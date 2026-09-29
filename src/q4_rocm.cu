#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

extern "C" {
#include "q4.h"
}

#define Q4_IQ_DECL static __constant__
#include "q4_iq_grids.inc"

static __constant__ int8_t q4_kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

#define Q4_HIP_CHECK(x)                                                        \
    do {                                                                       \
        hipError_t _e = (x);                                                   \
        if (_e != hipSuccess) {                                                \
            fprintf(stderr, "q4 hip: %s (%s:%d)\n", hipGetErrorString(_e),     \
                    __FILE__, __LINE__);                                       \
            return false;                                                      \
        }                                                                      \
    } while (0)

static bool g_ok = false;
static float *g_dx = nullptr;
static float *g_dy = nullptr;
static uint8_t *g_dw = nullptr;
static size_t g_dx_n = 0, g_dy_n = 0, g_dw_n = 0;
/* Out-of-place write-back target for gdn_scan_k's shared q/k conv rows:
 * blocks may finish (and write) before stragglers have read hist. */
static float *g_scanqk = nullptr;
static size_t g_scanqk_n = 0;
static uint32_t *g_tk_hist, *g_tk_st, *g_tk_cnt, *g_tk_off;
static int32_t *g_tk_chosen;
static float *g_attn_part; /* (n_kvh<=4,S,gqa<=12,hd+2) split-dec partials */
static float *g_sk_part;
static struct { uint32_t t; uint64_t r, c, b, n; } g_shapes[256];
static int g_shape_n;
/* Expert arenas: 0 = main model, 1 = MTP draft head. Separate because the
 * slot sizes differ and the caches are independent LRU sets. */
#define Q4_HIP_ARENAS 2
static uint8_t *g_exp[Q4_HIP_ARENAS] = {nullptr, nullptr};
static uint64_t g_exp_slot[Q4_HIP_ARENAS] = {0, 0};
static uint32_t g_exp_n[Q4_HIP_ARENAS] = {0, 0};
static hipStream_t g_copy = nullptr;
/* All compute launches go on this stream so stream capture can graph them.
 * hipStreamLegacy capture is unsupported on ROCm, so a real stream it is. */
static hipStream_t g_str = nullptr;
/* 1 while g_str is capturing a graph: blocking/legacy-stream calls are
 * illegal then, so helpers must take their pure-kernel path. */
static int g_capturing = 0;

/* Step-profiler marker events (allocated up front so nothing is created
 * during graph capture). */
#define Q4_MK_SLOTS 16
static hipEvent_t g_mk0[Q4_MK_SLOTS], g_mk1[Q4_MK_SLOTS];
static int g_mk_armed[Q4_MK_SLOTS];

static bool exp_ok(int arena) { return arena >= 0 && arena < Q4_HIP_ARENAS; }

static bool grow(void **p, size_t *cap, size_t need) {
    if (*cap >= need && *p) return true;
    if (*p) (void)hipFree(*p);
    *p = nullptr;
    *cap = 0;
    if (hipMalloc(p, need) != hipSuccess) {
        *p = nullptr;
        (void)hipGetLastError();
        return false;
    }
    *cap = need;
    return true;
}

static bool moe_stage_bufs(void);
static bool xq8_bufs(void);
static uint32_t *g_bar; /* grid-barrier counter/flag for gdn_tail_k */
static int32_t *g_pos_d;   /* decode position read by graph-captured kernels */
static uint32_t g_last_graph_nodes;
static int32_t *g_pos_h;   /* pinned staging for pos publishes */

extern "C" bool q4_hip_init(void) {
    if (g_ok) return true;
    setenv("HIP_VISIBLE_DEVICES", "0", 0);
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess || n <= 0) return false;
    if (hipSetDevice(0) != hipSuccess) return false;
    if (hipStreamCreateWithFlags(&g_str, hipStreamNonBlocking) != hipSuccess)
        g_str = nullptr; /* nullptr == legacy default: still correct, uncapturable */
    if (hipStreamCreate(&g_copy) != hipSuccess) g_copy = nullptr;
    g_ok = true;
    /* Pre-create step-profiler marker events here: creating them lazily would
     * hit hipEventCreate inside a graph capture and invalidate it. */
    {
        const char *e = getenv("Q4_STEP_PROF");
        if (e && e[0] == '1')
            for (int i = 0; i < Q4_MK_SLOTS; i++) {
                (void)hipEventCreate(&g_mk0[i]);
                (void)hipEventCreate(&g_mk1[i]);
            }
    }
    (void)moe_stage_bufs(); /* avoid first-time hipMalloc inside a capture */
    if (hipMalloc((void **)&g_bar, 2 * sizeof(uint32_t)) == hipSuccess)
        (void)hipMemset(g_bar, 0, 2 * sizeof(uint32_t));
    (void)hipMalloc((void **)&g_pos_d, sizeof(int32_t));
    (void)hipHostMalloc((void **)&g_pos_h, sizeof(int32_t), 0);
    return true;
}

extern "C" void q4_hip_experts_free(void) {
    for (int a = 0; a < Q4_HIP_ARENAS; a++) {
        if (g_exp[a]) (void)hipFree(g_exp[a]);
        g_exp[a] = nullptr;
        g_exp_slot[a] = 0;
        g_exp_n[a] = 0;
    }
}

extern "C" void q4_hip_shutdown(void) {
    q4_hip_experts_free();
    if (g_copy) {
        (void)hipStreamDestroy(g_copy);
        g_copy = nullptr;
    }
    if (g_dx) (void)hipFree(g_dx);
    if (g_dy) (void)hipFree(g_dy);
    if (g_dw) (void)hipFree(g_dw);
    if (g_bar) {
        (void)hipFree(g_bar);
        g_bar = nullptr;
    }
    if (g_pos_d) {
        (void)hipFree(g_pos_d);
        g_pos_d = nullptr;
    }
    if (g_pos_h) {
        (void)hipHostFree(g_pos_h);
        g_pos_h = nullptr;
    }
    if (g_scanqk) (void)hipFree(g_scanqk);
    if (g_tk_hist) (void)hipFree(g_tk_hist);
    if (g_tk_st) (void)hipFree(g_tk_st);
    if (g_tk_cnt) (void)hipFree(g_tk_cnt);
    if (g_tk_off) (void)hipFree(g_tk_off);
    if (g_tk_chosen) (void)hipFree(g_tk_chosen);
    if (g_attn_part) (void)hipFree(g_attn_part);
    if (g_sk_part) (void)hipFree(g_sk_part);
    g_tk_hist = g_tk_st = g_tk_cnt = g_tk_off = nullptr;
    g_tk_chosen = nullptr;
    g_attn_part = nullptr;
    g_sk_part = nullptr;
    if (getenv("Q4_SHAPES"))
        for (int i = 0; i < g_shape_n; i++)
            fprintf(stderr, "SHAPE ty=%u r=%llu c=%llu nb=%llu n=%llu\n",
                    g_shapes[i].t, (unsigned long long)g_shapes[i].r,
                    (unsigned long long)g_shapes[i].c,
                    (unsigned long long)g_shapes[i].b,
                    (unsigned long long)g_shapes[i].n);
    g_dx = g_dy = nullptr;
    g_dw = nullptr;
    g_scanqk = nullptr;
    g_dx_n = g_dy_n = g_dw_n = g_scanqk_n = 0;
    g_ok = false;
}

extern "C" bool q4_hip_experts_alloc_a(int arena, uint32_t n_slots,
                                       uint64_t slot_bytes) {
    if (!g_ok || !exp_ok(arena) || n_slots == 0 || slot_bytes == 0) return false;
    if (g_exp[arena]) (void)hipFree(g_exp[arena]);
    g_exp[arena] = nullptr;
    g_exp_n[arena] = 0;
    g_exp_slot[arena] = 0;
    size_t n = (size_t)n_slots * (size_t)slot_bytes;
    if (hipMalloc((void **)&g_exp[arena], n) != hipSuccess) {
        g_exp[arena] = nullptr;
        (void)hipGetLastError();
        return false;
    }
    g_exp_n[arena] = n_slots;
    g_exp_slot[arena] = slot_bytes;
    return true;
}

extern "C" bool q4_hip_experts_alloc(uint32_t n_slots, uint64_t slot_bytes) {
    return q4_hip_experts_alloc_a(0, n_slots, slot_bytes);
}

extern "C" uint32_t q4_hip_experts_n_a(int arena) {
    return exp_ok(arena) && g_exp[arena] ? g_exp_n[arena] : 0;
}

extern "C" uint32_t q4_hip_experts_n(void) {
    return q4_hip_experts_n_a(0);
}

/* Synchronous copy that still orders after g_str work. With a NonBlocking
 * compute stream, a blocking hipMemcpy on the legacy stream does NOT wait
 * for g_str kernels — it must be enqueued on g_str itself. During graph
 * capture it degrades to a memcpy node (host-visible results only after the
 * replay; capture regions must not read dst on the host right away). */
static bool sync_copy(void *dst, const void *src, size_t n,
                      hipMemcpyKind kind) {
    if (!g_str) return hipMemcpy(dst, src, n, kind) == hipSuccess;
    if (g_capturing)
        return hipMemcpyAsync(dst, src, n, kind, g_str) == hipSuccess;
    if (hipMemcpyAsync(dst, src, n, kind, g_str) != hipSuccess) return false;
    return hipStreamSynchronize(g_str) == hipSuccess;
}

extern "C" bool q4_hip_experts_put_a(int arena, uint32_t slot,
                                     const uint8_t *host, uint64_t n) {
    if (!exp_ok(arena) || !g_exp[arena] || slot >= g_exp_n[arena] || !host ||
        n == 0)
        return false;
    if (n > g_exp_slot[arena]) n = g_exp_slot[arena];
    return sync_copy(g_exp[arena] + (size_t)slot * (size_t)g_exp_slot[arena],
                     host, (size_t)n, hipMemcpyHostToDevice);
}

extern "C" bool q4_hip_experts_put(uint32_t slot, const uint8_t *host, uint64_t n) {
    return q4_hip_experts_put_a(0, slot, host, n);
}

extern "C" bool q4_hip_experts_put_async_a(int arena, uint32_t slot,
                                           const uint8_t *host, uint64_t n) {
    if (!exp_ok(arena) || !g_exp[arena] || slot >= g_exp_n[arena] || !host ||
        n == 0)
        return false;
    if (n > g_exp_slot[arena]) n = g_exp_slot[arena];
    hipStream_t st = g_copy ? g_copy : 0;
    Q4_HIP_CHECK(hipMemcpyAsync(g_exp[arena] + (size_t)slot * (size_t)g_exp_slot[arena],
                                host, (size_t)n, hipMemcpyHostToDevice, st));
    return true;
}

extern "C" bool q4_hip_experts_put_async(uint32_t slot, const uint8_t *host,
                                         uint64_t n) {
    return q4_hip_experts_put_async_a(0, slot, host, n);
}

extern "C" void q4_hip_copy_wait(void) {
    if (g_copy) (void)hipStreamSynchronize(g_copy);
    else (void)hipDeviceSynchronize();
}

extern "C" const uint8_t *q4_hip_experts_dev_a(int arena, uint32_t slot) {
    if (!exp_ok(arena) || !g_exp[arena] || slot >= g_exp_n[arena]) return nullptr;
    return g_exp[arena] + (size_t)slot * (size_t)g_exp_slot[arena];
}

extern "C" const uint8_t *q4_hip_experts_base_a(int arena) {
    return exp_ok(arena) ? g_exp[arena] : nullptr;
}
extern "C" uint64_t q4_hip_experts_stride_a(int arena) {
    return exp_ok(arena) ? g_exp_slot[arena] : 0;
}

extern "C" const uint8_t *q4_hip_experts_dev(uint32_t slot) {
    return q4_hip_experts_dev_a(0, slot);
}

extern "C" bool q4_hip_ok(void) { return g_ok; }

extern "C" bool q4_hip_host_register(void *p, size_t n) {
    if (!g_ok || !p || !n) return false;
    if (hipHostRegister(p, n, hipHostRegisterMapped) != hipSuccess) {
        (void)hipGetLastError();
        return false;
    }
    return true;
}
/* Device-accessible alias of a host-registered buffer (zero-copy). */
extern "C" void *q4_hip_host_devptr(void *p) {
    if (!g_ok || !p) return nullptr;
    void *d = nullptr;
    if (hipHostGetDevicePointer(&d, p, 0) != hipSuccess) {
        (void)hipGetLastError();
        return nullptr;
    }
    return d;
}
extern "C" void q4_hip_host_unregister(void *p) {
    if (g_ok && p) (void)hipHostUnregister(p);
}

extern "C" void q4_hip_sync(void) {
    if (g_ok) (void)hipDeviceSynchronize();
}

extern "C" void q4_hip_clear(void) {
    (void)hipGetLastError();
}

extern "C" void *q4_hip_malloc(size_t n) {
    if (!g_ok || n == 0) return nullptr;
    void *p = nullptr;
    if (hipMalloc(&p, n) != hipSuccess) {
        (void)hipGetLastError();
        return nullptr;
    }
    return p;
}

extern "C" void q4_hip_free(void *p) {
    if (p) (void)hipFree(p);
}

extern "C" bool q4_hip_h2d(void *d, const void *h, size_t n) {
    if (!d || !h || n == 0) return false;
    return sync_copy(d, h, n, hipMemcpyHostToDevice);
}

extern "C" bool q4_hip_d2h(void *h, const void *d, size_t n) {
    if (!d || !h || n == 0) return false;
    return sync_copy(h, d, n, hipMemcpyDeviceToHost);
}

extern "C" int q4_hip_clock_khz(void) {
    static int khz;
    if (!khz) {
        int v = 0;
        if (hipDeviceGetAttribute(&v, hipDeviceAttributeClockRate, 0) !=
            hipSuccess)
            return 0;
        khz = v;
    }
    return khz;
}

__global__ static void gemv_q8_0_k(float *out, const uint8_t *w, const float *x,
                                   uint32_t in_dim, uint32_t out_dim, float scale,
                                   uint32_t n_batch, const int32_t *which) {
    uint32_t row = blockIdx.x;
    if (row >= out_dim) return;
    uint32_t blocks = (in_dim + 31u) / 32u;
    const uint8_t *wr = w + (size_t)row * (size_t)blocks * 34ull;
    __shared__ float sh[64];
    for (uint32_t tok = blockIdx.y; tok < n_batch; tok += gridDim.y) {
        const float *xt = x + (size_t)(which ? which[tok] : (int32_t)tok) * in_dim;
        float acc = 0.f;
        for (uint32_t b = threadIdx.x; b < blocks; b += blockDim.x) {
            uint32_t i0 = b * 32u;
            uint32_t n = in_dim - i0 < 32u ? in_dim - i0 : 32u;
            __half dh;
            memcpy(&dh, wr + b * 34u, 2);
            const int8_t *qs = (const int8_t *)(wr + b * 34u + 2);
            float dot = 0.f;
            for (uint32_t i = 0; i < n; i++) dot += (float)qs[i] * xt[i0 + i];
            acc += __half2float(dh) * dot;
        }
        sh[threadIdx.x] = acc;
        __syncthreads();
        for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            out[(size_t)tok * out_dim + row] = sh[0] * scale;
        __syncthreads();
    }
}

__device__ static float bf16_f(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

__global__ static void gemv_bf16_k(float *out, const uint8_t *w, const float *x,
                                   uint32_t in_dim, uint32_t out_dim, float scale,
                                   uint32_t n_batch, const int32_t *which) {
    uint32_t row = blockIdx.x;
    if (row >= out_dim) return;
    const uint16_t *wr = (const uint16_t *)(w + (size_t)row * in_dim * 2u);
    __shared__ float sh[64];
    for (uint32_t tok = blockIdx.y; tok < n_batch; tok += gridDim.y) {
        const float *xt = x + (size_t)(which ? which[tok] : (int32_t)tok) * in_dim;
        float acc = 0.f;
        for (uint32_t i = threadIdx.x; i < in_dim; i += blockDim.x)
            acc += bf16_f(wr[i]) * xt[i];
        sh[threadIdx.x] = acc;
        __syncthreads();
        for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            out[(size_t)tok * out_dim + row] = sh[0] * scale;
        __syncthreads();
    }
}

/* Warp-per-row bf16 GEMV for the decode path (n_batch==1): each warp owns a
 * row, lanes take 4B bf16 pairs and shuffle-reduce — no shared memory, no
 * block barriers.  The old block-per-row kernel spent most of its time in
 * the __syncthreads reduction with only ~20B/thread of real work. */
__global__ static void wgv_bf16_k(float *out, const uint8_t *w,
                                  const float *x, uint32_t in_dim,
                                  uint32_t out_dim, float scale) {
    const uint32_t wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    if (wid >= out_dim) return;
    const uint16_t *wr = (const uint16_t *)(w + (size_t)wid * in_dim * 2u);
    float acc = 0.f;
    if (!(in_dim & 1u) && (((uintptr_t)w & 3u) == 0)) {
        /* Even in_dim + aligned base: pairs come in 4B loads. */
        const uint32_t *wr32 = (const uint32_t *)wr;
        const uint32_t n2 = in_dim >> 1;
        for (uint32_t i = lane; i < n2; i += 32u) {
            uint32_t pr = wr32[i];
            float lo, hi;
            uint32_t ul = pr << 16, uh = pr & 0xffff0000u;
            memcpy(&lo, &ul, 4);
            memcpy(&hi, &uh, 4);
            acc = fmaf(lo, x[2u * i], fmaf(hi, x[2u * i + 1u], acc));
        }
    } else {
        for (uint32_t i = lane; i < in_dim; i += 32u)
            acc = fmaf(bf16_f(wr[i]), x[i], acc);
    }
    for (uint32_t o = 16; o; o >>= 1)
        acc += __shfl_xor_sync(0xffffffffull, acc, o);
    if (lane == 0) out[wid] = acc * scale;
}

/* Tiled many-token GEMM (prefill), same shape as gemm_q8_0_k: each block
 * owns a 32x32 (row x token) tile, stages dequantized W and f32 X tiles in
 * shared, one thread per (1 row, 4 toks). W is re-read only T/32 times.
 * WT is float (f32 weights) or uint16_t (bf16). */
__device__ static inline float wt_f(float v) { return v; }
__device__ static inline float wt_f(uint16_t v) { return bf16_f(v); }
template<typename WT>
__global__ static void gemm_tiled_k(float *out, const WT *w, const float *x,
                                   uint32_t in_dim, uint32_t out_dim,
                                   float scale, uint32_t n_batch,
                                   const int32_t *which) {
    __shared__ float ws[32][33];
    __shared__ float xs[32][33];
    const uint32_t r0 = blockIdx.x * 32u;
    const uint32_t t0 = blockIdx.y * 32u;
    const uint32_t nb = (in_dim + 31u) / 32u;
    const uint32_t wr = threadIdx.x / 8u;
    const uint32_t kk = (threadIdx.x % 8u) * 4u;
    const uint32_t warp = threadIdx.x / 32u;
    const uint32_t lane = threadIdx.x % 32u;
    const uint32_t tr = warp * 4u + lane / 8u;
    const uint32_t tt = (lane % 8u) * 4u;
    float acc[4] = {};
    for (uint32_t kb = 0; kb < nb; kb++) {
        uint32_t i0 = kb * 32u;
        {
            uint32_t r = r0 + wr;
            float v[4] = {0.f, 0.f, 0.f, 0.f};
            if (r < out_dim) {
                const WT *xr = w + (size_t)r * in_dim + i0 + kk;
#pragma unroll
                for (int j = 0; j < 4; j++)
                    v[j] = i0 + kk + j < in_dim ? wt_f(xr[j]) : 0.f;
            }
            ws[wr][kk + 0] = v[0];
            ws[wr][kk + 1] = v[1];
            ws[wr][kk + 2] = v[2];
            ws[wr][kk + 3] = v[3];
        }
        {
            uint32_t t = t0 + wr;
            float v[4] = {0.f, 0.f, 0.f, 0.f};
            if (t < n_batch) {
                const float *xr =
                    x + (size_t)(which ? which[t] : (int32_t)t) * in_dim + i0 +
                    kk;
#pragma unroll
                for (int j = 0; j < 4; j++)
                    v[j] = i0 + kk + j < in_dim ? xr[j] : 0.f;
            }
            xs[wr][kk + 0] = v[0];
            xs[wr][kk + 1] = v[1];
            xs[wr][kk + 2] = v[2];
            xs[wr][kk + 3] = v[3];
        }
        __syncthreads();
#pragma unroll 8
        for (uint32_t k = 0; k < 32u; k++) {
            float wv = ws[tr][k];
#pragma unroll
            for (int j = 0; j < 4; j++) acc[j] += wv * xs[tt + j][k];
        }
        __syncthreads();
    }
    {
        uint32_t r = r0 + tr;
        if (r < out_dim) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                uint32_t t = t0 + tt + (uint32_t)j;
                if (t < n_batch) out[(size_t)t * out_dim + r] = acc[j] * scale;
            }
        }
    }
}

__global__ static void gemv_f32_k(float *out, const float *w, const float *x,
                                  uint32_t in_dim, uint32_t out_dim, float scale,
                                  uint32_t n_batch, const int32_t *which) {
    uint32_t row = blockIdx.x;
    if (row >= out_dim) return;
    const float *wr = w + (size_t)row * in_dim;
    __shared__ float sh[64];
    for (uint32_t tok = blockIdx.y; tok < n_batch; tok += gridDim.y) {
        const float *xt = x + (size_t)(which ? which[tok] : (int32_t)tok) * in_dim;
        float acc = 0.f;
        for (uint32_t i = threadIdx.x; i < in_dim; i += blockDim.x)
            acc += wr[i] * xt[i];
        sh[threadIdx.x] = acc;
        __syncthreads();
        for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            out[(size_t)tok * out_dim + row] = sh[0] * scale;
        __syncthreads();
    }
}

__device__ static inline float f16d(const uint8_t *p) {
    __half h;
    memcpy(&h, p, 2);
    return __half2float(h);
}

__device__ static inline float iq_u8(uint64_t v, int j) {
    return (float)((v >> (8 * j)) & 0xffu);
}

__device__ static float dot_iq2s(const uint8_t *blk, const float *x) {
    const float d = f16d(blk);
    const uint8_t *qs = blk + 2;
    const uint8_t *qh = blk + 66;
    const uint8_t *scales = blk + 74;
    const uint8_t *signs = qs + 32;
    float acc = 0.f;
    uint32_t xi = 0;
    for (int ib = 0; ib < 8; ib++) {
        const float db0 = d * (0.5f + (float)(scales[ib] & 0xf)) * 0.25f;
        const float db1 = d * (0.5f + (float)(scales[ib] >> 4)) * 0.25f;
        for (int l = 0; l < 4; l++) {
            const float dl = l < 2 ? db0 : db1;
            const int idx = qs[l] | ((qh[ib] << (8 - 2 * l)) & 0x300);
            const uint64_t gv = q4_iq2s_grid[idx];
            const uint8_t sg = signs[l];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float s = (sg & q4_kmask_iq2xs[j]) ? -1.f : 1.f;
                acc += dl * iq_u8(gv, j) * s * x[xi++];
            }
        }
        qs += 4;
        signs += 4;
    }
    return acc;
}

__device__ static float dot_iq3s(const uint8_t *blk, const float *x) {
    const float d = f16d(blk);
    const uint8_t *qs = blk + 2;
    const uint8_t *qh = blk + 66;
    const uint8_t *signs = blk + 74;
    const uint8_t *scales = blk + 106;
    float acc = 0.f;
    uint32_t xi = 0;
    for (int ib = 0; ib < 8; ib += 2) {
        const float db1 = d * (1.f + 2.f * (float)(scales[ib / 2] & 0xf));
        const float db2 = d * (1.f + 2.f * (float)(scales[ib / 2] >> 4));
        for (int half = 0; half < 2; half++) {
            const float db = half == 0 ? db1 : db2;
            const uint8_t qhb = qh[half];
            for (int l = 0; l < 4; l++) {
                const int i1 = qs[2 * l] | ((qhb << (8 - 2 * l)) & 256);
                const int i2 = qs[2 * l + 1] | ((qhb << (7 - 2 * l)) & 256);
                const uint32_t g1 = q4_iq3s_grid[i1];
                const uint32_t g2 = q4_iq3s_grid[i2];
                const uint8_t sg = signs[l];
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    acc += db * iq_u8(g1, j) *
                           ((sg & q4_kmask_iq2xs[j]) ? -1.f : 1.f) * x[xi + j];
                    acc += db * iq_u8(g2, j) *
                           ((sg & q4_kmask_iq2xs[j + 4]) ? -1.f : 1.f) *
                           x[xi + 4 + j];
                }
                xi += 8;
            }
            qs += 8;
            signs += 4;
        }
        qh += 2;
    }
    return acc;
}

__device__ static float dot_iq3xxs(const uint8_t *blk, const float *x) {
    const float d = f16d(blk);
    const uint8_t *qs = blk + 2;
    const uint8_t *ss = qs + 64;
    float acc = 0.f;
    uint32_t xi = 0;
    for (int ib = 0; ib < 8; ib++) {
        uint32_t aux32;
        memcpy(&aux32, ss + 4 * ib, 4);
        const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
        for (int l = 0; l < 4; l++) {
            const uint8_t sg = q4_ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
            const uint32_t g1 = q4_iq3xxs_grid[qs[2 * l]];
            const uint32_t g2 = q4_iq3xxs_grid[qs[2 * l + 1]];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                acc += db * iq_u8(g1, j) *
                       ((sg & (1u << j)) ? -1.f : 1.f) * x[xi + j];
                acc += db * iq_u8(g2, j) *
                       ((sg & (1u << (j + 4))) ? -1.f : 1.f) * x[xi + 4 + j];
            }
            xi += 8;
        }
        qs += 8;
    }
    return acc;
}

__device__ static float dot_iq4xs(const uint8_t *blk, const float *x) {
    const float d = f16d(blk);
    uint32_t sh;
    memcpy(&sh, blk + 2, 2);
    const uint8_t *sl = blk + 4;
    const uint8_t *qs = blk + 8;
    float acc = 0.f;
    uint32_t xi = 0;
    for (int ib = 0; ib < 8; ib++) {
        const int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0xf) |
                       ((int)((sh >> (2 * ib)) & 3u) << 4);
        const float dl = d * (float)(ls - 32);
#pragma unroll
        for (int j = 0; j < 16; j++) {
            acc += dl * (float)q4_kvalues_iq4nl[qs[j] & 0xf] * x[xi + j];
            acc += dl * (float)q4_kvalues_iq4nl[qs[j] >> 4] * x[xi + 16 + j];
        }
        qs += 16;
        xi += 32;
    }
    return acc;
}

__device__ static float dot_iq4nl(const uint8_t *blk, const float *x) {
    const float d = f16d(blk);
    const uint8_t *qs = blk + 2;
    float acc = 0.f;
    for (int j = 0; j < 16; j++) {
        const uint8_t q = qs[j];
        acc += d * (float)q4_kvalues_iq4nl[q & 0xf] * x[j];
        acc += d * (float)q4_kvalues_iq4nl[q >> 4] * x[j + 16];
    }
    return acc;
}

__device__ static float dot_q40(const uint8_t *blk, const float *x) {
    const float d = f16d(blk);
    const uint8_t *qs = blk + 2;
    float acc = 0.f;
    for (int j = 0; j < 16; j++) {
        acc += d * (float)((int)(qs[j] & 0xf) - 8) * x[j];
        acc += d * (float)((int)(qs[j] >> 4) - 8) * x[j + 16];
    }
    return acc;
}

__device__ static float dot_q50(const uint8_t *blk, const float *x) {
    const float d = f16d(blk);
    uint32_t qh;
    memcpy(&qh, blk + 2, 4);
    const uint8_t *qs = blk + 6;
    float acc = 0.f;
    for (int j = 0; j < 16; j++) {
        acc += d * (float)((int)((qs[j] & 0xf) | (((qh >> j) & 1u) << 4)) - 16) *
               x[j];
        acc += d * (float)((int)((qs[j] >> 4) |
                                 (((qh >> (j + 16)) & 1u) << 4)) - 16) *
               x[j + 16];
    }
    return acc;
}

__device__ static float dot_q20(const uint8_t *blk, const float *x) {
    const float d = f16d(blk);
    const uint8_t *qs = blk + 2;
    float acc = 0.f;
    for (int j = 0; j < 64; j++)
        acc += d * (float)((int)((qs[j >> 2] >> ((j & 3) * 2)) & 3u) - 1) * x[j];
    return acc;
}

__device__ static float dot_q6k(const uint8_t *blk, const float *x) {
    const uint8_t *ql = blk;
    const uint8_t *qh = blk + 128;
    const int8_t *sc = (const int8_t *)(blk + 192);
    const float d = f16d(blk + 208);
    float acc = 0.f;
    for (int n128 = 0; n128 < 256; n128 += 128) {
        for (int l = 0; l < 32; l++) {
            const int is = l / 16;
            const int q1 = (int)(ql[l] & 0xF) | ((int)(qh[l] & 3) << 4);
            const int q2 = (int)(ql[l + 32] & 0xF) | (((int)(qh[l] >> 2) & 3) << 4);
            const int q3 = (int)(ql[l] >> 4) | (((int)(qh[l] >> 4) & 3) << 4);
            const int q4 = (int)(ql[l + 32] >> 4) | (((int)(qh[l] >> 6) & 3) << 4);
            acc += d * (float)sc[is] * (float)(q1 - 32) * x[n128 + l];
            acc += d * (float)sc[is + 2] * (float)(q2 - 32) * x[n128 + l + 32];
            acc += d * (float)sc[is + 4] * (float)(q3 - 32) * x[n128 + l + 64];
            acc += d * (float)sc[is + 6] * (float)(q4 - 32) * x[n128 + l + 96];
        }
        ql += 64;
        qh += 32;
        sc += 8;
    }
    return acc;
}


#define Q4_GEMV_BLK(name, dotfn, blk_bytes, blk_vals)                          \
    __global__ static void name(float *out, const uint8_t *w, const float *x, \
                                uint32_t in_dim, uint32_t out_dim, float scale,\
                                uint32_t n_batch, const int32_t *which) {      \
        uint32_t row = blockIdx.x;                                             \
        if (row >= out_dim) return;                                            \
        uint32_t nb = in_dim / (blk_vals);                                     \
        const uint8_t *wr = w + (size_t)row * nb * (blk_bytes);                \
        __shared__ float sh[64];                                               \
        for (uint32_t tok = blockIdx.y; tok < n_batch; tok += gridDim.y) {     \
            const float *xt =                                                  \
                x + (size_t)(which ? which[tok] : (int32_t)tok) * in_dim;      \
            float acc = 0.f;                                                   \
            for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x)            \
                acc += dotfn(wr + (size_t)b * (blk_bytes),                     \
                             xt + (size_t)b * (blk_vals));                     \
            sh[threadIdx.x] = acc;                                             \
            __syncthreads();                                                   \
            for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {               \
                if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];   \
                __syncthreads();                                               \
            }                                                                  \
            if (threadIdx.x == 0)                                              \
                out[(size_t)tok * out_dim + row] = sh[0] * scale;              \
            __syncthreads();                                                   \
        }                                                                      \
    }

Q4_GEMV_BLK(gemv_iq2_s_k, dot_iq2s, 82u, 256u)
Q4_GEMV_BLK(gemv_iq3_s_k, dot_iq3s, 110u, 256u)
Q4_GEMV_BLK(gemv_iq4_nl_k, dot_iq4nl, 18u, 32u)
Q4_GEMV_BLK(gemv_iq3xxs_k, dot_iq3xxs, 98u, 256u)
Q4_GEMV_BLK(gemv_iq4xs_k, dot_iq4xs, 136u, 256u)
Q4_GEMV_BLK(gemv_q4_0_k, dot_q40, 18u, 32u)
Q4_GEMV_BLK(gemv_q5_0_k, dot_q50, 22u, 32u)
Q4_GEMV_BLK(gemv_q2_0_k, dot_q20, 18u, 64u)
/* Head logits gemv: split every 256-value block into 16 units of 4 quads;
 * each thread reads one unit's ql/qh as uint32 vectors. Deterministic: the
 * per-row result is the fixed blockDim reduction of thread partials. */
__global__ static void gemv_q6_k_k(float *out, const uint8_t *w, const float *x,
                                   uint32_t in_dim, uint32_t out_dim,
                                   float scale, uint32_t n_batch,
                                   const int32_t *which) {
    uint32_t row = blockIdx.x;
    if (row >= out_dim) return;
    uint32_t nb = in_dim / 256u;
    const uint8_t *wr = w + (size_t)row * nb * 210u;
    __shared__ float sh[128];
    for (uint32_t tok = blockIdx.y; tok < n_batch; tok += gridDim.y) {
        const float *xt =
            x + (size_t)(which ? which[tok] : (int32_t)tok) * in_dim;
        float acc = 0.f;
        uint32_t nu = nb * 16u;
        for (uint32_t u = threadIdx.x; u < nu; u += blockDim.x) {
            const uint8_t *blk = wr + (size_t)(u >> 4) * 210u;
            uint32_t qu = u & 15u;
            uint32_t h2 = qu >> 3, l4 = (qu & 7u) * 4u;
            uint32_t qlv0 = *(const uint32_t *)(blk + h2 * 64u + l4);
            uint32_t qlv1 = *(const uint32_t *)(blk + h2 * 64u + l4 + 32u);
            uint32_t qhv = *(const uint32_t *)(blk + 128u + h2 * 32u + l4);
            const int8_t *sc = (const int8_t *)(blk + 192u) + h2 * 8;
            float d = f16d(blk + 208u);
            uint32_t is = l4 >> 4;
            const float *xt4 = xt + (u >> 4) * 256u + h2 * 128u + l4;
            for (uint32_t j = 0; j < 4; j++) {
                uint32_t qlo = (qlv0 >> (j * 8)) & 0xffu;
                uint32_t qlh = (qlv1 >> (j * 8)) & 0xffu;
                uint32_t qhb = (qhv >> (j * 8)) & 0xffu;
                acc += d * (float)sc[is] *
                       (float)(int((qlo & 15u) | ((qhb & 3u) << 4)) - 32) *
                       xt4[j];
                acc += d * (float)sc[is + 2u] *
                       (float)(int((qlh & 15u) | (((qhb >> 2) & 3u) << 4)) -
                               32) *
                       xt4[j + 32u];
                acc += d * (float)sc[is + 4u] *
                       (float)(int((qlo >> 4) | (((qhb >> 4) & 3u) << 4)) - 32) *
                       xt4[j + 64u];
                acc += d * (float)sc[is + 6u] *
                       (float)(int((qlh >> 4) | (((qhb >> 6) & 3u) << 4)) - 32) *
                       xt4[j + 96u];
            }
        }
        sh[threadIdx.x] = acc;
        __syncthreads();
        for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            out[(size_t)tok * out_dim + row] = sh[0] * scale;
        __syncthreads();
    }
}

/* Batched routed-expert rows: grid.x = output rows, grid.y = expert index.
 * Each expert's weight pack lives at base + slots[e] * stride; x is either
 * shared (xs = 0) or per-expert (xs = in_dim). One launch per stage instead
 * of one launch per expert — removes ~3*n small kernels per MoE layer. */
#define Q4_MOE_BLK(name, dotfn, bb, bv)                                        \
    __global__ static void name(float *out, const uint8_t *base,               \
                                uint64_t stride, const int32_t *slots,         \
                                const float *x, uint32_t xs,                   \
                                uint32_t in_dim, uint32_t out_dim) {           \
        uint32_t e = blockIdx.y, row = blockIdx.x;                             \
        uint32_t nb = in_dim / (bv);                                           \
        const uint8_t *wr = base + (size_t)slots[e] * (stride) +               \
                            (size_t)row * nb * (bb);                           \
        const float *xt = x + (size_t)e * xs;                                  \
        __shared__ float sh[64];                                               \
        float acc = 0.f;                                                       \
        for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x)                \
            acc += dotfn(wr + (size_t)b * (bb), xt + (size_t)b * (bv));        \
        sh[threadIdx.x] = acc;                                                 \
        __syncthreads();                                                       \
        for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {                   \
            if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];       \
            __syncthreads();                                                   \
        }                                                                      \
        if (threadIdx.x == 0)                                                  \
            out[(size_t)e * out_dim + row] = sh[0];                            \
    }
Q4_MOE_BLK(moe_iq2_bk, dot_iq2s, 82u, 256u)
Q4_MOE_BLK(moe_iq3_bk, dot_iq3s, 110u, 256u)
Q4_MOE_BLK(moe_iq4_bk, dot_iq4nl, 18u, 32u)
Q4_MOE_BLK(moe_iq3xxs_bk, dot_iq3xxs, 98u, 256u)
Q4_MOE_BLK(moe_iq4xs_bk, dot_iq4xs, 136u, 256u)
Q4_MOE_BLK(moe_q40_bk, dot_q40, 18u, 32u)
Q4_MOE_BLK(moe_q50_bk, dot_q50, 22u, 32u)
Q4_MOE_BLK(moe_q20_bk, dot_q20, 18u, 64u)

/* ---- warp-per-row decode GEMVs (Q4_WARPGV, default on) -------------------
 * Decode GEMVs are latency-bound: one 64-thread block per row underfills the
 * GPU and reads quant bytes serially.  These kernels give each warp one
 * output row and map the 32 lanes onto 8-element "octs" inside each quant
 * block, so every warp-load is fully coalesced (q8_0: lane -> 16 elems of
 * block l/2; iq2_s/iq3_s: lane -> oct lane of the single block per iter;
 * iq4_nl: lane -> quarter (l&3) of block l/4).  Lane-local fp32 accumulate,
 * then a fixed-order __shfl_down_sync tree — deterministic and graph-safe.
 * The fp sum order differs from the 64-thread kernels; diffs are bounded
 * reorder noise (~1e-5 relative), same as the gemv_f32_sk path. */
static int wgv_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_WARPGV");
        v = !(e && e[0] == '0');
    }
    return v;
}

static int wgvn_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_WGVN");
        v = !(e && e[0] == '0');
    }
    return v;
}

__device__ static inline float wgv_shfl_red(float acc) {
    for (uint32_t o = 16; o; o >>= 1)
        acc += __shfl_down_sync(0xffffffffull, acc, o);
    return acc;
}

/* q8_0: lane covers 16 elems (half block) via 8x u16 + 4x float2; warp covers
 * 16 blocks (512 vals) per iter.  Requires in_dim % 32 == 0. */
__device__ static float wgd_q8h(const uint8_t *blk, const float *x,
                                uint32_t m) {
    const float d = f16d(blk);
    float dot = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        uint16_t q2;
        memcpy(&q2, blk + 2 + m + 2 * j, 2);
        float2 xv = *(const float2 *)(x + m + 2 * j);
        dot += (float)(int8_t)(q2 & 0xff) * xv.x +
               (float)(int8_t)(q2 >> 8) * xv.y;
    }
    return d * dot;
}

/* iq2_s: lane covers oct 8*lane..+7 of a 256-block (ib = lane/4, l = lane&3,
 * j = 0..7).  qs/sign byte index == lane → coalesced 32B warp loads. */
__device__ static float wgd_iq2s(const uint8_t *blk, const float *x,
                                 uint32_t lane) {
    const float d = f16d(blk);
    const uint32_t ib = lane >> 2, l = lane & 3u;
    const float sc = (l < 2) ? (float)(blk[74 + ib] & 0xf)
                             : (float)(blk[74 + ib] >> 4);
    const float dl = d * (0.5f + sc) * 0.25f;
    const int idx = blk[2 + ib * 4 + l] |
                    ((blk[66 + ib] << (8 - 2 * l)) & 0x300);
    const uint64_t gv = q4_iq2s_grid[idx];
    const uint8_t sg = blk[2 + 32 + ib * 4 + l];
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const float s = (sg & (1u << j)) ? -1.f : 1.f;
        acc += iq_u8(gv, j) * s * x[j];
    }
    return dl * acc;
}

/* iq3_s: lane covers oct 8*lane..+7 of a 256-block (p = lane/8,
 * half = (lane>>2)&1, l = lane&3, jj = 0..7). */
__device__ static float wgd_iq3s(const uint8_t *blk, const float *x,
                                 uint32_t lane) {
    const float d = f16d(blk);
    const uint32_t p = lane >> 3, half = (lane >> 2) & 1u, l = lane & 3u;
    const float sc = (half == 0) ? (float)(blk[106 + p] & 0xf)
                                 : (float)(blk[106 + p] >> 4);
    const float db = d * (1.f + 2.f * sc);
    const uint8_t qhb = blk[66 + p * 2 + half];
    const int i1 = blk[2 + p * 16 + half * 8 + 2 * l] |
                   ((qhb << (8 - 2 * l)) & 256);
    const int i2 = blk[2 + p * 16 + half * 8 + 2 * l + 1] |
                   ((qhb << (7 - 2 * l)) & 256);
    const uint32_t g1 = q4_iq3s_grid[i1];
    const uint32_t g2 = q4_iq3s_grid[i2];
    const uint8_t sg = blk[74 + p * 8 + half * 4 + l];
    float acc = 0.f;
#pragma unroll
    for (int jj = 0; jj < 8; jj++) {
        const uint32_t gi = jj & 3u;
        const uint32_t gv = (jj < 4) ? g1 : g2;
        const float s = (sg & (1u << jj)) ? -1.f : 1.f;
        acc += (float)((gv >> (8 * gi)) & 0xffu) * s * x[jj];
    }
    return db * acc;
}

/* iq3_xxs: lane covers oct 8*lane..+7 of a 256-block (ib = lane>>2,
 * l = lane&3).  qs[96] = 64 grid bytes + 32 sign/scale bytes (aux32). */
__device__ static float wgd_iq3xxs(const uint8_t *blk, const float *x,
                                   uint32_t lane) {
    const float d = f16d(blk);
    const uint32_t ib = lane >> 2, l = lane & 3u;
    uint32_t aux32;
    memcpy(&aux32, blk + 2 + 64 + 4 * ib, 4);
    const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
    const uint8_t sg = q4_ksigns_iq2xs[(aux32 >> (7 * l)) & 127u];
    const uint8_t *q3 = blk + 2 + ib * 8;
    const uint32_t g1 = q4_iq3xxs_grid[q3[2 * l]];
    const uint32_t g2 = q4_iq3xxs_grid[q3[2 * l + 1]];
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const uint32_t gi = j & 3u;
        const uint32_t gv = (j < 4) ? g1 : g2;
        const float s = (sg & (1u << j)) ? -1.f : 1.f;
        acc += (float)((gv >> (8 * gi)) & 0xffu) * s * x[j];
    }
    return db * acc;
}

/* iq4_xs: lane covers oct 8*lane..+7 of a 256-block (ib = lane>>2,
 * q = lane&3).  Block: f16 d, u16 scales_h, scales_l[4], qs[128]. */
__device__ static float wgd_iq4xs(const uint8_t *blk, const float *x,
                                  uint32_t lane) {
    const float d = f16d(blk);
    const uint32_t ib = lane >> 2, q = lane & 3u;
    uint32_t sh;
    memcpy(&sh, blk + 2, 2);
    const int ls = ((blk[4 + (ib >> 1)] >> (4 * (ib & 1u))) & 0xf) |
                   ((int)((sh >> (2 * ib)) & 3u) << 4);
    const float dl = d * (float)(ls - 32);
    const uint8_t *qb = blk + 8 + ib * 16 + (q & 1u) * 8u;
    const uint32_t hi = q >> 1;
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const uint8_t nib = hi ? (qb[j] >> 4) : (qb[j] & 0xf);
        acc += (float)q4_kvalues_iq4nl[nib] * x[j];
    }
    return dl * acc;
}

/* iq4_nl: lane covers quarter (lane&3)*8..+7 of a 32-block via 4x u16 qs
 * loads (warp covers 8 blocks/iter). */
__device__ static float wgd_iq4nl(const uint8_t *blk, const float *x,
                                  uint32_t lane) {
    const float d = f16d(blk);
    const uint32_t q = lane & 3u;
    const uint32_t byte0 = (q & 1u) * 8u;
    const uint32_t hi = q >> 1;
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        uint16_t qb;
        memcpy(&qb, blk + 2 + byte0 + 2 * j, 2);
        const uint8_t v0 = hi ? ((qb >> 4) & 0xf) : (qb & 0xf);
        const uint8_t v1 = hi ? (qb >> 12) : ((qb >> 8) & 0xf);
        acc += (float)q4_kvalues_iq4nl[v0] * x[2 * j] +
               (float)q4_kvalues_iq4nl[v1] * x[2 * j + 1];
    }
    return d * acc;
}

/* q4_0: same lane geometry as iq4nl (quarter (lane&3)*8..+7), linear
 * nibble map instead of the table. */
__device__ static float wgd_q40(const uint8_t *blk, const float *x,
                                uint32_t lane) {
    const float d = f16d(blk);
    const uint32_t q = lane & 3u;
    const uint32_t byte0 = (q & 1u) * 8u;
    const uint32_t hi = q >> 1;
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        uint16_t qb;
        memcpy(&qb, blk + 2 + byte0 + 2 * j, 2);
        const int v0 = (int)(hi ? ((qb >> 4) & 0xf) : (qb & 0xf)) - 8;
        const int v1 = (int)(hi ? (qb >> 12) : ((qb >> 8) & 0xf)) - 8;
        acc += (float)v0 * x[2 * j] + (float)v1 * x[2 * j + 1];
    }
    return d * acc;
}

/* q5_0: q4_0 layout + 32 high bits at blk+2 (bit i = elem i), offset -16. */
__device__ static float wgd_q50(const uint8_t *blk, const float *x,
                                uint32_t lane) {
    const float d = f16d(blk);
    uint32_t qh;
    memcpy(&qh, blk + 2, 4);
    const uint32_t q = lane & 3u;
    const uint32_t byte0 = (q & 1u) * 8u;
    const uint32_t hi = q >> 1;
    const uint32_t e0 = q * 8u;
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        uint16_t qb;
        memcpy(&qb, blk + 6 + byte0 + 2 * j, 2);
        const uint32_t ea = e0 + 2 * j, eb = ea + 1;
        const int v0 = (int)((hi ? ((qb >> 4) & 0xf) : (qb & 0xf)) |
                             (((qh >> ea) & 1u) << 4)) - 16;
        const int v1 = (int)((hi ? (qb >> 12) : ((qb >> 8) & 0xf)) |
                             (((qh >> eb) & 1u) << 4)) - 16;
        acc += (float)v0 * x[2 * j] + (float)v1 * x[2 * j + 1];
    }
    return d * acc;
}

/* q2_0: 64-elem block = fp16 d + 16B of 2-bit quads; x is pre-offset to the
 * lane's 4-elem group, byte picks the packed byte.  Values (q - 1) * d. */
__device__ static float wgd_q20(const uint8_t *blk, const float *x,
                                uint32_t byte) {
    const float d = f16d(blk);
    const uint32_t qv = blk[2 + byte];
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++)
        acc += (float)((int)((qv >> (j * 2)) & 3u) - 1) * x[j];
    return d * acc;
}

/* ---- int8-activation helpers (gfx11 v_dot4_i32_iu8) --------------------
 * The wmoe_* float path spends most of its issue slots on per-element
 * load+cvt+fma after the weight decode.  With activations pre-quantized
 * to per-32 int8 groups (xq8_k below), the same decoders emit packed
 * i8x4 and reduce 4 products per instruction.  Q4_WGQ8=0 falls back. */

/* iq4_nl 16-entry table as a register LUT: two v_perm_b32 byte selects
 * blended on nibble bit3.  Beats 8 serialized __constant__ byte gathers. */
__device__ static inline uint32_t kv_iq4nl_lut4(uint32_t idx4) {
    const uint32_t sel = idx4 & 0x07070707u;
    const uint32_t mhi = ((idx4 >> 3) & 0x01010101u) * 0xFFu;
    const uint32_t lo = (uint32_t)__builtin_amdgcn_perm(0xF6EADDCFu,
                                                       0xBFAD9881u, sel);
    const uint32_t hi = (uint32_t)__builtin_amdgcn_perm(0x71594535u,
                                                       0x26190D01u, sel);
    return (lo & ~mhi) | (hi & mhi);
}

/* iq4_nl int8: same lane geometry as wgd_iq4nl — quarter (lane&3)*8 of a
 * 32-element block — but returns an i32 partial dot and the block scale
 * via *dp so the caller multiplies by dw*dx once per 8 elements. */
__device__ static int32_t wgd_iq4nl_i8(const uint8_t *blk, const int8_t *xq,
                                       uint32_t lane, float *dp) {
    *dp = f16d(blk);
    uint64_t qv;
    memcpy(&qv, blk + 2 + (lane & 1u) * 8u, 8);
    if (lane & 2u) qv >>= 4;
    const uint32_t w0 = kv_iq4nl_lut4((uint32_t)qv);
    const uint32_t w1 = kv_iq4nl_lut4((uint32_t)(qv >> 32));
    uint32_t xl, xh;
    memcpy(&xl, xq + (lane & 3u) * 8u, 4);
    memcpy(&xh, xq + (lane & 3u) * 8u + 4u, 4);
    int32_t s = 0;
    s = __builtin_amdgcn_sudot4(true, (int)w0, true, (int)xl, s, false);
    s = __builtin_amdgcn_sudot4(true, (int)w1, true, (int)xh, s, false);
    return s;
}

/* q2_0 int8: lane covers elements byte*4..+3 of a 64-element block.  The
 * 2-bit field map {0,1,2,3} -> {-1,0,1,2} is a 4-entry byte LUT. */
__device__ static int32_t wgd_q20_i8(const uint8_t *blk, uint32_t xu,
                                     uint32_t byte, float *dp) {
    *dp = f16d(blk);
    const uint32_t f = blk[2 + byte];
    const uint32_t sel = (f & 3u) | (((f >> 2) & 3u) << 8) |
                         (((f >> 4) & 3u) << 16) | (((f >> 6) & 3u) << 24);
    const uint32_t w = (uint32_t)__builtin_amdgcn_perm(0, 0x020100FFu, sel);
    int32_t s = 0;
    s = __builtin_amdgcn_sudot4(true, (int)w, true, (int)xu, s, false);
    return s;
}

/* Sign-byte -> per-byte 0xFF masks.  M = bits{7,14,21,28} spreads low 4
 * bits to byte MSBs; >>7 then *0xFF widens them.  No packed-negate exists
 * on gfx11, so signed-magnitude octs split into two masked u8 dots. */
__device__ static inline void sg_masks(uint32_t sg, uint32_t *m0,
                                       uint32_t *m1) {
    *m0 = ((((sg & 0x0Fu) * 0x10204080u) & 0x80808080u) >> 7) * 0xFFu;
    *m1 = (((((sg >> 4) & 0x0Fu) * 0x10204080u) & 0x80808080u) >> 7) * 0xFFu;
}

/* Oct dot for sign-magnitude grids: gv holds 8 unsigned magnitudes, sg
 * marks the negative bytes.  s = dot(w&~m, x) - dot(w&m, x). */
__device__ static inline int32_t dot8_sg(uint64_t gv, uint32_t sg,
                                         const int8_t *xq) {
    uint32_t m0, m1, xl, xh;
    sg_masks(sg, &m0, &m1);
    memcpy(&xl, xq, 4);
    memcpy(&xh, xq + 4, 4);
    const uint32_t w0 = (uint32_t)gv, w1 = (uint32_t)(gv >> 32);
    int32_t p = 0, n = 0;
    p = __builtin_amdgcn_sudot4(false, (int)(w0 & ~m0), true, (int)xl, p,
                                false);
    p = __builtin_amdgcn_sudot4(false, (int)(w1 & ~m1), true, (int)xh, p,
                                false);
    n = __builtin_amdgcn_sudot4(false, (int)(w0 & m0), true, (int)xl, n,
                                false);
    n = __builtin_amdgcn_sudot4(false, (int)(w1 & m1), true, (int)xh, n,
                                false);
    return p - n;
}

/* iq2_s int8: same lane geometry as wgd_iq2s (oct 8*lane..+7). */
__device__ static int32_t wgd_iq2s_i8(const uint8_t *blk, const int8_t *xq,
                                      uint32_t lane, float *dp) {
    const float d = f16d(blk);
    const uint32_t ib = lane >> 2, l = lane & 3u;
    const float sc = (l < 2) ? (float)(blk[74 + ib] & 0xf)
                             : (float)(blk[74 + ib] >> 4);
    *dp = d * (0.5f + sc) * 0.25f;
    const int idx = blk[2 + ib * 4 + l] |
                    ((blk[66 + ib] << (8 - 2 * l)) & 0x300);
    const uint64_t gv = q4_iq2s_grid[idx];
    const uint8_t sg = blk[2 + 32 + ib * 4 + l];
    return dot8_sg(gv, sg, xq + lane * 8u);
}

/* iq3_s int8: same lane geometry as wgd_iq3s. */
__device__ static int32_t wgd_iq3s_i8(const uint8_t *blk, const int8_t *xq,
                                      uint32_t lane, float *dp) {
    const float d = f16d(blk);
    const uint32_t p = lane >> 3, half = (lane >> 2) & 1u, l = lane & 3u;
    const float sc = (half == 0) ? (float)(blk[106 + p] & 0xf)
                                 : (float)(blk[106 + p] >> 4);
    *dp = d * (1.f + 2.f * sc);
    const uint8_t qhb = blk[66 + p * 2 + half];
    const int i1 = blk[2 + p * 16 + half * 8 + 2 * l] |
                   ((qhb << (8 - 2 * l)) & 256);
    const int i2 = blk[2 + p * 16 + half * 8 + 2 * l + 1] |
                   ((qhb << (7 - 2 * l)) & 256);
    uint64_t gv = (uint64_t)q4_iq3s_grid[i2] << 32 | q4_iq3s_grid[i1];
    const uint8_t sg = blk[74 + p * 8 + half * 4 + l];
    return dot8_sg(gv, sg, xq + lane * 8u);
}

/* iq3_xxs int8: same lane geometry as wgd_iq3xxs. */
__device__ static int32_t wgd_iq3xxs_i8(const uint8_t *blk, const int8_t *xq,
                                        uint32_t lane, float *dp) {
    const float d = f16d(blk);
    const uint32_t ib = lane >> 2, l = lane & 3u;
    uint32_t aux32;
    memcpy(&aux32, blk + 2 + 64 + 4 * ib, 4);
    *dp = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
    const uint8_t sg = q4_ksigns_iq2xs[(aux32 >> (7 * l)) & 127u];
    const uint8_t *q3 = blk + 2 + ib * 8;
    uint64_t gv = (uint64_t)q4_iq3xxs_grid[q3[2 * l + 1]] << 32 |
                  q4_iq3xxs_grid[q3[2 * l]];
    return dot8_sg(gv, sg, xq + lane * 8u);
}

/* iq4_xs int8: same lane geometry as wgd_iq4xs — nibble LUT like iq4_nl,
 * no sign byte. */
__device__ static int32_t wgd_iq4xs_i8(const uint8_t *blk, const int8_t *xq,
                                       uint32_t lane, float *dp) {
    const float d = f16d(blk);
    const uint32_t ib = lane >> 2, q = lane & 3u;
    uint32_t sh;
    memcpy(&sh, blk + 2, 2);
    const int ls = ((blk[4 + (ib >> 1)] >> (4 * (ib & 1u))) & 0xf) |
                   ((int)((sh >> (2 * ib)) & 3u) << 4);
    *dp = d * (float)(ls - 32);
    uint64_t qv;
    memcpy(&qv, blk + 8 + ib * 16 + (q & 1u) * 8u, 8);
    if (q >> 1) qv >>= 4;
    const uint32_t w0 = kv_iq4nl_lut4((uint32_t)qv);
    const uint32_t w1 = kv_iq4nl_lut4((uint32_t)(qv >> 32));
    uint32_t xl, xh;
    memcpy(&xl, xq + lane * 8u, 4);
    memcpy(&xh, xq + lane * 8u + 4u, 4);
    int32_t s = 0;
    s = __builtin_amdgcn_sudot4(true, (int)w0, true, (int)xl, s, false);
    s = __builtin_amdgcn_sudot4(true, (int)w1, true, (int)xh, s, false);
    return s;
}

/* f32: lane covers 8 consecutive floats (2x float4); warp = 256/iter. */
__device__ static float wgd_f32o(const float *wr, const float *x) {
    const float4 a = *(const float4 *)(wr);
    const float4 b = *(const float4 *)(wr + 4);
    const float4 xa = *(const float4 *)(x);
    const float4 xb = *(const float4 *)(x + 4);
    return a.x * xa.x + a.y * xa.y + a.z * xa.z + a.w * xa.w +
           b.x * xb.x + b.y * xb.y + b.z * xb.z + b.w * xb.w;
}

#define Q4_WGV_ROW()                                                           \
    uint32_t row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);        \
    uint32_t lane = threadIdx.x & 31u;                                         \
    if (row >= out_dim) return

/* Per-format warp-per-row kernels: one output row per warp. */
__global__ static void wgv_q8_0_k(float *out, const uint8_t *w, const float *x,
                                  uint32_t in_dim, uint32_t out_dim,
                                  float scale, float act) {
    Q4_WGV_ROW();
    const uint32_t nb = in_dim >> 5;
    const uint8_t *wr = w + (size_t)row * nb * 34ull;
    const uint32_t b = lane >> 1, m = (lane & 1u) * 16u;
    float acc = 0.f;
    for (uint32_t b0 = 0; b0 < nb; b0 += 16u) {
        uint32_t bb = b0 + b;
        if (bb < nb)
            acc += wgd_q8h(wr + (size_t)bb * 34ull, x + bb * 32u, m);
    }
    acc = wgv_shfl_red(acc);
    if (lane == 0) {
        float v = acc * scale;
        if (act != 0.f) {
            v *= act;
            v = v / (1.f + expf(-v));
        }
        out[row] = v;
    }
}

#define Q4_WGV_IQ256(name, octfn, bb)                                          \
    __global__ static void name(float *out, const uint8_t *w, const float *x,  \
                                uint32_t in_dim, uint32_t out_dim,             \
                                float scale) {                                 \
        Q4_WGV_ROW();                                                          \
        uint32_t nb = in_dim >> 8;                                             \
        const uint8_t *wr = w + (size_t)row * nb * (bb);                       \
        float acc = 0.f;                                                       \
        for (uint32_t b0 = 0; b0 < nb; b0 += 2u) {                             \
            acc += octfn(wr + (size_t)b0 * (bb),                               \
                         x + b0 * 256u + lane * 8u, lane);                     \
            if (b0 + 1 < nb)                                                   \
                acc += octfn(wr + (size_t)(b0 + 1) * (bb),                     \
                             x + (b0 + 1) * 256u + lane * 8u, lane);           \
        }                                                                      \
        acc = wgv_shfl_red(acc);                                               \
        if (lane == 0) out[row] = acc * scale;                                 \
    }
Q4_WGV_IQ256(wgv_iq2_s_k, wgd_iq2s, 82ull)
Q4_WGV_IQ256(wgv_iq3_s_k, wgd_iq3s, 110ull)
Q4_WGV_IQ256(wgv_iq3xxs_k, wgd_iq3xxs, 98ull)
Q4_WGV_IQ256(wgv_iq4xs_k, wgd_iq4xs, 136ull)

__global__ static void wgv_iq4_nl_k(float *out, const uint8_t *w,
                                    const float *x, uint32_t in_dim,
                                    uint32_t out_dim, float scale) {
    Q4_WGV_ROW();
    const uint32_t nb = in_dim >> 5;
    const uint8_t *wr = w + (size_t)row * nb * 18ull;
    const uint32_t b = lane >> 2;
    float acc = 0.f;
    for (uint32_t b0 = 0; b0 < nb; b0 += 8u) {
        uint32_t bb = b0 + b;
        if (bb < nb)
            acc += wgd_iq4nl(wr + (size_t)bb * 18ull,
                             x + bb * 32u + (lane & 3u) * 8u, lane);
    }
    acc = wgv_shfl_red(acc);
    if (lane == 0) out[row] = acc * scale;
}

__global__ static void wgv_f32_k(float *out, const float *w, const float *x,
                                 uint32_t in_dim, uint32_t out_dim,
                                 float scale) {
    Q4_WGV_ROW();
    const float *wr = w + (size_t)row * in_dim;
    float acc = 0.f;
    uint32_t e0 = lane * 8u;
    for (; e0 + 8u <= in_dim; e0 += 256u)
        acc += wgd_f32o(wr + e0, x + e0);
    for (uint32_t e = (in_dim & ~7u) + lane; e < in_dim; e += 32u)
        acc += wr[e] * x[e];
    acc = wgv_shfl_red(acc);
    if (lane == 0) out[row] = acc * scale;
}

/* Warp-per-row Q6_K decode GEMV (the lm_head — the largest GEMV per token).
 * Same math as gemv_q6_k_k with the 16-elem units strided across lanes and a
 * shuffle reduce; the block version serialised 16 units over 128 threads and
 * paid a full shared-memory reduction per row. */
__global__ static void wgv_q6_k_k(float *out, const uint8_t *w, const float *x,
                                  uint32_t in_dim, uint32_t out_dim,
                                  float scale) {
    Q4_WGV_ROW();
    uint32_t nb = in_dim / 256u;
    const uint8_t *wr = w + (size_t)row * nb * 210u;
    float acc = 0.f;
    uint32_t nu = nb * 16u;
    for (uint32_t u = lane; u < nu; u += 32u) {
        const uint8_t *blk = wr + (size_t)(u >> 4) * 210u;
        uint32_t qu = u & 15u;
        uint32_t h2 = qu >> 3, l4 = (qu & 7u) * 4u;
        uint32_t qlv0 = *(const uint32_t *)(blk + h2 * 64u + l4);
        uint32_t qlv1 = *(const uint32_t *)(blk + h2 * 64u + l4 + 32u);
        uint32_t qhv = *(const uint32_t *)(blk + 128u + h2 * 32u + l4);
        const int8_t *sc = (const int8_t *)(blk + 192u) + h2 * 8;
        float d = f16d(blk + 208u);
        uint32_t is = l4 >> 4;
        const float *xt4 = x + (u >> 4) * 256u + h2 * 128u + l4;
        for (uint32_t j = 0; j < 4; j++) {
            uint32_t qlo = (qlv0 >> (j * 8)) & 0xffu;
            uint32_t qlh = (qlv1 >> (j * 8)) & 0xffu;
            uint32_t qhb = (qhv >> (j * 8)) & 0xffu;
            acc += d * (float)sc[is] *
                   (float)(int((qlo & 15u) | ((qhb & 3u) << 4)) - 32) *
                   xt4[j];
            acc += d * (float)sc[is + 2u] *
                   (float)(int((qlh & 15u) | (((qhb >> 2) & 3u) << 4)) - 32) *
                   xt4[j + 32u];
            acc += d * (float)sc[is + 4u] *
                   (float)(int((qlo >> 4) | (((qhb >> 4) & 3u) << 4)) - 32) *
                   xt4[j + 64u];
            acc += d * (float)sc[is + 6u] *
                   (float)(int((qlh >> 4) | (((qhb >> 6) & 3u) << 4)) - 32) *
                   xt4[j + 96u];
        }
    }
    acc = wgv_shfl_red(acc);
    if (lane == 0) out[row] = acc * scale;
}

__device__ static void scale_min_k4_dev(int is, const uint8_t *sc,
                                        float *d1, float *m1);

/* q4_k/q5_k warp GEMV: 8 sub-blocks of 32 elems per 256-el superblock;
 * lane l covers 8 elements of sub-block (l>>2) at position (l&3)*8. */
__global__ static void wgv_q4_k_k(float *out, const uint8_t *w, const float *x,
                                  uint32_t in_dim, uint32_t out_dim,
                                  float scale) {
    Q4_WGV_ROW();
    const uint32_t nb = in_dim / 256u;
    const uint8_t *wr = w + (size_t)row * nb * 144u;
    const uint32_t is = lane >> 2;            /* sub-block 0..7 */
    const uint32_t j64 = is >> 1, half = is & 1u;
    const uint32_t pos = (lane & 3u) * 8u;    /* 8 elems inside sub-block */
    const uint32_t xoff = j64 * 64u + half * 32u + pos;
    float acc = 0.f;
    for (uint32_t b = 0; b < nb; b++) {
        const uint8_t *blk = wr + (size_t)b * 144u;
        float d1, m1;
        scale_min_k4_dev((int)is, blk + 4u, &d1, &m1);
        const uint8_t *qs = blk + 16u + j64 * 32u + pos;
        const float *xb = x + b * 256u + xoff;
        float ds = 0.f, xs = 0.f;
#pragma unroll
        for (uint32_t j = 0; j < 8; j++) {
            const uint32_t q = half ? (uint32_t)(qs[j] >> 4)
                                    : (uint32_t)(qs[j] & 0xFu);
            ds += (float)q * xb[j];
            xs += xb[j];
        }
        acc += d1 * ds - m1 * xs;
    }
    acc = wgv_shfl_red(acc);
    if (lane == 0) out[row] = acc * scale;
}

__global__ static void wgv_q5_k_k(float *out, const uint8_t *w, const float *x,
                                  uint32_t in_dim, uint32_t out_dim,
                                  float scale) {
    Q4_WGV_ROW();
    const uint32_t nb = in_dim / 256u;
    const uint8_t *wr = w + (size_t)row * nb * 176u;
    const uint32_t is = lane >> 2;
    const uint32_t j64 = is >> 1, half = is & 1u;
    const uint32_t pos = (lane & 3u) * 8u;
    const uint32_t xoff = j64 * 64u + half * 32u + pos;
    float acc = 0.f;
    for (uint32_t b = 0; b < nb; b++) {
        const uint8_t *blk = wr + (size_t)b * 176u;
        float d1, m1;
        scale_min_k4_dev((int)is, blk + 4u, &d1, &m1);
        const uint8_t *ql = blk + 48u + j64 * 32u + pos;
        const uint8_t *qh = blk + 16u + pos;
        const float *xb = x + b * 256u + xoff;
        float ds = 0.f, xs = 0.f;
#pragma unroll
        for (uint32_t j = 0; j < 8; j++) {
            const uint32_t q = ((half ? (uint32_t)(ql[j] >> 4)
                                     : (uint32_t)(ql[j] & 0xFu))) |
                               (((uint32_t)(qh[j] >> is) & 1u) << 4);
            ds += (float)q * xb[j];
            xs += xb[j];
        }
        acc += d1 * ds - m1 * xs;
    }
    acc = wgv_shfl_red(acc);
    if (lane == 0) out[row] = acc * scale;
}

/* 32-value-block warp GEMV (q4_0/q5_0): same geometry as wgv_iq4_nl_k —
 * lane covers a quarter of each block, 8 blocks per warp-iter. */
#define Q4_WGV_B32(name, octfn, bb)                                            \
    __global__ static void name(float *out, const uint8_t *w, const float *x,  \
                                uint32_t in_dim, uint32_t out_dim,             \
                                float scale) {                                 \
        Q4_WGV_ROW();                                                          \
        const uint32_t nb = in_dim >> 5;                                       \
        const uint8_t *wr = w + (size_t)row * nb * (bb);                       \
        const uint32_t b = lane >> 2;                                          \
        float acc = 0.f;                                                       \
        for (uint32_t b0 = 0; b0 < nb; b0 += 8u) {                             \
            uint32_t bb2 = b0 + b;                                             \
            if (bb2 < nb)                                                      \
                acc += octfn(wr + (size_t)bb2 * (bb),                          \
                             x + bb2 * 32u + (lane & 3u) * 8u, lane);          \
        }                                                                      \
        acc = wgv_shfl_red(acc);                                               \
        if (lane == 0) out[row] = acc * scale;                                 \
    }
Q4_WGV_B32(wgv_q4_0_k, wgd_q40, 18ull)
Q4_WGV_B32(wgv_q5_0_k, wgd_q50, 22ull)

/* q2_0: 64-elem blocks, one byte (4 elems) per lane, 2 blocks/warp-iter. */
__global__ static void wgv_q2_0_k(float *out, const uint8_t *w, const float *x,
                                  uint32_t in_dim, uint32_t out_dim,
                                  float scale) {
    Q4_WGV_ROW();
    const uint32_t nb = in_dim >> 6;
    const uint8_t *wr = w + (size_t)row * nb * 18ull;
    const uint32_t b = lane >> 4;
    const uint32_t byte = lane & 15u;
    float acc = 0.f;
    for (uint32_t b0 = 0; b0 < nb; b0 += 2u) {
        uint32_t bb = b0 + b;
        if (bb < nb)
            acc += wgd_q20(wr + (size_t)bb * 18ull,
                           x + bb * 64u + byte * 4u, byte);
    }
    acc = wgv_shfl_red(acc);
    if (lane == 0) out[row] = acc * scale;
}

/* ---- wgvn: warp-per-row for 2..8-token batches (MTP verify, short
 * prefill).  Each warp owns one output row and carries n_batch
 * accumulators; acc[t] receives adds in exactly the order the n_batch==1
 * wgv kernel applies them, so out[t*out_dim + row] is bit-identical to a
 * single-token call on x_t. ---- */
#define WGVN_MAX 8u
#define Q4_WGVN_EPILOG()                                                       \
    for (uint32_t t = 0; t < nt; t++) {                                        \
        float v = wgv_shfl_red(acc[t]);                                        \
        if (lane == 0) out[(size_t)t * out_dim + row] = v * scale;             \
    }

__global__ static void wgvn_q8_0_k(float *out, const uint8_t *w,
                                   const float *x, uint32_t in_dim,
                                   uint32_t out_dim, float scale,
                                   uint32_t nt) {
    Q4_WGV_ROW();
    const uint32_t nb = in_dim >> 5;
    const uint8_t *wr = w + (size_t)row * nb * 34ull;
    const uint32_t b = lane >> 1, m = (lane & 1u) * 16u;
    float acc[WGVN_MAX];
    for (uint32_t t = 0; t < nt; t++) acc[t] = 0.f;
    for (uint32_t b0 = 0; b0 < nb; b0 += 16u) {
        uint32_t bb = b0 + b;
        if (bb < nb) {
            const uint8_t *blk = wr + (size_t)bb * 34ull;
            for (uint32_t t = 0; t < nt; t++)
                acc[t] += wgd_q8h(blk, x + (size_t)t * in_dim + bb * 32u, m);
        }
    }
    Q4_WGVN_EPILOG();
}

#define Q4_WGVN_IQ256(name, octfn, bb)                                         \
    __global__ static void name(float *out, const uint8_t *w, const float *x,  \
                                uint32_t in_dim, uint32_t out_dim,             \
                                float scale, uint32_t nt) {                    \
        Q4_WGV_ROW();                                                          \
        uint32_t nb = in_dim >> 8;                                             \
        const uint8_t *wr = w + (size_t)row * nb * (bb);                       \
        float acc[WGVN_MAX];                                                   \
        for (uint32_t t = 0; t < nt; t++) acc[t] = 0.f;                        \
        for (uint32_t b0 = 0; b0 < nb; b0 += 2u) {                             \
            const uint8_t *b0p = wr + (size_t)b0 * (bb);                       \
            const uint8_t *b1p = wr + (size_t)(b0 + 1u) * (bb);                \
            for (uint32_t t = 0; t < nt; t++) {                                \
                const float *xt = x + (size_t)t * in_dim + b0 * 256u +         \
                                    lane * 8u;                                 \
                acc[t] += octfn(b0p, xt, lane);                                \
                if (b0 + 1 < nb)                                               \
                    acc[t] += octfn(b1p, xt + 256u, lane);                     \
            }                                                                  \
        }                                                                      \
        Q4_WGVN_EPILOG();                                                      \
    }
Q4_WGVN_IQ256(wgvn_iq2_s_k, wgd_iq2s, 82ull)
Q4_WGVN_IQ256(wgvn_iq3_s_k, wgd_iq3s, 110ull)
Q4_WGVN_IQ256(wgvn_iq3xxs_k, wgd_iq3xxs, 98ull)
Q4_WGVN_IQ256(wgvn_iq4xs_k, wgd_iq4xs, 136ull)

/* 32-value-block geometry for iq4_nl / q4_0 / q5_0: lane covers a quarter
 * of each block, 8 blocks per warp-iter — same as wgv_iq4_nl_k/B32. */
#define Q4_WGVN_B32(name, octfn, bb)                                           \
    __global__ static void name(float *out, const uint8_t *w, const float *x,  \
                                uint32_t in_dim, uint32_t out_dim,             \
                                float scale, uint32_t nt) {                    \
        Q4_WGV_ROW();                                                          \
        const uint32_t nb = in_dim >> 5;                                       \
        const uint8_t *wr = w + (size_t)row * nb * (bb);                       \
        const uint32_t b = lane >> 2;                                          \
        float acc[WGVN_MAX];                                                   \
        for (uint32_t t = 0; t < nt; t++) acc[t] = 0.f;                        \
        for (uint32_t b0 = 0; b0 < nb; b0 += 8u) {                             \
            uint32_t bb2 = b0 + b;                                             \
            if (bb2 < nb) {                                                    \
                const uint8_t *blk = wr + (size_t)bb2 * (bb);                  \
                for (uint32_t t = 0; t < nt; t++)                              \
                    acc[t] += octfn(blk, x + (size_t)t * in_dim +              \
                                    bb2 * 32u + (lane & 3u) * 8u, lane);       \
            }                                                                  \
        }                                                                      \
        Q4_WGVN_EPILOG();                                                      \
    }
Q4_WGVN_B32(wgvn_iq4_nl_k, wgd_iq4nl, 18ull)
Q4_WGVN_B32(wgvn_q4_0_k, wgd_q40, 18ull)
Q4_WGVN_B32(wgvn_q5_0_k, wgd_q50, 22ull)

__global__ static void wgvn_q2_0_k(float *out, const uint8_t *w,
                                   const float *x, uint32_t in_dim,
                                   uint32_t out_dim, float scale,
                                   uint32_t nt) {
    Q4_WGV_ROW();
    const uint32_t nb = in_dim >> 6;
    const uint8_t *wr = w + (size_t)row * nb * 18ull;
    const uint32_t b = lane >> 4;
    const uint32_t byte = lane & 15u;
    float acc[WGVN_MAX];
    for (uint32_t t = 0; t < nt; t++) acc[t] = 0.f;
    for (uint32_t b0 = 0; b0 < nb; b0 += 2u) {
        uint32_t bb = b0 + b;
        if (bb < nb) {
            const uint8_t *blk = wr + (size_t)bb * 18ull;
            for (uint32_t t = 0; t < nt; t++)
                acc[t] += wgd_q20(blk, x + (size_t)t * in_dim + bb * 64u +
                                  byte * 4u, byte);
        }
    }
    Q4_WGVN_EPILOG();
}

__global__ static void wgvn_f32_k(float *out, const float *w, const float *x,
                                  uint32_t in_dim, uint32_t out_dim,
                                  float scale, uint32_t nt) {
    Q4_WGV_ROW();
    const float *wr = w + (size_t)row * in_dim;
    float acc[WGVN_MAX];
    for (uint32_t t = 0; t < nt; t++) acc[t] = 0.f;
    uint32_t e0 = lane * 8u;
    for (; e0 + 8u <= in_dim; e0 += 256u)
        for (uint32_t t = 0; t < nt; t++)
            acc[t] += wgd_f32o(wr + e0, x + (size_t)t * in_dim + e0);
    for (uint32_t e = (in_dim & ~7u) + lane; e < in_dim; e += 32u)
        for (uint32_t t = 0; t < nt; t++)
            acc[t] += wr[e] * x[(size_t)t * in_dim + e];
    Q4_WGVN_EPILOG();
}

__global__ static void wgvn_bf16_k(float *out, const uint8_t *w,
                                   const float *x, uint32_t in_dim,
                                   uint32_t out_dim, float scale,
                                   uint32_t nt) {
    Q4_WGV_ROW();
    const uint16_t *wr = (const uint16_t *)(w + (size_t)row * in_dim * 2u);
    float acc[WGVN_MAX];
    for (uint32_t t = 0; t < nt; t++) acc[t] = 0.f;
    if (!(in_dim & 1u) && (((uintptr_t)w & 3u) == 0)) {
        const uint32_t *wr32 = (const uint32_t *)wr;
        const uint32_t n2 = in_dim >> 1;
        for (uint32_t i = lane; i < n2; i += 32u) {
            uint32_t pr = wr32[i];
            float lo, hi;
            uint32_t ul = pr << 16, uh = pr & 0xffff0000u;
            memcpy(&lo, &ul, 4);
            memcpy(&hi, &uh, 4);
            for (uint32_t t = 0; t < nt; t++)
                acc[t] = fmaf(lo, x[(size_t)t * in_dim + 2u * i],
                              fmaf(hi, x[(size_t)t * in_dim + 2u * i + 1u],
                                   acc[t]));
        }
    } else {
        for (uint32_t i = lane; i < in_dim; i += 32u)
            for (uint32_t t = 0; t < nt; t++)
                acc[t] = fmaf(bf16_f(wr[i]), x[(size_t)t * in_dim + i],
                              acc[t]);
    }
    for (uint32_t t = 0; t < nt; t++) {
        for (uint32_t o = 16; o; o >>= 1)
            acc[t] += __shfl_xor_sync(0xffffffffull, acc[t], o);
        if (lane == 0) out[(size_t)t * out_dim + row] = acc[t] * scale;
    }
}

__global__ static void wgvn_q4_k_k(float *out, const uint8_t *w,
                                   const float *x, uint32_t in_dim,
                                   uint32_t out_dim, float scale,
                                   uint32_t nt) {
    Q4_WGV_ROW();
    const uint32_t nb = in_dim / 256u;
    const uint8_t *wr = w + (size_t)row * nb * 144u;
    const uint32_t is = lane >> 2;
    const uint32_t j64 = is >> 1, half = is & 1u;
    const uint32_t pos = (lane & 3u) * 8u;
    const uint32_t xoff = j64 * 64u + half * 32u + pos;
    float acc[WGVN_MAX];
    for (uint32_t t = 0; t < nt; t++) acc[t] = 0.f;
    for (uint32_t b = 0; b < nb; b++) {
        const uint8_t *blk = wr + (size_t)b * 144u;
        float d1, m1;
        scale_min_k4_dev((int)is, blk + 4u, &d1, &m1);
        const uint8_t *qs = blk + 16u + j64 * 32u + pos;
        for (uint32_t t = 0; t < nt; t++) {
            const float *xb = x + (size_t)t * in_dim + b * 256u + xoff;
            float ds = 0.f, xs = 0.f;
#pragma unroll
            for (uint32_t j = 0; j < 8; j++) {
                const uint32_t q = half ? (uint32_t)(qs[j] >> 4)
                                        : (uint32_t)(qs[j] & 0xFu);
                ds += (float)q * xb[j];
                xs += xb[j];
            }
            acc[t] += d1 * ds - m1 * xs;
        }
    }
    Q4_WGVN_EPILOG();
}

__global__ static void wgvn_q5_k_k(float *out, const uint8_t *w,
                                   const float *x, uint32_t in_dim,
                                   uint32_t out_dim, float scale,
                                   uint32_t nt) {
    Q4_WGV_ROW();
    const uint32_t nb = in_dim / 256u;
    const uint8_t *wr = w + (size_t)row * nb * 176u;
    const uint32_t is = lane >> 2;
    const uint32_t j64 = is >> 1, half = is & 1u;
    const uint32_t pos = (lane & 3u) * 8u;
    const uint32_t xoff = j64 * 64u + half * 32u + pos;
    float acc[WGVN_MAX];
    for (uint32_t t = 0; t < nt; t++) acc[t] = 0.f;
    for (uint32_t b = 0; b < nb; b++) {
        const uint8_t *blk = wr + (size_t)b * 176u;
        float d1, m1;
        scale_min_k4_dev((int)is, blk + 4u, &d1, &m1);
        const uint8_t *ql = blk + 48u + j64 * 32u + pos;
        const uint8_t *qh = blk + 16u + pos;
        for (uint32_t t = 0; t < nt; t++) {
            const float *xb = x + (size_t)t * in_dim + b * 256u + xoff;
            float ds = 0.f, xs = 0.f;
#pragma unroll
            for (uint32_t j = 0; j < 8; j++) {
                const uint32_t q = ((half ? (uint32_t)(ql[j] >> 4)
                                          : (uint32_t)(ql[j] & 0xFu))) |
                                   (((uint32_t)(qh[j] >> is) & 1u) << 4);
                ds += (float)q * xb[j];
                xs += xb[j];
            }
            acc[t] += d1 * ds - m1 * xs;
        }
    }
    Q4_WGVN_EPILOG();
}

/* Warp-per-(row, expert) for the batched routed-expert stages.  grid.y is a
 * small fixed stripe count (WMOE_GY): each block loops e += gridDim.y so the
 * launch stays small when the device-side count is only known at exec —
 * 64-way grids spent most of their block-dispatch time on early exits. */
#define WMOE_GY 8u
#define Q4_WMOE_IQ256(name, octfn, bb)                                         \
    __global__ static void name(float *out, const uint8_t *base,               \
                                uint64_t stride, const int32_t *slots,         \
                                const float *x, uint32_t xs,                   \
                                uint32_t in_dim, uint32_t out_dim,             \
                                const int32_t *n_d) {                          \
        const uint32_t lim = n_d ? (uint32_t)*n_d : gridDim.y;                 \
        uint32_t nb = in_dim >> 8;                                             \
        Q4_WGV_ROW();                                                          \
        for (uint32_t e = blockIdx.y; e < lim; e += gridDim.y) {               \
            const uint8_t *wr = base + (size_t)slots[e] * stride +             \
                                (size_t)row * nb * (bb);                       \
            const float *xt = x + (size_t)e * xs;                              \
            float acc = 0.f;                                                   \
            for (uint32_t b0 = 0; b0 < nb; b0 += 4u) {                         \
                acc += octfn(wr + (size_t)b0 * (bb),                           \
                             xt + b0 * 256u + lane * 8u, lane);                \
                if (b0 + 1 < nb)                                               \
                    acc += octfn(wr + (size_t)(b0 + 1) * (bb),                 \
                                 xt + (b0 + 1) * 256u + lane * 8u, lane);      \
                if (b0 + 2 < nb)                                               \
                    acc += octfn(wr + (size_t)(b0 + 2) * (bb),                 \
                                 xt + (b0 + 2) * 256u + lane * 8u, lane);      \
                if (b0 + 3 < nb)                                               \
                    acc += octfn(wr + (size_t)(b0 + 3) * (bb),                 \
                                 xt + (b0 + 3) * 256u + lane * 8u, lane);      \
            }                                                                  \
            acc = wgv_shfl_red(acc);                                           \
            if (lane == 0) out[(size_t)e * out_dim + row] = acc;               \
        }                                                                      \
    }
Q4_WMOE_IQ256(wmoe_iq2_bk, wgd_iq2s, 82ull)
Q4_WMOE_IQ256(wmoe_iq3_bk, wgd_iq3s, 110ull)
Q4_WMOE_IQ256(wmoe_iq3xxs_bk, wgd_iq3xxs, 98ull)
Q4_WMOE_IQ256(wmoe_iq4xs_bk, wgd_iq4xs, 136ull)

/* gate+up fused: one warp does both rows of the same expert, sharing the
 * slot lookup and x read — halves launches and fixed per-block cost. */
#define Q4_WMOE_GU(name, octfn, bb)                                            \
    __global__ static void name(float *og, float *ou, const uint8_t *bg,       \
                                const uint8_t *bu, uint64_t stride,            \
                                const int32_t *slots, const float *x,          \
                                uint32_t xs, uint32_t in_dim,                  \
                                uint32_t out_dim, const int32_t *n_d) {        \
        const uint32_t lim = n_d ? (uint32_t)*n_d : gridDim.y;                 \
        uint32_t nb = in_dim >> 8;                                             \
        Q4_WGV_ROW();                                                          \
        for (uint32_t e = blockIdx.y; e < lim; e += gridDim.y) {               \
            const uint8_t *wg = bg + (size_t)slots[e] * stride +               \
                                (size_t)row * nb * (bb);                       \
            const uint8_t *wu = bu + (size_t)slots[e] * stride +               \
                                (size_t)row * nb * (bb);                       \
            const float *xt = x + (size_t)e * xs;                              \
            float ag = 0.f, au = 0.f;                                          \
            for (uint32_t b0 = 0; b0 < nb; b0 += 4u) {                         \
                ag += octfn(wg + (size_t)b0 * (bb),                            \
                            xt + b0 * 256u + lane * 8u, lane);                 \
                au += octfn(wu + (size_t)b0 * (bb),                            \
                            xt + b0 * 256u + lane * 8u, lane);                 \
                if (b0 + 1 < nb) {                                             \
                    ag += octfn(wg + (size_t)(b0 + 1) * (bb),                  \
                                xt + (b0 + 1) * 256u + lane * 8u, lane);       \
                    au += octfn(wu + (size_t)(b0 + 1) * (bb),                  \
                                xt + (b0 + 1) * 256u + lane * 8u, lane);       \
                }                                                              \
                if (b0 + 2 < nb) {                                             \
                    ag += octfn(wg + (size_t)(b0 + 2) * (bb),                  \
                                xt + (b0 + 2) * 256u + lane * 8u, lane);       \
                    au += octfn(wu + (size_t)(b0 + 2) * (bb),                  \
                                xt + (b0 + 2) * 256u + lane * 8u, lane);       \
                }                                                              \
                if (b0 + 3 < nb) {                                             \
                    ag += octfn(wg + (size_t)(b0 + 3) * (bb),                  \
                                xt + (b0 + 3) * 256u + lane * 8u, lane);       \
                    au += octfn(wu + (size_t)(b0 + 3) * (bb),                  \
                                xt + (b0 + 3) * 256u + lane * 8u, lane);       \
                }                                                              \
            }                                                                  \
            ag = wgv_shfl_red(ag);                                             \
            au = wgv_shfl_red(au);                                             \
            if (lane == 0) {                                                   \
                og[(size_t)e * out_dim + row] = ag;                            \
                ou[(size_t)e * out_dim + row] = au;                            \
            }                                                                  \
        }                                                                      \
    }
Q4_WMOE_GU(wmoe_iq2_gu_bk, wgd_iq2s, 82ull)
Q4_WMOE_GU(wmoe_iq3_gu_bk, wgd_iq3s, 110ull)
Q4_WMOE_GU(wmoe_iq3xxs_gu_bk, wgd_iq3xxs, 98ull)
Q4_WMOE_GU(wmoe_iq4xs_gu_bk, wgd_iq4xs, 136ull)

/* int8-activation gate+up: same geometry, but x comes pre-quantized in
 * per-32 i8 groups (xq8_k).  The oct's activation scale is
 * xd8[b0*8 + lane/4]; the int8 partial is scaled by dl*dx once per oct. */
#define Q4_WMOE_GU8(name, octfn, bb)                                           \
    __global__ static void name(float *og, float *ou, const uint8_t *bg,       \
                                const uint8_t *bu, uint64_t stride,            \
                                const int32_t *slots, const int8_t *xq8,       \
                                const float *xd8, uint32_t in_dim,             \
                                uint32_t out_dim, const int32_t *n_d) {        \
        const uint32_t lim = n_d ? (uint32_t)*n_d : gridDim.y;                 \
        uint32_t nb = in_dim >> 8;                                             \
        Q4_WGV_ROW();                                                          \
        for (uint32_t e = blockIdx.y; e < lim; e += gridDim.y) {               \
            const uint8_t *wg = bg + (size_t)slots[e] * stride +               \
                                (size_t)row * nb * (bb);                       \
            const uint8_t *wu = bu + (size_t)slots[e] * stride +               \
                                (size_t)row * nb * (bb);                       \
            float ag = 0.f, au = 0.f;                                          \
            for (uint32_t b0 = 0; b0 < nb; b0 += 4u) {                         \
                const uint32_t bb0 = b0, bb1 = b0 + 1, bb2 = b0 + 2,           \
                               bb3 = b0 + 3;                                   \
                if (bb0 < nb) {                                                \
                    float dg, du;                                              \
                    int32_t sg = octfn(wg + (size_t)bb0 * (bb),                \
                                       xq8 + bb0 * 256u, lane, &dg);           \
                    int32_t su = octfn(wu + (size_t)bb0 * (bb),                \
                                       xq8 + bb0 * 256u, lane, &du);           \
                    const float dx = xd8[bb0 * 8u + (lane >> 2)];              \
                    ag += dg * dx * (float)sg;                                 \
                    au += du * dx * (float)su;                                 \
                }                                                              \
                if (bb1 < nb) {                                                \
                    float dg, du;                                              \
                    int32_t sg = octfn(wg + (size_t)bb1 * (bb),                \
                                       xq8 + bb1 * 256u, lane, &dg);           \
                    int32_t su = octfn(wu + (size_t)bb1 * (bb),                \
                                       xq8 + bb1 * 256u, lane, &du);           \
                    const float dx = xd8[bb1 * 8u + (lane >> 2)];              \
                    ag += dg * dx * (float)sg;                                 \
                    au += du * dx * (float)su;                                 \
                }                                                              \
                if (bb2 < nb) {                                                \
                    float dg, du;                                              \
                    int32_t sg = octfn(wg + (size_t)bb2 * (bb),                \
                                       xq8 + bb2 * 256u, lane, &dg);           \
                    int32_t su = octfn(wu + (size_t)bb2 * (bb),                \
                                       xq8 + bb2 * 256u, lane, &du);           \
                    const float dx = xd8[bb2 * 8u + (lane >> 2)];              \
                    ag += dg * dx * (float)sg;                                 \
                    au += du * dx * (float)su;                                 \
                }                                                              \
                if (bb3 < nb) {                                                \
                    float dg, du;                                              \
                    int32_t sg = octfn(wg + (size_t)bb3 * (bb),                \
                                       xq8 + bb3 * 256u, lane, &dg);           \
                    int32_t su = octfn(wu + (size_t)bb3 * (bb),                \
                                       xq8 + bb3 * 256u, lane, &du);           \
                    const float dx = xd8[bb3 * 8u + (lane >> 2)];              \
                    ag += dg * dx * (float)sg;                                 \
                    au += du * dx * (float)su;                                 \
                }                                                              \
            }                                                                  \
            ag = wgv_shfl_red(ag);                                             \
            au = wgv_shfl_red(au);                                             \
            if (lane == 0) {                                                   \
                og[(size_t)e * out_dim + row] = ag;                            \
                ou[(size_t)e * out_dim + row] = au;                            \
            }                                                                  \
        }                                                                      \
    }
Q4_WMOE_GU8(wmoe_iq2_gu_q8, wgd_iq2s_i8, 82ull)
Q4_WMOE_GU8(wmoe_iq3_gu_q8, wgd_iq3s_i8, 110ull)
Q4_WMOE_GU8(wmoe_iq3xxs_gu_q8, wgd_iq3xxs_i8, 98ull)
Q4_WMOE_GU8(wmoe_iq4xs_gu_q8, wgd_iq4xs_i8, 136ull)

__global__ static void wmoe_iq4_gu_bk(float *og, float *ou, const uint8_t *bg,
                                      const uint8_t *bu, uint64_t stride,
                                      const int32_t *slots, const float *x,
                                      uint32_t xs, uint32_t in_dim,
                                      uint32_t out_dim, const int32_t *n_d) {
    const uint32_t lim = n_d ? (uint32_t)*n_d : gridDim.y;
    const uint32_t nb = in_dim >> 5;
    Q4_WGV_ROW();
    for (uint32_t e = blockIdx.y; e < lim; e += gridDim.y) {
        const uint8_t *wg =
            bg + (size_t)slots[e] * stride + (size_t)row * nb * 18ull;
        const uint8_t *wu =
            bu + (size_t)slots[e] * stride + (size_t)row * nb * 18ull;
        const float *xt = x + (size_t)e * xs;
        const uint32_t b = lane >> 2;
        float ag = 0.f, au = 0.f;
        for (uint32_t b0 = 0; b0 < nb; b0 += 8u) {
            uint32_t bb = b0 + b;
            if (bb < nb) {
                ag += wgd_iq4nl(wg + (size_t)bb * 18ull,
                                xt + bb * 32u + (lane & 3u) * 8u, lane);
                au += wgd_iq4nl(wu + (size_t)bb * 18ull,
                                xt + bb * 32u + (lane & 3u) * 8u, lane);
            }
        }
        ag = wgv_shfl_red(ag);
        au = wgv_shfl_red(au);
        if (lane == 0) {
            og[(size_t)e * out_dim + row] = ag;
            ou[(size_t)e * out_dim + row] = au;
        }
    }
}

/* Down-projection with fused weighted accumulate: each row atomically adds
 * w[e]*dot into y, removing the n_expert×n_embd intermediate + wsum pass. */
#define Q4_WMOE_YA(name, octfn, bb)                                            \
    __global__ static void name(float *y, const uint8_t *base,                 \
                                uint64_t stride, const int32_t *slots,         \
                                const float *wts, const float *x,              \
                                uint32_t xs, uint32_t in_dim,                  \
                                uint32_t out_dim, const int32_t *n_d) {        \
        const uint32_t lim = n_d ? (uint32_t)*n_d : gridDim.y;                 \
        uint32_t nb = in_dim >> 8;                                             \
        Q4_WGV_ROW();                                                          \
        for (uint32_t e = blockIdx.y; e < lim; e += gridDim.y) {               \
            const uint8_t *wr = base + (size_t)slots[e] * stride +             \
                                (size_t)row * nb * (bb);                       \
            const float *xt = x + (size_t)e * xs;                              \
            float acc = 0.f;                                                   \
            for (uint32_t b0 = 0; b0 < nb; b0 += 4u) {                         \
                acc += octfn(wr + (size_t)b0 * (bb),                           \
                             xt + b0 * 256u + lane * 8u, lane);                \
                if (b0 + 1 < nb)                                               \
                    acc += octfn(wr + (size_t)(b0 + 1) * (bb),                 \
                                 xt + (b0 + 1) * 256u + lane * 8u, lane);      \
                if (b0 + 2 < nb)                                               \
                    acc += octfn(wr + (size_t)(b0 + 2) * (bb),                 \
                                 xt + (b0 + 2) * 256u + lane * 8u, lane);      \
                if (b0 + 3 < nb)                                               \
                    acc += octfn(wr + (size_t)(b0 + 3) * (bb),                 \
                                 xt + (b0 + 3) * 256u + lane * 8u, lane);      \
            }                                                                  \
            acc = wgv_shfl_red(acc);                                           \
            if (lane == 0) atomicAdd(y + row, acc * wts[e]);                   \
        }                                                                      \
    }
Q4_WMOE_YA(wmoe_iq2_ya_bk, wgd_iq2s, 82ull)
Q4_WMOE_YA(wmoe_iq3_ya_bk, wgd_iq3s, 110ull)
Q4_WMOE_YA(wmoe_iq3xxs_ya_bk, wgd_iq3xxs, 98ull)
Q4_WMOE_YA(wmoe_iq4xs_ya_bk, wgd_iq4xs, 136ull)

/* int8-activation IQ256 down-proj: per-expert quantized x slices; oct's
 * activation scale is xd8[e][b0*8 + lane/4]. */
#define Q4_WMOE_YA8(name, octfn, bb)                                           \
    __global__ static void name(float *y, const uint8_t *base,                 \
                                uint64_t stride, const int32_t *slots,         \
                                const float *wts, const int8_t *xq8,           \
                                const float *xd8, uint32_t in_dim,             \
                                uint32_t out_dim, const int32_t *n_d) {        \
        const uint32_t lim = n_d ? (uint32_t)*n_d : gridDim.y;                 \
        uint32_t nb = in_dim >> 8;                                             \
        Q4_WGV_ROW();                                                          \
        for (uint32_t e = blockIdx.y; e < lim; e += gridDim.y) {               \
            const uint8_t *wr = base + (size_t)slots[e] * stride +             \
                                (size_t)row * nb * (bb);                       \
            const int8_t *xq = xq8 + (size_t)e * in_dim;                       \
            const float *xd = xd8 + (size_t)e * (in_dim >> 5);                 \
            float acc = 0.f;                                                   \
            for (uint32_t b0 = 0; b0 < nb; b0 += 4u) {                         \
                for (uint32_t k = 0; k < 4u; k++) {                            \
                    const uint32_t bi = b0 + k;                                \
                    if (bi < nb) {                                             \
                        float dw;                                              \
                        const int32_t s = octfn(wr + (size_t)bi * (bb),        \
                                                xq + bi * 256u, lane, &dw);    \
                        acc += dw * xd[bi * 8u + (lane >> 2)] * (float)s;      \
                    }                                                          \
                }                                                              \
            }                                                                  \
            acc = wgv_shfl_red(acc);                                           \
            if (lane == 0) atomicAdd(y + row, acc * wts[e]);                   \
        }                                                                      \
    }
Q4_WMOE_YA8(wmoe_iq2_ya_q8, wgd_iq2s_i8, 82ull)
Q4_WMOE_YA8(wmoe_iq3_ya_q8, wgd_iq3s_i8, 110ull)
Q4_WMOE_YA8(wmoe_iq3xxs_ya_q8, wgd_iq3xxs_i8, 98ull)
Q4_WMOE_YA8(wmoe_iq4xs_ya_q8, wgd_iq4xs_i8, 136ull)

__global__ static void wmoe_iq4_ya_bk(float *y, const uint8_t *base,
                                      uint64_t stride, const int32_t *slots,
                                      const float *wts, const float *x,
                                      uint32_t xs, uint32_t in_dim,
                                      uint32_t out_dim, const int32_t *n_d) {
    const uint32_t lim = n_d ? (uint32_t)*n_d : gridDim.y;
    const uint32_t nb = in_dim >> 5;
    Q4_WGV_ROW();
    for (uint32_t e = blockIdx.y; e < lim; e += gridDim.y) {
        const uint8_t *wr = base + (size_t)slots[e] * stride +
                            (size_t)row * nb * 18ull;
        const float *xt = x + (size_t)e * xs;
        const uint32_t b = lane >> 2;
        float acc = 0.f;
        for (uint32_t b0 = 0; b0 < nb; b0 += 8u) {
            uint32_t bb = b0 + b;
            if (bb < nb)
                acc += wgd_iq4nl(wr + (size_t)bb * 18ull,
                                 xt + bb * 32u + (lane & 3u) * 8u, lane);
        }
        acc = wgv_shfl_red(acc);
        if (lane == 0) atomicAdd(y + row, acc * wts[e]);
    }
}

__global__ static void wmoe_iq4_bk(float *out, const uint8_t *base,
                                   uint64_t stride, const int32_t *slots,
                                   const float *x, uint32_t xs,
                                   uint32_t in_dim, uint32_t out_dim,
                                   const int32_t *n_d) {
    const uint32_t lim = n_d ? (uint32_t)*n_d : gridDim.y;
    const uint32_t nb = in_dim >> 5;
    Q4_WGV_ROW();
    for (uint32_t e = blockIdx.y; e < lim; e += gridDim.y) {
        const uint8_t *wr = base + (size_t)slots[e] * stride +
                            (size_t)row * nb * 18ull;
        const float *xt = x + (size_t)e * xs;
        const uint32_t b = lane >> 2;
        float acc = 0.f;
        for (uint32_t b0 = 0; b0 < nb; b0 += 8u) {
            uint32_t bb = b0 + b;
            if (bb < nb)
                acc += wgd_iq4nl(wr + (size_t)bb * 18ull,
                                 xt + bb * 32u + (lane & 3u) * 8u, lane);
        }
        acc = wgv_shfl_red(acc);
        if (lane == 0) out[(size_t)e * out_dim + row] = acc;
    }
}

/* 32-value-block expert variants (q4_0/q5_0): same shape as wmoe_iq4_*. */
#define Q4_WMOE_B32(name, octfn, bb)                                           \
    __global__ static void name(float *out, const uint8_t *base,               \
                                uint64_t stride, const int32_t *slots,         \
                                const float *x, uint32_t xs,                   \
                                uint32_t in_dim, uint32_t out_dim,             \
                                const int32_t *n_d) {                          \
        const uint32_t lim = n_d ? (uint32_t)*n_d : gridDim.y;                 \
        const uint32_t nb = in_dim >> 5;                                       \
        Q4_WGV_ROW();                                                          \
        for (uint32_t e = blockIdx.y; e < lim; e += gridDim.y) {               \
            const uint8_t *wr = base + (size_t)slots[e] * stride +             \
                                (size_t)row * nb * (bb);                       \
            const float *xt = x + (size_t)e * xs;                              \
            const uint32_t b = lane >> 2;                                      \
            float acc = 0.f;                                                   \
            for (uint32_t b0 = 0; b0 < nb; b0 += 8u) {                         \
                uint32_t bb2 = b0 + b;                                         \
                if (bb2 < nb)                                                  \
                    acc += octfn(wr + (size_t)bb2 * (bb),                      \
                                 xt + bb2 * 32u + (lane & 3u) * 8u, lane);     \
            }                                                                  \
            acc = wgv_shfl_red(acc);                                           \
            if (lane == 0) out[(size_t)e * out_dim + row] = acc;               \
        }                                                                      \
    }
Q4_WMOE_B32(wmoe_q40_bk, wgd_q40, 18ull)
Q4_WMOE_B32(wmoe_q50_bk, wgd_q50, 22ull)

#define Q4_WMOE_GU32(name, octfn, bb)                                          \
    __global__ static void name(float *og, float *ou, const uint8_t *bg,       \
                                const uint8_t *bu, uint64_t stride,            \
                                const int32_t *slots, const float *x,          \
                                uint32_t xs, uint32_t in_dim,                  \
                                uint32_t out_dim, const int32_t *n_d) {        \
        const uint32_t lim = n_d ? (uint32_t)*n_d : gridDim.y;                 \
        const uint32_t nb = in_dim >> 5;                                       \
        Q4_WGV_ROW();                                                          \
        for (uint32_t e = blockIdx.y; e < lim; e += gridDim.y) {               \
            const uint8_t *wg = bg + (size_t)slots[e] * stride +               \
                                (size_t)row * nb * (bb);                       \
            const uint8_t *wu = bu + (size_t)slots[e] * stride +               \
                                (size_t)row * nb * (bb);                       \
            const float *xt = x + (size_t)e * xs;                              \
            const uint32_t b = lane >> 2;                                      \
            float ag = 0.f, au = 0.f;                                          \
            for (uint32_t b0 = 0; b0 < nb; b0 += 8u) {                         \
                uint32_t bb2 = b0 + b;                                         \
                if (bb2 < nb) {                                                \
                    ag += octfn(wg + (size_t)bb2 * (bb),                       \
                                xt + bb2 * 32u + (lane & 3u) * 8u, lane);      \
                    au += octfn(wu + (size_t)bb2 * (bb),                       \
                                xt + bb2 * 32u + (lane & 3u) * 8u, lane);      \
                }                                                              \
            }                                                                  \
            ag = wgv_shfl_red(ag);                                             \
            au = wgv_shfl_red(au);                                             \
            if (lane == 0) {                                                   \
                og[(size_t)e * out_dim + row] = ag;                            \
                ou[(size_t)e * out_dim + row] = au;                            \
            }                                                                  \
        }                                                                      \
    }
Q4_WMOE_GU32(wmoe_q40_gu_bk, wgd_q40, 18ull)
Q4_WMOE_GU32(wmoe_q50_gu_bk, wgd_q50, 22ull)

#define Q4_WMOE_YA32(name, octfn, bb)                                          \
    __global__ static void name(float *y, const uint8_t *base,                 \
                                uint64_t stride, const int32_t *slots,         \
                                const float *wts, const float *x,              \
                                uint32_t xs, uint32_t in_dim,                  \
                                uint32_t out_dim, const int32_t *n_d) {        \
        const uint32_t lim = n_d ? (uint32_t)*n_d : gridDim.y;                 \
        const uint32_t nb = in_dim >> 5;                                       \
        Q4_WGV_ROW();                                                          \
        for (uint32_t e = blockIdx.y; e < lim; e += gridDim.y) {               \
            const uint8_t *wr = base + (size_t)slots[e] * stride +             \
                                (size_t)row * nb * (bb);                       \
            const float *xt = x + (size_t)e * xs;                              \
            const uint32_t b = lane >> 2;                                      \
            float acc = 0.f;                                                   \
            for (uint32_t b0 = 0; b0 < nb; b0 += 8u) {                         \
                uint32_t bb2 = b0 + b;                                         \
                if (bb2 < nb)                                                  \
                    acc += octfn(wr + (size_t)bb2 * (bb),                      \
                                 xt + bb2 * 32u + (lane & 3u) * 8u, lane);     \
            }                                                                  \
            acc = wgv_shfl_red(acc);                                           \
            if (lane == 0) atomicAdd(y + row, acc * wts[e]);                   \
        }                                                                      \
    }
Q4_WMOE_YA32(wmoe_q40_ya_bk, wgd_q40, 18ull)
Q4_WMOE_YA32(wmoe_q50_ya_bk, wgd_q50, 22ull)

/* q2_0 experts: 64-elem blocks, one byte (4 elems) per lane, 2 blocks per
 * warp-iter (lane>>4 picks the block, lane&15 the byte). */
__global__ static void wmoe_q20_bk(float *out, const uint8_t *base,
                                   uint64_t stride, const int32_t *slots,
                                   const float *x, uint32_t xs,
                                   uint32_t in_dim, uint32_t out_dim,
                                   const int32_t *n_d) {
    const uint32_t lim = n_d ? (uint32_t)*n_d : gridDim.y;
    const uint32_t nb = in_dim >> 6;
    Q4_WGV_ROW();
    for (uint32_t e = blockIdx.y; e < lim; e += gridDim.y) {
        const uint8_t *wr = base + (size_t)slots[e] * stride +
                            (size_t)row * nb * 18ull;
        const float *xt = x + (size_t)e * xs;
        const uint32_t b = lane >> 4, byte = lane & 15u;
        float acc = 0.f;
        for (uint32_t b0 = 0; b0 < nb; b0 += 2u) {
            uint32_t bb = b0 + b;
            if (bb < nb)
                acc += wgd_q20(wr + (size_t)bb * 18ull,
                               xt + bb * 64u + byte * 4u, byte);
        }
        acc = wgv_shfl_red(acc);
        if (lane == 0) out[(size_t)e * out_dim + row] = acc;
    }
}

__global__ static void wmoe_q20_gu_bk(float *og, float *ou, const uint8_t *bg,
                                      const uint8_t *bu, uint64_t stride,
                                      const int32_t *slots, const float *x,
                                      uint32_t xs, uint32_t in_dim,
                                      uint32_t out_dim, const int32_t *n_d) {
    const uint32_t lim = n_d ? (uint32_t)*n_d : gridDim.y;
    const uint32_t nb = in_dim >> 6;
    Q4_WGV_ROW();
    for (uint32_t e = blockIdx.y; e < lim; e += gridDim.y) {
        const uint8_t *wg = bg + (size_t)slots[e] * stride +
                            (size_t)row * nb * 18ull;
        const uint8_t *wu = bu + (size_t)slots[e] * stride +
                            (size_t)row * nb * 18ull;
        const float *xt = x + (size_t)e * xs;
        const uint32_t b = lane >> 4, byte = lane & 15u;
        float ag = 0.f, au = 0.f;
        for (uint32_t b0 = 0; b0 < nb; b0 += 2u) {
            uint32_t bb = b0 + b;
            if (bb < nb) {
                ag += wgd_q20(wg + (size_t)bb * 18ull,
                              xt + bb * 64u + byte * 4u, byte);
                au += wgd_q20(wu + (size_t)bb * 18ull,
                              xt + bb * 64u + byte * 4u, byte);
            }
        }
        ag = wgv_shfl_red(ag);
        au = wgv_shfl_red(au);
        if (lane == 0) {
            og[(size_t)e * out_dim + row] = ag;
            ou[(size_t)e * out_dim + row] = au;
        }
    }
}

__global__ static void wmoe_q20_ya_bk(float *y, const uint8_t *base,
                                      uint64_t stride, const int32_t *slots,
                                      const float *wts, const float *x,
                                      uint32_t xs, uint32_t in_dim,
                                      uint32_t out_dim, const int32_t *n_d) {
    const uint32_t lim = n_d ? (uint32_t)*n_d : gridDim.y;
    const uint32_t nb = in_dim >> 6;
    Q4_WGV_ROW();
    for (uint32_t e = blockIdx.y; e < lim; e += gridDim.y) {
        const uint8_t *wr = base + (size_t)slots[e] * stride +
                            (size_t)row * nb * 18ull;
        const float *xt = x + (size_t)e * xs;
        const uint32_t b = lane >> 4, byte = lane & 15u;
        float acc = 0.f;
        for (uint32_t b0 = 0; b0 < nb; b0 += 2u) {
            uint32_t bb = b0 + b;
            if (bb < nb)
                acc += wgd_q20(wr + (size_t)bb * 18ull,
                               xt + bb * 64u + byte * 4u, byte);
        }
        acc = wgv_shfl_red(acc);
        if (lane == 0) atomicAdd(y + row, acc * wts[e]);
    }
}

/* int8-activation ya variants: x slices were quantized into per-32 i8
 * groups by xq8_k (see below).  Same warp-per-row geometry as the float
 * kernels; the per-element fma tail collapses into i32 dots scaled once
 * per weight block. */
__global__ static void wmoe_iq4_ya_q8(float *y, const uint8_t *base,
                                      uint64_t stride, const int32_t *slots,
                                      const float *wts, const int8_t *xq8,
                                      const float *xd8, uint32_t in_dim,
                                      uint32_t out_dim, const int32_t *n_d) {
    const uint32_t lim = n_d ? (uint32_t)*n_d : gridDim.y;
    const uint32_t nb = in_dim >> 5;
    Q4_WGV_ROW();
    for (uint32_t e = blockIdx.y; e < lim; e += gridDim.y) {
        const uint8_t *wr = base + (size_t)slots[e] * stride +
                            (size_t)row * nb * 18ull;
        const int8_t *xq = xq8 + (size_t)e * in_dim;
        const float *xd = xd8 + (size_t)e * (in_dim >> 5);
        const uint32_t b = lane >> 2;
        float acc = 0.f;
        for (uint32_t b0 = 0; b0 < nb; b0 += 8u) {
            uint32_t bb = b0 + b;
            if (bb < nb) {
                float dw;
                int32_t s = wgd_iq4nl_i8(wr + (size_t)bb * 18ull,
                                         xq + bb * 32u, lane, &dw);
                acc += dw * xd[bb] * (float)s;
            }
        }
        acc = wgv_shfl_red(acc);
        if (lane == 0) atomicAdd(y + row, acc * wts[e]);
    }
}

__global__ static void wmoe_q20_ya_q8(float *y, const uint8_t *base,
                                      uint64_t stride, const int32_t *slots,
                                      const float *wts, const int8_t *xq8,
                                      const float *xd8, uint32_t in_dim,
                                      uint32_t out_dim, const int32_t *n_d) {
    const uint32_t lim = n_d ? (uint32_t)*n_d : gridDim.y;
    const uint32_t nb = in_dim >> 6;
    Q4_WGV_ROW();
    for (uint32_t e = blockIdx.y; e < lim; e += gridDim.y) {
        const uint8_t *wr = base + (size_t)slots[e] * stride +
                            (size_t)row * nb * 18ull;
        const int8_t *xq = xq8 + (size_t)e * in_dim;
        const float *xd = xd8 + (size_t)e * (in_dim >> 5);
        const uint32_t b = lane >> 4, byte = lane & 15u;
        float acc = 0.f;
        for (uint32_t b0 = 0; b0 < nb; b0 += 2u) {
            uint32_t bb = b0 + b;
            if (bb < nb) {
                /* block bb = elements bb*64..+63 -> q8 groups 2*bb + hi.
                 * lane's 4 elems sit at group offset (byte&7)*4. */
                const uint32_t grp = bb * 2u + (byte >> 3);
                float dw;
                uint32_t xu;
                memcpy(&xu, xq + grp * 32u + (byte & 7u) * 4u, 4);
                int32_t s = wgd_q20_i8(wr + (size_t)bb * 18ull, xu, byte,
                                       &dw);
                acc += dw * xd[grp] * (float)s;
            }
        }
        acc = wgv_shfl_red(acc);
        if (lane == 0) atomicAdd(y + row, acc * wts[e]);
    }
}

__global__ static void moe_act_k(float *g, const float *u, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = g[i];
        g[i] = v / (1.f + expf(-v)) * u[i];
    }
}

/* Device-count variant: n = *n_d * n_ff — grid is sized for the worst case. */
__global__ static void moe_act_dk(float *g, const float *u, uint32_t n_ff,
                                  const int32_t *n_d) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (uint64_t)*n_d * n_ff) {
        float v = g[i];
        g[i] = v / (1.f + expf(-v)) * u[i];
    }
}

/* y[i] += sum_e w[e] * dd[e][i] — replaces n separate axpy launches. */
__global__ static void moe_wsum_k(float *y, const float *dd, const float *w,
                                  uint32_t ne, uint32_t dim) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    float s = 0.f;
    for (uint32_t e = 0; e < ne; e++) s += w[e] * dd[(size_t)e * dim + i];
    y[i] += s;
}

__global__ static void moe_wsum_dk(float *y, const float *dd, const float *w,
                                   const int32_t *n_d, uint32_t dim) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    uint32_t ne = (uint32_t)*n_d;
    float s = 0.f;
    for (uint32_t e = 0; e < ne; e++) s += w[e] * dd[(size_t)e * dim + i];
    y[i] += s;
}

/* ---- multi-segment GEMV: up to 4 weight tensors sharing one x, one
 * launch (gdn qkv+gate+beta+alpha, shared-expert gate+up, ...). ---- */
struct seg4_p {
    const uint8_t *w[8];
    float *o[8];
    uint32_t rows[8];
    uint32_t ty[8];
    uint32_t off[8];    /* cumulative row starts */
    uint32_t nseg;
};

__global__ static void gemv_m_k(seg4_p S, const float *x, uint32_t in_dim) {
    uint32_t g = blockIdx.x, s = 0;
    while (s + 1 < S.nseg && g >= S.off[s + 1]) s++;
    uint32_t row = g - S.off[s];
    const uint8_t *w = S.w[s];
    __shared__ float sh[64];
    float acc = 0.f;
    if (S.ty[s] == Q4_T_Q8_0) {
        uint32_t nb = in_dim / 32u;
        const uint8_t *wr = w + (size_t)row * nb * 34ull;
        for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) {
            __half dh; memcpy(&dh, wr + b * 34u, 2);
            const int8_t *qs = (const int8_t *)(wr + b * 34u + 2);
            float dot = 0.f;
            for (uint32_t i = 0; i < 32; i++)
                dot += (float)qs[i] * x[b * 32u + i];
            acc += __half2float(dh) * dot;
        }
    } else { /* F32 row */
        const float *wr = (const float *)w + (size_t)row * in_dim;
        for (uint32_t i = threadIdx.x; i < in_dim; i += blockDim.x)
            acc += wr[i] * x[i];
    }
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t t = blockDim.x >> 1; t > 0; t >>= 1) {
        if (threadIdx.x < t) sh[threadIdx.x] += sh[threadIdx.x + t];
        __syncthreads();
    }
    if (threadIdx.x == 0) S.o[s][row] = sh[0];
}

/* Warp-per-row variant of gemv_m_k: warp handles concatenated row g; the
 * segment search is warp-uniform.  Q8_0 rows use the 16-elem/lane oct path,
 * F32 rows the float4x2 oct path. */
__global__ static void gemv_mw_k(seg4_p S, const float *x, uint32_t in_dim) {
    uint32_t g = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    uint32_t lane = threadIdx.x & 31u;
    uint32_t s = 0;
    while (s + 1 < S.nseg && g >= S.off[s + 1]) s++;
    uint32_t row = g - S.off[s];
    if (row >= S.rows[s]) return;
    const uint8_t *w = S.w[s];
    float acc = 0.f;
    if (S.ty[s] == Q4_T_Q8_0) {
        uint32_t nb = in_dim >> 5;
        const uint8_t *wr = w + (size_t)row * nb * 34ull;
        const uint32_t b = lane >> 1, m = (lane & 1u) * 16u;
        for (uint32_t b0 = 0; b0 < nb; b0 += 16u) {
            uint32_t bb = b0 + b;
            if (bb < nb)
                acc += wgd_q8h(wr + (size_t)bb * 34ull, x + bb * 32u, m);
        }
    } else { /* F32 row */
        const float *wr = (const float *)w + (size_t)row * in_dim;
        uint32_t e0 = lane * 8u;
        for (; e0 + 8u <= in_dim; e0 += 256u)
            acc += wgd_f32o(wr + e0, x + e0);
        for (uint32_t e = (in_dim & ~7u) + lane; e < in_dim; e += 32u)
            acc += wr[e] * x[e];
    }
    acc = wgv_shfl_red(acc);
    if (lane == 0) S.o[s][row] = acc;
}

extern "C" bool q4_hip_gemv_m(const q4_seg_t *segs, uint32_t nseg,
                              const float *d_x, uint32_t in_dim) {
    if (!g_ok || !segs || !d_x || !nseg || nseg > 8 || !in_dim) return false;
    seg4_p P;
    memset(&P, 0, sizeof P);
    P.nseg = nseg;
    uint32_t tot = 0;
    for (uint32_t i = 0; i < nseg; i++) {
        if (!segs[i].w || !segs[i].out || !segs[i].rows) return false;
        if (segs[i].ty != Q4_T_Q8_0 && segs[i].ty != Q4_T_F32) return false;
        if (segs[i].ty == Q4_T_Q8_0 && in_dim % 32u) return false;
        P.w[i] = segs[i].w;
        P.o[i] = segs[i].out;
        P.rows[i] = segs[i].rows;
        P.ty[i] = segs[i].ty;
        P.off[i] = tot;
        tot += segs[i].rows;
    }
    /* The warp path needs float4-aligned x reads: every row start must keep
     * in_dim % 4 (q8_0 segments already imply %32). */
    if (wgv_on() && in_dim % 4u == 0) {
        gemv_mw_k<<<dim3((tot + 7u) / 8u), 256, 0, g_str>>>(P, d_x, in_dim);
    } else {
        gemv_m_k<<<dim3(tot), 64, 0, g_str>>>(P, d_x, in_dim);
    }
    return hipGetLastError() == hipSuccess;
}

/* Small elementwise fusions (each replaces 2 launches). */
__global__ static void mul_sig_k(float *y, const float *a, const float *g,
                                 uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] / (1.f + expf(-g[i]));
}
extern "C" bool q4_hip_mul_sig(float *y, const float *a, const float *g,
                               uint64_t n) {
    if (!g_ok || !y || !a || !g || !n) return false;
    mul_sig_k<<<dim3((uint32_t)((n + 255) / 256)), 256, 0, g_str>>>(y, a, g, n);
    return hipGetLastError() == hipSuccess;
}
/* Fused mul_sig + mean_hc for the decode hc_mix tail: one node instead of
 * two, and skips materialising the gated intermediate.  Same expression
 * and same accumulation order as the two kernels it replaces. */
__global__ static void mulsig_mean_k(const float *a, const float *g,
                                     float *mixed, uint32_t n_embd,
                                     uint32_t hc) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    float s = 0.f;
    for (uint32_t c = 0; c < hc; c++) {
        size_t o = (size_t)c * n_embd + i;
        s += a[o] / (1.f + expf(-g[o]));
    }
    mixed[i] = s / (float)hc;
}
extern "C" bool q4_hip_mulsig_mean(const float *a, const float *g, float *mixed,
                                   uint32_t n_embd, uint32_t hc) {
    if (!g_ok || !a || !g || !mixed || !n_embd || !hc) return false;
    mulsig_mean_k<<<dim3((n_embd + 255) / 256), 256, 0, g_str>>>(a, g, mixed,
                                                               n_embd, hc);
    return hipGetLastError() == hipSuccess;
}

__global__ static void scale_silu_k(float *x, float s, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i] * s;
        x[i] = v / (1.f + expf(-v));
    }
}
extern "C" bool q4_hip_scale_silu(float *x, float s, uint64_t n) {
    if (!g_ok || !x || !n) return false;
    scale_silu_k<<<dim3((uint32_t)((n + 255) / 256)), 256, 0, g_str>>>(x, s, n);
    return hipGetLastError() == hipSuccess;
}
__global__ static void silu_mul_k(float *g, const float *u, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = g[i];
        g[i] = v / (1.f + expf(-v)) * u[i];
    }
}
extern "C" bool q4_hip_silu_mul(float *g, const float *u, uint64_t n) {
    if (!g_ok || !g || !u || !n) return false;
    silu_mul_k<<<dim3((uint32_t)((n + 255) / 256)), 256, 0, g_str>>>(g, u, n);
    return hipGetLastError() == hipSuccess;
}
__global__ static void dot_sig_k(const float *a, const float *b, float *out,
                                 uint32_t n) {
    __shared__ float sh[256];
    float acc = 0.f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) acc += a[i] * b[i];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[0] = 1.f / (1.f + expf(-sh[0]));
}
extern "C" bool q4_hip_dot_sigmoid(const float *a, const float *b, float *out,
                                   uint32_t n) {
    if (!g_ok || !a || !b || !out) return false;
    dot_sig_k<<<1, 256, 0, g_str>>>(a, b, out, n);
    return hipGetLastError() == hipSuccess;
}

static int32_t *g_es_d, *g_es_h; /* expert slot + weight staging (≤64 each) */
static float *g_ew_d, *g_ew_h;
static int32_t *g_en_d, *g_en_h; /* device-visible hit count for graphable exec */
static bool moe_stage_bufs(void) {
    if (!g_es_d) {
        /* Mapped host memory doubles as the device buffers: the exec kernels
         * read slot ids + weights over PCIe (≤512 B, L2-cached), which drops
         * two H2D copy nodes per MoE layer out of every decode graph. */
        if (hipHostMalloc((void **)&g_es_h, 64 * sizeof(int32_t),
                          hipHostMallocMapped) != hipSuccess ||
            hipHostMalloc((void **)&g_ew_h, 64 * sizeof(float),
                          hipHostMallocMapped) != hipSuccess ||
            hipHostMalloc((void **)&g_en_h, 64 * sizeof(int32_t),
                          hipHostMallocMapped) != hipSuccess ||
            hipHostGetDevicePointer((void **)&g_es_d, g_es_h, 0) !=
                hipSuccess ||
            hipHostGetDevicePointer((void **)&g_ew_d, g_ew_h, 0) !=
                hipSuccess ||
            hipHostGetDevicePointer((void **)&g_en_d, g_en_h, 0) !=
                hipSuccess) {
            g_es_d = nullptr;
            return false;
        }
        *g_en_h = 0;
    }
    (void)xq8_bufs(); /* int8-activation staging — outside any capture */
    return true;
}

static bool moe_ty_ok(uint32_t ty, uint32_t in_dim) {
    switch (ty) {
    case Q4_T_IQ2_S:
    case Q4_T_IQ3_S:
    case Q4_T_IQ3_XXS:
    case Q4_T_IQ4_XS:
        return in_dim % 256u == 0;
    case Q4_T_Q2_0:
        return in_dim % 64u == 0;
    case Q4_T_IQ4_NL:
    case Q4_T_Q4_0:
    case Q4_T_Q5_0:
        return in_dim % 32u == 0;
    default:
        return false;
    }
}

static bool moe_rows(uint32_t ty, const uint8_t *base, uint64_t stride,
                     const int32_t *slots, const float *x, uint32_t xs,
                     uint32_t in_dim, uint32_t out_dim, float *out, uint32_t ne,
                     const int32_t *n_d) {
    if (!in_dim || !out_dim || (!ne && !n_d)) return false;
    if (wgv_on()) {
        dim3 wgrd((out_dim + 7u) / 8u, n_d ? WMOE_GY : ne), wblk(256);
        switch (ty) {
        case Q4_T_IQ2_S:
            if (in_dim % 256u) return false;
            wmoe_iq2_bk<<<wgrd, wblk, 0, g_str>>>(out, base, stride, slots, x,
                                                  xs, in_dim, out_dim, n_d);
            break;
        case Q4_T_IQ3_S:
            if (in_dim % 256u) return false;
            wmoe_iq3_bk<<<wgrd, wblk, 0, g_str>>>(out, base, stride, slots, x,
                                                  xs, in_dim, out_dim, n_d);
            break;
        case Q4_T_IQ3_XXS:
            if (in_dim % 256u) return false;
            wmoe_iq3xxs_bk<<<wgrd, wblk, 0, g_str>>>(out, base, stride, slots,
                                                     x, xs, in_dim, out_dim,
                                                     n_d);
            break;
        case Q4_T_IQ4_XS:
            if (in_dim % 256u) return false;
            wmoe_iq4xs_bk<<<wgrd, wblk, 0, g_str>>>(out, base, stride, slots,
                                                    x, xs, in_dim, out_dim,
                                                    n_d);
            break;
        case Q4_T_IQ4_NL:
            if (in_dim % 32u) return false;
            wmoe_iq4_bk<<<wgrd, wblk, 0, g_str>>>(out, base, stride, slots, x,
                                                  xs, in_dim, out_dim, n_d);
            break;
        case Q4_T_Q4_0:
            if (in_dim % 32u) return false;
            wmoe_q40_bk<<<wgrd, wblk, 0, g_str>>>(out, base, stride, slots, x,
                                                 xs, in_dim, out_dim, n_d);
            break;
        case Q4_T_Q5_0:
            if (in_dim % 32u) return false;
            wmoe_q50_bk<<<wgrd, wblk, 0, g_str>>>(out, base, stride, slots, x,
                                                 xs, in_dim, out_dim, n_d);
            break;
        case Q4_T_Q2_0:
            if (in_dim % 64u) return false;
            wmoe_q20_bk<<<wgrd, wblk, 0, g_str>>>(out, base, stride, slots, x,
                                                 xs, in_dim, out_dim, n_d);
            break;
        default:
            return false;
        }
        return hipGetLastError() == hipSuccess;
    }
    if (n_d) return false;   /* legacy row-per-block kernels take host counts */
    dim3 grd(out_dim, ne), blk(64);
    switch (ty) {
    case Q4_T_IQ2_S:
        if (in_dim % 256u) return false;
        moe_iq2_bk<<<grd, blk, 0, g_str>>>(out, base, stride, slots, x, xs,
                                           in_dim, out_dim);
        break;
    case Q4_T_IQ3_S:
        if (in_dim % 256u) return false;
        moe_iq3_bk<<<grd, blk, 0, g_str>>>(out, base, stride, slots, x, xs,
                                           in_dim, out_dim);
        break;
    case Q4_T_IQ3_XXS:
        if (in_dim % 256u) return false;
        moe_iq3xxs_bk<<<grd, blk, 0, g_str>>>(out, base, stride, slots, x, xs,
                                              in_dim, out_dim);
        break;
    case Q4_T_IQ4_XS:
        if (in_dim % 256u) return false;
        moe_iq4xs_bk<<<grd, blk, 0, g_str>>>(out, base, stride, slots, x, xs,
                                             in_dim, out_dim);
        break;
    case Q4_T_IQ4_NL:
        if (in_dim % 32u) return false;
        moe_iq4_bk<<<grd, blk, 0, g_str>>>(out, base, stride, slots, x, xs,
                                           in_dim, out_dim);
        break;
    case Q4_T_Q4_0:
        if (in_dim % 32u) return false;
        moe_q40_bk<<<grd, blk, 0, g_str>>>(out, base, stride, slots, x, xs,
                                           in_dim, out_dim);
        break;
    case Q4_T_Q5_0:
        if (in_dim % 32u) return false;
        moe_q50_bk<<<grd, blk, 0, g_str>>>(out, base, stride, slots, x, xs,
                                           in_dim, out_dim);
        break;
    case Q4_T_Q2_0:
        if (in_dim % 64u) return false;
        moe_q20_bk<<<grd, blk, 0, g_str>>>(out, base, stride, slots, x, xs,
                                           in_dim, out_dim);
        break;
    default:
        return false;
    }
    return hipGetLastError() == hipSuccess;
}

/* ---- int8 activation staging for the *_q8 moe kernels ------------------
 * xq8_k quantizes each 32-float group to i8x32 + one fp32 scale, one warp
 * per group.  Buffers cover 64 expert slices of up to 16384 floats. */
#define XQ8_GPS_MAX 512u /* groups per slice: slen/32, slen <= 16384 */
#define XQ8_NS_MAX 64u
static int8_t *g_xq8;
static float *g_xq8d;
static bool xq8_bufs(void) {
    if (!g_xq8) {
        if (hipMalloc((void **)&g_xq8, XQ8_NS_MAX * XQ8_GPS_MAX * 32u) !=
                hipSuccess ||
            hipMalloc((void **)&g_xq8d, XQ8_NS_MAX * XQ8_GPS_MAX * 4u) !=
                hipSuccess) {
            g_xq8 = nullptr;
            return false;
        }
    }
    return true;
}

static int wq8_on(void) {
    static int v = -1;
    if (v < 0) v = getenv("Q4_WGQ8") ? atoi(getenv("Q4_WGQ8")) : 1;
    return v;
}

static int wq8_iq4ya_on(void) {
    static int v = -1;
    if (v < 0)
        v = wq8_on() && getenv("Q4_WGQ8_IQ4YA") &&
            atoi(getenv("Q4_WGQ8_IQ4YA"));
    return v;
}

__global__ static void xq8_k(const float *x, uint32_t stride, int8_t *q8,
                             float *d8, uint32_t gps, const int32_t *n_d) {
    const uint32_t g = blockIdx.x, lane = threadIdx.x & 31u;
    const uint32_t e = g / gps, gl = g - e * gps;
    if (n_d && e >= (uint32_t)*n_d) return;
    const float *xs = x + (size_t)e * stride + gl * 32u;
    const float v = xs[lane];
    float amax = fabsf(v);
    for (int o = 16; o; o >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffull, amax, o));
    q8[(size_t)g * 32 + lane] = (int8_t)__float2int_rn(v * (amax ? 127.f / amax : 0.f));
    if (!lane) d8[g] = amax * (1.f / 127.f);
}

/* Quantize `ns` x-slices of `slen` floats (slice e at x + e*xs) on g_str.
 * n_d != NULL: device count bounds the slices actually written. */
static bool xq8_launch(const float *x, uint32_t xs, uint32_t slen,
                       uint32_t ns, const int32_t *n_d) {
    if (!slen || (slen & 31u) || (slen >> 5) > XQ8_GPS_MAX) return false;
    if (!xq8_bufs()) return false;
    const uint32_t gps = slen >> 5;
    const uint32_t ng = (n_d ? XQ8_NS_MAX : (xs ? ns : 1u)) * gps;
    xq8_k<<<ng, 32, 0, g_str>>>(x, xs, g_xq8, g_xq8d, gps, n_d);
    return hipGetLastError() == hipSuccess;
}

/* Fused gate+up: one launch when gt == ut (the common case). */
static bool moe_rows_gu(uint32_t ty, const uint8_t *bg, const uint8_t *bu,
                        uint64_t stride, const int32_t *slots, const float *x,
                        uint32_t xs, uint32_t in_dim, uint32_t out_dim,
                        float *og, float *ou, uint32_t ne,
                        const int32_t *n_d) {
    if (!wgv_on() || !in_dim || !out_dim || (!ne && !n_d)) return false;
    dim3 wgrd((out_dim + 7u) / 8u, n_d ? WMOE_GY : ne), wblk(256);
    /* xs==0 means all experts share one activation — quantize it once into
     * g_xq8 and take the i8-dot kernels. */
    const bool q8ok = wq8_on() && xs == 0 && (in_dim % 256u) == 0 &&
                      (ty == Q4_T_IQ2_S || ty == Q4_T_IQ3_S ||
                       ty == Q4_T_IQ3_XXS || ty == Q4_T_IQ4_XS) &&
                      xq8_launch(x, 0, in_dim, 1, NULL);
    switch (ty) {
    case Q4_T_IQ2_S:
        if (in_dim % 256u) return false;
        if (q8ok) {
            wmoe_iq2_gu_q8<<<wgrd, wblk, 0, g_str>>>(og, ou, bg, bu, stride,
                slots, g_xq8, g_xq8d, in_dim, out_dim, n_d);
            break;
        }
        wmoe_iq2_gu_bk<<<wgrd, wblk, 0, g_str>>>(og, ou, bg, bu, stride, slots,
                                                 x, xs, in_dim, out_dim, n_d);
        break;
    case Q4_T_IQ3_S:
        if (in_dim % 256u) return false;
        if (q8ok) {
            wmoe_iq3_gu_q8<<<wgrd, wblk, 0, g_str>>>(og, ou, bg, bu, stride,
                slots, g_xq8, g_xq8d, in_dim, out_dim, n_d);
            break;
        }
        wmoe_iq3_gu_bk<<<wgrd, wblk, 0, g_str>>>(og, ou, bg, bu, stride, slots,
                                                 x, xs, in_dim, out_dim, n_d);
        break;
    case Q4_T_IQ3_XXS:
        if (in_dim % 256u) return false;
        if (q8ok) {
            wmoe_iq3xxs_gu_q8<<<wgrd, wblk, 0, g_str>>>(og, ou, bg, bu, stride,
                slots, g_xq8, g_xq8d, in_dim, out_dim, n_d);
            break;
        }
        wmoe_iq3xxs_gu_bk<<<wgrd, wblk, 0, g_str>>>(og, ou, bg, bu, stride,
                                                    slots, x, xs, in_dim,
                                                    out_dim, n_d);
        break;
    case Q4_T_IQ4_XS:
        if (in_dim % 256u) return false;
        if (q8ok) {
            wmoe_iq4xs_gu_q8<<<wgrd, wblk, 0, g_str>>>(og, ou, bg, bu, stride,
                slots, g_xq8, g_xq8d, in_dim, out_dim, n_d);
            break;
        }
        wmoe_iq4xs_gu_bk<<<wgrd, wblk, 0, g_str>>>(og, ou, bg, bu, stride,
                                                   slots, x, xs, in_dim,
                                                   out_dim, n_d);
        break;
    case Q4_T_IQ4_NL:
        if (in_dim % 32u) return false;
        wmoe_iq4_gu_bk<<<wgrd, wblk, 0, g_str>>>(og, ou, bg, bu, stride, slots,
                                                 x, xs, in_dim, out_dim, n_d);
        break;
    case Q4_T_Q4_0:
        if (in_dim % 32u) return false;
        wmoe_q40_gu_bk<<<wgrd, wblk, 0, g_str>>>(og, ou, bg, bu, stride, slots,
                                                x, xs, in_dim, out_dim, n_d);
        break;
    case Q4_T_Q5_0:
        if (in_dim % 32u) return false;
        wmoe_q50_gu_bk<<<wgrd, wblk, 0, g_str>>>(og, ou, bg, bu, stride, slots,
                                                x, xs, in_dim, out_dim, n_d);
        break;
    case Q4_T_Q2_0:
        if (in_dim % 64u) return false;
        wmoe_q20_gu_bk<<<wgrd, wblk, 0, g_str>>>(og, ou, bg, bu, stride, slots,
                                                x, xs, in_dim, out_dim, n_d);
        break;
    default:
        return false;
    }
    return hipGetLastError() == hipSuccess;
}

/* Down proj + fused weighted accumulate into y (atomic epilogue). */
static bool moe_rows_ya(uint32_t ty, const uint8_t *base, uint64_t stride,
                        const int32_t *slots, const float *wts,
                        const float *x, uint32_t xs, uint32_t in_dim,
                        uint32_t out_dim, float *y, uint32_t ne,
                        const int32_t *n_d) {
    if (!wgv_on() || !in_dim || !out_dim || (!ne && !n_d) || !wts)
        return false;
    dim3 wgrd((out_dim + 7u) / 8u, n_d ? WMOE_GY : ne), wblk(256);
    switch (ty) {
    case Q4_T_IQ2_S:
        if (in_dim % 256u) return false;
        if (wq8_on() && xs == in_dim &&
            xq8_launch(x, xs, in_dim, ne, n_d)) {
            wmoe_iq2_ya_q8<<<wgrd, wblk, 0, g_str>>>(y, base, stride, slots,
                wts, g_xq8, g_xq8d, in_dim, out_dim, n_d);
            break;
        }
        wmoe_iq2_ya_bk<<<wgrd, wblk, 0, g_str>>>(y, base, stride, slots, wts,
                                                 x, xs, in_dim, out_dim, n_d);
        break;
    case Q4_T_IQ3_S:
        if (in_dim % 256u) return false;
        if (wq8_on() && xs == in_dim &&
            xq8_launch(x, xs, in_dim, ne, n_d)) {
            wmoe_iq3_ya_q8<<<wgrd, wblk, 0, g_str>>>(y, base, stride, slots,
                wts, g_xq8, g_xq8d, in_dim, out_dim, n_d);
            break;
        }
        wmoe_iq3_ya_bk<<<wgrd, wblk, 0, g_str>>>(y, base, stride, slots, wts,
                                                 x, xs, in_dim, out_dim, n_d);
        break;
    case Q4_T_IQ3_XXS:
        if (in_dim % 256u) return false;
        if (wq8_on() && xs == in_dim &&
            xq8_launch(x, xs, in_dim, ne, n_d)) {
            wmoe_iq3xxs_ya_q8<<<wgrd, wblk, 0, g_str>>>(y, base, stride, slots,
                wts, g_xq8, g_xq8d, in_dim, out_dim, n_d);
            break;
        }
        wmoe_iq3xxs_ya_bk<<<wgrd, wblk, 0, g_str>>>(y, base, stride, slots,
                                                    wts, x, xs, in_dim,
                                                    out_dim, n_d);
        break;
    case Q4_T_IQ4_XS:
        if (in_dim % 256u) return false;
        /* Same lightweight-LUT class as iq4_nl: bandwidth-bound down rows,
         * marginal isolated win — keep float unless opted in. */
        if (wq8_iq4ya_on() && xs == in_dim &&
            xq8_launch(x, xs, in_dim, ne, n_d)) {
            wmoe_iq4xs_ya_q8<<<wgrd, wblk, 0, g_str>>>(y, base, stride, slots,
                wts, g_xq8, g_xq8d, in_dim, out_dim, n_d);
            break;
        }
        wmoe_iq4xs_ya_bk<<<wgrd, wblk, 0, g_str>>>(y, base, stride, slots, wts,
                                                   x, xs, in_dim, out_dim,
                                                   n_d);
        break;
    case Q4_T_IQ4_NL:
        if (in_dim % 32u) return false;
        /* Measured slower in-model (+10% kernel time) despite winning in the
         * isolated bench: at n_ff=640 the row is only 20 blocks, so the
         * extra xd8/xq8 dependent loads sit on the acc chain while weight
         * loads from L1 slots already bound the loop.  Keep float. */
        if (wq8_iq4ya_on() && xs == in_dim &&
            xq8_launch(x, xs, in_dim, ne, n_d)) {
            wmoe_iq4_ya_q8<<<wgrd, wblk, 0, g_str>>>(y, base, stride, slots,
                wts, g_xq8, g_xq8d, in_dim, out_dim, n_d);
            break;
        }
        wmoe_iq4_ya_bk<<<wgrd, wblk, 0, g_str>>>(y, base, stride, slots, wts,
                                                 x, xs, in_dim, out_dim, n_d);
        break;
    case Q4_T_Q4_0:
        if (in_dim % 32u) return false;
        wmoe_q40_ya_bk<<<wgrd, wblk, 0, g_str>>>(y, base, stride, slots, wts,
                                                x, xs, in_dim, out_dim, n_d);
        break;
    case Q4_T_Q5_0:
        if (in_dim % 32u) return false;
        wmoe_q50_ya_bk<<<wgrd, wblk, 0, g_str>>>(y, base, stride, slots, wts,
                                                x, xs, in_dim, out_dim, n_d);
        break;
    case Q4_T_Q2_0:
        if (in_dim % 64u) return false;
        if (wq8_on() && xs == in_dim &&
            xq8_launch(x, xs, in_dim, ne, n_d)) {
            wmoe_q20_ya_q8<<<wgrd, wblk, 0, g_str>>>(y, base, stride, slots,
                wts, g_xq8, g_xq8d, in_dim, out_dim, n_d);
            break;
        }
        wmoe_q20_ya_bk<<<wgrd, wblk, 0, g_str>>>(y, base, stride, slots, wts,
                                                x, xs, in_dim, out_dim, n_d);
        break;
    default:
        return false;
    }
    return hipGetLastError() == hipSuccess;
}

/* All n hit-experts in three launches when gt == ut (fused gate+up,
 * silu*mul, down with atomic weighted accumulate into d_y); falls back to
 * the five-launch chain for mixed types. Caller guarantees slots' data
 * already landed (upload watermark or copy-stream drain). */
extern "C" bool q4_hip_moe_exec(const uint8_t *base, uint64_t stride,
                                const int32_t *slots, const float *wts,
                                uint32_t n, uint32_t gt, uint64_t og,
                                uint32_t ut, uint64_t ou, uint32_t dt,
                                uint64_t od, const float *d_x, float *d_h,
                                float *d_h2, float *d_down, float *d_y,
                                uint32_t n_ff, uint32_t in_dim,
                                uint32_t n_embd) {
    if (!g_ok || !base || !slots || !wts || !n || n > 64) return false;
    /* Reject unsupported types BEFORE any launch so the caller's fallback
     * never double-accumulates work we already queued. */
    if (!moe_ty_ok(gt, in_dim) || !moe_ty_ok(ut, in_dim) ||
        !moe_ty_ok(dt, n_ff))
        return false;
    if (!moe_stage_bufs()) return false;
    memcpy(g_es_h, slots, n * sizeof(int32_t));
    memcpy(g_ew_h, wts, n * sizeof(float));
    /* slots/wts live in mapped host memory; the kernels below fetch them
     * over PCIe themselves — no copy nodes on the stream. */
    if (gt == ut &&
        moe_rows_gu(gt, base + og, base + ou, stride, g_es_d, d_x, 0, in_dim,
                    n_ff, d_h, d_h2, n, NULL)) {
        /* fused path committed */
    } else if (!moe_rows(gt, base + og, stride, g_es_d, d_x, 0, in_dim, n_ff,
                         d_h, n, NULL) ||
               !moe_rows(ut, base + ou, stride, g_es_d, d_x, 0, in_dim, n_ff,
                         d_h2, n, NULL)) {
        return false;
    }
    uint64_t na = (uint64_t)n * n_ff;
    moe_act_k<<<dim3((uint32_t)((na + 255) / 256)), 256, 0, g_str>>>(d_h, d_h2,
                                                                    na);
    /* All types were vetted above, so a ya failure here is a launch error —
     * do NOT fall back or the down contribution would be applied twice. */
    if (wgv_on())
        return moe_rows_ya(dt, base + od, stride, g_es_d, g_ew_d, d_h, n_ff,
                           n_ff, n_embd, d_y, n, NULL);
    if (!moe_rows(dt, base + od, stride, g_es_d, d_h, n_ff, n_ff, n_embd,
                  d_down, n, NULL))
        return false;
    moe_wsum_k<<<dim3((n_embd + 255) / 256), 256, 0, g_str>>>(d_y, d_down,
                                                            g_ew_d, n, n_embd);
    return hipGetLastError() == hipSuccess;
}

/* Preflight for the per-token small-batch exec: same type/shape rule as
 * q4_hip_moe_exec so callers can decide the fast path before any launch. */
extern "C" bool q4_hip_moe_tok_ok(uint32_t gt, uint32_t ut, uint32_t dt,
                                  uint32_t in_dim, uint32_t n_ff) {
    return g_ok && wgv_on() && moe_ty_ok(gt, in_dim) &&
           moe_ty_ok(ut, in_dim) && moe_ty_ok(dt, n_ff);
}

/* Small-batch MoE hits (2..8 tokens, MTP verify / short prefill): slots
 * and weights live in DEVICE arrays laid out token-major with h_off
 * prefix offsets, so back-to-back calls never touch the mapped staging
 * arrays q4_hip_moe_exec uses (those would race with queued kernels).
 * Per token: fused gate+up rows into d_h/d_h2 at the token's hit offset,
 * one fused silu*mul over the whole hit count, then down rows with the
 * fused weighted atomicAdd into that token's d_y. */
extern "C" bool q4_hip_moe_exec_tok(const uint8_t *base, uint64_t stride,
                                      const int32_t *d_slots,
                                      const float *d_wts,
                                      const uint32_t *h_off, uint32_t n_tok,
                                      uint32_t gt, uint64_t og, uint32_t ut,
                                      uint64_t ou, uint32_t dt, uint64_t od,
                                      const float *d_x, float *d_h,
                                      float *d_h2, float *d_y, uint32_t n_ff,
                                      uint32_t in_dim, uint32_t n_embd) {
    if (!g_ok || !base || !d_slots || !d_wts || !h_off || !n_tok)
        return false;
    /* Preflight everything before the first launch: a false return means
     * nothing was queued and the caller may take its old path. */
    if (!wgv_on() || !moe_ty_ok(gt, in_dim) || !moe_ty_ok(ut, in_dim) ||
        !moe_ty_ok(dt, n_ff))
        return false;
    uint32_t tot = h_off[n_tok];
    if (!tot) return true;
    for (uint32_t t = 0; t < n_tok; t++) {
        uint32_t cnt = h_off[t + 1] - h_off[t];
        if (!cnt) continue;
        if (gt != ut ||
            !moe_rows_gu(gt, base + og, base + ou, stride,
                         d_slots + h_off[t], d_x + (size_t)t * in_dim, 0,
                         in_dim, n_ff, d_h + (size_t)h_off[t] * n_ff,
                         d_h2 + (size_t)h_off[t] * n_ff, cnt, NULL))
            return false;
    }
    uint64_t na = (uint64_t)tot * n_ff;
    moe_act_k<<<dim3((uint32_t)((na + 255) / 256)), 256, 0, g_str>>>(d_h, d_h2,
                                                                     na);
    for (uint32_t t = 0; t < n_tok; t++) {
        uint32_t cnt = h_off[t + 1] - h_off[t];
        if (!cnt) continue;
        if (!moe_rows_ya(dt, base + od, stride, d_slots + h_off[t],
                         d_wts + h_off[t], d_h + (size_t)h_off[t] * n_ff,
                         n_ff, n_ff, n_embd, d_y + (size_t)t * n_embd, cnt,
                         NULL))
            return false;
    }
    return hipGetLastError() == hipSuccess;
}

/* ---- device-side routing (graph-resident MoE layer) --------------------
 * route_dev_k runs the router softmax+top-k entirely on the GPU, resolves
 * each pick to an L1 slot through the device-visible residency table
 * (published by the copy stream), splits into GPU-hit / CPU-miss lists and
 * publishes misses to a pinned mailbox consumed by the host dispatcher.
 * One block; thread 0 does the scan — 512 experts × insertion-sort top-k is
 * only a few µs and stays bit-identical to the host q4_topk order. */
typedef struct {
    volatile unsigned long long seq;   /* publish stamp (++ per job) */
    volatile unsigned long long done;  /* host dispatcher's completion stamp */
    int32_t ids[64];                   /* ids+wts contiguous: warp-burst copy */
    float wts[64];
    uint32_t n;                        /* miss count */
    volatile uint32_t err;             /* set by host if slot was overrun */
    /* Fire-and-forget routing profile channel: every call publishes the
     * full top-k list + an rseq stamp.  The host may drop entries (it's a
     * histogram, not a protocol) and never stamps anything back, so the
     * GPU never waits on it.  Routed reads are torn-tolerant by design. */
    volatile unsigned long long rseq;
    volatile int32_t routed[16];
} q4_mail_t;

/* Warp-cooperative router: lanes split the 512-wide max/exp scans (the
 * serial passes before were ~190µs of dependent global-load latency), the
 * probs land in shared memory, then lane 0 runs the host-identical serial
 * sum/insertion/renormalize — bit-identical ordering vs q4_topk. */
__global__ static void route_dev_k(
        const float *d_rlog, const int32_t *devtab,
        int32_t *d_es, float *d_ew, int32_t *d_nhit,
        q4_mail_t *mail, unsigned long long *d_seq,
        unsigned long long *d_rseq,
        unsigned long long *d_prof,
        uint32_t n_exp, uint32_t topk) {
    __shared__ float pr[1024];
    __shared__ unsigned long long s_pay[64]; /* ids[64] then wts[64] */
    const uint32_t lane = threadIdx.x;
    const unsigned long long t0 = clock64();
    float mx = -INFINITY;
    for (uint32_t i = lane; i < n_exp; i += 32u) mx = fmaxf(mx, d_rlog[i]);
    for (int o = 16; o; o >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffull, mx, o));
    float ps = 0.f;
    for (uint32_t i = lane; i < n_exp; i += 32u) {
        float e = expf(d_rlog[i] - mx);
        pr[i] = e;
        ps += e;
    }
    for (int o = 16; o; o >>= 1)
        ps += __shfl_xor_sync(0xffffffffull, ps, o);
    const float inv = ps > 0.f ? 1.f / ps : 0.f;
    for (uint32_t i = lane; i < n_exp; i += 32u) pr[i] *= inv;
    __shared__ uint32_t s_nm;
    if (lane == 0) {
        /* ids/wts live in shared: locals with dynamic indexing would spill
         * to DRAM-backed scratch and make every insertion compare ~400ns. */
        int32_t *ids = (int32_t *)s_pay;
        float *wts = (float *)(s_pay + 32);
        for (uint32_t i = 0; i < topk; i++) {
            ids[i] = -1;
            wts[i] = -INFINITY;
        }
        for (uint32_t i = 0; i < n_exp; i++) {
            float v = pr[i];
            uint32_t p = topk;
            while (p > 0 && (ids[p - 1] < 0 || v > wts[p - 1])) p--;
            if (p == topk) continue;
            for (uint32_t j = topk - 1; j > p; j--) {
                ids[j] = ids[j - 1];
                wts[j] = wts[j - 1];
            }
            ids[p] = (int32_t)i;
            wts[p] = v;
        }
        float s = 0.f;
        for (uint32_t i = 0; i < topk; i++) s += wts[i];
        float rinv = s > 0.f ? 1.f / s : 0.f;
        uint32_t nh = 0, nm = 0;
        /* Profile channel: publish the full top-k to the mailbox BEFORE
         * the miss compaction below rewrites ids[] in place.  d_rseq ==
         * NULL keeps this at zero cost when profiling is off. */
        if (d_rseq) {
            for (uint32_t k = 0; k < topk && k < 16; k++)
                mail->routed[k] = ids[k];
        }
        /* Compact misses into the low end of the same shared arrays —
         * nm <= k, so writes only touch entries already consumed. */
        for (uint32_t k = 0; k < topk; k++) {
            float w = wts[k] * rinv;
            int32_t slot = devtab[ids[k]];
            if (slot >= 0) {
                d_es[nh] = slot;
                d_ew[nh] = w;
                nh++;
            } else {
                ids[nm] = ids[k];
                wts[nm] = w;
                nm++;
            }
        }
        *d_nhit = (int32_t)nh;
        s_nm = nm;
    }
    __syncwarp();
    /* Publish only real jobs: the in-graph wait on mail->done back-
     * pressures the pipeline, so a mailbox slot is always consumed before
     * the same layer writes it again next token.  The stamp comes from a
     * device-side per-layer counter — a PCIe read-modify-write of the
     * mailbox itself costs ~3µs.  ids+wts go as a 512B warp burst (a few
     * TLPs) instead of nm scattered 4B sysmem stores; every lane fences so
     * the burst is globally visible before lane 0's seq store is posted. */
    if (s_nm) {
        unsigned long long *dst = (unsigned long long *)mail->ids;
        for (uint32_t i = lane; i < 64; i += 32u) dst[i] = s_pay[i];
        __threadfence_system();
        __syncwarp();
        if (lane == 0) {
            mail->n = s_nm;
            __threadfence_system();
            mail->seq = ++(*d_seq);
        }
    }
    /* Profile stamp: published even when no CPU misses so the host
     * histogram sees hit-only tokens too.  No wait protocol — the
     * dispatcher just counts what it sees before the next bump.  The
     * fence only runs when the channel is enabled. */
    if (d_rseq) {
        __threadfence_system();
        __syncwarp();
        if (lane == 0) mail->rseq = ++(*d_rseq);
    }
    if (lane == 0 && d_prof) {
        d_prof[0] += clock64() - t0;
        d_prof[1] += 1;
    }
}

/* Device-array MoE exec: slots/wts/count all live in device memory written
 * by route_dev_k earlier in the same stream — graphable, zero host staging. */
extern "C" bool q4_hip_moe_exec_dev(const uint8_t *base, uint64_t stride,
                                    const int32_t *d_es, const float *d_ew,
                                    const int32_t *d_nhit, uint32_t gt,
                                    uint64_t og, uint32_t ut, uint64_t ou,
                                    uint32_t dt, uint64_t od,
                                    const float *d_x, float *d_h, float *d_h2,
                                    float *d_down, float *d_y, uint32_t n_ff,
                                    uint32_t in_dim, uint32_t n_embd) {
    if (!g_ok || !base || !d_es || !d_ew || !d_nhit) return false;
    if (!moe_ty_ok(gt, in_dim) || !moe_ty_ok(ut, in_dim) ||
        !moe_ty_ok(dt, n_ff))
        return false;
    if (gt == ut &&
        moe_rows_gu(gt, base + og, base + ou, stride, d_es, d_x, 0, in_dim,
                    n_ff, d_h, d_h2, 0, d_nhit)) {
        /* fused path committed */
    } else if (!moe_rows(gt, base + og, stride, d_es, d_x, 0, in_dim, n_ff,
                         d_h, 0, d_nhit) ||
               !moe_rows(ut, base + ou, stride, d_es, d_x, 0, in_dim, n_ff,
                         d_h2, 0, d_nhit)) {
        return false;
    }
    moe_act_dk<<<dim3((uint32_t)((64ull * n_ff + 255) / 256)), 256, 0,
               g_str>>>(d_h, d_h2, n_ff, d_nhit);
    /* Types vetted above — a ya failure is a launch error; never fall back
     * (the down contribution would land twice). */
    if (wgv_on())
        return moe_rows_ya(dt, base + od, stride, d_es, d_ew, d_h, n_ff, n_ff,
                           n_embd, d_y, 0, d_nhit);
    if (!moe_rows(dt, base + od, stride, d_es, d_h, n_ff, n_ff, n_embd,
                  d_down, 0, d_nhit))
        return false;
    moe_wsum_dk<<<dim3((n_embd + 255) / 256), 256, 0, g_str>>>(d_y, d_down,
                                                             d_ew, d_nhit,
                                                             n_embd);
    return hipGetLastError() == hipSuccess;
}

/* Host-staged hit-exec for the hybrid path.  q4_hip_moe_stage writes the
 * slot/weight/count words into mapped host memory (safe to call between gA
 * and gB every token); q4_hip_moe_exec_staged launches the device-count
 * kernels reading those words — identical inside a graph capture, which is
 * what lets the hit-exec join gB instead of being a host-launch seam. */
extern "C" bool q4_hip_moe_stage(const int32_t *slots, const float *wts,
                                 uint32_t n) {
    if (!moe_stage_bufs() || n > 64) return false;
    memcpy(g_es_h, slots, n * sizeof(int32_t));
    memcpy(g_ew_h, wts, n * sizeof(float));
    *g_en_h = (int32_t)n;
    return true;
}

extern "C" bool q4_hip_moe_exec_staged(
        const uint8_t *base, uint64_t stride, uint32_t gt, uint64_t og,
        uint32_t ut, uint64_t ou, uint32_t dt, uint64_t od, const float *d_x,
        float *d_h, float *d_h2, float *d_down, float *d_y, uint32_t n_ff,
        uint32_t in_dim, uint32_t n_embd) {
    if (!g_es_d) return false;
    return q4_hip_moe_exec_dev(base, stride, g_es_d, g_ew_d, g_en_d, gt, og,
                               ut, ou, dt, od, d_x, d_h, d_h2, d_down, d_y,
                               n_ff, in_dim, n_embd);
}

extern "C" bool q4_hip_route_dev(const float *d_rlog, const int32_t *devtab,
                                 int32_t *d_es, float *d_ew, int32_t *d_nhit,
                                 void *mail, unsigned long long *d_seq,
                                 unsigned long long *d_rseq,
                                 unsigned long long *d_prof,
                                 uint32_t n_exp, uint32_t topk) {
    if (!g_ok || !d_rlog || !devtab || !d_es || !d_ew || !d_nhit || !mail ||
        !d_seq || n_exp > 1024)
        return false;
    route_dev_k<<<dim3(1), dim3(32), 0, g_str>>>(d_rlog, devtab, d_es, d_ew,
                                                 d_nhit, (q4_mail_t *)mail,
                                                 d_seq, d_rseq, d_prof,
                                                 n_exp, topk);
    return hipGetLastError() == hipSuccess;
}

/* Enqueue a 4-byte write on the dedicated copy stream — used to publish or
 * clear device residency-table entries so the writes stay ordered with the
 * expert payload uploads on that stream.  Falls back to a one-thread kernel
 * when stream write-value ops are unavailable. */
__global__ static void set_i32_k(int32_t *dst, int32_t v) { *dst = v; }

extern "C" bool q4_hip_copy_i32(int32_t *dst, int32_t v) {
    if (!dst) return false;
    if (g_copy)
        return hipStreamWriteValue32(g_copy, dst, (uint32_t)v, 0) ==
               hipSuccess;
    set_i32_k<<<dim3(1), dim3(32), 0, 0>>>(dst, v);
    return hipGetLastError() == hipSuccess;
}

/* memset/memcpy on the copy stream — keeps device-table writes ordered with
 * expert payload uploads (which also ride g_copy). */
extern "C" bool q4_hip_copy_memset(void *dst, int v, size_t n) {
    if (!dst || !n) return false;
    hipStream_t st = g_copy ? g_copy : 0;
    return hipMemsetAsync(dst, v, n, st) == hipSuccess;
}

extern "C" bool q4_hip_copy_h2d(void *dst, const void *src, size_t n) {
    if (!dst || !src || !n) return false;
    hipStream_t st = g_copy ? g_copy : 0;
    return hipMemcpyAsync(dst, src, n, hipMemcpyHostToDevice, st) ==
           hipSuccess;
}

/* Dequant one IQ block into plain floats so a prefill row can be reused
 * across every token that routed here. */
__device__ static void deq_iq2s_blk(const uint8_t *blk, float *dst) {
    const float d = f16d(blk);
    const uint8_t *qs = blk + 2;
    const uint8_t *qh = blk + 66;
    const uint8_t *scales = blk + 74;
    const uint8_t *signs = qs + 32;
    uint32_t xi = 0;
    for (int ib = 0; ib < 8; ib++) {
        const float db0 = d * (0.5f + (float)(scales[ib] & 0xf)) * 0.25f;
        const float db1 = d * (0.5f + (float)(scales[ib] >> 4)) * 0.25f;
        for (int l = 0; l < 4; l++) {
            const float dl = l < 2 ? db0 : db1;
            const int idx = qs[l] | ((qh[ib] << (8 - 2 * l)) & 0x300);
            const uint64_t gv = q4_iq2s_grid[idx];
            const uint8_t sg = signs[l];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float s = (sg & q4_kmask_iq2xs[j]) ? -1.f : 1.f;
                dst[xi++] = dl * iq_u8(gv, j) * s;
            }
        }
        qs += 4;
        signs += 4;
    }
}

__device__ static void deq_iq3s_blk(const uint8_t *blk, float *dst) {
    const float d = f16d(blk);
    const uint8_t *qs = blk + 2;
    const uint8_t *qh = blk + 66;
    const uint8_t *signs = blk + 74;
    const uint8_t *scales = blk + 106;
    uint32_t xi = 0;
    for (int ib = 0; ib < 8; ib += 2) {
        const float db1 = d * (1.f + 2.f * (float)(scales[ib / 2] & 0xf));
        const float db2 = d * (1.f + 2.f * (float)(scales[ib / 2] >> 4));
        for (int half = 0; half < 2; half++) {
            const float db = half == 0 ? db1 : db2;
            const uint8_t qhb = qh[half];
            for (int l = 0; l < 4; l++) {
                const int i1 = qs[2 * l] | ((qhb << (8 - 2 * l)) & 256);
                const int i2 = qs[2 * l + 1] | ((qhb << (7 - 2 * l)) & 256);
                const uint32_t g1 = q4_iq3s_grid[i1];
                const uint32_t g2 = q4_iq3s_grid[i2];
                const uint8_t sg = signs[l];
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    dst[xi + j] = db * iq_u8(g1, j) *
                                  ((sg & q4_kmask_iq2xs[j]) ? -1.f : 1.f);
                    dst[xi + 4 + j] = db * iq_u8(g2, j) *
                                      ((sg & q4_kmask_iq2xs[j + 4]) ? -1.f
                                                                   : 1.f);
                }
                xi += 8;
            }
            qs += 8;
            signs += 4;
        }
        qh += 2;
    }
}

__device__ static void deq_iq4nl_blk(const uint8_t *blk, float *dst) {
    const float d = f16d(blk);
    const uint8_t *qs = blk + 2;
    for (int j = 0; j < 16; j++) {
        const uint8_t q = qs[j];
        dst[j] = d * (float)q4_kvalues_iq4nl[q & 0xf];
        dst[j + 16] = d * (float)q4_kvalues_iq4nl[q >> 4];
    }
}

/* Per-element dequant: inverse map of deq_*_blk so the shared-memory fill can
 * fan out over all threads instead of one thread per quant block. */
__device__ static float deq_iq2s_elem(const uint8_t *blk, uint32_t r) {
    const float d = f16d(blk);
    const uint32_t ib = r >> 5, l = (r & 31u) >> 3, j = r & 7u;
    const float sc = (l < 2) ? (float)(blk[74 + ib] & 0xf)
                             : (float)(blk[74 + ib] >> 4);
    const float dl = d * (0.5f + sc) * 0.25f;
    const int idx = blk[2 + ib * 4 + l] | ((blk[66 + ib] << (8 - 2 * l)) & 0x300);
    const float s = (blk[2 + 32 + ib * 4 + l] & (1u << j)) ? -1.f : 1.f;
    return dl * iq_u8(q4_iq2s_grid[idx], j) * s;
}

__device__ static float deq_iq3s_elem(const uint8_t *blk, uint32_t r) {
    const float d = f16d(blk);
    const uint32_t p = r >> 6, half = (r >> 5) & 1u, rr = r & 31u;
    const uint32_t l = rr >> 3, jj = rr & 7u;
    const float sc = (half == 0) ? (float)(blk[106 + p] & 0xf)
                                 : (float)(blk[106 + p] >> 4);
    const float db = d * (1.f + 2.f * sc);
    const uint8_t qhb = blk[66 + p * 2 + half];
    const int i1 = blk[2 + p * 16 + half * 8 + 2 * l] |
                   ((qhb << (8 - 2 * l)) & 256);
    const int i2 = blk[2 + p * 16 + half * 8 + 2 * l + 1] |
                   ((qhb << (7 - 2 * l)) & 256);
    const uint8_t sg = blk[74 + p * 8 + half * 4 + l];
    const float s = (sg & q4_kmask_iq2xs[jj]) ? -1.f : 1.f;
    if (jj < 4u) return db * iq_u8(q4_iq3s_grid[i1], jj) * s;
    return db * iq_u8(q4_iq3s_grid[i2], jj - 4u) * s;
}

__device__ static float deq_iq4nl_elem(const uint8_t *blk, uint32_t r) {
    const float d = f16d(blk);
    const uint8_t q = blk[2 + (r & 15u)];
    const uint8_t v = (r < 16u) ? (q & 0xf) : (q >> 4);
    return d * (float)q4_kvalues_iq4nl[v];
}

/* Batched dequant: 32 contiguous elements at once (one 32-run within the
 * quantization block; e0 is the element offset, multiple of 32).  Scales,
 * grid bytes and signs are fetched once per 8/32-element run instead of
 * once per element — ~10x fewer instructions in the WMMA tile loads.
 * dst is float or __half; T picks the store conversion. */
__device__ static inline void deq_st(float v, float *d, uint32_t i) {
    d[i] = v;
}
__device__ static inline void deq_st(float v, __half *d, uint32_t i) {
    d[i] = __float2half(v);
}

template<typename T>
__device__ static void deq_iq2s_x32(const uint8_t *blk, uint32_t e0, T *dst) {
    const float d = f16d(blk);
    const uint32_t ib = e0 >> 5;
    const uint8_t *qs = blk + 2 + ib * 4;
    const uint8_t *sg = blk + 34 + ib * 4;
    const uint32_t qh = blk[66 + ib];
    const uint32_t sc = blk[74 + ib];
    const float dl0 = d * (0.5f + (float)(sc & 0xf)) * 0.25f;
    const float dl1 = d * (0.5f + (float)(sc >> 4)) * 0.25f;
#pragma unroll
    for (int l = 0; l < 4; l++) {
        const float dl = l < 2 ? dl0 : dl1;
        const uint32_t idx = qs[l] | ((qh << (8 - 2 * l)) & 0x300u);
        const uint64_t gv = q4_iq2s_grid[idx];
        const uint32_t s = sg[l];
#pragma unroll
        for (int j = 0; j < 8; j++)
            deq_st(dl * iq_u8(gv, j) * ((s & (1u << j)) ? -1.f : 1.f),
                   dst, (uint32_t)(l * 8 + j));
    }
}

template<typename T>
__device__ static void deq_iq3s_x32(const uint8_t *blk, uint32_t e0, T *dst) {
    const float d = f16d(blk);
    const uint32_t p = e0 >> 6, half = (e0 >> 5) & 1u;
    const uint32_t sc = blk[106 + p];
    const float db =
        d * (1.f + 2.f * (float)(half ? (sc >> 4) : (sc & 0xf)));
    const uint32_t qhb = blk[66 + p * 2 + half];
    const uint8_t *qs = blk + 2 + p * 16 + half * 8;
    const uint8_t *sg = blk + 74 + p * 8 + half * 4;
#pragma unroll
    for (int l = 0; l < 4; l++) {
        const uint32_t i1 = qs[2 * l] | ((qhb << (8 - 2 * l)) & 256u);
        const uint32_t i2 = qs[2 * l + 1] | ((qhb << (7 - 2 * l)) & 256u);
        const uint32_t g1 = q4_iq3s_grid[i1];
        const uint32_t g2 = q4_iq3s_grid[i2];
        const uint32_t s = sg[l];
#pragma unroll
        for (int m = 0; m < 4; m++) {
            deq_st(db * iq_u8(g1, m) *
                       ((s & q4_kmask_iq2xs[m]) ? -1.f : 1.f),
                   dst, (uint32_t)(l * 8 + m));
            deq_st(db * iq_u8(g2, m) *
                       ((s & q4_kmask_iq2xs[m + 4]) ? -1.f : 1.f),
                   dst, (uint32_t)(l * 8 + 4 + m));
        }
    }
}

template<typename T>
__device__ static void deq_iq4nl_x32(const uint8_t *blk, uint32_t, T *dst) {
    const float d = f16d(blk);
    const uint8_t *qs = blk + 2;
#pragma unroll
    for (int j = 0; j < 16; j++) {
        const uint32_t q = qs[j];
        deq_st(d * (float)q4_kvalues_iq4nl[q & 0xf], dst, (uint32_t)j);
        deq_st(d * (float)q4_kvalues_iq4nl[q >> 4], dst, (uint32_t)(j + 16));
    }
}

/* iq3_xxs: 256-elem block {f16 d, qs[96]}; qs[0..63] grid indices,
 * qs[64..95] = 8 aux32 sign/scale words. */
__device__ static float deq_iq3xxs_elem(const uint8_t *blk, uint32_t r) {
    const float d = f16d(blk);
    const uint32_t ib = r >> 5, rr = r & 31u;
    const uint32_t l = rr >> 3, jj = rr & 7u;
    uint32_t aux32;
    memcpy(&aux32, blk + 2 + 64 + 4 * ib, 4);
    const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
    const uint8_t sg = q4_ksigns_iq2xs[(aux32 >> (7 * l)) & 127u];
    const uint32_t gv = q4_iq3xxs_grid[blk[2 + ib * 8 + 2 * l + (jj >> 2)]];
    const float s = (sg & (1u << jj)) ? -1.f : 1.f;
    return db * iq_u8(gv, jj & 3u) * s;
}

template<typename T>
__device__ static void deq_iq3xxs_x32(const uint8_t *blk, uint32_t e0, T *dst) {
    const float d = f16d(blk);
    const uint32_t ib = e0 >> 5;
    uint32_t aux32;
    memcpy(&aux32, blk + 2 + 64 + 4 * ib, 4);
    const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
    const uint8_t *q3 = blk + 2 + ib * 8;
#pragma unroll
    for (int l = 0; l < 4; l++) {
        const uint8_t s = q4_ksigns_iq2xs[(aux32 >> (7 * l)) & 127u];
        const uint32_t g1 = q4_iq3xxs_grid[q3[2 * l]];
        const uint32_t g2 = q4_iq3xxs_grid[q3[2 * l + 1]];
#pragma unroll
        for (int m = 0; m < 4; m++) {
            deq_st(db * iq_u8(g1, m) * ((s & (1u << m)) ? -1.f : 1.f),
                   dst, (uint32_t)(l * 8 + m));
            deq_st(db * iq_u8(g2, m) * ((s & (1u << (m + 4))) ? -1.f : 1.f),
                   dst, (uint32_t)(l * 8 + 4 + m));
        }
    }
}

/* iq4_xs: 256-elem block {f16 d, u16 scales_h, u8 scales_l[4], qs[128]}. */
__device__ static float deq_iq4xs_elem(const uint8_t *blk, uint32_t r) {
    const float d = f16d(blk);
    const uint32_t ib = r >> 5, rr = r & 31u;
    uint32_t sh;
    memcpy(&sh, blk + 2, 2);
    const int ls = ((blk[4 + (ib >> 1)] >> (4 * (ib & 1u))) & 0xf) |
                   ((int)((sh >> (2 * ib)) & 3u) << 4);
    const float dl = d * (float)(ls - 32);
    const uint8_t q = blk[8 + ib * 16 + (rr & 15u)];
    const uint8_t v = (rr < 16u) ? (q & 0xf) : (q >> 4);
    return dl * (float)q4_kvalues_iq4nl[v];
}

template<typename T>
__device__ static void deq_iq4xs_x32(const uint8_t *blk, uint32_t e0, T *dst) {
    const float d = f16d(blk);
    const uint32_t ib = e0 >> 5;
    uint32_t sh;
    memcpy(&sh, blk + 2, 2);
    const int ls = ((blk[4 + (ib >> 1)] >> (4 * (ib & 1u))) & 0xf) |
                   ((int)((sh >> (2 * ib)) & 3u) << 4);
    const float dl = d * (float)(ls - 32);
    const uint8_t *qs = blk + 8 + ib * 16;
#pragma unroll
    for (int j = 0; j < 16; j++) {
        deq_st(dl * (float)q4_kvalues_iq4nl[qs[j] & 0xf], dst, (uint32_t)j);
        deq_st(dl * (float)q4_kvalues_iq4nl[qs[j] >> 4], dst,
               (uint32_t)(j + 16));
    }
}

__device__ static float deq_q8_0_elem(const uint8_t *blk, uint32_t r) {
    return f16d(blk) * (float)(const int8_t)blk[2 + r];
}

template<typename T>
__device__ static void deq_q8_0_x32(const uint8_t *blk, uint32_t, T *dst) {
    const float d = f16d(blk);
#pragma unroll
    for (int j = 0; j < 32; j++)
        deq_st(d * (float)(const int8_t)blk[2 + j], dst, (uint32_t)j);
}

/* q4_0: 32-value block = fp16 d + 16 packed nibbles, lo nibbles are elements
 * 0..15, hi 16..31. */
__device__ static float deq_q40_elem(const uint8_t *blk, uint32_t r) {
    const float d = f16d(blk);
    const uint8_t q = blk[2 + (r & 15u)];
    const int v = (r < 16u) ? (q & 0xf) : (q >> 4);
    return d * (float)(v - 8);
}

template<typename T>
__device__ static void deq_q40_x32(const uint8_t *blk, uint32_t, T *dst) {
    const float d = f16d(blk);
    const uint8_t *qs = blk + 2;
#pragma unroll
    for (int j = 0; j < 16; j++) {
        const uint32_t q = qs[j];
        deq_st(d * (float)((int)(q & 0xf) - 8), dst, (uint32_t)j);
        deq_st(d * (float)((int)(q >> 4) - 8), dst, (uint32_t)(j + 16));
    }
}

/* q5_0: q4_0 nibble order + high bits packed as a u32 at blk+2 (bit i =
 * element i), offset -16. */
__device__ static float deq_q50_elem(const uint8_t *blk, uint32_t r) {
    const float d = f16d(blk);
    uint32_t qh;
    memcpy(&qh, blk + 2, 4);
    const uint8_t q = blk[6 + (r & 15u)];
    const int v = (int)(((r < 16u) ? (q & 0xf) : (q >> 4)) |
                        (((qh >> r) & 1u) << 4)) - 16;
    return d * (float)v;
}

template<typename T>
__device__ static void deq_q50_x32(const uint8_t *blk, uint32_t, T *dst) {
    const float d = f16d(blk);
    uint32_t qh;
    memcpy(&qh, blk + 2, 4);
    const uint8_t *qs = blk + 6;
#pragma unroll
    for (int j = 0; j < 16; j++) {
        const uint32_t q = qs[j];
        deq_st(d * (float)((int)((q & 0xf) | (((qh >> j) & 1u) << 4)) - 16),
               dst, (uint32_t)j);
        deq_st(d * (float)((int)((q >> 4) |
                                 (((qh >> (j + 16)) & 1u) << 4)) - 16),
               dst, (uint32_t)(j + 16));
    }
}

/* q2_0: 64-value block = fp16 d + 16 bytes of 2-bit quads, (q - 1) * d.
 * x32 covers half a block: e0 is 0 or 32. */
__device__ static float deq_q20_elem(const uint8_t *blk, uint32_t r) {
    const float d = f16d(blk);
    return d * (float)((int)((blk[2 + (r >> 2)] >> ((r & 3u) * 2)) & 3u) - 1);
}

template<typename T>
__device__ static void deq_q20_x32(const uint8_t *blk, uint32_t e0, T *dst) {
    const float d = f16d(blk);
    const uint8_t *qs = blk + 2 + (e0 >> 2);
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const uint32_t qv = qs[j];
#pragma unroll
        for (int m = 0; m < 4; m++)
            deq_st(d * (float)((int)((qv >> (m * 2)) & 3u) - 1),
                   dst, (uint32_t)(j * 4 + m));
    }
}

/* q6_k: 256-value superblock, 210 bytes (ql[128] lo nibbles, qh[64] hi
 * bit-pairs, sc[16] int8 per-16 scales, fp16 d at +208). */
__device__ static float deq_q6_k_elem(const uint8_t *blk, uint32_t r) {
    const float d = f16d(blk + 208u);
    const uint32_t half = r >> 7, rr = r & 127u;
    const uint32_t g = rr >> 5, l = rr & 31u;
    const uint32_t ql = blk[half * 64u + l + (g & 1u) * 32u];
    const uint32_t qh = blk[128u + half * 32u + l];
    const uint32_t q =
        ((g & 2u) ? (ql >> 4) : (ql & 0xFu)) | (((qh >> (2u * g)) & 3u) << 4);
    const int32_t sc =
        (int32_t)(const int8_t)blk[192u + half * 8u + g * 2u + (l >> 4)];
    return d * (float)sc * (float)((int32_t)q - 32);
}

template<typename T>
__device__ static void deq_q6_k_x32(const uint8_t *blk, uint32_t e0, T *dst) {
    const float d = f16d(blk + 208u);
    const uint32_t half = e0 >> 7, g = (e0 & 127u) >> 5;
    const uint8_t *ql = blk + half * 64u + (g & 1u) * 32u;
    const uint8_t *qh = blk + 128u + half * 32u;
    const int8_t *sc = (const int8_t *)(blk + 192u + half * 8u + g * 2u);
    const uint32_t qsh = 2u * g;
    const uint32_t hi = g & 2u;
    const float d0 = d * (float)sc[0], d1 = d * (float)sc[1];
#pragma unroll
    for (uint32_t j = 0; j < 32; j++) {
        const uint32_t q = (hi ? (uint32_t)(ql[j] >> 4)
                               : (uint32_t)(ql[j] & 0xFu)) |
                           ((((uint32_t)qh[j]) >> qsh) & 3u) << 4;
        deq_st((j < 16u ? d0 : d1) * (float)((int32_t)q - 32), dst, j);
    }
}

/* q4_k / q5_k: 256-value superblock. d fp16@0, dmin fp16@2, 12 packed
 * 6-bit scale/min bytes @4 (8 sub-blocks of 32 elems via scale_min_k4
 * indexing), then q4_k: qs[128]@16; q5_k adds qh[32]@16, ql[128]@48.
 * Element e: j64 = e>>6, l = e&31, half = (e>>5)&1, is = j64*2+half. */
__device__ static void scale_min_k4_dev(int is, const uint8_t *sc,
                                        float *d1, float *m1) {
    uint8_t s, m;
    if (is < 4) {
        s = sc[is] & 63;
        m = sc[is + 4] & 63;
    } else {
        s = (sc[is + 4] & 0xF) | ((sc[is - 4] >> 6) << 4);
        m = (sc[is + 4] >> 4) | ((sc[is] >> 6) << 4);
    }
    *d1 = f16d(sc - 4) * (float)s;
    *m1 = f16d(sc - 2) * (float)m;
}

__device__ static float deq_q4k_elem(const uint8_t *blk, uint32_t r) {
    const uint32_t j64 = r >> 6, l = r & 31u, half = (r >> 5) & 1u;
    float d1, m1;
    scale_min_k4_dev((int)(j64 * 2u + half), blk + 4u, &d1, &m1);
    const uint32_t q = blk[16u + j64 * 32u + l];
    return d1 * (float)(half ? (q >> 4) : (q & 0xFu)) - m1;
}

template<typename T>
__device__ static void deq_q4k_x32(const uint8_t *blk, uint32_t e0, T *dst) {
    const uint32_t j64 = e0 >> 6, half = (e0 >> 5) & 1u;
    float d1, m1;
    scale_min_k4_dev((int)(j64 * 2u + half), blk + 4u, &d1, &m1);
    const uint8_t *qs = blk + 16u + j64 * 32u;
#pragma unroll
    for (uint32_t j = 0; j < 32; j++) {
        const uint32_t q = half ? (uint32_t)(qs[j] >> 4)
                                : (uint32_t)(qs[j] & 0xFu);
        deq_st(d1 * (float)q - m1, dst, j);
    }
}

__device__ static float deq_q5k_elem(const uint8_t *blk, uint32_t r) {
    const uint32_t j64 = r >> 6, l = r & 31u, half = (r >> 5) & 1u;
    const uint32_t is = j64 * 2u + half;
    float d1, m1;
    scale_min_k4_dev((int)is, blk + 4u, &d1, &m1);
    const uint32_t q = blk[48u + j64 * 32u + l];
    const uint32_t q5 = (half ? (q >> 4) : (q & 0xFu)) |
                        (((blk[16u + l] >> is) & 1u) << 4);
    return d1 * (float)q5 - m1;
}

template<typename T>
__device__ static void deq_q5k_x32(const uint8_t *blk, uint32_t e0, T *dst) {
    const uint32_t j64 = e0 >> 6, half = (e0 >> 5) & 1u;
    const uint32_t is = j64 * 2u + half;
    float d1, m1;
    scale_min_k4_dev((int)is, blk + 4u, &d1, &m1);
    const uint8_t *ql = blk + 48u + j64 * 32u;
    const uint8_t *qh = blk + 16u;
#pragma unroll
    for (uint32_t j = 0; j < 32; j++) {
        const uint32_t q = half ? (uint32_t)(ql[j] >> 4)
                                : (uint32_t)(ql[j] & 0xFu);
        deq_st(d1 * (float)(q | (((qh[j] >> is) & 1u) << 4)) - m1, dst, j);
    }
}

/* 32-element tile fill for the WMMA A-loads: covers MOEW_KC columns of one
 * row via KC/32 runs; falls back to scalar deq at the kn/nrows fringe so
 * tail chunks and clipped rows still produce zeros where needed. */
#define Q4_DEQ_TILE(As, wp, r0, r, k0, sb, kn, nrows, nb, BLK, BBYTES,       \
                    deq_elem, deq32)                                        \
    do {                                                                   \
        const uint32_t e2_ = (k0) + (sb) * 32u;                             \
        const uint8_t *b_ = (wp) + (size_t)((r0) + (r)) * (nb) * (BBYTES) + \
                            (size_t)(e2_ / (BLK)) * (BBYTES);               \
        if ((sb) * 32u + 32u <= (kn) && (r0) + (r) < (nrows)) {             \
            deq32(b_, e2_ % (BLK), &(As)[(r)][(sb) * 32u]);                 \
        } else {                                                           \
            for (uint32_t j = 0; j < 32u; j++) {                            \
                const uint32_t kc_ = (sb) * 32u + j;                        \
                (As)[(r)][kc_] = __float2half(                              \
                    kc_ < (kn) && (r0) + (r) < (nrows)                      \
                        ? deq_elem(b_, (e2_ % (BLK)) + j)                   \
                        : 0.f);                                             \
            }                                                              \
        }                                                                  \
    } while (0)

/* One row per block. All threads cooperatively dequant the row into shared
 * memory (element-indexed), then each warp dots one token: lanes split ncols
 * (coalesced x reads) and shfl-reduce. */
#define Q4_GEMM_REUSE(name, deq_elem, BLK, BBYTES)                             \
    __global__ static void name(const uint8_t *w, const float *x,              \
                                const int32_t *which, float *out,              \
                                uint32_t nrows, uint32_t ncols, float scale,   \
                                uint32_t n_batch) {                            \
        uint32_t row = blockIdx.x;                                             \
        if (row >= nrows) return;                                             \
        extern __shared__ float rowv[];                                       \
        uint32_t nb = ncols / (BLK);                                           \
        const uint8_t *wr = w + (size_t)row * nb * (BBYTES);                   \
        for (uint32_t e = threadIdx.x; e < ncols; e += blockDim.x)            \
            rowv[e] = deq_elem(wr + (size_t)(e / (BLK)) * (BBYTES),           \
                               e % (BLK));                                     \
        __syncthreads();                                                       \
        uint32_t lane = threadIdx.x & 31u;                                    \
        for (uint32_t tok = threadIdx.x >> 5; tok < n_batch;                  \
             tok += blockDim.x >> 5) {                                        \
            const float *xt =                                                 \
                x + (size_t)(which ? which[tok] : (int32_t)tok) * ncols;      \
            float acc = 0.f;                                                   \
            for (uint32_t i = lane; i < ncols; i += 32)                       \
                acc += rowv[i] * xt[i];                                       \
            for (uint32_t o = 16; o; o >>= 1)                                 \
                acc += __shfl_down(acc, o);                                   \
            if (lane == 0)                                                    \
                out[(size_t)tok * nrows + row] = acc * scale;                 \
        }                                                                      \
    }

Q4_GEMM_REUSE(gemm_iq2_reuse, deq_iq2s_elem, 256u, 82u)
Q4_GEMM_REUSE(gemm_iq3_reuse, deq_iq3s_elem, 256u, 110u)
Q4_GEMM_REUSE(gemm_iq4_reuse, deq_iq4nl_elem, 32u, 18u)
Q4_GEMM_REUSE(gemm_iq3xxs_reuse, deq_iq3xxs_elem, 256u, 98u)
Q4_GEMM_REUSE(gemm_iq4xs_reuse, deq_iq4xs_elem, 256u, 136u)
Q4_GEMM_REUSE(gemm_q40_reuse, deq_q40_elem, 32u, 18u)
Q4_GEMM_REUSE(gemm_q50_reuse, deq_q50_elem, 32u, 22u)
Q4_GEMM_REUSE(gemm_q20_reuse, deq_q20_elem, 64u, 18u)

/* Batched prefill expert GEMM: grid (row, expert-entry).  Entry e has ec[e]
 * assigned tokens, listed at atok[eb[e]..eb[e]+ec[e]) (global token ids into
 * x), or — when atok == NULL — assignment-indexed rows of x directly.
 * Weights live in the L1 slot arena: base + slots[e]*stride + woff.
 * One launch replaces n_experts small GEMM launches. */
#define Q4_MOE_PF(name, deq_elem, BLK, BBYTES)                                 \
    __global__ static void name(float *out, const uint8_t *wbase,              \
                                uint64_t stride, uint64_t woff,                \
                                const int32_t *slots, const uint32_t *eb,      \
                                const uint32_t *ec, const int32_t *atok,       \
                                const float *x, uint32_t nrows,                \
                                uint32_t ncols, float scale) {                 \
        /* two rows per block: one x load feeds two dots */                   \
        const uint32_t R = 2u;                                                \
        uint32_t r0 = blockIdx.x * R, e = blockIdx.y;                         \
        uint32_t K = ec[e];                                                   \
        if (!K) return;                                                       \
        extern __shared__ float rowv[]; /* [R][ncols] */                      \
        uint32_t nb = ncols / (BLK);                                          \
        const uint8_t *wp = wbase + (size_t)slots[e] * stride + woff;         \
        for (uint32_t t = threadIdx.x; t < ncols * R; t += blockDim.x) {      \
            uint32_t rr = t / ncols, i = t - rr * ncols;                      \
            uint32_t row = r0 + rr < nrows ? r0 + rr : r0;                    \
            rowv[t] = deq_elem(wp + (size_t)row * nb * (BBYTES) +             \
                               (size_t)(i / (BLK)) * (BBYTES), i % (BLK));    \
        }                                                                     \
        __syncthreads();                                                       \
        uint32_t lane = threadIdx.x & 31u, nw = blockDim.x >> 5;              \
        uint32_t a0 = eb[e];                                                  \
        for (uint32_t k = threadIdx.x >> 5; k < K; k += nw) {                 \
            const float *xt =                                                 \
                x + (size_t)(atok ? atok[a0 + k] : (int32_t)(a0 + k)) *       \
                    ncols;                                                    \
            float acc[R];                                                     \
            for (uint32_t r = 0; r < R; r++) acc[r] = 0.f;                    \
            for (uint32_t i = lane; i < ncols; i += 32) {                     \
                float xv = xt[i];                                             \
                for (uint32_t r = 0; r < R; r++)                              \
                    acc[r] += rowv[r * ncols + i] * xv;                       \
            }                                                                 \
            for (uint32_t r = 0; r < R; r++) {                                \
                float a = acc[r];                                             \
                for (uint32_t o = 16; o; o >>= 1)                             \
                    a += __shfl_down(a, o);                                   \
                if (lane == 0 && r0 + r < nrows)                              \
                    out[(size_t)(a0 + k) * nrows + r0 + r] = a * scale;       \
            }                                                                 \
        }                                                                      \
    }

Q4_MOE_PF(moepf_iq2_k, deq_iq2s_elem, 256u, 82u)
Q4_MOE_PF(moepf_iq3_k, deq_iq3s_elem, 256u, 110u)
Q4_MOE_PF(moepf_iq4_k, deq_iq4nl_elem, 32u, 18u)
Q4_MOE_PF(moepf_iq3xxs_k, deq_iq3xxs_elem, 256u, 98u)
Q4_MOE_PF(moepf_iq4xs_k, deq_iq4xs_elem, 256u, 136u)
Q4_MOE_PF(moepf_q40_k, deq_q40_elem, 32u, 18u)
Q4_MOE_PF(moepf_q50_k, deq_q50_elem, 32u, 22u)
Q4_MOE_PF(moepf_q20_k, deq_q20_elem, 64u, 18u)

/* Fused gate+up variant: same shape/type, same token list — one smem pair of
 * rows (gate row r, up row r), one launch, half the x traffic. */
#define Q4_MOE_PF2(name, deq_elem, BLK, BBYTES)                                \
    __global__ static void name(float *out0, float *out1,                      \
                                const uint8_t *wbase, uint64_t stride,         \
                                uint64_t woff0, uint64_t woff1,                \
                                const int32_t *slots, const uint32_t *eb,      \
                                const uint32_t *ec, const int32_t *atok,       \
                                const float *x, uint32_t nrows,                \
                                uint32_t ncols, float scale) {                 \
        uint32_t row = blockIdx.x, e = blockIdx.y;                            \
        uint32_t K = ec[e];                                                   \
        if (!K) return;                                                       \
        extern __shared__ float rowv[]; /* [0]=gate row, [1]=up row */        \
        uint32_t nb = ncols / (BLK);                                          \
        const uint8_t *wp = wbase + (size_t)slots[e] * stride;                \
        for (uint32_t i = threadIdx.x; i < ncols; i += blockDim.x) {          \
            uint32_t bl = i / (BLK), rr = i % (BLK);                          \
            rowv[i] = deq_elem(wp + woff0 + (size_t)row * nb * (BBYTES) +     \
                               (size_t)bl * (BBYTES), rr);                    \
            rowv[ncols + i] =                                                 \
                deq_elem(wp + woff1 + (size_t)row * nb * (BBYTES) +           \
                         (size_t)bl * (BBYTES), rr);                          \
        }                                                                     \
        __syncthreads();                                                       \
        uint32_t lane = threadIdx.x & 31u, nw = blockDim.x >> 5;              \
        uint32_t a0 = eb[e];                                                  \
        for (uint32_t k = threadIdx.x >> 5; k < K; k += nw) {                 \
            const float *xt =                                                 \
                x + (size_t)(atok ? atok[a0 + k] : (int32_t)(a0 + k)) *       \
                    ncols;                                                    \
            float acc = 0.f, acc1 = 0.f;                                      \
            for (uint32_t i = lane; i < ncols; i += 32) {                     \
                float xv = xt[i];                                             \
                acc += rowv[i] * xv;                                          \
                acc1 += rowv[ncols + i] * xv;                                 \
            }                                                                 \
            for (uint32_t o = 16; o; o >>= 1) {                               \
                acc += __shfl_down(acc, o);                                   \
                acc1 += __shfl_down(acc1, o);                                 \
            }                                                                 \
            if (lane == 0) {                                                  \
                out0[(size_t)(a0 + k) * nrows + row] = acc * scale;           \
                out1[(size_t)(a0 + k) * nrows + row] = acc1 * scale;          \
            }                                                                 \
        }                                                                      \
    }

Q4_MOE_PF2(moepf2_iq2_k, deq_iq2s_elem, 256u, 82u)
Q4_MOE_PF2(moepf2_iq3_k, deq_iq3s_elem, 256u, 110u)
Q4_MOE_PF2(moepf2_iq4_k, deq_iq4nl_elem, 32u, 18u)
Q4_MOE_PF2(moepf2_iq3xxs_k, deq_iq3xxs_elem, 256u, 98u)
Q4_MOE_PF2(moepf2_iq4xs_k, deq_iq4xs_elem, 256u, 136u)
Q4_MOE_PF2(moepf2_q40_k, deq_q40_elem, 32u, 18u)
Q4_MOE_PF2(moepf2_q50_k, deq_q50_elem, 32u, 22u)
Q4_MOE_PF2(moepf2_q20_k, deq_q20_elem, 64u, 18u)

/* RDNA3 wave32 WMMA (16x16x16 f16 in, f32 acc): lane holds one A row
 * (all 16 k), one B column (all 16 k), and output column lane%16 of rows
 * 2i+lane/16 of its wave's tile (verified empirically).  Block = 4 waves x
 * 32 = 128 threads covering a 64-row tile; up to 64 tokens per pass in four
 * 16-column WMMA tiles.  Weight tiles are dequantized into LDS as f16 once
 * per token pass; X is staged to f16 LDS per K-chunk. */
typedef _Float16 wmma_a16 __attribute__((ext_vector_type(16)));
typedef float wmma_c8 __attribute__((ext_vector_type(8)));
#define MOEW_ROWS 64
#define MOEW_TOKS 64
#define MOEW_KC 64
#define MOEW_PAD 2

/* Fused gate+up + silu(g)*u: writes act output to out[a0+t][row]. */
#define Q4_MOEW2(name, deq_elem, deq32, BLK, BBYTES)                                 \
    __global__ static void name(float *out, const uint8_t *wbase,             \
            uint64_t stride, uint64_t woff0, uint64_t woff1,                  \
            const int32_t *slots, const uint32_t *eb, const uint32_t *ec,     \
            const int32_t *atok, const float *x, uint32_t nrows,              \
            uint32_t ncols) {                                                \
        const uint32_t r0 = blockIdx.x * MOEW_ROWS, e = blockIdx.y;           \
        const uint32_t K = ec[e], a0 = eb[e];                                 \
        if (!K) return;                                                      \
        __shared__ __half As[2][MOEW_ROWS][MOEW_KC + MOEW_PAD];               \
        __shared__ __half Xs[MOEW_KC][MOEW_TOKS + MOEW_PAD];                  \
        const uint8_t *wp = wbase + (size_t)slots[e] * stride;                \
        const uint32_t nb = ncols / (BLK);                                    \
        const int wave = threadIdx.x >> 5, l = threadIdx.x & 31;              \
        for (uint32_t t0 = 0; t0 < K; t0 += MOEW_TOKS) {                      \
            const uint32_t tn = K - t0 < MOEW_TOKS ? K - t0 : MOEW_TOKS;      \
            wmma_c8 ag[MOEW_TOKS / 16], au[MOEW_TOKS / 16];                   \
            for (uint32_t nt = 0; nt < MOEW_TOKS / 16; nt++) {                \
                ag[nt] = (wmma_c8){};                                         \
                au[nt] = (wmma_c8){};                                         \
            }                                                                 \
            for (uint32_t k0 = 0; k0 < ncols; k0 += MOEW_KC) {                \
                const uint32_t kn = ncols - k0 < MOEW_KC ? ncols - k0         \
                                                         : MOEW_KC;           \
                for (uint32_t i = threadIdx.x;                                \
                     i < 2u * MOEW_ROWS * (MOEW_KC / 32u);                    \
                     i += blockDim.x) {                                       \
                    const uint32_t t = i / (MOEW_ROWS * (MOEW_KC / 32u));     \
                    const uint32_t rr = i % (MOEW_ROWS * (MOEW_KC / 32u));    \
                    const uint32_t r = rr / (MOEW_KC / 32u);                  \
                    const uint32_t sb = rr % (MOEW_KC / 32u);                 \
                    Q4_DEQ_TILE(As[t], wp + (t ? woff1 : woff0), r0, r, k0,   \
                                sb, kn, nrows, nb, BLK, BBYTES, deq_elem,     \
                                deq32);                                       \
                }                                                             \
                for (uint32_t i = threadIdx.x; i < MOEW_KC * MOEW_TOKS;       \
                     i += blockDim.x) {                                       \
                    uint32_t tt = i / MOEW_KC, kc = i % MOEW_KC;              \
                    float v = 0.f;                                            \
                    if (kc < kn && tt < tn) {                                 \
                        int32_t tok =                                         \
                            atok ? atok[a0 + t0 + tt]                         \
                                 : (int32_t)(a0 + t0 + tt);                   \
                        v = x[(size_t)tok * ncols + k0 + kc];                 \
                    }                                                         \
                    Xs[kc][tt] = __float2half(v);                             \
                }                                                             \
                __syncthreads();                                              \
                for (uint32_t ks = 0; ks < kn; ks += 16) {                    \
                    wmma_a16 ag16, au16;                                      \
                    const uint32_t rw = wave * 16 + (l & 15);                 \
                    for (uint32_t i = 0; i < 16; i++) {                       \
                        ag16[i] = __half2float(As[0][rw][ks + i]);            \
                        au16[i] = __half2float(As[1][rw][ks + i]);            \
                    }                                                         \
                    for (uint32_t nt = 0; nt * 16 < tn; nt++) {               \
                        wmma_a16 b16;                                         \
                        for (uint32_t i = 0; i < 16; i++)                     \
                            b16[i] = __half2float(                            \
                                Xs[ks + i][nt * 16 + (l & 15)]);              \
                        ag[nt] =                                              \
                            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(       \
                                ag16, b16, ag[nt]);                           \
                        au[nt] =                                              \
                            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(       \
                                au16, b16, au[nt]);                           \
                    }                                                         \
                }                                                             \
                __syncthreads();                                              \
            }                                                                 \
            for (uint32_t nt = 0; nt * 16 < tn; nt++)                         \
                for (uint32_t i = 0; i < 8; i++) {                            \
                    uint32_t r = r0 + wave * 16 + 2 * i + (l >> 4);           \
                    uint32_t tt = nt * 16 + (l & 15);                         \
                    if (r < nrows && tt < tn) {                               \
                        float g = ag[nt][i], u = au[nt][i];                   \
                        out[(size_t)(a0 + t0 + tt) * nrows + r] =             \
                            g / (1.f + expf(-g)) * u;                         \
                    }                                                         \
                }                                                             \
        }                                                                     \
    }

/* Single tensor (down): plain GEMM, out[a0+t][row] = W . x[tok]. */
#define Q4_MOEW(name, deq_elem, deq32, BLK, BBYTES)                                  \
    __global__ static void name(float *out, const uint8_t *wbase,             \
            uint64_t stride, uint64_t woff,                                   \
            const int32_t *slots, const uint32_t *eb, const uint32_t *ec,     \
            const int32_t *atok, const float *x, uint32_t nrows,              \
            uint32_t ncols) {                                                \
        const uint32_t r0 = blockIdx.x * MOEW_ROWS, e = blockIdx.y;           \
        const uint32_t K = ec[e], a0 = eb[e];                                 \
        if (!K) return;                                                      \
        __shared__ __half As[MOEW_ROWS][MOEW_KC + MOEW_PAD];                  \
        __shared__ __half Xs[MOEW_KC][MOEW_TOKS + MOEW_PAD];                  \
        const uint8_t *wp = wbase + (size_t)slots[e] * stride + woff;         \
        const uint32_t nb = ncols / (BLK);                                    \
        const int wave = threadIdx.x >> 5, l = threadIdx.x & 31;              \
        for (uint32_t t0 = 0; t0 < K; t0 += MOEW_TOKS) {                      \
            const uint32_t tn = K - t0 < MOEW_TOKS ? K - t0 : MOEW_TOKS;      \
            wmma_c8 acc[MOEW_TOKS / 16];                                      \
            for (uint32_t nt = 0; nt < MOEW_TOKS / 16; nt++)                  \
                acc[nt] = (wmma_c8){};                                        \
            for (uint32_t k0 = 0; k0 < ncols; k0 += MOEW_KC) {                \
                const uint32_t kn = ncols - k0 < MOEW_KC ? ncols - k0         \
                                                         : MOEW_KC;           \
                for (uint32_t i = threadIdx.x;                                \
                     i < MOEW_ROWS * (MOEW_KC / 32u); i += blockDim.x) {       \
                    const uint32_t r = i / (MOEW_KC / 32u);                   \
                    const uint32_t sb = i % (MOEW_KC / 32u);                  \
                    Q4_DEQ_TILE(As, wp, r0, r, k0, sb, kn, nrows, nb, BLK,    \
                                BBYTES, deq_elem, deq32);                     \
                }                                                             \
                for (uint32_t i = threadIdx.x; i < MOEW_KC * MOEW_TOKS;       \
                     i += blockDim.x) {                                       \
                    uint32_t tt = i / MOEW_KC, kc = i % MOEW_KC;              \
                    float v = 0.f;                                            \
                    if (kc < kn && tt < tn) {                                 \
                        int32_t tok =                                         \
                            atok ? atok[a0 + t0 + tt]                         \
                                 : (int32_t)(a0 + t0 + tt);                   \
                        v = x[(size_t)tok * ncols + k0 + kc];                 \
                    }                                                         \
                    Xs[kc][tt] = __float2half(v);                             \
                }                                                             \
                __syncthreads();                                              \
                for (uint32_t ks = 0; ks < kn; ks += 16) {                    \
                    wmma_a16 a16;                                             \
                    const uint32_t rw = wave * 16 + (l & 15);                 \
                    for (uint32_t i = 0; i < 16; i++)                         \
                        a16[i] = __half2float(As[rw][ks + i]);                \
                    for (uint32_t nt = 0; nt * 16 < tn; nt++) {               \
                        wmma_a16 b16;                                         \
                        for (uint32_t i = 0; i < 16; i++)                     \
                            b16[i] = __half2float(                            \
                                Xs[ks + i][nt * 16 + (l & 15)]);              \
                        acc[nt] =                                             \
                            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(       \
                                a16, b16, acc[nt]);                           \
                    }                                                         \
                }                                                             \
                __syncthreads();                                              \
            }                                                                 \
            for (uint32_t nt = 0; nt * 16 < tn; nt++)                         \
                for (uint32_t i = 0; i < 8; i++) {                            \
                    uint32_t r = r0 + wave * 16 + 2 * i + (l >> 4);           \
                    uint32_t tt = nt * 16 + (l & 15);                         \
                    if (r < nrows && tt < tn)                                 \
                        out[(size_t)(a0 + t0 + tt) * nrows + r] =             \
                            acc[nt][i];                                       \
                }                                                             \
        }                                                                     \
    }

/* Dense many-token GEMM on WMMA: same 64x64x64 tiling as the expert
 * kernels (128-token tiles regressed: 8 c8 accumulators spill). BLK/BBYTES
 * are the quant superblock's value/byte size. */
#define Q8W_TOKS 64u
#define Q4_DGEMW(name, BLK, BBYTES, deq_elem, deq32)                         \
__global__ static void name(float *out, const uint8_t *w, const float *x,   \
                                  uint32_t in_dim, uint32_t out_dim,        \
                                  uint32_t n_batch, float scale) {          \
    const uint32_t r0 = blockIdx.x * MOEW_ROWS, t0 = blockIdx.y * Q8W_TOKS; \
    __shared__ __half As[MOEW_ROWS][MOEW_KC + MOEW_PAD];                    \
    __shared__ __half Xs[MOEW_KC][Q8W_TOKS + MOEW_PAD];                     \
    const uint32_t nb = in_dim / (BLK);                                     \
    const int wave = threadIdx.x >> 5, l = threadIdx.x & 31;                \
    const uint32_t tn = n_batch - t0 < Q8W_TOKS ? n_batch - t0 : Q8W_TOKS;  \
    wmma_c8 acc[Q8W_TOKS / 16];                                             \
    for (uint32_t nt = 0; nt < Q8W_TOKS / 16; nt++) acc[nt] = (wmma_c8){};  \
    for (uint32_t k0 = 0; k0 < in_dim; k0 += MOEW_KC) {                     \
        const uint32_t kn = in_dim - k0 < MOEW_KC ? in_dim - k0 : MOEW_KC;  \
        for (uint32_t i = threadIdx.x; i < MOEW_ROWS * (MOEW_KC / 32u);     \
             i += blockDim.x) {                                             \
            const uint32_t r = i / (MOEW_KC / 32u);                         \
            const uint32_t sb = i % (MOEW_KC / 32u);                        \
            Q4_DEQ_TILE(As, w, r0, r, k0, sb, kn, out_dim, nb, BLK,         \
                        BBYTES, deq_elem, deq32);                           \
        }                                                                 \
        for (uint32_t i = threadIdx.x; i < MOEW_KC * Q8W_TOKS;              \
             i += blockDim.x) {                                             \
            uint32_t tt = i / MOEW_KC, kc = i % MOEW_KC;                    \
            float v = 0.f;                                                  \
            if (kc < kn && tt < tn)                                         \
                v = x[(size_t)(t0 + tt) * in_dim + k0 + kc];                \
            Xs[kc][tt] = __float2half(v);                                   \
        }                                                                 \
        __syncthreads();                                                    \
        for (uint32_t ks = 0; ks < kn; ks += 16) {                          \
            wmma_a16 a16;                                                   \
            const uint32_t rw = wave * 16 + (l & 15);                       \
            for (uint32_t i = 0; i < 16; i++)                               \
                a16[i] = __half2float(As[rw][ks + i]);                      \
            for (uint32_t nt = 0; nt * 16 < tn; nt++) {                     \
                wmma_a16 b16;                                               \
                for (uint32_t i = 0; i < 16; i++)                           \
                    b16[i] = __half2float(Xs[ks + i][nt * 16 + (l & 15)]);  \
                acc[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(       \
                    a16, b16, acc[nt]);                                     \
            }                                                             \
        }                                                                 \
        __syncthreads();                                                    \
    }                                                                     \
    for (uint32_t nt = 0; nt * 16 < tn; nt++)                               \
        for (uint32_t i = 0; i < 8; i++) {                                  \
            uint32_t r = r0 + wave * 16 + 2 * i + (l >> 4);                 \
            uint32_t tt = nt * 16 + (l & 15);                               \
            if (r < out_dim && tt < tn)                                     \
                out[(size_t)(t0 + tt) * out_dim + r] = acc[nt][i] * scale;  \
        }                                                                 \
}

Q4_DGEMW(gemm_q8w_k, 32u, 34u, deq_q8_0_elem, deq_q8_0_x32)
Q4_DGEMW(gemm_q6w_k, 256u, 210u, deq_q6_k_elem, deq_q6_k_x32)
Q4_DGEMW(gemm_q4kw_k, 256u, 144u, deq_q4k_elem, deq_q4k_x32)
Q4_DGEMW(gemm_q5kw_k, 256u, 176u, deq_q5k_elem, deq_q5k_x32)
Q4_DGEMW(gemm_q40w_k, 32u, 18u, deq_q40_elem, deq_q40_x32)
Q4_DGEMW(gemm_q50w_k, 32u, 22u, deq_q50_elem, deq_q50_x32)
Q4_DGEMW(gemm_q20w_k, 64u, 18u, deq_q20_elem, deq_q20_x32)
Q4_DGEMW(gemm_iq4xsw_k, 256u, 136u, deq_iq4xs_elem, deq_iq4xs_x32)


Q4_MOEW2(moew2_iq2_k, deq_iq2s_elem, deq_iq2s_x32, 256u, 82u)
Q4_MOEW2(moew2_iq3_k, deq_iq3s_elem, deq_iq3s_x32, 256u, 110u)
Q4_MOEW2(moew2_iq4_k, deq_iq4nl_elem, deq_iq4nl_x32, 32u, 18u)
Q4_MOEW2(moew2_iq3xxs_k, deq_iq3xxs_elem, deq_iq3xxs_x32, 256u, 98u)
Q4_MOEW2(moew2_iq4xs_k, deq_iq4xs_elem, deq_iq4xs_x32, 256u, 136u)
Q4_MOEW2(moew2_q40_k, deq_q40_elem, deq_q40_x32, 32u, 18u)
Q4_MOEW2(moew2_q50_k, deq_q50_elem, deq_q50_x32, 32u, 22u)
Q4_MOEW2(moew2_q20_k, deq_q20_elem, deq_q20_x32, 64u, 18u)
Q4_MOEW(moew_iq2_k, deq_iq2s_elem, deq_iq2s_x32, 256u, 82u)
Q4_MOEW(moew_iq3_k, deq_iq3s_elem, deq_iq3s_x32, 256u, 110u)
Q4_MOEW(moew_iq4_k, deq_iq4nl_elem, deq_iq4nl_x32, 32u, 18u)
Q4_MOEW(moew_iq3xxs_k, deq_iq3xxs_elem, deq_iq3xxs_x32, 256u, 98u)
Q4_MOEW(moew_iq4xs_k, deq_iq4xs_elem, deq_iq4xs_x32, 256u, 136u)
Q4_MOEW(moew_q40_k, deq_q40_elem, deq_q40_x32, 32u, 18u)
Q4_MOEW(moew_q50_k, deq_q50_elem, deq_q50_x32, 32u, 22u)
Q4_MOEW(moew_q20_k, deq_q20_elem, deq_q20_x32, 64u, 18u)

/* Q4_GEMV_PROF=1: per-type call/byte counters. Free (no events), works inside
 * graph capture too since it is pure host-side accounting.
 * Q4_GEMV_PROF=2: additionally bracket each launch with stream events and
 * accumulate per-type GPU intervals at report time. Events are recorded on
 * g_str so no host sync is needed; intervals include in-stream gaps. */
static uint64_t g_gemv_calls[64];
static uint64_t g_gemv_bytes[64];
static double g_gemv_ms[64];
#define Q4_GEV_MAX 200000
static hipEvent_t g_gev_a[Q4_GEV_MAX], g_gev_b[Q4_GEV_MAX];
static uint8_t g_gev_ty[Q4_GEV_MAX];
static uint32_t g_gev_n;
static int g_gemv_mode = -1;
static int gemv_mode(void) {
    if (g_gemv_mode < 0) {
        const char *e = getenv("Q4_GEMV_PROF");
        g_gemv_mode = e ? atoi(e) : 0;
    }
    return g_gemv_mode;
}
static void gev_rec(int which, uint32_t type) {
    /* During graph capture eventRecord would be an illegal host call that
     * leaves a sticky error; profiling only covers eager launches then. */
    if (g_gev_n >= Q4_GEV_MAX || g_capturing) return;
    uint32_t i = g_gev_n;
    if (which == 0) {
        if (!g_gev_a[i])
            if (hipEventCreate(&g_gev_a[i]) != hipSuccess) return;
        if (hipEventRecord(g_gev_a[i], g_str) != hipSuccess) return;
        g_gev_ty[i] = (uint8_t)type;
    } else {
        if (!g_gev_b[i])
            if (hipEventCreate(&g_gev_b[i]) != hipSuccess) return;
        if (hipEventRecord(g_gev_b[i], g_str) != hipSuccess) return;
        g_gev_n++;
    }
}
#define GEV_PRE(t) do { if (gemv_mode() >= 2) gev_rec(0, (t)); } while (0)
#define GEV_POST() do { if (gemv_mode() >= 2) gev_rec(1, 0); } while (0)

extern "C" void q4_hip_gemv_prof_reset(void) {
    memset(g_gemv_calls, 0, sizeof(g_gemv_calls));
    memset(g_gemv_bytes, 0, sizeof(g_gemv_bytes));
    memset(g_gemv_ms, 0, sizeof(g_gemv_ms));
    g_gev_n = 0;
}

extern "C" void q4_hip_gemv_prof_report(void) {
    const char *tn[64] = {0};
    tn[Q4_T_F32] = "f32"; tn[Q4_T_F16] = "f16"; tn[Q4_T_Q5_1] = "q5_1";
    tn[Q4_T_Q8_0] = "q8_0"; tn[Q4_T_Q4_K] = "q4_k"; tn[Q4_T_Q5_K] = "q5_k";
    tn[Q4_T_Q6_K] = "q6_k"; tn[Q4_T_IQ4_NL] = "iq4_nl";
    tn[Q4_T_IQ3_S] = "iq3_s"; tn[Q4_T_IQ2_S] = "iq2_s"; tn[Q4_T_BF16] = "bf16";
    tn[Q4_T_IQ3_XXS] = "iq3_xxs"; tn[Q4_T_IQ4_XS] = "iq4_xs";
    tn[Q4_T_Q4_0] = "q4_0"; tn[Q4_T_Q5_0] = "q5_0"; tn[Q4_T_Q2_0] = "q2_0";
    /* Resolve recorded event pairs into per-type GPU ms (drain-free: events
     * already complete by report time). */
    if (g_gev_n) {
        (void)hipDeviceSynchronize();
        for (uint32_t i = 0; i < g_gev_n; i++) {
            float ms = 0.f;
            if (hipEventElapsedTime(&ms, g_gev_a[i], g_gev_b[i]) == hipSuccess)
                g_gemv_ms[g_gev_ty[i]] += ms;
        }
    }
    for (uint32_t t = 0; t < 64; t++) {
        if (!g_gemv_calls[t]) continue;
        const char *nm = tn[t] ? tn[t] : "?";
        fprintf(stderr, "gemv-prof %-8s calls %llu  bytes %.2f GiB  gpu %.0f ms\n",
                nm,
                (unsigned long long)g_gemv_calls[t],
                (double)g_gemv_bytes[t] / (1 << 30), g_gemv_ms[t]);
    }
}

extern "C" bool q4_hip_moe_pf2(uint32_t type, const uint8_t *wbase,
                               uint64_t stride, uint64_t woff0, uint64_t woff1,
                               const int32_t *d_slots, const uint32_t *d_eb,
                               const uint32_t *d_ec, const int32_t *d_atok,
                               const float *d_x, float *d_out0, float *d_out1,
                               uint32_t nrows, uint32_t ncols, uint32_t ne) {
    if (!g_ok || !wbase || !d_slots || !d_eb || !d_ec || !d_x || !d_out0 ||
        !d_out1 || !ne || !nrows || !ncols || ncols > 4096u ||
        ncols * 2u * sizeof(float) > 48u * 1024u)
        return false;
    size_t shmem = (size_t)ncols * 2u * sizeof(float);
    dim3 grid(nrows, ne);
    if (gemv_mode() > 0 && type < 64) {
        uint64_t rb = (uint64_t)q4_row_bytes(type, ncols);
        g_gemv_calls[type] += 2 * ne;
        g_gemv_bytes[type] += rb * nrows * 2 * ne;
    }
    GEV_PRE(type);
    switch (type) {
    case Q4_T_IQ2_S:
        if (ncols % 256u) return false;
        moepf2_iq2_k<<<grid, 256, shmem, g_str>>>(d_out0, d_out1, wbase, stride,
                                                woff0, woff1, d_slots, d_eb,
                                                d_ec, d_atok, d_x, nrows, ncols,
                                                1.f);
        break;
    case Q4_T_IQ3_S:
        if (ncols % 256u) return false;
        moepf2_iq3_k<<<grid, 256, shmem, g_str>>>(d_out0, d_out1, wbase, stride,
                                                woff0, woff1, d_slots, d_eb,
                                                d_ec, d_atok, d_x, nrows, ncols,
                                                1.f);
        break;
    case Q4_T_IQ3_XXS:
        if (ncols % 256u) return false;
        moepf2_iq3xxs_k<<<grid, 256, shmem, g_str>>>(d_out0, d_out1, wbase,
                                                    stride, woff0, woff1,
                                                    d_slots, d_eb, d_ec,
                                                    d_atok, d_x, nrows, ncols,
                                                    1.f);
        break;
    case Q4_T_IQ4_XS:
        if (ncols % 256u) return false;
        moepf2_iq4xs_k<<<grid, 256, shmem, g_str>>>(d_out0, d_out1, wbase,
                                                   stride, woff0, woff1,
                                                   d_slots, d_eb, d_ec,
                                                   d_atok, d_x, nrows, ncols,
                                                   1.f);
        break;
    case Q4_T_IQ4_NL:
        if (ncols % 32u) return false;
        moepf2_iq4_k<<<grid, 256, shmem, g_str>>>(d_out0, d_out1, wbase, stride,
                                                woff0, woff1, d_slots, d_eb,
                                                d_ec, d_atok, d_x, nrows, ncols,
                                                1.f);
        break;
    case Q4_T_Q4_0:
        if (ncols % 32u) return false;
        moepf2_q40_k<<<grid, 256, shmem, g_str>>>(d_out0, d_out1, wbase, stride,
                                                woff0, woff1, d_slots, d_eb,
                                                d_ec, d_atok, d_x, nrows, ncols,
                                                1.f);
        break;
    case Q4_T_Q5_0:
        if (ncols % 32u) return false;
        moepf2_q50_k<<<grid, 256, shmem, g_str>>>(d_out0, d_out1, wbase, stride,
                                                woff0, woff1, d_slots, d_eb,
                                                d_ec, d_atok, d_x, nrows, ncols,
                                                1.f);
        break;
    case Q4_T_Q2_0:
        if (ncols % 64u) return false;
        moepf2_q20_k<<<grid, 256, shmem, g_str>>>(d_out0, d_out1, wbase, stride,
                                                woff0, woff1, d_slots, d_eb,
                                                d_ec, d_atok, d_x, nrows, ncols,
                                                1.f);
        break;
    default:
        return false;
    }
    GEV_POST();
    return hipGetLastError() == hipSuccess;
}

extern "C" bool q4_hip_moe_pf(uint32_t type, const uint8_t *wbase,
                              uint64_t stride, uint64_t woff,
                              const int32_t *d_slots, const uint32_t *d_eb,
                              const uint32_t *d_ec, const int32_t *d_atok,
                              const float *d_x, float *d_out, uint32_t nrows,
                              uint32_t ncols, uint32_t ne) {
    if (!g_ok || !wbase || !d_slots || !d_eb || !d_ec || !d_x || !d_out ||
        !ne || !nrows || !ncols || ncols > 4096u ||
        ncols * 2u * sizeof(float) > 48u * 1024u)
        return false;
    size_t shmem = (size_t)ncols * 2u * sizeof(float);
    dim3 grid((nrows + 1u) / 2u, ne);
    if (gemv_mode() > 0 && type < 64) {
        uint64_t rb = (uint64_t)q4_row_bytes(type, ncols);
        g_gemv_calls[type] += ne;
        g_gemv_bytes[type] += rb * nrows * ne;
    }
    GEV_PRE(type);
    switch (type) {
    case Q4_T_IQ2_S:
        if (ncols % 256u) return false;
        moepf_iq2_k<<<grid, 256, shmem, g_str>>>(d_out, wbase, stride, woff,
                                               d_slots, d_eb, d_ec, d_atok,
                                               d_x, nrows, ncols, 1.f);
        break;
    case Q4_T_IQ3_S:
        if (ncols % 256u) return false;
        moepf_iq3_k<<<grid, 256, shmem, g_str>>>(d_out, wbase, stride, woff,
                                               d_slots, d_eb, d_ec, d_atok,
                                               d_x, nrows, ncols, 1.f);
        break;
    case Q4_T_IQ3_XXS:
        if (ncols % 256u) return false;
        moepf_iq3xxs_k<<<grid, 256, shmem, g_str>>>(d_out, wbase, stride, woff,
                                                   d_slots, d_eb, d_ec,
                                                   d_atok, d_x, nrows, ncols,
                                                   1.f);
        break;
    case Q4_T_IQ4_XS:
        if (ncols % 256u) return false;
        moepf_iq4xs_k<<<grid, 256, shmem, g_str>>>(d_out, wbase, stride, woff,
                                                  d_slots, d_eb, d_ec,
                                                  d_atok, d_x, nrows, ncols,
                                                  1.f);
        break;
    case Q4_T_IQ4_NL:
        if (ncols % 32u) return false;
        moepf_iq4_k<<<grid, 256, shmem, g_str>>>(d_out, wbase, stride, woff,
                                               d_slots, d_eb, d_ec, d_atok,
                                               d_x, nrows, ncols, 1.f);
        break;
    case Q4_T_Q4_0:
        if (ncols % 32u) return false;
        moepf_q40_k<<<grid, 256, shmem, g_str>>>(d_out, wbase, stride, woff,
                                               d_slots, d_eb, d_ec, d_atok,
                                               d_x, nrows, ncols, 1.f);
        break;
    case Q4_T_Q5_0:
        if (ncols % 32u) return false;
        moepf_q50_k<<<grid, 256, shmem, g_str>>>(d_out, wbase, stride, woff,
                                               d_slots, d_eb, d_ec, d_atok,
                                               d_x, nrows, ncols, 1.f);
        break;
    case Q4_T_Q2_0:
        if (ncols % 64u) return false;
        moepf_q20_k<<<grid, 256, shmem, g_str>>>(d_out, wbase, stride, woff,
                                               d_slots, d_eb, d_ec, d_atok,
                                               d_x, nrows, ncols, 1.f);
        break;
    default:
        return false;
    }
    GEV_POST();
    return hipGetLastError() == hipSuccess;
}


static int wmma_off(void) {
    static int v = -1;
    if (v < 0)
        v = getenv("Q4_WMMA") && getenv("Q4_WMMA")[0] == '0';
    return v;
}

/* Fused gate+up+silu WMMA prefill path; returns false for unsupported types
 * or when Q4_WMMA=0 so the caller falls back to q4_hip_moe_pf2 + moe_act. */
extern "C" bool q4_hip_moe_wmma2(uint32_t type, const uint8_t *wbase,
                                 uint64_t stride, uint64_t woff0,
                                 uint64_t woff1, const int32_t *d_slots,
                                 const uint32_t *d_eb, const uint32_t *d_ec,
                                 const int32_t *d_atok, const float *d_x,
                                 float *d_out, uint32_t nrows, uint32_t ncols,
                                 uint32_t ne) {
    if (!g_ok || wmma_off() || !wbase || !d_slots || !d_eb || !d_ec || !d_x ||
        !d_out || !ne || !nrows || !ncols)
        return false;
    dim3 grid((nrows + MOEW_ROWS - 1) / MOEW_ROWS, ne);
    if (gemv_mode() > 0 && type < 64) {
        uint64_t rb = (uint64_t)q4_row_bytes(type, ncols);
        g_gemv_calls[type] += 2 * ne;
        g_gemv_bytes[type] += rb * nrows * 2 * ne;
    }
    GEV_PRE(type);
    switch (type) {
    case Q4_T_IQ2_S:
        if (ncols % 256u) return false;
        moew2_iq2_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff0,
                                             woff1, d_slots, d_eb, d_ec,
                                             d_atok, d_x, nrows, ncols);
        break;
    case Q4_T_IQ3_S:
        if (ncols % 256u) return false;
        moew2_iq3_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff0,
                                             woff1, d_slots, d_eb, d_ec,
                                             d_atok, d_x, nrows, ncols);
        break;
    case Q4_T_IQ3_XXS:
        if (ncols % 256u) return false;
        moew2_iq3xxs_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff0,
                                                woff1, d_slots, d_eb, d_ec,
                                                d_atok, d_x, nrows, ncols);
        break;
    case Q4_T_IQ4_XS:
        if (ncols % 256u) return false;
        moew2_iq4xs_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff0,
                                               woff1, d_slots, d_eb, d_ec,
                                               d_atok, d_x, nrows, ncols);
        break;
    case Q4_T_IQ4_NL:
        if (ncols % 32u) return false;
        moew2_iq4_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff0,
                                             woff1, d_slots, d_eb, d_ec,
                                             d_atok, d_x, nrows, ncols);
        break;
    case Q4_T_Q4_0:
        if (ncols % 32u) return false;
        moew2_q40_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff0,
                                            woff1, d_slots, d_eb, d_ec,
                                            d_atok, d_x, nrows, ncols);
        break;
    case Q4_T_Q5_0:
        if (ncols % 32u) return false;
        moew2_q50_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff0,
                                            woff1, d_slots, d_eb, d_ec,
                                            d_atok, d_x, nrows, ncols);
        break;
    case Q4_T_Q2_0:
        if (ncols % 64u) return false;
        moew2_q20_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff0,
                                            woff1, d_slots, d_eb, d_ec,
                                            d_atok, d_x, nrows, ncols);
        break;
    default:
        return false;
    }
    GEV_POST();
    return hipGetLastError() == hipSuccess;
}

extern "C" bool q4_hip_moe_wmma(uint32_t type, const uint8_t *wbase,
                                uint64_t stride, uint64_t woff,
                                const int32_t *d_slots, const uint32_t *d_eb,
                                const uint32_t *d_ec, const int32_t *d_atok,
                                const float *d_x, float *d_out,
                                uint32_t nrows, uint32_t ncols, uint32_t ne) {
    if (!g_ok || wmma_off() || !wbase || !d_slots || !d_eb || !d_ec || !d_x ||
        !d_out || !ne || !nrows || !ncols)
        return false;
    dim3 grid((nrows + MOEW_ROWS - 1) / MOEW_ROWS, ne);
    if (gemv_mode() > 0 && type < 64) {
        uint64_t rb = (uint64_t)q4_row_bytes(type, ncols);
        g_gemv_calls[type] += ne;
        g_gemv_bytes[type] += rb * nrows * ne;
    }
    GEV_PRE(type);
    switch (type) {
    case Q4_T_IQ2_S:
        if (ncols % 256u) return false;
        moew_iq2_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff,
                                            d_slots, d_eb, d_ec, d_atok, d_x,
                                            nrows, ncols);
        break;
    case Q4_T_IQ3_S:
        if (ncols % 256u) return false;
        moew_iq3_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff,
                                            d_slots, d_eb, d_ec, d_atok, d_x,
                                            nrows, ncols);
        break;
    case Q4_T_IQ3_XXS:
        if (ncols % 256u) return false;
        moew_iq3xxs_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff,
                                               d_slots, d_eb, d_ec, d_atok,
                                               d_x, nrows, ncols);
        break;
    case Q4_T_IQ4_XS:
        if (ncols % 256u) return false;
        moew_iq4xs_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff,
                                              d_slots, d_eb, d_ec, d_atok,
                                              d_x, nrows, ncols);
        break;
    case Q4_T_IQ4_NL:
        if (ncols % 32u) return false;
        moew_iq4_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff,
                                            d_slots, d_eb, d_ec, d_atok, d_x,
                                            nrows, ncols);
        break;
    case Q4_T_Q4_0:
        if (ncols % 32u) return false;
        moew_q40_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff,
                                            d_slots, d_eb, d_ec, d_atok, d_x,
                                            nrows, ncols);
        break;
    case Q4_T_Q5_0:
        if (ncols % 32u) return false;
        moew_q50_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff,
                                            d_slots, d_eb, d_ec, d_atok, d_x,
                                            nrows, ncols);
        break;
    case Q4_T_Q2_0:
        if (ncols % 64u) return false;
        moew_q20_k<<<grid, 128, 0, g_str>>>(d_out, wbase, stride, woff,
                                           d_slots, d_eb, d_ec, d_atok, d_x,
                                           nrows, ncols);
        break;
    default:
        return false;
    }
    GEV_POST();
    return hipGetLastError() == hipSuccess;
}

/* silu(g)*u in place on g, over n contiguous floats — same math as
 * q4_hip_silu + q4_hip_mul but one launch. */
extern "C" bool q4_hip_moe_act(float *d_g, const float *d_u, uint64_t n) {
    if (!g_ok || !d_g || !d_u || !n) return false;
    moe_act_k<<<dim3((uint32_t)((n + 255) / 256)), 256, 0, g_str>>>(d_g, d_u,
                                                                  n);
    return hipGetLastError() == hipSuccess;
}

/* ---- device argmax + max-prob (temp-0 draft sampling) ------------------
 * Two kernels, wave32-safe (no wave64 assumptions): block partials
 * (max, first-index argmax, sum exp(x-blockmax)) then a one-block combine.
 * Tie-break everywhere: the smaller element index wins, matching the host
 * argmax() walk exactly. */

#define AMX_MAXB 1024u
typedef struct {
    float mx;
    uint32_t am;
    float se;
} amx_part_t;
static amx_part_t *g_amx_part; /* device */
static uint32_t *g_amx_h, *g_amx_d; /* mapped host [0]=id [1]=f32-bits prob */

static bool amx_bufs(void) {
    if (g_amx_part) return true;
    if (hipMalloc((void **)&g_amx_part, AMX_MAXB * sizeof(amx_part_t)) !=
        hipSuccess)
        return false;
    if (hipHostMalloc((void **)&g_amx_h, 2 * sizeof(uint32_t),
                      hipHostMallocMapped) != hipSuccess ||
        hipHostGetDevicePointer((void **)&g_amx_d, g_amx_h, 0) !=
            hipSuccess) {
        (void)hipFree(g_amx_part);
        g_amx_part = nullptr;
        return false;
    }
    return true;
}

/* Pairwise (max, first-index) combine used by both reduction stages. */
__device__ static inline void amx_pick(float vm, uint32_t va, float *mx,
                                       uint32_t *am) {
    if (vm > *mx || (vm == *mx && va < *am)) {
        *mx = vm;
        *am = va;
    }
}

__global__ static void amx_part_k(const float *x, uint32_t n, amx_part_t *p) {
    const uint32_t lane = threadIdx.x & 31u, w = threadIdx.x >> 5;
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stp = gridDim.x * blockDim.x;
    float mx = -INFINITY;
    uint32_t am = 0xffffffffu;
    for (uint32_t i = tid; i < n; i += stp) {
        const float v = x[i];
        amx_pick(v, i, &mx, &am);
    }
    for (uint32_t o = 16; o; o >>= 1)
        amx_pick(__shfl_down(mx, o), __shfl_down(am, o), &mx, &am);
    __shared__ float sm[8];
    __shared__ uint32_t sa[8];
    __shared__ float s_bmx;
    if (lane == 0) {
        sm[w] = mx;
        sa[w] = am;
    }
    __syncthreads();
    if (w == 0) {
        mx = lane < 8 ? sm[lane] : -INFINITY;
        am = lane < 8 ? sa[lane] : 0xffffffffu;
        for (uint32_t o = 4; o; o >>= 1)
            amx_pick(__shfl_down(mx, o), __shfl_down(am, o), &mx, &am);
        if (lane == 0) {
            s_bmx = mx;
            p[blockIdx.x].am = am;
        }
    }
    __syncthreads();
    const float bmx = s_bmx;
    float se = 0.f;
    for (uint32_t i = tid; i < n; i += stp) {
        const float d = x[i] - bmx;
        se += (d == d) ? __expf(d) : 0.f; /* -inf - -inf = NaN */
    }
    for (uint32_t o = 16; o; o >>= 1) se += __shfl_down(se, o);
    __shared__ float ss[8];
    if (lane == 0) ss[w] = se;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) t += ss[i];
        p[blockIdx.x].mx = bmx;
        p[blockIdx.x].se = t;
    }
}

__global__ static void amx_comb_k(const amx_part_t *p, uint32_t nb,
                                  uint32_t *out) {
    const uint32_t lane = threadIdx.x & 31u, w = threadIdx.x >> 5;
    float mx = -INFINITY;
    uint32_t am = 0xffffffffu;
    for (uint32_t i = threadIdx.x; i < nb; i += blockDim.x)
        amx_pick(p[i].mx, p[i].am, &mx, &am);
    for (uint32_t o = 16; o; o >>= 1)
        amx_pick(__shfl_down(mx, o), __shfl_down(am, o), &mx, &am);
    __shared__ float sm[8];
    __shared__ uint32_t sa[8];
    __shared__ float s_gmx;
    __shared__ uint32_t s_gam;
    if (lane == 0) {
        sm[w] = mx;
        sa[w] = am;
    }
    __syncthreads();
    if (w == 0) {
        mx = lane < 8 ? sm[lane] : -INFINITY;
        am = lane < 8 ? sa[lane] : 0xffffffffu;
        for (uint32_t o = 4; o; o >>= 1)
            amx_pick(__shfl_down(mx, o), __shfl_down(am, o), &mx, &am);
        if (lane == 0) {
            s_gmx = mx;
            s_gam = am;
        }
    }
    __syncthreads();
    const float gmx = s_gmx;
    float se = 0.f;
    for (uint32_t i = threadIdx.x; i < nb; i += blockDim.x) {
        const float w = __expf(p[i].mx - gmx);
        se += (w == w) ? w * p[i].se : 0.f;
    }
    for (uint32_t o = 16; o; o >>= 1) se += __shfl_down(se, o);
    __shared__ float ss[8];
    if (lane == 0) ss[w] = se;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) t += ss[i];
        out[0] = s_gam;
        float pr = (t > 0.f) ? 1.f / t : 0.f;
        memcpy(out + 1, &pr, 4);
    }
}

/* Device argmax of d_x[0..n) plus the softmax probability of the max
 * (p_max = 1 / sum_i exp(x_i - max)).  D2H's 8 bytes instead of n*4. */
extern "C" bool q4_hip_argmax_prob(const float *d_x, uint32_t n, int32_t *h_id,
                                   float *h_p) {
    if (!g_ok || !d_x || !n || !h_id || !h_p) return false;
    if (!amx_bufs()) return false;
    uint32_t nb = (n + 2047u) / 2048u;
    if (nb > AMX_MAXB) nb = AMX_MAXB;
    amx_part_k<<<nb, 256, 0, g_str>>>(d_x, n, g_amx_part);
    amx_comb_k<<<1, 256, 0, g_str>>>(g_amx_part, nb, g_amx_d);
    if (hipGetLastError() != hipSuccess) return false;
    if (hipStreamSynchronize(g_str) != hipSuccess) return false;
    *h_id = (int32_t)g_amx_h[0];
    float p;
    memcpy(&p, g_amx_h + 1, 4);
    *h_p = p;
    return true;
}

__device__ static inline void scale_min_k4(int j, const uint8_t *q, uint8_t *d,
                                           uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

__global__ static void gemv_q4_k_k(float *out, const uint8_t *w, const float *x,
                                   uint32_t in_dim, uint32_t out_dim, float scale,
                                   uint32_t n_batch, const int32_t *which) {
    uint32_t row = blockIdx.x;
    if (row >= out_dim) return;
    uint32_t nb = in_dim / 256u;
    const uint8_t *wr = w + (size_t)row * (size_t)nb * 144ull;
    __shared__ float sh[64];
    for (uint32_t tok = blockIdx.y; tok < n_batch; tok += gridDim.y) {
        const float *xt = x + (size_t)(which ? which[tok] : (int32_t)tok) * in_dim;
        float acc = 0.f;
        for (uint32_t i = threadIdx.x; i < nb; i += blockDim.x) {
            const uint8_t *blk = wr + i * 144u;
            float d = f16d(blk);
            float minv = f16d(blk + 2);
            const uint8_t *sc = blk + 4;
            const uint8_t *q = blk + 16;
            const float *xr = xt + i * 256u;
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t s1, m1, s2, m2;
                scale_min_k4(is + 0, sc, &s1, &m1);
                scale_min_k4(is + 1, sc, &s2, &m2);
                float d1 = d * s1, mm1 = minv * m1;
                float d2 = d * s2, mm2 = minv * m2;
                for (int l = 0; l < 32; l++)
                    acc += (d1 * (q[l] & 0xF) - mm1) * xr[j + l];
                for (int l = 0; l < 32; l++)
                    acc += (d2 * (q[l] >> 4) - mm2) * xr[j + 32 + l];
                q += 32;
                is += 2;
            }
        }
        sh[threadIdx.x] = acc;
        __syncthreads();
        for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            out[(size_t)tok * out_dim + row] = sh[0] * scale;
        __syncthreads();
    }
}

__global__ static void gemv_q5_1_k(float *out, const uint8_t *w, const float *x,
                                   uint32_t in_dim, uint32_t out_dim, float scale,
                                   uint32_t n_batch, const int32_t *which) {
    uint32_t row = blockIdx.x;
    if (row >= out_dim) return;
    uint32_t nb = in_dim / 32u;
    const uint8_t *wr = w + (size_t)row * (size_t)nb * 24ull;
    __shared__ float sh[64];
    for (uint32_t tok = blockIdx.y; tok < n_batch; tok += gridDim.y) {
        const float *xt = x + (size_t)(which ? which[tok] : (int32_t)tok) * in_dim;
        float acc = 0.f;
        for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) {
            const uint8_t *blk = wr + (size_t)b * 24u;
            float d = f16d(blk);
            float m = f16d(blk + 2);
            uint32_t qh;
            memcpy(&qh, blk + 4, 4);
            const uint8_t *qs = blk + 8;
            const float *xr = xt + b * 32u;
            for (int j = 0; j < 16; j++) {
                uint8_t xh0 = (uint8_t)(((qh >> (j + 0)) << 4) & 0x10);
                uint8_t xh1 = (uint8_t)((qh >> (j + 12)) & 0x10);
                int x0 = (qs[j] & 0x0F) | xh0;
                int x1 = (qs[j] >> 4) | xh1;
                acc += (x0 * d + m) * xr[j];
                acc += (x1 * d + m) * xr[j + 16];
            }
        }
        sh[threadIdx.x] = acc;
        __syncthreads();
        for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            out[(size_t)tok * out_dim + row] = sh[0] * scale;
        __syncthreads();
    }
}

__global__ static void gemv_q5_k_k(float *out, const uint8_t *w, const float *x,
                                   uint32_t in_dim, uint32_t out_dim, float scale,
                                   uint32_t n_batch, const int32_t *which) {
    uint32_t row = blockIdx.x;
    if (row >= out_dim) return;
    uint32_t nb = in_dim / 256u;
    const uint8_t *wr = w + (size_t)row * (size_t)nb * 176ull;
    __shared__ float sh[64];
    for (uint32_t tok = blockIdx.y; tok < n_batch; tok += gridDim.y) {
        const float *xt = x + (size_t)(which ? which[tok] : (int32_t)tok) * in_dim;
        float acc = 0.f;
        for (uint32_t i = threadIdx.x; i < nb; i += blockDim.x) {
            const uint8_t *blk = wr + i * 176u;
            float d = f16d(blk);
            float minv = f16d(blk + 2);
            const uint8_t *sc = blk + 4;
            const uint8_t *qh = blk + 16;
            const uint8_t *ql = blk + 48;
            const float *xr = xt + i * 256u;
            int is = 0;
            uint8_t u1 = 1, u2 = 2;
            for (int j = 0; j < 256; j += 64) {
                uint8_t s1, m1, s2, m2;
                scale_min_k4(is + 0, sc, &s1, &m1);
                scale_min_k4(is + 1, sc, &s2, &m2);
                float d1 = d * s1, mm1 = minv * m1;
                float d2 = d * s2, mm2 = minv * m2;
                for (int l = 0; l < 32; l++)
                    acc += (d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - mm1) *
                           xr[j + l];
                for (int l = 0; l < 32; l++)
                    acc += (d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - mm2) *
                           xr[j + 32 + l];
                ql += 32;
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        sh[threadIdx.x] = acc;
        __syncthreads();
        for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            out[(size_t)tok * out_dim + row] = sh[0] * scale;
        __syncthreads();
    }
}

/* Tiled Q8_0 x F32 GEMM for batched prefill projections.
 * W Q8_0 [nrows x ncols], X F32 [n_batch x ncols], Y F32 [n_batch x nrows].
 * Block: 256 threads, tile 32 rows x 32 tokens x 32 cols (one Q8_0 block). */
__global__ static void gemm_q8_0_k(float *out, const uint8_t *w, const float *x,
                                   uint32_t in_dim, uint32_t out_dim,
                                   uint32_t n_batch, float scale) {
    __shared__ float ws[32][33];
    __shared__ float xs[32][33];
    const uint32_t r0 = blockIdx.x * 32u;
    const uint32_t t0 = blockIdx.y * 32u;
    const uint32_t nb = in_dim / 32u;
    const uint32_t wr = threadIdx.x / 8u;      /* 0..31: row stripe */
    const uint32_t kk = (threadIdx.x % 8u) * 4u; /* 0,4,...,28 */
    const uint32_t warp = threadIdx.x / 32u;   /* 0..7 */
    const uint32_t lane = threadIdx.x % 32u;   /* 0..31 */
    /* each thread owns exactly 1 row x 4 tokens of the 32x32 tile */
    const uint32_t tr = warp * 4u + lane / 8u; /* 0..31 */
    const uint32_t tt = (lane % 8u) * 4u;      /* 0,4,...,28 */
    float acc[4] = {};
    for (uint32_t kb = 0; kb < nb; kb++) {
        /* load W tile: rows r0+0..31, q8 block kb */
        {
            uint32_t r = r0 + wr;
            float d = 0.f;
            int8_t qv[4] = {0, 0, 0, 0};
            if (r < out_dim) {
                const uint8_t *blk = w + ((size_t)r * nb + kb) * 34ull;
                __half dh;
                memcpy(&dh, blk, 2);
                d = __half2float(dh);
                memcpy(qv, blk + 2 + kk, 4);
            }
            ws[wr][kk + 0] = d * (float)qv[0];
            ws[wr][kk + 1] = d * (float)qv[1];
            ws[wr][kk + 2] = d * (float)qv[2];
            ws[wr][kk + 3] = d * (float)qv[3];
        }
        /* load X tile: toks t0+0..31, cols kb*32+0..31 */
        {
            uint32_t t = t0 + wr;
            float v[4] = {0.f, 0.f, 0.f, 0.f};
            if (t < n_batch) {
                const float *xr = x + (size_t)t * in_dim + kb * 32u + kk;
                v[0] = xr[0];
                v[1] = xr[1];
                v[2] = xr[2];
                v[3] = xr[3];
            }
            xs[wr][kk + 0] = v[0];
            xs[wr][kk + 1] = v[1];
            xs[wr][kk + 2] = v[2];
            xs[wr][kk + 3] = v[3];
        }
        __syncthreads();
#pragma unroll 8
        for (uint32_t k = 0; k < 32u; k++) {
            float wv = ws[tr][k];
#pragma unroll
            for (int j = 0; j < 4; j++) acc[j] += wv * xs[tt + j][k];
        }
        __syncthreads();
    }
    {
        uint32_t r = r0 + tr;
        if (r < out_dim) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                uint32_t t = t0 + tt + (uint32_t)j;
                if (t < n_batch) out[(size_t)t * out_dim + r] = acc[j] * scale;
            }
        }
    }
}

/* Split-K gemv for tiny out_dim: grid (out_dim, S), partials to scratch,
 * deterministic ascending sum in the second kernel. */
__global__ static void gemv_f32_sk(float *part, const float *w, const float *x,
                                   uint32_t in_dim, uint32_t S) {
    uint32_t row = blockIdx.x, s = blockIdx.y;
    uint32_t per = (in_dim + S - 1) / S;
    uint32_t i0 = s * per, i1 = i0 + per < in_dim ? i0 + per : in_dim;
    const float *wr = w + (size_t)row * in_dim;
    float acc = 0.f;
    for (uint32_t i = i0 + threadIdx.x; i < i1; i += blockDim.x)
        acc += wr[i] * x[i];
    __shared__ float sh[256];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t sp = blockDim.x >> 1; sp > 0; sp >>= 1) {
        if (threadIdx.x < sp) sh[threadIdx.x] += sh[threadIdx.x + sp];
        __syncthreads();
    }
    if (threadIdx.x == 0) part[(size_t)row * S + s] = sh[0];
}

__global__ static void gemv_sk_sum(float *out, const float *part,
                                   uint32_t out_dim, uint32_t S, float scale) {
    uint32_t row = blockIdx.x;
    if (row >= out_dim) return;
    float acc = 0.f;
    for (uint32_t s = 0; s < S; s++) acc += part[(size_t)row * S + s];
    out[row] = acc * scale;
}

#define GEMV_SK_MAXROWS 256u
#define GEMV_SK_S 64u

static bool launch(uint32_t type, const uint8_t *d_w, uint64_t nrows,
                   uint64_t ncols, const float *d_x, float *d_y, float scale,
                   uint32_t n_batch, const int32_t *which) {
    if (n_batch == 0) n_batch = 1;
    if (gemv_mode() > 0 && type < 64) {
        uint64_t rb = (uint64_t)q4_row_bytes(type, ncols);
        g_gemv_calls[type]++;
        g_gemv_bytes[type] += rb * nrows;
    }
    static int shapes = -1;
    if (shapes < 0) shapes = getenv("Q4_SHAPES") != NULL;
    if (shapes) {
        bool dup = false;
        for (int i = 0; i < g_shape_n; i++)
            if (g_shapes[i].t == type && g_shapes[i].r == nrows &&
                g_shapes[i].c == ncols && g_shapes[i].b == n_batch) {
                g_shapes[i].n++;
                dup = true;
            }
        if (!dup && g_shape_n < 256) {
            g_shapes[g_shape_n].t = type;
            g_shapes[g_shape_n].r = nrows;
            g_shapes[g_shape_n].c = ncols;
            g_shapes[g_shape_n].b = n_batch;
            g_shapes[g_shape_n].n = 1;
            g_shape_n++;
        }
    }
#define GEMV_DONE() return true
    /* Decode GEMV (single token, no which remap): warp-per-row kernels.  The
     * fp sum order differs from the 64-thread path — bounded reorder noise,
     * same class as the f32-sk path.  Q4_WARPGV=0 falls back to the old
     * kernels bit-identically. */
    if (n_batch == 1 && !which && wgv_on()) {
        const dim3 wg((uint32_t)((nrows + 7u) / 8u));
        switch (type) {
        case Q4_T_Q8_0:
            if (ncols % 32u) break;
            GEV_PRE(type);
            wgv_q8_0_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                              (uint32_t)nrows, scale, 0.f);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_IQ2_S:
            if (ncols % 256u) break;
            GEV_PRE(type);
            wgv_iq2_s_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                               (uint32_t)nrows, scale);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_IQ3_S:
            if (ncols % 256u) break;
            GEV_PRE(type);
            wgv_iq3_s_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                               (uint32_t)nrows, scale);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_IQ3_XXS:
            if (ncols % 256u) break;
            GEV_PRE(type);
            wgv_iq3xxs_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                                (uint32_t)nrows, scale);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_IQ4_XS:
            if (ncols % 256u) break;
            GEV_PRE(type);
            wgv_iq4xs_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                               (uint32_t)nrows, scale);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_IQ4_NL:
            if (ncols % 32u) break;
            GEV_PRE(type);
            wgv_iq4_nl_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                                (uint32_t)nrows, scale);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_Q4_0:
            if (ncols % 32u) break;
            GEV_PRE(type);
            wgv_q4_0_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                              (uint32_t)nrows, scale);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_Q5_0:
            if (ncols % 32u) break;
            GEV_PRE(type);
            wgv_q5_0_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                              (uint32_t)nrows, scale);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_Q2_0:
            if (ncols % 64u) break;
            GEV_PRE(type);
            wgv_q2_0_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                              (uint32_t)nrows, scale);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_F32:
            /* keep the split-K path for tiny outputs: it wins there */
            if ((nrows <= GEMV_SK_MAXROWS && nrows < 96u && ncols >= 1024 &&
                 g_sk_part) ||
                ncols % 4u)
                break;
            GEV_PRE(type);
            wgv_f32_k<<<wg, 256, 0, g_str>>>(d_y, (const float *)d_w, d_x,
                                             (uint32_t)ncols, (uint32_t)nrows,
                                             scale);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_BF16:
            GEV_PRE(type);
            wgv_bf16_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                              (uint32_t)nrows, scale);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_Q6_K:
            if (ncols % 256u) break;
            GEV_PRE(type);
            wgv_q6_k_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                              (uint32_t)nrows, scale);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_Q4_K:
            if (ncols % 256u) break;
            GEV_PRE(type);
            wgv_q4_k_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                              (uint32_t)nrows, scale);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_Q5_K:
            if (ncols % 256u) break;
            GEV_PRE(type);
            wgv_q5_k_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                              (uint32_t)nrows, scale);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        default:
            break;
        }
    }
    /* Small multi-token batches (MTP verify, short prefill): warp-per-row
     * with one accumulator per token, same per-token add order as wgv —
     * each row is bit-identical to the single-token call.  Q4_WGVN=0
     * restores the old paths. */
    if (n_batch >= 2 && n_batch <= WGVN_MAX && !which && wgv_on() &&
        wgvn_on()) {
        const dim3 wg((uint32_t)((nrows + 7u) / 8u));
        switch (type) {
        case Q4_T_Q8_0:
            if (ncols % 32u) break;
            GEV_PRE(type);
            wgvn_q8_0_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                               (uint32_t)nrows, scale,
                                               n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_IQ2_S:
            if (ncols % 256u) break;
            GEV_PRE(type);
            wgvn_iq2_s_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                                (uint32_t)nrows, scale,
                                                n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_IQ3_S:
            if (ncols % 256u) break;
            GEV_PRE(type);
            wgvn_iq3_s_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                                (uint32_t)nrows, scale,
                                                n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_IQ3_XXS:
            if (ncols % 256u) break;
            GEV_PRE(type);
            wgvn_iq3xxs_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                                 (uint32_t)nrows, scale,
                                                 n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_IQ4_XS:
            if (ncols % 256u) break;
            GEV_PRE(type);
            wgvn_iq4xs_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                                (uint32_t)nrows, scale,
                                                n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_IQ4_NL:
            if (ncols % 32u) break;
            GEV_PRE(type);
            wgvn_iq4_nl_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                                 (uint32_t)nrows, scale,
                                                 n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_Q4_0:
            if (ncols % 32u) break;
            GEV_PRE(type);
            wgvn_q4_0_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                               (uint32_t)nrows, scale,
                                               n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_Q5_0:
            if (ncols % 32u) break;
            GEV_PRE(type);
            wgvn_q5_0_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                               (uint32_t)nrows, scale,
                                               n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_Q2_0:
            if (ncols % 64u) break;
            GEV_PRE(type);
            wgvn_q2_0_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                               (uint32_t)nrows, scale,
                                               n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_F32:
            if (ncols % 4u) break;
            GEV_PRE(type);
            wgvn_f32_k<<<wg, 256, 0, g_str>>>(d_y, (const float *)d_w, d_x,
                                              (uint32_t)ncols, (uint32_t)nrows,
                                              scale, n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_BF16:
            GEV_PRE(type);
            wgvn_bf16_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                               (uint32_t)nrows, scale,
                                               n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        /* Q4_T_Q6_K intentionally absent: the inline q6 accumulation does
         * not compile bit-identically to wgv_q6_k_k inside the t loop —
         * falls through to the batched gemv_q6_k_k (the old path). */
        case Q4_T_Q4_K:
            if (ncols % 256u) break;
            GEV_PRE(type);
            wgvn_q4_k_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                               (uint32_t)nrows, scale,
                                               n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        case Q4_T_Q5_K:
            if (ncols % 256u) break;
            GEV_PRE(type);
            wgvn_q5_k_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                               (uint32_t)nrows, scale,
                                               n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        default:
            break;
        }
    }
    /* New-type dense batches: WMMA beats the reuse kernel once enough tokens
     * amortize the tile setup. which-gathers still take the reuse path. */
    if ((type == Q4_T_Q4_0 || type == Q4_T_Q5_0 || type == Q4_T_Q2_0 ||
         type == Q4_T_IQ4_XS) &&
        n_batch >= 16 && !wmma_off() && !which &&
        (type == Q4_T_IQ4_XS ? ncols % 256u == 0 : ncols % 64u == 0)) {
        GEV_PRE(type);
        dim3 g((uint32_t)((nrows + MOEW_ROWS - 1) / MOEW_ROWS),
               (n_batch + 63u) / 64u);
        if (type == Q4_T_Q4_0)
            gemm_q40w_k<<<g, 128, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                              (uint32_t)nrows, n_batch, scale);
        else if (type == Q4_T_Q5_0)
            gemm_q50w_k<<<g, 128, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                              (uint32_t)nrows, n_batch, scale);
        else if (type == Q4_T_IQ4_XS)
            gemm_iq4xsw_k<<<g, 128, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                                (uint32_t)nrows, n_batch,
                                                scale);
        else
            gemm_q20w_k<<<g, 128, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                              (uint32_t)nrows, n_batch, scale);
        GEV_POST();
        Q4_HIP_CHECK(hipGetLastError());
        return true;
    }
    /* Several tokens share an expert. Dequant the row once (element-parallel),
     * then warp-per-token dots — faster than the per-thread serial-dot GEMV
     * even for a single token. */
    if (ncols <= 4096u && ncols * sizeof(float) <= 48u * 1024u) {
        size_t shmem = (size_t)ncols * sizeof(float);
        if (type == Q4_T_IQ2_S && ncols % 256u == 0) {
            GEV_PRE(type);
            gemm_iq2_reuse<<<(uint32_t)nrows, 256, shmem, g_str>>>(
                d_w, d_x, which, d_y, (uint32_t)nrows, (uint32_t)ncols, scale,
                n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        }
        if (type == Q4_T_IQ3_S && ncols % 256u == 0) {
            GEV_PRE(type);
            gemm_iq3_reuse<<<(uint32_t)nrows, 256, shmem, g_str>>>(
                d_w, d_x, which, d_y, (uint32_t)nrows, (uint32_t)ncols, scale,
                n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        }
        if (type == Q4_T_IQ4_NL && ncols % 32u == 0) {
            GEV_PRE(type);
            gemm_iq4_reuse<<<(uint32_t)nrows, 256, shmem, g_str>>>(
                d_w, d_x, which, d_y, (uint32_t)nrows, (uint32_t)ncols, scale,
                n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        }
        if (type == Q4_T_IQ3_XXS && ncols % 256u == 0) {
            GEV_PRE(type);
            gemm_iq3xxs_reuse<<<(uint32_t)nrows, 256, shmem, g_str>>>(
                d_w, d_x, which, d_y, (uint32_t)nrows, (uint32_t)ncols, scale,
                n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        }
        if (type == Q4_T_IQ4_XS && ncols % 256u == 0) {
            GEV_PRE(type);
            gemm_iq4xs_reuse<<<(uint32_t)nrows, 256, shmem, g_str>>>(
                d_w, d_x, which, d_y, (uint32_t)nrows, (uint32_t)ncols, scale,
                n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        }
        if (type == Q4_T_Q4_0 && ncols % 32u == 0) {
            GEV_PRE(type);
            gemm_q40_reuse<<<(uint32_t)nrows, 256, shmem, g_str>>>(
                d_w, d_x, which, d_y, (uint32_t)nrows, (uint32_t)ncols, scale,
                n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        }
        if (type == Q4_T_Q5_0 && ncols % 32u == 0) {
            GEV_PRE(type);
            gemm_q50_reuse<<<(uint32_t)nrows, 256, shmem, g_str>>>(
                d_w, d_x, which, d_y, (uint32_t)nrows, (uint32_t)ncols, scale,
                n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        }
        if (type == Q4_T_Q2_0 && ncols % 64u == 0) {
            GEV_PRE(type);
            gemm_q20_reuse<<<(uint32_t)nrows, 256, shmem, g_str>>>(
                d_w, d_x, which, d_y, (uint32_t)nrows, (uint32_t)ncols, scale,
                n_batch);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            GEMV_DONE();
        }
    }
    if (type == Q4_T_Q8_0 && n_batch >= 16 && ncols % 32u == 0) {
        if (!wmma_off() && ncols % 16u == 0) {
            GEV_PRE(type);
            dim3 g((uint32_t)((nrows + MOEW_ROWS - 1) / MOEW_ROWS),
                   (n_batch + 63u) / 64u);
            gemm_q8w_k<<<g, 128, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                           (uint32_t)nrows, n_batch, scale);
            GEV_POST();
            Q4_HIP_CHECK(hipGetLastError());
            return true;
        }
        GEV_PRE(type);
        dim3 g((uint32_t)((nrows + 31u) / 32u), (n_batch + 31u) / 32u);
        gemm_q8_0_k<<<g, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols, (uint32_t)nrows,
                                n_batch, scale);
        GEV_POST();
        Q4_HIP_CHECK(hipGetLastError());
        return true;
    }
    if (type == Q4_T_Q6_K && n_batch >= 16 && ncols % 256u == 0 &&
        !wmma_off() && !which) {
        GEV_PRE(type);
        dim3 g((uint32_t)((nrows + MOEW_ROWS - 1) / MOEW_ROWS),
               (n_batch + 63u) / 64u);
        gemm_q6w_k<<<g, 128, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                         (uint32_t)nrows, n_batch, scale);
        GEV_POST();
        Q4_HIP_CHECK(hipGetLastError());
        return true;
    }
    if ((type == Q4_T_Q4_K || type == Q4_T_Q5_K) && n_batch >= 16 &&
        ncols % 256u == 0 && !wmma_off() && !which) {
        GEV_PRE(type);
        dim3 g((uint32_t)((nrows + MOEW_ROWS - 1) / MOEW_ROWS),
               (n_batch + 63u) / 64u);
        if (type == Q4_T_Q4_K)
            gemm_q4kw_k<<<g, 128, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                              (uint32_t)nrows, n_batch, scale);
        else
            gemm_q5kw_k<<<g, 128, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                              (uint32_t)nrows, n_batch, scale);
        GEV_POST();
        Q4_HIP_CHECK(hipGetLastError());
        return true;
    }
    uint32_t gy = n_batch < 64u ? n_batch : 64u;
    dim3 grid((uint32_t)nrows, gy);
    dim3 block(64);
    GEV_PRE(type);
    switch (type) {
    case Q4_T_Q8_0:
        gemv_q8_0_k<<<grid, block, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                     (uint32_t)nrows, scale, n_batch, which);
        break;
    case Q4_T_Q4_K:
        gemv_q4_k_k<<<grid, block, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                     (uint32_t)nrows, scale, n_batch, which);
        break;
    case Q4_T_Q5_1:
        gemv_q5_1_k<<<grid, block, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                     (uint32_t)nrows, scale, n_batch, which);
        break;
    case Q4_T_Q5_K:
        gemv_q5_k_k<<<grid, block, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                     (uint32_t)nrows, scale, n_batch, which);
        break;
    case Q4_T_Q6_K:
        if (ncols % 256u) return false;
        gemv_q6_k_k<<<grid, dim3(128), 0, g_str>>>(d_y, d_w, d_x,
                                     (uint32_t)ncols, (uint32_t)nrows, scale,
                                     n_batch, which);
        break;
    case Q4_T_IQ2_S:
        if (ncols % 256u) return false;
        gemv_iq2_s_k<<<grid, block, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                      (uint32_t)nrows, scale, n_batch, which);
        break;
    case Q4_T_IQ3_S:
        if (ncols % 256u) return false;
        gemv_iq3_s_k<<<grid, block, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                      (uint32_t)nrows, scale, n_batch, which);
        break;
    case Q4_T_IQ3_XXS:
        if (ncols % 256u) return false;
        gemv_iq3xxs_k<<<grid, block, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                       (uint32_t)nrows, scale, n_batch, which);
        break;
    case Q4_T_IQ4_XS:
        if (ncols % 256u) return false;
        gemv_iq4xs_k<<<grid, block, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                      (uint32_t)nrows, scale, n_batch, which);
        break;
    case Q4_T_IQ4_NL:
        if (ncols % 32u) return false;
        gemv_iq4_nl_k<<<grid, block, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                       (uint32_t)nrows, scale, n_batch, which);
        break;
    case Q4_T_Q4_0:
        if (ncols % 32u) return false;
        gemv_q4_0_k<<<grid, block, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                       (uint32_t)nrows, scale, n_batch, which);
        break;
    case Q4_T_Q5_0:
        if (ncols % 32u) return false;
        gemv_q5_0_k<<<grid, block, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                       (uint32_t)nrows, scale, n_batch, which);
        break;
    case Q4_T_Q2_0:
        if (ncols % 64u) return false;
        gemv_q2_0_k<<<grid, block, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                       (uint32_t)nrows, scale, n_batch, which);
        break;
    case Q4_T_F32:
        static int f32pf = -1;
        if (f32pf < 0)
            f32pf = !(getenv("Q4_F32_PF") && getenv("Q4_F32_PF")[0] == '0');
        if (n_batch > 1 && f32pf) {
            dim3 g32((unsigned)((nrows + 31u) / 32u),
                     (unsigned)((n_batch + 31u) / 32u));
            gemm_tiled_k<float><<<g32, 256, 0, g_str>>>(
                d_y, (const float *)d_w, d_x, (uint32_t)ncols,
                (uint32_t)nrows, scale, n_batch, which);
        } else if (n_batch == 1 && !which && nrows <= GEMV_SK_MAXROWS &&
                   nrows < 96u && ncols >= 1024 && g_sk_part) {
            dim3 gs((uint32_t)nrows, GEMV_SK_S);
            gemv_f32_sk<<<gs, 256, 0, g_str>>>(g_sk_part,
                                               (const float *)d_w, d_x,
                                               (uint32_t)ncols, GEMV_SK_S);
            gemv_sk_sum<<<(uint32_t)nrows, 32, 0, g_str>>>(d_y, g_sk_part,
                                                         (uint32_t)nrows,
                                                         GEMV_SK_S, scale);
        } else {
            gemv_f32_k<<<grid, block, 0, g_str>>>(d_y, (const float *)d_w, d_x,
                                        (uint32_t)ncols, (uint32_t)nrows, scale,
                                        n_batch, which);
        }
        break;
    case Q4_T_BF16:
        if (n_batch > 1) {
            dim3 g32((unsigned)((nrows + 31u) / 32u),
                     (unsigned)((n_batch + 31u) / 32u));
            gemm_tiled_k<uint16_t><<<g32, 256, 0, g_str>>>(
                d_y, (const uint16_t *)d_w, d_x, (uint32_t)ncols,
                (uint32_t)nrows, scale, n_batch, which);
        } else {
            gemv_bf16_k<<<grid, block, 0, g_str>>>(d_y, d_w, d_x,
                                                   (uint32_t)ncols,
                                                   (uint32_t)nrows, scale,
                                                   n_batch, which);
        }
        break;
    default:
        return false;
    }
    GEV_POST();
    Q4_HIP_CHECK(hipGetLastError());
    GEMV_DONE();
}

extern "C" bool q4_hip_gemv_dev(uint32_t ggml_type, const uint8_t *d_w,
                                uint64_t nrows, uint64_t ncols, const float *x,
                                float *y, float scale) {
    if (!g_ok || !d_w || !x || !y || nrows == 0 || ncols == 0) return false;
    if (!grow((void **)&g_dx, &g_dx_n, ncols * sizeof(float))) return false;
    if (!grow((void **)&g_dy, &g_dy_n, nrows * sizeof(float))) return false;
    if (!sync_copy(g_dx, x, ncols * sizeof(float), hipMemcpyHostToDevice))
        return false;
    if (!launch(ggml_type, d_w, nrows, ncols, g_dx, g_dy, scale, 1, NULL)) return false;
    return sync_copy(y, g_dy, nrows * sizeof(float), hipMemcpyDeviceToHost);
}

extern "C" bool q4_hip_gemv(uint32_t ggml_type, const uint8_t *w, uint64_t nrows,
                            uint64_t ncols, const float *x, float *y,
                            float scale) {
    if (!g_ok || !w) return false;
    uint64_t rb = q4_row_bytes(ggml_type, ncols);
    if (rb == 0) return false;
    size_t wn = (size_t)rb * (size_t)nrows;
    if (!grow((void **)&g_dw, &g_dw_n, wn)) return false;
    if (!sync_copy(g_dw, w, wn, hipMemcpyHostToDevice)) return false;
    return q4_hip_gemv_dev(ggml_type, g_dw, nrows, ncols, x, y, scale);
}

extern "C" bool q4_hip_gemv_dd(uint32_t ggml_type, const uint8_t *d_w,
                               uint64_t nrows, uint64_t ncols, const float *d_x,
                               float *d_y, float scale) {
    if (!g_ok || !d_w || !d_x || !d_y || nrows == 0 || ncols == 0) return false;
    return launch(ggml_type, d_w, nrows, ncols, d_x, d_y, scale, 1, NULL);
}

/* Warp GEMV with a fused epilogue: out = silu(dot * scale * act) when
 * act != 0 — absorbs the scale_silu elementwise pass between the two
 * hc_mix GEMVs.  Only the q8_0 warp path supports it; callers fall back
 * to gemv + scale_silu. */
extern "C" bool q4_hip_gemv_dd_act(uint32_t ggml_type, const uint8_t *d_w,
                                   uint64_t nrows, uint64_t ncols,
                                   const float *d_x, float *d_y, float scale,
                                   float act) {
    if (!g_ok || !d_w || !d_x || !d_y || nrows == 0 || ncols == 0) return false;
    if (!wgv_on() || act == 0.f) return false;
    if (ggml_type == Q4_T_Q8_0 && ncols % 32u == 0) {
        const dim3 wg((uint32_t)((nrows + 7u) / 8u));
        wgv_q8_0_k<<<wg, 256, 0, g_str>>>(d_y, d_w, d_x, (uint32_t)ncols,
                                          (uint32_t)nrows, scale, act);
        return hipGetLastError() == hipSuccess;
    }
    return false;
}

extern "C" bool q4_hip_gemv_dd_n(uint32_t ggml_type, const uint8_t *d_w,
                                 uint64_t nrows, uint64_t ncols, const float *d_x,
                                 float *d_y, float scale, uint32_t n_batch) {
    return q4_hip_gemv_dd_n_which(ggml_type, d_w, nrows, ncols, d_x, d_y, scale,
                                  n_batch, NULL);
}

extern "C" bool q4_hip_gemv_dd_n_which(uint32_t ggml_type, const uint8_t *d_w,
                                       uint64_t nrows, uint64_t ncols,
                                       const float *d_x, float *d_y, float scale,
                                       uint32_t n_batch, const int32_t *which) {
    if (!g_ok || !d_w || !d_x || !d_y || nrows == 0 || ncols == 0) return false;
    return launch(ggml_type, d_w, nrows, ncols, d_x, d_y, scale,
                  n_batch ? n_batch : 1, which);
}

__global__ static void unary_silu_k(float *d, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = d[i];
        d[i] = v / (1.f + expf(-v));
    }
}
__global__ static void unary_sig_k(float *d, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = 1.f / (1.f + expf(-d[i]));
}
__global__ static void unary_scale_k(float *d, float s, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] *= s;
}
__global__ static void mul_k(float *y, const float *a, const float *b, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] * b[i];
}
__global__ static void add_k(float *y, const float *a, const float *b, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}
__global__ static void axpy_k(float *y, const float *x, float a, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}
__global__ static void fill_k(float *d, float v, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = v;
}
__global__ static void copy_k(float *d, const float *s, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = s[i];
}

static bool launch1(uint64_t n, dim3 *g, dim3 *b) {
    *b = dim3(256);
    *g = dim3((uint32_t)((n + 255) / 256));
    return n > 0;
}

extern "C" bool q4_hip_silu(float *d, uint64_t n) {
    dim3 g, b;
    if (!g_ok || !d || !launch1(n, &g, &b)) return false;
    unary_silu_k<<<g, b, 0, g_str>>>(d, n);
    return hipGetLastError() == hipSuccess;
}
extern "C" bool q4_hip_sigmoid(float *d, uint64_t n) {
    dim3 g, b;
    if (!g_ok || !d || !launch1(n, &g, &b)) return false;
    unary_sig_k<<<g, b, 0, g_str>>>(d, n);
    return hipGetLastError() == hipSuccess;
}
extern "C" bool q4_hip_scale(float *d, float s, uint64_t n) {
    dim3 g, b;
    if (!g_ok || !d || !launch1(n, &g, &b)) return false;
    unary_scale_k<<<g, b, 0, g_str>>>(d, s, n);
    return hipGetLastError() == hipSuccess;
}
extern "C" bool q4_hip_mul(float *y, const float *a, const float *b, uint64_t n) {
    dim3 g, bl;
    if (!g_ok || !y || !a || !b || !launch1(n, &g, &bl)) return false;
    mul_k<<<g, bl, 0, g_str>>>(y, a, b, n);
    return hipGetLastError() == hipSuccess;
}
extern "C" bool q4_hip_add(float *y, const float *a, const float *b, uint64_t n) {
    dim3 g, bl;
    if (!g_ok || !y || !a || !b || !launch1(n, &g, &bl)) return false;
    add_k<<<g, bl, 0, g_str>>>(y, a, b, n);
    return hipGetLastError() == hipSuccess;
}
extern "C" bool q4_hip_axpy(float *y, const float *x, float a, uint64_t n) {
    dim3 g, b;
    if (!g_ok || !y || !x || !launch1(n, &g, &b)) return false;
    axpy_k<<<g, b, 0, g_str>>>(y, x, a, n);
    return hipGetLastError() == hipSuccess;
}
extern "C" bool q4_hip_fill(float *d, float v, uint64_t n) {
    if (!g_ok || !d || n == 0) return false;
    if (v == 0.f && !g_capturing && !g_str) {
        Q4_HIP_CHECK(hipMemset(d, 0, n * sizeof(float)));
        return true;
    }
    dim3 g, b;
    launch1(n, &g, &b);
    fill_k<<<g, b, 0, g_str>>>(d, v, n);
    return hipGetLastError() == hipSuccess;
}
extern "C" bool q4_hip_copy(float *d, const float *s, uint64_t n) {
    if (!g_ok || !d || !s || n == 0) return false;
    return sync_copy(d, s, n * sizeof(float), hipMemcpyDeviceToDevice);
}

__global__ static void grouped_rms_k(const float *x, const float *w, float *y,
                                     uint32_t n_embd, uint32_t hc, float eps) {
    uint32_t c = blockIdx.x;
    uint32_t tok = blockIdx.y;
    const float *xc = x + ((size_t)tok * hc + c) * n_embd;
    float *yc = y + ((size_t)tok * hc + c) * n_embd;
    const float *wc = w ? w + (size_t)c * n_embd : nullptr;
    float ss = 0.f;
    for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x)
        ss += xc[i] * xc[i];
    __shared__ float sh[256];
    sh[threadIdx.x] = ss;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float scale = rsqrtf(sh[0] / (float)n_embd + eps);
    for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x)
        yc[i] = xc[i] * scale * (wc ? wc[i] : 1.f);
}

extern "C" bool q4_hip_grouped_rms_n(const float *x, const float *w, float *y,
                                     uint32_t n_embd, uint32_t hc, float eps,
                                     uint32_t n_tok) {
    if (!g_ok || !x || !y || n_embd == 0 || hc == 0 || n_tok == 0) return false;
    grouped_rms_k<<<dim3(hc, n_tok), 256, 0, g_str>>>(x, w, y, n_embd, hc, eps);
    return hipGetLastError() == hipSuccess;
}
extern "C" bool q4_hip_grouped_rms(const float *x, const float *w, float *y,
                                   uint32_t n_embd, uint32_t hc, float eps) {
    return q4_hip_grouped_rms_n(x, w, y, n_embd, hc, eps, 1);
}

__global__ static void mean_hc_k(const float *gated, float *mixed, uint32_t n_embd,
                                 uint32_t hc) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t tok = blockIdx.y;
    if (i >= n_embd) return;
    const float *g = gated + (size_t)tok * hc * n_embd;
    float s = 0.f;
    for (uint32_t c = 0; c < hc; c++) s += g[(size_t)c * n_embd + i];
    mixed[(size_t)tok * n_embd + i] = s / (float)hc;
}
extern "C" bool q4_hip_mean_hc_n(const float *gated, float *mixed, uint32_t n_embd,
                                 uint32_t hc, uint32_t n_tok) {
    if (!g_ok || !gated || !mixed) return false;
    dim3 b(256), g((n_embd + 255) / 256, n_tok ? n_tok : 1);
    mean_hc_k<<<g, b, 0, g_str>>>(gated, mixed, n_embd, hc);
    return hipGetLastError() == hipSuccess;
}
extern "C" bool q4_hip_mean_hc(const float *gated, float *mixed, uint32_t n_embd,
                               uint32_t hc) {
    return q4_hip_mean_hc_n(gated, mixed, n_embd, hc, 1);
}

__global__ static void hc_combine_k(float *res, const float *block,
                                    const float *inject, uint32_t n_embd,
                                    uint32_t hc) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t tok = blockIdx.y;
    if (i >= n_embd) return;
    float inv = 1.f / (float)hc;
    float *rt = res + (size_t)tok * hc * n_embd;
    const float *bt = block + (size_t)tok * n_embd;
    const float *it = inject + (size_t)tok * hc;
    for (uint32_t c = 0; c < hc; c++) {
        float w = 2.f / (1.f + expf(-it[c] * inv));
        rt[(size_t)c * n_embd + i] += bt[i] * w;
    }
}
extern "C" bool q4_hip_hc_combine_n(float *res, const float *block,
                                    const float *inject, uint32_t n_embd,
                                    uint32_t hc, uint32_t n_tok) {
    if (!g_ok || !res || !block || !inject) return false;
    dim3 b(256), g((n_embd + 255) / 256, n_tok ? n_tok : 1);
    hc_combine_k<<<g, b, 0, g_str>>>(res, block, inject, n_embd, hc);
    return hipGetLastError() == hipSuccess;
}
extern "C" bool q4_hip_hc_combine(float *res, const float *block,
                                  const float *inject, uint32_t n_embd,
                                  uint32_t hc) {
    return q4_hip_hc_combine_n(res, block, inject, n_embd, hc, 1);
}

__global__ static void repeat_hc_k(const float *emb, float *res, uint32_t n_embd,
                                   uint32_t hc) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    float v = emb[i];
    for (uint32_t c = 0; c < hc; c++) res[(size_t)c * n_embd + i] = v;
}
extern "C" bool q4_hip_repeat_hc(const float *emb, float *res, uint32_t n_embd,
                                 uint32_t hc) {
    if (!g_ok || !emb || !res) return false;
    dim3 b(256), g((n_embd + 255) / 256);
    repeat_hc_k<<<g, b, 0, g_str>>>(emb, res, n_embd, hc);
    return hipGetLastError() == hipSuccess;
}

__global__ static void dot_k(const float *a, const float *b, float *out, uint32_t n) {
    float ss = 0.f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) ss += a[i] * b[i];
    __shared__ float sh[256];
    sh[threadIdx.x] = ss;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = sh[0];
}
extern "C" bool q4_hip_dot(const float *a, const float *b, float *out, uint32_t n) {
    if (!g_ok || !a || !b || !out) return false;
    dot_k<<<1, 256, 0, g_str>>>(a, b, out, n);
    return hipGetLastError() == hipSuccess;
}

__global__ static void gdn_conv_k(const float *hist, const float *qkv,
                                  const float *w, float *out, uint32_t ch,
                                  uint32_t ksz) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    float acc = 0.f;
    for (uint32_t k = 0; k + 1 < ksz; k++)
        acc += w[k + c * ksz] * hist[k * ch + c];
    acc += w[(ksz - 1) + c * ksz] * qkv[c];
    out[c] = acc / (1.f + expf(-acc));
}

/* Fused conv + history shift: one thread owns channel c end-to-end so the
 * read-then-update of hist needs no extra kernel (saves 1 launch/layer). */
__global__ static void gdn_convshift_k(float *hist, const float *qkv,
                                       const float *w, float *out, uint32_t ch,
                                       uint32_t ksz) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    float acc = 0.f;
    for (uint32_t k = 0; k + 1 < ksz; k++)
        acc += w[k + c * ksz] * hist[k * ch + c];
    acc += w[(ksz - 1) + c * ksz] * qkv[c];
    out[c] = acc / (1.f + expf(-acc));
    for (uint32_t k = 0; k + 2 < ksz; k++)
        hist[k * ch + c] = hist[(k + 1) * ch + c];
    hist[(ksz - 2) * ch + c] = qkv[c];
}
extern "C" bool q4_hip_gdn_convshift(float *hist, const float *qkv,
                                     const float *w, float *out, uint32_t ch,
                                     uint32_t ksz) {
    if (!g_ok || !hist || !qkv || !w || !out || ksz < 2) return false;
    dim3 b(256), g((ch + 255) / 256);
    gdn_convshift_k<<<g, b, 0, g_str>>>(hist, qkv, w, out, ch, ksz);
    return hipGetLastError() == hipSuccess;
}
extern "C" bool q4_hip_gdn_conv(const float *hist, const float *qkv, const float *w,
                                float *out, uint32_t ch, uint32_t ksz) {
    if (!g_ok || !hist || !qkv || !w || !out) return false;
    dim3 b(256), g((ch + 255) / 256);
    gdn_conv_k<<<g, b, 0, g_str>>>(hist, qkv, w, out, ch, ksz);
    return hipGetLastError() == hipSuccess;
}

__global__ static void gdn_shift_k(float *hist, const float *qkv, uint32_t ch,
                                   uint32_t ksz) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    for (uint32_t k = 0; k + 2 < ksz; k++)
        hist[k * ch + c] = hist[(k + 1) * ch + c];
    hist[(ksz - 2) * ch + c] = qkv[c];
}
extern "C" bool q4_hip_gdn_shift_hist(float *hist, const float *qkv, uint32_t ch,
                                      uint32_t ksz) {
    if (!g_ok || !hist || !qkv || ksz < 2) return false;
    dim3 b(256), g((ch + 255) / 256);
    gdn_shift_k<<<g, b, 0, g_str>>>(hist, qkv, ch, ksz);
    return hipGetLastError() == hipSuccess;
}

__global__ static void l2_heads_k(float *x, uint32_t d, float eps) {
    uint32_t h = blockIdx.x;
    float *v = x + (size_t)h * d;
    float ss = 0.f;
    for (uint32_t i = threadIdx.x; i < d; i += blockDim.x) ss += v[i] * v[i];
    __shared__ float sh[128];
    sh[threadIdx.x] = ss;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float scale = rsqrtf(sh[0] + eps);
    for (uint32_t i = threadIdx.x; i < d; i += blockDim.x) v[i] *= scale;
}
extern "C" bool q4_hip_l2_heads(float *x, uint32_t n_h, uint32_t d, float eps) {
    if (!g_ok || !x || n_h == 0) return false;
    l2_heads_k<<<n_h, 128, 0, g_str>>>(x, d, eps);
    return hipGetLastError() == hipSuccess;
}

__device__ static inline float softplus_d(float x) {
    if (x > 20.f) return x;
    if (x < -20.f) return expf(x);
    return logf(1.f + expf(x));
}

__global__ static void gdn_step_k(float *S, const float *q, const float *k,
                                  const float *v, const float *beta,
                                  const float *alpha, const float *A,
                                  const float *dt, float *y, uint32_t n_kh,
                                  uint32_t d, float qscale) {
    uint32_t vh = blockIdx.x;
    uint32_t col = threadIdx.x;
    if (col >= d) return;
    /* GGUF V heads are tiled: pair V[vh] with K[vh % n_k]. */
    uint32_t kh = n_kh ? (vh % n_kh) : 0;
    const float *qh = q + kh * d;
    const float *khv = k + kh * d;
    float gv = expf(A[vh] * softplus_d(alpha[vh] + dt[vh]));
    float b = 1.f / (1.f + expf(-beta[vh])); /* sigmoid folded: raw logits in */
    float *Sh = S + (size_t)vh * d * d;
    float kv = 0.f;
    for (uint32_t i = 0; i < d; i++) kv += Sh[i * d + col] * khv[i];
    float delta = (v[vh * d + col] - gv * kv) * b;
    float att = 0.f;
    for (uint32_t i = 0; i < d; i++) {
        float s = gv * Sh[i * d + col] + khv[i] * delta;
        Sh[i * d + col] = s;
        att += s * qh[i];
    }
    y[vh * d + col] = att * qscale;
}
extern "C" bool q4_hip_gdn_step(float *S, const float *q, const float *k,
                                const float *v, const float *beta,
                                const float *alpha, const float *A,
                                const float *dt, float *y, uint32_t n_vh,
                                uint32_t n_kh, uint32_t d, float qscale) {
    if (!g_ok || !S || !q || !k || !v || !beta || !alpha || !A || !dt || !y)
        return false;
    gdn_step_k<<<n_vh, d, 0, g_str>>>(S, q, k, v, beta, alpha, A, dt, y, n_kh, d, qscale);
    return hipGetLastError() == hipSuccess;
}

/* Generation-count grid barrier: safe here because nblocks (≤48) is far
 * below the resident limit, so every block coexists on the device.
 * g_bar is declared near the top of this file. */
__device__ static void gbar(uint32_t *cnt, volatile uint32_t *fl,
                            uint32_t nb) {
    __syncthreads();
    if (threadIdx.x == 0) {
        uint32_t gen = *fl;
        if (atomicAdd(cnt, 1u) + 1 == nb) {
            *cnt = 0;
            atomicExch((uint32_t *)fl, gen + 1);
        } else {
            while (*fl == gen) __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();
}

/* Whole GDN tail in ONE launch: conv+hist-shift, q/k l2 norms, gated
 * delta-net step, rms+sigmoid(z) gate — all per-value-head work lives inside
 * a single block so no inter-block sync is needed. Requires d_k == d_v.
 * The hist update waits on a grid barrier: q/k channel ranges are shared by
 * several vh blocks, so every block must finish reading before anyone
 * writes. */
__global__ static void gdn_tail_k(float *hist, const float *qkv,
                                  const float *cw, uint32_t ch, uint32_t ksz,
                                  float *S, const float *beta,
                                  const float *alpha, const float *A,
                                  const float *dt, const float *nw,
                                  const float *z, float *core, uint32_t n_kh,
                                  uint32_t d, float eps, float qscale,
                                  uint32_t *bar) {
    uint32_t vh = blockIdx.x;
    uint32_t col = threadIdx.x;
    if (col >= d) return;
    uint32_t kh = n_kh ? (vh % n_kh) : 0;
    __shared__ float qs[128], ks[128], vs[128], sh[128];
    const uint32_t base[3] = { kh * d, d * n_kh + kh * d,
                               2 * d * n_kh + vh * d };
    float *dst[3] = { qs, ks, vs };
    /* conv + silu for this block's q,k,v channel slices (read-only pass) */
    for (int s = 0; s < 3; s++) {
        uint32_t c = base[s] + col;
        float acc = 0.f;
        for (uint32_t j = 0; j + 1 < ksz; j++)
            acc += cw[j + c * ksz] * hist[j * ch + c];
        acc += cw[(ksz - 1) + c * ksz] * qkv[c];
        dst[s][col] = acc / (1.f + expf(-acc));
    }
    /* all conv reads done everywhere → hist updates are now race-free */
    gbar(bar, bar + 1, gridDim.x);
    /* q/k channel updates: only the first block of each kh group writes
     * (the three vh blocks sharing a kh would otherwise triple-write). */
    if (vh < n_kh) {
        for (int s = 0; s < 2; s++) {
            uint32_t c = base[s] + col;
            for (uint32_t j = 0; j + 2 < ksz; j++)
                hist[j * ch + c] = hist[(j + 1) * ch + c];
            hist[(ksz - 2) * ch + c] = qkv[c];
        }
    }
    /* v channels are block-private: every block shifts its own slice */
    {
        uint32_t c = base[2] + col;
        for (uint32_t j = 0; j + 2 < ksz; j++)
            hist[j * ch + c] = hist[(j + 1) * ch + c];
        hist[(ksz - 2) * ch + c] = qkv[c];
    }
    __syncthreads();
    /* l2-normalize q and k head slices (v stays raw) */
    for (int s = 0; s < 2; s++) {
        float *v = dst[s];
        sh[col] = v[col] * v[col];
        __syncthreads();
        for (uint32_t t = d >> 1; t > 0; t >>= 1) {
            if (col < t) sh[col] += sh[col + t];
            __syncthreads();
        }
        float sc = rsqrtf(sh[0] + eps);
        __syncthreads();
        v[col] *= sc;
        __syncthreads();
    }
    /* gated delta-net step: beta sigmoid is applied here (raw logits in) */
    float gv = expf(A[vh] * softplus_d(alpha[vh] + dt[vh]));
    float b = 1.f / (1.f + expf(-beta[vh]));
    float *Sh = S + (size_t)vh * d * d;
    float kv = 0.f;
    for (uint32_t i = 0; i < d; i++) kv += Sh[i * d + col] * ks[i];
    float delta = (vs[col] - gv * kv) * b;
    float att = 0.f;
    for (uint32_t i = 0; i < d; i++) {
        float s = gv * Sh[i * d + col] + ks[i] * delta;
        Sh[i * d + col] = s;
        att += s * qs[i];
    }
    float y = att * qscale;
    /* rms + sigmoid(z) gate epilogue, written straight to core */
    sh[col] = y * y;
    __syncthreads();
    for (uint32_t t = d >> 1; t > 0; t >>= 1) {
        if (col < t) sh[col] += sh[col + t];
        __syncthreads();
    }
    float sc = rsqrtf(sh[0] / (float)d + eps);
    float zz = z[(size_t)vh * d + col];
    core[(size_t)vh * d + col] =
        (y * sc * (nw ? nw[col] : 1.f)) / (1.f + expf(-zz));
}
extern "C" bool q4_hip_gdn_tail(float *hist, const float *qkv, const float *cw,
                                uint32_t ch, uint32_t ksz, float *S,
                                const float *beta, const float *alpha,
                                const float *A, const float *dt,
                                const float *nw, const float *z, float *core,
                                uint32_t n_vh, uint32_t n_kh, uint32_t d,
                                float eps, float qscale) {
    if (!g_ok || !g_bar || !hist || !qkv || !cw || !S || !beta || !alpha ||
        !A || !dt || !z || !core || !n_vh || !n_kh || n_vh % n_kh ||
        d != 128 || ksz < 2)
        return false;
    gdn_tail_k<<<n_vh, d, 0, g_str>>>(hist, qkv, cw, ch, ksz, S, beta, alpha,
                                      A, dt, nw, z, core, n_kh, d, eps,
                                      qscale, g_bar);
    return hipGetLastError() == hipSuccess;
}

/* Whole-chunk GDN scan (prefill): one launch per layer. Block vh keeps its
 * column of the delta-net state S in registers across all T tokens and a
 * private conv-history slice in shared, so the per-token launches and the
 * grid barrier of gdn_tail_k disappear. Math is op-for-op identical:
 * conv+silu -> hist shift -> l2(q,k) -> delta step -> rms+sigmoid(z) gate. */
__global__ static void gdn_scan_k(float *hist, const float *qkv,
                                  const float *cw, uint32_t ch, uint32_t ksz,
                                  float *S, const float *beta,
                                  const float *alpha, const float *A,
                                  const float *dt, const float *nw,
                                  const float *z, float *core, uint32_t n_tok,
                                  uint32_t n_kh, uint32_t d, float eps,
                                  float qscale, float *qkh) {
    constexpr uint32_t D = 128, KMAX = 8, QS = D / 4;
    uint32_t vh = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t col = tid >> 2;        /* 4 threads share each column */
    uint32_t qt = tid & 3;          /* this thread's row quarter */
    if (col >= d || ksz > KMAX) return;
    uint32_t n_vh = gridDim.x;
    uint32_t kh = n_kh ? (vh % n_kh) : 0;
    const uint32_t base[3] = { kh * d, d * n_kh + kh * d,
                               2 * d * n_kh + vh * d };
    __shared__ float qs[D], ks[D], vs[D], sh[2 * D];
    __shared__ float h[KMAX][3 * D]; /* private hist rows for q,k,v slices */
    /* Load this block's hist slices (q,k of its kh; v of its vh). */
    for (uint32_t e = tid; e < 3 * D; e += blockDim.x) {
        uint32_t s = e / D, cc = e - s * D;
        for (uint32_t j = 0; j + 1 < ksz; j++)
            h[j][e] = hist[(size_t)j * ch + base[s] + cc];
    }
    float *Sh = S + (size_t)vh * d * d;
    float scol[QS];
#pragma unroll
    for (uint32_t i = 0; i < QS; i++)
        scol[i] = Sh[(size_t)(qt * QS + i) * d + col];
    __syncthreads();
    for (uint32_t t = 0; t < n_tok; t++) {
        const float *x = qkv + (size_t)t * ch;
        /* conv + silu on the private slices, then private shift+append */
        for (uint32_t e = tid; e < 3 * D; e += blockDim.x) {
            uint32_t s = e / D, cc = e - s * D;
            uint32_t c = base[s] + cc;
            float acc = 0.f;
            for (uint32_t j = 0; j + 1 < ksz; j++)
                acc += cw[j + (size_t)c * ksz] * h[j][e];
            acc += cw[(ksz - 1) + (size_t)c * ksz] * x[c];
            float v = acc / (1.f + expf(-acc));
            (s == 0 ? qs : s == 1 ? ks : vs)[cc] = v;
            for (uint32_t j = 0; j + 2 < ksz; j++)
                h[j][e] = h[j + 1][e];
            h[ksz - 2][e] = x[c];
        }
        __syncthreads();
        /* l2-normalize q and k in shared (v stays raw): both sums in one
         * reduction pass — halves the barrier count per token step. */
        if (tid < D) {
            sh[tid] = qs[tid] * qs[tid];
            sh[D + tid] = ks[tid] * ks[tid];
        }
        __syncthreads();
        for (uint32_t r = D >> 1; r > 0; r >>= 1) {
            if (tid < r) {
                sh[tid] += sh[tid + r];
                sh[D + tid] += sh[D + tid + r];
            }
            __syncthreads();
        }
        {
            float scq = rsqrtf(sh[0] + eps), sck = rsqrtf(sh[D] + eps);
            __syncthreads();
            if (tid < D) {
                qs[tid] *= scq;
                ks[tid] *= sck;
            }
            __syncthreads();
        }
        /* gated delta-net step on the register quarter-column */
        float gv = expf(A[vh] * softplus_d(alpha[(size_t)t * n_vh + vh] + dt[vh]));
        float b = 1.f / (1.f + expf(-beta[(size_t)t * n_vh + vh]));
        float kv = 0.f;
#pragma unroll
        for (uint32_t i = 0; i < QS; i++)
            kv += scol[i] * ks[qt * QS + i];
        kv += __shfl_xor(kv, 1);
        kv += __shfl_xor(kv, 2);
        float delta = (vs[col] - gv * kv) * b;
        float att = 0.f;
#pragma unroll
        for (uint32_t i = 0; i < QS; i++) {
            float s = gv * scol[i] + ks[qt * QS + i] * delta;
            scol[i] = s;
            att += s * qs[qt * QS + i];
        }
        att += __shfl_xor(att, 1);
        att += __shfl_xor(att, 2);
        float y = att * qscale;
        /* rms + sigmoid(z) gate epilogue straight to core */
        if (qt == 0) sh[col] = y * y;
        __syncthreads();
        for (uint32_t r = D >> 1; r > 0; r >>= 1) {
            if (tid < r) sh[tid] += sh[tid + r];
            __syncthreads();
        }
        float sc = rsqrtf(sh[0] / (float)d + eps);
        if (qt == 0) {
            float zz = z[(size_t)t * n_vh * d + (size_t)vh * d + col];
            core[(size_t)t * n_vh * d + (size_t)vh * d + col] =
                (y * sc * (nw ? nw[col] : 1.f)) / (1.f + expf(-zz));
        }
        __syncthreads();
    }
    /* Write back the final state: S quarter-column and the private hist rows.
     * The q/k rows are shared across kh groups and read at kernel start by
     * every block, so they must not be overwritten in place — they go to the
     * qkh scratch and are copied back by the host after this launch. The v
     * rows are block-private and stay in place. */
#pragma unroll
    for (uint32_t i = 0; i < QS; i++)
        Sh[(size_t)(qt * QS + i) * d + col] = scol[i];
    for (uint32_t e = tid; e < 3 * D; e += blockDim.x) {
        uint32_t s = e / D, cc = e - s * D;
        if (s < 2 && vh >= n_kh) continue;
        float *dst = s < 2 ? qkh : hist;
        for (uint32_t j = 0; j + 1 < ksz; j++)
            dst[(size_t)j * ch + base[s] + cc] = h[j][e];
    }
}
extern "C" bool q4_hip_gdn_scan(float *hist, const float *qkv, const float *cw,
                                uint32_t ch, uint32_t ksz, float *S,
                                const float *beta, const float *alpha,
                                const float *A, const float *dt,
                                const float *nw, const float *z, float *core,
                                uint32_t n_tok, uint32_t n_vh, uint32_t n_kh,
                                uint32_t d, float eps, float qscale) {
    if (!g_ok || !hist || !qkv || !cw || !S || !beta || !alpha || !A || !dt ||
        !z || !core || !n_tok || !n_vh || !n_kh || n_vh % n_kh || d != 128 ||
        ksz < 2)
        return false;
    if (!grow((void **)&g_scanqk, &g_scanqk_n,
              (size_t)(ksz - 1) * ch * sizeof(float)))
        return false;
    gdn_scan_k<<<n_vh, 512, 0, g_str>>>(hist, qkv, cw, ch, ksz, S, beta, alpha,
                                      A, dt, nw, z, core, n_tok, n_kh, d, eps,
                                      qscale, g_scanqk);
    if (hipGetLastError() != hipSuccess) return false;
    /* Stream-ordered copy-back of the shared q/k channel rows [0, 2*d*n_kh). */
    for (uint32_t j = 0; j + 1 < ksz; j++)
        if (hipMemcpyAsync(hist + (size_t)j * ch, g_scanqk + (size_t)j * ch,
                           (size_t)2 * n_kh * d * sizeof(float),
                           hipMemcpyDeviceToDevice, g_str) != hipSuccess)
            return false;
    return true;
}


__global__ static void rms_gate_heads_k(float *h, const float *w, const float *z,
                                        uint32_t d, float eps) {
    uint32_t vh = blockIdx.x;
    float *hv = h + (size_t)vh * d;
    const float *zv = z + (size_t)vh * d;
    float ss = 0.f;
    for (uint32_t i = threadIdx.x; i < d; i += blockDim.x) ss += hv[i] * hv[i];
    __shared__ float sh[128];
    sh[threadIdx.x] = ss;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float scale = rsqrtf(sh[0] / (float)d + eps);
    for (uint32_t i = threadIdx.x; i < d; i += blockDim.x) {
        float nrm = hv[i] * scale * (w ? w[i] : 1.f);
        hv[i] = nrm / (1.f + expf(-zv[i]));
    }
}
/* Publish the decode position for graph-captured kernels. Stream-ordered so
 * replays launched afterwards read the new value. */
extern "C" void q4_hip_pos_push(int32_t pos) {
    if (!g_pos_d || !g_pos_h || !g_str) return;
    *g_pos_h = pos;
    (void)hipMemcpyAsync(g_pos_d, g_pos_h, sizeof(int32_t),
                         hipMemcpyHostToDevice, g_str);
}
extern "C" int q4_hip_capturing(void) { return g_capturing; }
extern "C" int q4_hip_pos_dev_ok(void) {
    return g_pos_d != nullptr && g_pos_h != nullptr;
}

__global__ static void qsa_q_prep_k(const float *qg, const float *wn, float *q,
                                    float *g, uint32_t hd, uint32_t n_rot,
                                    int32_t pos0, float base, float eps,
                                    uint32_t n_head, const int32_t *pp) {
    uint32_t h = blockIdx.x;
    uint32_t t = blockIdx.y;
    int32_t pos = (pp ? *pp : pos0) + (int32_t)t;
    const float *src = qg + ((size_t)t * n_head + h) * hd * 2;
    float *qd = q + ((size_t)t * n_head + h) * hd;
    float *gd = g + ((size_t)t * n_head + h) * hd;
    for (uint32_t i = threadIdx.x; i < hd; i += blockDim.x) gd[i] = src[hd + i];
    float ss = 0.f;
    for (uint32_t i = threadIdx.x; i < hd; i += blockDim.x) ss += src[i] * src[i];
    __shared__ float sh[256];
    sh[threadIdx.x] = ss;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float scale = rsqrtf(sh[0] / (float)hd + eps);
    for (uint32_t i = threadIdx.x; i < hd; i += blockDim.x)
        qd[i] = src[i] * scale * (wn ? wn[i] : 1.f);
    __syncthreads();
    uint32_t np = n_rot / 2;
    for (uint32_t p = threadIdx.x; p < np; p += blockDim.x) {
        float freq = powf(base, -(float)p / (float)np);
        float ang = (float)pos * freq;
        float c = cosf(ang), si = sinf(ang);
        float x0 = qd[2 * p], x1 = qd[2 * p + 1];
        qd[2 * p] = x0 * c - x1 * si;
        qd[2 * p + 1] = x0 * si + x1 * c;
    }
}

__global__ static void qsa_k_prep_k(float *k, const float *wn, uint32_t hd,
                                    uint32_t n_rot, int32_t pos0, float base,
                                    float eps, uint32_t n_kvh,
                                    const int32_t *pp) {
    uint32_t h = blockIdx.x;
    uint32_t t = blockIdx.y;
    int32_t pos = (pp ? *pp : pos0) + (int32_t)t;
    float *kh = k + ((size_t)t * n_kvh + h) * hd;
    float ss = 0.f;
    for (uint32_t i = threadIdx.x; i < hd; i += blockDim.x) ss += kh[i] * kh[i];
    __shared__ float sh[256];
    sh[threadIdx.x] = ss;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float scale = rsqrtf(sh[0] / (float)hd + eps);
    for (uint32_t i = threadIdx.x; i < hd; i += blockDim.x)
        kh[i] = kh[i] * scale * (wn ? wn[i] : 1.f);
    __syncthreads();
    uint32_t np = n_rot / 2;
    for (uint32_t p = threadIdx.x; p < np; p += blockDim.x) {
        float freq = powf(base, -(float)p / (float)np);
        float ang = (float)pos * freq;
        float c = cosf(ang), si = sinf(ang);
        float x0 = kh[2 * p], x1 = kh[2 * p + 1];
        kh[2 * p] = x0 * c - x1 * si;
        kh[2 * p + 1] = x0 * si + x1 * c;
    }
}

extern "C" bool q4_hip_qsa_q_prep_n(const float *d_qg, const float *d_wn, float *d_q,
                                    float *d_g, uint32_t n_head, uint32_t hd,
                                    uint32_t n_rot, int32_t pos0, float base,
                                    float eps, uint32_t n_tok) {
    if (!g_ok || !d_qg || !d_q || !d_g || n_head == 0 || n_tok == 0) return false;
    qsa_q_prep_k<<<dim3(n_head, n_tok), 256, 0, g_str>>>(d_qg, d_wn, d_q, d_g, hd, n_rot, pos0,
                                               base, eps, n_head, nullptr);
    return hipGetLastError() == hipSuccess;
}
/* Device-position variant for graph capture: pos comes from g_pos_d. */
extern "C" bool q4_hip_qsa_q_prep_d(const float *d_qg, const float *d_wn,
                                    float *d_q, float *d_g, uint32_t n_head,
                                    uint32_t hd, uint32_t n_rot, float base,
                                    float eps) {
    if (!g_ok || !g_pos_d || !d_qg || !d_q || !d_g || n_head == 0) return false;
    qsa_q_prep_k<<<dim3(n_head, 1), 256, 0, g_str>>>(d_qg, d_wn, d_q, d_g, hd,
                                                   n_rot, 0, base, eps, n_head,
                                                   g_pos_d);
    return hipGetLastError() == hipSuccess;
}
extern "C" bool q4_hip_qsa_q_prep(const float *d_qg, const float *d_wn, float *d_q,
                                  float *d_g, uint32_t n_head, uint32_t hd,
                                  uint32_t n_rot, int32_t pos, float base,
                                  float eps) {
    return q4_hip_qsa_q_prep_n(d_qg, d_wn, d_q, d_g, n_head, hd, n_rot, pos, base,
                               eps, 1);
}
extern "C" bool q4_hip_qsa_k_prep_n(float *d_k, const float *d_wn, uint32_t n_kvh,
                                    uint32_t hd, uint32_t n_rot, int32_t pos0,
                                    float base, float eps, uint32_t n_tok) {
    if (!g_ok || !d_k || n_kvh == 0 || n_tok == 0) return false;
    qsa_k_prep_k<<<dim3(n_kvh, n_tok), 256, 0, g_str>>>(d_k, d_wn, hd, n_rot, pos0, base, eps,
                                              n_kvh, nullptr);
    return hipGetLastError() == hipSuccess;
}
extern "C" bool q4_hip_qsa_k_prep_d(float *d_k, const float *d_wn,
                                    uint32_t n_kvh, uint32_t hd, uint32_t n_rot,
                                    float base, float eps) {
    if (!g_ok || !g_pos_d || !d_k || n_kvh == 0) return false;
    qsa_k_prep_k<<<dim3(n_kvh, 1), 256, 0, g_str>>>(d_k, d_wn, hd, n_rot, 0,
                                                  base, eps, n_kvh, g_pos_d);
    return hipGetLastError() == hipSuccess;
}
extern "C" bool q4_hip_qsa_k_prep(float *d_k, const float *d_wn, uint32_t n_kvh,
                                  uint32_t hd, uint32_t n_rot, int32_t pos,
                                  float base, float eps) {
    return q4_hip_qsa_k_prep_n(d_k, d_wn, n_kvh, hd, n_rot, pos, base, eps, 1);
}

extern "C" bool q4_hip_rms_gate_heads(float *h, const float *w, const float *z,
                                      uint32_t n_h, uint32_t d, float eps) {
    if (!g_ok || !h || !z || n_h == 0) return false;
    rms_gate_heads_k<<<n_h, 128, 0, g_str>>>(h, w, z, d, eps);
    return hipGetLastError() == hipSuccess;
}

__global__ static void softmax_topk_k(const float *x, int32_t *ids, float *wts,
                                      int n, int k) {
    __shared__ float buf[512];
    int t = (int)threadIdx.x;
    buf[t] = (t < n) ? x[t] : -INFINITY;
    __syncthreads();
    float m = -INFINITY;
    for (int i = t; i < n; i += (int)blockDim.x)
        if (buf[i] > m) m = buf[i];
    __shared__ float sh[512];
    sh[t] = m;
    __syncthreads();
    for (int s = (int)blockDim.x >> 1; s > 0; s >>= 1) {
        if (t < s && sh[t + s] > sh[t]) sh[t] = sh[t + s];
        __syncthreads();
    }
    m = sh[0];
    float e = (t < n) ? expf(buf[t] - m) : 0.f;
    buf[t] = e;
    __syncthreads();
    float sum = 0.f;
    for (int i = t; i < n; i += (int)blockDim.x) sum += buf[i];
    sh[t] = sum;
    __syncthreads();
    for (int s = (int)blockDim.x >> 1; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        __syncthreads();
    }
    float inv = sh[0] > 0.f ? 1.f / sh[0] : 0.f;
    if (t < n) buf[t] *= inv;
    __syncthreads();
    if (t == 0) {
        for (int j = 0; j < k; j++) {
            int bi = 0;
            float bv = -1.f;
            for (int i = 0; i < n; i++) {
                if (buf[i] > bv) {
                    bv = buf[i];
                    bi = i;
                }
            }
            ids[j] = bi;
            wts[j] = bv;
            buf[bi] = -1.f;
        }
        float s = 0.f;
        for (int j = 0; j < k; j++) s += wts[j];
        if (s > 0.f) {
            float r = 1.f / s;
            for (int j = 0; j < k; j++) wts[j] *= r;
        }
    }
}

extern "C" bool q4_hip_softmax_topk(const float *d_logits, uint32_t n, uint32_t k,
                                    int32_t *h_ids, float *h_wts) {
    if (!g_ok || !d_logits || !h_ids || !h_wts || n == 0 || k == 0 || n > 512)
        return false;
    static int32_t *d_ids = nullptr;
    static float *d_wts = nullptr;
    if (!d_ids) {
        if (hipMalloc((void **)&d_ids, 64 * sizeof(int32_t)) != hipSuccess)
            return false;
        if (hipMalloc((void **)&d_wts, 64 * sizeof(float)) != hipSuccess)
            return false;
    }
    softmax_topk_k<<<1, 512, 0, g_str>>>(d_logits, d_ids, d_wts, (int)n, (int)k);
    if (hipGetLastError() != hipSuccess) return false;
    if (!sync_copy(h_ids, d_ids, k * sizeof(int32_t), hipMemcpyDeviceToHost))
        return false;
    return sync_copy(h_wts, d_wts, k * sizeof(float), hipMemcpyDeviceToHost);
}

__device__ static float q4k_block_dot(const uint8_t *blk, const float *xr) {
    float d = f16d(blk);
    float minv = f16d(blk + 2);
    const uint8_t *sc = blk + 4;
    const uint8_t *q = blk + 16;
    float acc = 0.f;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uint8_t s1, m1, s2, m2;
        scale_min_k4(is + 0, sc, &s1, &m1);
        scale_min_k4(is + 1, sc, &s2, &m2);
        float d1 = d * s1, mm1 = minv * m1;
        float d2 = d * s2, mm2 = minv * m2;
        for (int l = 0; l < 32; l++)
            acc += (d1 * (q[l] & 0xF) - mm1) * xr[j + l];
        for (int l = 0; l < 32; l++)
            acc += (d2 * (q[l] >> 4) - mm2) * xr[j + 32 + l];
        q += 32;
        is += 2;
    }
    return acc;
}

__global__ static void gateup_q4k_k(const uint8_t *gate, const uint8_t *up,
                                    const float *x, float *h, uint32_t n_ff,
                                    uint32_t n_embd) {
    uint32_t row = blockIdx.x;
    uint32_t tok = blockIdx.y;
    if (row >= n_ff) return;
    x += (size_t)tok * n_embd;
    h += (size_t)tok * n_ff;
    uint32_t nb = n_embd / 256u;
    uint64_t rb = (uint64_t)nb * 144ull;
    const uint8_t *gr = gate + row * rb;
    const uint8_t *ur = up + row * rb;
    float acc_g = 0.f, acc_u = 0.f;
    for (uint32_t i = threadIdx.x; i < nb; i += blockDim.x) {
        acc_g += q4k_block_dot(gr + i * 144u, x + i * 256u);
        acc_u += q4k_block_dot(ur + i * 144u, x + i * 256u);
    }
    __shared__ float sg[64], su[64];
    sg[threadIdx.x] = acc_g;
    su[threadIdx.x] = acc_u;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sg[threadIdx.x] += sg[threadIdx.x + s];
            su[threadIdx.x] += su[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float g = sg[0];
        h[row] = (g / (1.f + expf(-g))) * su[0];
    }
}

__global__ static void gateup_q4k_n_k(const uint8_t *gate, const uint8_t *up,
                                      const float *x, float *h, uint32_t n_ff,
                                      uint32_t n_embd, uint32_t n_tok,
                                      const int32_t *which) {
    uint32_t row = blockIdx.x;
    if (row >= n_ff) return;
    uint32_t nb = n_embd / 256u;
    uint64_t rb = (uint64_t)nb * 144ull;
    const uint8_t *gr = gate + row * rb;
    const uint8_t *ur = up + row * rb;
    __shared__ float sg[64], su[64];
    for (uint32_t tok = blockIdx.y; tok < n_tok; tok += gridDim.y ? gridDim.y : 1) {
        const float *xt = x + (size_t)(which ? which[tok] : (int32_t)tok) * n_embd;
        float acc_g = 0.f, acc_u = 0.f;
        for (uint32_t i = threadIdx.x; i < nb; i += blockDim.x) {
            acc_g += q4k_block_dot(gr + i * 144u, xt + i * 256u);
            acc_u += q4k_block_dot(ur + i * 144u, xt + i * 256u);
        }
        sg[threadIdx.x] = acc_g;
        su[threadIdx.x] = acc_u;
        __syncthreads();
        for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sg[threadIdx.x] += sg[threadIdx.x + s];
                su[threadIdx.x] += su[threadIdx.x + s];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            float g = sg[0];
            h[(size_t)tok * n_ff + row] = (g / (1.f + expf(-g))) * su[0];
        }
        __syncthreads();
    }
}

extern "C" bool q4_hip_gateup_q4k_n(const uint8_t *d_gate, const uint8_t *d_up,
                                    const float *d_x, float *d_h, uint32_t n_ff,
                                    uint32_t n_embd, uint32_t n_tok) {
    return q4_hip_gateup_q4k_n_which(d_gate, d_up, d_x, d_h, n_ff, n_embd, n_tok,
                                     NULL);
}

extern "C" bool q4_hip_gateup_q4k_n_which(const uint8_t *d_gate,
                                          const uint8_t *d_up, const float *d_x,
                                          float *d_h, uint32_t n_ff,
                                          uint32_t n_embd, uint32_t n_tok,
                                          const int32_t *which) {
    if (!g_ok || !d_gate || !d_up || !d_x || !d_h) return false;
    if (!n_tok) n_tok = 1;
    uint32_t gy = n_tok < 64u ? n_tok : 64u;
    gateup_q4k_n_k<<<dim3(n_ff, gy), 64, 0, g_str>>>(d_gate, d_up, d_x, d_h, n_ff, n_embd,
                                           n_tok, which);
    return hipGetLastError() == hipSuccess;
}
extern "C" bool q4_hip_gateup_q4k(const uint8_t *d_gate, const uint8_t *d_up,
                                  const float *d_x, float *d_h, uint32_t n_ff,
                                  uint32_t n_embd) {
    return q4_hip_gateup_q4k_n(d_gate, d_up, d_x, d_h, n_ff, n_embd, 1);
}

__device__ static float kv_ld(const float *p) { return *p; }
__device__ static float kv_ld(const __half *p) { return __half2float(*p); }

/* One query (head, token). KV tile is staged in shared memory so the K
 * load is coalesced, and softmax max/sum run once per tile instead of
 * once per thread. kv_f16 stores the cache as fp16. */
template <int KIND, int INDEXED>
__global__ static void gqa_decode_fast(const float *q, const uint8_t *kcache,
                                       const uint8_t *vcache, const float *gate,
                                       float *out, uint32_t n_head, uint32_t n_kvh,
                                       uint32_t hd, uint32_t kv_base, uint32_t max_kv,
                                       float scale, const int32_t *idx,
                                       uint32_t width) {
    constexpr uint32_t TILE = 32;
    uint32_t h = blockIdx.x;
    uint32_t t = blockIdx.y;
    if (h >= n_head || n_kvh == 0 || hd == 0 || hd > 256) return;
    uint32_t gqa = n_head / n_kvh;
    if (gqa == 0) return;
    uint32_t kh = h / gqa;
    uint32_t n_kv = kv_base + t + 1;
    if (n_kv > max_kv) n_kv = max_kv;
    uint32_t n_pos = INDEXED ? width : n_kv;
    const float *qh = q + ((size_t)t * n_head + h) * hd;
    uint32_t i = threadIdx.x;

    __shared__ float qs[256];
    __shared__ float acc[256];
    __shared__ float tile[TILE * 256];
    __shared__ float wt[TILE];
    __shared__ float red[256];
    if (i < hd) {
        qs[i] = qh[i];
        acc[i] = 0.f;
    }
    __syncthreads();

    int have_m = 0;
    float m = 0.f, sum = 0.f;
    for (uint32_t base = 0; base < n_pos; base += TILE) {
        uint32_t lim = n_pos - base;
        if (lim > TILE) lim = TILE;
        uint32_t tile_n = TILE * hd;
        uint32_t q8b = (hd / 32u) * 34u;
        if constexpr (KIND == 2) {
            if (i < lim) {
                int p = INDEXED ? idx[(size_t)t * width + base + i]
                                : (int)(base + i);
                const uint8_t *src =
                    p >= 0 ? kcache + ((size_t)p * n_kvh + kh) * q8b : nullptr;
                float *dst = tile + (size_t)i * hd;
                if (!src) {
                    for (uint32_t d = 0; d < hd; d++) dst[d] = 0.f;
                } else {
                for (uint32_t b = 0; b < hd / 32u; b++) {
                    float sc = f16d(src);
                    const int8_t *qs8 = (const int8_t *)(src + 2);
                    for (int j = 0; j < 32; j++)
                        dst[b * 32u + j] = sc * (float)qs8[j];
                    src += 34;
                }
                }
            }
        } else {
            for (uint32_t e = i; e < tile_n; e += blockDim.x) {
                uint32_t pp = e / hd;
                uint32_t dd = e - pp * hd;
                float v = 0.f;
                int p = -1;
                if (pp < lim)
                    p = INDEXED ? idx[(size_t)t * width + base + pp]
                                : (int)(base + pp);
                if (p >= 0) {
                    const uint8_t *elt =
                        kcache + (((size_t)p * n_kvh + kh) * hd + dd) *
                                     (KIND == 1 ? 2u : 4u);
                    v = KIND == 1 ? __half2float(*(const __half *)elt)
                                  : *(const float *)elt;
                }
                tile[e] = __float2half(v);
            }
        }
        __syncthreads();
        /* 8 threads share one KV position so all 256 lanes do the dot. */
        uint32_t pos = i % TILE;
        uint32_t sub = i / TILE;
        uint32_t d0 = sub * (hd / 8u);
        uint32_t d1 = (sub == 7u) ? hd : d0 + (hd / 8u);
        float partial = 0.f;
        if (pos < lim) {
            const float *kk = tile + (size_t)pos * hd;
            for (uint32_t d = d0; d < d1; d++) partial += qs[d] * kk[d];
        }
        red[i] = partial;
        __syncthreads();
        float dot = -1.0e30f;
        if (i < TILE && i < lim) {
            int p = INDEXED ? idx[(size_t)t * width + base + i] : (int)(base + i);
            if (p >= 0) {
                dot = 0.f;
                for (uint32_t k = 0; k < 8; k++) dot += red[k * TILE + i];
                dot *= scale;
            }
        }
        __syncthreads();
        red[i] = (i < TILE) ? dot : -1.0e30f;
        __syncthreads();
        for (uint32_t s = 128; s > 0; s >>= 1) {
            if (i < s) red[i] = fmaxf(red[i], red[i + s]);
            __syncthreads();
        }
        float tm = red[0];
        int live = tm > -1.0e20f;
        float nm = m, alpha = 1.f;
        if (live) {
            if (!have_m) {
                nm = tm;
                alpha = 0.f;
                have_m = 1;
            } else if (tm > m) {
                nm = tm;
                alpha = expf(m - nm);
            } else {
                nm = m;
                alpha = 1.f;
            }
        }
        float w = 0.f;
        if (live && i < lim && dot > -1.0e20f) w = expf(dot - nm);
        if (i < TILE) wt[i] = w;
        __syncthreads();
        red[i] = (i < TILE) ? wt[i] : 0.f;
        __syncthreads();
        for (uint32_t s = 16; s > 0; s >>= 1) {
            if (i < s) red[i] += red[i + s];
            __syncthreads();
        }
        float tsum = red[0];
        if (i < hd) acc[i] *= alpha;
        sum *= alpha;
        __syncthreads();
        if constexpr (KIND == 2) {
            if (i < lim) {
                int p = INDEXED ? idx[(size_t)t * width + base + i]
                                : (int)(base + i);
                const uint8_t *src =
                    p >= 0 ? vcache + ((size_t)p * n_kvh + kh) * q8b : nullptr;
                float *dst = tile + (size_t)i * hd;
                if (!src) {
                    for (uint32_t d = 0; d < hd; d++) dst[d] = 0.f;
                } else {
                for (uint32_t b = 0; b < hd / 32u; b++) {
                    float sc = f16d(src);
                    const int8_t *qs8 = (const int8_t *)(src + 2);
                    for (int j = 0; j < 32; j++)
                        dst[b * 32u + j] = sc * (float)qs8[j];
                    src += 34;
                }
                }
            }
        } else {
            for (uint32_t e = i; e < tile_n; e += blockDim.x) {
                uint32_t pp = e / hd;
                uint32_t dd = e - pp * hd;
                float v = 0.f;
                int p = -1;
                if (pp < lim)
                    p = INDEXED ? idx[(size_t)t * width + base + pp]
                                : (int)(base + pp);
                if (p >= 0) {
                    const uint8_t *elt =
                        vcache + (((size_t)p * n_kvh + kh) * hd + dd) *
                                     (KIND == 1 ? 2u : 4u);
                    v = KIND == 1 ? __half2float(*(const __half *)elt)
                                  : *(const float *)elt;
                }
                tile[e] = __float2half(v);
            }
        }
        __syncthreads();
        if (i < hd) {
            float part = 0.f;
            for (uint32_t j = 0; j < lim; j++)
                part += wt[j] * tile[(size_t)j * hd + i];
            acc[i] += part;
        }
        sum += tsum;
        m = nm;
        __syncthreads();
    }
    float inv = sum > 0.f ? 1.f / sum : 0.f;
    if (i < hd) {
        float gv = 1.f / (1.f + expf(-gate[((size_t)t * n_head + h) * hd + i]));
        out[((size_t)t * n_head + h) * hd + i] = acc[i] * inv * gv;
    }
}

/* Decode attention, split-KV ("flash decoding") + GQA sharing: grid
 * (n_kvh, S). One workgroup runs ALL gqa query heads of its KV head over a
 * contiguous slice of the position list (indexed: slice of the sel list;
 * dense: slice of [0, n_kv)). The K/V tile is dequantized once per slice and
 * shared across heads. Per-head online-softmax partials (m, l, acc[hd]) go
 * to `part` for gqa_dec_comb_k to merge. n_tok == 1 only. */
#define GQA_DEC_SPLIT_S 128u
#define GQA_DEC_MAX_GQA 12u
/* Runtime split count for the decode path (env Q4_DEC_SPLIT_S, capped at
 * the compile-time partial buffers / combine shmem bound). */
static uint32_t g_dec_split_s(void) {
    static int s = -1;
    if (s < 0) {
        const char *e = getenv("Q4_DEC_SPLIT_S");
        long v = e ? strtol(e, nullptr, 10) : (long)GQA_DEC_SPLIT_S;
        if (v < 1) v = 1;
        if (v > (long)GQA_DEC_SPLIT_S) v = GQA_DEC_SPLIT_S;
        s = (int)v;
    }
    return (uint32_t)s;
}
template <int KIND, bool INDEXED>
__global__ static void gqa_dec_split_k(const float *q, const uint8_t *kcache,
                                       const uint8_t *vcache, float *part,
                                       uint32_t n_head, uint32_t n_kvh,
                                       uint32_t hd, uint32_t kv_base,
                                       uint32_t max_kv, float scale,
                                       const int32_t *idx, uint32_t width,
                                       uint32_t S) {
    constexpr uint32_t TILE = 32;
    uint32_t kh = blockIdx.x, s = blockIdx.y;
    uint32_t gqa = n_head / n_kvh;
    if (!gqa || !hd || hd > 256 || gqa > GQA_DEC_MAX_GQA) return;
    uint32_t n_kv = kv_base + 1;
    if (n_kv > max_kv) n_kv = max_kv;
    uint32_t n_pos = INDEXED ? width : n_kv;
    uint32_t per = (n_pos + S - 1) / S;
    uint32_t p0 = s * per;
    uint32_t p1 = p0 + per < n_pos ? p0 + per : n_pos;
    uint32_t i = threadIdx.x;
    float *pv = part + (((size_t)kh * S + s) * gqa) * (hd + 2u);

    __shared__ float qs[GQA_DEC_MAX_GQA * 256];
    __shared__ float acc[GQA_DEC_MAX_GQA * 256];
    __shared__ float tile[TILE * 256];
    __shared__ float wt[GQA_DEC_MAX_GQA * TILE];
    __shared__ float s_m[GQA_DEC_MAX_GQA];
    __shared__ float s_su[GQA_DEC_MAX_GQA];
    __shared__ float s_al[GQA_DEC_MAX_GQA];
    __shared__ uint32_t s_have[GQA_DEC_MAX_GQA];
    if (i < hd)
        for (uint32_t g = 0; g < gqa; g++) {
            qs[g * 256 + i] = q[((size_t)(kh * gqa + g)) * hd + i];
            acc[g * 256 + i] = 0.f;
        }
    if (i < GQA_DEC_MAX_GQA) {
        s_m[i] = 0.f;
        s_su[i] = 0.f;
        s_al[i] = 1.f;
        s_have[i] = 0;
    }
    __syncthreads();

    for (uint32_t base = p0; base < p1; base += TILE) {
        uint32_t lim = p1 - base;
        if (lim > TILE) lim = TILE;
        uint32_t q8b = (hd / 32u) * 34u;
        if constexpr (KIND == 2) {
            /* One 34-byte quant block per thread: TILE*(hd/32) = 256 work
             * items, so with TILE=32 and 256 threads each thread dequants a
             * single 32-value block instead of a whole row serially. */
            const uint32_t nb = hd / 32u;
            const uint32_t pp = i / nb, bb = i % nb;
            if (pp < lim) {
                int p = INDEXED ? idx[base + pp] : (int)(base + pp);
                float *dst = tile + (size_t)pp * hd + bb * 32u;
                if (p < 0) {
                    for (uint32_t j = 0; j < 32; j++) dst[j] = 0.f;
                } else {
                    const uint8_t *src =
                        kcache + ((size_t)p * n_kvh + kh) * q8b + bb * 34u;
                    const float sc = f16d(src);
                    const int8_t *qs8 = (const int8_t *)(src + 2);
                    for (uint32_t j = 0; j < 32; j++)
                        dst[j] = sc * (float)qs8[j];
                }
            }
        } else {
            uint32_t tile_n = TILE * hd;
            for (uint32_t e = i; e < tile_n; e += blockDim.x) {
                uint32_t pp = e / hd;
                uint32_t dd = e - pp * hd;
                float v = 0.f;
                int p = -1;
                if (pp < lim)
                    p = INDEXED ? idx[base + pp] : (int)(base + pp);
                if (p >= 0) {
                    const uint8_t *elt =
                        kcache + (((size_t)p * n_kvh + kh) * hd + dd) *
                                     (KIND == 1 ? 2u : 4u);
                    v = KIND == 1 ? __half2float(*(const __half *)elt)
                                  : *(const float *)elt;
                }
                tile[e] = __float2half(v);
            }
        }
        __syncthreads();
        /* Warp-per-position scores for all heads: 8 warps cover TILE=32
         * positions (4 each); every lane reduces its head dot by shuffle —
         * no block barriers inside the head loop. */
        uint32_t wid = i >> 5, lane = i & 31;
        for (uint32_t pos = wid; pos < TILE; pos += 8) {
            int p = -1;
            if (pos < lim)
                p = INDEXED ? idx[base + pos] : (int)(base + pos);
            const float *kk = tile + (size_t)pos * hd;
            for (uint32_t g = 0; g < gqa; g++) {
                float a = 0.f;
                if (p >= 0) {
                    const float *qh = qs + g * 256;
                    for (uint32_t d = lane; d < hd; d += 32)
                        a += qh[d] * kk[d];
                }
                for (uint32_t o = 16; o; o >>= 1)
                    a += __shfl_down_sync(~0ull, a, o);
                if (!lane) wt[g * TILE + pos] = p >= 0 ? a * scale : -1.0e30f;
            }
        }
        __syncthreads();
        /* Warp-per-head online softmax update over the tile scores. */
        for (uint32_t g = wid; g < gqa; g += 8) {
            float mx = -1.0e30f;
            for (uint32_t p = lane; p < lim; p += 32)
                mx = fmaxf(mx, wt[g * TILE + p]);
            for (uint32_t o = 16; o; o >>= 1)
                mx = fmaxf(mx, __shfl_down_sync(~0ull, mx, o));
            mx = __shfl_sync(~0ull, mx, 0);
            int live = mx > -1.0e20f;
            float mo = s_m[g], nm = mo, alpha = 1.f;
            if (live) {
                if (!s_have[g]) {
                    nm = mx;
                    alpha = 0.f;
                } else if (mx > mo) {
                    nm = mx;
                    alpha = expf(mo - nm);
                }
            }
            float su = 0.f;
            for (uint32_t p = lane; p < lim; p += 32) {
                float w = (live && wt[g * TILE + p] > -1.0e20f)
                              ? expf(wt[g * TILE + p] - nm)
                              : 0.f;
                wt[g * TILE + p] = w;
                su += w;
            }
            for (uint32_t o = 16; o; o >>= 1)
                su += __shfl_down_sync(~0ull, su, o);
            if (!lane) {
                s_m[g] = nm;
                s_su[g] = s_su[g] * alpha + su;
                s_al[g] = alpha;
                if (live) s_have[g] = 1;
            }
        }
        __syncthreads();
        if (i < hd)
            for (uint32_t g = 0; g < gqa; g++)
                acc[g * 256 + i] *= s_al[g];
        __syncthreads();
        /* V tile + per-head accumulate (same one-block-per-thread split) */
        if constexpr (KIND == 2) {
            const uint32_t nb = hd / 32u;
            const uint32_t pp = i / nb, bb = i % nb;
            if (pp < lim) {
                int p = INDEXED ? idx[base + pp] : (int)(base + pp);
                float *dst = tile + (size_t)pp * hd + bb * 32u;
                if (p < 0) {
                    for (uint32_t j = 0; j < 32; j++) dst[j] = 0.f;
                } else {
                    const uint8_t *src =
                        vcache + ((size_t)p * n_kvh + kh) * q8b + bb * 34u;
                    const float sc = f16d(src);
                    const int8_t *qs8 = (const int8_t *)(src + 2);
                    for (uint32_t j = 0; j < 32; j++)
                        dst[j] = sc * (float)qs8[j];
                }
            }
        } else {
            uint32_t tile_n = TILE * hd;
            for (uint32_t e = i; e < tile_n; e += blockDim.x) {
                uint32_t pp = e / hd;
                uint32_t dd = e - pp * hd;
                float v = 0.f;
                int p = -1;
                if (pp < lim)
                    p = INDEXED ? idx[base + pp] : (int)(base + pp);
                if (p >= 0) {
                    const uint8_t *elt =
                        vcache + (((size_t)p * n_kvh + kh) * hd + dd) *
                                     (KIND == 1 ? 2u : 4u);
                    v = KIND == 1 ? __half2float(*(const __half *)elt)
                                  : *(const float *)elt;
                }
                tile[e] = __float2half(v);
            }
        }
        __syncthreads();
        if (i < hd)
            for (uint32_t g = 0; g < gqa; g++) {
                float a = 0.f;
                for (uint32_t j = 0; j < lim; j++)
                    a += wt[g * TILE + j] * tile[(size_t)j * hd + i];
                acc[g * 256 + i] += a;
            }
        __syncthreads();
    }
    for (uint32_t g = 0; g < gqa; g++) {
        if (i < hd) pv[g * (hd + 2u) + 2u + i] = acc[g * 256 + i];
        if (i == 0) {
            pv[g * (hd + 2u)] = s_m[g];
            pv[g * (hd + 2u) + 1u] = s_su[g];
        }
    }
}

/* Merge S partials per head; applies the sigmoid gate exactly like
 * gqa_decode_fast's epilogue. */
__global__ static void gqa_dec_comb_k(const float *part, const float *gate,
                                      float *out, uint32_t n_head,
                                      uint32_t n_kvh, uint32_t hd,
                                      uint32_t S) {
    uint32_t h = blockIdx.x;
    uint32_t gqa = n_head / n_kvh;
    if (!gqa || h >= n_head || hd > 256) return;
    uint32_t kh = h / gqa, g = h % gqa;
    uint32_t i = threadIdx.x;
    __shared__ float ms[GQA_DEC_SPLIT_S], ls[GQA_DEC_SPLIT_S], sh[256];
    if (i < S) {
        const float *pv = part + (((size_t)kh * S + i) * gqa + g) * (hd + 2u);
        ms[i] = pv[0];
        ls[i] = pv[1];
    }
    __syncthreads();
    sh[i] = (i < S) ? ms[i] : -1.0e30f;
    __syncthreads();
    for (uint32_t sp = 128; sp > 0; sp >>= 1) {
        if (i < sp) sh[i] = fmaxf(sh[i], sh[i + sp]);
        __syncthreads();
    }
    float M = sh[0];
    float ws = (i < S && ls[i] > 0.f) ? expf(ms[i] - M) : 0.f;
    sh[i] = (i < S) ? ls[i] * ws : 0.f;
    __syncthreads();
    for (uint32_t sp = 128; sp > 0; sp >>= 1) {
        if (i < sp) sh[i] += sh[i + sp];
        __syncthreads();
    }
    float wsum = sh[0];
    if (i < S) ms[i] = ws;
    __syncthreads();
    float inv = wsum > 0.f ? 1.f / wsum : 0.f;
    if (i < hd) {
        float a = 0.f;
        for (uint32_t sp = 0; sp < S; sp++) {
            const float *pv =
                part + (((size_t)kh * S + sp) * gqa + g) * (hd + 2u);
            a += ms[sp] * pv[2u + i];
        }
        float gv = 1.f / (1.f + expf(-gate[(size_t)h * hd + i]));
        out[(size_t)h * hd + i] = a * inv * gv;
    }
}

/* Prefill attention: grid (n_kvh, n_tok), one block runs all gqa query heads
 * of one token against one KV head. The 32-position K/V tile is dequantized
 * once into shared and reused by every head (gqa_decode_fast re-reads it per
 * head). Per-head accumulators live in registers: thread i owns dim i of all
 * heads. Same online-softmax semantics as gqa_decode_fast. */
template <int KIND, bool INDEXED>
__global__ static void gqa_prefill_k(const float *q, const uint8_t *kcache,
                                     const uint8_t *vcache, const float *gate,
                                     float *out, uint32_t n_head,
                                     uint32_t n_kvh, uint32_t hd,
                                     uint32_t kv_base, uint32_t max_kv,
                                     float scale, const int32_t *idx,
                                     uint32_t width) {
    constexpr uint32_t TILE = 32, GQA = 12;
    uint32_t kh = blockIdx.x;
    uint32_t t = blockIdx.y;
    uint32_t gqa = n_head / n_kvh;
    if (kh >= n_kvh || gqa == 0 || gqa > GQA || hd == 0 || hd > 256) return;
    uint32_t n_kv = kv_base + t + 1;
    if (n_kv > max_kv) n_kv = max_kv;
    uint32_t n_pos = INDEXED ? width : n_kv;
    uint32_t i = threadIdx.x;
    uint32_t w = i >> 5, lane = i & 31;

    __shared__ float qs[GQA * 256];
    __shared__ __half tile[TILE * 256];
    __shared__ float wt[GQA * TILE];
    __shared__ float dots[GQA * TILE];
    __shared__ float mm[GQA], ssum[GQA], al[GQA];

    for (uint32_t e = i; e < gqa * hd; e += blockDim.x) {
        uint32_t g = e / hd, d = e - g * hd;
        qs[e] = q[((size_t)t * n_head + kh * gqa + g) * hd + d];
    }
    if (i < GQA) { mm[i] = -1.0e30f; ssum[i] = 0.f; }
    float acc[GQA];
    for (uint32_t g = 0; g < GQA; g++) acc[g] = 0.f;
    __syncthreads();

    uint32_t q8b = (hd / 32u) * 34u;
    for (uint32_t base = 0; base < n_pos; base += TILE) {
        uint32_t lim = n_pos - base;
        if (lim > TILE) lim = TILE;
        if constexpr (KIND == 2) {
            /* warp per position: lanes split hd (one element per 34B
             * q8_0 block lane-step) — the old i<lim path left 224 threads
             * idle while each of 32 threads dequantized a whole row. */
            for (uint32_t pos = w; pos < lim; pos += blockDim.x >> 5) {
                int p = INDEXED ? idx[(size_t)t * width + base + pos]
                                : (int)(base + pos);
                const uint8_t *src =
                    p >= 0 ? kcache + ((size_t)p * n_kvh + kh) * q8b
                           : nullptr;
                __half *dst = tile + (size_t)pos * hd;
                for (uint32_t d = lane; d < hd; d += 32) {
                    float v = 0.f;
                    if (src) {
                        const uint8_t *b = src + (d >> 5) * 34u;
                        v = f16d(b) * (float)(const int8_t)b[2 + (d & 31u)];
                    }
                    dst[d] = __float2half(v);
                }
            }
        } else {
            uint32_t tile_n = TILE * hd;
            for (uint32_t e = i; e < tile_n; e += blockDim.x) {
                uint32_t pp = e / hd;
                uint32_t dd = e - pp * hd;
                float v = 0.f;
                int p = -1;
                if (pp < lim)
                    p = INDEXED ? idx[(size_t)t * width + base + pp]
                                : (int)(base + pp);
                if (p >= 0) {
                    const uint8_t *elt =
                        kcache + (((size_t)p * n_kvh + kh) * hd + dd) *
                                     (KIND == 1 ? 2u : 4u);
                    v = KIND == 1 ? __half2float(*(const __half *)elt)
                                  : *(const float *)elt;
                }
                tile[e] = __float2half(v);
            }
        }
        __syncthreads();
        /* 8 warps cover the gqa*32 (g,pos) dots; lanes split hd, shfl-reduce. */
        for (uint32_t pr = w; pr < gqa * TILE; pr += 8) {
            uint32_t g = pr / TILE, pos = pr - g * TILE;
            float d = -1.0e30f;
            if (pos < lim) {
                int p = INDEXED ? idx[(size_t)t * width + base + pos]
                                : (int)(base + pos);
                if (p >= 0) {
                    d = 0.f;
                    const __half *kk = tile + (size_t)pos * hd;
                    const float *qq = qs + (size_t)g * hd;
                    for (uint32_t dd = lane; dd < hd; dd += 32)
                        d += qq[dd] * __half2float(kk[dd]);
                    for (uint32_t o = 16; o; o >>= 1)
                        d += __shfl_down(d, o);
                    if (lane == 0) d *= scale;
                }
            }
            if (lane == 0) dots[pr] = d;
        }
        __syncthreads();
        /* One warp per head (12 heads over 8 warps): tile max, online update,
         * weights, weight sum. Scalars shared via mm/ssum/al. */
        for (uint32_t g = w; g < gqa; g += 8) {
            float dv = (lane < TILE && lane < lim) ? dots[g * TILE + lane]
                                                   : -1.0e30f;
            float tm = dv;
            for (uint32_t o = 16; o; o >>= 1)
                tm = fmaxf(tm, __shfl_down(tm, o));
            tm = __shfl(tm, 0);
            int live = tm > -1.0e20f;
            float m = mm[g], nm = m, alpha = 1.f;
            if (live) {
                if (tm > m) {
                    nm = tm;
                    alpha = expf(m - nm);
                }
            }
            float wv = (live && dv > -1.0e20f) ? expf(dv - nm) : 0.f;
            if (lane < TILE) wt[g * TILE + lane] = wv;
            float ts = wv;
            for (uint32_t o = 16; o; o >>= 1)
                ts += __shfl_down(ts, o);
            if (lane == 0) {
                mm[g] = nm;
                ssum[g] = ssum[g] * alpha + ts;
                al[g] = alpha;
            }
        }
        __syncthreads();
        if constexpr (KIND == 2) {
            for (uint32_t pos = w; pos < lim; pos += blockDim.x >> 5) {
                int p = INDEXED ? idx[(size_t)t * width + base + pos]
                                : (int)(base + pos);
                const uint8_t *src =
                    p >= 0 ? vcache + ((size_t)p * n_kvh + kh) * q8b
                           : nullptr;
                __half *dst = tile + (size_t)pos * hd;
                for (uint32_t d = lane; d < hd; d += 32) {
                    float v = 0.f;
                    if (src) {
                        const uint8_t *b = src + (d >> 5) * 34u;
                        v = f16d(b) * (float)(const int8_t)b[2 + (d & 31u)];
                    }
                    dst[d] = __float2half(v);
                }
            }
        } else {
            uint32_t tile_n = TILE * hd;
            for (uint32_t e = i; e < tile_n; e += blockDim.x) {
                uint32_t pp = e / hd;
                uint32_t dd = e - pp * hd;
                float v = 0.f;
                int p = -1;
                if (pp < lim)
                    p = INDEXED ? idx[(size_t)t * width + base + pp]
                                : (int)(base + pp);
                if (p >= 0) {
                    const uint8_t *elt =
                        vcache + (((size_t)p * n_kvh + kh) * hd + dd) *
                                     (KIND == 1 ? 2u : 4u);
                    v = KIND == 1 ? __half2float(*(const __half *)elt)
                                  : *(const float *)elt;
                }
                tile[e] = __float2half(v);
            }
        }
        __syncthreads();
        /* acc[g] = acc[g]*alpha[g] + sum_j wt[g][j]*tile[j][i] for dim i. */
        if (i < hd) {
            for (uint32_t g = 0; g < gqa; g++) {
                float a = acc[g] * al[g];
                const float *wrow = wt + g * TILE;
                for (uint32_t j = 0; j < lim; j++)
                    a += wrow[j] * __half2float(tile[(size_t)j * hd + i]);
                acc[g] = a;
            }
        }
        __syncthreads();
    }
    if (i < hd) {
        for (uint32_t g = 0; g < gqa; g++) {
            uint32_t h = kh * gqa + g;
            float inv = ssum[g] > 0.f ? 1.f / ssum[g] : 0.f;
            float gv = 1.f / (1.f + expf(-gate[((size_t)t * n_head + h) * hd + i]));
            out[((size_t)t * n_head + h) * hd + i] = acc[g] * inv * gv;
        }
    }
}

__global__ static void f32_to_f16_k(const float *src, __half *dst, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

extern "C" bool q4_hip_f32_to_f16(void *dst_f16, const float *src, uint64_t n) {
    if (!g_ok || !dst_f16 || !src || !n) return false;
    uint32_t block = 256;
    uint32_t grid = (uint32_t)((n + block - 1) / block);
    f32_to_f16_k<<<grid, block, 0, g_str>>>(src, (__half *)dst_f16, n);
    return hipGetLastError() == hipSuccess;
}

__global__ static void f32_to_q8_k(const float *src, uint8_t *dst, uint64_t nblocks) {
    uint64_t b = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const float *x = src + b * 32u;
    float amax = 0.f;
    for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(x[i]));
    float d = amax / 127.f;
    float id = d > 0.f ? 1.f / d : 0.f;
    uint8_t *o = dst + b * 34u;
    __half h = __float2half(d);
    memcpy(o, &h, 2);
    for (int i = 0; i < 32; i++) {
        int q = (int)rintf(x[i] * id);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        o[2 + i] = (uint8_t)(int8_t)q;
    }
}

extern "C" bool q4_hip_f32_to_q8(void *dst_q8, const float *src, uint64_t n) {
    if (!g_ok || !dst_q8 || !src || n == 0 || (n % 32u) != 0) return false;
    uint64_t nb = n / 32u;
    uint32_t block = 256;
    uint32_t grid = (uint32_t)((nb + block - 1) / block);
    f32_to_q8_k<<<grid, block, 0, g_str>>>(src, (uint8_t *)dst_q8, nb);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool q4_hip_qsa_decode_n(const float *d_q, const void *d_kcache,
                                    const void *d_vcache, const float *d_gate,
                                    float *d_out, uint32_t n_head, uint32_t n_kvh,
                                    uint32_t hd, uint32_t n_tok, uint32_t kv_base,
                                    uint32_t max_kv, float scale, int kv_f16) {
    if (!g_ok || !d_q || !d_kcache || !d_vcache || !d_gate || !d_out) return false;
    if (n_tok == 0 || hd > 256 || hd == 0 || n_head == 0 || n_kvh == 0) return false;
    (void)hipGetLastError();
    /* n_tok > 1 is prefill: one block per (kv head, token) shares each K/V
     * tile across all its query heads. n_tok == 1 (decode, incl. the graph)
     * keeps gqa_decode_fast. */
    static int qsa_pf = -1;
    if (qsa_pf < 0)
        qsa_pf = !(getenv("Q4_QSA_PF") && getenv("Q4_QSA_PF")[0] == '0');
    int prefill = n_tok > 1 && n_head / n_kvh <= 12 && qsa_pf;
    if (!prefill && n_tok == 1 && g_attn_part && n_kvh <= 4 && hd <= 256 &&
        n_head / n_kvh <= GQA_DEC_MAX_GQA &&
        !(getenv("Q4_DEC_SPLIT") && getenv("Q4_DEC_SPLIT")[0] == '0')) {
        const uint32_t S = g_dec_split_s();
        dim3 g2(n_kvh, S);
#define DEC_SPLIT(K)                                                          \
        do {                                                                  \
            gqa_dec_split_k<K, false><<<g2, 256, 0, g_str>>>(                   \
                (const float *)d_q, (const uint8_t *)d_kcache,                \
                (const uint8_t *)d_vcache, g_attn_part, n_head, n_kvh, hd,    \
                kv_base, max_kv, scale, nullptr, 0, S);                       \
            gqa_dec_comb_k<<<n_head, 256, 0, g_str>>>(g_attn_part, d_gate,    \
                                                    d_out, n_head, n_kvh,   \
                                                    hd, S);                 \
        } while (0)
        if (kv_f16 == 2) DEC_SPLIT(2);
        else if (kv_f16 == 1) DEC_SPLIT(1);
        else DEC_SPLIT(0);
#undef DEC_SPLIT
        if (hipGetLastError() != hipSuccess) return false;
        return true;
    }
    dim3 g(prefill ? n_kvh : n_head, n_tok);
    if (kv_f16 == 2)
        (prefill ? gqa_prefill_k<2, 0> : gqa_decode_fast<2, 0>)<<<g, 256, 0, g_str>>>((const float *)d_q, (const uint8_t *)d_kcache,
                                          (const uint8_t *)d_vcache, d_gate, d_out, n_head,
                                          n_kvh, hd, kv_base, max_kv, scale, nullptr, 0);
    else if (kv_f16 == 1)
        (prefill ? gqa_prefill_k<1, 0> : gqa_decode_fast<1, 0>)<<<g, 256, 0, g_str>>>((const float *)d_q, (const uint8_t *)d_kcache,
                                          (const uint8_t *)d_vcache, d_gate, d_out, n_head,
                                          n_kvh, hd, kv_base, max_kv, scale, nullptr, 0);
    else
        (prefill ? gqa_prefill_k<0, 0> : gqa_decode_fast<0, 0>)<<<g, 256, 0, g_str>>>((const float *)d_q, (const uint8_t *)d_kcache,
                                          (const uint8_t *)d_vcache, d_gate, d_out, n_head,
                                          n_kvh, hd, kv_base, max_kv, scale, nullptr, 0);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) {
        fprintf(stderr, "q4 hip: qsa launch %s\n", hipGetErrorString(e));
        return false;
    }
    return true;
}
extern "C" bool q4_hip_qsa_decode_idx(const float *d_q, const void *d_kcache,
                                      const void *d_vcache, const float *d_gate,
                                      float *d_out, const int32_t *idx,
                                      uint32_t width, uint32_t n_head,
                                      uint32_t n_kvh, uint32_t hd, uint32_t n_tok,
                                      uint32_t max_kv, float scale, int kv_f16) {
    if (!g_ok || !d_q || !d_kcache || !d_vcache || !d_gate || !d_out || !idx)
        return false;
    if (n_tok == 0 || width == 0 || hd > 256 || hd == 0) return false;
    (void)hipGetLastError();
    static int qsa_pf = -1;
    if (qsa_pf < 0)
        qsa_pf = !(getenv("Q4_QSA_PF") && getenv("Q4_QSA_PF")[0] == '0');
    int prefill = n_tok > 1 && n_head / n_kvh <= 12 && qsa_pf;
    if (!prefill && n_tok == 1 && g_attn_part && n_kvh <= 4 && hd <= 256 &&
        n_head / n_kvh <= GQA_DEC_MAX_GQA &&
        !(getenv("Q4_DEC_SPLIT") && getenv("Q4_DEC_SPLIT")[0] == '0')) {
        const uint32_t S = g_dec_split_s();
        dim3 g2(n_kvh, S);
#define DEC_SPLITI(K)                                                         \
        do {                                                                  \
            gqa_dec_split_k<K, true><<<g2, 256, 0, g_str>>>(                    \
                (const float *)d_q, (const uint8_t *)d_kcache,                \
                (const uint8_t *)d_vcache, g_attn_part, n_head, n_kvh, hd,    \
                0, max_kv, scale, idx, width, S);                             \
            gqa_dec_comb_k<<<n_head, 256, 0, g_str>>>(g_attn_part, d_gate,    \
                                                    d_out, n_head, n_kvh,   \
                                                    hd, S);                 \
        } while (0)
        if (kv_f16 == 2) DEC_SPLITI(2);
        else if (kv_f16 == 1) DEC_SPLITI(1);
        else DEC_SPLITI(0);
#undef DEC_SPLITI
        if (hipGetLastError() != hipSuccess) return false;
        return true;
    }
    dim3 g(prefill ? n_kvh : n_head, n_tok);
    if (kv_f16 == 2)
        (prefill ? gqa_prefill_k<2, 1> : gqa_decode_fast<2, 1>)<<<g, 256, 0, g_str>>>((const float *)d_q, (const uint8_t *)d_kcache,
                                          (const uint8_t *)d_vcache, d_gate, d_out,
                                          n_head, n_kvh, hd, 0, max_kv, scale, idx,
                                          width);
    else if (kv_f16 == 1)
        (prefill ? gqa_prefill_k<1, 1> : gqa_decode_fast<1, 1>)<<<g, 256, 0, g_str>>>((const float *)d_q, (const uint8_t *)d_kcache,
                                          (const uint8_t *)d_vcache, d_gate, d_out,
                                          n_head, n_kvh, hd, 0, max_kv, scale, idx,
                                          width);
    else
        (prefill ? gqa_prefill_k<0, 1> : gqa_decode_fast<0, 1>)<<<g, 256, 0, g_str>>>((const float *)d_q, (const uint8_t *)d_kcache,
                                          (const uint8_t *)d_vcache, d_gate, d_out,
                                          n_head, n_kvh, hd, 0, max_kv, scale, idx,
                                          width);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) {
        fprintf(stderr, "q4 hip: qsa idx launch %s\n", hipGetErrorString(e));
        return false;
    }
    return true;
}

/* Mean of `ratio` raw fp16 keys, RMSNorm, RoPE at the block's first position. */
__global__ static void qsa_pool_k(const __half *raw, float *pool, const float *wn,
                                  uint32_t hd, uint32_t ratio, uint32_t b0,
                                  uint32_t n_rot, float rope_base, float eps) {
    uint32_t b = b0 + blockIdx.x;
    uint32_t d = threadIdx.x;
    if (hd != 128 || d >= hd) return;
    __shared__ float v[128];
    __shared__ float red[128];
    float s = 0.f;
    const __half *src = raw + (size_t)b * ratio * hd;
    for (uint32_t j = 0; j < ratio; j++)
        s += __half2float(src[(size_t)j * hd + d]);
    v[d] = s / (float)ratio;
    __syncthreads();
    red[d] = v[d] * v[d];
    __syncthreads();
    for (uint32_t st = 64; st > 0; st >>= 1) {
        if (d < st) red[d] += red[d + st];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / (float)hd + eps);
    v[d] = v[d] * inv * (wn ? wn[d] : 1.f);
    __syncthreads();
    if ((d & 1u) == 0 && d < n_rot) {
        uint32_t p = d >> 1;
        float freq = powf(rope_base, -(float)p / (float)(n_rot / 2u));
        float ang = (float)(b * ratio) * freq;
        float c = cosf(ang), si = sinf(ang);
        float x0 = v[d], x1 = v[d + 1];
        v[d] = x0 * c - x1 * si;
        v[d + 1] = x0 * si + x1 * c;
    }
    __syncthreads();
    pool[(size_t)b * hd + d] = v[d];
}

extern "C" bool q4_hip_qsa_pool(const void *raw_f16, float *pool, const float *wn,
                                uint32_t hd, uint32_t ratio, uint32_t b0,
                                uint32_t b1, uint32_t n_rot, float rope_base,
                                float eps) {
    if (!g_ok || !raw_f16 || !pool || b1 <= b0 || hd != 128 || ratio == 0)
        return false;
    qsa_pool_k<<<b1 - b0, 128, 0, g_str>>>((const __half *)raw_f16, pool, wn, hd, ratio, b0,
                                 n_rot, rope_base, eps);
    return hipGetLastError() == hipSuccess;
}

/* One query per block. Q stays in shared memory; each thread scores a stride of blocks. */
__global__ static void qsa_score_k(const float *q, const float *pool, float *score,
                                   uint32_t n_blocks, uint32_t stride, uint32_t n_head,
                                   uint32_t hd, uint32_t pos0, uint32_t ratio,
                                   const int32_t *pp) {
    uint32_t t = blockIdx.x;
    uint32_t pos = (pp ? (uint32_t)*pp : pos0) + t;
    uint32_t n_vis = ratio ? (pos + 1) / ratio : 0;
    if (pp) n_blocks = n_vis; /* device pos: bound derived on-device */
    uint32_t d = threadIdx.x;
    __shared__ float qh[4][128];
    if (d < hd && n_head <= 4) {
        for (uint32_t h = 0; h < n_head; h++)
            qh[h][d] = q[((size_t)t * n_head + h) * hd + d];
    }
    __syncthreads();
    for (uint32_t b = d; b < n_blocks; b += blockDim.x) {
        float s = -1.0e30f;
        if (b < n_vis) {
            s = 0.f;
            const float *kb = pool + (size_t)b * hd;
            for (uint32_t h = 0; h < n_head; h++) {
                float dot = 0.f;
                const float *qq = qh[h];
                for (uint32_t i = 0; i < hd; i++) dot += qq[i] * kb[i];
                s += fmaxf(dot, 0.f);
            }
        }
        score[(size_t)t * stride + b] = s;
    }
}

extern "C" bool q4_hip_qsa_score(const float *q, const float *pool, float *score,
                                 uint32_t n_tok, uint32_t n_blocks, uint32_t stride,
                                 uint32_t n_head, uint32_t hd, uint32_t pos0,
                                 uint32_t ratio) {
    if (!g_ok || !q || !pool || !score || n_tok == 0 || hd != 128 || n_head == 0 ||
        n_head > 4)
        return false;
    qsa_score_k<<<n_tok, 128, 0, g_str>>>(q, pool, score, n_blocks, stride, n_head, hd, pos0,
                                ratio, nullptr);
    return hipGetLastError() == hipSuccess;
}
/* Device-position single-token variant: n_blocks derived inside the kernel. */
/* Decode (device-pos, graph-captured) sparse indexer. Fixed grids only:
 * n_vis comes from *pp at replay time, so workgroup counts and scratch
 * sizes must not depend on it. */

__device__ static void qsa_write_sel(int32_t *sel, const int *chosen,
                                     int nchosen, uint32_t t, uint32_t pos,
                                     uint32_t n_vis, uint32_t ratio,
                                     uint32_t width);

#define QSA_SD_G 128   /* score workgroups */
#define QSA_SD_TILE 64 /* pooled blocks staged per workgroup pass (32 KiB) */
#define QSA_TK_G 64    /* topk workgroups (contiguous slices for ordering) */

__device__ static uint32_t qsa_tk_xkey(float f) {
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__global__ static void qsa_score_dk(const float *q, const float *pool,
                                    float *score, uint32_t n_head, uint32_t hd,
                                    uint32_t ratio, const int32_t *pp) {
    uint32_t pos = (uint32_t)*pp;
    uint32_t n_vis = ratio ? (pos + 1) / ratio : 0;
    __shared__ float qh[4][128];
    __shared__ float kb[QSA_SD_TILE][128];
    uint32_t d = threadIdx.x;
    if (d < hd)
        for (uint32_t h = 0; h < n_head; h++) qh[h][d] = q[h * hd + d];
    for (uint32_t t0 = blockIdx.x * QSA_SD_TILE; t0 < n_vis;
         t0 += gridDim.x * QSA_SD_TILE) {
        __syncthreads();
        uint32_t nt = n_vis - t0;
        if (nt > QSA_SD_TILE) nt = QSA_SD_TILE;
        for (uint32_t b2 = 0; b2 < nt; b2++)
            for (uint32_t i = d; i < hd; i += blockDim.x)
                kb[b2][i] = pool[((size_t)t0 + b2) * hd + i];
        __syncthreads();
        if (d < nt) {
            float s = 0.f;
            const float *k = kb[d];
            for (uint32_t h = 0; h < n_head; h++) {
                float dot = 0.f;
                const float *qq = qh[h];
                for (uint32_t i = 0; i < hd; i++) dot += qq[i] * k[i];
                s += fmaxf(dot, 0.f);
            }
            score[t0 + d] = s;
        }
    }
}

/* Exact top-need by (key desc, index asc): 4 radix passes over the 8 MSB
 * digits of the order-preserving uint key, then ordered compaction. st:
 * [0] prefix bits above the current digit, [1] (unused), [2] hi_cnt,
 * [3] threshold key, [4] n_gt total. */

__global__ static void qsa_tk_init_k(uint32_t *hist, uint32_t *st) {
    uint32_t d = threadIdx.x;
    if (d < 256) hist[d] = 0;
    if (d == 0) { st[0] = 0; st[2] = 0; }
}

__global__ static void qsa_tk_hist_k(const float *score, const uint32_t *st,
                                     uint32_t *hist, uint32_t ratio,
                                     uint32_t k_blk, const int32_t *pp,
                                     uint32_t pass) {
    uint32_t pos = (uint32_t)*pp;
    uint32_t n_vis = ratio ? (pos + 1) / ratio : 0;
    if (n_vis <= k_blk) return;
    __shared__ uint32_t h[256];
    uint32_t d = threadIdx.x;
    h[d] = 0;
    __syncthreads();
    uint32_t slice = (n_vis + QSA_TK_G - 1) / QSA_TK_G;
    uint32_t b0 = blockIdx.x * slice;
    uint32_t b1 = b0 + slice < n_vis ? b0 + slice : n_vis;
    uint32_t sh = 24 - pass * 8;
    uint32_t pre = st[0];
    for (uint32_t b = b0 + d; b < b1; b += blockDim.x) {
        uint32_t u = qsa_tk_xkey(score[b]);
        if (pass == 0 || (u >> (sh + 8)) == pre)
            atomicAdd(&h[(u >> sh) & 255], 1u);
    }
    __syncthreads();
    if (h[d]) atomicAdd(&hist[d], h[d]);
}

__global__ static void qsa_tk_pick_k(uint32_t *hist, uint32_t *st,
                                     uint32_t ratio, uint32_t k_blk,
                                     const int32_t *pp, uint32_t last) {
    uint32_t pos = (uint32_t)*pp;
    uint32_t n_vis = ratio ? (pos + 1) / ratio : 0;
    uint32_t need = n_vis < k_blk ? n_vis : k_blk;
    uint32_t d = threadIdx.x;
    if (n_vis <= k_blk) {
        if (d < 256) hist[d] = 0;
        return;
    }
    __shared__ int cnt[256];
    cnt[d] = (int)hist[d];
    __syncthreads();
    if (d == 0) {
        uint32_t target = need - st[2];
        int acc = 0;
        uint32_t dg = 0;
        for (int i = 255; i >= 0; i--) {
            if ((uint32_t)(acc + cnt[i]) >= target) {
                dg = (uint32_t)i;
                break;
            }
            acc += cnt[i];
        }
        st[2] += (uint32_t)acc;
        st[0] = (st[0] << 8) | dg;
        if (last) {
            st[3] = st[0];
            st[4] = st[2];
        }
    }
    __syncthreads();
    hist[d] = 0;
}

__global__ static void qsa_tk_cnt_k(const float *score, const uint32_t *st,
                                    uint32_t *cnt, uint32_t ratio,
                                    uint32_t k_blk, const int32_t *pp) {
    uint32_t pos = (uint32_t)*pp;
    uint32_t n_vis = ratio ? (pos + 1) / ratio : 0;
    if (n_vis <= k_blk) return;
    __shared__ uint32_t cg, ce;
    uint32_t d = threadIdx.x;
    if (d == 0) { cg = 0; ce = 0; }
    __syncthreads();
    uint32_t thr = st[3];
    uint32_t slice = (n_vis + QSA_TK_G - 1) / QSA_TK_G;
    uint32_t b0 = blockIdx.x * slice;
    uint32_t b1 = b0 + slice < n_vis ? b0 + slice : n_vis;
    for (uint32_t b = b0 + d; b < b1; b += blockDim.x) {
        uint32_t u = qsa_tk_xkey(score[b]);
        if (u > thr) atomicAdd(&cg, 1u);
        else if (u == thr) atomicAdd(&ce, 1u);
    }
    __syncthreads();
    if (d == 0) {
        cnt[2 * blockIdx.x] = cg;
        cnt[2 * blockIdx.x + 1] = ce;
    }
}

__global__ static void qsa_tk_scan_k(const uint32_t *cnt, uint32_t *off) {
    if (threadIdx.x == 0) {
        uint32_t ag = 0, ae = 0;
        for (uint32_t i = 0; i < QSA_TK_G; i++) {
            off[2 * i] = ag;
            off[2 * i + 1] = ae;
            ag += cnt[2 * i];
            ae += cnt[2 * i + 1];
        }
    }
}

__global__ static void qsa_tk_emit_k(const float *score, const uint32_t *st,
                                     const uint32_t *off, int32_t *chosen,
                                     uint32_t ratio, uint32_t k_blk,
                                     const int32_t *pp) {
    uint32_t pos = (uint32_t)*pp;
    uint32_t n_vis = ratio ? (pos + 1) / ratio : 0;
    uint32_t need = n_vis < k_blk ? n_vis : k_blk;
    if (n_vis <= k_blk) return;
    __shared__ int cg[256], ce[256];
    uint32_t d = threadIdx.x;
    uint32_t thr = st[3];
    uint32_t rem = need - st[4]; /* ==thr slots still free */
    uint32_t slice = (n_vis + QSA_TK_G - 1) / QSA_TK_G;
    uint32_t b0 = blockIdx.x * slice;
    uint32_t b1 = b0 + slice < n_vis ? b0 + slice : n_vis;
    uint32_t sub = (b1 - b0 + blockDim.x - 1) / blockDim.x;
    uint32_t c0 = b0 + d * sub;
    uint32_t c1 = c0 + sub < b1 ? c0 + sub : b1;
    int ng = 0, ne = 0;
    for (uint32_t b = c0; b < c1; b++) {
        uint32_t u = qsa_tk_xkey(score[b]);
        if (u > thr) ng++;
        else if (u == thr) ne++;
    }
    cg[d] = ng;
    ce[d] = ne;
    __syncthreads();
    if (d == 0) {
        int ag = 0, ae = 0;
        for (uint32_t i = 0; i < blockDim.x; i++) {
            int tg = cg[i], te = ce[i];
            cg[i] = ag; ce[i] = ae;
            ag += tg; ae += te;
        }
    }
    __syncthreads();
    uint32_t wg = off[2 * blockIdx.x] + (uint32_t)cg[d];
    uint32_t we = off[2 * blockIdx.x + 1] + (uint32_t)ce[d];
    for (uint32_t b = c0; b < c1; b++) {
        uint32_t u = qsa_tk_xkey(score[b]);
        if (u > thr)
            chosen[wg++] = (int32_t)b;
        else if (u == thr) {
            if (we < rem) chosen[st[4] + we] = (int32_t)b;
            we++;
        }
    }
}

__global__ static void qsa_tk_sel_k(const uint32_t *st, const int32_t *chosen,
                                    int32_t *sel, uint32_t ratio,
                                    uint32_t k_blk, uint32_t width,
                                    const int32_t *pp) {
    uint32_t pos = (uint32_t)*pp;
    uint32_t n_vis = ratio ? (pos + 1) / ratio : 0;
    uint32_t d = threadIdx.x;
    if (n_vis <= k_blk) {
        for (uint32_t w = d; w < width; w += blockDim.x)
            sel[w] = w <= pos ? (int32_t)w : -1;
        return;
    }
    /* Parallel expansion of qsa_write_sel: entry w<need*ratio maps to
     * chosen[w/ratio]*ratio + w%ratio; the tail appends the newest partial
     * block positions n_vis*ratio..pos; the rest is -1 padding. */
    uint32_t need = n_vis < k_blk ? n_vis : k_blk;
    uint32_t ncs = need * ratio;
    for (uint32_t w = d; w < width; w += blockDim.x) {
        int32_t v;
        if (w < ncs) {
            uint32_t i = w / ratio, j = w - i * ratio;
            v = (int32_t)((uint32_t)chosen[i] * ratio + j);
        } else {
            uint32_t tail = n_vis * ratio + (w - ncs);
            v = tail <= pos ? (int32_t)tail : -1;
        }
        sel[w] = v;
    }
}

extern "C" bool q4_hip_qsa_topk_alloc(void) {
    static size_t hc, sc, cc, oc, xc, pc, skc;
    return grow((void **)&g_tk_hist, &hc, 256 * 4) &&
           grow((void **)&g_tk_st, &sc, 8 * 4) &&
           grow((void **)&g_tk_cnt, &cc, 2 * QSA_TK_G * 4) &&
           grow((void **)&g_tk_off, &oc, 2 * QSA_TK_G * 4) &&
           grow((void **)&g_tk_chosen, &xc, 512 * 4) &&
           grow((void **)&g_attn_part, &pc,
                4 * GQA_DEC_SPLIT_S * GQA_DEC_MAX_GQA * (256 + 2) * 4) &&
           grow((void **)&g_sk_part, &skc, GEMV_SK_MAXROWS * GEMV_SK_S * 4);
}

extern "C" bool q4_hip_qsa_score_d(const float *q, const float *pool,
                                   float *score, uint32_t stride,
                                   uint32_t n_head, uint32_t hd,
                                   uint32_t ratio) {
    if (!g_ok || !g_pos_d || !q || !pool || !score || hd != 128 ||
        n_head == 0 || n_head > 4)
        return false;
    qsa_score_dk<<<QSA_SD_G, 128, 0, g_str>>>(q, pool, score, n_head, hd,
                                            ratio, g_pos_d);
    return hipGetLastError() == hipSuccess;
}

__device__ static void qsa_write_sel(int32_t *sel, const int *chosen, int nchosen,
                                      uint32_t t, uint32_t pos, uint32_t n_vis,
                                      uint32_t ratio, uint32_t width) {
    uint32_t w = 0;
    for (int i = 0; i < nchosen && w < width; i++) {
        uint32_t b = (uint32_t)chosen[i];
        for (uint32_t j = 0; j < ratio && w < width; j++)
            sel[(size_t)t * width + w++] = (int32_t)(b * ratio + j);
    }
    uint32_t tail = n_vis * ratio;
    while (tail <= pos && w < width) sel[(size_t)t * width + w++] = (int32_t)tail++;
    while (w < width) sel[(size_t)t * width + w++] = -1;
}

/* Threshold search, then one pass that keeps the blocks at or above it. */
__global__ static void qsa_topk_k(const float *score, int32_t *sel, uint32_t stride,
                                  uint32_t k_blk, uint32_t ratio, uint32_t pos0,
                                  uint32_t width, const int32_t *pp) {
    uint32_t t = blockIdx.x;
    uint32_t tx = threadIdx.x;
    uint32_t pos = (pp ? (uint32_t)*pp : pos0) + t;
    uint32_t n_vis = ratio ? (pos + 1) / ratio : 0;
    uint32_t need = n_vis < k_blk ? n_vis : k_blk;
    const float *row = score + (size_t)t * stride;
    __shared__ float red[128];
    __shared__ int cred[128];
    __shared__ int chosen[512];
    __shared__ int nch;
    __shared__ float bound[2];

    if (need == n_vis) {
        /* Every complete block fits, so the selection is the dense prefix. */
        for (uint32_t w = tx; w < width; w += blockDim.x)
            sel[(size_t)t * width + w] = w <= pos ? (int32_t)w : -1;
        return;
    }

    /* Exact k-th key via 4 passes of 8-bit radix on the order-preserving
     * uint transform; then ordered compaction of u>thr ascending plus
     * u==thr ascending until `need` is full. */
    __shared__ uint32_t hist[256];
    __shared__ uint32_t st[2]; /* [0] prefix [1] hi_cnt */
    if (tx == 0) { st[0] = 0; st[1] = 0; }
    for (int p = 0; p < 4; p++) {
        for (uint32_t i = tx; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        uint32_t sh = 24 - p * 8;
        uint32_t pre = st[0];
        for (uint32_t b = tx; b < n_vis; b += blockDim.x) {
            uint32_t u = qsa_tk_xkey(row[b]);
            if (p == 0 || (u >> (sh + 8)) == pre)
                atomicAdd(&hist[(u >> sh) & 255], 1u);
        }
        __syncthreads();
        if (tx == 0) {
            uint32_t target = need - st[1];
            uint32_t acc = 0, dg = 0;
            for (int i = 255; i >= 0; i--) {
                if (acc + hist[i] >= target) { dg = (uint32_t)i; break; }
                acc += hist[i];
            }
            st[1] += acc;
            st[0] = (st[0] << 8) | dg;
        }
        __syncthreads();
    }
    uint32_t thr = st[0];
    int cgt = 0;
    for (uint32_t b = tx; b < n_vis; b += blockDim.x)
        if (qsa_tk_xkey(row[b]) > thr) cgt++;
    cred[tx] = cgt;
    __syncthreads();
    for (uint32_t stp = blockDim.x >> 1; stp > 0; stp >>= 1) {
        if (tx < stp) cred[tx] += cred[tx + stp];
        __syncthreads();
    }
    int n_gt = cred[0];
    if (tx == 0) nch = 0;
    __syncthreads();
    /* Exact threshold: n_gt < need and n_gt + n_eq >= need always. The
     * emit order matches the pre-radix kernel: blocks above the threshold
     * ascending, then threshold ties ascending. */
    {
        uint32_t slice = (n_vis + blockDim.x - 1) / blockDim.x;
        uint32_t b0 = tx * slice;
        uint32_t b1 = b0 + slice < n_vis ? b0 + slice : n_vis;
        int c = 0;
        for (uint32_t b = b0; b < b1; b++)
            if (qsa_tk_xkey(row[b]) > thr) c++;
        cred[tx] = c;
        __syncthreads();
        if (tx == 0) {
            int acc = 0;
            for (uint32_t i = 0; i < blockDim.x; i++) {
                int t = cred[i]; cred[i] = acc; acc += t;
            }
            nch = acc;
        }
        __syncthreads();
        int off = cred[tx];
        for (uint32_t b = b0; b < b1 && off < (int)k_blk; b++)
            if (qsa_tk_xkey(row[b]) > thr) chosen[off++] = (int)b;
        __syncthreads();
        if (tx == 0) {
            for (uint32_t b = 0; b < n_vis && nch < (int)need; b++)
                if (qsa_tk_xkey(row[b]) == thr) chosen[nch++] = (int)b;
            nch = (int)(nch < (int)need ? nch : need);
        }
        __syncthreads();
        {
            uint32_t ncs = (uint32_t)nch * ratio;
            for (uint32_t w = tx; w < width; w += blockDim.x) {
                int32_t v;
                if (w < ncs) {
                    uint32_t i = w / ratio, j = w - i * ratio;
                    v = (int32_t)((uint32_t)chosen[i] * ratio + j);
                } else {
                    uint32_t tail = n_vis * ratio + (w - ncs);
                    v = tail <= pos ? (int32_t)tail : -1;
                }
                sel[(size_t)t * width + w] = v;
            }
        }
    }
}

extern "C" bool q4_hip_qsa_topk(const float *score, int32_t *sel, uint32_t n_tok,
                                uint32_t n_blocks, uint32_t stride, uint32_t k_blk,
                                uint32_t ratio, uint32_t pos0, uint32_t width) {
    (void)n_blocks;
    if (!g_ok || !score || !sel || n_tok == 0 || k_blk == 0 || k_blk > 512 ||
        width == 0)
        return false;
    qsa_topk_k<<<n_tok, 128, 0, g_str>>>(score, sel, stride, k_blk, ratio, pos0, width,
                               nullptr);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) {
        fprintf(stderr, "q4 hip: qsa topk %s\n", hipGetErrorString(e));
        return false;
    }
    return true;
}
extern "C" bool q4_hip_qsa_topk_d(const float *score, int32_t *sel,
                                  uint32_t stride, uint32_t k_blk,
                                  uint32_t ratio, uint32_t width) {
    if (!g_ok || !g_pos_d || !score || !sel || k_blk == 0 || k_blk > 512 ||
        width == 0)
        return false;
    if (!g_tk_hist)
        return false;
    qsa_tk_init_k<<<1, 256, 0, g_str>>>(g_tk_hist, g_tk_st);
    for (uint32_t p = 0; p < 4; p++) {
        qsa_tk_hist_k<<<QSA_TK_G, 256, 0, g_str>>>(score, g_tk_st, g_tk_hist,
                                                 ratio, k_blk, g_pos_d, p);
        qsa_tk_pick_k<<<1, 256, 0, g_str>>>(g_tk_hist, g_tk_st, ratio, k_blk,
                                            g_pos_d, p == 3);
    }
    qsa_tk_cnt_k<<<QSA_TK_G, 256, 0, g_str>>>(score, g_tk_st, g_tk_cnt, ratio,
                                            k_blk, g_pos_d);
    qsa_tk_scan_k<<<1, 128, 0, g_str>>>(g_tk_cnt, g_tk_off);
    qsa_tk_emit_k<<<QSA_TK_G, 256, 0, g_str>>>(score, g_tk_st, g_tk_off,
                                             g_tk_chosen, ratio, k_blk,
                                             g_pos_d);
    qsa_tk_sel_k<<<1, 128, 0, g_str>>>(g_tk_st, g_tk_chosen, sel, ratio,
                                       k_blk, width, g_pos_d);
    return hipGetLastError() == hipSuccess;
}

/* Pool one indexer block with bounds computed from the device position.
 * Always launched in graphs; self-gates when no block completed. */
__global__ static void qsa_pool_dk(const __half *raw, float *pool,
                                   const float *wn, uint32_t hd,
                                   uint32_t ratio, uint32_t n_rot,
                                   float rope_base, float eps,
                                   const int32_t *pp) {
    uint32_t pos = (uint32_t)*pp;
    uint32_t b0 = pos / ratio;
    uint32_t b1 = (pos + 1) / ratio;
    if (b1 <= b0 || hd != 128 || threadIdx.x >= hd) return;
    uint32_t b = b0;
    uint32_t d = threadIdx.x;
    __shared__ float v[128];
    __shared__ float red[128];
    float s = 0.f;
    const __half *src = raw + (size_t)b * ratio * hd;
    for (uint32_t j = 0; j < ratio; j++)
        s += __half2float(src[(size_t)j * hd + d]);
    v[d] = s / (float)ratio;
    __syncthreads();
    red[d] = v[d] * v[d];
    __syncthreads();
    for (uint32_t st = 64; st > 0; st >>= 1) {
        if (d < st) red[d] += red[d + st];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / (float)hd + eps);
    v[d] = v[d] * inv * (wn ? wn[d] : 1.f);
    __syncthreads();
    if ((d & 1u) == 0 && d < n_rot) {
        uint32_t p = d >> 1;
        float freq = powf(rope_base, -(float)p / (float)(n_rot / 2u));
        float ang = (float)(b * ratio) * freq;
        float c = cosf(ang), si = sinf(ang);
        float x0 = v[d], x1 = v[d + 1];
        v[d] = x0 * c - x1 * si;
        v[d + 1] = x0 * si + x1 * c;
    }
    __syncthreads();
    pool[(size_t)b * hd + d] = v[d];
}
extern "C" bool q4_hip_qsa_pool_d(const void *raw_f16, float *pool,
                                  const float *wn, uint32_t hd, uint32_t ratio,
                                  uint32_t n_rot, float rope_base, float eps) {
    if (!g_ok || !g_pos_d || !raw_f16 || !pool || hd != 128 || ratio == 0)
        return false;
    qsa_pool_dk<<<1, 128, 0, g_str>>>((const __half *)raw_f16, pool, wn, hd,
                                      ratio, n_rot, rope_base, eps, g_pos_d);
    return hipGetLastError() == hipSuccess;
}

/* Store raw indexer key at the device position's slot. */
__global__ static void f32_to_f16_at_k(const float *src, __half *dst0,
                                       uint32_t stride, uint64_t n,
                                       const int32_t *pp) {
    __half *dst = dst0 + (size_t)*pp * stride;
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}
extern "C" bool q4_hip_f32_to_f16_d(void *dst_base, uint32_t stride_elems,
                                    const float *src, uint64_t n) {
    if (!g_ok || !g_pos_d || !dst_base || !src || !n) return false;
    uint32_t block = 256;
    uint32_t grid = (uint32_t)((n + block - 1) / block);
    f32_to_f16_at_k<<<grid, block, 0, g_str>>>(src, (__half *)dst_base,
                                             stride_elems, n, g_pos_d);
    return hipGetLastError() == hipSuccess;
}

/* Append K and V (n floats each) into the caches at the device position.
 * kind: 0 f32, 1 f16, 2 q8-per-32 — mirrors kv_write on the host path. */
__global__ static void kv_append_k(const float *k, const float *v, uint8_t *kc,
                                   uint8_t *vc, uint32_t n, uint32_t bpt,
                                   int kind, const int32_t *pp) {
    size_t off = (size_t)(uint32_t)*pp * bpt;
    uint8_t *dk = kc + off, *dv = vc + off;
    if (kind == 2) {
        uint32_t nb = n / 32u;
        uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
        if (b >= 2 * nb) return;
        uint32_t i = b < nb ? b : b - nb;
        const float *x = (b < nb ? k : v) + (size_t)i * 32u;
        uint8_t *o = (b < nb ? dk : dv) + (size_t)i * 34u;
        float amax = 0.f;
        for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(x[j]));
        float d = amax / 127.f;
        float id = d > 0.f ? 1.f / d : 0.f;
        __half h = __float2half(d);
        memcpy(o, &h, 2);
        for (int j = 0; j < 32; j++) {
            int q = (int)rintf(x[j] * id);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            o[2 + j] = (uint8_t)(int8_t)q;
        }
    } else {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= 2 * n) return;
        const float *x = i < n ? k : v;
        uint32_t j = i < n ? i : i - n;
        uint8_t *o = (i < n ? dk : dv) + (size_t)j * (kind == 1 ? 2u : 4u);
        if (kind == 1) {
            __half h = __float2half(x[j]);
            memcpy(o, &h, 2);
        } else {
            memcpy(o, &x[j], 4);
        }
    }
}
extern "C" bool q4_hip_kv_append_d(void *kcache, void *vcache, const float *k,
                                   const float *v, uint32_t n, uint32_t bpt,
                                   int kind) {
    if (!g_ok || !g_pos_d || !kcache || !vcache || !k || !v || !n || !bpt)
        return false;
    uint32_t unit = kind == 2 ? n / 32u : n;
    if (kind == 2 && (n % 32u)) return false;
    uint32_t grid = (2 * unit + 255) / 256;
    kv_append_k<<<grid, 256, 0, g_str>>>(k, v, (uint8_t *)kcache,
                                         (uint8_t *)vcache, n, bpt, kind,
                                         g_pos_d);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool q4_hip_qsa_decode(const float *d_q, const void *d_kcache,
                                  const void *d_vcache, const float *d_gate,
                                  float *d_out, uint32_t n_head, uint32_t n_kvh,
                                  uint32_t hd, uint32_t n_kv, uint32_t max_kv,
                                  float scale, int kv_f16) {
    if (n_kv == 0) return false;
    return q4_hip_qsa_decode_n(d_q, d_kcache, d_vcache, d_gate, d_out, n_head,
                               n_kvh, hd, 1, n_kv - 1, max_kv, scale, kv_f16);
}

/* PLE batched over T tokens (prefill). Per-(token,stream) gate:
 * dot = sum(kn_c*qn_c)/sqrt(n_embd); gate = sigmoid(copysign(sqrt(max(|dot|,
 * 1e-6)), dot)); kn[t,c,i] <- val[t,i] * gate (kn is dead after the dot). */
__global__ static void ple_gate_k(float *kn, const float *qn, const float *val,
                                  uint32_t n_embd, uint32_t hc) {
    uint32_t c = blockIdx.x, t = blockIdx.y;
    float *k = kn + ((size_t)t * hc + c) * n_embd;
    const float *qv = qn + ((size_t)t * hc + c) * n_embd;
    float d = 0.f;
    for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x)
        d += k[i] * qv[i];
    __shared__ float sh[256], sg;
    sh[threadIdx.x] = d;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float dot = sh[0] * (1.0f / sqrtf((float)n_embd));
        float mag = sqrtf(fmaxf(fabsf(dot), 1e-6f));
        sg = 1.f / (1.f + expf(-copysignf(mag, dot)));
    }
    __syncthreads();
    float gate = sg;
    const float *v = val + (size_t)t * n_embd;
    for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x)
        k[i] = v[i] * gate;
}

extern "C" bool q4_hip_ple_gate(float *kn, const float *qn, const float *val,
                                uint32_t n_embd, uint32_t hc, uint32_t n_tok) {
    if (!g_ok || !kn || !qn || !val || !n_embd || !hc || !n_tok) return false;
    ple_gate_k<<<dim3(hc, n_tok), 256, 0, g_str>>>(kn, qn, val, n_embd, hc);
    return hipGetLastError() == hipSuccess;
}

/* Causal dilated depthwise conv over the batch: token t taps conv-norm rows
 * t - (kern-1-k)*dil, reading the (hist+1)-row state for offsets < 0.
 * co[t,c] = silu(sum_k kw[k + c*kern] * src[t-(kern-1-k)*dil, c]). */
__global__ static void ple_conv_k(float *co, const float *cn,
                                  const float *state, const float *kw,
                                  uint32_t hc_dim, uint32_t kern, uint32_t dil,
                                  uint32_t hist) {
    uint32_t t = blockIdx.x;
    uint32_t c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= hc_dim) return;
    float acc = 0.f;
    for (uint32_t k = 0; k < kern; k++) {
        int64_t o = (int64_t)t - (int64_t)(kern - 1 - k) * (int64_t)dil;
        const float *src = o >= 0 ? cn + (size_t)o * hc_dim
                                  : state + ((size_t)o + hist + 1) * hc_dim;
        acc += kw[k + (size_t)c * kern] * src[c];
    }
    co[(size_t)t * hc_dim + c] = acc / (1.f + expf(-acc));
}

extern "C" bool q4_hip_ple_conv(float *co, const float *cn, const float *state,
                                const float *kw, uint32_t hc_dim, uint32_t kern,
                                uint32_t dil, uint32_t hist, uint32_t n_tok) {
    if (!g_ok || !co || !cn || !state || !kw || !hc_dim || !kern || !dil ||
        !n_tok)
        return false;
    dim3 g(n_tok, (hc_dim + 255) / 256);
    ple_conv_k<<<g, 256, 0, g_str>>>(co, cn, state, kw, hc_dim, kern, dil, hist);
    return hipGetLastError() == hipSuccess;
}

/* New conv state = last H conv-norm rows; positions before the batch come
 * from the tail of the old state. dst is the ping-pong buffer (no overlap). */
__global__ static void ple_state_k(float *dst, const float *old,
                                   const float *cn, uint32_t T, uint32_t hc_dim,
                                   uint32_t H) {
    uint64_t e = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (uint64_t)H * hc_dim) return;
    uint32_t r = (uint32_t)(e / hc_dim), c = (uint32_t)(e % hc_dim);
    int64_t idx = (int64_t)T - (int64_t)H + (int64_t)r;
    dst[e] = idx >= 0 ? cn[(size_t)idx * hc_dim + c]
                      : old[((size_t)T + r) * hc_dim + c];
}

extern "C" bool q4_hip_ple_state(float *dst, const float *old, const float *cn,
                                 uint32_t n_tok, uint32_t hc_dim, uint32_t H) {
    if (!g_ok || !dst || !old || !cn || !hc_dim || !H) return false;
    uint64_t n = (uint64_t)H * hc_dim;
    ple_state_k<<<(unsigned)((n + 255) / 256), 256, 0, g_str>>>(dst, old, cn,
                                                              n_tok, hc_dim, H);
    return hipGetLastError() == hipSuccess;
}

/* Deterministic per-token reduce of expert outputs:
 * dy[tok] += sum_k rw[tok*topk+k] * pd[rsrc[tok*topk+k]]  (rsrc < 0 skipped). */
__global__ static void reduce_expert_k(float *dy, const float *pd,
                                       const int32_t *rsrc, const float *rw,
                                       uint32_t topk, uint32_t n_embd) {
    uint32_t tok = blockIdx.y;
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    float acc = 0.f;
    for (uint32_t k = 0; k < topk; k++) {
        int32_t src = rsrc[tok * topk + k];
        if (src >= 0) acc += rw[tok * topk + k] * pd[(size_t)src * n_embd + i];
    }
    dy[(size_t)tok * n_embd + i] += acc;
}

extern "C" bool q4_hip_expert_reduce(float *d_y, const float *d_pd,
                                     const int32_t *d_rsrc, const float *d_rw,
                                     uint32_t n_tok, uint32_t topk,
                                     uint32_t n_embd) {
    if (!g_ok || !d_y || !d_pd || !d_rsrc || !d_rw || !n_tok || !topk) return false;
    dim3 b(256), g((n_embd + 255u) / 256u, n_tok);
    reduce_expert_k<<<g, b, 0, g_str>>>(d_y, d_pd, d_rsrc, d_rw, topk, n_embd);
    return hipGetLastError() == hipSuccess;
}

/* dy[t*n+i] += sg[t] * dx[t*n+i] */
__global__ static void axpy_rows_k(float *dy, const float *dx, const float *sg,
                                   uint32_t n) {
    uint32_t t = blockIdx.y;
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dy[(size_t)t * n + i] += sg[t] * dx[(size_t)t * n + i];
}

extern "C" bool q4_hip_axpy_rows(float *d_y, const float *d_x, const float *d_sg,
                                 uint32_t n, uint32_t n_tok) {
    if (!g_ok || !d_y || !d_x || !d_sg || !n || !n_tok) return false;
    dim3 b(256), g((n + 255u) / 256u, n_tok);
    axpy_rows_k<<<g, b, 0, g_str>>>(d_y, d_x, d_sg, n);
    return hipGetLastError() == hipSuccess;
}

/* ---- decode-graph support -------------------------------------------------
 * Stream capture on g_str. The decode layer body is a fixed launch chain per
 * layer; capturing it once and replaying removes ~40 launch submissions per
 * layer (GPU was ~10% busy: submission-bound).
 * -------------------------------------------------------------------------*/

extern "C" int q4_hip_graph_begin(void) {
    if (!g_ok || !g_str || g_capturing) return 0;
    /* ThreadLocal: io threads keep enqueuing async H2D on g_copy while we
     * capture g_str; Global mode would abort the capture on their calls. */
    hipError_t e =
        hipStreamBeginCapture(g_str, hipStreamCaptureModeThreadLocal);
    if (e != hipSuccess) {
        static int once = 0;
        if (!once++) fprintf(stderr, "q4: graph begin: %s\n",
                             hipGetErrorString(e));
        return 0;
    }
    g_capturing = 1;
    return 1;
}

extern "C" void *q4_hip_graph_end(void) {
    g_capturing = 0;
    hipGraph_t g = nullptr;
    hipError_t e = hipStreamEndCapture(g_str, &g);
    if (e != hipSuccess || !g) {
        static int once = 0;
        if (!once++) fprintf(stderr, "q4: graph end: %s\n",
                             hipGetErrorString(e));
        /* A capture-invalidating call leaves a sticky last-error on this
         * thread; drain it or the first hipGetLastError check in the
         * eager re-run reports a phantom failure. Worse, on ROCm a failed
         * EndCapture can leave g_str permanently in capture state, making
         * every later launch fail — recreate the stream when that happens. */
        (void)hipGetLastError();
        hipStreamCaptureStatus cs = hipStreamCaptureStatusNone;
        if (hipStreamIsCapturing(g_str, &cs) == hipSuccess &&
            cs != hipStreamCaptureStatusNone) {
            hipStream_t ns = nullptr;
            (void)hipStreamDestroy(g_str);
            if (hipStreamCreateWithFlags(&ns, hipStreamNonBlocking) ==
                hipSuccess) {
                g_str = ns;
            } else {
                g_str = nullptr;
                (void)hipGetLastError();
            }
            static int once2 = 0;
            if (!once2++)
                fprintf(stderr, "q4: graph end: stream stuck capturing — "
                                "recreated (%s)\n", g_str ? "ok" : "failed");
            (void)hipGetLastError(); /* the query itself set last-error */
        }
        return nullptr;
    }
    size_t nn = 0;
    (void)hipGraphGetNodes(g, nullptr, &nn);
    g_last_graph_nodes = (uint32_t)nn;
    hipGraphExec_t ex = nullptr;
    e = hipGraphInstantiate(&ex, g, nullptr, nullptr, 0);
    (void)hipGraphDestroy(g);
    if (e != hipSuccess) {
        static int once = 0;
        if (!once++) fprintf(stderr, "q4: graph instantiate: %s\n",
                             hipGetErrorString(e));
        return nullptr;
    }
    /* Pre-upload node schedules to the device so replay skips most of the
     * per-launch host work. */
    (void)hipGraphUpload(ex, g_str);
    return (void *)ex;
}

extern "C" uint32_t q4_hip_graph_nodes(void *exec) {
    (void)exec;
    return g_last_graph_nodes;
}

extern "C" bool q4_hip_graph_launch(void *exec) {
    if (!exec) return false;
    return hipGraphLaunch((hipGraphExec_t)exec, g_str) == hipSuccess;
}

extern "C" void q4_hip_graph_free(void *exec) {
    if (exec) (void)hipGraphExecDestroy((hipGraphExec_t)exec);
}

/* Pinned-memory sync between the compute stream and the host/CPU expert
 * pool.  Both sides use monotonically increasing counters so nothing is
 * baked into launch arguments: graph replays keep working untouched. */
__global__ static void flag_bump_k(volatile uint32_t *p) { ++*p; }

/* Stream-ordered "routing data is in host memory" marker: enqueued right
 * after the router/x D2H, so the host may consume them while later kernels
 * (shared expert) still run.  Host spins on the flag, one slot per layer. */
extern "C" bool q4_hip_flag_bump(void *flag_d) {
    if (!g_ok || !flag_d) return false;
    flag_bump_k<<<1, 1, 0, g_str>>>((volatile uint32_t *)flag_d);
    return hipGetLastError() == hipSuccess;
}

/* One graph node replacing fill(d_y) + dot_sigmoid + flag_bump in the MoE
 * decode path: zero the routed-merge accumulator, compute the shared-gate
 * scalar sigmoid(sgate . x), then bump the per-layer "routing is in host
 * memory" flag.  Runs right after the router/x D2H copies on the same
 * stream, so the flag still marks copy completion.  d_sgate/flag may be
 * NULL (no sgate tensor / no flag scheme). */
__global__ static void moe_pack_k(float *d_y, const float *d_x,
                                  const float *d_sgate, float *d_dot,
                                  volatile uint32_t *flag, uint32_t n_embd) {
    __shared__ float red[8];
    const uint32_t i = threadIdx.x;
    float acc = 0.f;
    for (uint32_t j = i; j < n_embd; j += 256u) {
        d_y[j] = 0.f;
        if (d_sgate) acc += d_x[j] * d_sgate[j];
    }
    if (d_sgate) {
        for (int o = 16; o; o >>= 1) acc += __shfl_down(acc, o);
        if ((i & 31u) == 0) red[i >> 5] = acc;
        __syncthreads();
        if (i < 8) {
            acc = red[i];
            for (int o = 4; o; o >>= 1) acc += __shfl_down(acc, o);
            if (i == 0) *d_dot = 1.f / (1.f + expf(-acc));
        }
    }
    __syncthreads();
    if (i == 0 && flag) ++*flag;
}
extern "C" bool q4_hip_moe_pack(float *d_y, const float *d_x,
                                const float *d_sgate, float *d_dot,
                                void *flag_d, uint32_t n_embd) {
    if (!g_ok || !d_y || !d_x || !n_embd) return false;
    moe_pack_k<<<1, 256, 0, g_str>>>(d_y, d_x, d_sgate, d_dot,
                                     (volatile uint32_t *)flag_d, n_embd);
    return hipGetLastError() == hipSuccess;
}

/* Fused MoE tail for decode: y += hy; y += sh * sig(dot); then the
 * hyper-connection residual combine — one node for three.  dot already
 * holds the sigmoid scalar (moe_pack / dot_sigmoid); NULL means no shared
 * gate (plain add). */
__global__ static void moe_final_k(float *res, const float *y, const float *hy,
                                   const float *sh, const float *dot,
                                   const float *inj, uint32_t n_embd,
                                   uint32_t hc) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    float v = y[i] + hy[i] + sh[i] * (dot ? *dot : 1.f);
    float inv = 1.f / (float)hc;
    for (uint32_t c = 0; c < hc; c++) {
        float w = 2.f / (1.f + expf(-inj[c] * inv));
        res[(size_t)c * n_embd + i] += v * w;
    }
}
extern "C" bool q4_hip_moe_final(float *res, const float *y, const float *hy,
                                 const float *sh, const float *dot,
                                 const float *inj, uint32_t n_embd,
                                 uint32_t hc) {
    if (!g_ok || !res || !y || !hy || !sh || !inj || !n_embd || !hc)
        return false;
    moe_final_k<<<dim3((n_embd + 255) / 256), 256, 0, g_str>>>(
        res, y, hy, sh, dot, inj, n_embd, hc);
    return hipGetLastError() == hipSuccess;
}

/* The CPU-miss wait runs as a HOST callback node inside the stream: a GPU
 * kernel cannot spin on host-registered memory — device reads of mapped
 * sysmem are L2-cached and host writes do not snoop GPU caches (measured:
 * a polling kernel saw done=46 for 45+ s after the host wrote 48, while a
 * just-launched read of expect=48 was fresh).  hipLaunchHostFunc inside a
 * captured graph replays fine, the callback reads pinned host memory with
 * full coherence, and stream order is preserved: downstream merge kernels
 * do not run until it returns. */

typedef struct {
    volatile unsigned long long *done;
    volatile unsigned long long *expect;
    volatile unsigned long long *progress; /* pool-wide monotone floor */
} cpx_wait_ctx;

#define CPX_WAITQ 1024
static cpx_wait_ctx g_cpxq[CPX_WAITQ];
static uint32_t g_cpxq_i;
/* Host-side prof: sum µs, calls, timeouts, (last_seen<<20|expect) */
static unsigned long long g_cpx_us, g_cpx_n, g_cpx_to, g_cpx_ls;

static void cpx_wait_hostfn(void *arg) {
    cpx_wait_ctx *w = (cpx_wait_ctx *)arg;
    struct timespec ts0, tprog, ts;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    tprog = ts0;
    unsigned long long hb = w->progress ? *w->progress : 0;
    unsigned long long last = 0;
    for (;;) {
        /* Re-read expect each pass: the eager path can raise it after this
         * node is queued, and the callback must track the latest target. */
        const unsigned long long e =
            __atomic_load_n(w->expect, __ATOMIC_ACQUIRE);
        last = __atomic_load_n(w->done, __ATOMIC_ACQUIRE);
        if (last >= e) break;
        if (w->progress) {
            unsigned long long now =
                __atomic_load_n(w->progress, __ATOMIC_ACQUIRE);
            if (now != hb) { hb = now; clock_gettime(CLOCK_MONOTONIC, &tprog); }
        }
        clock_gettime(CLOCK_MONOTONIC, &ts);
        /* ~45 s of pool-wide silence = wedged pipeline; still under the
         * host watchdog's 180 s stall trip. */
        if (ts.tv_sec - tprog.tv_sec > 45) {
            __atomic_add_fetch(&g_cpx_to, 1, __ATOMIC_RELAXED);
            __atomic_store_n(&g_cpx_ls, (last << 20) | (e & 0xfffffull),
                             __ATOMIC_RELAXED);
            break;
        }
        _mm_pause();
    }
    clock_gettime(CLOCK_MONOTONIC, &ts);
    __atomic_add_fetch(&g_cpx_n, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&g_cpx_us,
        (unsigned long long)(ts.tv_sec - ts0.tv_sec) * 1000000ull +
        (unsigned long long)(ts.tv_nsec - ts0.tv_nsec) / 1000ull,
        __ATOMIC_RELAXED);
}

extern "C" bool q4_hip_cpx_wait(void *done_h, void *expect_h,
                                void *progress_h) {
    if (!g_ok || !done_h || !expect_h) return false;
    uint32_t i = __atomic_fetch_add(&g_cpxq_i, 1, __ATOMIC_RELAXED) &
                 (CPX_WAITQ - 1);
    g_cpxq[i].done = (volatile unsigned long long *)done_h;
    g_cpxq[i].expect = (volatile unsigned long long *)expect_h;
    g_cpxq[i].progress = (volatile unsigned long long *)progress_h;
    return hipLaunchHostFunc(g_str, cpx_wait_hostfn, &g_cpxq[i]) ==
           hipSuccess;
}

/* Avg wait µs/call (resets cycle/count on read) or -1; calls via *calls. */
extern "C" double q4_hip_cpx_wait_prof(unsigned long long *calls) {
    unsigned long long us = __atomic_exchange_n(&g_cpx_us, 0,
                                                __ATOMIC_RELAXED);
    unsigned long long n = __atomic_exchange_n(&g_cpx_n, 0,
                                               __ATOMIC_RELAXED);
    if (calls) *calls = n;
    if (!n) return -1.0;
    return (double)us / (double)n;
}

/* Cumulative count of cpx wait timeouts (survives profiling reads). */
extern "C" unsigned long long q4_hip_cpx_wait_timeouts(void) {
    return __atomic_load_n(&g_cpx_to, __ATOMIC_RELAXED);
}

/* Last timeout's diagnostic: (seen done)<<20 | (expect & 2^20-1). */
extern "C" unsigned long long q4_hip_cpx_wait_lastseen(void) {
    return __atomic_load_n(&g_cpx_ls, __ATOMIC_RELAXED);
}

/* Copies on the compute stream: inside capture they become memcpy nodes;
 * outside capture they behave like any async enqueue (caller must sync). */
extern "C" bool q4_hip_d2h_async(void *h, const void *d, uint64_t n) {
    if (!g_ok || !h || !d || !n) return false;
    return hipMemcpyAsync(h, d, (size_t)n, hipMemcpyDeviceToHost, g_str) ==
           hipSuccess;
}
extern "C" bool q4_hip_h2d_async(void *d, const void *h, uint64_t n) {
    if (!g_ok || !d || !h || !n) return false;
    return hipMemcpyAsync(d, h, (size_t)n, hipMemcpyHostToDevice, g_str) ==
           hipSuccess;
}
extern "C" bool q4_hip_stream_sync(void) {
    if (!g_ok) return false;
    return (g_str ? hipStreamSynchronize(g_str) : hipDeviceSynchronize()) ==
           hipSuccess;
}

/* GPU-side timing: events recorded on g_str bracket a phase; q4_hip_mark_ms
 * drains slot and returns elapsed ms (GPU busy + in-stream idle). Slots:
 * 0 = whole decode window, 1 = graph A, 2 = graph B, 3 = head graph,
 * 4 = eager attn, 5 = eager moe-a, 6 = eager moe-b. */
/* Boundary marker: records an event on g_str. Safe inside graph capture (it
 * becomes an event-record node timestamped at replay). No-op unless
 * Q4_STEP_PROF=1 so call sites need no env plumbing. */
static int mk_env(void) {
    static int en = -1;
    if (en < 0) {
        const char *e = getenv("Q4_STEP_PROF");
        en = e && e[0] == '1';
        /* Pre-create all marker events now: hipEventCreate during stream
         * capture would invalidate the capture. */
        if (en && g_ok && g_str)
            for (int i = 0; i < Q4_MK_SLOTS; i++) {
                (void)hipEventCreate(&g_mk0[i]);
                (void)hipEventCreate(&g_mk1[i]);
            }
    }
    return en;
}
extern "C" void q4_hip_mark_pt(int slot) {
    if (!mk_env() || !g_ok || !g_str || slot < 0 || slot >= Q4_MK_SLOTS)
        return;
    if (!g_mk0[slot]) return;
    hipError_t rc = hipEventRecord(g_mk0[slot], g_str);
    static int dbg = 0;
    if (slot >= 12 && dbg < 8) {
        dbg++;
        fprintf(stderr, "q4: mark_pt slot %d rc=%d cap=%d\n", slot, (int)rc,
                g_capturing);
    }
    if (rc == hipSuccess)
        g_mk_armed[slot] = 2; /* 2 = boundary recorded */
}
/* Elapsed ms between two boundary events; syncs sb first (caller must be
 * outside capture). */
extern "C" double q4_hip_mark_elapsed(int sa, int sb) {
    if (sa < 0 || sb < 0 || sa >= Q4_MK_SLOTS || sb >= Q4_MK_SLOTS) return -1.0;
    if (!g_mk0[sa] || !g_mk0[sb]) return -1.0;
    /* Never-recorded events would make hipEventElapsedTime fail and leave a
     * sticky error that poisons subsequent hipGetLastError checks. */
    if (g_mk_armed[sa] != 2 || g_mk_armed[sb] != 2) {
        static int d0 = 0;
        if (sb >= 12 && d0++ < 6)
            fprintf(stderr, "q4: mark_elapsed %d->%d armed %d/%d\n",
                    sa, sb, g_mk_armed[sa], g_mk_armed[sb]);
        return -1.0;
    }
    hipError_t sr = hipEventSynchronize(g_mk0[sb]);
    if (sr != hipSuccess) {
        static int d1 = 0;
        if (sb >= 12 && d1++ < 6)
            fprintf(stderr, "q4: mark_elapsed %d->%d sync rc=%d\n",
                    sa, sb, (int)sr);
        (void)hipGetLastError(); /* keep profiler errors off the work path */
        return -1.0;
    }
    float ms = 0.f;
    hipError_t er = hipEventElapsedTime(&ms, g_mk0[sa], g_mk0[sb]);
    if (er != hipSuccess) {
        static int d2 = 0;
        if (sb >= 12 && d2++ < 6)
            fprintf(stderr, "q4: mark_elapsed %d->%d elapsed rc=%d\n",
                    sa, sb, (int)er);
        (void)hipGetLastError();
        return -1.0;
    }
    return (double)ms;
}
extern "C" void q4_hip_mark_begin_slot(int slot) {
    if (!g_ok || !g_str || g_capturing || slot < 0 || slot >= Q4_MK_SLOTS)
        return;
    if (!g_mk0[slot]) {
        if (hipEventCreate(&g_mk0[slot]) != hipSuccess ||
            hipEventCreate(&g_mk1[slot]) != hipSuccess) {
            g_mk0[slot] = g_mk1[slot] = nullptr;
            return;
        }
    }
    if (hipEventRecord(g_mk0[slot], g_str) == hipSuccess)
        g_mk_armed[slot] = 1;
    else
        (void)hipGetLastError(); /* profiler must not poison the work path */
}
extern "C" double q4_hip_mark_ms_slot(int slot) {
    if (!g_mk0[slot] || !g_str || g_mk_armed[slot] != 1) return -1.0;
    g_mk_armed[slot] = 0;
    float ms = 0.f;
    if (hipEventRecord(g_mk1[slot], g_str) != hipSuccess ||
        hipEventSynchronize(g_mk1[slot]) != hipSuccess ||
        hipEventElapsedTime(&ms, g_mk0[slot], g_mk1[slot]) != hipSuccess) {
        (void)hipGetLastError();
        return -1.0;
    }
    return (double)ms;
}
extern "C" void q4_hip_mark_begin(void) { q4_hip_mark_begin_slot(0); }
extern "C" double q4_hip_mark_ms(void) { return q4_hip_mark_ms_slot(0); }

__global__ static void axpy_da_k(float *y, const float *x, const float *a,
                                 uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a[0] * x[i];
}

/* y += (*d_alpha) * x  — alpha lives on device so a graph can be replayed
 * without patching kernel args. */
extern "C" bool q4_hip_axpy_dalpha(float *y, const float *x,
                                   const float *d_alpha, uint64_t n) {
    if (!g_ok || !y || !x || !d_alpha || !n) return false;
    dim3 b(256), g((uint32_t)((n + 255) / 256));
    axpy_da_k<<<g, b, 0, g_str>>>(y, x, d_alpha, n);
    return hipGetLastError() == hipSuccess;
}
