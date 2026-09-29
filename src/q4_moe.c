#include "q4.h"

#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

double q4_prof_moe_io;
double q4_prof_moe_d2h;
double q4_prof_moe_gemm;
double q4_prof_moe_cw;
double q4_prof_moe_xd2h;   /* d_x D2H for CPU experts */
double q4_prof_moe_topk;   /* router topk D2H */
double q4_prof_moe_sg;     /* sgate dot + D2H */
double q4_prof_moe_join;   /* cpuex join */
double q4_prof_moe_merg;   /* h_y H2D + axpy */
double q4_prof_moe_ng;     /* routed uses executed on GPU L1 (decode) */
double q4_prof_moe_nc;     /* routed uses dispatched to the CPU pool */
double q4_prof_moe_ncall;  /* hostpart calls (decode layers x tokens) */

static double moe_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

typedef struct {
    int32_t layer;
    uint8_t *router;    /* F32 [n_embd, n_expert] */
    uint8_t *sgate;     /* F32 [n_embd] */
    uint8_t *gate_s;
    uint8_t *up_s;
    uint8_t *down_s;
    const q4_tensor *t_router, *t_sgate, *t_gate_s, *t_up_s, *t_down_s;
} q4_moe_dense;

static q4_moe_dense g_dense = { .layer = -1 };

static void dense_free(q4_moe_dense *d) {
    free(d->router);
    free(d->sgate);
    free(d->gate_s);
    free(d->up_s);
    free(d->down_s);
    memset(d, 0, sizeof(*d));
    d->layer = -1;
}

static uint8_t *load_t(const q4_gguf *g, const q4_tensor *t) {
    if (!t) return NULL;
    uint8_t *p = malloc((size_t)t->nbytes);
    if (!p) return NULL;
    if (!q4_tensor_read(g, t, p, t->nbytes)) {
        free(p);
        return NULL;
    }
    return p;
}

static bool dense_load(const q4_gguf *g, int32_t layer) {
    if (g_dense.layer == layer) return true;
    dense_free(&g_dense);
    char n[Q4_MAX_NAME];
    snprintf(n, sizeof(n), "blk.%d.ffn_gate_inp.weight", layer);
    g_dense.t_router = q4_find_tensor(g, n);
    snprintf(n, sizeof(n), "blk.%d.ffn_gate_inp_shexp.weight", layer);
    g_dense.t_sgate = q4_find_tensor(g, n);
    snprintf(n, sizeof(n), "blk.%d.ffn_gate_shexp.weight", layer);
    g_dense.t_gate_s = q4_find_tensor(g, n);
    snprintf(n, sizeof(n), "blk.%d.ffn_up_shexp.weight", layer);
    g_dense.t_up_s = q4_find_tensor(g, n);
    snprintf(n, sizeof(n), "blk.%d.ffn_down_shexp.weight", layer);
    g_dense.t_down_s = q4_find_tensor(g, n);
    if (!g_dense.t_router || !g_dense.t_gate_s || !g_dense.t_up_s ||
        !g_dense.t_down_s)
        return false;
    g_dense.router = load_t(g, g_dense.t_router);
    g_dense.gate_s = load_t(g, g_dense.t_gate_s);
    g_dense.up_s = load_t(g, g_dense.t_up_s);
    g_dense.down_s = load_t(g, g_dense.t_down_s);
    if (g_dense.t_sgate) g_dense.sgate = load_t(g, g_dense.t_sgate);
    if (!g_dense.router || !g_dense.gate_s || !g_dense.up_s || !g_dense.down_s)
        return false;
    g_dense.layer = layer;
    return true;
}

static bool gemv_t(const q4_tensor *t, const uint8_t *w, const uint8_t *d_w,
                   const float *x, float *y) {
    if (!t || t->n_dims < 2) return false;
    uint64_t ncols = t->ne[0];
    uint64_t nrows = t->ne[1];
    if (q4_hip_ok()) {
        if (d_w) return q4_hip_gemv_dev(t->ggml_type, d_w, nrows, ncols, x, y, 1.f);
        if (w) return q4_hip_gemv(t->ggml_type, w, nrows, ncols, x, y, 1.f);
    }
    if (!w) return false;
    memset(y, 0, (size_t)nrows * sizeof(float));
    return q4_gemv(t->ggml_type, w, nrows, ncols, x, y, 1.f);
}

/* sgate (ffn_gate_inp_shexp) is F32 in some builds and BF16 in others;
 * every consumer wants f32.  The device variant dequantizes the tensor
 * from the gguf and uploads once, cached by tensor pointer. */
static const float *moe_sgate_f32d(const q4_gguf *g, const q4_store *st,
                                   const q4_tensor *t) {
    static const q4_tensor *sk[64];
    static float *sv[64];
    static int sn;
    if (!t || !g || !st) return NULL;
    if (t->ggml_type == Q4_T_F32)
        return (const float *)q4_store_dev(st, t);
    for (int i = 0; i < sn; i++)
        if (sk[i] == t) return sv[i];
    if (sn >= 64) return NULL;
    const uint64_t n = t->ne[0];
    uint8_t *raw = (uint8_t *)malloc((size_t)t->nbytes);
    float *h = (float *)malloc(n * sizeof(float));
    float *d = NULL;
    if (raw && h && q4_tensor_read(g, t, raw, t->nbytes) &&
        q4_dequant_row(t->ggml_type, raw, n, h)) {
        d = q4_hip_malloc(n * sizeof(float));
        if (!d || !q4_hip_h2d(d, h, n * sizeof(float))) d = NULL;
    }
    free(raw);
    free(h);
    if (!d) return NULL;
    sk[sn] = t;
    sv[sn] = d;
    sn++;
    return d;
}

/* Host-side counterpart for the CPU moe_body path. */
static const float *moe_sgate_f32h(const uint8_t *raw, const q4_tensor *t) {
    static const q4_tensor *sk[64];
    static float *sv[64];
    static int sn;
    if (!t || !raw) return NULL;
    if (t->ggml_type == Q4_T_F32) return (const float *)raw;
    for (int i = 0; i < sn; i++)
        if (sk[i] == t) return sv[i];
    if (sn >= 64) return NULL;
    float *h = (float *)malloc(t->ne[0] * sizeof(float));
    if (!h || !q4_dequant_row(t->ggml_type, raw, t->ne[0], h)) {
        free(h);
        return NULL;
    }
    sk[sn] = t;
    sv[sn] = h;
    sn++;
    return h;
}

static bool moe_body(const q4_gguf *g, q4_expert_cache *c, int32_t layer,
                     const float *x, float *y, const uint8_t *router,
                     const q4_tensor *t_router, const uint8_t *d_router,
                     const uint8_t *sgate, const q4_tensor *t_sgate,
                     const uint8_t *gate_s, const q4_tensor *t_gate_s,
                     const uint8_t *d_gate_s, const uint8_t *up_s,
                     const q4_tensor *t_up_s, const uint8_t *d_up_s,
                     const uint8_t *down_s, const q4_tensor *t_down_s,
                     const uint8_t *d_down_s) {
    const uint32_t n_embd = g->n_embd;
    const uint32_t n_ff = g->n_ff_exp ? g->n_ff_exp : 640;
    const uint32_t n_exp = g->n_expert;
    const uint32_t topk = g->n_expert_used;
    const q4_tensor *gate, *up, *down;
    uint64_t og, ou, od, tot;
    if (!q4_expert_parts(g, layer, &gate, &up, &down, &og, &ou, &od, &tot))
        return false;

    float *logits = malloc((size_t)n_exp * sizeof(float));
    float *h = malloc((size_t)n_ff * sizeof(float));
    float *h2 = malloc((size_t)n_ff * sizeof(float));
    float *down_buf = malloc((size_t)n_embd * sizeof(float));
    float *shared = malloc((size_t)n_embd * sizeof(float));
    if (!logits || !h || !h2 || !down_buf || !shared) {
        free(logits); free(h); free(h2); free(down_buf); free(shared);
        return false;
    }
    if (!gemv_t(t_router, router, d_router, x, logits)) {
        free(logits); free(h); free(h2); free(down_buf); free(shared);
        return false;
    }
    q4_softmax(logits, n_exp);
    int32_t ids[64];
    float wts[64];
    if (topk > 64) {
        free(logits); free(h); free(h2); free(down_buf); free(shared);
        return false;
    }
    q4_topk(logits, n_exp, topk, ids, wts, true);
    q4_expert_route_note(c, layer, ids, (int)topk);

    memset(y, 0, (size_t)n_embd * sizeof(float));
    for (uint32_t k = 0; k < topk; k++) {
        /* Partial-resident tail experts are not in the image: touch reports
         * -1, so bounce them straight off the GGUF instead of failing. */
        uint8_t *pk = NULL;
        int64_t nread = q4_expert_cache_touch(c, layer, ids[k]);
        if (nread < 0) {
            pk = malloc((size_t)tot);
            if (!pk || q4_expert_pread(c, layer, ids[k], pk) < 0) {
                free(pk);
                free(logits); free(h); free(h2); free(down_buf); free(shared);
                return false;
            }
            nread = (int64_t)tot;
        }
        const uint8_t *pack = pk ? pk : q4_expert_cache_data(c, layer, ids[k]);
        int32_t slot = q4_expert_cache_slot(c, layer, ids[k]);
        const uint8_t *dpack = NULL;
        if (pack && slot >= 0 && !pk && q4_hip_ok() && q4_hip_experts_dev(0)) {
            if (nread > 0) q4_hip_experts_put((uint32_t)slot, pack, (uint64_t)nread);
            dpack = q4_hip_experts_dev((uint32_t)slot);
        }
        if (!pack) {
            free(logits); free(h); free(h2); free(down_buf); free(shared);
            return false;
        }
        if (!gemv_t(gate, pack + og, dpack ? dpack + og : NULL, x, h)) {
            free(pk);
            free(logits); free(h); free(h2); free(down_buf); free(shared);
            return false;
        }
        q4_silu(h, n_ff);
        if (!gemv_t(up, pack + ou, dpack ? dpack + ou : NULL, x, h2)) {
            free(pk);
            free(logits); free(h); free(h2); free(down_buf); free(shared);
            return false;
        }
        for (uint32_t i = 0; i < n_ff; i++) h[i] *= h2[i];
        if (!gemv_t(down, pack + od, dpack ? dpack + od : NULL, h, down_buf)) {
            free(pk);
            free(logits); free(h); free(h2); free(down_buf); free(shared);
            return false;
        }
        const float w = wts[k];
        for (uint32_t i = 0; i < n_embd; i++) y[i] += w * down_buf[i];
        free(pk);
    }

    if (!gemv_t(t_gate_s, gate_s, d_gate_s, x, h)) {
        free(logits); free(h); free(h2); free(down_buf); free(shared);
        return false;
    }
    q4_silu(h, n_ff);
    if (!gemv_t(t_up_s, up_s, d_up_s, x, h2)) {
        free(logits); free(h); free(h2); free(down_buf); free(shared);
        return false;
    }
    for (uint32_t i = 0; i < n_ff; i++) h[i] *= h2[i];
    if (!gemv_t(t_down_s, down_s, d_down_s, h, shared)) {
        free(logits); free(h); free(h2); free(down_buf); free(shared);
        return false;
    }
    float sg = 1.0f;
    if (sgate && t_sgate && t_sgate->ne[0] == n_embd) {
        const float *gw = moe_sgate_f32h(sgate, t_sgate);
        if (gw) {
            float acc = 0;
            for (uint32_t i = 0; i < n_embd; i++) acc += gw[i] * x[i];
            sg = 1.0f / (1.0f + expf(-acc));
        }
    }
    for (uint32_t i = 0; i < n_embd; i++) y[i] += sg * shared[i];
    free(logits); free(h); free(h2); free(down_buf); free(shared);
    return true;
}

bool q4_moe_layer(const q4_gguf *g, q4_expert_cache *c, int32_t layer,
                  const float *x, float *y) {
    if (!g || !c || !x || !y || layer < 0) return false;
    if (!g->n_embd || !g->n_expert || !g->n_expert_used) return false;
    if (!dense_load(g, layer)) return false;
    return moe_body(g, c, layer, x, y, g_dense.router, g_dense.t_router, NULL,
                    g_dense.sgate, g_dense.t_sgate, g_dense.gate_s,
                    g_dense.t_gate_s, NULL, g_dense.up_s, g_dense.t_up_s, NULL,
                    g_dense.down_s, g_dense.t_down_s, NULL);
}

/* ---- CPU-expert decode path, split so the two GPU halves can be captured
 * into hipGraphs (decode is launch-bound: GPU ~10% busy).
 *   A:  router gemv + shared expert + sgate dot + async D2H of logits/x
 *   host: stream sync, softmax+topk, CPU dispatch, CPU join
 *   B:  expert-merge axpy + sgate axpy (+ hc_combine by the caller)
 * The pinned host buffers are shared file state; one session at a time. */

static float *moe_hx, *moe_hy, *moe_hh, *moe_hrlog;
static float *moe_dhy;                 /* device alias of moe_hy */
static uint32_t moe_hcap;
/* Pinned sync block: bytes [0,256) hold per-layer "routing ready" bump
 * flags written by a GPU kernel after the D2H; bytes 256/264 hold the
 * expect/done seq words published by the CPU expert pool for the in-stream
 * wait kernel; bytes 272/280 hold the devroute want/done job counters
 * (want is bumped by route_dev_k on the GPU, done by the dispatcher
 * thread after join).  All counters are monotonic → graph-replay safe. */
#define MOE_SYNC_BYTES 288
static volatile uint32_t *moe_route_h;   /* u32[64] */
static uint32_t *moe_route_d;
static volatile unsigned long long *moe_cpx_h; /* {expect, done} */
static unsigned long long *moe_cpx_d;
static volatile unsigned long long *moe_dwant_h; /* devroute {want, done} */
static unsigned long long *moe_dwant_d;
static uint32_t moe_route_last[64];
static uint64_t moe_eager_tag;  /* dispatch tag of the in-flight eager job */
static int32_t moe_ids[64];
static float moe_wts[64];
static float *moe_d_sh, *moe_d_dot, *moe_d_down, *moe_dy;
static float *moe_dx, *moe_dh, *moe_dh2;
static const uint8_t *moe_d_sgate;
static const q4_tensor *moe_t_sgate;
static uint32_t moe_nembd;
static int moe_want_cpu = -1;

/* ---- device-resident routing ------------------------------------------
 * The whole MoE layer — router gemv, top-k, hit/miss split, hit-expert
 * exec, CPU-miss wait and the merged tail — is graphable.  route_dev_k
 * publishes CPU misses through a pinned per-layer mailbox; a dispatcher
 * thread feeds them to the CPU pool and bumps the done counter the in-graph
 * wait kernel spins on.  Decoding never leaves the stream. */
/* Layout must match q4_mail_t in q4_rocm.cu.  ids+wts sit in one
 * contiguous 512B region so the GPU can publish them as a warp burst. */
typedef struct {
    volatile unsigned long long seq;   /* bump per job (single writer/layer) */
    volatile unsigned long long done;  /* dispatcher's completion stamp */
    int32_t ids[64];
    float wts[64];
    uint32_t n;                        /* miss count */
    volatile uint32_t err;             /* overwrite/error flag (host-checked) */
    volatile unsigned long long rseq;  /* profile stamp (fire-and-forget) */
    volatile int32_t routed[16];       /* full top-k for the hit histogram */
} moe_mail_t;

static moe_mail_t *moe_mail;        /* pinned[Q4_MAX_LAYER] */
static void *moe_mail_d;            /* device alias */
static float *moe_hyL;              /* pinned per-layer CPU output */
static float *moe_dhyL;             /* device alias */
static int32_t *moe_des;            /* device hit slots [64] */
static float *moe_dew;              /* device hit weights [64] */
static int32_t *moe_dnh;            /* device hit count */
static unsigned long long *moe_dseq; /* per-layer route sequence counters */
static unsigned long long *moe_drseq; /* per-layer profile stamps */
static unsigned long long *moe_dprof; /* route_dev cycles/calls counters */
static pthread_t moe_disp_th;
static _Atomic int moe_disp_on;
static int moe_disp_started;
static const q4_gguf *moe_disp_g;
static q4_expert_cache *moe_disp_c;
static int moe_disp_nl;
static _Atomic int moe_dev_err;
static _Atomic unsigned long long moe_dev_jobs, moe_dev_nmiss;
static _Atomic unsigned long long moe_dev_lat_us;  /* sum of job latency */

void q4_moe_dev_stats(unsigned long long *jobs, unsigned long long *miss,
                      unsigned long long *lat_us) {
    *jobs = atomic_load(&moe_dev_jobs);
    *miss = atomic_load(&moe_dev_nmiss);
    *lat_us = atomic_load(&moe_dev_lat_us);
}

/* Average on-GPU duration of route_dev_k (µs) since the last call.  Reads
 * back the kernel's clock64 counters — call only with the stream idle. */
double q4_moe_route_prof(void) {
    if (!moe_dprof) return -1.0;
    unsigned long long h[2] = { 0, 0 };
    if (!q4_hip_d2h(h, moe_dprof, sizeof(h)) || !h[1]) return -1.0;
    if (!q4_hip_copy_memset(moe_dprof, 0, sizeof(h))) return -1.0;
    q4_hip_copy_wait();
    double khz = (double)q4_hip_clock_khz();
    if (khz <= 0.0) return -1.0;
    return (double)h[0] / (double)h[1] / khz * 1e3;
}

static int moe_cpu_bufs(uint32_t n_embd, uint32_t n_exp, uint32_t topk,
                        uint32_t n_ff);
static int moe_cpu_want(void);

static void *moe_dispatch_loop(void *arg) {
    (void)arg;
    const q4_gguf *g = moe_disp_g;
    q4_expert_cache *c = moe_disp_c;
    unsigned long long last[Q4_MAX_LAYER];
    memset(last, 0, sizeof(last));
    unsigned long long last_r[Q4_MAX_LAYER];
    memset(last_r, 0, sizeof(last_r));
    const uint32_t disp_topk = g->n_expert_used > 16 ? 16 :
                               (g->n_expert_used ? g->n_expert_used : 10);
    /* In-flight slot per layer: jobs overlap in the pool, so the scan
     * publishes as soon as a mailbox entry appears and stamps done as each
     * slot finishes — never blocking on the layer's own compute. */
    int pend_slot[Q4_MAX_LAYER];
    unsigned long long pend_seq[Q4_MAX_LAYER];
    uint64_t pend_tag[Q4_MAX_LAYER]; /* cpuex dispatch tag — slot may be
                                      * retired+republished while pending,
                                      * so poll by tag, not by slot state */
    double pend_t[Q4_MAX_LAYER];
    for (int i = 0; i < Q4_MAX_LAYER; i++) pend_slot[i] = -1;
    int32_t ids[64];
    float wts[64];
    /* Deferred warm-up staging: the scan pushes (layer, ids) here and the
     * ring drains while the CPU pool crunches — staging I/O never delays
     * the next dispatch. */
    int32_t sq_layer[128], sq_ids[128][64];
    uint32_t sq_n[128], sq_r = 0, sq_w = 0;
    while (moe_disp_on) {
        while (sq_r != sq_w) {
            uint32_t i = sq_r & 127;
            /* Never block the drain on a busy io pool: the completion pass
             * below stamps 'done' for the in-graph cpx_wait — parking here
             * starves it until the wait kernel's timeout aborts decode. */
            if (q4_expert_stage_batch_try(c, sq_layer[i], sq_ids[i],
                                          sq_n[i], 1) == -2)
                break;
            sq_r++;
        }
        int did = 0;
        /* Completion pass: retire finished slots, stamp this layer's done —
         * the in-graph wait key is per-layer so interleaved (eager)
         * dispatches can't produce a false pass. */
        for (int L = 0; L < moe_disp_nl && L < Q4_MAX_LAYER; L++) {
            if (pend_slot[L] < 0) continue;
            if (!q4_cpuex_tag_done(pend_slot[L], pend_tag[L])) continue;
            (void)q4_cpuex_join_tag(pend_slot[L], pend_tag[L]);
            atomic_fetch_add(&moe_dev_lat_us,
                             (unsigned long long)(
                                 (moe_now() - pend_t[L]) * 1e6));
            __atomic_store_n(&moe_mail[L].done, pend_seq[L],
                             __ATOMIC_RELEASE);
            pend_slot[L] = -1;
            did = 1;
        }
        (void)q4_cpuex_retire();
        for (int L = 0; L < moe_disp_nl && L < Q4_MAX_LAYER; L++) {
            /* Profile channel: count the full top-k whenever the kernel
             * publishes a new routed list.  Fire-and-forget — never waited
             * on by either side, so a missed bump only thins the histogram. */
            unsigned long long rs = moe_mail[L].rseq;
            if (rs != last_r[L]) {
                int32_t rid[16];
                for (uint32_t k = 0; k < disp_topk; k++)
                    rid[k] = moe_mail[L].routed[k];
                if (moe_mail[L].rseq == rs) {   /* torn: skip this sample */
                    q4_expert_route_note(c, L, rid, (int)disp_topk);
                    last_r[L] = rs;
                }
            }
            unsigned long long s0 = moe_mail[L].seq;
            if (s0 == last[L]) continue;
            /* A +1 step is a fresh job; a bigger jump means a mailbox slot
             * was overwritten unread — a CPU job's output was lost and the
             * merged token was already wrong.  Flag it for the decode loop. */
            if (last[L] && s0 != last[L] + 1)
                atomic_store_explicit(&moe_dev_err, 1,
                                      memory_order_relaxed);
            uint32_t n = moe_mail[L].n;
            if (n > 64) n = 64;
            memcpy(ids, (const void *)moe_mail[L].ids, n * sizeof(int32_t));
            memcpy(wts, (const void *)moe_mail[L].wts, n * sizeof(float));
            if (moe_mail[L].seq != s0) continue;   /* torn: retry next pass */
            did = 1;
            if (!n) { last[L] = s0; continue; }
            uint64_t tag = 0;
            int slot = q4_cpuex_dispatch2(g, c, L, moe_hx, ids, wts, n,
                                          moe_hyL + (size_t)L * moe_nembd,
                                          &tag);
            if (slot < 0) {
                if (slot == -2)
                    atomic_store_explicit(&moe_dev_err, 1,
                                          memory_order_relaxed);
                /* -1: all slots busy — leave last[L] so the scan retries
                 * next pass; the mailbox entry is untouched until the GPU
                 * publishes the next token (which waits on this one). */
                continue;
            }
            last[L] = s0;
            pend_slot[L] = slot;
            pend_seq[L] = s0;
            pend_tag[L] = tag;
            pend_t[L] = moe_now();
            atomic_fetch_add(&moe_dev_jobs, 1);
            atomic_fetch_add(&moe_dev_nmiss, n);
            if (sq_w - sq_r < 128) {
                uint32_t wi = sq_w & 127;
                sq_layer[wi] = L;
                sq_n[wi] = n;
                memcpy(sq_ids[wi], ids, n * sizeof(int32_t));
                sq_w++;
            }
        }
        if (!did && sq_r == sq_w) _mm_pause();
    }
    return NULL;
}

/* Devroute init: device residency table, mailbox, per-layer CPU output
 * staging and the dispatcher thread.  Returns 1 when the whole-layer
 * graphable path may be used. */
static int moe_route_prof; /* Q4_ROUTE_PROF set -> kernel publishes routed[] */
static int moe_dev_init(const q4_gguf *g, q4_expert_cache *c) {
    static int tried;
    if (tried) return moe_disp_started && moe_mail != NULL;
    tried = 1;
    {
        const char *rp = getenv("Q4_ROUTE_PROF");
        moe_route_prof = (rp && rp[0]) ? 1 : 0;
    }
    if (!moe_cpu_bufs(g->n_embd, g->n_expert, g->n_expert_used,
                      g->n_ff_exp ? g->n_ff_exp : 640))
        return 0;
    if (!q4_expert_dev_enable(c)) return 0;
    size_t mail_n = sizeof(moe_mail_t) * Q4_MAX_LAYER;
    moe_mail = malloc(mail_n);
    moe_hyL = malloc((size_t)Q4_MAX_LAYER * g->n_embd * sizeof(float));
    if (!moe_mail || !moe_hyL) goto fail;
    memset(moe_mail, 0, mail_n);
    memset(moe_hyL, 0, (size_t)Q4_MAX_LAYER * g->n_embd * sizeof(float));
    q4_hip_host_register(moe_mail, mail_n);
    q4_hip_host_register(moe_hyL,
                         (size_t)Q4_MAX_LAYER * g->n_embd * sizeof(float));
    moe_mail_d = q4_hip_host_devptr(moe_mail);
    moe_dhyL = (float *)q4_hip_host_devptr(moe_hyL);
    moe_dseq = (unsigned long long *)q4_hip_malloc(
        2 * Q4_MAX_LAYER * sizeof(uint64_t) + 64 * sizeof(int32_t) +
        64 * sizeof(float) + sizeof(int32_t) + 4 + 16);
    if (!moe_mail_d || !moe_dhyL || !moe_dseq || !moe_dwant_d) goto fail;
    moe_des = (int32_t *)(moe_dseq + Q4_MAX_LAYER);
    moe_dew = (float *)(moe_des + 64);
    moe_dnh = moe_des + 128;
    moe_dprof = (unsigned long long *)(moe_dnh + 2);
    moe_drseq = (unsigned long long *)((char *)moe_dprof + 16);
    if (((uintptr_t)moe_dprof & 7) || ((uintptr_t)moe_drseq & 7) ||
        !q4_hip_copy_memset(moe_dseq, 0,
                            Q4_MAX_LAYER * sizeof(uint64_t)) ||
        !q4_hip_copy_memset(moe_dprof, 0, 16) ||
        !q4_hip_copy_memset(moe_drseq, 0,
                            Q4_MAX_LAYER * sizeof(uint64_t)))
        goto fail;
    q4_hip_copy_wait();
    moe_disp_g = g;
    moe_disp_c = c;
    moe_disp_nl = (int)g->n_layer;
    moe_disp_on = 1;
    if (pthread_create(&moe_disp_th, NULL, moe_dispatch_loop, NULL) != 0) {
        moe_disp_on = 0;
        goto fail;
    }
    moe_disp_started = 1;
    fprintf(stderr, "q4: devroute live (mailbox %d slots, devtab %ux%u)\n",
            moe_disp_nl, (unsigned)g->n_layer, (unsigned)g->n_expert);
    return 1;
fail:
    moe_disp_on = 0;
    free(moe_mail); moe_mail = NULL;
    free(moe_hyL); moe_hyL = NULL;
    return 0;
}

int q4_moe_dev_active(void) {
    return moe_disp_started && moe_mail != NULL;
}

/* Stop and join the dispatcher.  Required before the expert cache it reads
 * (moe_disp_c) is freed — a live dispatcher on a freed cache segfaulted
 * teardown after an aborted decode.  A dispatcher blocked inside
 * stage_batch still finishes the in-flight job before noticing the flag. */
void q4_moe_dev_stop(void) {
    if (!moe_disp_started) return;
    moe_disp_on = 0;
    pthread_join(moe_disp_th, NULL);
    moe_disp_started = 0;
}

void q4_moe_dev_warm(const q4_gguf *g, const q4_store *st, int32_t layer) {
    if (!g || !st || layer < 0) return;
    char nm[Q4_MAX_NAME];
    snprintf(nm, sizeof nm, "blk.%d.ffn_gate_inp_shexp.weight", (int)layer);
    const q4_tensor *t = q4_find_tensor(g, nm);
    if (t && t->ggml_type != Q4_T_F32 && t->ne[0] == g->n_embd)
        (void)moe_sgate_f32d(g, st, t);
}

int q4_moe_dev_route(const q4_gguf *g, q4_expert_cache *c) {
    if (!moe_cpu_want() || !g || !c || !g->n_embd || !q4_hip_ok()) return 0;
    const char *e = getenv("Q4_DEVROUTE");
    if (e && e[0] == '0') return 0;
    return moe_dev_init(g, c);
}

static int moe_cpu_want(void) {
    if (moe_want_cpu < 0) {
        const char *ce = getenv("Q4_CPU_EXPERTS");
        moe_want_cpu = !(ce && ce[0] == '0');
    }
    return moe_want_cpu;
}

static int moe_cpu_bufs(uint32_t n_embd, uint32_t n_exp, uint32_t topk,
                        uint32_t n_ff) {
    if (moe_hcap >= n_embd && moe_hh) return 1;
    if (moe_hx) q4_hip_host_unregister(moe_hx);
    if (moe_hy) q4_hip_host_unregister(moe_hy);
    if (moe_hrlog) q4_hip_host_unregister(moe_hrlog);
    if (moe_route_h) q4_hip_host_unregister((void *)moe_route_h);
    if (moe_mail) q4_hip_host_unregister(moe_mail);
    if (moe_hyL) q4_hip_host_unregister(moe_hyL);
    free(moe_hx); free(moe_hy); free(moe_hh); free(moe_hrlog);
    free((void *)moe_route_h);
    free(moe_mail); free(moe_hyL);
    moe_mail = NULL; moe_mail_d = NULL; moe_hyL = NULL; moe_dhyL = NULL;
    moe_route_h = NULL; moe_route_d = NULL;
    moe_cpx_h = NULL; moe_cpx_d = NULL;
    moe_dwant_h = NULL; moe_dwant_d = NULL;
    moe_hx = malloc((size_t)n_embd * 4);
    moe_hy = malloc((size_t)n_embd * 4);
    moe_hh = malloc((size_t)topk * n_ff * 4);
    moe_hrlog = malloc((size_t)n_exp * 4);
    uint8_t *syncb = malloc(MOE_SYNC_BYTES);
    moe_hcap = 0;
    moe_dhy = NULL;
    if (moe_hx && moe_hy && moe_hh && moe_hrlog) {
        q4_hip_host_register(moe_hx, (size_t)n_embd * 4);
        q4_hip_host_register(moe_hy, (size_t)n_embd * 4);
        q4_hip_host_register(moe_hrlog, (size_t)n_exp * 4);
        moe_dhy = (float *)q4_hip_host_devptr(moe_hy);
        if (moe_dhy) moe_hcap = n_embd;
    }
    if (syncb) {
        memset(syncb, 0, MOE_SYNC_BYTES);
        q4_hip_host_register(syncb, MOE_SYNC_BYTES);
        uint8_t *sd = q4_hip_host_devptr(syncb);
        if (sd) {
            moe_route_h = (volatile uint32_t *)syncb;
            moe_route_d = (uint32_t *)sd;
            moe_cpx_h = (volatile unsigned long long *)(syncb + 256);
            moe_cpx_d = (unsigned long long *)(sd + 256);
            moe_dwant_h = (volatile unsigned long long *)(syncb + 272);
            moe_dwant_d = (unsigned long long *)(sd + 272);
            memset(moe_route_last, 0, sizeof(moe_route_last));
            q4_cpuex_syncpin((unsigned long long *)moe_cpx_h,
                             (unsigned long long *)moe_cpx_h + 1);
        } else {
            free(syncb);
        }
    }
    return moe_hcap >= n_embd && moe_dhy != NULL;
}

/* 1 when the CPU-expert path is live (env on + buffers registered); decode
 * uses this to pick the graphable flow. */
int q4_moe_cpu_active(const q4_gguf *g) {
    if (!moe_cpu_want() || !g || !g->n_embd) return 0;
    return moe_cpu_bufs(g->n_embd, g->n_expert, g->n_expert_used,
                        g->n_ff_exp ? g->n_ff_exp : 640);
}

static const uint8_t *moe_devw(const q4_gguf *g, const q4_store *st,
                               int32_t layer, const char *pat,
                               const q4_tensor **t_out) {
    char n[Q4_MAX_NAME];
    snprintf(n, sizeof(n), pat, layer);
    const q4_tensor *t = q4_find_tensor(g, n);
    if (t_out) *t_out = t;
    return t ? q4_store_dev(st, t) : NULL;
}


/* GPU part 1 (graphable): y=0, router gemv, shared expert, sgate dot,
 * async D2H of router logits and d_x into pinned host buffers. */
bool q4_moe_gpu_a(const q4_gguf *g, const q4_store *st, int32_t layer,
                  float *d_x, float *d_y, float *d_ws) {
    const uint32_t n_embd = g->n_embd;
    const uint32_t n_ff = g->n_ff_exp ? g->n_ff_exp : 640;
    const uint32_t n_exp = g->n_expert;
    const q4_tensor *t_router, *t_gate_s, *t_up_s, *t_down_s;
    const uint8_t *d_router = moe_devw(g, st, layer,
                                     "blk.%d.ffn_gate_inp.weight", &t_router);
    const uint8_t *d_gate_s = moe_devw(g, st, layer,
                                     "blk.%d.ffn_gate_shexp.weight", &t_gate_s);
    const uint8_t *d_up_s = moe_devw(g, st, layer,
                                   "blk.%d.ffn_up_shexp.weight", &t_up_s);
    const uint8_t *d_down_s = moe_devw(g, st, layer,
                                     "blk.%d.ffn_down_shexp.weight", &t_down_s);
    if (!d_router || !d_gate_s || !d_up_s || !d_down_s) return false;
    const uint8_t *d_sgate = moe_devw(g, st, layer,
                                    "blk.%d.ffn_gate_inp_shexp.weight",
                                    &moe_t_sgate);
    float *d_rlog = d_ws;
    float *d_h = d_rlog + n_exp;
    float *d_h2 = d_h + n_ff;
    float *d_down = d_h2 + n_ff;
    float *d_sh = d_down + n_embd;
    float *d_dot = d_sh + n_embd;
    /* Batched routed-expert scratch (per-expert rows, ≤64 experts). */
    float *d_hall = d_dot + 16;
    float *d_h2all = d_hall + 64 * n_ff;
    float *d_dall = d_h2all + 64 * n_ff;
    if (!q4_hip_gemv_dd(t_router->ggml_type, d_router, t_router->ne[1],
                        t_router->ne[0], d_x, d_rlog, 1.f))
        return false;
    /* Routing D2H first, then a bump flag: the host only waits for these
     * copies, so the shared expert below stays overlapped with topk,
     * CPU dispatch and the hit-expert launches.  The pack kernel also
     * fills d_y and computes the sgate scalar, saving two more nodes. */
    if (!q4_hip_d2h_async(moe_hrlog, d_rlog, (size_t)n_exp * sizeof(float)) ||
        !q4_hip_d2h_async(moe_hx, d_x, (size_t)n_embd * sizeof(float)))
        return false;
    {
        const float *sg = (d_sgate && moe_t_sgate &&
                           moe_t_sgate->ne[0] == n_embd)
                              ? moe_sgate_f32d(g, st, moe_t_sgate)
                              : NULL;
        if (!q4_hip_moe_pack(d_y, d_x, sg, d_dot,
                             (moe_route_d && layer >= 0 && layer < 64)
                                 ? (void *)(moe_route_d + layer)
                                 : NULL,
                             n_embd))
            return false;
        moe_d_dot = sg ? d_dot : NULL;
    }
    /* shared expert (GPU) while the CPU will do the routed ones: gate+up
     * fused into one multi-seg GEMV, then fused silu*mul. */
    q4_seg_t sh_segs[2] = {
        { d_gate_s, d_h, t_gate_s->ne[1], t_gate_s->ggml_type },
        { d_up_s, d_h2, t_up_s->ne[1], t_up_s->ggml_type },
    };
    if (q4_hip_gemv_m(sh_segs, 2, d_x, n_embd)) {
        if (!q4_hip_silu_mul(d_h, d_h2, n_ff)) return false;
    } else {
        if (!q4_hip_gemv_dd(t_gate_s->ggml_type, d_gate_s, t_gate_s->ne[1],
                            t_gate_s->ne[0], d_x, d_h, 1.f))
            return false;
        if (!q4_hip_silu(d_h, n_ff)) return false;
        if (!q4_hip_gemv_dd(t_up_s->ggml_type, d_up_s, t_up_s->ne[1],
                            t_up_s->ne[0], d_x, d_h2, 1.f))
            return false;
        if (!q4_hip_mul(d_h, d_h, d_h2, n_ff)) return false;
    }
    if (!q4_hip_gemv_dd(t_down_s->ggml_type, d_down_s, t_down_s->ne[1],
                        t_down_s->ne[0], d_h, d_sh, 1.f))
        return false;
    moe_d_sgate = d_sgate;
    moe_d_sh = d_sh;
    moe_d_down = d_dall;
    moe_dy = d_y;
    moe_dx = d_x;
    moe_dh = d_hall;
    moe_dh2 = d_h2all;
    moe_nembd = n_embd;
    q4_hip_mark_pt(11); /* gA end boundary (Q4_STEP_PROF only) */
    return true;
}

/* Whole-layer graphable MoE (device-resident routing).  Runs entirely on
 * the compute stream: router gemv → x D2H → device top-k+slot split →
 * shared expert → hit-expert exec → in-stream wait for CPU misses → merge.
 * The dispatcher thread handles the mailboxed misses.  Requires
 * q4_moe_dev_route() == 1; the caller owns the fallback when this fails. */
bool q4_moe_gpu_dev(const q4_gguf *g, const q4_store *st, q4_expert_cache *c,
                    int32_t layer, float *d_res, const float *d_inj,
                    uint32_t hc, float *d_x, float *d_y, float *d_ws) {
    if (!moe_disp_started || !moe_mail_d || !moe_dhyL || !moe_dwant_d)
        return false;
    if (layer < 0 || layer >= moe_disp_nl || layer >= Q4_MAX_LAYER)
        return false;
    const uint32_t n_embd = g->n_embd;
    const uint32_t n_ff = g->n_ff_exp ? g->n_ff_exp : 640;
    const uint32_t n_exp = g->n_expert;
    const uint32_t topk = g->n_expert_used;
    if (!n_exp || !topk || topk > 64) return false;
    const q4_tensor *t_router, *t_gate_s, *t_up_s, *t_down_s;
    const uint8_t *d_router = moe_devw(g, st, layer,
                                     "blk.%d.ffn_gate_inp.weight", &t_router);
    const uint8_t *d_gate_s = moe_devw(g, st, layer,
                                     "blk.%d.ffn_gate_shexp.weight", &t_gate_s);
    const uint8_t *d_up_s = moe_devw(g, st, layer,
                                   "blk.%d.ffn_up_shexp.weight", &t_up_s);
    const uint8_t *d_down_s = moe_devw(g, st, layer,
                                     "blk.%d.ffn_down_shexp.weight", &t_down_s);
    if (!d_router || !d_gate_s || !d_up_s || !d_down_s) return false;
    const uint8_t *d_sgate = moe_devw(g, st, layer,
                                    "blk.%d.ffn_gate_inp_shexp.weight",
                                    &moe_t_sgate);
    int32_t *devtab = q4_expert_dev_row(c, layer);
    const q4_tensor *tg, *tu, *td;
    uint64_t og, ou, od, tot;
    if (!devtab ||
        !q4_expert_parts(g, layer, &tg, &tu, &td, &og, &ou, &od, &tot))
        return false;
    int arena = q4_expert_cache_arena(c);
    const uint8_t *base = q4_hip_experts_base_a(arena);
    uint64_t stride = q4_hip_experts_stride_a(arena);
    if (!base || !stride) return false;
    float *d_rlog = d_ws;
    float *d_h = d_rlog + n_exp;
    float *d_h2 = d_h + n_ff;
    float *d_down = d_h2 + n_ff;
    float *d_sh = d_down + n_embd;
    float *d_dot = d_sh + n_embd;
    float *d_hall = d_dot + 16;
    float *d_h2all = d_hall + 64 * n_ff;
    float *d_dall = d_h2all + 64 * n_ff;
    if (!q4_hip_gemv_dd(t_router->ggml_type, d_router, t_router->ne[1],
                        t_router->ne[0], d_x, d_rlog, 1.f))
        return false;
    /* x D2H before the mailbox publish: the seq bump implies x landed. */
    if (!q4_hip_d2h_async(moe_hx, d_x, (size_t)n_embd * sizeof(float)))
        return false;
    moe_mail_t *mail = (moe_mail_t *)moe_mail_d + layer;
    if (!q4_hip_route_dev(d_rlog, devtab, moe_des, moe_dew, moe_dnh,
                          mail, moe_dseq + layer,
                          moe_route_prof ? moe_drseq + layer : NULL,
                          moe_dprof, n_exp, topk))
        return false;
    {
        const float *sg = (d_sgate && moe_t_sgate &&
                           moe_t_sgate->ne[0] == n_embd)
                              ? moe_sgate_f32d(g, st, moe_t_sgate)
                              : NULL;
        if (!q4_hip_moe_pack(d_y, d_x, sg, d_dot, NULL, n_embd))
            return false;
        moe_d_dot = sg ? d_dot : NULL;
    }
    q4_seg_t sh_segs[2] = {
        { d_gate_s, d_h, t_gate_s->ne[1], t_gate_s->ggml_type },
        { d_up_s, d_h2, t_up_s->ne[1], t_up_s->ggml_type },
    };
    if (q4_hip_gemv_m(sh_segs, 2, d_x, n_embd)) {
        if (!q4_hip_silu_mul(d_h, d_h2, n_ff)) return false;
    } else {
        if (!q4_hip_gemv_dd(t_gate_s->ggml_type, d_gate_s, t_gate_s->ne[1],
                            t_gate_s->ne[0], d_x, d_h, 1.f))
            return false;
        if (!q4_hip_silu(d_h, n_ff)) return false;
        if (!q4_hip_gemv_dd(t_up_s->ggml_type, d_up_s, t_up_s->ne[1],
                            t_up_s->ne[0], d_x, d_h2, 1.f))
            return false;
        if (!q4_hip_mul(d_h, d_h, d_h2, n_ff)) return false;
    }
    if (!q4_hip_gemv_dd(t_down_s->ggml_type, d_down_s, t_down_s->ne[1],
                        t_down_s->ne[0], d_h, d_sh, 1.f))
        return false;
    /* GPU-resident top-k experts: slots/wts/count live on the device. */
    if (!q4_hip_moe_exec_dev(base, stride, moe_des, moe_dew, moe_dnh,
                             tg->ggml_type, og, tu->ggml_type, ou,
                             td->ggml_type, od, d_x, d_hall, d_h2all, d_dall,
                             d_y, n_ff, tg->ne[0], n_embd))
        return false;
    /* In-stream wait for THIS layer's CPU misses only: a host-callback node
     * spins on the pinned mailbox (device-side polling of host memory is
     * not coherent on gfx1100).  Progress word = the cpuex completion
     * floor: the wait stays alive while ANY job finishes. */
    moe_mail_t *hmail = moe_mail + layer;
    if (!q4_hip_cpx_wait((void *)&hmail->done, (void *)&hmail->seq,
                         moe_cpx_h ? (void *)(moe_cpx_h + 1)
                                   : (void *)&hmail->done))
        return false;
    return q4_hip_moe_final(d_res, d_y, moe_dhyL + (size_t)layer * n_embd,
                          d_sh, moe_d_dot, d_inj, n_embd, hc);
}

/* Token-boundary hook: bumps the expert tick (deferred evictions refill)
 * and reports dispatcher errors.  Returns 0 ok, -1 on mailbox overrun. */
int q4_moe_dev_tick(q4_expert_cache *c) {
    if (!moe_disp_started) return 0;
    if (q4_hip_cpx_wait_timeouts()) {
        /* Self-diagnose the wedge: did the GPU publish seqs, did the
         * dispatcher see them, did cpuex slots ever finish? */
        int stuck = -1;
        if (moe_mail)
            for (int L = 0; L < moe_disp_nl; L++)
                if (moe_mail[L].seq > moe_mail[L].done) { stuck = L; break; }
        unsigned long long ls = q4_hip_cpx_wait_lastseen();
        fprintf(stderr, "q4: cpx_wait timed out — CPU-miss pipeline wedged; "
                        "aborting decode (jobs=%llu miss=%llu err=%d "
                        "stuck-layer=%d seq=%llu done=%llu cpx exp=%llu "
                        "done=%llu kseen=%llu kexp=%llu)\n",
                (unsigned long long)atomic_load(&moe_dev_jobs),
                (unsigned long long)atomic_load(&moe_dev_nmiss),
                atomic_load_explicit(&moe_dev_err, memory_order_relaxed),
                stuck,
                stuck >= 0 ? (unsigned long long)moe_mail[stuck].seq : 0,
                stuck >= 0 ? (unsigned long long)moe_mail[stuck].done : 0,
                moe_cpx_h ? (unsigned long long)moe_cpx_h[0] : 0,
                moe_cpx_h ? (unsigned long long)moe_cpx_h[1] : 0,
                ls >> 20, ls & 0xfffffull);
        q4_cpuex_diag();
        return -1;
    }
    if (atomic_load_explicit(&moe_dev_err, memory_order_relaxed)) {
        fprintf(stderr, "q4: devroute mailbox overrun — aborting decode\n");
        return -1;
    }
    q4_expert_dev_tick(c);
    return 0;
}

static bool moe_gpu_chain(const q4_gguf *g, q4_expert_cache *c, int32_t layer,
                          const float *d_x, float *d_y, const int32_t *ids,
                          const float *wts, uint32_t topk, float *d_h,
                          float *d_h2, float *d_down, uint32_t n_ff,
                          uint32_t n_embd);
static bool moe_gpu_exec(const q4_gguf *g, q4_expert_cache *c, int32_t layer,
                         const float *d_x, float *d_y, const int32_t *ids,
                         const float *wts, uint32_t n, float *d_h,
                         float *d_h2, float *d_down, uint32_t n_ff,
                         uint32_t n_embd);
static bool moe_gpu_stage(const q4_gguf *g, q4_expert_cache *c, int32_t layer,
                          const int32_t *ids, const float *wts, uint32_t n);
static bool moe_gpu_exec_staged(const q4_gguf *g, q4_expert_cache *c,
                                int32_t layer, const float *d_x, float *d_y,
                                float *d_h, float *d_h2, float *d_down,
                                uint32_t n_ff, uint32_t n_embd);

/* Staged hit-exec state.  moe_staged_exec: -1 undecided, 0 = staging broken
 * so hits run eagerly in hostpart, 1 = hits published through the mapped
 * arrays and executed by the per-layer gB graph (or gpu_b on the eager
 * tail). */
static int32_t moe_hslots[64];
static float moe_hwts[64];
static uint32_t moe_nhits;
static int moe_staged_exec = -1;

/* Host middle: drain the stream, softmax+topk, dispatch routed experts on
 * the CPU pool, wait for them.  Returns 1 when the CPU path produced h_y;
 * 0 means "dispatch refused" and the caller must run the eager GPU chain;
 * -1 is a hard error. */
/* Wait for the "routing ready" bump (stream-ordered after the two D2H
 * copies), NOT for the whole stream: the shared expert keeps running while
 * the host does topk and dispatches the CPU pool.  Falls back to a full
 * drain if the flag mechanism is unavailable or stalls. */
#define MOE_ROUTE_SPIN 20000000ull /* ~100 ms of pause, then drain */
static int moe_route_wait(int32_t layer) {
    if (!moe_route_h || layer < 0 || layer >= 64)
        return q4_hip_stream_sync() ? 0 : -1;
    volatile uint32_t *f = moe_route_h + layer;
    uint64_t spins = 0;
    while (*f == moe_route_last[layer]) {
        if (++spins >= MOE_ROUTE_SPIN) {
            if (!q4_hip_stream_sync()) return -1;
            break;
        }
        _mm_pause();
    }
    moe_route_last[layer] = *f;
    return 0;
}

int q4_moe_hostpart(const q4_gguf *g, q4_expert_cache *c, int32_t layer) {
    const uint32_t n_exp = g->n_expert;
    const uint32_t topk = g->n_expert_used;
    const uint32_t n_ff = g->n_ff_exp ? g->n_ff_exp : 640;
    if (moe_route_wait(layer) < 0) return -1;
    /* The previous layer's job may still be live on the pool (its join was
     * deferred to the in-stream wait kernel).  Reap before touching the
     * shared host buffers; normally already done so this is instant. */
    if (!q4_cpuex_reap()) return -1;
    double t0 = moe_now();
    q4_softmax(moe_hrlog, n_exp);
    q4_topk(moe_hrlog, n_exp, topk, moe_ids, moe_wts, true);
    q4_expert_route_note(c, layer, moe_ids, (int)topk);
    q4_prof_moe_topk += moe_now() - t0;
    /* Drain async expert H2D enqueued by earlier layers, then split the top-k:
     * experts already resident in the GPU L1 arena run there (GPU is ~90%
     * idle), misses go to the CPU pool reading the DRAM resident image. */
    int32_t gids[64], cids[64];
    float gwt[64], cwt[64];
    uint32_t ng = 0, nc = 0;
    int hybrid = q4_hip_ok() && q4_expert_cache_arena(c) >= 0 &&
                 moe_dh && moe_dh2;
    if (hybrid) {
        double tcw = moe_now();
        q4_hip_copy_wait();
        q4_prof_moe_cw += moe_now() - tcw;
        q4_expert_uploads_mark(c);
        /* The async stager mutates the L1 map under the io mutex; join it
         * before the lock-free gpu_slot reads below.  Its job was posted a
         * full layer ago, so this is normally a no-op. */
        q4_expert_stage_join(c);
        int arena = q4_expert_cache_arena(c);
        for (uint32_t k = 0; k < topk; k++) {
            int32_t s = q4_expert_gpu_slot(c, layer, moe_ids[k]);
            if (s >= 0 && (uint32_t)s < q4_hip_experts_n_a(arena)) {
                gids[ng] = moe_ids[k];
                gwt[ng++] = moe_wts[k];
            } else {
                cids[nc] = moe_ids[k];
                cwt[nc++] = moe_wts[k];
            }
        }
    } else {
        for (uint32_t k = 0; k < topk; k++) {
            cids[nc] = moe_ids[k];
            cwt[nc++] = moe_wts[k];
        }
    }
    q4_prof_moe_ng += ng;
    q4_prof_moe_nc += nc;
    q4_prof_moe_ncall += 1;
    int disp = 1;
    int slot = -1;
    if (nc) {
        /* -1 means all ring slots are busy (e.g. devroute jobs in flight):
         * retire completions and retry briefly before falling back. */
        for (int t = 0; t < 400; t++) {
            slot = q4_cpuex_dispatch2(g, c, layer, moe_hx, cids, cwt, nc,
                                      moe_hy, &moe_eager_tag);
            if (slot != -1) break;
            (void)q4_cpuex_retire();
            _mm_pause();
        }
        disp = slot >= 0;
    } else {
        memset(moe_hy, 0, (size_t)moe_nembd * sizeof(float));
    }
    if (disp && hybrid) {
        /* Warm the CPU-routed experts into GPU L1 for later tokens while the
         * CPU pool crunches. Hit slots are protected from eviction during
         * the batch; warm puts land via the copy stream. */
        double t1 = moe_now();
        if (ng) q4_expert_protect(c, layer, gids, ng, 1);
        /* Q4_MISS_STAGE=0 skips uploading CPU-routed misses into GPU L1 —
         * with a rotating expert tail the binds churn LRU and burn PCIe
         * without ever re-hitting.  Only valid when L2 is fully resident —
         * otherwise the touch is what pages the expert into DRAM. */
        static int ms = -1;
        if (ms < 0) {
            const char *e = getenv("Q4_MISS_STAGE");
            ms = (!e || e[0] != '0') || !q4_expert_is_resident(c);
        }
        /* Hand the bind batch to the stager thread: it finishes the
         * synchronous stage off the critical path and releases the hit
         * protection itself.  The fallback keeps the old inline order. */
        int posted = 0;
        if (nc && ms)
            posted = q4_expert_stage_async(c, layer, cids, nc, gids, ng);
        if (ng && !posted) q4_expert_protect(c, layer, gids, ng, 0);
        q4_prof_moe_io += moe_now() - t1;
        t1 = moe_now();
        /* Publish hits through the mapped arrays; the exec kernels live in
         * the per-layer gB graph (captured or launched by the caller), which
         * removes ~4 host launches per MoE layer from the decode seam.  If
         * staging can't work at all (no arena, alloc failure on the first
         * token) fall back to eager launches permanently. */
        if (moe_staged_exec != 0) {
            if (moe_gpu_stage(g, c, layer, gids, gwt, ng))
                moe_staged_exec = 1;
            else if (moe_staged_exec > 0)
                return -1;      /* a gB graph already contains the exec */
            else
                moe_staged_exec = 0;
        }
        if (moe_staged_exec == 0 &&
            ng && !moe_gpu_exec(g, c, layer, moe_dx, moe_dy, gids, gwt, ng,
                                moe_dh, moe_dh2, moe_d_down, n_ff, moe_nembd))
            return -1;
        q4_prof_moe_gemm += moe_now() - t1;
    }
    if (!disp) {
        /* Dispatch refused mid-flight: hit-expert work is only staged (or,
         * on the permanent eager path, already launched validly); the tail
         * recomputes all routed experts over the same d_y. */
        return 0;
    }
    t0 = moe_now();
    if (nc && !moe_cpx_d && !q4_cpuex_join_tag(slot, moe_eager_tag))
        return -1;
    /* With the pinned seq block the pool join is an in-stream wait kernel
     * inside gB (or queued by gpu_b on the eager tail) — nothing to do here. */
    q4_prof_moe_join += moe_now() - t0;
    return 1;
}

/* GPU part 2 + hit-exec + pool join + residual combine (decode graph gB).
 * Same function serves graph capture and the eager tail — it only issues
 * launches, and every input arrives through mapped/device words. */
bool q4_moe_gpu_b_hc(const q4_gguf *g, q4_expert_cache *c, int32_t layer,
                     float *d_res, const float *d_inj, uint32_t hc) {
    const uint32_t n_ff = g->n_ff_exp ? g->n_ff_exp : 640;
    if (moe_staged_exec == 1 &&
        !moe_gpu_exec_staged(g, c, layer, moe_dx, moe_dy, moe_dh, moe_dh2,
                             moe_d_down, n_ff, moe_nembd))
        return false;
    if (moe_cpx_h &&
        !q4_hip_cpx_wait((void *)(moe_cpx_h + 1), (void *)moe_cpx_h,
                         (void *)(moe_cpx_h + 1)))
        return false;
    return q4_hip_moe_final(d_res, moe_dy, moe_dhy, moe_d_sh, moe_d_dot,
                            d_inj, moe_nembd, hc);
}

/* GPU part 2 (eager tail of the graphed path): staged hit-exec, pool join,
 * merge CPU expert output, apply shared expert. */
bool q4_moe_gpu_b(const q4_gguf *g, q4_expert_cache *c, int32_t layer) {
    const uint32_t n_ff = g->n_ff_exp ? g->n_ff_exp : 640;
    if (moe_staged_exec == 1 &&
        !moe_gpu_exec_staged(g, c, layer, moe_dx, moe_dy, moe_dh, moe_dh2,
                             moe_d_down, n_ff, moe_nembd))
        return false;
    if (moe_cpx_h &&
        !q4_hip_cpx_wait((void *)(moe_cpx_h + 1), (void *)moe_cpx_h,
                         (void *)(moe_cpx_h + 1)))
        return false;
    const uint32_t n_embd = moe_nembd;
    double t0 = moe_now();
    /* GPU axpy reads the pinned host buffer straight over PCIe; skips the
     * explicit H2D copy and its stream sync. */
    if (!q4_hip_axpy(moe_dy, moe_dhy, 1.f, n_embd)) return false;
    q4_prof_moe_merg += moe_now() - t0;
    if (moe_d_dot)
        return q4_hip_axpy_dalpha(moe_dy, moe_d_sh, moe_d_dot, n_embd);
    return q4_hip_axpy(moe_dy, moe_d_sh, 1.f, n_embd);
}

/* Shared-expert tail only (routed experts were merged by the caller). */
static bool moe_shared_merge(void) {
    if (moe_d_dot)
        return q4_hip_axpy_dalpha(moe_dy, moe_d_sh, moe_d_dot, moe_nembd);
    return q4_hip_axpy(moe_dy, moe_d_sh, 1.f, moe_nembd);
}

/* After a "dispatch refused" hostpart: run the routed experts on GPU using
 * the ids/wts already computed, then merge the shared expert. Never
 * captures; called from graphed decode's fallback and from _dev. */
bool q4_moe_gpu_tail(const q4_gguf *g, q4_expert_cache *c, int32_t layer,
                     float *d_x, float *d_y, float *d_ws) {
    const uint32_t n_embd = g->n_embd;
    const uint32_t n_ff = g->n_ff_exp ? g->n_ff_exp : 640;
    const uint32_t n_exp = g->n_expert;
    const uint32_t topk = g->n_expert_used;
    float *d_rlog = d_ws;
    float *d_h = d_rlog + n_exp + 16;         /* ≤64 experts batched */
    float *d_h2 = d_h + 64 * n_ff;
    float *d_down = d_h2 + 64 * n_ff;
    if (!moe_gpu_chain(g, c, layer, d_x, d_y, moe_ids, moe_wts, topk, d_h,
                       d_h2, d_down, n_ff, n_embd))
        return false;
    return moe_shared_merge();
}

/* Routed-expert GEMV chain for ids whose L1 slots already contain the data.
 * Launches on the compute stream; overlaps CPU expert work. Batched: gate,
 * up, fused activation, down and weighted-sum run as five launches for the
 * whole hit set (before: ~4 small launches per expert). */
static bool moe_gpu_exec(const q4_gguf *g, q4_expert_cache *c, int32_t layer,
                         const float *d_x, float *d_y, const int32_t *ids,
                         const float *wts, uint32_t n, float *d_h,
                         float *d_h2, float *d_down, uint32_t n_ff,
                         uint32_t n_embd) {
    const q4_tensor *gate, *up, *down;
    uint64_t og, ou, od, tot;
    if (!q4_expert_parts(g, layer, &gate, &up, &down, &og, &ou, &od, &tot))
        return false;
    int arena = q4_expert_cache_arena(c);
    const uint8_t *base = q4_hip_experts_base_a(arena);
    uint64_t stride = q4_hip_experts_stride_a(arena);
    uint32_t na = q4_hip_experts_n_a(arena);
    if (!base || !stride) return false;
    int32_t slots[64];
    float wt[64];
    for (uint32_t k = 0; k < n; k++) {
        int32_t slot = q4_expert_cache_slot(c, layer, ids[k]);
        if (slot < 0 || (uint32_t)slot >= na) return false;
        slots[k] = slot;
        wt[k] = wts[k];
    }
    if (q4_hip_moe_exec(base, stride, slots, wt, n, gate->ggml_type, og,
                        up->ggml_type, ou, down->ggml_type, od, d_x, d_h,
                        d_h2, d_down, d_y, n_ff, gate->ne[0], n_embd))
        return true;
    /* Unsupported weight type: per-expert fallback. */
    const uint8_t *dpacks[64];
    for (uint32_t k = 0; k < n; k++)
        dpacks[k] = base + (size_t)slots[k] * stride;
    int use_fuse = (gate->ggml_type == Q4_T_Q4_K &&
                    up->ggml_type == Q4_T_Q4_K);
    for (uint32_t k = 0; k < n; k++) {
        const uint8_t *dpack = dpacks[k];
        if (use_fuse) {
            if (!q4_hip_gateup_q4k(dpack + og, dpack + ou, d_x, d_h, n_ff,
                                   n_embd))
                return false;
        } else {
            if (!q4_hip_gemv_dd(gate->ggml_type, dpack + og, gate->ne[1],
                                gate->ne[0], d_x, d_h, 1.f))
                return false;
            if (!q4_hip_silu(d_h, n_ff)) return false;
            if (!q4_hip_gemv_dd(up->ggml_type, dpack + ou, up->ne[1],
                                up->ne[0], d_x, d_h2, 1.f))
                return false;
            if (!q4_hip_mul(d_h, d_h, d_h2, n_ff)) return false;
        }
        if (!q4_hip_gemv_dd(down->ggml_type, dpack + od, down->ne[1],
                            down->ne[0], d_h, d_down, 1.f))
            return false;
        if (!q4_hip_axpy(d_y, (float *)d_down, wts[k], n_embd)) return false;
    }
    return true;
}

/* Resolve ids→L1 slots and publish slots/weights/count through the mapped
 * staging arrays.  The hit-exec kernels read those words at exec time, so
 * once captured into gB the same launches replay correctly every token. */
static bool moe_gpu_stage(const q4_gguf *g, q4_expert_cache *c, int32_t layer,
                          const int32_t *ids, const float *wts, uint32_t n) {
    (void)g;
    int arena = q4_expert_cache_arena(c);
    if (arena < 0) return false;
    uint32_t na = q4_hip_experts_n_a(arena);
    for (uint32_t k = 0; k < n; k++) {
        int32_t slot = q4_expert_cache_slot(c, layer, ids[k]);
        if (slot < 0 || (uint32_t)slot >= na) return false;
        moe_hslots[k] = slot;
        moe_hwts[k] = wts[k];
    }
    moe_nhits = n;
    return q4_hip_moe_stage(moe_hslots, moe_hwts, n);
}

/* Launch hit-exec reading the staged arrays — identical op sequence whether
 * called inside a gB capture or eagerly on the tail path. */
static bool moe_gpu_exec_staged(const q4_gguf *g, q4_expert_cache *c,
                                int32_t layer, const float *d_x, float *d_y,
                                float *d_h, float *d_h2, float *d_down,
                                uint32_t n_ff, uint32_t n_embd) {
    const q4_tensor *gate, *up, *down;
    uint64_t og, ou, od, tot;
    if (!q4_expert_parts(g, layer, &gate, &up, &down, &og, &ou, &od, &tot))
        return false;
    int arena = q4_expert_cache_arena(c);
    const uint8_t *base = q4_hip_experts_base_a(arena);
    uint64_t stride = q4_hip_experts_stride_a(arena);
    if (!base || !stride) return false;
    if (q4_hip_moe_exec_staged(base, stride, gate->ggml_type, og,
                               up->ggml_type, ou, down->ggml_type, od, d_x,
                               d_h, d_h2, d_down, d_y, n_ff, gate->ne[0],
                               n_embd))
        return true;
    /* Unsupported weight type: per-expert fallback over the staged set. */
    const uint8_t *dpacks[64];
    for (uint32_t k = 0; k < moe_nhits; k++)
        dpacks[k] = base + (size_t)moe_hslots[k] * stride;
    int use_fuse = (gate->ggml_type == Q4_T_Q4_K &&
                    up->ggml_type == Q4_T_Q4_K);
    for (uint32_t k = 0; k < moe_nhits; k++) {
        const uint8_t *dpack = dpacks[k];
        if (use_fuse) {
            if (!q4_hip_gateup_q4k(dpack + og, dpack + ou, d_x, d_h, n_ff,
                                   n_embd))
                return false;
        } else {
            if (!q4_hip_gemv_dd(gate->ggml_type, dpack + og, gate->ne[1],
                                gate->ne[0], d_x, d_h, 1.f))
                return false;
            if (!q4_hip_silu(d_h, n_ff)) return false;
            if (!q4_hip_gemv_dd(up->ggml_type, dpack + ou, up->ne[1],
                                up->ne[0], d_x, d_h2, 1.f))
                return false;
            if (!q4_hip_mul(d_h, d_h, d_h2, n_ff)) return false;
        }
        if (!q4_hip_gemv_dd(down->ggml_type, dpack + od, down->ne[1],
                            down->ne[0], d_h, d_down, 1.f))
            return false;
        if (!q4_hip_axpy(d_y, (float *)d_down, moe_hwts[k], n_embd))
            return false;
    }
    return true;
}

/* Eager GPU routed-expert chain: stage misses to L1, run the GEMV chain.
 * Used when the CPU path is off or refused dispatch mid-flight. */
static bool moe_gpu_chain(const q4_gguf *g, q4_expert_cache *c, int32_t layer,
                          const float *d_x, float *d_y, const int32_t *ids,
                          const float *wts, uint32_t topk, float *d_h,
                          float *d_h2, float *d_down, uint32_t n_ff,
                          uint32_t n_embd) {
    double t0 = moe_now();
    if (q4_expert_stage_batch_sync(c, layer, ids, topk, 1) != 0)
        return false;
    q4_hip_copy_wait();
    q4_prof_moe_io += moe_now() - t0;
    t0 = moe_now();
    bool ok = moe_gpu_exec(g, c, layer, d_x, d_y, ids, wts, topk, d_h, d_h2,
                           d_down, n_ff, n_embd);
    q4_prof_moe_gemm += moe_now() - t0;
    return ok;
}

bool q4_moe_layer_dev(const q4_gguf *g, const q4_store *st, q4_expert_cache *c,
                      int32_t layer, float *d_x, float *d_y, float *d_ws) {
    /* Decode after prefill: a leftover prefetch must not own the io pool.
     * No-op when none is running. */
    q4_expert_pf_join(c);
    if (!g || !st || !c || !d_x || !d_y || !d_ws || !q4_hip_ok()) return false;
    const uint32_t n_embd = g->n_embd;
    const uint32_t n_ff = g->n_ff_exp ? g->n_ff_exp : 640;
    const uint32_t n_exp = g->n_expert;
    const uint32_t topk = g->n_expert_used;
    if (n_exp > 512 || topk > 64) return false;
    float *d_rlog = d_ws;
    float *d_h = d_rlog + n_exp + 16;         /* ≤64 experts batched */
    float *d_h2 = d_h + 64 * n_ff;
    float *d_down = d_h2 + 64 * n_ff;

    if (q4_moe_cpu_active(g)) {
        if (!q4_moe_gpu_a(g, st, layer, d_x, d_y, d_ws)) return false;
        int rc = q4_moe_hostpart(g, c, layer);
        if (rc < 0) return false;
        if (rc > 0) return q4_moe_gpu_b(g, c, layer);
        /* dispatch refused: finish on the GPU with the ids we already have */
        return q4_moe_gpu_tail(g, c, layer, d_x, d_y, d_ws);
    }

    /* ---- original all-GPU path (Q4_CPU_EXPERTS=0 / buffer alloc failed) -- */
    const q4_tensor *t_router, *t_gate_s, *t_up_s, *t_down_s, *t_sgate;
    const uint8_t *d_router = moe_devw(g, st, layer,
                                     "blk.%d.ffn_gate_inp.weight", &t_router);
    const uint8_t *d_gate_s = moe_devw(g, st, layer,
                                     "blk.%d.ffn_gate_shexp.weight", &t_gate_s);
    const uint8_t *d_up_s = moe_devw(g, st, layer,
                                   "blk.%d.ffn_up_shexp.weight", &t_up_s);
    const uint8_t *d_down_s = moe_devw(g, st, layer,
                                     "blk.%d.ffn_down_shexp.weight",
                                     &t_down_s);
    (void)moe_devw(g, st, layer, "blk.%d.ffn_gate_inp_shexp.weight",
                   &t_sgate);
    if (!d_router || !d_gate_s || !d_up_s || !d_down_s) return false;
    float *d_sh = d_down + n_embd;
    float *d_dot = d_sh + n_embd;
    int32_t ids[64];
    float wts[64];
    if (!q4_hip_gemv_dd(t_router->ggml_type, d_router, t_router->ne[1],
                        t_router->ne[0], d_x, d_rlog, 1.f))
        return false;
    if (!q4_hip_fill(d_y, 0.f, n_embd)) return false;
    {
        double t0 = moe_now();
        if (!q4_hip_softmax_topk(d_rlog, n_exp, topk, ids, wts)) return false;
        q4_prof_moe_d2h += moe_now() - t0;
        q4_prof_moe_topk += moe_now() - t0;
    }
    if (!moe_gpu_chain(g, c, layer, d_x, d_y, ids, wts, topk, d_h, d_h2,
                       d_down, n_ff, n_embd))
        return false;
    if (!q4_hip_gemv_dd(t_gate_s->ggml_type, d_gate_s, t_gate_s->ne[1],
                        t_gate_s->ne[0], d_x, d_h, 1.f))
        return false;
    if (!q4_hip_silu(d_h, n_ff)) return false;
    if (!q4_hip_gemv_dd(t_up_s->ggml_type, d_up_s, t_up_s->ne[1], t_up_s->ne[0],
                        d_x, d_h2, 1.f))
        return false;
    if (!q4_hip_mul(d_h, d_h, d_h2, n_ff)) return false;
    if (!q4_hip_gemv_dd(t_down_s->ggml_type, d_down_s, t_down_s->ne[1],
                        t_down_s->ne[0], d_h, d_sh, 1.f))
        return false;
    float sg = 1.f;
    int sg_dev = 0;
    const float *d_sgf = (t_sgate && t_sgate->ne[0] == n_embd)
                             ? moe_sgate_f32d(g, st, t_sgate)
                             : NULL;
    if (d_sgf) {
        double t0 = moe_now();
        if (!q4_hip_dot(d_sgf, d_x, d_dot, n_embd))
            return false;
        if (!q4_hip_sigmoid(d_dot, 1)) return false;
        sg_dev = 1;
        q4_prof_moe_sg += moe_now() - t0;
    }
    if (sg_dev)
        return q4_hip_axpy_dalpha(d_y, d_sh, d_dot, n_embd);
    return q4_hip_axpy(d_y, d_sh, sg, n_embd);
}

/* Shared-expert + sgate tail for the n-token device path: gate_s -> silu,
 * up_s -> mul, down_s -> d_sh, then sgate-scaled or plain add into d_y.
 * Extracted so the small-batch hybrid path computes the shared
 * contribution identically. */
static bool moe_dev_n_shared(const q4_gguf *g, const q4_store *st,
                             const q4_tensor *t_gate_s,
                             const q4_tensor *t_up_s,
                             const q4_tensor *t_down_s,
                             const q4_tensor *t_sgate,
                             const uint8_t *d_gate_s, const uint8_t *d_up_s,
                             const uint8_t *d_down_s, const float *d_x,
                             float *d_h, float *d_h2, float *d_sh, float *d_sg,
                             float *d_y, uint32_t n_ff, uint32_t n_embd,
                             uint32_t n_tok) {
    if (!q4_hip_gemv_dd_n(t_gate_s->ggml_type, d_gate_s, t_gate_s->ne[1],
                          t_gate_s->ne[0], d_x, d_h, 1.f, n_tok) ||
        !q4_hip_silu(d_h, (uint64_t)n_tok * n_ff) ||
        !q4_hip_gemv_dd_n(t_up_s->ggml_type, d_up_s, t_up_s->ne[1],
                          t_up_s->ne[0], d_x, d_h2, 1.f, n_tok) ||
        !q4_hip_mul(d_h, d_h, d_h2, (uint64_t)n_tok * n_ff) ||
        !q4_hip_gemv_dd_n(t_down_s->ggml_type, d_down_s, t_down_s->ne[1],
                          t_down_s->ne[0], d_h, d_sh, 1.f, n_tok))
        return false;
    const float *d_sgf = (t_sgate && t_sgate->ne[0] == n_embd)
                             ? moe_sgate_f32d(g, st, t_sgate)
                             : NULL;
    if (d_sgf) {
        /* one batched dot per token: sg[t] = sigmoid(sgate . x[t]) */
        return q4_hip_gemv_dd_n(Q4_T_F32, (const uint8_t *)d_sgf, 1, n_embd,
                                d_x, d_sg, 1.f, n_tok) &&
               q4_hip_sigmoid(d_sg, n_tok) &&
               q4_hip_axpy_rows(d_y, d_sh, d_sg, n_embd, n_tok);
    }
    return q4_hip_add(d_y, d_y, d_sh, (uint64_t)n_tok * n_embd);
}

/* Small-batch hybrid MoE (2..8 tokens, MTP verify + short prefill): top-k
 * hits run on GPU through the wmoe rows (per-token slices of a device
 * slot/weight array), misses are dispatched to the CPU pool one job per
 * token — instead of the prefill path's synchronous PCIe staging of every
 * miss.  Host scratch is grow-once; device scratch is carved from d_ws. */
static int moe_sb_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_MOE_SB");
        v = !(e && e[0] == '0');
    }
    return v;
}

#define MOE_SB_MAXTOK 8u
#define MOE_SB_MAXK 64u

static float *sb_hx, *sb_hy, *sb_hlog;
static int32_t *sb_ids, *sb_cids, *sb_hslots;
static float *sb_wts, *sb_cwt, *sb_hwts;
static uint32_t *sb_nc;
static uint32_t sb_off[MOE_SB_MAXTOK + 1];
static int sb_jobs[MOE_SB_MAXTOK];
static uint64_t sb_tags[MOE_SB_MAXTOK];

static bool sb_grow(void) {
    static int done;
    if (done) return true;
    sb_hx = malloc(MOE_SB_MAXTOK * 16384 * sizeof(float));
    sb_hy = malloc(MOE_SB_MAXTOK * 16384 * sizeof(float));
    sb_hlog = malloc(MOE_SB_MAXTOK * 512 * sizeof(float));
    sb_ids = malloc(MOE_SB_MAXTOK * MOE_SB_MAXK * sizeof(int32_t));
    sb_cids = malloc(MOE_SB_MAXTOK * MOE_SB_MAXK * sizeof(int32_t));
    sb_hslots = malloc(MOE_SB_MAXTOK * MOE_SB_MAXK * sizeof(int32_t));
    sb_wts = malloc(MOE_SB_MAXTOK * MOE_SB_MAXK * sizeof(float));
    sb_cwt = malloc(MOE_SB_MAXTOK * MOE_SB_MAXK * sizeof(float));
    sb_hwts = malloc(MOE_SB_MAXTOK * MOE_SB_MAXK * sizeof(float));
    sb_nc = malloc(MOE_SB_MAXTOK * sizeof(uint32_t));
    done = sb_hx && sb_hy && sb_hlog && sb_ids && sb_cids && sb_hslots &&
           sb_wts && sb_cwt && sb_hwts && sb_nc;
    return done;
}

static int sb_prof(void) {
    static int v = -1;
    if (v < 0) v = getenv("Q4_PROFILE") && getenv("Q4_PROFILE")[0] != '0';
    return v;
}
static double sb_p_route, sb_p_topk, sb_p_bound, sb_p_disp, sb_p_enq,
              sb_p_join, sb_p_tail, sb_p_stage;
static uint64_t sb_p_calls, sb_p_tok, sb_p_hit, sb_p_hu, sb_p_miss, sb_p_mu;
#define SB_PT(acc)                                                           \
    do {                                                                     \
        if (sbp) {                                                           \
            double _n = moe_now();                                           \
            acc += _n - _tt;                                                 \
            _tt = _n;                                                        \
        }                                                                    \
    } while (0)

/* Join every outstanding CPU job from the sb dispatch loop (result
 * ignored — used on late-failure paths so no worker still writes sb_hy or
 * holds a pool slot when the caller falls back). */
static void sb_join_all(uint32_t n_tok) {
    for (uint32_t t = 0; t < n_tok; t++)
        if (sb_jobs[t] >= 0) {
            q4_cpuex_join_tag(sb_jobs[t], sb_tags[t]);
            sb_jobs[t] = -1;
        }
}

void q4_moe_sb_prof_print(FILE *f) {
    if (!f || !sb_p_calls) return;
    double c = (double)sb_p_calls;
    fprintf(f, "q4 sb prof: calls %llu  tok %llu  hit %llu (uniq %llu)  "
               "miss %llu (uniq %llu)  | route %.2f  topk %.2f  "
               "boundary %.2f  cpu-disp %.2f  gpu-enq %.2f  join %.2f  "
               "tail %.2f  stage %.2f ms/call\n",
            (unsigned long long)sb_p_calls, (unsigned long long)sb_p_tok,
            (unsigned long long)sb_p_hit, (unsigned long long)sb_p_hu,
            (unsigned long long)sb_p_miss, (unsigned long long)sb_p_mu,
            sb_p_route * 1e3 / c, sb_p_topk * 1e3 / c, sb_p_bound * 1e3 / c,
            sb_p_disp * 1e3 / c, sb_p_enq * 1e3 / c, sb_p_join * 1e3 / c,
            sb_p_tail * 1e3 / c, sb_p_stage * 1e3 / c);
    sb_p_route = sb_p_topk = sb_p_bound = sb_p_disp = sb_p_enq = 0;
    sb_p_join = sb_p_tail = sb_p_stage = 0;
    sb_p_calls = sb_p_tok = sb_p_hit = sb_p_hu = sb_p_miss = sb_p_mu = 0;
}

static bool moe_dev_n_sb(const q4_gguf *g, const q4_store *st,
                         q4_expert_cache *c, int32_t layer, float *d_x,
                         float *d_y, float *d_ws, uint32_t n_tok,
                         uint64_t ws_floats) {
    const int arena = q4_expert_cache_arena(c);
    if (arena < 0 || !sb_grow()) return false;
    const uint32_t n_embd = g->n_embd;
    const uint32_t n_ff = g->n_ff_exp ? g->n_ff_exp : 640;
    const uint32_t n_exp = g->n_expert;
    const uint32_t topk = g->n_expert_used;
    if (!n_exp || !topk || topk > MOE_SB_MAXK || n_exp > 512 ||
        n_embd > 16384)
        return false;
    const q4_tensor *gate, *up, *down;
    uint64_t og, ou, od, tot;
    if (!q4_expert_parts(g, layer, &gate, &up, &down, &og, &ou, &od, &tot))
        return false;
    if (gate->ggml_type != up->ggml_type || gate->ne[0] != up->ne[0] ||
        gate->ne[1] != up->ne[1] ||
        !q4_hip_moe_tok_ok(gate->ggml_type, up->ggml_type, down->ggml_type,
                           gate->ne[0], n_ff))
        return false;
    char nm[Q4_MAX_NAME];
    snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_inp.weight", layer);
    const q4_tensor *t_router = q4_find_tensor(g, nm);
    snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_shexp.weight", layer);
    const q4_tensor *t_gate_s = q4_find_tensor(g, nm);
    snprintf(nm, sizeof(nm), "blk.%d.ffn_up_shexp.weight", layer);
    const q4_tensor *t_up_s = q4_find_tensor(g, nm);
    snprintf(nm, sizeof(nm), "blk.%d.ffn_down_shexp.weight", layer);
    const q4_tensor *t_down_s = q4_find_tensor(g, nm);
    snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_inp_shexp.weight", layer);
    const q4_tensor *t_sgate = q4_find_tensor(g, nm);
    if (!t_router || !t_gate_s || !t_up_s || !t_down_s) return false;
    const uint8_t *d_router = q4_store_dev(st, t_router);
    const uint8_t *d_gate_s = q4_store_dev(st, t_gate_s);
    const uint8_t *d_up_s = q4_store_dev(st, t_up_s);
    const uint8_t *d_down_s = q4_store_dev(st, t_down_s);
    if (!d_router || !d_gate_s || !d_up_s || !d_down_s) return false;
    const uint8_t *abase = q4_hip_experts_base_a(arena);
    uint64_t astride = q4_hip_experts_stride_a(arena);
    if (!abase || !astride) return false;

    /* d_ws layout: rlog | slots | wts | h | h2 | shared out | sg | hy */
    float *d_rlog = d_ws;
    int32_t *d_slots = (int32_t *)(d_rlog + (size_t)n_tok * n_exp);
    float *d_wts = (float *)(d_slots + (size_t)n_tok * topk);
    float *d_h = d_wts + (size_t)n_tok * topk;
    float *d_h2 = d_h + (size_t)n_tok * topk * n_ff;
    float *d_sh = d_h2 + (size_t)n_tok * topk * n_ff;
    float *d_sg = d_sh + (size_t)n_tok * n_embd;
    float *d_hy = d_sg + n_tok;
    size_t need = (size_t)(d_hy + (size_t)n_tok * n_embd - d_ws);
    if (need > ws_floats) return false;

    int sbp = sb_prof();
    double _tt = sbp ? moe_now() : 0;
    if (!q4_hip_gemv_dd_n(t_router->ggml_type, d_router, t_router->ne[1],
                          t_router->ne[0], d_x, d_rlog, 1.f, n_tok))
        return false;
    {
        double t0 = moe_now();
        if (!q4_hip_d2h(sb_hlog, d_rlog, (size_t)n_tok * n_exp * sizeof(float)) ||
            !q4_hip_d2h(sb_hx, d_x, (size_t)n_tok * n_embd * sizeof(float)))
            return false;
        q4_prof_moe_d2h += moe_now() - t0;
    }
    SB_PT(sb_p_route);
    {
        double t0 = moe_now();
        for (uint32_t t = 0; t < n_tok; t++) {
            q4_softmax(sb_hlog + t * n_exp, n_exp);
            q4_topk(sb_hlog + t * n_exp, n_exp, topk, sb_ids + t * topk,
                    sb_wts + t * topk, true);
            q4_expert_route_note(c, layer, sb_ids + t * topk, (int)topk);
        }
        q4_prof_moe_topk += moe_now() - t0;
    }
    SB_PT(sb_p_topk);
    /* Boundary with prior work: reap an outstanding CPU job, drain async
     * H2D and the async stager, then the lock-free gpu_slot reads below are
     * safe. */
    if (!q4_cpuex_reap()) return false;
    {
        double t0 = moe_now();
        q4_hip_copy_wait();
        q4_prof_moe_cw += moe_now() - t0;
    }
    q4_expert_uploads_mark(c);
    q4_expert_stage_join(c);
    q4_expert_pf_join(c);
    q4_expert_clear_sticky(c);
    SB_PT(sb_p_bound);

    /* Split each token's top-k into GPU hits (token-major prefix) and CPU
     * misses, preserving top-k order within each list. */
    const uint32_t na = q4_hip_experts_n_a(arena);
    uint32_t nhit = 0, nmiss = 0;
    for (uint32_t t = 0; t < n_tok; t++) {
        sb_off[t] = nhit;
        uint32_t nc = 0;
        for (uint32_t k = 0; k < topk; k++) {
            int32_t e = sb_ids[t * topk + k];
            float w = sb_wts[t * topk + k];
            int32_t s = (e >= 0 && e < (int32_t)n_exp)
                            ? q4_expert_gpu_slot(c, layer, e)
                            : -1;
            if (s >= 0 && (uint32_t)s < na) {
                sb_hslots[nhit] = s;
                sb_hwts[nhit] = w;
                nhit++;
            } else {
                sb_cids[t * MOE_SB_MAXK + nc] = e;
                sb_cwt[t * MOE_SB_MAXK + nc] = w;
                nc++;
            }
        }
        sb_nc[t] = nc;
        nmiss += nc;
    }
    sb_off[n_tok] = nhit;
    q4_prof_moe_ng += nhit;
    q4_prof_moe_nc += nmiss;
    q4_prof_moe_ncall += n_tok;
    /* Distinct hit slots / miss ids for the call (miss union doubles as the
     * stage_async list below). */
    int32_t uni[MOE_SB_MAXTOK * MOE_SB_MAXK];
    uint32_t nu = 0;
    {
        uint32_t hu = 0;
        for (uint32_t i = 0; i < nhit; i++) {
            uint32_t u = 0;
            for (; u < i; u++) if (sb_hslots[u] == sb_hslots[i]) break;
            if (u == i) hu++;
        }
        for (uint32_t t = 0; t < n_tok; t++)
            for (uint32_t k = 0; k < sb_nc[t]; k++) {
                int32_t e = sb_cids[t * MOE_SB_MAXK + k];
                uint32_t u = 0;
                for (; u < nu; u++) if (uni[u] == e) break;
                if (u == nu) uni[nu++] = e;
            }
        if (sbp) {
            sb_p_calls++;
            sb_p_tok += n_tok;
            sb_p_hit += nhit;
            sb_p_hu += hu;
            sb_p_miss += nmiss;
            sb_p_mu += nu;
        }
    }

    /* CPU jobs first so the pool crunches while GPU hits queue. */
    for (uint32_t t = 0; t < n_tok; t++) {
        sb_jobs[t] = -1;
        if (!sb_nc[t]) {
            memset(sb_hy + (size_t)t * n_embd, 0, n_embd * sizeof(float));
            continue;
        }
        int slot = -1;
        for (int r = 0; r < 400; r++) {
            slot = q4_cpuex_dispatch2(g, c, layer,
                                      sb_hx + (size_t)t * n_embd,
                                      sb_cids + (size_t)t * MOE_SB_MAXK,
                                      sb_cwt + (size_t)t * MOE_SB_MAXK,
                                      sb_nc[t], sb_hy + (size_t)t * n_embd,
                                      &sb_tags[t]);
            if (slot != -1) break;
            (void)q4_cpuex_retire();
            _mm_pause();
        }
        if (slot == -1) {
            sb_join_all(n_tok);
            return false;   /* d_y untouched: the old path can run */
        }
        sb_jobs[t] = slot;
    }
    SB_PT(sb_p_disp);

    double t_gemm = moe_now();
    if (!q4_hip_fill(d_y, 0.f, (uint64_t)n_tok * n_embd)) {
        sb_join_all(n_tok);
        return false;
    }
    if (nhit) {
        if (!q4_hip_h2d(d_slots, sb_hslots, nhit * sizeof(int32_t)) ||
            !q4_hip_h2d(d_wts, sb_hwts, nhit * sizeof(float)) ||
            !q4_hip_moe_exec_tok(abase, astride, d_slots, d_wts, sb_off,
                                 n_tok, gate->ggml_type, og, up->ggml_type,
                                 ou, down->ggml_type, od, d_x, d_h, d_h2,
                                 d_y, n_ff, gate->ne[0], n_embd)) {
            sb_join_all(n_tok);
            return false;
        }
    }
    if (!moe_dev_n_shared(g, st, t_gate_s, t_up_s, t_down_s, t_sgate,
                          d_gate_s, d_up_s, d_down_s, d_x, d_h, d_h2, d_sh,
                          d_sg, d_y,
                          n_ff, n_embd, n_tok)) {
        sb_join_all(n_tok);
        return false;
    }
    q4_prof_moe_gemm += moe_now() - t_gemm;
    SB_PT(sb_p_enq);

    {
        double t0 = moe_now();
        for (uint32_t t = 0; t < n_tok; t++)
            if (sb_jobs[t] >= 0) {
                if (!q4_cpuex_join_tag(sb_jobs[t], sb_tags[t])) {
                    sb_join_all(n_tok);
                    return false;
                }
                sb_jobs[t] = -1;
            }
        q4_prof_moe_join += moe_now() - t0;
    }
    SB_PT(sb_p_join);
    if (!q4_hip_h2d(d_hy, sb_hy, (size_t)n_tok * n_embd * sizeof(float)) ||
        !q4_hip_axpy(d_y, d_hy, 1.f, (uint64_t)n_tok * n_embd)) {
        sb_join_all(n_tok);
        return false;
    }

    q4_hip_stream_sync();
    SB_PT(sb_p_tail);
    /* With MTP on, main-model tokens go through verify instead of
     * hostpart — stage the misses here so the L1 LRU stays warm.  The
     * stream sync above guarantees no queued kernel still reads a slot
     * the stager may rebind. */
    if (nmiss) {
        static int ms = -1;
        if (ms < 0) {
            const char *e = getenv("Q4_MISS_STAGE");
            ms = (!e || e[0] != '0') || !q4_expert_is_resident(c);
        }
        if (ms) {
            double t0 = moe_now();
            if (nu) q4_expert_stage_async(c, layer, uni, nu, NULL, 0);
            q4_prof_moe_io += moe_now() - t0;
            if (sbp) { double _n = moe_now(); sb_p_stage += _n - _tt; _tt = _n; }
        }
    }
    return true;
}

bool q4_moe_layer_dev_n(const q4_gguf *g, const q4_store *st, q4_expert_cache *c,
                        int32_t layer, float *d_x, float *d_y, float *d_ws,
                        uint32_t n_tok, uint64_t ws_floats) {
    if (n_tok <= 1)
        return q4_moe_layer_dev(g, st, c, layer, d_x, d_y, d_ws);
    if (n_tok <= MOE_SB_MAXTOK && moe_sb_on() && g && st && c && d_x && d_y &&
        d_ws && q4_hip_ok() &&
        moe_dev_n_sb(g, st, c, layer, d_x, d_y, d_ws, n_tok, ws_floats))
        return true;
    if (!g || !st || !c || !d_x || !d_y || !d_ws || !q4_hip_ok()) { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 1); return false; }
    const int arena = q4_expert_cache_arena(c);
    const uint32_t n_embd = g->n_embd;
    const uint32_t n_ff = g->n_ff_exp ? g->n_ff_exp : 640;
    const uint32_t n_exp = g->n_expert;
    const uint32_t topk = g->n_expert_used;
    if (n_exp > 512 || topk > 10) { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 3); return false; }
    const q4_tensor *gate, *up, *down;
    uint64_t og, ou, od, tot;
    if (!q4_expert_parts(g, layer, &gate, &up, &down, &og, &ou, &od, &tot))
        { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 4); return false; }
    char nm[Q4_MAX_NAME];
    snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_inp.weight", layer);
    const q4_tensor *t_router = q4_find_tensor(g, nm);
    snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_shexp.weight", layer);
    const q4_tensor *t_gate_s = q4_find_tensor(g, nm);
    snprintf(nm, sizeof(nm), "blk.%d.ffn_up_shexp.weight", layer);
    const q4_tensor *t_up_s = q4_find_tensor(g, nm);
    snprintf(nm, sizeof(nm), "blk.%d.ffn_down_shexp.weight", layer);
    const q4_tensor *t_down_s = q4_find_tensor(g, nm);
    snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_inp_shexp.weight", layer);
    const q4_tensor *t_sgate = q4_find_tensor(g, nm);
    if (!t_router || !t_gate_s || !t_up_s || !t_down_s) { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 5); return false; }
    const uint8_t *d_router = q4_store_dev(st, t_router);
    const uint8_t *d_gate_s = q4_store_dev(st, t_gate_s);
    const uint8_t *d_up_s = q4_store_dev(st, t_up_s);
    const uint8_t *d_down_s = q4_store_dev(st, t_down_s);
    if (!d_router || !d_gate_s || !d_up_s || !d_down_s) { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 6); return false; }

    /* Tokens are processed in groups of <=1024 rows: with the next layer's
     * experts prefetching on the copy stream, groups only bound the
     * workspace (Gmax assignment rows -> d_h/d_h2/d_pd) while every GEMM
     * queues back to back on the compute stream. */
    const uint32_t gsz = n_tok < 1024 ? n_tok : 1024;
    const uint32_t n_g = (n_tok + gsz - 1) / gsz;
    const uint32_t Gmax = gsz * topk;
    /* d_ws layout (floats): router logits | ph/ph2 (Gmax*n_ff) | shared out |
     * expert out pd (Gmax*n_embd) | assignment tables | sgate | group meta */
    float *d_rlog = d_ws;
    float *d_h = d_rlog + (size_t)n_tok * n_exp;
    float *d_h2 = d_h + (size_t)Gmax * n_ff;
    float *d_sh = d_h2 + (size_t)Gmax * n_ff;
    float *d_pd = d_sh + (size_t)n_tok * n_embd;
    int32_t *d_atok = (int32_t *)(d_pd + (size_t)Gmax * n_embd);
    int32_t *d_rsrc = d_atok + (size_t)n_tok * topk;
    float *d_rw = (float *)(d_rsrc + (size_t)n_tok * topk);
    float *d_sg = d_rw + (size_t)n_tok * topk;
    int32_t *d_meta = (int32_t *)(d_sg + n_tok);
    size_t need = (size_t)((const float *)(d_meta + 3 * 512 * n_g) - d_ws);
    if (need > ws_floats) { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 7); return false; }
    if (!q4_hip_gemv_dd_n(t_router->ggml_type, d_router, t_router->ne[1],
                          t_router->ne[0], d_x, d_rlog, 1.f, n_tok))
        { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 8); return false; }
    /* Host block: logits, topk, per-group histograms and assignment tables,
     * stage id lists, batched-GEMM metadata (chunk-sized -> heap). */
    size_t hfloats = (size_t)n_tok * n_exp + 2u * (size_t)n_tok * topk;
    size_t hints = 3u * (size_t)n_tok * topk + 4u * (size_t)n_g * 512 +
                   (size_t)n_g + 3u * (size_t)n_g * 512;
    uint8_t *hbuf = malloc(hfloats * sizeof(float) + hints * sizeof(int32_t));
    if (!hbuf) { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 9); return false; }
    float *hlog = (float *)hbuf;
    float *wts = hlog + (size_t)n_tok * n_exp;
    float *rw = wts + (size_t)n_tok * topk;
    int32_t *ids = (int32_t *)(rw + (size_t)n_tok * topk);
    int32_t *a_tok = ids + (size_t)n_tok * topk;
    int32_t *rsrc = a_tok + (size_t)n_tok * topk;
    uint32_t *ek = (uint32_t *)(rsrc + (size_t)n_tok * topk);
    uint32_t *ebase = ek + (size_t)n_g * 512;
    uint32_t *cur = ebase + (size_t)n_g * 512;
    int32_t *stg = (int32_t *)(cur + (size_t)n_g * 512);
    uint32_t *stgn = (uint32_t *)(stg + (size_t)n_g * 512);
    int32_t *meta = (int32_t *)(stgn + n_g);
    memset(ek, 0, (size_t)n_g * 512 * sizeof(uint32_t));
    memset(stgn, 0, (size_t)n_g * sizeof(uint32_t));
    {
        double t0 = moe_now();
        if (!q4_hip_d2h(hlog, d_rlog, (size_t)n_tok * n_exp * sizeof(float))) {
            free(hbuf);
            { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 10); return false; }
        }
        q4_prof_moe_d2h += moe_now() - t0;
    }
    for (uint32_t t = 0; t < n_tok; t++) {
        q4_softmax(hlog + t * n_exp, n_exp);
        q4_topk(hlog + t * n_exp, n_exp, topk, ids + t * topk,
                wts + t * topk, true);
        q4_expert_route_note(c, layer, ids + t * topk, (int)topk);
    }
    /* Assignment rows are group-local so d_h/d_pd stay group-sized. A group's
     * segment in the concat arrays is g*gsz*topk; rsrc/rw stay token-major. */
    for (uint32_t t = 0; t < n_tok; t++) {
        uint32_t gno = t / gsz;
        for (uint32_t k = 0; k < topk; k++) {
            int32_t e = ids[t * topk + k];
            if (e >= 0 && e < (int32_t)n_exp) ek[gno * 512 + e]++;
        }
    }
    for (uint32_t gno = 0; gno < n_g; gno++) {
        uint32_t b = 0;
        for (uint32_t e = 0; e < n_exp; e++) {
            ebase[gno * 512 + e] = b;
            b += ek[gno * 512 + e];
        }
        memcpy(cur + gno * 512, ebase + gno * 512, 512 * sizeof(uint32_t));
    }
    for (uint32_t t = 0; t < n_tok; t++) {
        uint32_t gno = t / gsz;
        uint32_t seg = gno * gsz * topk;
        for (uint32_t k = 0; k < topk; k++) {
            int32_t e = ids[t * topk + k];
            if (e < 0 || e >= (int32_t)n_exp) {
                rsrc[t * topk + k] = -1;
                rw[t * topk + k] = 0.f;
                continue;
            }
            uint32_t p = cur[gno * 512 + e]++;
            a_tok[seg + p] = (int32_t)t;
            rsrc[t * topk + k] = (int32_t)p;
            rw[t * topk + k] = wts[t * topk + k];
        }
    }
    for (uint32_t gno = 0; gno < n_g; gno++) {
        for (uint32_t e = 0; e < n_exp; e++)
            if (ek[gno * 512 + e])
                stg[gno * 512 + stgn[gno]++] = (int32_t)e;
    }
    if (!q4_hip_h2d(d_atok, a_tok, (size_t)n_tok * topk * sizeof(int32_t)) ||
        !q4_hip_h2d(d_rsrc, rsrc, (size_t)n_tok * topk * sizeof(int32_t)) ||
        !q4_hip_h2d(d_rw, rw, (size_t)n_tok * topk * sizeof(float))) {
        free(hbuf);
        { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 11); return false; }
    }
    if (!q4_hip_fill(d_y, 0.f, (uint64_t)n_tok * n_embd)) {
        free(hbuf);
        { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 12); return false; }
    }

    /* Boundary with the previous layer's prefetch: it has finished enqueuing
     * once joined (its g_copy tail lands at copy_wait below). L-1's sticky
     * set is released here; L-1's expert GEMMs retired at its own sync. */
    q4_expert_pf_join(c);
    q4_expert_clear_sticky(c);
    double t_gemm0 = moe_now();
    {
        double t0 = moe_now();
        for (uint32_t gno = 0; gno < n_g; gno++) {
            if (!stgn[gno]) continue;
            if (q4_expert_stage_batch_sync(c, layer, stg + gno * 512,
                                           stgn[gno], 1) != 0) {
                free(hbuf);
                { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 13); return false; }
            }
            /* Protect every already-staged group: a later group's misses
             * must not rebind slots whose meta is built below. The union
             * doubles as the prefetch-eviction shield once pf_begin runs. */
            q4_expert_sticky(c, layer, stg + gno * 512, stgn[gno]);
        }
        double t1 = moe_now();
        q4_hip_copy_wait();
        q4_prof_moe_cw += moe_now() - t1;
        q4_prof_moe_io += moe_now() - t0;
    }
    /* Slots are stable from here on (sticky + prefetch shield), so the
     * batched-GEMM metadata for every group uploads in one shot. */
    memset(meta, 0, (size_t)n_g * 3 * 512 * sizeof(int32_t));
    int all_slotted = 1;
    for (uint32_t gno = 0; gno < n_g; gno++) {
        int32_t *h_sl = meta + gno * 3 * 512;
        int32_t *h_eb = h_sl + 512, *h_ec = h_sl + 1024;
        for (uint32_t i = 0; i < stgn[gno]; i++) {
            int32_t e = stg[gno * 512 + i];
            int32_t slot = q4_expert_cache_slot(c, layer, e);
            if (slot < 0) { all_slotted = 0; break; }
            h_sl[i] = slot;
            h_eb[i] = (int32_t)ebase[gno * 512 + e];
            h_ec[i] = (int32_t)ek[gno * 512 + e];
        }
        if (!all_slotted) break;
    }
    if (all_slotted &&
        !q4_hip_h2d(d_meta, meta, (size_t)n_g * 3 * 512 * sizeof(int32_t)))
        all_slotted = 0;
    /* Layer L's experts are sticky; the next layer's set stages underneath
     * the GEMM launches below plus the following attention block. */
    int launched = n_tok >= 256 ? q4_expert_pf_begin(c, layer + 1) : 0;

    const uint8_t *abase = q4_hip_experts_base_a(arena);
    uint64_t astride = q4_hip_experts_stride_a(arena);
    int use_fuse = (gate->ggml_type == Q4_T_Q4_K && up->ggml_type == Q4_T_Q4_K);
    for (uint32_t gno = 0; gno < n_g; gno++) {
        uint32_t t0 = gno * gsz;
        uint32_t gtn = n_tok - t0 < gsz ? n_tok - t0 : gsz;
        uint32_t seg = gno * gsz * topk;
        if (!stgn[gno]) continue;
        /* Batched path: one kernel launch per tensor over the whole expert
         * group instead of ~5 launches per expert. */
        int batched = all_slotted && abase != NULL && astride != 0;
        if (batched) {
            const int32_t *gm = d_meta + gno * 3 * 512;
            const uint32_t *d_eb = (const uint32_t *)gm + 512;
            const uint32_t *d_ec = (const uint32_t *)gm + 1024;
            uint32_t n_st = stgn[gno];
            /* WMMA path first: fused gate+up+silu writing d_h directly,
             * then the same tiled kernel for down. Unsupported types (or
             * Q4_WMMA=0) make the externs return false -> moe_pf fallback. */
            int fused = 0;
            int ok = 0;
            if (n_tok >= 64 && gate->ggml_type == up->ggml_type &&
                gate->ne[0] == up->ne[0] && gate->ne[1] == up->ne[1])
                fused = ok = q4_hip_moe_wmma2(gate->ggml_type, abase, astride,
                                    og, ou, gm, d_eb, d_ec, d_atok + seg, d_x,
                                    d_h, gate->ne[1], gate->ne[0], n_st);
            if (!ok && gate->ggml_type == up->ggml_type &&
                gate->ne[0] == up->ne[0] && gate->ne[1] == up->ne[1])
                ok = q4_hip_moe_pf2(gate->ggml_type, abase, astride, og,
                                    ou, gm, d_eb, d_ec, d_atok + seg, d_x,
                                    d_h, d_h2, gate->ne[1], gate->ne[0],
                                    n_st);
            else if (!ok)
                ok = q4_hip_moe_pf(gate->ggml_type, abase, astride, og,
                                   gm, d_eb, d_ec, d_atok + seg, d_x, d_h,
                                   gate->ne[1], gate->ne[0], n_st) &&
                     q4_hip_moe_pf(up->ggml_type, abase, astride, ou,
                                   gm, d_eb, d_ec, d_atok + seg, d_x, d_h2,
                                   up->ne[1], up->ne[0], n_st);
            if (ok && !fused) {
                uint32_t A_g = ebase[gno * 512 + stg[gno * 512 + n_st - 1]] +
                               ek[gno * 512 + stg[gno * 512 + n_st - 1]];
                ok = q4_hip_moe_act(d_h, d_h2, (uint64_t)A_g * n_ff);
            }
            int down_w = 0;
            if (ok && n_tok >= 64)
                down_w = q4_hip_moe_wmma(down->ggml_type, abase, astride, od,
                                         gm, d_eb, d_ec, NULL, d_h, d_pd,
                                         down->ne[1], down->ne[0], n_st);
            if (ok && !down_w)
                ok = q4_hip_moe_pf(down->ggml_type, abase, astride, od,
                                   gm, d_eb, d_ec, NULL, d_h, d_pd,
                                   down->ne[1], down->ne[0], n_st);
            batched = ok;
        }
        if (!batched) {
            /* fall through to the per-expert loop on unsupported types */
            for (uint32_t e = 0; e < n_exp; e++) {
                uint32_t K = ek[gno * 512 + e];
                if (!K) continue;
                uint32_t base = ebase[gno * 512 + e];
                int32_t slot = q4_expert_cache_slot(c, layer, (int32_t)e);
                const uint8_t *dpack =
                    slot >= 0 ? q4_hip_experts_dev_a(arena, (uint32_t)slot)
                              : NULL;
                if (!dpack) {
                    free(hbuf);
                    { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 14); return false; }
                }
                float *ph = d_h + (size_t)base * n_ff;
                float *ph2 = d_h2 + (size_t)base * n_ff;
                float *pd = d_pd + (size_t)base * n_embd;
                const int32_t *which = d_atok + seg + base;
                if (use_fuse) {
                    if (!q4_hip_gateup_q4k_n_which(dpack + og, dpack + ou,
                                                   d_x, ph, n_ff, n_embd, K,
                                                   which)) {
                        free(hbuf);
                        { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 15); return false; }
                    }
                } else {
                    if (!q4_hip_gemv_dd_n_which(gate->ggml_type, dpack + og,
                                                gate->ne[1], gate->ne[0],
                                                d_x, ph, 1.f, K, which) ||
                        !q4_hip_silu(ph, (uint64_t)K * n_ff) ||
                        !q4_hip_gemv_dd_n_which(up->ggml_type, dpack + ou,
                                                up->ne[1], up->ne[0],
                                                d_x, ph2, 1.f, K, which) ||
                        !q4_hip_mul(ph, ph, ph2, (uint64_t)K * n_ff)) {
                        free(hbuf);
                        { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 16); return false; }
                    }
                }
                /* ph rows are assignment-indexed: plain batched GEMV to pd. */
                if (!q4_hip_gemv_dd_n(down->ggml_type, dpack + od,
                                      down->ne[1], down->ne[0], ph, pd,
                                      1.f, K)) {
                    free(hbuf);
                    { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 17); return false; }
                }
            }
        }
        /* dy[t] += sum_k rw * pd[rsrc]: group-local rows, token-window view. */
        if (!q4_hip_expert_reduce(d_y + (size_t)t0 * n_embd, d_pd,
                                  d_rsrc + (size_t)t0 * topk,
                                  d_rw + (size_t)t0 * topk, gtn, topk,
                                  n_embd)) {
            free(hbuf);
            { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 18); return false; }
        }
    }
    q4_hip_sync();
    q4_prof_moe_gemm += moe_now() - t_gemm0;
    if (!launched) {
        /* No prefetch to shield: restore the decode-time protection of just
         * the last token's experts (the all-used set must not linger). */
        q4_expert_clear_sticky(c);
        q4_expert_sticky(c, layer, ids + (size_t)(n_tok - 1) * topk, topk);
    }
    /* With a prefetch in flight the all-used sticky stays: it is released
     * at the next layer's join, after these GEMMs have long retired. */
    if (!moe_dev_n_shared(g, st, t_gate_s, t_up_s, t_down_s, t_sgate,
                          d_gate_s, d_up_s, d_down_s, d_x, d_h, d_h2, d_sh,
                          d_sg, d_y,
                          n_ff, n_embd, n_tok)) {
        free(hbuf);
        { if (getenv("Q4_MTP_DEBUG")) fprintf(stderr, "q4 mtp: moe _n fail #%d\n", 19); return false; }
    }
    free(hbuf);
    return true;
}
bool q4_moe_layer_store(const q4_gguf *g, const q4_store *st, q4_expert_cache *c,
                        int32_t layer, const float *x, float *y) {
    if (!g || !st || !c || !x || !y) return false;
    char n[Q4_MAX_NAME];
    snprintf(n, sizeof(n), "blk.%d.ffn_gate_inp.weight", layer);
    const q4_tensor *t_router = q4_find_tensor(g, n);
    snprintf(n, sizeof(n), "blk.%d.ffn_gate_inp_shexp.weight", layer);
    const q4_tensor *t_sgate = q4_find_tensor(g, n);
    snprintf(n, sizeof(n), "blk.%d.ffn_gate_shexp.weight", layer);
    const q4_tensor *t_gate_s = q4_find_tensor(g, n);
    snprintf(n, sizeof(n), "blk.%d.ffn_up_shexp.weight", layer);
    const q4_tensor *t_up_s = q4_find_tensor(g, n);
    snprintf(n, sizeof(n), "blk.%d.ffn_down_shexp.weight", layer);
    const q4_tensor *t_down_s = q4_find_tensor(g, n);
    if (!t_router || !t_gate_s || !t_up_s || !t_down_s) return false;
    return moe_body(g, c, layer, x, y, q4_store_get(st, t_router), t_router,
                    q4_store_dev(st, t_router), q4_store_get(st, t_sgate),
                    t_sgate, q4_store_get(st, t_gate_s), t_gate_s,
                    q4_store_dev(st, t_gate_s), q4_store_get(st, t_up_s),
                    t_up_s, q4_store_dev(st, t_up_s), q4_store_get(st, t_down_s),
                    t_down_s, q4_store_dev(st, t_down_s));
}
