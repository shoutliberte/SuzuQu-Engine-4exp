#define _GNU_SOURCE
#include "q4.h"

#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* CPU-side routed experts for the single-token decode path.
 *
 * The GPU spends most of a decode step waiting on PCIe H2D copies of expert
 * weights. DRAM bandwidth on the host is ~3x PCIe x4, so running the routed
 * FFNs on CPU cores while the GPU handles shared-expert + attention work is
 * a large net win. Weights are read straight out of the DRAM-resident expert
 * image (or L1/L2 slots; direct GGUF row pread as fallback) -- nothing is
 * dequantized eagerly, GEMVs are fused single-pass dot products.
 */

#define Q4_IQ_DECL static const
#include "q4_iq_grids.inc"

#define CPX_MAX_TH 16
#define CPX_CHUNK_A 64   /* rows per chunk, gate/up (n_ff) */
#define CPX_CHUNK_B 64   /* rows per chunk, down (n_embd) */
#define CPX_MAX_E 16

/* ---------------- fused row dot kernels (AVX2) ---------------- */

static inline float cpx_hsum(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
    return _mm_cvtss_f32(lo);
}

/* 8 signed grid bytes -> 8 floats with sign byte applied. */
static inline __m256 cpx_w8(uint64_t gv, uint32_t sg) {
    const __m256i kbit =
        _mm256_set_epi32(128, 64, 32, 16, 8, 4, 2, 1);
    const __m256 ksgn = _mm256_set1_ps(-0.0f);
    __m128i w8 = _mm_cvtsi64_si128((int64_t)gv);
    __m256 w = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w8));
    __m256i m = _mm256_cmpeq_epi32(
        _mm256_and_si256(_mm256_set1_epi32((int)sg), kbit), kbit);
    return _mm256_xor_ps(w, _mm256_and_ps(_mm256_castsi256_ps(m), ksgn));
}

/* Float view of the 1024-entry IQ2_S codebook. A plain 8-float load beats
 * the int8->int32->float convert chain; the table is 32 KiB and stays in L1.
 * Built once on first dispatch. */
static float cpx_fgrid2s[1024][8];
static int cpx_fgrid_ok;
static void cpx_fgrid_init(void) {
    if (cpx_fgrid_ok) return;
    for (int i = 0; i < 1024; i++)
        for (int j = 0; j < 8; j++)
            cpx_fgrid2s[i][j] = (int8_t)((q4_iq2s_grid[i] >> (8 * j)) & 0xff);
    __sync_synchronize();
    cpx_fgrid_ok = 1;
}

static inline __m256 cpx_fw8(int idx, uint32_t sg) {
    const __m256i kbit =
        _mm256_set_epi32(128, 64, 32, 16, 8, 4, 2, 1);
    const __m256 ksgn = _mm256_set1_ps(-0.0f);
    __m256 w = _mm256_loadu_ps(cpx_fgrid2s[idx]);
    __m256i m = _mm256_cmpeq_epi32(
        _mm256_and_si256(_mm256_set1_epi32((int)sg), kbit), kbit);
    return _mm256_xor_ps(w, _mm256_and_ps(_mm256_castsi256_ps(m), ksgn));
}

static float dot_iq2_s(const uint8_t *w, const float *x, uint64_t n) {
    /* One vector accumulator per 256-block: the per-sub-block scales are
     * folded in vectorially instead of paying a serial hsum every 32
     * elements (~4x fewer reductions). */
    __m256 va = _mm256_setzero_ps();
    const uint64_t nb = n / 256;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 82;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = q4_f16_to_f32(dh);
        const uint8_t *qs = blk + 2;
        const uint8_t *signs = blk + 34;
        const uint8_t *qh = blk + 66;
        const uint8_t *sc = blk + 74;
        const float *xb = x + i * 256;
        for (int ib = 0; ib < 8; ib++) {
            const float db0 = d * (0.5f + (float)(sc[ib] & 0xf)) * 0.25f;
            const float db1 = d * (0.5f + (float)(sc[ib] >> 4)) * 0.25f;
            __m256 v0 = _mm256_setzero_ps(), v1 = _mm256_setzero_ps();
            for (int l = 0; l < 4; l++) {
                int idx = qs[l] | ((qh[ib] << (8 - 2 * l)) & 0x300);
                __m256 wv = cpx_fw8(idx, signs[l]);
                __m256 xv = _mm256_loadu_ps(xb + ib * 32 + l * 8);
                if (l < 2) v0 = _mm256_fmadd_ps(wv, xv, v0);
                else v1 = _mm256_fmadd_ps(wv, xv, v1);
            }
            va = _mm256_fmadd_ps(_mm256_set1_ps(db0), v0, va);
            va = _mm256_fmadd_ps(_mm256_set1_ps(db1), v1, va);
            qs += 4;
            signs += 4;
        }
    }
    return cpx_hsum(va);
}

static float dot_iq3_s(const uint8_t *w, const float *x, uint64_t n) {
    __m256 va = _mm256_setzero_ps();
    const uint64_t nb = n / 256;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 110;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = q4_f16_to_f32(dh);
        const uint8_t *qs = blk + 2;
        const uint8_t *qh = blk + 66;
        const uint8_t *signs = blk + 74;
        const uint8_t *sc = blk + 106;
        const float *xb = x + i * 256;
        for (int ib = 0; ib < 4; ib++) {
            const float db1 = d * (1.f + 2.f * (float)(sc[ib] & 0xf));
            const float db2 = d * (1.f + 2.f * (float)(sc[ib] >> 4));
            __m256 v0 = _mm256_setzero_ps(), v1 = _mm256_setzero_ps();
            for (int half = 0; half < 2; half++) {
                const uint8_t qhb = qh[half];
                for (int l = 0; l < 4; l++) {
                    const int i1 = qs[2 * l] | ((qhb << (8 - 2 * l)) & 256);
                    const int i2 = qs[2 * l + 1] | ((qhb << (7 - 2 * l)) & 256);
                    uint64_t gv = (uint64_t)q4_iq3s_grid[i1] |
                                  ((uint64_t)q4_iq3s_grid[i2] << 32);
                    __m256 wv = cpx_w8(gv, signs[l]);
                    __m256 xv =
                        _mm256_loadu_ps(xb + ib * 64 + half * 32 + l * 8);
                    if (half == 0) v0 = _mm256_fmadd_ps(wv, xv, v0);
                    else v1 = _mm256_fmadd_ps(wv, xv, v1);
                }
                qs += 8;
                signs += 4;
            }
            va = _mm256_fmadd_ps(_mm256_set1_ps(db1), v0, va);
            va = _mm256_fmadd_ps(_mm256_set1_ps(db2), v1, va);
            qh += 2;
        }
    }
    return cpx_hsum(va);
}

/* IQ3_XXS: 98-byte block = f16 d + 64 grid bytes + 32 sign/scale bytes.
 * Same oct-vector trick as iq3_s: two uint32 grid entries make one w8. */
static float dot_iq3_xxs(const uint8_t *w, const float *x, uint64_t n) {
    __m256 va = _mm256_setzero_ps();
    const uint64_t nb = n / 256;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 98;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = q4_f16_to_f32(dh);
        const uint8_t *qs = blk + 2;
        const uint8_t *ss = qs + 64;
        const float *xb = x + i * 256;
        for (int ib = 0; ib < 8; ib++) {
            uint32_t aux32;
            memcpy(&aux32, ss + 4 * ib, 4);
            const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
            __m256 v = _mm256_setzero_ps();
            for (int l = 0; l < 4; l++) {
                const uint8_t sg = q4_ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
                const uint64_t gv = (uint64_t)q4_iq3xxs_grid[qs[2 * l]] |
                                    ((uint64_t)q4_iq3xxs_grid[qs[2 * l + 1]]
                                     << 32);
                v = _mm256_fmadd_ps(cpx_w8(gv, sg),
                                    _mm256_loadu_ps(xb + ib * 32 + l * 8), v);
            }
            va = _mm256_fmadd_ps(_mm256_set1_ps(db), v, va);
            qs += 8;
        }
    }
    return cpx_hsum(va);
}

static const int8_t cpx_iq4nl_kv[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

static inline __m256 cpx_iq4nl_vec(const uint8_t *blk, const float *xb,
                                   __m128i kv, __m128i m4) {
    __m128i qv = _mm_loadu_si128((const __m128i *)(blk + 2));
    __m128i lo = _mm_shuffle_epi8(kv, _mm_and_si128(qv, m4));
    __m128i hi =
        _mm_shuffle_epi8(kv, _mm_and_si128(_mm_srli_epi16(qv, 4), m4));
    __m256 v = _mm256_mul_ps(
        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)), _mm256_loadu_ps(xb));
    v = _mm256_fmadd_ps(
        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))),
        _mm256_loadu_ps(xb + 8), v);
    v = _mm256_fmadd_ps(
        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)),
        _mm256_loadu_ps(xb + 16), v);
    v = _mm256_fmadd_ps(
        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8))),
        _mm256_loadu_ps(xb + 24), v);
    return v;
}

static float dot_iq4_nl(const uint8_t *w, const float *x, uint64_t n) {
    __m256 va = _mm256_setzero_ps();
    const uint64_t nb = n / 32;
    const __m128i kv = _mm_loadu_si128((const __m128i *)cpx_iq4nl_kv);
    const __m128i m4 = _mm_set1_epi8(0x0f);
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 18;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        va = _mm256_fmadd_ps(_mm256_set1_ps(q4_f16_to_f32(dh)),
                             cpx_iq4nl_vec(blk, x + i * 32, kv, m4), va);
    }
    return cpx_hsum(va);
}

/* IQ4_XS: 136-byte block = f16 d + u16 scales_h + scales_l[4] + qs[128];
 * per-ib (32 vals): same nibble table as IQ4_NL but a 6-bit scale. */
static float dot_iq4_xs(const uint8_t *w, const float *x, uint64_t n) {
    __m256 va = _mm256_setzero_ps();
    const uint64_t nb = n / 256;
    const __m128i kv = _mm_loadu_si128((const __m128i *)cpx_iq4nl_kv);
    const __m128i m4 = _mm_set1_epi8(0x0f);
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 136;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = q4_f16_to_f32(dh);
        uint16_t sh;
        memcpy(&sh, blk + 2, 2);
        const uint8_t *sl = blk + 4;
        const uint8_t *qs = blk + 8;
        const float *xb = x + i * 256;
        for (int ib = 0; ib < 8; ib++) {
            const int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0xf) |
                           ((int)((sh >> (2 * ib)) & 3) << 4);
            const float dl = d * (float)(ls - 32);
            __m128i qv = _mm_loadu_si128((const __m128i *)(qs + ib * 16));
            __m128i lo = _mm_shuffle_epi8(kv, _mm_and_si128(qv, m4));
            __m128i hi = _mm_shuffle_epi8(
                kv, _mm_and_si128(_mm_srli_epi16(qv, 4), m4));
            __m256 v = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)),
                _mm256_loadu_ps(xb + ib * 32));
            v = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(
                    _mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))),
                _mm256_loadu_ps(xb + ib * 32 + 8), v);
            v = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)),
                _mm256_loadu_ps(xb + ib * 32 + 16), v);
            v = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(
                    _mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8))),
                _mm256_loadu_ps(xb + ib * 32 + 24), v);
            va = _mm256_fmadd_ps(_mm256_set1_ps(dl), v, va);
        }
    }
    return cpx_hsum(va);
}

static float dot_q8_0(const uint8_t *w, const float *x, uint64_t n) {
    __m256 va = _mm256_setzero_ps();
    const uint64_t nb = n / 32;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 34;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = q4_f16_to_f32(dh);
        const int8_t *qs = (const int8_t *)(blk + 2);
        const float *xb = x + i * 32;
        __m256 v = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                _mm_loadl_epi64((const __m128i *)qs))),
            _mm256_loadu_ps(xb));
        v = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                _mm_loadl_epi64((const __m128i *)(qs + 8)))),
            _mm256_loadu_ps(xb + 8), v);
        v = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                _mm_loadl_epi64((const __m128i *)(qs + 16)))),
            _mm256_loadu_ps(xb + 16), v);
        v = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                _mm_loadl_epi64((const __m128i *)(qs + 24)))),
            _mm256_loadu_ps(xb + 24), v);
        va = _mm256_fmadd_ps(_mm256_set1_ps(d), v, va);
    }
    return cpx_hsum(va);
}

/* q4_0: same nibble geometry as iq4_nl (lo->0..15, hi->16..31), linear
 * (nibble - 8) map instead of the kvalues table. */
static float dot_q4_0(const uint8_t *w, const float *x, uint64_t n) {
    __m256 va = _mm256_setzero_ps();
    const uint64_t nb = n / 32;
    const __m128i m4 = _mm_set1_epi8(0x0f);
    const __m128i m8 = _mm_set1_epi8(8);
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 18;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = q4_f16_to_f32(dh);
        const float *xb = x + i * 32;
        __m128i qv = _mm_loadu_si128((const __m128i *)(blk + 2));
        __m128i lo = _mm_sub_epi8(_mm_and_si128(qv, m4), m8);
        __m128i hi =
            _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(qv, 4), m4), m8);
        __m256 v = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)),
            _mm256_loadu_ps(xb));
        v = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))),
            _mm256_loadu_ps(xb + 8), v);
        v = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)),
            _mm256_loadu_ps(xb + 16), v);
        v = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8))),
            _mm256_loadu_ps(xb + 24), v);
        va = _mm256_fmadd_ps(_mm256_set1_ps(d), v, va);
    }
    return cpx_hsum(va);
}

/* q5_0: q4_0 nibbles (at +6) + 32 high bits (u32 at +2, bit i = elem i),
 * offset -16. */
static float dot_q5_0(const uint8_t *w, const float *x, uint64_t n) {
    __m256 va = _mm256_setzero_ps();
    const uint64_t nb = n / 32;
    const __m128i m4 = _mm_set1_epi8(0x0f);
    const __m128i m16 = _mm_set1_epi8(16);
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 22;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = q4_f16_to_f32(dh);
        uint32_t qh;
        memcpy(&qh, blk + 2, 4);
        const float *xb = x + i * 32;
        __m128i qv = _mm_loadu_si128((const __m128i *)(blk + 6));
        /* high bits: elems 0..15 = qh bits 0..15 (lo nibbles),
         * 16..31 = bits 16..31 (hi nibbles) */
        __m128i hlo = _mm_set_epi8(
            (int8_t)((qh >> 15) & 1), (int8_t)((qh >> 14) & 1),
            (int8_t)((qh >> 13) & 1), (int8_t)((qh >> 12) & 1),
            (int8_t)((qh >> 11) & 1), (int8_t)((qh >> 10) & 1),
            (int8_t)((qh >> 9) & 1), (int8_t)((qh >> 8) & 1),
            (int8_t)((qh >> 7) & 1), (int8_t)((qh >> 6) & 1),
            (int8_t)((qh >> 5) & 1), (int8_t)((qh >> 4) & 1),
            (int8_t)((qh >> 3) & 1), (int8_t)((qh >> 2) & 1),
            (int8_t)((qh >> 1) & 1), (int8_t)(qh & 1));
        __m128i hhi = _mm_set_epi8(
            (int8_t)((qh >> 31) & 1), (int8_t)((qh >> 30) & 1),
            (int8_t)((qh >> 29) & 1), (int8_t)((qh >> 28) & 1),
            (int8_t)((qh >> 27) & 1), (int8_t)((qh >> 26) & 1),
            (int8_t)((qh >> 25) & 1), (int8_t)((qh >> 24) & 1),
            (int8_t)((qh >> 23) & 1), (int8_t)((qh >> 22) & 1),
            (int8_t)((qh >> 21) & 1), (int8_t)((qh >> 20) & 1),
            (int8_t)((qh >> 19) & 1), (int8_t)((qh >> 18) & 1),
            (int8_t)((qh >> 17) & 1), (int8_t)((qh >> 16) & 1));
        __m128i lo = _mm_sub_epi8(
            _mm_or_si128(_mm_and_si128(qv, m4), _mm_slli_epi16(hlo, 4)), m16);
        __m128i hi = _mm_sub_epi8(
            _mm_or_si128(_mm_and_si128(_mm_srli_epi16(qv, 4), m4),
                         _mm_slli_epi16(hhi, 4)),
            m16);
        __m256 v = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)),
            _mm256_loadu_ps(xb));
        v = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))),
            _mm256_loadu_ps(xb + 8), v);
        v = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)),
            _mm256_loadu_ps(xb + 16), v);
        v = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8))),
            _mm256_loadu_ps(xb + 24), v);
        va = _mm256_fmadd_ps(_mm256_set1_ps(d), v, va);
    }
    return cpx_hsum(va);
}

/* q2_0: 64-elem block, fp16 d + 16B of 2-bit quads (byte j = elems 4j..4j+3,
 * LSB-first), values (q - 1) * d. */
static float dot_q2_0(const uint8_t *w, const float *x, uint64_t n) {
    __m256 va = _mm256_setzero_ps();
    const uint64_t nb = n / 64;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 18;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = q4_f16_to_f32(dh);
        const uint8_t *qs = blk + 2;
        const float *xb = x + i * 64;
        int8_t buf[64];
        for (int j = 0; j < 16; j++) {
            const uint32_t qv = qs[j];
            buf[4 * j + 0] = (int8_t)((int)((qv >> 0) & 3) - 1);
            buf[4 * j + 1] = (int8_t)((int)((qv >> 2) & 3) - 1);
            buf[4 * j + 2] = (int8_t)((int)((qv >> 4) & 3) - 1);
            buf[4 * j + 3] = (int8_t)((int)((qv >> 6) & 3) - 1);
        }
        __m256 v = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                _mm_loadl_epi64((const __m128i *)buf))),
            _mm256_loadu_ps(xb));
        for (int k = 1; k < 8; k++)
            v = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                    _mm_loadl_epi64((const __m128i *)(buf + 8 * k)))),
                _mm256_loadu_ps(xb + 8 * k), v);
        va = _mm256_fmadd_ps(_mm256_set1_ps(d), v, va);
    }
    return cpx_hsum(va);
}

static float dot_f16(const uint8_t *w, const float *x, uint64_t n) {
    const uint16_t *h = (const uint16_t *)w;
    __m256 v = _mm256_setzero_ps();
    uint64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i hw = _mm_loadu_si128((const __m128i *)(h + i));
        v = _mm256_fmadd_ps(_mm256_cvtph_ps(hw), _mm256_loadu_ps(x + i), v);
    }
    float acc = cpx_hsum(v);
    for (; i < n; i++) acc += q4_f16_to_f32(h[i]) * x[i];
    return acc;
}

static float dot_bf16(const uint8_t *w, const float *x, uint64_t n) {
    const uint16_t *h = (const uint16_t *)w;
    __m256 v = _mm256_setzero_ps();
    uint64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i lo = _mm_loadu_si128((const __m128i *)(h + i));
        __m256i w32 = _mm256_cvtepu16_epi32(lo);
        __m256 wf = _mm256_castsi256_ps(_mm256_slli_epi32(w32, 16));
        v = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x + i), v);
    }
    float acc = cpx_hsum(v);
    for (; i < n; i++) acc += q4_bf16_to_f32(h[i]) * x[i];
    return acc;
}

static float dot_f32(const uint8_t *w, const float *x, uint64_t n) {
    const float *fw = (const float *)w;
    __m256 v0 = _mm256_setzero_ps(), v1 = _mm256_setzero_ps();
    uint64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        v0 = _mm256_fmadd_ps(_mm256_loadu_ps(fw + i), _mm256_loadu_ps(x + i),
                             v0);
        v1 = _mm256_fmadd_ps(_mm256_loadu_ps(fw + i + 8),
                             _mm256_loadu_ps(x + i + 8), v1);
    }
    for (; i + 8 <= n; i += 8)
        v0 = _mm256_fmadd_ps(_mm256_loadu_ps(fw + i), _mm256_loadu_ps(x + i),
                             v0);
    float acc = cpx_hsum(_mm256_add_ps(v0, v1));
    for (; i < n; i++) acc += fw[i] * x[i];
    return acc;
}

/* Generic fallback: dequant a row then plain dot. */
static float dot_generic(uint32_t type, const uint8_t *w, const float *x,
                         uint64_t n, float *buf) {
    if (n > 8192) return 0.f;
    q4_dequant_row(type, w, n, buf);
    float s = 0.f;
    for (uint64_t i = 0; i < n; i++) s += buf[i] * x[i];
    return s;
}

static float cpx_row_dot(uint32_t type, const uint8_t *w, const float *x,
                         uint64_t ncols, float *dbuf) {
    switch (type) {
    case Q4_T_IQ2_S:  return dot_iq2_s(w, x, ncols);
    case Q4_T_IQ3_S:  return dot_iq3_s(w, x, ncols);
    case Q4_T_IQ4_NL: return dot_iq4_nl(w, x, ncols);
    case Q4_T_IQ3_XXS: return dot_iq3_xxs(w, x, ncols);
    case Q4_T_IQ4_XS:  return dot_iq4_xs(w, x, ncols);
    case Q4_T_Q4_0:   return dot_q4_0(w, x, ncols);
    case Q4_T_Q5_0:   return dot_q5_0(w, x, ncols);
    case Q4_T_Q2_0:   return dot_q2_0(w, x, ncols);
    case Q4_T_Q8_0:   return dot_q8_0(w, x, ncols);
    case Q4_T_F16:    return dot_f16(w, x, ncols);
    case Q4_T_BF16:   return dot_bf16(w, x, ncols);
    case Q4_T_F32:    return dot_f32(w, x, ncols);
    default:          return dot_generic(type, w, x, ncols, dbuf);
    }
}

/* ---------------- int8-activation kernels (Strata-style) ----------------
 * The float path above decodes each weight to fp32 lanes and FMAs against
 * raw x.  The int8 path instead quantizes the activation ONCE per job
 * (x for gate/up, h for down) into per-32-element groups of {i8 codes,
 * fp32 scale, i32 code sum} and computes each 32-value dot in integer:
 * maddubs(u8 magnitudes, i8 activations) -> madd(i16 scales) -> i32, one
 * cvt+fmadd per 32-elem group.  Same format bytes, ~3x fewer ops per
 * weight element; rounding shifts the dot by ~1% (activation quant noise)
 * — gated by Q4_CPXQ8 (default on, =0 restores the float path).
 * Structure follows Strata's iq_avx2.cpp: grid magnitudes are the unsigned
 * maddubs operand, sign bits are folded into the activation via sign_epi8,
 * per-16/per-32 weight scales ride in the i16 madd multiplier. */

typedef struct {
    int8_t  q[32];
    float   d;   /* amax/127 */
    int32_t bs;  /* sum(q): offset-correction formats (q2_0/q4_0/...) */
} cpx_q8;        /* 40 B per 32 elements */

static inline int32_t cpx_hsum_i32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    lo = _mm_add_epi32(lo, _mm256_extracti128_si256(v, 1));
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0x4e));
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0xb1));
    return _mm_cvtsi128_si32(lo);
}

/* x[n] -> o[n/32]; n%32 must be 0 (checked at dispatch).  Rounding is
 * cvtps' nearest-even — matches ggml's 12582912 magic.  amax==0 emits
 * d=0/q=0/bs=0 which contributes nothing to any dot. */
static void cpx_quant8(const float *x, uint64_t n, cpx_q8 *o) {
    const __m256 mabs =
        _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256i ord = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    const __m256i one8 = _mm256_set1_epi8(1);
    const __m256i one16 = _mm256_set1_epi16(1);
    for (uint64_t g = 0; g < n / 32; g++) {
        const float *xb = x + 32 * g;
        __m256 x0 = _mm256_loadu_ps(xb), x1 = _mm256_loadu_ps(xb + 8);
        __m256 x2 = _mm256_loadu_ps(xb + 16), x3 = _mm256_loadu_ps(xb + 24);
        __m256 am = _mm256_max_ps(
            _mm256_max_ps(_mm256_and_ps(x0, mabs), _mm256_and_ps(x1, mabs)),
            _mm256_max_ps(_mm256_and_ps(x2, mabs), _mm256_and_ps(x3, mabs)));
        __m128 m = _mm_max_ps(_mm256_castps256_ps128(am),
                              _mm256_extractf128_ps(am, 1));
        m = _mm_max_ps(m, _mm_movehl_ps(m, m));
        m = _mm_max_ps(m, _mm_movehdup_ps(m));
        const float a = _mm_cvtss_f32(m);
        o[g].d = a / 127.f;
        const __m256 sc = _mm256_set1_ps(a > 0.f ? 127.f / a : 0.f);
        __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(x0, sc));
        __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(x1, sc));
        __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(x2, sc));
        __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(x3, sc));
        __m256i p = _mm256_packs_epi16(
            _mm256_packs_epi32(i0, i1), _mm256_packs_epi32(i2, i3));
        p = _mm256_permutevar8x32_epi32(p, ord);
        _mm256_storeu_si256((__m256i *)o[g].q, p);
        o[g].bs =
            cpx_hsum_i32(_mm256_madd_epi16(_mm256_maddubs_epi16(one8, p),
                                           one16));
    }
}

/* ksigns_iq2xs as sign bytes: byte k = 0xFF when bit k set, else 0x01 —
 * four scalar u64 loads build a whole 32-value sign vector (the IQ3_XXS
 * aux word holds four 7-bit sign indices).  Lazy init, like cpx_fgrid2s. */
static uint64_t cpx_esigns[128];
static int cpx_esigns_ok;
static void cpx_esigns_init(void) {
    if (cpx_esigns_ok) return;
    for (int i = 0; i < 128; i++) {
        uint64_t r = 0;
        for (int k = 0; k < 8; k++)
            r |= (uint64_t)(((q4_ksigns_iq2xs[i] >> k) & 1) ? 0xFF : 0x01)
                 << (8 * k);
        cpx_esigns[i] = r;
    }
    __sync_synchronize();
    cpx_esigns_ok = 1;
}

/* 32 sign bits -> 32 bytes of -1/+1 (ggml bit_selector pattern). */
static inline __m256i cpx_sgn_vec(uint32_t m) {
    const __m128i bm = _mm_set1_epi32((int)m);
    const __m128i lo = _mm_shuffle_epi8(
        bm, _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1));
    const __m128i hi = _mm_shuffle_epi8(
        bm, _mm_setr_epi8(2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3));
    const __m256i bits = _mm256_inserti128_si256(_mm256_castsi128_si256(lo),
                                                 hi, 1);
    const __m256i sel = _mm256_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, (char)0x80, 1, 2, 4, 8, 16, 32, 64,
        (char)0x80, 1, 2, 4, 8, 16, 32, 64, (char)0x80, 1, 2, 4, 8, 16,
        32, 64, (char)0x80);
    const __m256i nz = _mm256_cmpeq_epi8(_mm256_and_si256(bits, sel), sel);
    return _mm256_or_si256(nz, _mm256_set1_epi8(1));
}

/* i16 lane scales: lanes 0-7 = a (vals 0-15), 8-15 = b (16-31). */
static inline __m256i cpx_sc16(int a, int b) {
    return _mm256_inserti128_si256(
        _mm256_castsi128_si256(_mm_set1_epi16((short)a)),
        _mm_set1_epi16((short)b), 1);
}

static inline uint32_t cpx_u32(const uint8_t *p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}
static inline uint64_t cpx_u64(const uint8_t *p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

/* One 32-elem group: g = u8 magnitudes, sgn = i8 ±1 applied to the
 * activation, sc = i16 weight scales -> i32 pair sums. */
static inline __m256i cpx_dot32(__m256i g, __m256i sgn, __m256i sc,
                                const int8_t *yq) {
    const __m256i yv = _mm256_loadu_si256((const __m256i *)yq);
    const __m256i ys = _mm256_sign_epi8(yv, sgn);
    return _mm256_madd_epi16(_mm256_maddubs_epi16(g, ys), sc);
}

/* Same, when the decoded values are already signed i8 (IQ4_NL/IQ4_XS
 * kvalues): magnitude via abs, sign applied to the activation.  Requires
 * no zero values (kvalues min |v| = 1) — sign(_,0)=0 would zero a product
 * that should be v*y... v=0 only pairs y with 0 which is correct anyway. */
static inline __m256i cpx_dot32_s(__m256i v, const int8_t *yq) {
    const __m256i yv = _mm256_loadu_si256((const __m256i *)yq);
    return _mm256_maddubs_epi16(_mm256_abs_epi8(v),
                                _mm256_sign_epi8(yv, v));
}

static int cpx_q8ok(uint32_t t) {
    switch (t) {
    case Q4_T_IQ2_S:
    case Q4_T_IQ3_S:
    case Q4_T_IQ3_XXS:
    case Q4_T_IQ4_XS:
    case Q4_T_IQ4_NL:
    case Q4_T_Q4_0:
    case Q4_T_Q5_0:
    case Q4_T_Q8_0:
    case Q4_T_Q2_0:
        return 1;
    }
    return 0;
}

static int cpx_q8_env(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q4_CPXQ8");
        v = !(e && e[0] == '0');
    }
    return v;
}

/* IQ2_S 82B block: qs[32] grid bytes + qs[32] sign bytes + qh[8] + sc[8].
 * Per 32-val group ib (j,half): 4 grid lookups, sign u32, two 16-scales. */
static float dq8_iq2_s(const uint8_t *w, const cpx_q8 *x, uint64_t n) {
    __m256 accf = _mm256_setzero_ps();
    const uint64_t nb = n / 256;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 82;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = q4_f16_to_f32(dh) * 0.125f;
        const uint8_t *qs = blk + 2;
        const uint8_t *qh = blk + 66;
        for (int j = 0; j < 4; j++) {
            const uint64_t m = cpx_u64(blk + 34 + 8 * j);
            for (int half = 0; half < 2; half++) {
                const int o = 4 * half;
                const uint32_t h = qh[2 * j + half];
                const __m256i g = _mm256_set_epi64x(
                    (long long)q4_iq2s_grid[qs[8 * j + o + 3] |
                                            ((h << 2) & 0x300)],
                    (long long)q4_iq2s_grid[qs[8 * j + o + 2] |
                                            ((h << 4) & 0x300)],
                    (long long)q4_iq2s_grid[qs[8 * j + o + 1] |
                                            ((h << 6) & 0x300)],
                    (long long)q4_iq2s_grid[qs[8 * j + o] |
                                            ((h << 8) & 0x300)]);
                const __m256i sgn =
                    cpx_sgn_vec(half ? (uint32_t)(m >> 32) : (uint32_t)m);
                const uint8_t sb = blk[74 + 2 * j + half];
                const __m256i sc =
                    cpx_sc16(2 * (sb & 15) + 1, 2 * (sb >> 4) + 1);
                const cpx_q8 *xg = x + 8 * i + 2 * j + half;
                accf = _mm256_fmadd_ps(
                    _mm256_set1_ps(d * xg->d),
                    _mm256_cvtepi32_ps(cpx_dot32(g, sgn, sc, xg->q)),
                    accf);
            }
        }
    }
    return cpx_hsum(accf);
}

/* IQ3_S 110B: qs[64] (u32 grid per byte), qh[8], signs[32], scales[4]. */
static float dq8_iq3_s(const uint8_t *w, const cpx_q8 *x, uint64_t n) {
    __m256 accf = _mm256_setzero_ps();
    const uint64_t nb = n / 256;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 110;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = q4_f16_to_f32(dh);
        for (int j = 0; j < 4; j++) {
            const uint8_t *q = blk + 2 + 16 * j;
            const uint64_t m = cpx_u64(blk + 74 + 8 * j);
            const uint8_t s = blk[106 + j];
            for (int half = 0; half < 2; half++) {
                const uint8_t *qb = q + 8 * half;
                const uint32_t h = blk[66 + 2 * j + half];
                /* 4 u64 lanes = 8 u32 grid entries: 4 merged loads beat
                 * set_epi32's per-lane insert chain. */
#define G3(k) (uint64_t)q4_iq3s_grid[qb[k] | (((h >> k) & 1u) << 8)]
                const __m256i g = _mm256_set_epi64x(
                    (long long)(G3(6) | (G3(7) << 32)),
                    (long long)(G3(4) | (G3(5) << 32)),
                    (long long)(G3(2) | (G3(3) << 32)),
                    (long long)(G3(0) | (G3(1) << 32)));
#undef G3
                const __m256i sgn =
                    cpx_sgn_vec(half ? (uint32_t)(m >> 32) : (uint32_t)m);
                const __m256i sc = _mm256_set1_epi16(
                    (short)(half ? 2 * (s >> 4) + 1 : 2 * (s & 15) + 1));
                const cpx_q8 *xg = x + 8 * i + 2 * j + half;
                accf = _mm256_fmadd_ps(
                    _mm256_set1_ps(d * xg->d),
                    _mm256_cvtepi32_ps(cpx_dot32(g, sgn, sc, xg->q)),
                    accf);
            }
        }
    }
    return cpx_hsum(accf);
}

/* IQ3_XXS 98B: 64 grid bytes + 8 aux u32 (4x 7-bit sign idx + 4-bit sc). */
static float dq8_iq3_xxs(const uint8_t *w, const cpx_q8 *x, uint64_t n) {
    __m256 accf = _mm256_setzero_ps();
    const uint64_t nb = n / 256;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 98;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = q4_f16_to_f32(dh) * 0.25f;
        for (int j = 0; j < 4; j++) {
            for (int half = 0; half < 2; half++) {
                const uint8_t *q = blk + 2 + 16 * j + 8 * half;
                const uint32_t aw =
                    cpx_u32(blk + 2 + 64 + 8 * j + 4 * half);
#define G3X(k) (uint64_t)q4_iq3xxs_grid[q[k]]
                const __m256i g = _mm256_set_epi64x(
                    (long long)(G3X(6) | (G3X(7) << 32)),
                    (long long)(G3X(4) | (G3X(5) << 32)),
                    (long long)(G3X(2) | (G3X(3) << 32)),
                    (long long)(G3X(0) | (G3X(1) << 32)));
#undef G3X
                const __m256i sgn = _mm256_set_epi64x(
                    (long long)cpx_esigns[(aw >> 21) & 127],
                    (long long)cpx_esigns[(aw >> 14) & 127],
                    (long long)cpx_esigns[(aw >> 7) & 127],
                    (long long)cpx_esigns[aw & 127]);
                const __m256i sc =
                    _mm256_set1_epi16((short)(2 * (int)(aw >> 28) + 1));
                const cpx_q8 *xg = x + 8 * i + 2 * j + half;
                accf = _mm256_fmadd_ps(
                    _mm256_set1_ps(d * xg->d),
                    _mm256_cvtepi32_ps(cpx_dot32(g, sgn, sc, xg->q)),
                    accf);
            }
        }
    }
    return cpx_hsum(accf);
}

/* IQ4_NL nibble LUT -> 32 signed values in one 256 reg (lo 16 vals in the
 * low lane, hi in the high). */
static inline __m256i cpx_iq4nl_i8(const uint8_t *q16, __m128i kv,
                                   __m128i m4) {
    const __m128i qv = _mm_loadu_si128((const __m128i *)q16);
    const __m128i lo = _mm_shuffle_epi8(kv, _mm_and_si128(qv, m4));
    const __m128i hi =
        _mm_shuffle_epi8(kv, _mm_and_si128(_mm_srli_epi16(qv, 4), m4));
    return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

/* IQ4_XS 136B: u16 scales_h + scales_l[4] + qs[128], per-32 6-bit scale. */
static float dq8_iq4_xs(const uint8_t *w, const cpx_q8 *x, uint64_t n) {
    __m256 accf = _mm256_setzero_ps();
    const uint64_t nb = n / 256;
    const __m128i kv = _mm_loadu_si128((const __m128i *)cpx_iq4nl_kv);
    const __m128i m4 = _mm_set1_epi8(0x0f);
    const __m256i one16 = _mm256_set1_epi16(1);
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 136;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = q4_f16_to_f32(dh);
        uint16_t sh;
        memcpy(&sh, blk + 2, 2);
        const uint8_t *sl = blk + 4;
        const uint8_t *qs = blk + 8;
        for (int ib = 0; ib < 8; ib++) {
            const int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0xf) |
                           ((int)((sh >> (2 * ib)) & 3) << 4);
            const __m256i v = cpx_iq4nl_i8(qs + 16 * ib, kv, m4);
            const cpx_q8 *xg = x + 8 * i + ib;
            const __m256i p = cpx_dot32_s(v, xg->q);
            accf = _mm256_fmadd_ps(
                _mm256_set1_ps(d * xg->d),
                _mm256_cvtepi32_ps(
                    _mm256_madd_epi16(p, _mm256_set1_epi16((short)(ls - 32)))),
                accf);
        }
    }
    (void)one16;
    return cpx_hsum(accf);
}

/* IQ4_NL 18B block: f16 d + 16 nibble bytes, no sub-scale. */
static float dq8_iq4_nl(const uint8_t *w, const cpx_q8 *x, uint64_t n) {
    __m256 accf = _mm256_setzero_ps();
    const uint64_t nb = n / 32;
    const __m128i kv = _mm_loadu_si128((const __m128i *)cpx_iq4nl_kv);
    const __m128i m4 = _mm_set1_epi8(0x0f);
    const __m256i one16 = _mm256_set1_epi16(1);
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 18;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const __m256i v = cpx_iq4nl_i8(blk + 2, kv, m4);
        const cpx_q8 *xg = x + i;
        accf = _mm256_fmadd_ps(
            _mm256_set1_ps(q4_f16_to_f32(dh) * xg->d),
            _mm256_cvtepi32_ps(_mm256_madd_epi16(cpx_dot32_s(v, xg->q),
                                                 one16)),
            accf);
    }
    return cpx_hsum(accf);
}

/* Offset formats: unsigned codes c in u8, value = (c - off) * d, so
 * dot = d*(dx*sum(c*q8) - off*dx*bs).  v' u8 codes x i8 activations. */
static inline __m256i cpx_dot32_u(__m256i v, const int8_t *yq) {
    const __m256i yv = _mm256_loadu_si256((const __m256i *)yq);
    return _mm256_madd_epi16(_mm256_maddubs_epi16(v, yv),
                             _mm256_set1_epi16(1));
}

static float dq8_q4_0(const uint8_t *w, const cpx_q8 *x, uint64_t n) {
    __m256 accf = _mm256_setzero_ps();
    float cs = 0.f;
    const uint64_t nb = n / 32;
    const __m128i m4 = _mm_set1_epi8(0x0f);
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 18;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const __m128i qv = _mm_loadu_si128((const __m128i *)(blk + 2));
        const __m128i lo = _mm_and_si128(qv, m4);
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(qv, 4), m4);
        const __m256i v =
            _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
        const cpx_q8 *xg = x + i;
        const float dd = q4_f16_to_f32(dh) * xg->d;
        accf = _mm256_fmadd_ps(_mm256_set1_ps(dd),
                               _mm256_cvtepi32_ps(cpx_dot32_u(v, xg->q)),
                               accf);
        cs += dd * (float)xg->bs;
    }
    return cpx_hsum(accf) - 8.f * cs;
}

static float dq8_q5_0(const uint8_t *w, const cpx_q8 *x, uint64_t n) {
    __m256 accf = _mm256_setzero_ps();
    float cs = 0.f;
    const uint64_t nb = n / 32;
    const __m128i m4 = _mm_set1_epi8(0x0f);
    const __m128i m1 = _mm_set1_epi8(0x10);
    const __m128i sel16 = _mm_setr_epi8(1, 2, 4, 8, 16, 32, 64,
                                      (char)0x80, 1, 2, 4, 8, 16, 32,
                                      64, (char)0x80);
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 22;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        uint32_t qh;
        memcpy(&qh, blk + 2, 4);
        const __m128i qv = _mm_loadu_si128((const __m128i *)(blk + 6));
        /* high bits -> byte lanes: elems 0-15 use qh bits 0-15, 16-31 use
         * bits 16-31.  Broadcast the u16s then test each bit position. */
        const __m128i hv = _mm_set1_epi32((int)qh);
        const __m128i hb_lo = _mm_and_si128(
            _mm_shuffle_epi8(hv, _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1,
                                               1, 1, 1, 1, 1, 1, 1)),
            sel16);
        const __m128i hb_hi = _mm_and_si128(
            _mm_shuffle_epi8(hv, _mm_setr_epi8(2, 2, 2, 2, 2, 2, 2, 2, 3,
                                               3, 3, 3, 3, 3, 3, 3)),
            sel16);
        const __m128i lo = _mm_or_si128(
            _mm_and_si128(qv, m4),
            _mm_and_si128(_mm_cmpeq_epi8(hb_lo, sel16), m1));
        const __m128i hi = _mm_or_si128(
            _mm_and_si128(_mm_srli_epi16(qv, 4), m4),
            _mm_and_si128(_mm_cmpeq_epi8(hb_hi, sel16), m1));
        const __m256i v =
            _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
        const cpx_q8 *xg = x + i;
        const float dd = q4_f16_to_f32(dh) * xg->d;
        accf = _mm256_fmadd_ps(_mm256_set1_ps(dd),
                               _mm256_cvtepi32_ps(cpx_dot32_u(v, xg->q)),
                               accf);
        cs += dd * (float)xg->bs;
    }
    return cpx_hsum(accf) - 16.f * cs;
}

static float dq8_q8_0(const uint8_t *w, const cpx_q8 *x, uint64_t n) {
    __m256 accf = _mm256_setzero_ps();
    const uint64_t nb = n / 32;
    const __m256i one16 = _mm256_set1_epi16(1);
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 34;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        /* w is already signed i8: abs/sign into the activation.  w = -128
         * wraps abs to 0x80 = u8 128 — still the correct magnitude. */
        const __m256i v =
            _mm256_loadu_si256((const __m256i *)(blk + 2));
        const cpx_q8 *xg = x + i;
        accf = _mm256_fmadd_ps(
            _mm256_set1_ps(q4_f16_to_f32(dh) * xg->d),
            _mm256_cvtepi32_ps(
                _mm256_madd_epi16(cpx_dot32_s(v, xg->q), one16)),
            accf);
    }
    return cpx_hsum(accf);
}

/* Q2_0 18B block: f16 d + 16 bytes of 2-bit quads = 64 vals, two groups.
 * Byte b -> u32 [c0,c1,c2,c3] via a 256-entry unpack table (1 KiB, L1). */
static uint32_t cpx_q20tab[256];
static int cpx_q20tab_ok;
static void cpx_q20tab_init(void) {
    if (cpx_q20tab_ok) return;
    for (int b = 0; b < 256; b++)
        cpx_q20tab[b] = (uint32_t)((b >> 0) & 3) |
                        ((uint32_t)((b >> 2) & 3) << 8) |
                        ((uint32_t)((b >> 4) & 3) << 16) |
                        ((uint32_t)((b >> 6) & 3) << 24);
    __sync_synchronize();
    cpx_q20tab_ok = 1;
}

static float dq8_q2_0(const uint8_t *w, const cpx_q8 *x, uint64_t n) {
    __m256 accf = _mm256_setzero_ps();
    float cs = 0.f;
    const uint64_t nb = n / 64;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + i * 18;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = q4_f16_to_f32(dh);
        const uint8_t *qs = blk + 2;
        const __m256i v0 = _mm256_setr_epi32(
            (int)cpx_q20tab[qs[0]], (int)cpx_q20tab[qs[1]],
            (int)cpx_q20tab[qs[2]], (int)cpx_q20tab[qs[3]],
            (int)cpx_q20tab[qs[4]], (int)cpx_q20tab[qs[5]],
            (int)cpx_q20tab[qs[6]], (int)cpx_q20tab[qs[7]]);
        const __m256i v1 = _mm256_setr_epi32(
            (int)cpx_q20tab[qs[8]], (int)cpx_q20tab[qs[9]],
            (int)cpx_q20tab[qs[10]], (int)cpx_q20tab[qs[11]],
            (int)cpx_q20tab[qs[12]], (int)cpx_q20tab[qs[13]],
            (int)cpx_q20tab[qs[14]], (int)cpx_q20tab[qs[15]]);
        for (int h = 0; h < 2; h++) {
            const cpx_q8 *xg = x + 2 * i + h;
            const float dd = d * xg->d;
            accf = _mm256_fmadd_ps(
                _mm256_set1_ps(dd),
                _mm256_cvtepi32_ps(
                    cpx_dot32_u(h ? v1 : v0, xg->q)),
                accf);
            cs += dd * (float)xg->bs;
        }
    }
    return cpx_hsum(accf) - cs;
}

static int cpx_dot8(uint32_t type, const uint8_t *w, const cpx_q8 *xq,
                    uint64_t n, float *out) {
    switch (type) {
    case Q4_T_IQ2_S:   *out = dq8_iq2_s(w, xq, n);   return 1;
    case Q4_T_IQ3_S:   *out = dq8_iq3_s(w, xq, n);   return 1;
    case Q4_T_IQ3_XXS: *out = dq8_iq3_xxs(w, xq, n); return 1;
    case Q4_T_IQ4_XS:  *out = dq8_iq4_xs(w, xq, n);  return 1;
    case Q4_T_IQ4_NL:  *out = dq8_iq4_nl(w, xq, n);  return 1;
    case Q4_T_Q4_0:    *out = dq8_q4_0(w, xq, n);    return 1;
    case Q4_T_Q5_0:    *out = dq8_q5_0(w, xq, n);    return 1;
    case Q4_T_Q8_0:    *out = dq8_q8_0(w, xq, n);    return 1;
    case Q4_T_Q2_0:    *out = dq8_q2_0(w, xq, n);    return 1;
    }
    return 0;
}

/* ---------------- job + worker pool ---------------- */

typedef struct {
    const q4_gguf *g;
    q4_expert_cache *c;
    int32_t layer;
    const float *x;
    int32_t ids[CPX_MAX_E];
    float wts[CPX_MAX_E];
    uint32_t k;
    float *y;                     /* [n_embd] routed result */
    float *h;                     /* [k][n_ff] shared intermediates */
    const q4_tensor *tg, *tu, *td;
    uint64_t og, ou, od;
    uint32_t n_embd, n_ff;
    uint32_t chunks_a;            /* k * ceil(n_ff/CHUNK_A) */
    uint32_t chunks_b;            /* k * ceil(n_embd/CHUNK_B) */
    _Atomic uint32_t next_a;
    _Atomic uint32_t next_be[CPX_MAX_E]; /* per-expert phase-B cursors */
    _Atomic uint32_t a_done_e[CPX_MAX_E];/* per-expert phase-A progress */
    _Atomic uint32_t b_done;
    _Atomic int      hold;        /* workers pinned on this job object */
    _Atomic uint64_t gen;         /* == tag of the live generation */
    int ok;
    /* int8-activation path: x/h quantized to cpx_q8 groups at dispatch /
     * at the last phase-A chunk.  q8a: gate/up use xq8.  q8b: down uses
     * hq8[e] once hq_rdy[e] is published by the worker that completed
     * that expert's phase A. */
    int q8a, q8b;
    const cpx_q8 *xq8;            /* [n_embd/32] per-slot buffer */
    cpx_q8 *hq8;                  /* [k][n_ff/32] per-slot buffer */
    uint32_t hgn;                 /* n_ff/32 */
    _Atomic uint32_t hq_rdy[CPX_MAX_E];
} cpx_job;

/* Multi-slot job ring: the device-route dispatcher publishes one job per
 * MoE layer per token; with a single job slot each layer's ~200µs compute
 * would serialize (~5 ms/token).  Slots let several layers' jobs overlap —
 * workers sweep all slots claiming chunks, so a job queued behind an active
 * one still starts as soon as a worker frees up. */
#define CPX_SLOTS 8

typedef struct {
    pthread_t       th[CPX_MAX_TH];
    int             n_th;
    int             quit;
    pthread_mutex_t mu;
    pthread_cond_t  cv_job;
    pthread_cond_t  cv_done;
    uint64_t        seq;          /* global issue counter (host side only) */
    /* Job objects are static (one per slot): workers pin a slot with
     * jobs[].hold++ then verify gen — stale pointers can never dangle
     * because the objects are never freed. */
    cpx_job         jobs[CPX_SLOTS];
    _Atomic int     live_a[CPX_SLOTS];   /* published (release) */
    _Atomic uint64_t tag_a[CPX_SLOTS];   /* publish stamp (seq) per slot */
    _Atomic uint64_t done_a[CPX_SLOTS];  /* completion stamp per slot */
    /* in-order completion floor for the legacy global wait word: slots
     * finish out of order, so comp[] tracks which seqs are done and the
     * floor advances only across contiguous completions. */
    uint64_t        comp[64];
    uint64_t        done_floor;
    /* per-slot intermediates (jobs overlap — h cannot be shared) */
    float          *hbuf[CPX_SLOTS];
    size_t          hsz[CPX_SLOTS];
    /* per-slot quantized activations (xq8 = n_embd/32 groups,
     * hq8 = k * n_ff/32 groups) */
    cpx_q8         *xqbuf[CPX_SLOTS];
    size_t          xqsz[CPX_SLOTS];
    cpx_q8         *hqbuf[CPX_SLOTS];
    size_t          hqsz[CPX_SLOTS];
    /* per-worker scratch for the pread fallback path */
    uint8_t        *scratch[CPX_MAX_TH];
    uint32_t        scratch_sz[CPX_MAX_TH];
} cpx_pool;

static cpx_pool g_pool;
static int g_pool_init;
static int g_n_threads = -1;

/* Pinned seq words shared with the GPU wait kernel (set by the MoE layer,
 * which owns the host-registered block).  The GPU spins on done >= expect
 * instead of the host joining, so both must stay monotonic. */
static volatile unsigned long long *g_cpx_expect;
static volatile unsigned long long *g_cpx_done;

void q4_cpuex_syncpin(unsigned long long *expect,
                      unsigned long long *done) {
    g_cpx_expect = expect;
    g_cpx_done = done;
}

static int cpx_threads(void) {
    if (g_n_threads >= 0) return g_n_threads;
    const char *e = getenv("Q4_CPU_THREADS");
    int v = e && e[0] ? atoi(e) : 12;
    if (v < 1) v = 1;
    if (v > CPX_MAX_TH) v = CPX_MAX_TH;
    long np = sysconf(_SC_NPROCESSORS_ONLN);
    if (np > 0 && v > np - 1) v = (int)(np - 1);
    if (v < 1) v = 1;
    g_n_threads = v;
    return v;
}

/* Rows [r0,r1) of `part` for expert eid. pack = packed expert base or NULL
 * (then read the needed rows straight out of the GGUF). */
static const uint8_t *cpx_part_rows(cpx_job *j, int w, const q4_tensor *t,
                                    uint64_t part_off_in_pack,
                                    int32_t eid, uint32_t r0, uint32_t r1) {
    const uint64_t rb = q4_row_bytes(t->ggml_type, t->ne[0]);
    const uint64_t slice = t->nbytes / (uint64_t)t->n_experts;
    const uint8_t *pack = q4_expert_host_ptr(j->c, j->layer, eid);
    if (pack) return pack + part_off_in_pack + (uint64_t)r0 * rb;
    /* Fallback: pread only the rows this chunk needs. */
    uint64_t need = (uint64_t)(r1 - r0) * rb;
    cpx_pool *p = &g_pool;
    if (p->scratch_sz[w] < need) {
        free(p->scratch[w]);
        p->scratch[w] = malloc((size_t)need);
        p->scratch_sz[w] = p->scratch[w] ? (uint32_t)need : 0;
    }
    if (!p->scratch[w]) return NULL;
    const q4_file *f = &j->g->files[t->shard];
    uint64_t off = f->data_off + t->offset + slice * (uint64_t)eid +
                   (uint64_t)r0 * rb;
    uint8_t *dst = p->scratch[w];
    uint64_t got = 0;
    while (got < need) {
        ssize_t r = pread(f->fd, dst + got, (size_t)(need - got), (off_t)(off + got));
        if (r <= 0) return NULL;
        got += (uint64_t)r;
    }
    return dst;
}

static inline float cpx_silu(float v) { return v / (1.f + expf(-v)); }

/* Atomic float add: per-expert phase-B chunks overlap the same y range
 * across experts, so the accumulation must be race-free. */
static inline void cpx_fadd(float *p, float v) {
    uint32_t old;
    memcpy(&old, p, 4);
    for (;;) {
        float nv;
        uint32_t nw;
        memcpy(&nv, &old, 4);
        nv += v;
        memcpy(&nw, &nv, 4);
        if (__atomic_compare_exchange_n((uint32_t *)p, &old, nw, true,
                                        __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            return;
    }
}

static void cpx_chunk_a(cpx_job *j, int w, uint32_t i) {
    const uint32_t cpa = (j->n_ff + CPX_CHUNK_A - 1) / CPX_CHUNK_A;
    const uint32_t e = i / cpa;
    const uint32_t r0 = (i % cpa) * CPX_CHUNK_A;
    const uint32_t r1 = r0 + CPX_CHUNK_A < j->n_ff ? r0 + CPX_CHUNK_A : j->n_ff;
    const int32_t eid = j->ids[e];
    const uint8_t *wg = cpx_part_rows(j, w, j->tg, j->og, eid, r0, r1);
    const uint8_t *wu = cpx_part_rows(j, w, j->tu, j->ou, eid, r0, r1);
    if (!wg || !wu) {
        j->ok = 0;
        return;
    }
    const uint64_t rbg = q4_row_bytes(j->tg->ggml_type, j->tg->ne[0]);
    const uint64_t rbu = q4_row_bytes(j->tu->ggml_type, j->tu->ne[0]);
    float *h = j->h + (size_t)e * j->n_ff;
    float dbuf[8192];
    for (uint32_t r = r0; r < r1; r++) {
        float gv, uv;
        if (j->q8a) {
            if (!cpx_dot8(j->tg->ggml_type,
                          wg + (size_t)(r - r0) * rbg, j->xq8, j->n_embd,
                          &gv))
                gv = cpx_row_dot(j->tg->ggml_type,
                                 wg + (size_t)(r - r0) * rbg, j->x,
                                 j->n_embd, dbuf);
            if (!cpx_dot8(j->tu->ggml_type,
                          wu + (size_t)(r - r0) * rbu, j->xq8, j->n_embd,
                          &uv))
                uv = cpx_row_dot(j->tu->ggml_type,
                                 wu + (size_t)(r - r0) * rbu, j->x,
                                 j->n_embd, dbuf);
        } else {
            gv = cpx_row_dot(j->tg->ggml_type,
                             wg + (size_t)(r - r0) * rbg, j->x, j->n_embd,
                             dbuf);
            uv = cpx_row_dot(j->tu->ggml_type,
                             wu + (size_t)(r - r0) * rbu, j->x, j->n_embd,
                             dbuf);
        }
        h[r] = cpx_silu(gv) * uv;
    }
}

/* Phase-B chunk for a single expert: gated on that expert's phase A so
 * the down-projection of an early-finishing expert overlaps the gate/up
 * work of the later ones. */
static void cpx_chunk_be(cpx_job *j, int w, uint32_t e, uint32_t i) {
    const uint32_t r0 = i * CPX_CHUNK_B;
    const uint32_t r1 = r0 + CPX_CHUNK_B < j->n_embd ? r0 + CPX_CHUNK_B
                                                    : j->n_embd;
    const uint64_t rb = q4_row_bytes(j->td->ggml_type, j->td->ne[0]);
    const uint64_t ncols = j->td->ne[0];
    float dbuf[8192];
    const uint8_t *wd =
        cpx_part_rows(j, w, j->td, j->od, j->ids[e], r0, r1);
    if (!wd) {
        j->ok = 0;
        return;
    }
    const float *h = j->h + (size_t)e * j->n_ff;
    const cpx_q8 *hq = j->hq8 + (size_t)e * j->hgn;
    const float wt = j->wts[e];
    for (uint32_t r = r0; r < r1; r++) {
        float dv;
        if (j->q8b &&
            cpx_dot8(j->td->ggml_type, wd + (size_t)(r - r0) * rb, hq,
                     ncols, &dv))
            cpx_fadd(&j->y[r], wt * dv);
        else
            cpx_fadd(&j->y[r],
                     wt * cpx_row_dot(j->td->ggml_type,
                                      wd + (size_t)(r - r0) * rb, h,
                                      ncols, dbuf));
    }
}

#define CPX_SPIN 200000 /* ~1 ms of _mm_pause before falling back to sleep */

/* Advance the in-order completion floor and publish it to the pinned word
 * the GPU wait kernel spins on.  Called (under p->mu) by the worker that
 * finishes a job's last chunk. */
static void cpx_note_complete(cpx_pool *p, uint64_t seq) {
    pthread_mutex_lock(&p->mu);
    p->comp[seq & 63] = seq;
    while (p->comp[(p->done_floor + 1) & 63] == p->done_floor + 1) {
        p->done_floor++;
        p->comp[p->done_floor & 63] = 0;
    }
    if (g_cpx_done)
        __atomic_store_n(g_cpx_done, p->done_floor, __ATOMIC_RELEASE);
    pthread_cond_broadcast(&p->cv_done);
    pthread_mutex_unlock(&p->mu);
}

static int cpx_any_job(cpx_pool *p) {
    for (int s = 0; s < CPX_SLOTS; s++)
        if (atomic_load_explicit(&p->live_a[s], memory_order_acquire) &&
            atomic_load_explicit(&p->done_a[s], memory_order_acquire) !=
                atomic_load_explicit(&p->tag_a[s], memory_order_relaxed))
            return 1;
    return 0;
}

static void *cpx_worker(void *arg) {
    intptr_t w = (intptr_t)arg;
    cpx_pool *p = &g_pool;
    for (;;) {
        int progressed = 0;
        for (int s = 0; s < CPX_SLOTS; s++) {
            uint64_t tag =
                atomic_load_explicit(&p->tag_a[s], memory_order_acquire);
            if (!atomic_load_explicit(&p->live_a[s],
                                      memory_order_acquire) ||
                atomic_load_explicit(&p->done_a[s],
                                     memory_order_acquire) == tag)
                continue;
            cpx_job *j = &p->jobs[s];
            /* Pin the object, then re-verify the generation: the dispatcher
             * may have retired/republished the slot between our tag read and
             * the pin. */
            atomic_fetch_add_explicit(&j->hold, 1, memory_order_seq_cst);
            if (atomic_load_explicit(&j->gen, memory_order_acquire) != tag) {
                atomic_fetch_sub(&j->hold, 1);
                continue;
            }
            const uint32_t cpa =
                (j->n_ff + CPX_CHUNK_A - 1) / CPX_CHUNK_A;
            const uint32_t cpb =
                (j->n_embd + CPX_CHUNK_B - 1) / CPX_CHUNK_B;
            for (;;) {
                uint32_t i = atomic_fetch_add_explicit(&j->next_a, 1,
                                                       memory_order_relaxed);
                if (i >= j->chunks_a) break;
                cpx_chunk_a(j, (int)w, i);
                progressed = 1;
                /* Per-expert A progress releases that expert's B chunks.
                 * The worker that posts the LAST A chunk also quantizes
                 * that expert's h for the int8 dot path, then publishes
                 * hq_rdy — B chunks gate on it so they never race ahead
                 * of the quantization. */
                const uint32_t ei = i / cpa;
                if (atomic_fetch_add_explicit(&j->a_done_e[ei], 1,
                                              memory_order_release) +
                        1 ==
                        cpa &&
                    j->q8b) {
                    cpx_quant8(j->h + (size_t)ei * j->n_ff, j->n_ff,
                               j->hq8 + (size_t)ei * j->hgn);
                    atomic_store_explicit(&j->hq_rdy[ei], 1,
                                          memory_order_release);
                }
            }
            /* Claim one B chunk at a time from the first B-ready expert,
             * then rescan: an expert's down projection can start as soon
             * as its own gate/up chunks finish. */
            for (;;) {
                int claimed = 0;
                for (uint32_t e = 0; e < j->k && !claimed; e++) {
                    if (atomic_load_explicit(&j->a_done_e[e],
                                           memory_order_acquire) != cpa)
                        continue;
                    if (j->q8b &&
                        !atomic_load_explicit(&j->hq_rdy[e],
                                              memory_order_acquire))
                        continue;
                    uint32_t i = atomic_fetch_add_explicit(&j->next_be[e],
                                                           1,
                                                           memory_order_relaxed);
                    if (i >= cpb) continue;
                    cpx_chunk_be(j, (int)w, e, i);
                    progressed = 1;
                    claimed = 1;
                    if (atomic_fetch_add(&j->b_done, 1) + 1 == j->chunks_b) {
                        atomic_store_explicit(&p->done_a[s], tag,
                                              memory_order_release);
                        cpx_note_complete(p, tag);
                    }
                }
                if (!claimed) break;
            }
            atomic_fetch_sub(&j->hold, 1);
        }
        if (progressed) continue;
        if (p->quit) return NULL;
        /* Idle: spin briefly (jobs arrive every ~0.5 ms and a condvar wakeup
         * costs tens of us), then sleep until a dispatch broadcasts. */
        uint64_t spins = 0;
        while (spins < CPX_SPIN && !p->quit && !cpx_any_job(p)) {
            _mm_pause();
            spins++;
        }
        if (p->quit) return NULL;
        if (cpx_any_job(p)) continue;
        pthread_mutex_lock(&p->mu);
        while (!p->quit && !cpx_any_job(p))
            pthread_cond_wait(&p->cv_job, &p->mu);
        pthread_mutex_unlock(&p->mu);
        if (p->quit) return NULL;
    }
}

static bool cpx_pool_start(void) {
    if (g_pool_init) return g_pool.n_th > 0;
    g_pool_init = 1;
    memset(&g_pool, 0, sizeof(g_pool));
    pthread_mutex_init(&g_pool.mu, NULL);
    pthread_cond_init(&g_pool.cv_job, NULL);
    pthread_cond_init(&g_pool.cv_done, NULL);
    int want = cpx_threads();
    for (int i = 0; i < want; i++) {
        if (pthread_create(&g_pool.th[g_pool.n_th], NULL, cpx_worker,
                           (void *)(intptr_t)g_pool.n_th) != 0)
            break;
        g_pool.n_th++;
    }
    fprintf(stderr, "q4: cpuexp pool %d threads\n", g_pool.n_th);
    return g_pool.n_th > 0;
}

/* Publish a job into a free slot.  Returns the slot index, -1 on transient
 * backpressure (all slots busy — callers retry after polling/reaping) or
 * -2 on a hard error.  Intermediates are per-slot so overlapping jobs never
 * share h. */
int q4_cpuex_dispatch2(const q4_gguf *g, q4_expert_cache *c, int32_t layer,
                       const float *x, const int32_t *ids, const float *wts,
                       uint32_t k, float *y, uint64_t *tag_out) {
    if (!g || !c || !x || !ids || !wts || !y || !cpx_pool_start())
        return -2;
    cpx_fgrid_init();
    const q4_tensor *tg, *tu, *td;
    uint64_t og, ou, od, tot;
    if (!q4_expert_parts(g, layer, &tg, &tu, &td, &og, &ou, &od, &tot))
        return -2;
    const uint32_t n_embd = g->n_embd;
    const uint32_t n_ff = g->n_ff_exp ? g->n_ff_exp : 640;
    if (k == 0 || k > CPX_MAX_E) return -2;
    cpx_pool *p = &g_pool;
    pthread_mutex_lock(&p->mu);
    int s = -1;
    for (int i = 0; i < CPX_SLOTS; i++) {
        /* A slot is reusable once its last generation completed and no
         * worker still holds the static job object. */
        if ((!atomic_load_explicit(&p->live_a[i], memory_order_acquire) ||
             atomic_load_explicit(&p->done_a[i], memory_order_acquire) ==
                 atomic_load_explicit(&p->tag_a[i], memory_order_relaxed)) &&
            atomic_load_explicit(&p->jobs[i].hold, memory_order_acquire) ==
                0) {
            s = i;
            break;
        }
    }
    if (s < 0) {
        pthread_mutex_unlock(&p->mu);
        return -1;
    }
    const size_t hneed = (size_t)k * n_ff * sizeof(float);
    if (p->hsz[s] < hneed) {
        free(p->hbuf[s]);
        p->hbuf[s] = (float *)malloc(hneed);
        p->hsz[s] = p->hbuf[s] ? hneed : 0;
    }
    const int q8a = cpx_q8_env() && (n_embd % 32) == 0 &&
                    cpx_q8ok(tg->ggml_type) && cpx_q8ok(tu->ggml_type);
    const int q8b = q8a && (n_ff % 32) == 0 && cpx_q8ok(td->ggml_type);
    const size_t xqneed = q8a ? (n_embd / 32) * sizeof(cpx_q8) : 0;
    const size_t hqneed = q8b ? (size_t)k * (n_ff / 32) * sizeof(cpx_q8)
                              : 0;
    if (p->xqsz[s] < xqneed) {
        free(p->xqbuf[s]);
        p->xqbuf[s] = (cpx_q8 *)malloc(xqneed);
        p->xqsz[s] = p->xqbuf[s] ? xqneed : 0;
    }
    if (p->hqsz[s] < hqneed) {
        free(p->hqbuf[s]);
        p->hqbuf[s] = (cpx_q8 *)malloc(hqneed);
        p->hqsz[s] = p->hqbuf[s] ? hqneed : 0;
    }
    cpx_job *j = &p->jobs[s];
    j->g = g;
    j->c = c;
    j->layer = layer;
    j->x = x;
    j->k = k;
    memcpy(j->ids, ids, k * sizeof(int32_t));
    memcpy(j->wts, wts, k * sizeof(float));
    j->y = y;
    j->h = p->hbuf[s];
    if (!j->h) {
        pthread_mutex_unlock(&p->mu);
        return -2;
    }
    j->tg = tg;
    j->tu = tu;
    j->td = td;
    j->og = og;
    j->ou = ou;
    j->od = od;
    j->n_embd = n_embd;
    j->n_ff = n_ff;
    j->chunks_a = k * ((n_ff + CPX_CHUNK_A - 1) / CPX_CHUNK_A);
    j->chunks_b = k * ((n_embd + CPX_CHUNK_B - 1) / CPX_CHUNK_B);
    atomic_store_explicit(&j->next_a, 0, memory_order_relaxed);
    memset(j->next_be, 0, k * sizeof j->next_be[0]);
    memset(j->a_done_e, 0, k * sizeof j->a_done_e[0]);
    atomic_store_explicit(&j->b_done, 0, memory_order_relaxed);
    j->ok = 1;
    j->q8a = q8a && p->xqbuf[s];
    j->q8b = q8b && p->xqbuf[s] && p->hqbuf[s];
    j->xq8 = j->q8a ? p->xqbuf[s] : NULL;
    j->hq8 = j->q8b ? p->hqbuf[s] : NULL;
    j->hgn = n_ff / 32;
    memset(j->hq_rdy, 0, k * sizeof j->hq_rdy[0]);
    if (j->q8a) {
        cpx_fgrid_init();
        cpx_esigns_init();
        cpx_q20tab_init();
        cpx_quant8(x, n_embd, p->xqbuf[s]);
    }
    memset(y, 0, (size_t)n_embd * sizeof(float));
    const uint64_t tag = ++p->seq;
    atomic_store_explicit(&j->gen, tag, memory_order_relaxed);
    atomic_store_explicit(&p->tag_a[s], tag, memory_order_relaxed);
    /* live last: workers see the fully-initialized job. */
    atomic_store_explicit(&p->live_a[s], 1, memory_order_release);
    if (g_cpx_expect)
        __atomic_store_n(g_cpx_expect, tag, __ATOMIC_RELEASE);
    pthread_cond_broadcast(&p->cv_job);
    pthread_mutex_unlock(&p->mu);
    if (tag_out) *tag_out = tag;
    return s;
}

/* Tag-aware completion poll: callers that hold a dispatch tag must NOT
 * poll done_a == tag_a — the slot may have been retired and republished
 * (tag_a advanced) while the polled job itself already completed. done_a
 * is monotone (tags come from one global seq), so >= is the right test. */
bool q4_cpuex_tag_done(int slot, uint64_t tag) {
    cpx_pool *p = &g_pool;
    if (slot < 0 || slot >= CPX_SLOTS || !tag) return false;
    return atomic_load_explicit(&p->done_a[slot], memory_order_acquire) >=
           tag;
}

/* Wait on a specific dispatch tag: after slot reuse, tag_a advances past
 * the caller's tag — waiting for done_a == (re-read tag) could spin forever
 * (done_a overshoots old tags). Compare >= the captured tag instead. Only
 * clears live when this tag is still the slot's current job — clearing a
 * NEWER job's live flag would leak a running job's slot for re-dispatch. */
bool q4_cpuex_join_tag(int slot, uint64_t tag) {
    cpx_pool *p = &g_pool;
    if (slot < 0 || slot >= CPX_SLOTS || !tag) return false;
    uint64_t spins = 0;
    while (atomic_load_explicit(&p->done_a[slot], memory_order_acquire) <
               tag &&
           spins < CPX_SPIN) {
        _mm_pause();
        spins++;
    }
    pthread_mutex_lock(&p->mu);
    while (atomic_load_explicit(&p->done_a[slot], memory_order_acquire) <
           tag)
        pthread_cond_wait(&p->cv_done, &p->mu);
    if (atomic_load_explicit(&p->tag_a[slot], memory_order_relaxed) == tag)
        atomic_store_explicit(&p->live_a[slot], 0, memory_order_release);
    int ok = p->jobs[slot].ok;
    pthread_mutex_unlock(&p->mu);
    return ok != 0;
}

/* Non-blocking completion poll for a dispatched slot. */
bool q4_cpuex_slot_done(int slot) {
    cpx_pool *p = &g_pool;
    if (slot < 0 || slot >= CPX_SLOTS) return false;
    return atomic_load_explicit(&p->done_a[slot], memory_order_acquire) ==
               atomic_load_explicit(&p->tag_a[slot], memory_order_relaxed) &&
           atomic_load_explicit(&p->tag_a[slot], memory_order_relaxed) != 0;
}

/* Wait for a dispatched slot, retire it, return its ok flag. */
bool q4_cpuex_join2(int slot) {
    cpx_pool *p = &g_pool;
    if (slot < 0 || slot >= CPX_SLOTS) return false;
    uint64_t tag =
        atomic_load_explicit(&p->tag_a[slot], memory_order_acquire);
    uint64_t spins = 0;
    while (atomic_load_explicit(&p->done_a[slot], memory_order_acquire) !=
               tag &&
           spins < CPX_SPIN) {
        _mm_pause();
        spins++;
    }
    pthread_mutex_lock(&p->mu);
    while (atomic_load_explicit(&p->done_a[slot], memory_order_acquire) !=
           tag)
        pthread_cond_wait(&p->cv_done, &p->mu);
    atomic_store_explicit(&p->live_a[slot], 0, memory_order_release);
    int ok = p->jobs[slot].ok;
    pthread_mutex_unlock(&p->mu);
    return ok != 0;
}

/* Wedge diagnostic: dump each slot's publish/complete stamps and the job's
 * internal progress so a stuck job shows WHERE it stopped (A vs B phase,
 * which expert's gate/up never finished). */
void q4_cpuex_diag(void) {
    cpx_pool *p = &g_pool;
    fprintf(stderr, "q4: cpuex floor=%llu seq=%llu slots:",
            (unsigned long long)p->done_floor,
            (unsigned long long)p->seq);
    for (int s = 0; s < CPX_SLOTS; s++) {
        int lv = atomic_load_explicit(&p->live_a[s], memory_order_acquire);
        uint64_t tg = atomic_load_explicit(&p->tag_a[s],
                                           memory_order_relaxed);
        uint64_t dn = atomic_load_explicit(&p->done_a[s],
                                           memory_order_acquire);
        if (!lv && !tg) continue;
        cpx_job *j = &p->jobs[s];
        fprintf(stderr, " [s%d live=%d tag=%llu done=%llu A %u/%u B %u/%u]",
                s, lv, (unsigned long long)tg, (unsigned long long)dn,
                (unsigned)atomic_load(&j->next_a), (unsigned)j->chunks_a,
                (unsigned)atomic_load(&j->b_done), (unsigned)j->chunks_b);
    }
    fprintf(stderr, "\n");
}

/* Retire finished slots so their next dispatch is immediate; returns how
 * many slots are still running. */
int q4_cpuex_retire(void) {
    cpx_pool *p = &g_pool;
    int live = 0;
    for (int s = 0; s < CPX_SLOTS; s++) {
        if (!atomic_load_explicit(&p->live_a[s], memory_order_acquire))
            continue;
        if (atomic_load_explicit(&p->done_a[s], memory_order_acquire) ==
            atomic_load_explicit(&p->tag_a[s], memory_order_relaxed))
            atomic_store_explicit(&p->live_a[s], 0, memory_order_release);
        else
            live++;
    }
    return live;
}

bool q4_cpuex_reap(void) {
    /* Slots self-retire; kept for source compatibility. */
    return true;
}

/* ---------------- selftest ----------------
 * Exercises every int8-activation dot against the float path on random
 * valid encodings (random payload bytes + a proper fp16 scale).  The int8
 * path quantizes x, so it can never be bit-equal — the bound checked is
 * activation-quantization noise: |dq8 - f32| <= tol * (1 + |f32|). */
int q4_cpx_selftest(void) {
    static const struct {
        uint32_t type;
        uint64_t bsz;      /* bytes per super-block */
        uint64_t nblocks;  /* blocks at n=512 */
    } cases[] = {
        {Q4_T_IQ2_S,   82,  2},
        {Q4_T_IQ3_S,   110, 2},
        {Q4_T_IQ3_XXS, 98,  2},
        {Q4_T_IQ4_XS,  136, 2},
        {Q4_T_IQ4_NL,  18,  16},
        {Q4_T_Q4_0,    18,  16},
        {Q4_T_Q5_0,    22,  16},
        {Q4_T_Q8_0,    34,  16},
        {Q4_T_Q2_0,    18,  8},
    };
    const uint64_t n = 512;
    cpx_fgrid_init();
    cpx_esigns_init();
    cpx_q20tab_init();
    uint64_t st = 0x9e3779b97f4a7c15ull;
    float x[512];
    cpx_q8 xq[512 / 32];
    uint8_t w[34 * 16];   /* largest case: Q8_0, 544 B for n=512 */
    float wf[512];
    float dbuf[8192];
    int bad = 0;
    for (int trial = 0; trial < 4; trial++) {
        for (uint64_t i = 0; i < n; i++) {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            x[i] = ((float)(int)(st >> 40) / (float)(1 << 22)) * 0.5f;
        }
        cpx_quant8(x, n, xq);
        for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
            const uint64_t nbytes = cases[c].bsz * cases[c].nblocks;
            for (uint64_t b = 0; b < nbytes; b++) {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                w[b] = (uint8_t)(st >> 56);
            }
            /* fp16 block scales: d = 0.25 + blk/8 keeps magnitudes sane. */
            for (uint64_t blk = 0; blk < cases[c].nblocks; blk++) {
                uint16_t h16 = _cvtss_sh(0.25f + 0.125f * (float)blk, 0);
                memcpy(w + blk * cases[c].bsz, &h16, 2);
            }
            float ref = cpx_row_dot(cases[c].type, w, x, n, dbuf);
            float got = 0.f;
            if (!cpx_dot8(cases[c].type, w, xq, n, &got)) {
                fprintf(stderr, "cpx q8: type %u not wired\n",
                        cases[c].type);
                bad = 1;
                continue;
            }
            /* Error bound: |x - q*d| <= d/2 per element, so the dot error
             * is bounded by sum |w_i| * dx/2 — absolute, not relative. */
            if (!q4_dequant_row(cases[c].type, w, n, wf)) {
                bad = 1;
                continue;
            }
            float lim = 1e-3f;
            for (uint64_t i = 0; i < n; i++)
                lim += fabsf(wf[i]) * (xq[i / 32].d * 0.5f);
            float err = fabsf(got - ref);
            if (!(err <= lim)) {
                fprintf(stderr,
                        "cpx q8: type %u trial %d got %g ref %g"
                        " (err %g > %g)\n",
                        cases[c].type, trial, got, ref, err, lim);
                bad = 1;
            }
        }
    }
    /* quantization round-trip sanity: dequantized codes track x. */
    cpx_quant8(x, n, xq);
    for (uint64_t g = 0; g < n / 32; g++)
        for (int i = 0; i < 32; i++) {
            float xr = (float)xq[g].q[i] * xq[g].d;
            if (fabsf(xr - x[g * 32 + i]) > xq[g].d * 0.51f + 1e-6f) {
                fprintf(stderr, "cpx q8: quant g%llu i%d\n",
                        (unsigned long long)g, i);
                bad = 1;
            }
        }
    if (!bad) printf("ok cpx q8 kernels\n");
    return bad;
}

/* Microbenchmark: row-dot throughput of the float path vs the int8
 * path per format, at n_embd-like row length.  Usage: q4-test --cpxbench */
int q4_cpx_bench(void) {
    static const struct {
        uint32_t type;
        uint64_t bsz, nvals;
        const char *name;
    } cases[] = {
        {Q4_T_IQ2_S,   82,  256, "iq2_s"},
        {Q4_T_IQ3_S,   110, 256, "iq3_s"},
        {Q4_T_IQ3_XXS, 98,  256, "iq3_xxs"},
        {Q4_T_IQ4_XS,  136, 256, "iq4_xs"},
        {Q4_T_IQ4_NL,  18,  32,  "iq4_nl"},
        {Q4_T_Q4_0,    18,  32,  "q4_0"},
        {Q4_T_Q5_0,    22,  32,  "q5_0"},
        {Q4_T_Q8_0,    34,  32,  "q8_0"},
        {Q4_T_Q2_0,    18,  64,  "q2_0"},
        {Q4_T_F16,     2,   1,   "f16"},
    };
    const uint64_t n = 2560;   /* n_embd for the current models */
    const int iters = 40000;
    cpx_fgrid_init();
    cpx_esigns_init();
    cpx_q20tab_init();
    static uint8_t w[8192];
    static float x[2560];
    static cpx_q8 xq[2560 / 32];
    static float dbuf[8192];
    uint64_t st = 42;
    for (uint64_t i = 0; i < n; i++) {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        x[i] = ((float)(int)(st >> 40) / (float)(1 << 22)) * 0.5f;
    }
    cpx_quant8(x, n, xq);
    printf("%-8s %10s %10s %8s\n", "type", "f32 GB/s", "q8 GB/s", "speedup");
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        const uint64_t nb = n / cases[c].nvals;
        const uint64_t rb = nb * cases[c].bsz;
        if (rb > sizeof(w)) continue;
        for (uint64_t b = 0; b < rb; b++) {
            st ^= st << 13; st ^= st >> 7; st ^= st << 17;
            w[b] = (uint8_t)(st >> 56);
        }
        for (uint64_t blk = 0; blk < nb; blk++) {
            uint16_t h16 = _cvtss_sh(0.25f, 0);
            memcpy(w + blk * cases[c].bsz, &h16, 2);
        }
        if (cases[c].type == Q4_T_F16) {
            for (uint64_t i = 0; i < n; i++) {
                uint16_t h16 = _cvtss_sh(x[i] * 0.01f, 0);
                memcpy(w + 2 * i, &h16, 2);
            }
        }
        volatile float sink = 0;
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int it = 0; it < iters; it++)
            sink += cpx_row_dot(cases[c].type, w, x, n, dbuf);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double t_f = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
        float probe;
        if (!cpx_dot8(cases[c].type, w, xq, n, &probe)) {
            /* no int8 path for this format — don't report a fake speedup */
            double gf = rb * (double)iters / t_f / 1e9;
            printf("%-8s %10.1f %10s %8s\n", cases[c].name, gf, "n/a", "n/a");
            continue;
        }
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int it = 0; it < iters; it++) {
            float o;
            if (!cpx_dot8(cases[c].type, w, xq, n, &o)) o = 0.f;
            sink += o;
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double t_q = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
        double gf = rb * (double)iters / t_f / 1e9;
        double gq = rb * (double)iters / t_q / 1e9;
        printf("%-8s %10.1f %10.1f %7.2fx\n", cases[c].name, gf, gq,
               gq / gf);
        (void)sink;
    }
    return 0;
}
