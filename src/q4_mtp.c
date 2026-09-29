/* MTP (multi-token prediction) draft head for Qwen3.8-Flash-Next.
 *
 * Loads an Unsloth `mtp-...-shared-Q8_0.gguf` nextn head (arch qwen4exp,
 * blk.<n_layer-1> = one full MoE layer + eh_proj glue) and runs speculative
 * decoding on top of the q4 session: the head drafts up to N tokens, the
 * main model verifies them in one batched pass. Verification is exact, so
 * output matches non-MTP decoding; only the speed changes.
 *
 * Head forward (matches ggml-org/llama.cpp PR #28243, graph_mtp):
 *   e = rms_norm(emb(tok)) * enorm            (n_embd)
 *   h = grouped_rms(h_state) * hnorm          (hc x n_embd, per stream)
 *   res[c] = eh_proj([e | h[c]])              (concat order: e first)
 *   then one standard layer: hc_attn mix -> dense GQA attention -> MoE
 *   -> hc_ffn mix; res' = next hidden estimate,
 *   logits = output( hc_head_mixer(res') ).
 * The shared head borrows token_embd + output from the main model.
 *
 * Head KV: one QSA-style linear buffer. Entries carry their own rope
 * position; draft-chain entries (estimates) are overwritten by the repair
 * pass with entries computed from the real residuals, so the invariant is:
 * slots [0, kv_fill) always describe the accepted sequence.
 */

#include "q4.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MTP_WS_N (1u << 20)      /* d_ws floats */
#define MTP_PWS_N (32u << 20)    /* batched-MoE scratch floats (128 MiB) */
#define MTP_MAX_ROWS 8           /* max tokens per batched head pass */
#define MTP_MAX_DRAFT 3

/* d_ws regions (floats). d_resb lives in its own buffer because it must
 * survive every helper call within a batched pass. Worst case n = 8 rows:
 * XN needs n*(3*hc_dim+hc_rank) = 330240, QG needs n*25600 = 204800. */
#define MTP_OFF_XN 0             /* hc mix: n*(3*hc_dim + hc_rank) */
#define MTP_OFF_QG 344064        /* attention: n*25600 */
#define MTP_OFF_MIXED 557056     /* n*n_embd */
#define MTP_OFF_BLOCK 589824     /* n*n_embd */
#define MTP_OFF_INJ 622592       /* n*hc */
/* [655360, MTP_WS_N) unused headroom; the batched MoE gets m->d_pws. */

struct q4_mtp {
    q4_gguf g;            /* head gguf (nextn layer + glue) */
    q4_store *st;         /* head dense + shared expert store (VRAM) */
    q4_expert_cache *ex;  /* head routed experts */
    const q4_gguf *gm;    /* main model (token_embd, output, hparams) */
    q4_store *stm;
    int32_t layer;        /* head layer index */
    uint32_t n_embd, hc, hc_dim, hc_rank;
    uint32_t n_head, n_kvh, hd, n_rot;
    uint32_t n_vocab;
    float eps, rope_base;
    /* head KV */
    uint32_t kv_cap, kv_fill;
    float *d_kv_k, *d_kv_v;
    /* scratch */
    float *d_ws, *d_pws, *d_resb;
    float *d_res;         /* current chain estimate (hc_dim) */
    float *d_emb, *d_cat, *d_logits;
    float *h_logits;
    /* stats */
    uint64_t proposed, accepted, cycles, gated;
};

static int mtp_dbg(void) {
    static int v = -1;
    if (v < 0) v = getenv("Q4_MTP_DEBUG") && getenv("Q4_MTP_DEBUG")[0] != '0';
    return v;
}

static const q4_tensor *mt(const q4_gguf *g, const char *fmt, int layer) {
    char n[Q4_MAX_NAME];
    snprintf(n, sizeof(n), fmt, layer);
    return q4_find_tensor(g, n);
}

static bool mmd(const q4_store *st, const q4_tensor *t, const float *d_x,
                float *d_y) {
    if (!t || t->n_dims < 2 || !d_x || !d_y) return false;
    const uint8_t *d = q4_store_dev(st, t);
    if (!d) return false;
    return q4_hip_gemv_dd(t->ggml_type, d, t->ne[1], t->ne[0], d_x, d_y, 1.f);
}

static bool mmd_n(const q4_store *st, const q4_tensor *t, const float *d_x,
                  float *d_y, uint32_t n) {
    if (!t || t->n_dims < 2 || !d_x || !d_y || n == 0) return false;
    const uint8_t *d = q4_store_dev(st, t);
    if (!d) return false;
    return q4_hip_gemv_dd_n(t->ggml_type, d, t->ne[1], t->ne[0], d_x, d_y, 1.f,
                            n);
}

static const float *fd(const q4_store *st, const q4_tensor *t) {
    return t ? (const float *)q4_store_dev(st, t) : NULL;
}

static bool hc_mix_mn(q4_mtp *m, const float *d_res, float *d_mixed,
                      float *d_inj, bool attn, uint32_t n) {
    const q4_gguf *g = &m->g;
    const uint32_t hc_dim = m->hc_dim;
    const uint32_t n_embd = m->n_embd;
    const q4_tensor *wn = mt(g, attn ? "blk.%d.hc_attn_norm.weight"
                                     : "blk.%d.hc_ffn_norm.weight", m->layer);
    const q4_tensor *wd = mt(g, attn ? "blk.%d.hc_attn_down.weight"
                                     : "blk.%d.hc_ffn_down.weight", m->layer);
    const q4_tensor *wu = mt(g, attn ? "blk.%d.hc_attn_up.weight"
                                     : "blk.%d.hc_ffn_up.weight", m->layer);
    const q4_tensor *wi = mt(g, attn ? "blk.%d.hc_attn_inject.weight"
                                     : "blk.%d.hc_ffn_inject.weight", m->layer);
    float *xn = m->d_ws + MTP_OFF_XN;
    float *lo = xn + (size_t)n * hc_dim;
    float *gate = lo + (size_t)n * m->hc_rank;
    float *gated = gate + (size_t)n * hc_dim;
    if ((size_t)(gated - xn) + (size_t)n * hc_dim > MTP_OFF_QG) return false;
    if (!q4_hip_grouped_rms_n(d_res, fd(m->st, wn), xn, n_embd, m->hc, m->eps,
                              n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: hc mix rms failed\n");
        return false;
    }
    if (!mmd_n(m->st, wd, xn, lo, n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: hc down gemv failed\n");
        return false;
    }
    if (!q4_hip_scale(lo, 1.f / (float)m->hc, (uint64_t)n * m->hc_rank))
        return false;
    if (!q4_hip_silu(lo, (uint64_t)n * m->hc_rank)) return false;
    if (!mmd_n(m->st, wu, lo, gate, n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: hc up gemv failed\n");
        return false;
    }
    if (!q4_hip_sigmoid(gate, (uint64_t)n * hc_dim)) return false;
    if (!q4_hip_mul(gated, xn, gate, (uint64_t)n * hc_dim)) return false;
    if (!q4_hip_mean_hc_n(gated, d_mixed, n_embd, m->hc, n)) return false;
    if (d_inj && wi && !mmd_n(m->st, wi, xn, d_inj, n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: hc inject gemv failed\n");
        return false;
    }
    return true;
}

/* Dense GQA attention of the head layer over its own KV.
 * d_mix (n x n_embd) -> d_y (n x n_embd); positions base..base+n-1. */
static bool attn_mn(q4_mtp *m, int32_t base, const float *d_mix, float *d_y,
                    uint32_t n) {
    const q4_gguf *g = &m->g;
    const uint32_t hd = m->hd;
    if (m->kv_fill + n > m->kv_cap) return false;
    float *qg = m->d_ws + MTP_OFF_QG;
    float *k = qg + (size_t)n * m->n_head * hd * 2;
    float *v = k + (size_t)n * m->n_kvh * hd;
    float *qbat = v + (size_t)n * m->n_kvh * hd;
    float *gbat = qbat + (size_t)n * m->n_head * hd;
    float *attn = qg; /* overwrite packed Q after prep */
    if ((size_t)(gbat - qg) + (size_t)n * m->n_head * hd > MTP_OFF_MIXED)
        return false;
    if (!mmd_n(m->st, mt(g, "blk.%d.attn_q.weight", m->layer), d_mix, qg, n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: attn q gemv failed\n");
        return false;
    }
    if (!mmd_n(m->st, mt(g, "blk.%d.attn_k.weight", m->layer), d_mix, k, n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: attn k gemv failed\n");
        return false;
    }
    if (!mmd_n(m->st, mt(g, "blk.%d.attn_v.weight", m->layer), d_mix, v, n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: attn v gemv failed\n");
        return false;
    }
    size_t kvw = (size_t)m->n_kvh * hd;
    if (!q4_hip_qsa_q_prep_n(qg,
                             fd(m->st, mt(g, "blk.%d.attn_q_norm.weight", m->layer)),
                             qbat, gbat, m->n_head, hd, m->n_rot, base,
                             m->rope_base, m->eps, n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: q prep failed\n");
        return false;
    }
    if (!q4_hip_qsa_k_prep_n(k,
                             fd(m->st, mt(g, "blk.%d.attn_k_norm.weight", m->layer)),
                             m->n_kvh, hd, m->n_rot, base, m->rope_base, m->eps,
                             n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: k prep failed\n");
        return false;
    }
    if (!q4_hip_copy(m->d_kv_k + (size_t)m->kv_fill * kvw, k,
                     (size_t)n * kvw)) {
        if (mtp_dbg())
            fprintf(stderr, "q4 mtp: attn kv copy k failed fill=%u n=%u\n",
                    m->kv_fill, n);
        return false;
    }
    if (!q4_hip_copy(m->d_kv_v + (size_t)m->kv_fill * kvw, v,
                     (size_t)n * kvw)) {
        if (mtp_dbg())
            fprintf(stderr, "q4 mtp: attn kv copy v failed fill=%u n=%u\n",
                    m->kv_fill, n);
        return false;
    }
    m->kv_fill += n;
    float scale = 1.f / sqrtf((float)hd);
    if (!q4_hip_qsa_decode_n(qbat, m->d_kv_k, m->d_kv_v, gbat, attn, m->n_head,
                             m->n_kvh, hd, n, (uint32_t)base, m->kv_cap, scale, 0)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: qsa decode failed\n");
        return false;
    }
    if (!mmd_n(m->st, mt(g, "blk.%d.attn_output.weight", m->layer), attn, d_y,
               n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: attn out gemv failed\n");
        return false;
    }
    return true;
}

/* Host-decode an embedding row from the main model store. */
static bool emb_row(q4_mtp *m, int32_t tok, float *emb) {
    const q4_tensor *temb = q4_find_tensor(m->gm, "token_embd.weight");
    if (!temb || tok < 0 || (uint32_t)tok >= m->gm->n_vocab) return false;
    uint64_t rb = q4_row_bytes(temb->ggml_type, temb->ne[0]);
    const uint8_t *w = q4_store_get(m->stm, temb);
    if (!w || rb == 0) return false;
    if (m->n_embd > 2560) return false;
    return q4_dequant_row(temb->ggml_type, w + (uint64_t)tok * rb, temb->ne[0],
                          emb);
}

/* (token, hidden row) -> head residual row (eh_proj glue). */
static bool eh_proj_m(q4_mtp *m, int32_t tok, const float *d_h, float *d_out) {
    const q4_tensor *enorm = mt(&m->g, "blk.%d.nextn.enorm.weight", m->layer);
    const q4_tensor *hnorm = mt(&m->g, "blk.%d.nextn.hnorm.weight", m->layer);
    const q4_tensor *proj = mt(&m->g, "blk.%d.nextn.eh_proj.weight", m->layer);
    if (!enorm || !hnorm || !proj) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: eh_proj tensors missing\n");
        return false;
    }
    float emb[2560];
    if (!emb_row(m, tok, emb)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: emb_row failed tok=%d\n", tok);
        return false;
    }
    if (!q4_hip_h2d(m->d_emb, emb, m->n_embd * sizeof(float))) return false;
    float *e4 = m->d_ws + MTP_OFF_XN;             /* hc x n_embd */
    float *en = e4 + m->hc_dim;                   /* normed emb (n_embd) */
    float *h4 = en + m->n_embd;                   /* hc x n_embd */
    if ((size_t)(h4 - m->d_ws) + m->hc_dim > MTP_OFF_QG) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: eh_proj ws overflow\n");
        return false;
    }
    if (!q4_hip_grouped_rms(m->d_emb, fd(m->st, enorm), en, m->n_embd, 1,
                            m->eps)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: enorm rms failed\n");
        return false;
    }
    if (!q4_hip_repeat_hc(en, e4, m->n_embd, m->hc)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: repeat_hc failed\n");
        return false;
    }
    if (!q4_hip_grouped_rms(d_h, fd(m->st, hnorm), h4, m->n_embd, m->hc,
                            m->eps)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: hnorm rms failed\n");
        return false;
    }
    float *cat = m->d_cat; /* hc x 2*n_embd, row c = [e_c | h_c] */
    for (uint32_t c = 0; c < m->hc; c++) {
        if (!q4_hip_copy(cat + (size_t)c * 2 * m->n_embd,
                         e4 + (size_t)c * m->n_embd, m->n_embd)) {
            if (mtp_dbg())
                fprintf(stderr, "q4 mtp: eh_proj copy e failed c=%u\n", c);
            return false;
        }
        if (!q4_hip_copy(cat + (size_t)c * 2 * m->n_embd + m->n_embd,
                         h4 + (size_t)c * m->n_embd, m->n_embd)) {
            if (mtp_dbg())
                fprintf(stderr, "q4 mtp: eh_proj copy h failed c=%u\n", c);
            return false;
        }
    }
    if (!mmd_n(m->st, proj, cat, d_out, m->hc)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: eh_proj gemv failed\n");
        return false;
    }
    return true;
}

/* lm head: hc_head mixer (carries the output norm) + shared output GEMV.
 * Input: residual row (d_res). Fills m->h_logits. */
static bool logits_m(q4_mtp *m) {
    const q4_gguf *g = &m->g;
    const uint32_t hc_dim = m->hc_dim;
    const uint32_t n_embd = m->n_embd;
    float *xn = m->d_ws + MTP_OFF_XN;
    float *lo = xn + hc_dim;
    float *gate = lo + m->hc_rank;
    float *gated = gate + hc_dim;
    float *d_mixed = gated + hc_dim;
    if ((size_t)(d_mixed - m->d_ws) + n_embd > MTP_OFF_QG) return false;
    const q4_tensor *hn = mt(g, "blk.%d.nextn.hc_head_norm.weight", m->layer);
    const q4_tensor *hd_ = mt(g, "blk.%d.nextn.hc_head_down.weight", m->layer);
    const q4_tensor *hu = mt(g, "blk.%d.nextn.hc_head_up.weight", m->layer);
    if (!q4_hip_grouped_rms(m->d_res, fd(m->st, hn), xn, n_embd, m->hc, m->eps))
        return false;
    if (!mmd(m->st, hd_, xn, lo)) return false;
    if (!q4_hip_scale(lo, 1.f / (float)m->hc, m->hc_rank)) return false;
    if (!q4_hip_silu(lo, m->hc_rank)) return false;
    if (!mmd(m->st, hu, lo, gate)) return false;
    if (!q4_hip_sigmoid(gate, hc_dim)) return false;
    if (!q4_hip_mul(gated, xn, gate, hc_dim)) return false;
    if (!q4_hip_mean_hc(gated, d_mixed, n_embd, m->hc)) return false;
    const q4_tensor *out = q4_find_tensor(m->gm, "output.weight");
    if (!out) return false;
    return mmd(m->stm, out, d_mixed, m->d_logits);
}

/* One batched head pass over rows at positions base..base+n-1.
 * d_h_rows: n x hc_dim device rows (main-model residuals or the chain
 * estimate). Leaves the new estimate in m->d_res (last row).
 * logits: 0 = none, 1 = device d_logits only, 2 = also d2h m->h_logits. */
static bool fwd_m(q4_mtp *m, int32_t base, const int32_t *toks,
                  const float *d_h_rows, uint32_t n, uint32_t logits) {
    const q4_gguf *g = &m->g;
    const uint32_t n_embd = m->n_embd;
    const uint32_t hc_dim = m->hc_dim;
    if (!n || n > MTP_MAX_ROWS) return false;
    if (m->kv_fill + n > m->kv_cap) {
        if (mtp_dbg())
            fprintf(stderr, "q4 mtp: head KV full fill=%u n=%u cap=%u\n",
                    m->kv_fill, n, m->kv_cap);
        return false;
    }
    float *d_mixed = m->d_ws + MTP_OFF_MIXED;
    float *d_block = m->d_ws + MTP_OFF_BLOCK;
    float *d_inj = m->d_ws + MTP_OFF_INJ;
    for (uint32_t t = 0; t < n; t++)
        if (!eh_proj_m(m, toks[t], d_h_rows + (size_t)t * hc_dim,
                       m->d_resb + (size_t)t * hc_dim)) {
            if (mtp_dbg())
                fprintf(stderr, "q4 mtp: fwd eh_proj failed t=%u tok=%d\n", t,
                        toks[t]);
            return false;
        }
    if (!hc_mix_mn(m, m->d_resb, d_mixed, d_inj, true, n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: fwd hc_attn failed\n");
        return false;
    }
    if (!attn_mn(m, base, d_mixed, d_block, n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: fwd attn failed\n");
        return false;
    }
    if (!q4_hip_hc_combine_n(m->d_resb, d_block, d_inj, n_embd, m->hc, n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: fwd comb1 failed\n");
        return false;
    }
    if (!hc_mix_mn(m, m->d_resb, d_mixed, d_inj, false, n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: fwd hc_ffn failed\n");
        return false;
    }
    if (!q4_moe_layer_dev_n(g, m->st, m->ex, m->layer, d_mixed, d_block,
                            m->d_pws, n, MTP_PWS_N)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: fwd moe failed\n");
        return false;
    }
    if (!q4_hip_hc_combine_n(m->d_resb, d_block, d_inj, n_embd, m->hc, n)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: fwd comb2 failed\n");
        return false;
    }
    if (!q4_hip_copy(m->d_res, m->d_resb + (size_t)(n - 1) * hc_dim, hc_dim)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: fwd res copy failed\n");
        return false;
    }
    if (logits && !logits_m(m)) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: fwd logits failed\n");
        return false;
    }
    if (logits >= 2 &&
        !q4_hip_d2h(m->h_logits, m->d_logits, m->n_vocab * sizeof(float))) {
        if (mtp_dbg()) fprintf(stderr, "q4 mtp: fwd logits d2h failed\n");
        return false;
    }
    return true;
}

/* ---- open / close ---- */

q4_mtp *q4_mtp_open(const char *head_path, const q4_gguf *gmain,
                    q4_store *stmain, uint32_t head_kv, uint32_t n_l1,
                    uint64_t l2_bytes) {
    if (!head_path || !head_path[0] || !gmain || !stmain || !q4_hip_ok())
        return NULL;
    q4_mtp *m = calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->gm = gmain;
    m->stm = stmain;
    /* keep_mtp_layer=1: this file IS the draft block; peeling it would
     * drop blk.N from the head's own forward. */
    if (!q4_gguf_open_ex(&m->g, head_path, 1)) {
        fprintf(stderr, "q4 mtp: failed to open %s\n", head_path);
        free(m);
        return NULL;
    }
    /* Pairing check: the head must be a nextn export of THIS model.
     * Mismatched pairs are rejected rather than producing bad drafts. */
    if (!m->g.arch || strcmp(m->g.arch, "qwen4exp") != 0 ||
        m->g.n_embd != gmain->n_embd ||
        (m->g.hc_mult ? m->g.hc_mult : 4) !=
            (gmain->hc_mult ? gmain->hc_mult : 4) ||
        m->g.hc_rank != gmain->hc_rank ||
        m->g.n_expert != gmain->n_expert ||
        m->g.n_expert_used != gmain->n_expert_used ||
        m->g.n_head != gmain->n_head ||
        m->g.n_head_kv != gmain->n_head_kv ||
        (m->g.n_vocab && gmain->n_vocab && m->g.n_vocab != gmain->n_vocab)) {
        fprintf(stderr, "q4 mtp: head does not match this model\n");
        q4_mtp_close(m);
        return NULL;
    }
    if (m->g.n_layer < 1 || m->g.n_layer >= Q4_MAX_LAYER) {
        q4_mtp_close(m);
        return NULL;
    }
    m->layer = (int32_t)m->g.n_layer - 1;
    if (!mt(&m->g, "blk.%d.nextn.eh_proj.weight", m->layer) ||
        !mt(&m->g, "blk.%d.nextn.hnorm.weight", m->layer) ||
        !mt(&m->g, "blk.%d.ffn_gate_exps.weight", m->layer)) {
        fprintf(stderr, "q4 mtp: %s is not a qwen4exp MTP head\n", head_path);
        q4_mtp_close(m);
        return NULL;
    }
    m->n_embd = gmain->n_embd;
    m->hc = gmain->hc_mult ? gmain->hc_mult : 4;
    m->hc_dim = m->hc * m->n_embd;
    m->hc_rank = gmain->hc_rank ? gmain->hc_rank : 320;
    m->n_head = gmain->n_head ? gmain->n_head : 24;
    m->n_kvh = gmain->n_head_kv ? gmain->n_head_kv : 2;
    m->hd = gmain->n_embd_head_k ? gmain->n_embd_head_k : 256;
    m->n_rot = m->g.n_rot ? m->g.n_rot : (gmain->n_rot ? gmain->n_rot : 64);
    m->n_vocab = gmain->n_vocab;
    m->eps = gmain->rms_eps ? gmain->rms_eps : 1e-6f;
    m->rope_base = gmain->rope_freq_base ? gmain->rope_freq_base : 1e7f;
    m->kv_cap = head_kv ? head_kv : 65536;
    m->kv_fill = 0;

    m->st = q4_store_open(&m->g);
    if (!m->st) {
        fprintf(stderr, "q4 mtp: head dense store failed\n");
        q4_mtp_close(m);
        return NULL;
    }
    if (!q4_store_upload_hip(m->st)) {
        fprintf(stderr, "q4 mtp: head VRAM upload failed\n");
        q4_mtp_close(m);
        return NULL;
    }
    q4_store_free_host_except(m->st, &m->g);

    /* Head experts default to page cache, not resident DRAM: pinning the
     * head's 2.5 GiB on top of the 45 GiB main image is what pushed the box
     * into the 2026-09-29 global OOM (pinned pages resist kernel reaping).
     * Q4_MTP_RESIDENT=1 restores the old behaviour when RAM allows. */
    int hres = 0;
    {
        const char *mr = getenv("Q4_MTP_RESIDENT");
        if (mr && mr[0] && mr[0] != '0') hres = 1;
    }
    m->ex = q4_expert_cache_open_arena_r(&m->g, n_l1 ? n_l1 : 64, l2_bytes,
                                         1, hres);
    if (!m->ex) {
        fprintf(stderr, "q4 mtp: head expert cache failed\n");
        q4_mtp_close(m);
        return NULL;
    }

    size_t f = sizeof(float);
    m->d_ws = q4_hip_malloc(MTP_WS_N * f);
    m->d_pws = q4_hip_malloc(MTP_PWS_N * f);
    m->d_resb = q4_hip_malloc(MTP_MAX_ROWS * m->hc_dim * f);
    m->d_res = q4_hip_malloc(m->hc_dim * f);
    m->d_emb = q4_hip_malloc(m->n_embd * f);
    m->d_cat = q4_hip_malloc((size_t)m->hc * 2 * m->n_embd * f);
    m->d_logits = q4_hip_malloc((size_t)m->n_vocab * f);
    m->h_logits = malloc((size_t)m->n_vocab * f);
    m->d_kv_k = q4_hip_malloc((size_t)m->kv_cap * m->n_kvh * m->hd * f);
    m->d_kv_v = q4_hip_malloc((size_t)m->kv_cap * m->n_kvh * m->hd * f);
    if (!m->d_ws || !m->d_pws || !m->d_resb || !m->d_res || !m->d_emb ||
        !m->d_cat || !m->d_logits || !m->h_logits || !m->d_kv_k || !m->d_kv_v) {
        fprintf(stderr, "q4 mtp: device buffers failed\n");
        q4_mtp_close(m);
        return NULL;
    }
    q4_hip_fill(m->d_res, 0, m->hc_dim);
    return m;
}

void q4_mtp_close(q4_mtp *m) {
    if (!m) return;
    q4_hip_free(m->d_ws);
    q4_hip_free(m->d_pws);
    q4_hip_free(m->d_resb);
    q4_hip_free(m->d_res);
    q4_hip_free(m->d_emb);
    q4_hip_free(m->d_cat);
    q4_hip_free(m->d_logits);
    q4_hip_free(m->d_kv_k);
    q4_hip_free(m->d_kv_v);
    free(m->h_logits);
    q4_expert_cache_close(m->ex);
    q4_store_close(m->st);
    q4_gguf_close(&m->g);
    free(m);
}

/* VRAM beyond the head's own L1 slots. Includes the session-side verify
 * shadows (GDN snapshot ~118 MiB), which q4_sess_mtp_prep allocates. */
uint64_t q4_mtp_fixed_vram(const q4_mtp *m) {
    if (!m) return 0;
    uint64_t b = m->g.bytes_dense + m->g.bytes_shared;
    b += 2ull * m->kv_cap * m->n_kvh * m->hd * sizeof(float);
    b += (uint64_t)MTP_WS_N * sizeof(float);
    b += (uint64_t)MTP_PWS_N * sizeof(float);
    b += (uint64_t)MTP_MAX_ROWS * m->hc_dim * sizeof(float);
    b += (size_t)(m->hc_dim + m->n_embd + m->hc * 2 * m->n_embd + m->n_vocab) *
         sizeof(float);
    /* session shadows: GDN S + conv + PLE snaps + replay staging */
    b += q4_gdn_state_bytes(m->gm) + 12ull * Q4_GIB / 1024;
    return b;
}

uint64_t q4_mtp_slot_bytes(const q4_mtp *m) {
    return m && m->ex ? q4_expert_cache_slot_bytes(m->ex) : 0;
}

q4_expert_cache *q4_mtp_cache(q4_mtp *m) {
    return m ? m->ex : NULL;
}

void q4_mtp_stats(const q4_mtp *m, uint64_t *proposed, uint64_t *accepted,
                  uint64_t *cycles, uint64_t *gated) {
    if (proposed) *proposed = m ? m->proposed : 0;
    if (accepted) *accepted = m ? m->accepted : 0;
    if (cycles) *cycles = m ? m->cycles : 0;
    if (gated) *gated = m ? m->gated : 0;
}

/* ---- prefill + cycle ---- */

/* Build head KV for a chunk that q4_sess_prefill just processed. Rows come
 * from the session's prefill buffer; positions are the chunk's real ones. */
bool q4_mtp_prefill_chunk(q4_mtp *m, q4_sess *s, const int32_t *toks, int n) {
    if (!m || !s || !toks || n <= 0) return false;
    uint32_t n_kv = q4_sess_n_kv(s);
    if (n_kv < (uint32_t)n) {
        if (mtp_dbg())
            fprintf(stderr, "q4 mtp: prefill chunk n_kv=%u < n=%d\n", n_kv, n);
        return false;
    }
    const float *rows = q4_sess_pref_row(s, 0);
    if (!rows) {
        if (mtp_dbg())
            fprintf(stderr, "q4 mtp: no pref rows (d_pref NULL?)\n");
        return false; /* sequential-decode fallback: skip head KV */
    }
    if (m->kv_fill + (uint32_t)n > m->kv_cap) return true; /* frozen window */
    uint32_t base = n_kv - (uint32_t)n; /* first chunk position */
    const uint32_t hc_dim = m->hc_dim;
    for (int i = 0; i < n; i += MTP_MAX_ROWS) {
        int c = n - i < MTP_MAX_ROWS ? n - i : MTP_MAX_ROWS;
        if (!fwd_m(m, (int32_t)(base + (uint32_t)i), toks + i,
                   rows + (size_t)i * hc_dim, (uint32_t)c, false))
            return false;
    }
    return true;
}

/* Seed the chain when the prompt came entirely from the session cache (no
 * prefill ran): run the head once at the last prompt position using the
 * restored residual. Appends the matching real head-KV entry. */
bool q4_mtp_seed(q4_mtp *m, q4_sess *s, int32_t tok) {
    if (!m || !s) return false;
    uint32_t n_kv = q4_sess_n_kv(s);
    if (n_kv == 0) return false;
    const float *res = q4_sess_res_dev(s);
    if (!res || m->kv_fill >= m->kv_cap) return false;
    return fwd_m(m, (int32_t)n_kv - 1, &tok, res, 1, false);
}

static int32_t argmax(const float *l, uint32_t n) {
    int32_t b = 0;
    float v = l[0];
    for (uint32_t i = 1; i < n; i++)
        if (l[i] > v) {
            v = l[i];
            b = (int32_t)i;
        }
    return b;
}

/* Host fallback for the probability of the max under softmax. */
static float softpmax(const float *l, uint32_t n) {
    float mx = l[0];
    for (uint32_t i = 1; i < n; i++)
        if (l[i] > mx) mx = l[i];
    double s = 0;
    for (uint32_t i = 0; i < n; i++) s += exp((double)(l[i] - mx));
    return (float)(1.0 / s);
}

static float mtp_pmin(void) {
    static int init;
    static float v;
    if (!init) {
        const char *e = getenv("Q4_MTP_PMIN");
        v = e ? (float)atof(e) : 0.5f;
        init = 1;
    }
    return v;
}

static double mtp_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int mtp_prof(void) {
    static int v = -1;
    if (v < 0) v = getenv("Q4_PROFILE") && getenv("Q4_PROFILE")[0] != '0';
    return v;
}

static double t_draft, t_verify, t_logits, t_commit, t_repair;
static uint64_t t_cycles;
#define MTP_PT(acc)                                                          \
    do {                                                                     \
        if (prof) {                                                          \
            q4_hip_sync();                                                   \
            acc += mtp_now() - tt;                                           \
        }                                                                    \
        tt = mtp_now();                                                      \
    } while (0)

void q4_mtp_prof_print(q4_mtp *m, FILE *f) {
    (void)m;
    if (!f || !t_cycles) return;
    double c = (double)t_cycles;
    fprintf(f, "q4 mtp prof: cycles %llu\n"
               "  draft   %.3f s  %.2f ms/cyc\n"
               "  verify  %.3f s  %.2f ms/cyc\n"
               "  logits  %.3f s  %.2f ms/cyc\n"
               "  commit  %.3f s  %.2f ms/cyc\n"
               "  repair  %.3f s  %.2f ms/cyc\n",
            (unsigned long long)t_cycles,
            t_draft, t_draft * 1e3 / c,
            t_verify, t_verify * 1e3 / c,
            t_logits, t_logits * 1e3 / c,
            t_commit, t_commit * 1e3 / c,
            t_repair, t_repair * 1e3 / c);
    t_draft = t_verify = t_logits = t_commit = t_repair = 0;
    t_cycles = 0;
}

int q4_mtp_cycle(q4_mtp *m, q4_sess *s, int32_t c0, int n_draft, float temp,
                 int top_k, float top_p, unsigned *rng, int32_t *out_drafts,
                 int *n_out, int32_t *next_c0, const float **next_logits) {
    *n_out = 0;
    if (next_logits) *next_logits = NULL;
    if (!m || !s || n_draft <= 0 || !out_drafts || !next_c0) return -1;
    if (q4_sess_n_kv(s) + (uint32_t)n_draft + 2 >= q4_sess_max_kv(s))
        return -1;
    uint32_t p = q4_sess_n_kv(s); /* c0 sits at position p */
    int nd = n_draft > MTP_MAX_DRAFT ? MTP_MAX_DRAFT : n_draft;

    /* 1. Draft chain. Each step consumes (token, previous estimate) and
     * writes an estimate entry into the head KV (overwritten by repair). */
    int32_t draft[MTP_MAX_DRAFT];
    int32_t did[MTP_MAX_DRAFT][Q4_DIST_MAX];
    float dprob[MTP_MAX_DRAFT][Q4_DIST_MAX];
    int dk_n[MTP_MAX_DRAFT];
    uint32_t fill0 = m->kv_fill;
    int32_t h = c0;
    int prof = mtp_prof();
    double tt = prof ? mtp_now() : 0;
    const float pmin = mtp_pmin();
    for (int i = 0; i < nd; i++) {
        /* temp==0 needs only (argmax, p_max) on device — skip the full
         * logits D2H. temp>0 still wants the host row for dist sampling. */
        if (!fwd_m(m, (int32_t)p + i, &h, m->d_res, 1,
                   temp > 1e-5f ? 2u : 1u))
            return -1;
        float pc = 0.f;
        if (temp <= 1e-5f) {
            if (!q4_hip_argmax_prob(m->d_logits, m->n_vocab, &draft[i],
                                    &pc)) {
                if (!q4_hip_d2h(m->h_logits, m->d_logits,
                                m->n_vocab * sizeof(float)))
                    return -1;
                draft[i] = argmax(m->h_logits, m->n_vocab);
                pc = softpmax(m->h_logits, m->n_vocab);
            }
        } else {
            dk_n[i] = q4_dist_build(m->h_logits, m->n_vocab, temp, top_k, top_p,
                                    did[i], dprob[i]);
            draft[i] = q4_dist_sample(did[i], dprob[i], dk_n[i], rng);
            for (int j = 0; j < dk_n[i]; j++) {
                if (did[i][j] == draft[i]) {
                    pc = dprob[i][j];
                    break;
                }
            }
        }
        /* Strata-style confidence gate: past the first draft the head must
         * be >= pmin confident to keep extending the chain; below it the
         * chain stops here (this draft is dropped, not proposed). */
        if (i > 0 && pmin > 0.f && pc < pmin) {
            m->gated++;
            nd = i;
            break;
        }
        m->proposed++;
        h = draft[i];
    }
    MTP_PT(t_draft);

    /* 2. Verify [c0, d0 .. d_{nd-1}] in one batched main-model pass. */
    int32_t batch[MTP_MAX_DRAFT + 1];
    batch[0] = c0;
    for (int i = 0; i < nd; i++) batch[1 + i] = draft[i];
    int nb = nd + 1;
    if (!q4_sess_prefill_tokens(s, batch, nb)) return -1; /* nothing kept */
    MTP_PT(t_verify);

    /* Return protocol: 0 = ok (drafts in out_drafts, next token in
     * *next_c0). 1 = recovered, only c0 was kept (*n_out == 0), the caller
     * must continue in plain mode from q4_sess_logits(). 2 = recovered,
     * the accepted drafts in out_drafts WERE kept, continue plain after
     * emitting them. -1 = nothing was processed, the caller must run
     * q4_sess_decode_ex(c0) itself. */
    int32_t tid[Q4_DIST_MAX];
    float tprob[Q4_DIST_MAX];
    int n_acc = 1; /* c0 is a target sample: always verified */
    int rc = 0;
    for (int i = 0; i < nd; i++) {
        /* Target distribution at position p+i predicts position p+i+1. */
        if (!q4_sess_logits_at(s, i)) {
            rc = 1;
            break;
        }
        int acc;
        if (temp <= 1e-5f) {
            acc = (argmax(q4_sess_logits(s), m->n_vocab) == draft[i]);
            if (!acc) *next_c0 = argmax(q4_sess_logits(s), m->n_vocab);
        } else {
            int tk = q4_dist_build(q4_sess_logits(s), m->n_vocab, temp, top_k,
                                   top_p, tid, tprob);
            int32_t rs = 0;
            acc = q4_spec_accept(tid, tprob, tk, did[i], dprob[i], dk_n[i],
                                 draft[i], rng, &rs);
            if (!acc) *next_c0 = rs;
        }
        if (acc) {
            out_drafts[i] = draft[i];
            m->accepted++;
            n_acc = i + 2;
        } else {
            break;
        }
        if (i == nd - 1) {
            /* All drafts accepted: sample the bonus token from the last row
             * (position p+nd, predicting p+nd+1). */
            if (!q4_sess_logits_at(s, nd)) {
                rc = 1;
                n_acc = nd; /* the bonus sample is void */
                *next_c0 = 0;
            } else if (temp <= 1e-5f) {
                *next_c0 = argmax(q4_sess_logits(s), m->n_vocab);
            } else {
                int tk = q4_dist_build(q4_sess_logits(s), m->n_vocab, temp,
                                       top_k, top_p, tid, tprob);
                *next_c0 = q4_dist_sample(tid, tprob, tk, rng);
            }
        }
    }
    m->cycles++;

    if (mtp_dbg()) {
        fprintf(stderr, "q4 mtp: cycle p=%u drafts=[", p);
        for (int i = 0; i < nd; i++) fprintf(stderr, " %d", draft[i]);
        fprintf(stderr, "] n_acc=%d next_c0=%d top1_row0=%d\n", n_acc, *next_c0,
                argmax(q4_sess_logits(s), m->n_vocab));
    }
    if (prof) t_cycles++;
    MTP_PT(t_logits);

    /* 3. Keep the accepted prefix (rollback + replay when the tail was
     * rejected). */
    if (rc == 0 && !q4_sess_mtp_commit(s, n_acc)) {
        /* Redo from the intact pre-batch snapshot, keeping only c0. */
        if (q4_sess_mtp_commit(s, 1)) {
            rc = 1;
        } else {
            MTP_PT(t_commit);
            return -1; /* unrecoverable: caller re-decodes c0 */
        }
    }
    MTP_PT(t_commit);
    if (rc != 0) {
        *n_out = 0;
        q4_sess_logits_at(s, 0);
        if (next_logits) *next_logits = q4_sess_logits(s);
        return rc;
    }
    *n_out = n_acc - 1;

    /* 4. Repair the head KV with the real residuals of the accepted prefix
     * (overwrites the estimate entries written during drafting). */
    m->kv_fill = fill0;
    int repaired = 0;
    for (int t = 0; t < n_acc; t++) {
        const float *hrow = q4_sess_pref_row(s, t);
        if (!hrow) break;
        if (!fwd_m(m, (int32_t)p + t, &batch[t], hrow, 1, false)) break;
        repaired = t + 1;
    }
    MTP_PT(t_repair);
    if (repaired < n_acc) {
        /* Head KV partially repaired: drafts degrade but stay exact. The
         * caller keeps the accepted tokens and switches to plain mode. */
        q4_sess_logits_at(s, n_acc - 1);
        if (next_logits) *next_logits = q4_sess_logits(s);
        return 2;
    }
    /* m->d_res now estimates the hidden state at position p+n_acc: the seed
     * for the next cycle's first draft. */
    if (next_logits) *next_logits = q4_sess_logits(s);
    return 0;
}

void q4_mtp_reset(q4_mtp *m) {
    /* The session restarted (q4_sess_reset): head positions restart at 0,
     * so all carried-over entries would be stale duplicates. */
    if (m) m->kv_fill = 0;
}
