#include "q4.h"

#include <malloc.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct q4_engine {
    q4_gguf g;
    q4_store *st;
    q4_expert_cache *ex;
    q4_ple *ple;
    q4_tok *tok;
    q4_sess *sess;
    q4_mtp *mtp;
    int mtp_fail;
    uint32_t ctx;
    uint32_t slots;
    unsigned rng;
    float temp;
    int top_k;
    float top_p;
    int prompt_tokens;
    int completion_tokens;
    int show_thinking;
    int hit_limit;
    char *reason;
    int reason_n;
    int reason_cap;
    char name[128];
    pthread_mutex_t mu;
    void (*progress)(int prefill_i, int prefill_n, int gen_i, void *u);
    void *progress_u;
    /* Set by the HTTP client when the socket dies. Same thread as generate. */
    volatile int *cancel;
    int pref_base;   /* tokens completed before the current chunk */
    int pref_chunk;  /* tokens in the current chunk */
    int pref_total;  /* prompt tokens */
};

static void pref_layer_hook(uint32_t il, uint32_t nl, void *u) {
    q4_engine *e = u;
    if (!e || !e->progress || !nl) return;
    int done = e->pref_base + (int)((uint64_t)il * (uint64_t)e->pref_chunk / nl);
    e->progress(done, e->pref_total, 0, e->progress_u);
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static uint64_t mem_available(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[128];
    unsigned long avail = 0, freeb = 0, v;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "MemAvailable: %lu", &v) == 1) avail = v;
        else if (sscanf(line, "MemFree: %lu", &v) == 1) freeb = v;
    }
    fclose(f);
    return (avail ? avail : freeb) * 1024ull;
}

static uint64_t default_l2(void) {
    uint64_t want = 24ull * Q4_GIB;
    const char *e = getenv("Q4_L2_GIB");
    if (e && e[0]) {
        int v = atoi(e);
        want = v <= 0 ? 0 : (uint64_t)v * Q4_GIB;
    }
    /* Same floor the expert cache enforces. 16 GiB plus a 45 GiB resident
     * image does not fit in 60 GiB. */
    uint64_t reserve = 4ull * Q4_GIB;
    const char *r = getenv("Q4_DRAM_RESERVE_GIB");
    if (r && r[0]) {
        int v = atoi(r);
        reserve = v < 4 ? 4 : (uint64_t)v * Q4_GIB;
    }
    uint64_t avail = mem_available();
    if (avail && want + reserve > avail) {
        uint64_t cap = avail > reserve ? avail - reserve : 0;
        if (cap < want) {
            fprintf(stderr,
                    "q4: L2 %.0f GiB capped to %.0f GiB (avail %.0f, reserve %.0f)\n",
                    (double)want / Q4_GIB, (double)cap / Q4_GIB,
                    (double)avail / Q4_GIB, (double)reserve / Q4_GIB);
            want = cap;
        }
    }
    return want;
}

static uint32_t warm_pin_cap(const q4_engine *e);

static uint32_t default_ctx(uint32_t ctx, const q4_gguf *g) {
    if (!ctx) {
        const char *e = getenv("Q4_CTX");
        ctx = e && e[0] ? (uint32_t)atoi(e) : 262144;
    }
    if (!ctx) ctx = 262144;
    if (g && g->n_ctx_train && ctx > g->n_ctx_train) ctx = g->n_ctx_train;
    return ctx;
}

static int tok_is_special(const q4_tok *t, int32_t id) {
    if (!t) return 1;
    if (id == q4_tok_eos(t) || id == q4_tok_im_end(t)) return 1;
    char buf[64];
    int n = q4_tok_decode(t, &id, 1, buf, (int)sizeof(buf));
    if (n <= 0) return 1;
    return buf[0] == '<' && buf[1] == '|';
}

q4_engine *q4_engine_open_full(const char *model, uint32_t ctx, uint64_t l2_bytes,
                               const char *mtp_path) {
    if (!model || !model[0]) return NULL;
    q4_engine *e = calloc(1, sizeof(*e));
    if (!e) return NULL;
    e->temp = 0.7f;
    e->top_k = 20;
    e->top_p = 0.8f;
    e->rng = (unsigned)time(NULL) ^ 0x9e3779b9u;
    e->show_thinking =
        getenv("Q4_SHOW_THINKING") && getenv("Q4_SHOW_THINKING")[0] == '1';
    pthread_mutex_init(&e->mu, NULL);
    if (!q4_gguf_open(&e->g, model)) {
        fprintf(stderr, "q4: failed to open %s\n", model);
        pthread_mutex_destroy(&e->mu);
        free(e);
        return NULL;
    }
    snprintf(e->name, sizeof(e->name), "qwen3.8-flash-next");
    e->ctx = default_ctx(ctx, &e->g);
    if (!l2_bytes) l2_bytes = default_l2();

    bool hip = q4_hip_init();
    fprintf(stderr, "q4: hip %s  ctx %u  model %s\n", hip ? "ok gfx1100" : "cpu",
            e->ctx, e->name);

    double t0 = now_s();
    e->st = q4_store_open(&e->g);
    if (!e->st) {
        fprintf(stderr, "q4: dense store load failed\n");
        q4_engine_close(e);
        return NULL;
    }
    fprintf(stderr, "q4: dense host  %.2f s\n", now_s() - t0);
    bool dense_on_gpu = false;
    if (hip) {
        t0 = now_s();
        if (!q4_store_upload_hip(e->st))
            fprintf(stderr, "q4: VRAM upload failed, GEMV will H2D\n");
        else {
            fprintf(stderr, "q4: dense VRAM %.2f s  (%.2f GiB)\n", now_s() - t0,
                    (double)(e->g.bytes_dense + e->g.bytes_shared) / Q4_GIB);
            dense_on_gpu = true;
        }
    }
    /* Free host dense copies (keep token_embd + PLE conv): leaves DRAM for
     * the expert L2. Q4_FREE_HOST_DENSE=0 keeps the old behavior. */
    {
        const char *fd = getenv("Q4_FREE_HOST_DENSE");
        if (dense_on_gpu && (!fd || fd[0] != '0')) {
            q4_store_free_host_except(e->st, &e->g);
            /* glibc keeps the freed dense image in the heap, so MemAvailable
             * still looks too small for the 45 GiB resident alloc. */
            malloc_trim(0);
            fprintf(stderr, "q4: host dense freed (token_embd + PLE conv kept)\n");
        }
    }

    q4_machine m = q4_machine_this_pc(e->ctx);
    q4_place p = q4_place_plan(&e->g, &m);
    q4_place_print(&p, &e->g, stderr);
    fprintf(stderr, "q4: QSA KV %.2f GiB  GDN state %.2f GiB  (%u QSA layers)\n",
            (double)q4_qsa_kv_bytes(&e->g, e->ctx) / Q4_GIB,
            (double)q4_gdn_state_bytes(&e->g) / Q4_GIB, q4_n_qsa_layers(&e->g));

    uint32_t slots = p.l1_experts;
    /* Leave VRAM for ROCm scratch. Filling 24 GiB made OpenCode decode hang.
     * Q4_L1_SLOTS can only cap below the planner value, never push past it:
     * a bigger prefill chunk shrinks the budget the planner computed. */
    uint32_t cap = 8192;
    const char *ls = getenv("Q4_L1_SLOTS");
    if (ls && ls[0]) {
        int v = atoi(ls);
        if (v > 0) cap = (uint32_t)v;
        if (cap > p.l1_experts)
            fprintf(stderr,
                    "q4: Q4_L1_SLOTS %u exceeds planner %u; clamped\n",
                    cap, p.l1_experts);
    }
    if (slots > cap) slots = cap;
    if (slots < 64) slots = 64;
    e->slots = slots;

    /* Reserve VRAM for the MTP head before the main GPU L1 is sized. The
     * fixed part (dense + KV + scratch + verify shadows) plus a small L1. */
    bool want_mtp = hip && mtp_path && mtp_path[0];
    uint32_t mtp_slots = 128; /* a prefill group of 8 tokens routes up to
                               * 80 distinct head experts: 64 would evict
                               * mid-batch (moe _n fail #14) */
    {
        const char *ms0 = getenv("Q4_MTP_SLOTS");
        if (ms0 && ms0[0]) {
            int v = atoi(ms0);
            if (v > 0) mtp_slots = (uint32_t)v;
        }
    }
    if (want_mtp) {
        /* fixed (dense+KV+scratch+shadows) + head L1 slots, ~6 MiB each */
        uint64_t head_est =
            (768ull + (uint64_t)mtp_slots * 6ull) << 20;
        uint64_t msb = 3ull * (e->g.per_expert_bytes ? e->g.per_expert_bytes
                                                     : 3ull << 20);
        uint32_t reserve = (uint32_t)(head_est / msb);
        if (reserve < slots) {
            slots -= reserve;
            fprintf(stderr,
                    "q4: L1 slots -> %u (%u reserved for MTP head, "
                    "~%.2f GiB VRAM)\n",
                    slots, reserve, (double)head_est / Q4_GIB);
        }
    }

    e->ex = NULL;
    uint64_t l2b = l2_bytes;
    while (!e->ex) {
        e->ex = q4_expert_cache_open_ex(&e->g, slots, l2b);
        if (e->ex) break;
        if (l2b > 8ull * Q4_GIB) {
            l2b /= 2;
            fprintf(stderr, "q4: L2 alloc failed, retry %.0f GiB\n",
                    (double)l2b / Q4_GIB);
            continue;
        }
        fprintf(stderr, "q4: expert cache failed\n");
        q4_engine_close(e);
        return NULL;
    }
    fprintf(stderr, "q4: DRAM L2 %.2f GiB\n",
            (double)q4_expert_l2_bytes(e->ex) / Q4_GIB);
    if (q4_expert_l2_bytes(e->ex)) {
        t0 = now_s();
        fprintf(stderr, "q4: prefetching routed experts into DRAM...\n");
        q4_expert_prefetch(e->ex, stderr);
        fprintf(stderr, "q4: DRAM L2 filled %.2f s  %u experts\n", now_s() - t0,
                q4_expert_l2_filled(e->ex));
        const char *ml = getenv("Q4_MLOCK_L2");
        if (!ml || ml[0] != '0') {
            if (q4_expert_cache_mlock_l2(e->ex))
                fprintf(stderr, "q4: L2 mlocked (kswapd cannot steal experts)\n");
        }
    }

    /* Session first so QSA KV claims VRAM before L1 experts. */
    e->ple = q4_ple_open(&e->g);
    uint32_t try_ctx = e->ctx;
    while (try_ctx >= 4096) {
        e->sess = q4_sess_open(&e->g, e->st, e->ex, e->ple, try_ctx);
        if (e->sess) {
            e->ctx = try_ctx;
            break;
        }
        fprintf(stderr, "q4: session ctx %u failed, retry %u\n", try_ctx,
                try_ctx / 2);
        try_ctx /= 2;
    }
    if (!e->sess) {
        fprintf(stderr, "q4: session open failed\n");
        q4_engine_close(e);
        return NULL;
    }
    fprintf(stderr, "q4: session ctx %u  QSA %s\n", q4_sess_max_kv(e->sess),
            "gpu/host");

    if (hip) {
        uint64_t sb = q4_expert_cache_slot_bytes(e->ex);
        uint32_t s = slots;
        uint32_t got = 0;
        while (s >= 64) {
            if (q4_hip_experts_alloc(s, sb)) {
                got = s;
                break;
            }
            fprintf(stderr, "q4: GPU L1 %u slots failed, retry %u\n", s, s / 2);
            s /= 2;
        }
        if (!got) {
            fprintf(stderr, "q4: GPU L1 alloc failed\n");
            q4_engine_close(e);
            return NULL;
        }
        if (got < q4_expert_cache_n_slots(e->ex)) {
            fprintf(stderr, "q4: cap host L1 %u -> %u to match GPU\n",
                    q4_expert_cache_n_slots(e->ex), got);
            q4_expert_cache_cap_slots(e->ex, got);
        }
        e->slots = got;
        fprintf(stderr, "q4: GPU L1 %u slots  %.2f GiB\n", got,
                (double)got * sb / Q4_GIB);
        /* Opt-in for the partial-L2 staging arena only. Resident experts
         * have no host mirror; hipHostRegister of that image is not used. */
        if (getenv("Q4_PIN_ARENA") && getenv("Q4_PIN_ARENA")[0] == '1') {
            if (q4_expert_is_resident(e->ex)) {
                fprintf(stderr,
                        "q4: arena pin skipped (H2D reads the resident image)\n");
            } else {
                double tp = now_s();
                if (q4_expert_cache_pin_arena(e->ex))
                    fprintf(stderr, "q4: L1 host arena pinned (DMA H2D)  %.2f s\n",
                            now_s() - tp);
                else
                    fprintf(stderr, "q4: L1 host arena pin failed, pageable H2D\n");
            }
        }
    }

    /* MTP draft head: dense store + expert cache (arena 1) + verify shadows.
     * Any failure here disables MTP but keeps the engine usable. */
    if (want_mtp) {
        uint32_t head_kv = 65536;
        const char *kv = getenv("Q4_MTP_KV");
        if (kv && kv[0]) head_kv = (uint32_t)atoi(kv);
        const char *ms = getenv("Q4_MTP_SLOTS");
        if (ms && ms[0]) {
            int v = atoi(ms);
            if (v > 0) mtp_slots = (uint32_t)v;
        }
        /* Head experts read from the head GGUF's page cache by default.
         * A DRAM L2 (old default 2 GiB) is opt-in: Q4_MTP_L2_GIB=<n>. */
        uint64_t head_l2 = 0;
        const char *ml2 = getenv("Q4_MTP_L2_GIB");
        if (ml2 && ml2[0]) {
            int v = atoi(ml2);
            head_l2 = v <= 0 ? 0 : (uint64_t)v * Q4_GIB;
        }
        e->mtp = q4_mtp_open(mtp_path, &e->g, e->st, head_kv, mtp_slots, head_l2);
        if (!e->mtp) {
            fprintf(stderr, "q4: MTP disabled (head open failed)\n");
        } else {
            uint64_t msb = q4_mtp_slot_bytes(e->mtp);
            uint32_t hs = q4_expert_cache_n_slots(q4_mtp_cache(e->mtp));
            uint32_t got = 0;
            while (hs >= 16) {
                if (q4_hip_experts_alloc_a(1, hs, msb)) {
                    got = hs;
                    break;
                }
                fprintf(stderr, "q4: MTP L1 %u slots failed, retry %u\n", hs,
                        hs / 2);
                hs /= 2;
            }
            if (!got) {
                fprintf(stderr, "q4: MTP disabled (GPU L1 alloc failed)\n");
                q4_mtp_close(e->mtp);
                e->mtp = NULL;
            } else {
                if (got < q4_expert_cache_n_slots(q4_mtp_cache(e->mtp)))
                    q4_expert_cache_cap_slots(q4_mtp_cache(e->mtp), got);
                if (!q4_sess_mtp_prep(e->sess, 5)) {
                    fprintf(stderr, "q4: MTP disabled (session prep failed)\n");
                    q4_mtp_close(e->mtp);
                    e->mtp = NULL;
                } else {
                    if (q4_expert_l2_bytes(q4_mtp_cache(e->mtp))) {
                        double tp = now_s();
                        q4_expert_prefetch(q4_mtp_cache(e->mtp), stderr);
                        fprintf(stderr, "q4: MTP L2 filled %.2f s\n",
                                now_s() - tp);
                        const char *ml = getenv("Q4_MLOCK_L2");
                        if (!ml || ml[0] != '0')
                            q4_expert_cache_mlock_l2(q4_mtp_cache(e->mtp));
                    }
                    fprintf(stderr, "q4: MTP ready  draft head '%s'\n", mtp_path);
                }
            }
        }
    }

    e->tok = q4_tok_open(&e->g);
    if (!e->tok) {
        fprintf(stderr, "q4: tokenizer failed (missing ggml.tokens?)\n");
        q4_engine_close(e);
        return NULL;
    }
    return e;
}

q4_engine *q4_engine_open(const char *model, uint32_t ctx, uint64_t l2_bytes) {
    return q4_engine_open_full(model, ctx, l2_bytes, NULL);
}

void q4_engine_close(q4_engine *e) {
    if (!e) return;
    /* Dispatcher thread reads e->ex and e->g — join it before teardown. */
    q4_moe_dev_stop();
    q4_mtp_close(e->mtp);
    q4_sess_close(e->sess);
    q4_tok_close(e->tok);
    q4_ple_close(e->ple);
    q4_expert_cache_close(e->ex);
    q4_hip_experts_free();
    q4_store_close(e->st);
    q4_hip_shutdown();
    q4_gguf_close(&e->g);
    pthread_mutex_destroy(&e->mu);
    free(e->reason);
    free(e);
}

void q4_engine_set_sample(q4_engine *e, float temp, int top_k, float top_p) {
    if (!e) return;
    e->temp = temp;
    e->top_k = top_k;
    e->top_p = top_p;
}

void q4_engine_set_progress(q4_engine *e,
                            void (*cb)(int prefill_i, int prefill_n, int gen_i,
                                       void *u),
                            void *u) {
    if (!e) return;
    e->progress = cb;
    e->progress_u = u;
}

void q4_engine_set_cancel(q4_engine *e, volatile int *flag) {
    if (!e) return;
    e->cancel = flag;
}

uint32_t q4_engine_ctx(const q4_engine *e) {
    return e ? e->ctx : 0;
}

const char *q4_engine_name(const q4_engine *e) {
    return e ? e->name : "";
}

int q4_engine_last_prompt_tokens(const q4_engine *e) {
    return e ? e->prompt_tokens : 0;
}

int q4_engine_last_completion_tokens(const q4_engine *e) {
    return e ? e->completion_tokens : 0;
}

const char *q4_engine_last_reasoning(const q4_engine *e) {
    return e && e->reason ? e->reason : "";
}

int q4_engine_last_hit_limit(const q4_engine *e) {
    return e ? e->hit_limit : 0;
}

static uint64_t hash_str(const char *s) {
    uint64_t h = 14695981039346656037ull;
    if (!s) return 0;
    for (; *s; s++) {
        h ^= (unsigned char)*s;
        h *= 1099511628211ull;
    }
    return h ? h : 1;
}

static uint64_t first_user_hash(const char *const *roles, const char *const *contents,
                                int nmsg) {
    for (int i = 0; i < nmsg; i++)
        if (roles && roles[i] && strcmp(roles[i], "user") == 0)
            return hash_str(contents ? contents[i] : "");
    return 0;
}

static int reason_append(q4_engine *e, const char *p, int n) {
    if (!e || n <= 0) return 0;
    if (e->reason_n + n + 1 > e->reason_cap) {
        int cap = e->reason_cap ? e->reason_cap : 8192;
        while (cap < e->reason_n + n + 1) cap *= 2;
        if (cap > 8 << 20) cap = 8 << 20;
        if (e->reason_n + n + 1 > cap) return -1;
        char *q = realloc(e->reason, (size_t)cap);
        if (!q) return -1;
        e->reason = q;
        e->reason_cap = cap;
    }
    memcpy(e->reason + e->reason_n, p, (size_t)n);
    e->reason_n += n;
    e->reason[e->reason_n] = 0;
    return 0;
}

/* Decode one BPE piece into complete UTF-8, holding a trailing partial sequence. */
static int flush_utf8_piece(unsigned char *hold, int *nh, const char *piece, int pn,
                            char *emit, int max_emit, int *n_emit) {
    unsigned char tmp[264];
    if (*nh + pn > (int)sizeof(tmp)) *nh = 0;
    memcpy(tmp, hold, (size_t)*nh);
    memcpy(tmp + *nh, piece, (size_t)pn);
    int tot = *nh + pn, i = 0, eo = 0;
    while (i < tot) {
        unsigned char c = tmp[i];
        int L = 1;
        if (c < 0x80) L = 1;
        else if ((c & 0xe0) == 0xc0) L = 2;
        else if ((c & 0xf0) == 0xe0) L = 3;
        else if ((c & 0xf8) == 0xf0) L = 4;
        else {
            i++;
            continue;
        }
        if (i + L > tot) break;
        int ok = 1;
        for (int k = 1; k < L; k++)
            if ((tmp[i + k] & 0xc0) != 0x80) ok = 0;
        if (!ok) {
            i++;
            continue;
        }
        if (eo + L > max_emit) break;
        memcpy(emit + eo, tmp + i, (size_t)L);
        eo += L;
        i += L;
    }
    *nh = tot - i;
    if (*nh > 0) memcpy(hold, tmp + i, (size_t)*nh);
    *n_emit = eo;
    return 0;
}

/* Emit one generated token: think-state tracking, UTF-8 flush, streaming.
 * Returns 0 = continue, 1 = eos/im_end (stop), 2 = output full (stop). */
static int emit_piece(q4_engine *e, int32_t id, char *out, int *o, int max_out,
                      int *in_think, int *think_toks, unsigned char *hold,
                      int *nh, void (*stream_cb)(const char *, int, void *),
                      void *u, int t, int32_t im_end, int32_t eos, int32_t th,
                      int32_t te) {
    if (id == eos || (im_end >= 0 && id == im_end)) return 1;
    if (th >= 0 && id == th) {
        *in_think = 1;
        *nh = 0;
    } else if (te >= 0 && id == te) {
        if (*in_think)
            fprintf(stderr, "\nq4: </think> after %d tok\n", *think_toks);
        *in_think = 0;
        *nh = 0;
    } else if (!tok_is_special(e->tok, id) && (!*in_think || e->show_thinking)) {
        char piece[256];
        int pn = q4_tok_decode(e->tok, &id, 1, piece, (int)sizeof(piece));
        if (pn > 0) {
            char emitbuf[264];
            int eo = 0;
            flush_utf8_piece(hold, nh, piece, pn, emitbuf, (int)sizeof(emitbuf),
                             &eo);
            if (eo > 0) {
                emitbuf[eo] = 0;
                int kind = *in_think ? Q4_PIECE_REASONING : Q4_PIECE_CONTENT;
                if (kind == Q4_PIECE_REASONING) {
                    if (reason_append(e, emitbuf, eo) < 0) {
                        fprintf(stderr, "\nq4: reasoning buffer full at gen %d\n",
                                t);
                        return 2;
                    }
                } else {
                    if (*o + eo >= max_out) return 2;
                    memcpy(out + *o, emitbuf, (size_t)eo);
                    *o += eo;
                    out[*o] = 0;
                }
                if (stream_cb) stream_cb(emitbuf, kind, u);
                if (t == 0)
                    fprintf(stderr, "q4: first piece kind=%s [%s]\n",
                            kind == Q4_PIECE_REASONING ? "think" : "content",
                            emitbuf);
            }
        }
    }
    if (*in_think) (*think_toks)++;
    return 0;
}

static int client_gone(const q4_engine *e) {
    return e && e->cancel && *e->cancel;
}

static int prompt_cache_path(const char *path, char *out, size_t cap) {
    if (!path || !path[0] || !out || cap < 8) return -1;
    int n = snprintf(out, cap, "%s.prompt", path);
    if (n < 0 || (size_t)n >= cap) {
        if (cap) out[0] = 0;
        return -1;
    }
    return 0;
}

/* Longest snapshot whose ids are an exact prefix of ids[0..n).
 * A mismatch inside the snapshot cannot be restored: GDN state is
 * only valid at the snapshot's end. */
static int cache_prefix_len(q4_sess *sess, const char *path, const int32_t *ids,
                            int n, int32_t *scratch, int scratch_n) {
    if (!sess || !path || !path[0] || !ids || n <= 0 || !scratch || scratch_n <= 0)
        return 0;
    int sn = q4_sess_peek_ids(sess, path, scratch, scratch_n, NULL, NULL, NULL);
    if (sn <= 0 || sn > n) return 0;
    if (memcmp(scratch, ids, (size_t)sn * sizeof(int32_t)) != 0) return 0;
    return sn;
}

static int generate_locked(q4_engine *e, const char *const *roles,
                           const char *const *contents, int nmsg, char *out,
                           int max_out, int max_new,
                           void (*stream_cb)(const char *piece, int kind, void *u),
                           void *u) {
    if (!e || !e->sess || !e->tok || !out || max_out <= 1) return -1;
    if (client_gone(e)) return -1;
    if (max_new <= 0) max_new = 256;
    int32_t *ids = malloc((size_t)e->ctx * sizeof(int32_t));
    if (!ids) return -1;
    const char *cache_path = getenv("Q4_SESS_CACHE");
    char prompt_path[4096];
    prompt_path[0] = 0;
    if (cache_path && cache_path[0])
        prompt_cache_path(cache_path, prompt_path, sizeof(prompt_path));
    int n = 0;
    int spliced = 0;
    uint64_t uh = 0;
    int cache_seen = 0, cache_nmsg = 0, cache_user_ok = 0;
    if (cache_path && cache_path[0]) {
        int32_t *saved = malloc((size_t)e->ctx * sizeof(int32_t));
        int saved_prompt = 0, saved_nmsg = 0;
        uint64_t saved_uh = 0;
        int saved_n = saved ? q4_sess_peek_ids(e->sess, cache_path, saved,
                                               (int)e->ctx, &saved_prompt,
                                               &saved_nmsg, &saved_uh)
                            : -1;
        uh = first_user_hash(roles, contents, nmsg);
        if (saved_n > 0) {
            cache_seen = 1;
            cache_nmsg = saved_nmsg;
            cache_user_ok = uh && uh == saved_uh;
        }
        /* OpenCode rewrites the system prompt every turn (files, clock,
         * tools). GDN cannot skip a changed prefix, but we still splice
         * the saved ids — the conversation is identified by the first
         * user message, not the system tokens. A resend with the same
         * message count is not a splice; the prompt snapshot covers it. */
        if (saved_n > 0 && saved_nmsg > 0 && nmsg > saved_nmsg &&
            saved_nmsg <= nmsg && saved_n < (int)e->ctx - 64 && uh &&
            uh == saved_uh) {
            memcpy(ids, saved, (size_t)saved_n * sizeof(int32_t));
            n = saved_n;
            int32_t im_end = q4_tok_im_end(e->tok);
            if (im_end >= 0 && ids[n - 1] != im_end && n + 8 < (int)e->ctx)
                ids[n++] = im_end;
            n += q4_tok_encode(e->tok, "\n", ids + n, (int)e->ctx - n);
            int from = saved_nmsg;
            if (from < nmsg && roles[from] &&
                strcmp(roles[from], "assistant") == 0)
                from++;
            n = q4_tok_apply_messages_from(e->tok, roles, contents, nmsg, from,
                                           ids, (int)e->ctx, n);
            spliced = 1;
            fprintf(stderr,
                    "q4: session splice kept %d tok + %d new  (msg %d..%d)\n",
                    saved_n, n - saved_n, from, nmsg);
        }
        free(saved);
    }
    if (!spliced)
        n = q4_tok_apply_messages(e->tok, roles, contents, nmsg, ids, (int)e->ctx);
    if (n <= 0 || n + 8 >= (int)q4_sess_max_kv(e->sess)) {
        fprintf(stderr, "q4: prompt tokens %d do not fit ctx %u\n", n,
                q4_sess_max_kv(e->sess));
        free(ids);
        return -1;
    }
    {
        int room = (int)q4_sess_max_kv(e->sess) - n - 8;
        if (room < 1) room = 1;
        if (max_new > room) max_new = room;
    }
    q4_sess_reset(e->sess);
    q4_mtp_reset(e->mtp);
    q4_expert_clear_sticky(e->ex);
    e->prompt_tokens = n;
    e->completion_tokens = 0;
    e->hit_limit = 0;
    e->reason_n = 0;
    if (e->reason) e->reason[0] = 0;
    /* Prefix cache. The main file is the last finished turn (prompt plus
     * generated tokens). The .prompt file is the prompt boundary, written
     * when prefill finishes or the client disconnects, so a retry of the
     * same body resumes instead of recomputing. The longer exact prefix
     * wins. A longer snapshot cannot be rewound. */
    int start = 0;
    int hit_side = 0;
    if (cache_path && cache_path[0]) {
        int32_t *scratch = malloc((size_t)e->ctx * sizeof(int32_t));
        int main_len = 0, side_len = 0;
        if (scratch) {
            main_len = cache_prefix_len(e->sess, cache_path, ids, n, scratch,
                                        (int)e->ctx);
            side_len = cache_prefix_len(e->sess, prompt_path, ids, n, scratch,
                                        (int)e->ctx);
            free(scratch);
        }
        const char *use = NULL;
        if (side_len > main_len || (side_len == n && side_len >= main_len &&
                                    side_len > 0)) {
            use = prompt_path;
            hit_side = 1;
        } else if (main_len > 0) {
            use = cache_path;
        }
        if (use) {
            double tc = now_s();
            start = q4_sess_restore(e->sess, use, ids, n);
            if (start > 0)
                fprintf(stderr, "q4: session cache hit%s %d/%d tok (%.2f s)\n",
                        hit_side ? " prompt" : "", start, n, now_s() - tc);
        }
        if (start <= 0 && cache_seen)
            fprintf(stderr,
                    "q4: session cache miss (nmsg %d/%d user %s)\n", cache_nmsg,
                    nmsg, cache_user_ok ? "match" : "mismatch");
    }
    {
        FILE *st = fopen("/proc/self/status", "r");
        if (st) {
            char line[128];
            while (fgets(line, sizeof(line), st)) {
                unsigned long sw = 0;
                if (sscanf(line, "VmSwap: %lu", &sw) == 1 && sw > 256ull * 1024ull) {
                    fprintf(stderr,
                            "q4: WARNING self VmSwap %.1f GiB — L2 is in swap, "
                            "decode will collapse. Close browsers / do not delete "
                            "large files while q4 runs.\n",
                            (double)sw / (1024.0 * 1024.0));
                    break;
                }
            }
            fclose(st);
        }
    }
    fprintf(stderr, "q4: generate prompt=%d max_new=%d (layer-first prefill)\n", n,
            max_new);
    fflush(stderr);
    double t0 = now_s();
    const int chunk = (int)q4_pref_chunk();
    e->pref_base = 0;
    e->pref_chunk = chunk;
    e->pref_total = n;
    q4_sess_set_pref_hook(e->sess, pref_layer_hook, e);
    int filled = start;
    int ckpt_at = 0;
    int ckpt_fail = 0;
    double ckpt_dt = 0;
    for (int i = start; i < n && !q4_stop;) {
        if (client_gone(e)) break;
        int c = n - i;
        if (c > chunk) c = chunk;
        e->pref_base = i;
        e->pref_chunk = c;
        if (!q4_sess_prefill(e->sess, ids + i, c)) {
            q4_sess_set_pref_hook(e->sess, NULL, NULL);
            fprintf(stderr, "q4: prefill failed at %d / %d  id=%d\n", i, n,
                    ids[i]);
            free(ids);
            return -1;
        }
        if (e->mtp && !e->mtp_fail &&
            !q4_mtp_prefill_chunk(e->mtp, e->sess, ids + i, c)) {
            fprintf(stderr, "q4: MTP head prefill failed, disabling MTP\n");
            q4_mtp_close(e->mtp);
            e->mtp = NULL;
        }
        i += c;
        filled = i;
        fprintf(stderr, "\rprefill %d/%d", i, n);
        if (getenv("Q4_PROFILE"))
            fprintf(stderr, "  (chunk %.2f s  io %.2f)", now_s() - t0,
                    q4_prof_moe_io);
        fflush(stderr);
        if (e->progress) e->progress(i, n, 0, e->progress_u);
        /* One snapshot of tokens [0, i). That is enough to resume: GDN
         * state is valid at this boundary. Skip the intermediate chunks
         * while the client is still connected. The main file still holds
         * the previous finished turn until decode ends. */
        if (prompt_path[0] && (i == n || client_gone(e)) &&
            (uint32_t)i == q4_sess_n_kv(e->sess)) {
            double tc = now_s();
            if (q4_sess_save(e->sess, prompt_path, ids, i, i, nmsg, uh)) {
                ckpt_at = i;
                ckpt_dt = now_s() - tc;
            } else if (!ckpt_fail) {
                fprintf(stderr, "\nq4: prompt checkpoint failed at %d\n", i);
                ckpt_fail = 1;
            }
        }
        if (client_gone(e)) break;
    }
    q4_sess_set_pref_hook(e->sess, NULL, NULL);
    if (client_gone(e)) {
        fprintf(stderr, "\nq4: client closed at prefill %d/%d\n", filled, n);
        if (ckpt_at > 0)
            fprintf(stderr, "q4: prompt checkpoint %d tok (%.2f s)\n", ckpt_at,
                    ckpt_dt);
        fflush(stderr);
        free(ids);
        return -1;
    }
    /* A full-cache hit never entered the loop, so .prompt would stay
     * older than the prompt. Write it before decode appends tokens to
     * the main file and makes a same-body retry miss. */
    if (prompt_path[0] && start == n && !hit_side &&
        (uint32_t)n == q4_sess_n_kv(e->sess)) {
        double tc = now_s();
        if (q4_sess_save(e->sess, prompt_path, ids, n, n, nmsg, uh)) {
            ckpt_at = n;
            ckpt_dt = now_s() - tc;
        } else if (!ckpt_fail) {
            fprintf(stderr, "q4: prompt checkpoint failed at %d\n", n);
            ckpt_fail = 1;
        }
    }
    if (ckpt_at == n && ckpt_at > 0)
        fprintf(stderr, "q4: prompt checkpoint %d tok (%.2f s)\n", ckpt_at,
                ckpt_dt);
    if (!q4_sess_head(e->sess, NULL, NULL)) {
        fprintf(stderr, "\nq4: lm-head after prefill failed\n");
        free(ids);
        return -1;
    }
    double dt = now_s() - t0;
    {
        int did = n - start;
        fprintf(stderr, "\rprefill %d+%d/%d  %.2f t/s                \n", start, did,
                n, dt > 0 && did > 0 ? did / dt : 0);
    }
    if (getenv("Q4_PROFILE")) {
        extern double q4_prof_sec[8];
        extern double q4_prof_moe_cw;
        fprintf(stderr,
                "q4 prof prefill: moe io %.2f (copy_wait %.2f)  d2h %.2f  "
                "gemm %.2f s\n"
                "q4 prof sections: ple %.2f hc_attn %.2f qsa %.2f comb %.2f "
                "hc_ffn %.2f moe %.2f comb2 %.2f gdn %.2f s\n",
                q4_prof_moe_io, q4_prof_moe_cw, q4_prof_moe_d2h, q4_prof_moe_gemm,
                q4_prof_sec[0], q4_prof_sec[1], q4_prof_sec[2], q4_prof_sec[3],
                q4_prof_sec[4], q4_prof_sec[5], q4_prof_sec[6], q4_prof_sec[7]);
    }
    fflush(stderr);
    if (e->progress) e->progress(n, n, 0, e->progress_u);

    const float *logits = q4_sess_logits(e->sess);
    if (!logits) {
        free(ids);
        return -1;
    }
    int o = 0;
    out[0] = 0;
    int32_t im_end = q4_tok_im_end(e->tok);
    int32_t eos = q4_tok_eos(e->tok);
    int32_t th = q4_tok_id(e->tok, "<think>");
    int32_t te = q4_tok_id(e->tok, "</think>");
    /* Prompt already opened <think> when thinking is on; generation starts
     * inside reasoning. Tags themselves must not go into the answer body. */
    int in_think = q4_tok_think_enabled();
    int think_toks = 0;
    int32_t recent[32];
    unsigned char hold[8];
    int nh = 0;
    int32_t *gen_ids = cache_path && cache_path[0]
                           ? malloc((size_t)max_new * sizeof(int32_t))
                           : NULL;
    int n_gen = 0;
    t0 = now_s();
    int t = 0;
    int n_draft = 2;
    {
        const char *dn = getenv("Q4_MTP_N");
        if (dn && dn[0]) {
            int v = atoi(dn);
            if (v >= 1) n_draft = v;
        }
    }
    int mtp_mode = e->mtp && !e->mtp_fail;
    int32_t c0 = 0;
    if (mtp_mode) {
        /* Seed the chain when the whole prompt came from the session cache
         * and no prefill chunk ran this turn. */
        if (start >= n) {
            if (n <= 0 || !q4_mtp_seed(e->mtp, e->sess, ids[n - 1])) {
                fprintf(stderr, "q4: MTP seed failed, plain decode\n");
                mtp_mode = 0;
                e->mtp_fail = 1;
            }
        }
    }
    if (mtp_mode)
        c0 = q4_sample(logits, e->g.n_vocab, e->temp, e->top_k, e->top_p,
                       &e->rng);
    int stop = 0;
    int closed = 0;
    uint32_t repin_every = 64;
    {
        const char *re = getenv("Q4_REPIN_EVERY");
        if (re && re[0]) {
            long v = strtol(re, NULL, 10);
            repin_every = v < 0 ? 0 : (uint32_t)v;
        }
    }
    while (t < max_new && !q4_stop && !stop && !client_gone(e)) {
        if (isnan(logits[0]) || isinf(logits[0]) ||
            isnan(logits[e->g.n_vocab / 2]) || isinf(logits[e->g.n_vocab / 2]) ||
            isnan(logits[e->g.n_vocab - 1]) || isinf(logits[e->g.n_vocab - 1])) {
            fprintf(stderr, "\nq4: NaN logits at gen %d, stop\n", t);
            break;
        }
        int32_t id;
        if (mtp_mode)
            id = c0;
        else
            id = q4_sample(logits, e->g.n_vocab, e->temp, e->top_k, e->top_p,
                           &e->rng);
        int rc = emit_piece(e, id, out, &o, max_out, &in_think, &think_toks,
                            hold, &nh, stream_cb, u, t, im_end, eos, th, te);
        if (rc == 1) break;
        if (rc == 2) {
            e->hit_limit = 1;
            break;
        }
        t++;
        e->completion_tokens++;
        /* Adaptive re-pin: promote recurrent hot experts so the LRU tail
         * stops thrashing on working sets larger than L1. O(n_slots). */
        if (repin_every && e->ex && (t % repin_every) == 0)
            (void)q4_expert_pin_thresh(e->ex, warm_pin_cap(e), 8);
        if (gen_ids) gen_ids[n_gen++] = id;
        recent[(t - 1) & 31] = id;
        if (t % 4 == 0 || t == 1) {
            fprintf(stderr, "\rdecode %d/%d", t, max_new);
            fflush(stderr);
            if (e->progress) e->progress(n, n, t, e->progress_u);
        }
        if (t >= max_new) {
            e->hit_limit = 1;
            break;
        }
        if (mtp_mode) {
            if (q4_sess_n_kv(e->sess) + (uint32_t)n_draft + 2 >=
                q4_sess_max_kv(e->sess)) {
                e->hit_limit = 1;
                break;
            }
            int32_t drafts[4];
            int nd_out = 0;
            int32_t nxt = 0;
            const float *nlog = NULL;
            int crc = q4_mtp_cycle(e->mtp, e->sess, id, n_draft, e->temp,
                                   e->top_k, e->top_p, &e->rng, drafts, &nd_out,
                                   &nxt, &nlog);
            if (crc == 1) {
                /* Recovered with only c0 kept: continue in plain mode. */
                fprintf(stderr, "\nq4: MTP cycle recovered at gen %d\n", t);
                e->mtp_fail = 1;
                mtp_mode = 0;
                logits = q4_sess_logits(e->sess);
                if (!logits) break;
                continue;
            }
            if (crc < 0) {
                fprintf(stderr,
                        "\nq4: MTP cycle failed at gen %d, plain decode\n", t);
                e->mtp_fail = 1;
                mtp_mode = 0;
                /* c0 was emitted but not yet processed by the model. */
                if (!q4_sess_decode_ex(e->sess, id, NULL, NULL, true)) break;
                logits = q4_sess_logits(e->sess);
                if (!logits) break;
                continue;
            }
            if (crc == 2) {
                /* Partial repair: keep the accepted tokens, then plain. */
                fprintf(stderr, "\nq4: MTP partial repair at gen %d\n", t);
                e->mtp_fail = 1;
                mtp_mode = 0;
            }
            for (int i = 0; i < nd_out; i++) {
                rc = emit_piece(e, drafts[i], out, &o, max_out, &in_think,
                                &think_toks, hold, &nh, stream_cb, u, t, im_end,
                                eos, th, te);
                if (rc == 1) {
                    stop = 1;
                    break;
                }
                if (rc == 2) {
                    e->hit_limit = 1;
                    stop = 1;
                    break;
                }
                t++;
                e->completion_tokens++;
                if (gen_ids) gen_ids[n_gen++] = drafts[i];
                recent[(t - 1) & 31] = drafts[i];
                if (t % 4 == 0 || t == 1) {
                    fprintf(stderr, "\rdecode %d/%d", t, max_new);
                    fflush(stderr);
                    if (e->progress) e->progress(n, n, t, e->progress_u);
                }
                if (t >= max_new) {
                    e->hit_limit = 1;
                    stop = 1;
                    break;
                }
            }
            if (stop) break;
            if (crc == 2) {
                /* Partial repair: sample the next token ourselves. */
                logits = q4_sess_logits(e->sess);
                if (!logits) break;
                continue;
            }
            c0 = nxt;
            logits = nlog ? nlog : logits;
        } else {
            if (q4_sess_n_kv(e->sess) + 1 >= q4_sess_max_kv(e->sess)) {
                e->hit_limit = 1;
                break;
            }
            if (!q4_sess_decode_ex(e->sess, id, NULL, NULL, true)) {
                fprintf(stderr, "\nq4: decode failed at gen %d\n", t);
                break;
            }
            logits = q4_sess_logits(e->sess);
            if (!logits) break;
        }
        if (t >= 23) {
            int same = 1;
            for (int k = 0; k < 8; k++)
                if (recent[(t - 1 - k) & 31] != recent[(t - 9 - k) & 31]) {
                    same = 0;
                    break;
                }
            if (same) {
                fprintf(stderr, "\nq4: repeat loop at gen %d, stop\n", t);
                break;
            }
        }
    }
    if (t >= max_new) e->hit_limit = 1;
    if (in_think && think_toks)
        fprintf(stderr, "\nq4: still thinking at stop (%d think tok, limit=%d)\n",
                think_toks, e->hit_limit);
    dt = now_s() - t0;
    if (e->completion_tokens)
        fprintf(stderr, "\rdecode %d tok  %.2f t/s                \n",
                e->completion_tokens,
                dt > 0 ? e->completion_tokens / dt : 0);
    if (getenv("Q4_STEP_PROF") && e->completion_tokens) {
        double inv = 1.0 / e->completion_tokens;
        fprintf(stderr,
                "  step/tok ms   gA+sync %.2f  hostpart %.2f  gB %.2f  "
                "e-attn %.2f  e-moe %.2f  e-comb %.2f  embed %.2f  head %.2f\n",
                q4_prof_step[0] * inv * 1e3, q4_prof_step[1] * inv * 1e3,
                q4_prof_step[2] * inv * 1e3, q4_prof_step[3] * inv * 1e3,
                q4_prof_step[4] * inv * 1e3, q4_prof_step[5] * inv * 1e3,
                q4_prof_step[6] * inv * 1e3, q4_prof_step[7] * inv * 1e3);
        memset(q4_prof_step, 0, sizeof(q4_prof_step));
        if (q4_prof_gpu_sec[0] || q4_prof_gpu_sec[1] || q4_prof_gpu_sec[2] ||
            q4_prof_gpu_sec[3] || q4_prof_gpu_sec[4])
            fprintf(stderr,
                    "  gpu/tok ms    hc1 %.2f  attn %.2f  comb %.2f  "
                    "ffnmix %.2f  moea %.2f\n",
                    q4_prof_gpu_sec[0] * inv, q4_prof_gpu_sec[1] * inv,
                    q4_prof_gpu_sec[2] * inv, q4_prof_gpu_sec[3] * inv,
                    q4_prof_gpu_sec[4] * inv);
        if (q4_prof_gpu_sec[5] || q4_prof_gpu_sec[6] || q4_prof_gpu_sec[7])
            fprintf(stderr,
                    "  qsa/tok ms    prep %.2f  sel %.2f  dec %.2f\n",
                    q4_prof_gpu_sec[5] * inv, q4_prof_gpu_sec[6] * inv,
                    q4_prof_gpu_sec[7] * inv);
        memset(q4_prof_gpu_sec, 0, sizeof(q4_prof_gpu_sec));
    }
    if (getenv("Q4_STEP_PROF") && e->completion_tokens) {
        static unsigned long long pj, pm, pl;
        unsigned long long dj, dm, dl;
        q4_moe_dev_stats(&dj, &dm, &dl);
        if (dj - pj) {
            double rt = q4_moe_route_prof();
            fprintf(stderr,
                    "  devroute      jobs %llu  misses %llu  "
                    "%.0f us/job  route %.0f us\n",
                    dj - pj, dm - pm,
                    (double)(dl - pl) / (double)(dj - pj), rt);
        }
        {
            unsigned long long wc = 0;
            double wa = q4_hip_cpx_wait_prof(&wc);
            if (wc && wa >= 0)
                fprintf(stderr, "  cpxwait       %.0f us x%llu = %.1f ms/tok\n",
                        wa, wc, wa * (double)wc / 1e3 /
                                (double)e->completion_tokens);
            extern double q4_prof_gpu_ms;
            fprintf(stderr, "  gpu-busy/tok  %.2f ms\n",
                    q4_prof_gpu_ms / (double)e->completion_tokens);
            q4_prof_gpu_ms = 0;
        }
        pj = dj; pm = dm; pl = dl;
        extern double q4_prof_moe_cw;
        static double pi, pcw, pg, pt, pj2, png, pnc, pnl;
        if (q4_prof_moe_ncall > pnl)
            fprintf(stderr,
                    "  moe/layer     topk %.0f us  cw %.0f us  io %.0f us  "
                    "gemm %.0f us  join %.0f us  gpu-hit %.1f  cpu %.1f\n",
                    (q4_prof_moe_topk - pt) / (q4_prof_moe_ncall - pnl) * 1e6,
                    (q4_prof_moe_cw - pcw) / (q4_prof_moe_ncall - pnl) * 1e6,
                    (q4_prof_moe_io - pi) / (q4_prof_moe_ncall - pnl) * 1e6,
                    (q4_prof_moe_gemm - pg) / (q4_prof_moe_ncall - pnl) * 1e6,
                    (q4_prof_moe_join - pj2) / (q4_prof_moe_ncall - pnl) * 1e6,
                    (q4_prof_moe_ng - png) / (q4_prof_moe_ncall - pnl),
                    (q4_prof_moe_nc - pnc) / (q4_prof_moe_ncall - pnl));
        pi = q4_prof_moe_io; pcw = q4_prof_moe_cw; pg = q4_prof_moe_gemm;
        pt = q4_prof_moe_topk; pj2 = q4_prof_moe_join;
        png = q4_prof_moe_ng; pnc = q4_prof_moe_nc; pnl = q4_prof_moe_ncall;
    }
    if (e->mtp) {
        uint64_t prop = 0, acc = 0, cyc = 0, gat = 0;
        q4_mtp_stats(e->mtp, &prop, &acc, &cyc, &gat);
        if (cyc)
            fprintf(stderr,
                    "q4: mtp draft acceptance = %.5f (%llu accepted / %llu "
                    "drafted, %llu gated), mean len = %.2f (%llu cycles)\n",
                    prop ? (double)acc / (double)prop : 0.0,
                    (unsigned long long)acc, (unsigned long long)prop,
                    (unsigned long long)gat,
                    (double)e->completion_tokens / (double)cyc,
                    (unsigned long long)cyc);
        if (getenv("Q4_PROFILE"))
            q4_mtp_prof_print(e->mtp, stderr);
        if (getenv("Q4_PROFILE"))
            q4_moe_sb_prof_print(stderr);
    }
    if (getenv("Q4_PROFILE"))
        fprintf(stderr, "q4 prof decode: moe io %.2f  d2h %.2f  gemm %.2f s\n",
                q4_prof_moe_io, q4_prof_moe_d2h, q4_prof_moe_gemm);
    fflush(stderr);
    if (client_gone(e)) closed = 1;
    if (gen_ids) {
        if (n_gen <= 0 || q4_stop || closed) {
            fprintf(stderr,
                    "q4: session cache not saved (gen %d stop %d closed %d)\n",
                    n_gen, (int)q4_stop, closed);
        } else {
            int32_t *all = malloc((size_t)(n + n_gen) * sizeof(int32_t));
            if (all) {
                memcpy(all, ids, (size_t)n * sizeof(int32_t));
                memcpy(all + n, gen_ids, (size_t)n_gen * sizeof(int32_t));
                double tc = now_s();
                uh = first_user_hash(roles, contents, nmsg);
                if (q4_sess_save(e->sess, cache_path, all, n + n_gen, n, nmsg, uh))
                    fprintf(stderr,
                            "q4: session cache saved %d tok (prompt %d msg %d, %.2f s)\n",
                            n + n_gen, n, nmsg, now_s() - tc);
                free(all);
            }
        }
        free(gen_ids);
    }
    free(ids);
    return o;
}

int q4_engine_generate_msgs(q4_engine *e, const char *const *roles,
                            const char *const *contents, int nmsg, char *out,
                            int max_out, int max_new,
                            void (*stream_cb)(const char *piece, int kind, void *u),
                            void *u) {
    if (!e) return -1;
    pthread_mutex_lock(&e->mu);
    int n = generate_locked(e, roles, contents, nmsg, out, max_out, max_new,
                            stream_cb, u);
    pthread_mutex_unlock(&e->mu);
    return n;
}

int q4_engine_generate(q4_engine *e, const char *system, const char *user,
                       char *out, int max_out, int max_new,
                       void (*stream_cb)(const char *piece, int kind, void *u),
                       void *u) {
    const char *roles[2];
    const char *contents[2];
    int n = 0;
    if (system && system[0]) {
        roles[n] = "system";
        contents[n] = system;
        n++;
    }
    roles[n] = "user";
    contents[n] = user ? user : "";
    n++;
    return q4_engine_generate_msgs(e, roles, contents, n, out, max_out, max_new,
                                   stream_cb, u);
}

#define Q4_WARM_MAGIC 0x57345134u /* "Q4WS" LE */
#define Q4_WARM_VER 1u

/* One decode step touches n_layer * top-k distinct experts (480 here).
 * Pin ~1.5× that, and never more than 1/4 of L1, so a long prefill can
 * still evict. Q4_PIN_CAP is an exact top-N override. */
static uint32_t warm_pin_cap(const q4_engine *e) {
    uint32_t nslots = e && e->ex ? q4_expert_cache_n_slots(e->ex) : 0;
    return q4_expert_warm_cap(nslots, e->g.n_layer, e->g.n_expert_used);
}

static int pin_cap_exact(void) {
    const char *env = getenv("Q4_PIN_CAP");
    return env && env[0] && strtol(env, NULL, 10) > 0;
}

static void preview_line(FILE *fp, const char *s) {
    if (!fp || !s || !s[0]) return;
    char b[161];
    int n = 0;
    for (int i = 0; s[i] && n < 160; i++) {
        unsigned char c = (unsigned char)s[i];
        b[n++] = c < 32 ? ' ' : (char)c;
    }
    b[n] = 0;
    fprintf(fp, "  sample: %s\n", b);
}

int q4_engine_warm_save(q4_engine *e, const char *path) {
    if (!e || !path) return -1;
    uint32_t n = q4_expert_pinned_count(e->ex);
    if (!n) return -1;
    int32_t *layers = malloc((size_t)n * sizeof(int32_t));
    int32_t *eids = malloc((size_t)n * sizeof(int32_t));
    uint32_t *hits = malloc((size_t)n * sizeof(uint32_t));
    if (!layers || !eids || !hits) {
        free(layers); free(eids); free(hits);
        return -1;
    }
    q4_expert_pinned_list(e->ex, layers, eids, hits, n);
    struct {
        uint32_t magic, version;
        uint64_t bytes_total;
        uint32_t n_layer, n_expert, count;
    } h = {Q4_WARM_MAGIC, Q4_WARM_VER, e->g.bytes_total, e->g.n_layer,
           e->g.n_expert, n};
    char tmp[4096 + 16];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    bool ok = f && fwrite(&h, 1, sizeof(h), f) == sizeof(h) &&
              fwrite(layers, 1, (size_t)n * 4, f) == (size_t)n * 4 &&
              fwrite(eids, 1, (size_t)n * 4, f) == (size_t)n * 4 &&
              fwrite(hits, 1, (size_t)n * 4, f) == (size_t)n * 4;
    if (f) fclose(f);
    free(layers); free(eids); free(hits);
    if (!ok) {
        if (f) remove(tmp);
        return -1;
    }
    if (rename(tmp, path) != 0) {
        remove(tmp);
        return -1;
    }
    return (int)n;
}

int q4_engine_warm_load(q4_engine *e, const char *path, FILE *fp) {
    if (!e || !path) return -1;
    return q4_expert_warm_load(e->ex, &e->g, path, warm_pin_cap(e), fp);
}

int q4_engine_route_save(q4_engine *e, const char *path) {
    if (!e || !path) return -1;
    return q4_expert_route_save(e->ex, &e->g, path, stderr);
}

int q4_engine_pin_ja(q4_engine *e, FILE *fp) {
    if (!e) return -1;
    static const char *prompts[] = {
        "日本語で自己紹介してください。日常会話、技術、法律、文学のどれが得意か、"
        "短く答えてください。",
        "次の文章を丁寧なビジネス日本語に直してください。「明日の会議、資料まだだ"
        "から後で送る。よろしく。」",
        "Rustで簡単なHTTPサーバーを書く手順を、日本語で箇条書きにしてください。"
        "エラー処理と並行性にも触れて。",
        "二次方程式 ax^2+bx+c=0 の解の公式を導出し、日本語で途中式を説明して"
        "ください。",
        "日本国憲法第9条の趣旨を、高校生向けの日本語で200字以内にまとめて"
        "ください。",
        "夏目漱石『こころ』の先生とKの関係を、Spoilers込みで短く解説して。",
        "光合成の明反応と暗反応の違いを、日本語の科学用語を使って説明して。",
        "次を自然な日本語に訳して: Caching routed experts in VRAM avoids SSD "
        "round-trips on every decode step.",
        "このC言語のバグを日本語で指摘して。int n=strlen(s); char b[n]; "
        "strcpy(b,s); return b;",
        "京都の夏の過ごし方、屋内で涼しく働く人向けに、具体的なアドバイスを"
        "日本語で。",
    };
    const int np = (int)(sizeof(prompts) / sizeof(prompts[0]));
    if (fp)
        fprintf(fp, "q4: Japanese expert warmup (%d prompts)...\n", np);
    float save_temp = e->temp;
    int save_k = e->top_k;
    float save_p = e->top_p;
    e->temp = 0.f;
    e->top_k = 1;
    e->top_p = 1.f;
    uint32_t nslots = q4_expert_cache_n_slots(e->ex);
    /* Pin after every prompt. Pinning occupied slots 0..cap on the first
     * generate froze the first-touched experts and starved later prompts. */
    uint32_t cap = warm_pin_cap(e);
    uint32_t step = e->g.n_layer * (e->g.n_expert_used ? e->g.n_expert_used : 10u);
    if (fp) {
        fprintf(fp,
                "q4: warmup %d prompts, then pin %s %u / %u L1 (decode step %u)\n",
                np, pin_cap_exact() ? "top" : "stable", cap, nslots, step);
        fflush(fp);
    }
    char buf[4096];
    for (int i = 0; i < np && !q4_stop; i++) {
        if (fp) fprintf(fp, "  ja[%d/%d] ", i + 1, np);
        int n = q4_engine_generate(e, "日本語で簡潔に答えてください。", prompts[i],
                                   buf, (int)sizeof(buf), 16, NULL, NULL);
        if (fp)
            fprintf(fp, "gen %d B\n", n < 0 ? 0 : n);
        if (fp && i == 0) {
            const char *s = (n > 0) ? buf : q4_engine_last_reasoning(e);
            preview_line(fp, s);
        }
        fflush(fp ? fp : stderr);
    }
    uint32_t add = pin_cap_exact() ? q4_expert_pin_hottest(e->ex, cap)
                                   : q4_expert_pin_stable(e->ex, cap, 8);
    if (fp)
        fprintf(fp, "q4: pinned %u L1 slots (cap %u)\n", add, cap);
    e->temp = save_temp;
    e->top_k = save_k;
    e->top_p = save_p;
    if (fp) {
        uint64_t l1h = 0, l2h = 0, ssd = 0, bytes = 0;
        q4_expert_cache_stats_ex(e->ex, &l1h, &l2h, &ssd, &bytes);
        fprintf(fp,
                "q4: pinned %u / %u L1 slots  cache L1 %llu  L2 %llu  SSD miss %llu\n",
                q4_expert_pinned_count(e->ex), nslots,
                (unsigned long long)l1h, (unsigned long long)l2h,
                (unsigned long long)ssd);
    }
    q4_expert_clear_sticky(e->ex);
    q4_sess_reset(e->sess);
    return (int)q4_expert_pinned_count(e->ex);
}

const q4_gguf *q4_engine_gguf(const q4_engine *e) {
    return e ? &e->g : NULL;
}

static int32_t argmax_logits(const float *logits, uint32_t n) {
    int32_t id = 0;
    float best = logits ? logits[0] : 0.f;
    for (uint32_t i = 1; i < n; i++) {
        if (logits[i] > best) {
            best = logits[i];
            id = (int32_t)i;
        }
    }
    return id;
}

/* Teacher-forced quality probe: feed ids in 8-token verify windows and score
 * every next token from the target logits of its row. Window w covers
 * ids[o..o+7]; row r predicts ids[o+r+1]. Reports mean NLL, perplexity and
 * top-1 agreement; dump (if non-NULL) gets "pos target nll argmax" lines. */
int q4_engine_ppl(q4_engine *e, const int32_t *ids, int n, FILE *fp,
                  FILE *dump) {
    if (!e || !ids || n < 2 || !fp) return -1;
    uint32_t max_kv = q4_sess_max_kv(e->sess);
    if (n + 16 > (int)max_kv) n = (int)max_kv - 16;
    if (n < 2) return -1;
    const uint32_t nv = e->g.n_vocab;
    q4_sess_reset(e->sess);
    q4_expert_clear_sticky(e->ex);
    double nll_sum = 0;
    uint64_t scored = 0, top1 = 0;
    double t0 = now_s();
    for (int o = 0; o + 1 < n && !q4_stop; o += 8) {
        int w = n - o < 8 ? n - o : 8;
        if (!q4_sess_prefill_tokens(e->sess, ids + o, w)) {
            fprintf(fp, "PPL_ABORT prefill at %d\n", o);
            return -1;
        }
        for (int r = 0; r < w && o + r + 1 < n; r++) {
            if (!q4_sess_logits_at(e->sess, r)) {
                fprintf(fp, "PPL_ABORT logits at %d\n", o + r);
                return -1;
            }
            const float *lg = q4_sess_logits(e->sess);
            const int32_t tgt = ids[o + r + 1];
            if (!lg || tgt < 0 || (uint32_t)tgt >= nv) {
                fprintf(fp, "PPL_ABORT row %d\n", o + r);
                return -1;
            }
            int32_t am = argmax_logits(lg, nv);
            float mx = lg[am];
            double se = 0;
            for (uint32_t i = 0; i < nv; i++) se += exp((double)(lg[i] - mx));
            double nll = -((double)(lg[tgt] - mx) - log(se));
            nll_sum += nll;
            scored++;
            if (am == tgt) top1++;
            if (dump)
                fprintf(dump, "%d %d %.6f %d\n", o + r + 1, tgt, nll, am);
        }
        if (w == 8 && (o / 8) % 64 == 63) {
            fprintf(fp, "  ppl at %d  nll %.4f  top1 %.2f%%\n", o + w,
                    nll_sum / (double)scored, 100.0 * (double)top1 / (double)scored);
            fflush(fp);
        }
    }
    double dt = now_s() - t0;
    if (!scored) return -1;
    double mn = nll_sum / (double)scored;
    fprintf(fp, "PPL_OK tokens %llu  nll %.5f  ppl %.4f  top1 %.3f%%  %.1f s\n",
            (unsigned long long)scored, mn, exp(mn),
            100.0 * (double)top1 / (double)scored, dt);
    fflush(fp);
    return 0;
}

int q4_engine_bench(q4_engine *e, const int32_t *ids, int n, int n_dec, FILE *fp) {
    if (!e || !ids || n <= 0 || !fp) return -1;
    if (n_dec < 0) n_dec = 0;
    uint32_t max_kv = q4_sess_max_kv(e->sess);
    if (n_dec + 8 >= (int)max_kv) n_dec = (int)max_kv > 16 ? (int)max_kv - 16 : 0;
    if (n + n_dec + 4 > (int)max_kv) n = (int)max_kv - n_dec - 4;
    if (n < 1) return -1;
    q4_sess_reset(e->sess);
    q4_expert_clear_sticky(e->ex);
    uint64_t l1a = 0, l2a = 0, ssda = 0, ba = 0;
    q4_expert_cache_stats_ex(e->ex, &l1a, &l2a, &ssda, &ba);
    fprintf(fp,
            "bench start  tokens=%d decode=%d ctx=%u pinned=%u L1=%u\n",
            n, n_dec, q4_sess_max_kv(e->sess), q4_expert_pinned_count(e->ex),
            q4_expert_cache_n_slots(e->ex));
    fflush(fp);
    double t_all = now_s();
    int off = 0, chunk_i = 0;
    while (off < n && !q4_stop) {
        int csz = n - off;
        if (csz > 512) csz = 512;
        uint64_t l1b = 0, l2b = 0, ssdb = 0, bb = 0;
        q4_expert_cache_stats_ex(e->ex, &l1b, &l2b, &ssdb, &bb);
        uint64_t avail = mem_available();
        if (avail && avail < 8ull * Q4_GIB) {
            fprintf(fp, "BENCH_ABORT mem %.2f GiB\n", (double)avail / Q4_GIB);
            fflush(fp);
            return -1;
        }
        double t0 = now_s();
        if (!q4_sess_prefill(e->sess, ids + off, csz)) {
            fprintf(fp, "BENCH_ABORT prefill at %d\n", off);
            fflush(fp);
            return -1;
        }
        double dt = now_s() - t0;
        off += csz;
        chunk_i++;
        uint64_t l1c = 0, l2c = 0, ssdc = 0, bc = 0;
        q4_expert_cache_stats_ex(e->ex, &l1c, &l2c, &ssdc, &bc);
        uint64_t d1 = l1c - l1b, d2 = l2c - l2b, ds = ssdc - ssdb;
        uint64_t den = d1 + d2 + ds;
        fprintf(fp,
                "  chunk %d  +%d  at %d  %.2f s  %.1f tok/s  L1 %llu  L2 %llu  "
                "SSD %llu  hit %.0f%%  ram %.1f GiB\n",
                chunk_i, csz, off, dt, dt > 0 ? (double)csz / dt : 0,
                (unsigned long long)d1, (unsigned long long)d2,
                (unsigned long long)ds, den ? 100.0 * (double)d1 / (double)den : 0,
                (double)mem_available() / Q4_GIB);
        fflush(fp);
    }
    double tall = now_s() - t_all;
    if (!q4_sess_head(e->sess, NULL, NULL)) {
        fprintf(fp, "BENCH_ABORT lm-head\n");
        return -1;
    }
    const float *logits = q4_sess_logits(e->sess);
    if (!logits) {
        fprintf(fp, "BENCH_ABORT logits\n");
        return -1;
    }
    int32_t id = argmax_logits(logits, e->g.n_vocab);
    int32_t *gen = n_dec ? malloc((size_t)n_dec * sizeof(int32_t)) : NULL;
    int got = 0;
    double tdec = 0, tdec_rest = 0;
    uint64_t l1d0 = 0, l2d0 = 0, s0 = 0, b0 = 0;
    q4_expert_cache_stats_ex(e->ex, &l1d0, &l2d0, &s0, &b0);
    for (int i = 0; i < n_dec && !q4_stop; i++) {
        if (mem_available() && mem_available() < 8ull * Q4_GIB) {
            fprintf(fp, "BENCH_ABORT mem during decode\n");
            free(gen);
            return -1;
        }
        int32_t nxt = 0;
        float lg = 0.f;
        double t0 = now_s();
        if (!q4_sess_decode_ex(e->sess, id, &nxt, &lg, true)) {
            fprintf(fp, "BENCH_ABORT decode at %d\n", i);
            free(gen);
            return -1;
        }
        double dt = now_s() - t0;
        fprintf(fp, "  dec %d  id %d -> %d  %.3f s  %.2f tok/s\n", i, id, nxt, dt,
                dt > 0 ? 1.0 / dt : 0);
        fflush(fp);
        if (gen) gen[got] = nxt;
        id = nxt;
        got++;
        tdec += dt;
        if (i > 0) tdec_rest += dt;
    }
    uint64_t l1e = 0, l2e = 0, se = 0, be = 0;
    q4_expert_cache_stats_ex(e->ex, &l1e, &l2e, &se, &be);
    fprintf(fp, "prefill total  %d tok  %.2f s  %.1f tok/s\n", n, tall,
            tall > 0 ? (double)n / tall : 0);
    if (got > 0) {
        double w = tdec / got;
        fprintf(fp, "decode mean    %.3f s  %.2f tok/s  (%d tok)\n", w,
                w > 0 ? 1.0 / w : 0, got);
    }
    if (got > 1) {
        double w = tdec_rest / (got - 1);
        fprintf(fp, "decode rest    %.3f s  %.2f tok/s  (%d tok)\n", w,
                w > 0 ? 1.0 / w : 0, got - 1);
    }
    if (gen && got > 0 && e->tok) {
        char text[512];
        int nb = q4_tok_decode(e->tok, gen, got, text, (int)sizeof(text) - 1);
        if (nb < 0) nb = 0;
        text[nb] = 0;
        preview_line(fp, text);
    }
    fprintf(fp,
            "expert cache   L1 +%llu  L2 +%llu  SSD %llu  SSD %.2f GiB  pinned %u\n",
            (unsigned long long)(l1e - l1a), (unsigned long long)(l2e - l2a),
            (unsigned long long)(se - ssda), (double)(be - ba) / Q4_GIB,
            q4_expert_pinned_count(e->ex));
    uint64_t dd1 = l1e - l1d0, dd2 = l2e - l2d0;
    uint64_t dden = dd1 + dd2 + (se - s0);
    fprintf(fp, "decode hits    L1 %llu  L2 %llu  hit %.0f%%\n",
            (unsigned long long)dd1, (unsigned long long)dd2,
            dden ? 100.0 * (double)dd1 / (double)dden : 0);
    fprintf(fp, "BENCH_OK\n");
    fflush(fp);
    free(gen);
    return 0;
}
