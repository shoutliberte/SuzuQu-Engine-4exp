#ifndef Q4_H
#define Q4_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <pthread.h>

#define Q4_GIB (1024ull * 1024ull * 1024ull)
#define Q4_MAX_DIMS 8
#define Q4_MAX_NAME 256
#define Q4_MAX_FILES 16
#define Q4_MAX_SHARDS 8
#define Q4_MAX_PLE_NGRAM 8
#define Q4_MAX_PLE_HEADS 64
#define Q4_MAX_LAYER 128

/* ggml type ids (GGUF). */
enum {
    Q4_T_F32 = 0,
    Q4_T_F16 = 1,
    Q4_T_Q4_0 = 2,
    Q4_T_Q5_0 = 6,
    Q4_T_Q5_1 = 7,
    Q4_T_Q8_0 = 8,
    Q4_T_Q4_K = 12,
    Q4_T_Q5_K = 13,
    Q4_T_Q6_K = 14,
    Q4_T_IQ3_XXS = 18,
    Q4_T_IQ4_NL = 20,
    Q4_T_IQ3_S = 21,
    Q4_T_IQ2_S = 22,
    Q4_T_IQ4_XS = 23,
    Q4_T_BF16 = 30,
    Q4_T_Q2_0 = 42,
};

typedef enum {
    Q4_KIND_OTHER = 0,
    Q4_KIND_DENSE,   /* attn, norms, embeddings, output, HC, GDN, QSA */
    Q4_KIND_PLE,     /* n-gram table + layer-1 PLE mixers */
    Q4_KIND_SHARED,  /* shared expert FFN */
    Q4_KIND_ROUTED,  /* ffn_*_exps */
    Q4_KIND_MTP,     /* blk.N nextn draft; not part of the main 48-layer forward */
} q4_kind;

typedef struct {
    char     name[Q4_MAX_NAME];
    uint32_t n_dims;
    uint64_t ne[Q4_MAX_DIMS];
    uint32_t ggml_type;
    uint64_t offset;      /* into that shard's data section */
    uint64_t nbytes;
    uint32_t shard;       /* index into q4_gguf.files */
    q4_kind  kind;
    int32_t  layer;       /* -1 if not blk.N */
    int32_t  n_experts;   /* last dim of routed tensors, else 0 */
} q4_tensor;

typedef struct {
    char     path[4096];
    int      fd;
    uint64_t file_size;
    uint64_t data_off;    /* byte offset of tensor payload */
} q4_file;

typedef struct {
    q4_file   files[Q4_MAX_FILES];
    uint32_t  n_files;
    q4_tensor *tensors;
    uint32_t  n_tensors;
    char     *arch;       /* malloc'd, may be NULL */
    char     *name;
    uint32_t  n_layer;
    uint32_t  n_embd;
    uint32_t  n_ff;
    uint32_t  n_expert;
    uint32_t  n_expert_used;
    uint32_t  n_head;
    uint32_t  n_head_kv;
    uint32_t  n_ctx_train;
    uint32_t  n_ff_exp;
    uint32_t  n_ff_shexp;
    uint32_t  hc_mult;
    uint32_t  hc_rank;
    uint32_t  full_attn_interval;
    uint32_t  n_embd_ple; /* 160: PLE row width */
    uint32_t  n_embd_head_k;
    uint32_t  n_embd_head_v;
    uint32_t  n_rot;
    uint32_t  n_vocab;
    uint32_t  ssm_d_conv;
    uint32_t  ssm_d_state;
    uint32_t  ssm_n_group;
    uint32_t  ssm_dt_rank;
    uint32_t  ssm_d_inner;
    uint32_t  indexer_n_head;
    uint32_t  indexer_head_size;
    uint32_t  indexer_top_k;
    uint32_t  compress_ratio[Q4_MAX_LAYER];
    int32_t   rope_sections[4];
    float     rope_freq_base;
    float     rms_eps;
    uint32_t  bos_id;
    uint32_t  eos_id;
    uint32_t  pad_id;
    uint32_t  ple_ngram;
    uint32_t  ple_heads_per;
    uint32_t  ple_n_heads;
    uint32_t  ple_layer; /* 0-based; UINT32_MAX if none */
    uint32_t  ple_eos;
    uint32_t  ple_image;
    uint32_t  ple_conv_kernel;
    uint64_t  ple_mult[Q4_MAX_PLE_NGRAM];
    uint64_t  ple_off[Q4_MAX_PLE_HEADS];
    uint64_t  ple_vocab[Q4_MAX_PLE_HEADS];
    int32_t   ple_table; /* tensor index, or -1 */
    uint64_t  bytes_total;
    uint64_t  bytes_dense;
    uint64_t  bytes_shared;
    uint64_t  bytes_routed;
    uint64_t  bytes_ple;
    uint64_t  bytes_mtp; /* 0 in the current Unsloth GGUF */
    uint64_t  per_expert_bytes; /* mean over routed tensors / n_expert */
    char    **tok_vocab;
    uint32_t  n_tok_vocab;
    char    **tok_merges;
    uint32_t  n_tok_merges;
    char     *chat_template;
    int32_t  *name_hash;   /* open-addressing name -> tensor index */
    uint32_t  name_hash_cap;
} q4_gguf;

typedef struct {
    uint64_t vram_bytes;
    uint64_t dram_bytes;
    uint64_t vram_reserve_bytes;  /* desktop + ROCm scratch */
    uint64_t kv_bytes;            /* context workspace estimate */
    uint32_t ctx;
} q4_machine;

typedef struct {
    uint64_t dense_vram;     /* non-routed resident in VRAM */
    uint64_t l1_expert_vram; /* explicit expert slots */
    uint32_t l1_experts;
    uint64_t kv_vram;
    uint64_t indexer_vram; /* QSA raw keys + block pool + score scratch */
    uint64_t scratch_vram; /* prefill buffers + workspace + alloc slack */
    uint64_t reserve_vram;
    uint64_t vram_used;
    uint64_t vram_free;
    uint64_t l2_page_cache;  /* leftover DRAM hint, not locked */
    uint64_t ple_host;       /* 0 reserved: PLE is SSD-backed demand I/O */
    bool     dense_fits_vram;
    bool     ok;
    const char *note;
} q4_place;

typedef struct q4_expert_cache q4_expert_cache;

bool q4_gguf_open(q4_gguf *g, const char *path);
bool q4_gguf_open_ex(q4_gguf *g, const char *path, int keep_mtp_layer);
void q4_gguf_close(q4_gguf *g);
void q4_gguf_print(const q4_gguf *g, FILE *fp);

q4_machine q4_machine_this_pc(uint32_t ctx);
q4_place   q4_place_plan(const q4_gguf *g, const q4_machine *m);
void       q4_place_print(const q4_place *p, const q4_gguf *g, FILE *fp);
/* Prefill tokens per chunk (Q4_PREF_CHUNK, 512..4096, multiple of 256) and
 * the d_pws floats it requires for this model. */
uint32_t   q4_pref_chunk(void);
uint64_t   q4_pws_floats(const q4_gguf *g, uint32_t T);
uint64_t   q4_qsa_kv_bytes(const q4_gguf *g, uint32_t ctx);
uint64_t   q4_qsa_indexer_bytes(const q4_gguf *g, uint32_t ctx);
uint64_t   q4_gdn_state_bytes(const q4_gguf *g);
uint32_t   q4_n_qsa_layers(const q4_gguf *g);

q4_expert_cache *q4_expert_cache_open(const q4_gguf *g, uint32_t n_slots);
q4_expert_cache *q4_expert_cache_open_ex(const q4_gguf *g, uint32_t n_l1,
                                         uint64_t l2_bytes);
/* Same, uploading into a specific GPU expert arena (0 main / 1 MTP head). */
q4_expert_cache *q4_expert_cache_open_arena(const q4_gguf *g, uint32_t n_l1,
                                            uint64_t l2_bytes, int arena);
/* res_mode: <0 env, 0 force non-resident, 1 force resident (MTP head). */
q4_expert_cache *q4_expert_cache_open_arena_r(const q4_gguf *g, uint32_t n_l1,
                                              uint64_t l2_bytes, int arena,
                                              int res_mode);
void q4_expert_cache_close(q4_expert_cache *c);
/* Load expert `eid` of layer `layer` into a slot. Returns bytes copied from SSD. */
int64_t q4_expert_cache_touch(q4_expert_cache *c, int32_t layer, int32_t eid);
void q4_expert_cache_stats(const q4_expert_cache *c,
                           uint64_t *hits, uint64_t *misses, uint64_t *bytes);
void q4_expert_cache_stats_ex(const q4_expert_cache *c, uint64_t *l1_hits,
                              uint64_t *l2_hits, uint64_t *ssd_misses,
                              uint64_t *bytes);
/* Round-robin fill of DRAM L2. Prints progress to fp if non-NULL. */
uint32_t q4_expert_prefetch(q4_expert_cache *c, FILE *fp);
/* Parallel staging: make all ids resident in L1 via the reader pool
 * (Q4_IO_THREADS, default 6; 1 = serial). If gpu_upload != 0, newly filled
 * slots are queued for H2D (q4_hip_experts_put_async); caller must
 * q4_hip_copy_wait() before launching kernels that read them.
 * Returns 0 on success. */
int q4_expert_stage_batch(q4_expert_cache *c, int32_t layer, const int32_t *ids,
                          uint32_t n, int gpu_upload);
/* Synchronous variant: misses are always bound before return (no
 * devroute quiesce deferral).  For callers that consume slots immediately
 * — prefill meta uploads, the GPU fallback exec path. */
/* Decode-time async miss staging: post the miss list to the submitter
 * thread and return immediately.  `hits` (the GPU-routed experts this
 * token) stay protected until the worker finishes binding the misses.
 * Returns nonzero when the batch was handed off. */
int q4_expert_stage_async(q4_expert_cache *c, int32_t layer,
                          const int32_t *miss_ids, uint32_t n_miss,
                          const int32_t *hit_ids, uint32_t n_hits);
/* Wait out the submitter — call before any lock-free cache-map read on the
 * decode thread (the topk split's q4_expert_gpu_slot). */
void q4_expert_stage_join(q4_expert_cache *c);
int  q4_expert_stage_pending(void);
int q4_expert_stage_batch_sync(q4_expert_cache *c, int32_t layer,
                               const int32_t *ids, uint32_t n,
                               int gpu_upload);
/* Whole-layer expert prefetch on a helper thread (prefill overlap). The
 * caller stickies the in-flight layer's experts before pf_begin and joins
 * before its own stage_batch calls. */
int q4_expert_pf_begin(q4_expert_cache *c, int32_t layer);
int q4_expert_pf_join(q4_expert_cache *c);
/* hipHostRegister the L1 arena: H2D uploads become true DMA (~2x). */
bool q4_expert_cache_pin_arena(q4_expert_cache *c);
/* mlock DRAM L2 so kswapd cannot steal experts into zram. */
bool q4_expert_cache_mlock_l2(q4_expert_cache *c);

/* Layer-ahead readahead thread for prefill (Q4_AHEAD=0 disables). */
typedef struct q4_ahead {
    pthread_t th;
    q4_expert_cache *c;
    volatile uint32_t cur;
    volatile int stop;
    int started;
} q4_ahead;
void q4_expert_ahead_begin(q4_ahead *a, q4_expert_cache *c, uint32_t cur_layer);
void q4_expert_ahead_end(q4_ahead *a);
uint64_t q4_expert_l2_bytes(const q4_expert_cache *c);
uint32_t q4_expert_l2_filled(const q4_expert_cache *c);
uint32_t q4_expert_warm_cap(uint32_t nslots, uint32_t n_layer,
                            uint32_t topk);
int q4_expert_warm_load(q4_expert_cache *c, const q4_gguf *g,
                        const char *path, uint32_t cap, FILE *fp);
/* Routing hit histogram: note the full top-k for (layer), save the
 * nonzero entries in warm-file format for res_warm_parse. */
void q4_expert_route_note(q4_expert_cache *c, int32_t layer,
                          const int32_t *ids, int n);
int q4_expert_route_save(const q4_expert_cache *c, const q4_gguf *g,
                         const char *path, FILE *fp);
uint32_t q4_expert_pin_occupied(q4_expert_cache *c);
uint32_t q4_expert_pin_occupied_cap(q4_expert_cache *c, uint32_t max_pinned);
/* GPU arena this cache uploads into (0 main / 1 MTP head). */
int q4_expert_cache_arena(const q4_expert_cache *c);
/* Pin the listed (already resident) experts. Returns newly pinned count. */
uint32_t q4_expert_pin_list(q4_expert_cache *c, int32_t layer,
                            const int32_t *ids, uint32_t n);
/* Dump pinned experts into parallel arrays (may be NULL). Returns count. */
uint32_t q4_expert_pinned_list(const q4_expert_cache *c, int32_t *layers,
                               int32_t *eids, uint32_t *hits, uint32_t max);
uint32_t q4_expert_pin_hottest(q4_expert_cache *c, uint32_t max_pinned);
/* Pin occupied experts whose hits are within hottest/div of the top, up to
 * max_pinned. div 0 means 8. One-hit prefill noise stays evictable. */
uint32_t q4_expert_pin_stable(q4_expert_cache *c, uint32_t max_pinned,
                              uint32_t hottest_div);
/* Additive promote-only variant of pin_stable for in-decode re-pinning. */
uint32_t q4_expert_pin_thresh(q4_expert_cache *c, uint32_t max_pinned,
                              uint32_t hottest_div);
int q4_expert_is_resident(const q4_expert_cache *c);
void q4_expert_clear_sticky(q4_expert_cache *c);
void q4_expert_sticky(q4_expert_cache *c, int32_t layer, const int32_t *ids,
                      uint32_t n);
uint32_t q4_expert_pinned_count(const q4_expert_cache *c);
/* Packed expert: gate, then up, then down. NULL if not resident. */
const uint8_t *q4_expert_cache_data(q4_expert_cache *c, int32_t layer, int32_t eid);
/* Packed expert bytes on the host without binding/evicting a slot:
 * resident image, then L1 arena, then L2 slot. NULL -> read rows via
 * the GGUF fd (cpuexp does ranged preads itself). */
const uint8_t *q4_expert_host_ptr(const q4_expert_cache *c, int32_t layer,
                                  int32_t eid);
int64_t q4_expert_pread(q4_expert_cache *c, int32_t layer, int32_t eid,
                        uint8_t *dst);
bool q4_expert_parts(const q4_gguf *g, int32_t layer,
                     const q4_tensor **gate, const q4_tensor **up,
                     const q4_tensor **down, uint64_t *off_gate,
                     uint64_t *off_up, uint64_t *off_down, uint64_t *total);
int32_t q4_expert_cache_slot(const q4_expert_cache *c, int32_t layer, int32_t eid);
uint64_t q4_expert_cache_slot_bytes(const q4_expert_cache *c);
/* Hybrid CPU+GPU decode: async-H2D upload watermark + hit detection */
void    q4_expert_uploads_mark(q4_expert_cache *c);
int32_t q4_expert_gpu_slot(const q4_expert_cache *c, int32_t layer,
                           int32_t eid);
void    q4_expert_protect(q4_expert_cache *c, int32_t layer,
                          const int32_t *ids, uint32_t n, int on);
uint32_t q4_expert_cache_n_slots(const q4_expert_cache *c);
uint64_t q4_expert_packed_bytes(const q4_gguf *g);
void q4_expert_cache_cap_slots(q4_expert_cache *c, uint32_t n);
/* Device residency table (graph-resident routing): enable allocates the
 * n_layer*n_expert i32 table and publishes current bindings; row() gives
 * kernels the per-layer slice; tick() runs at a decode token boundary with
 * the compute stream drained — it makes deferred evictions refillable. */
int     q4_expert_dev_enable(q4_expert_cache *c);
int32_t *q4_expert_dev_row(q4_expert_cache *c, int32_t layer);
void    q4_expert_dev_tick(q4_expert_cache *c);

/* CPU-computed routed experts (decode path). x is n_embd floats.
 * Multi-slot ring: dispatch2 publishes into a free slot and returns its
 * index (-1 transient backpressure, -2 error); slot_done polls without
 * blocking; join2 waits and retires the slot.  Intermediates are per-slot,
 * so jobs from different layers overlap. */
int  q4_cpuex_dispatch2(const q4_gguf *g, q4_expert_cache *c, int32_t layer,
                        const float *x, const int32_t *ids, const float *wts,
                        uint32_t k, float *y, uint64_t *tag_out);
bool q4_cpuex_slot_done(int slot);
bool q4_cpuex_join2(int slot);
/* Tag-aware variants: safe across slot retire+republish (done_a is
 * monotone so >= works where ==-against-a-reread-tag can overshoot and
 * wedge). dispatch2 fills *tag_out; wait/poll on that tag. */
bool q4_cpuex_tag_done(int slot, uint64_t tag);
bool q4_cpuex_join_tag(int slot, uint64_t tag);
/* Retire completed slots; returns count still running. */
int  q4_cpuex_retire(void);
void q4_cpuex_diag(void);
/* Reaps a job whose completion was already consumed by the GPU wait kernel
 * (deferred-join path); instant when nothing is pending. */
bool q4_cpuex_reap(void);
/* Pinned seq words the workers publish into for the GPU wait kernel. */
void q4_cpuex_syncpin(unsigned long long *expect,
                      unsigned long long *done);
/* int8-activation dot kernels vs the float path; 0 = ok (no model needed). */
int  q4_cpx_selftest(void);
/* Row-dot microbenchmark per format; run via `q4-test --cpxbench`. */
int  q4_cpx_bench(void);
/* SIGABRT/SEGV/BUS/ILL/FPE -> stderr backtrace + real core (call early in
 * main).  Raises RLIMIT_CORE; weight arenas are MADV_DONTDUMP. */
void q4_install_crashdump(void);

typedef struct q4_ple q4_ple;

typedef enum {
    Q4_PLE_IO_PREAD = 0, /* SSD row reads; do not map 27 GiB */
    Q4_PLE_IO_MMAP  = 1, /* demand-paged mmap; DRAM is the kernel's cache */
} q4_ple_io;

q4_ple *q4_ple_open(const q4_gguf *g); /* PREAD */
q4_ple *q4_ple_open_io(const q4_gguf *g, q4_ple_io io);
void    q4_ple_close(q4_ple *p);
/* Hash n-grams of `n` token ids. hist is ngram-1 predecessors (EOS-padded).
 * writes n * ple_n_heads row indices. */
bool q4_ple_hash(const q4_ple *p, const int32_t *toks, int64_t n,
                 const int32_t *hist, int32_t *rows_out);
/* Dequant IQ4_NL rows into f32 [n_rows * n_embd_ple]. */
bool q4_ple_gather(const q4_ple *p, const int32_t *rows, int64_t n_rows,
                   float *out);

bool q4_parse_gib(const char *s, uint64_t *bytes);
bool q4_layer_is_qsa(const q4_gguf *g, int32_t layer);
const q4_tensor *q4_find_tensor(const q4_gguf *g, const char *name);
bool q4_tensor_read(const q4_gguf *g, const q4_tensor *t, void *dst, uint64_t n);
const char *q4_type_name(uint32_t ggml_type);
uint64_t q4_row_bytes(uint32_t ggml_type, uint64_t ncols);
float q4_f16_to_f32(uint16_t h);
float q4_bf16_to_f32(uint16_t h);

bool q4_dequant_row(uint32_t ggml_type, const uint8_t *src, uint64_t ncols,
                    float *dst);
/* y[nrows] += scale * W[nrows, ncols] @ x[ncols]  (W row-major, ggml ne[0]=ncols) */
bool q4_gemv(uint32_t ggml_type, const uint8_t *w, uint64_t nrows, uint64_t ncols,
             const float *x, float *y, float scale);
void q4_silu(float *x, uint64_t n);
void q4_softmax(float *x, uint64_t n);
/* Top-k of 512-or-less router logits. ids/wts length k. */
void q4_topk(const float *logits, uint32_t n, uint32_t k, int32_t *ids, float *wts,
             bool renormalize);

void q4_sigmoid(float *x, uint64_t n);
void q4_rms_norm(const float *x, const float *w, float *y, uint64_t n, float eps);
void q4_l2_norm(float *x, uint64_t n, float eps);
float q4_softplus(float x);

typedef struct q4_store q4_store;
q4_store *q4_store_open(const q4_gguf *g); /* host: dense + shared, not PLE table / routed */
void q4_store_close(q4_store *s);
const uint8_t *q4_store_get(const q4_store *s, const q4_tensor *t);
const uint8_t *q4_store_get_name(const q4_store *s, const char *name);
bool q4_store_upload_hip(q4_store *s);
const uint8_t *q4_store_dev(const q4_store *s, const q4_tensor *t);
/* Drop host copies after VRAM upload (frees ~4.6 GiB for expert L2).
 * Keeps tensors the GPU serve path still reads on host: token_embd + PLE conv. */
void q4_store_free_host_except(q4_store *s, const q4_gguf *g);

/* One MoE layer: shared (gated SiLU) + top-k routed. x,y are n_embd. */
bool q4_moe_layer(const q4_gguf *g, q4_expert_cache *c, int32_t layer,
                  const float *x, float *y);
bool q4_moe_layer_store(const q4_gguf *g, const q4_store *st, q4_expert_cache *c,
                        int32_t layer, const float *x, float *y);
/* Device activations. d_x/d_y/d_ws are GPU pointers. */
/* Split MoE for graph-captured decode: A = GPU pre-router+shared+D2H,
 * hostpart = sync+topk+CPU dispatch+join, B = GPU merge tail. */
int q4_moe_cpu_active(const q4_gguf *g);
bool q4_moe_gpu_a(const q4_gguf *g, const q4_store *st, int32_t layer,
                  float *d_x, float *d_y, float *d_ws);
int q4_moe_hostpart(const q4_gguf *g, q4_expert_cache *c, int32_t layer);
bool q4_moe_gpu_b(const q4_gguf *g, q4_expert_cache *c, int32_t layer);
bool q4_moe_gpu_b_hc(const q4_gguf *g, q4_expert_cache *c, int32_t layer,
                     float *d_res, const float *d_inj,
                     uint32_t hc);
/* Whole-layer graphable MoE (device routing): router gemv + device top-k +
 * hit-exec + in-graph wait for CPU misses + merged tail + hc combine.
 * q4_moe_dev_route() must have succeeded first (devtab + mailbox live). */
bool q4_moe_gpu_dev(const q4_gguf *g, const q4_store *st, q4_expert_cache *c,
                    int32_t layer, float *d_res, const float *d_inj,
                    uint32_t hc, float *d_x, float *d_y, float *d_ws);
int  q4_expert_stage_batch_try(q4_expert_cache *c, int32_t layer,
                               const int32_t *ids, uint32_t n, int gpu_upload);
int  q4_moe_dev_route(const q4_gguf *g, q4_expert_cache *c);
int  q4_moe_dev_active(void);
int  q4_moe_dev_tick(q4_expert_cache *c);
/* Warm lazily-converted weights (non-F32 sgate) BEFORE a gA graph capture:
 * the conversion does malloc + hipMalloc + a sync H2D copy, which would
 * invalidate an in-flight capture. */
void q4_moe_dev_warm(const q4_gguf *g, const q4_store *st, int32_t layer);
void q4_moe_dev_stop(void);
void q4_moe_dev_stats(unsigned long long *jobs, unsigned long long *miss,
                      unsigned long long *lat_us);
double q4_moe_route_prof(void);
/* "dispatch refused" tail: GPU routed-expert chain + shared merge using the
 * ids hostpart already computed. */
bool q4_moe_gpu_tail(const q4_gguf *g, q4_expert_cache *c, int32_t layer,
                     float *d_x, float *d_y, float *d_ws);
bool q4_moe_layer_dev(const q4_gguf *g, const q4_store *st, q4_expert_cache *c,
                      int32_t layer, float *d_x, float *d_y, float *d_ws);
/* Batched MoE for n_tok tokens. d_ws is ws_floats floats of scratch; tokens
 * are processed in <=1024-row groups to bound the workspace. */
bool q4_moe_layer_dev_n(const q4_gguf *g, const q4_store *st, q4_expert_cache *c,
                        int32_t layer, float *d_x, float *d_y, float *d_ws,
                        uint32_t n_tok, uint64_t ws_floats);

typedef struct q4_sess q4_sess;
q4_sess *q4_sess_open(const q4_gguf *g, q4_store *st, q4_expert_cache *c,
                      q4_ple *ple, uint32_t max_kv);
void q4_sess_close(q4_sess *s);
/* One decode token: embed + 48 layers + lm-head argmax. */
bool q4_sess_decode(q4_sess *s, int32_t token, int32_t *out_id, float *out_logit);
bool q4_sess_decode_ex(q4_sess *s, int32_t token, int32_t *out_id, float *out_logit,
                       bool want_logits);
/* Layer-first prefill of `n` tokens. Leaves residual at the last token. */
bool q4_sess_prefill(q4_sess *s, const int32_t *toks, int n);
/* Optional per-layer prefill progress hook (for keep-alive pings). */
void q4_sess_set_pref_hook(q4_sess *s,
                           void (*cb)(uint32_t layer, uint32_t n_layer, void *u),
                           void *u);
bool q4_sess_head(q4_sess *s, int32_t *out_id, float *out_logit);
void q4_sess_reset(q4_sess *s);
uint32_t q4_sess_n_kv(const q4_sess *s);
uint32_t q4_sess_max_kv(const q4_sess *s);
const float *q4_sess_logits(const q4_sess *s);
/* ---- MTP (multi-token prediction) verify support ----
 * The draft head needs per-token residuals of a just-prefilled chunk and a
 * way to roll the sequential state back to an accepted prefix. */
/* Device row i (hc_dim floats) of the last prefill_hip batch, or NULL. */
const float *q4_sess_pref_row(const q4_sess *s, int i);
/* Batched forward of n <= 8 tokens WITHOUT the readahead thread. Used for
 * MTP verify passes; also stashes per-token GDN inputs and PLE state so a
 * rejected prefix can be replayed from the pre-batch snapshot. */
bool q4_sess_prefill_tokens(q4_sess *s, const int32_t *toks, int n);
/* Allocate the verify shadow/stash buffers (call once after sess open). */
bool q4_sess_mtp_prep(q4_sess *s, int n_max);
/* Target logits at prefill row `row` (runs the lm-head mixer on that row;
 * afterwards q4_sess_logits() returns them). */
bool q4_sess_logits_at(q4_sess *s, int row);
/* Keep the state of the first n_acc tokens of the last verify batch and
 * discard the rest (rollback + replay). n_acc == batch size is a no-op. */
bool q4_sess_mtp_commit(q4_sess *s, int n_acc);
/* Device residual (hc_dim) of the last processed token. */
const float *q4_sess_res_dev(const q4_sess *s);
/* Session-state snapshot for chat prefix caching. Save writes
 * ids + full recurrent state (atomically via rename). Restore requires
 * the saved ids to be an exact prefix of the new prompt; returns the
 * matched prefix length (state loaded) or 0.
 * prompt_n is the token count before generation (so a later turn can
 * splice the raw generated ids instead of re-encoding the assistant). */
bool q4_sess_save(const q4_sess *s, const char *path, const int32_t *ids,
                  int n_ids, int prompt_n, int nmsg, uint64_t user_hash);
int q4_sess_restore(q4_sess *s, const char *path, const int32_t *ids,
                    int n_ids);
/* Header + token ids only (no KV/GDN). Returns n_ids, or -1. */
int q4_sess_peek_ids(const q4_sess *s, const char *path, int32_t *ids,
                     int max_ids, int *prompt_n, int *nmsg, uint64_t *user_hash);

typedef struct q4_tok q4_tok;
q4_tok *q4_tok_open(const q4_gguf *g);
void q4_tok_close(q4_tok *t);
int32_t q4_tok_id(const q4_tok *t, const char *s);
int q4_tok_encode(const q4_tok *t, const char *utf8, int32_t *out, int max_out);
int q4_tok_decode(const q4_tok *t, const int32_t *ids, int n, char *out, int max_out);
/* Largest index <= n that does not split a UTF-8 character. */
int q4_utf8_safe_end(const char *s, int n);
int q4_tok_apply_chat(const q4_tok *t, const char *system, const char *user,
                      int32_t *out, int max_out);
int q4_tok_apply_messages(const q4_tok *t, const char *const *roles,
                          const char *const *contents, int nmsg, int32_t *out,
                          int max_out);
/* Append messages[from..] onto an existing token prefix (length n). */
int q4_tok_apply_messages_from(const q4_tok *t, const char *const *roles,
                               const char *const *contents, int nmsg, int from,
                               int32_t *out, int max_out, int n);
int32_t q4_tok_eos(const q4_tok *t);
int32_t q4_tok_im_end(const q4_tok *t);
/* Q4_THINK=1 (serve.sh default): assistant-open leaves <think> unclosed. */
int q4_tok_think_enabled(void);
/* Qwen3.8 / qwen3_coder tool format. prefix is prose before the first
 * <tool_call>; names/args_json are malloc'd arrays of length ncalls. */
int q4_qwen_parse_tool_calls(const char *text, char **prefix, char ***names,
                             char ***args_json, int *ncalls);
void q4_qwen_free_tool_calls(char **names, char **args_json, int n);
/* tools_array is the raw JSON array from the request ("[{...},...]").
 * Descriptions and other schema prose are stripped so OpenCode+MCP dumps
 * do not 2–5× the system prompt. */
char *q4_qwen_compact_tools_json(const char *tools_array);
char *q4_qwen_tools_preamble(const char *tools_array, int with_think);
int32_t q4_sample(const float *logits, uint32_t n, float temp, int top_k,
                  float top_p, unsigned *rng);
/* Transformed sampling distribution used by q4_sample (top-k -> temp ->
 * softmax -> top-p), exposed for speculative decoding. */
#define Q4_DIST_MAX 128
int q4_dist_build(const float *logits, uint32_t n, float temp, int top_k,
                  float top_p, int32_t *ids, float *probs);
int32_t q4_dist_sample(const int32_t *ids, const float *probs, int k, unsigned *rng);
float q4_dist_prob(const int32_t *ids, const float *probs, int k, int32_t id);
/* One speculative-sampling accept test against the target distribution.
 * Returns 1 when `draft` is accepted; on 0, *resample holds a draw from
 * norm(max(0, p_target - p_draft)) so the emitted token stays exact. */
int q4_spec_accept(const int32_t *tids, const float *tp, int tn,
                   const int32_t *dids, const float *dp, int dn,
                   int32_t draft, unsigned *rng, int32_t *resample);

/* ---- MTP draft head (Unsloth mtp-*-shared GGUF for qwen4exp) ---- */
typedef struct q4_mtp q4_mtp;
q4_mtp *q4_mtp_open(const char *head_path, const q4_gguf *gmain, q4_store *stmain,
                    uint32_t head_kv, uint32_t n_l1, uint64_t l2_bytes);
void q4_mtp_close(q4_mtp *m);
/* VRAM the head needs beyond its expert L1 slots (dense + KV + scratch). */
uint64_t q4_mtp_fixed_vram(const q4_mtp *m);
uint64_t q4_mtp_slot_bytes(const q4_mtp *m);
q4_expert_cache *q4_mtp_cache(q4_mtp *m);
/* Run the head over a just-prefilled chunk (builds the head KV with the
 * real per-token residuals; must follow q4_sess_prefill of the same toks). */
bool q4_mtp_prefill_chunk(q4_mtp *m, q4_sess *s, const int32_t *toks, int n);
/* Chain seed for a fully-cached prompt (no prefill ran this turn). */
bool q4_mtp_seed(q4_mtp *m, q4_sess *s, int32_t tok);
/* Drop carried-over head KV when the session resets (positions restart). */
void q4_mtp_reset(q4_mtp *m);
/* One draft+verify cycle. c0 sits at position n_kv (already emitted). Drafts
 * up to n_draft tokens, verifies [c0, drafts...] in one batched pass, keeps
 * the accepted prefix and re-feeds the head with the real residuals.
 * Outputs: accepted draft ids (out_drafts, *n_out of them), the next token
 * (sampled exactly as q4_sample would) and the target logits it came from. */
int q4_mtp_cycle(q4_mtp *m, q4_sess *s, int32_t c0, int n_draft, float temp,
                 int top_k, float top_p, unsigned *rng, int32_t *out_drafts,
                 int *n_out, int32_t *next_c0, const float **next_logits);
/* Draft statistics: proposals, accepted, cycles, confidence-gated. */
void q4_mtp_stats(const q4_mtp *m, uint64_t *proposed, uint64_t *accepted,
                  uint64_t *cycles, uint64_t *gated);
/* Q4_PROFILE decode-cycle phase totals; prints then zeroes the counters. */
void q4_mtp_prof_print(q4_mtp *m, FILE *f);
/* Q4_PROFILE small-batch MoE (verify-path) totals; prints then zeroes. */
void q4_moe_sb_prof_print(FILE *f);

typedef struct q4_engine q4_engine;
q4_engine *q4_engine_open(const char *model, uint32_t ctx, uint64_t l2_bytes);
/* Same, with an optional MTP draft-head GGUF (Unsloth mtp-*-shared). */
q4_engine *q4_engine_open_full(const char *model, uint32_t ctx, uint64_t l2_bytes,
                               const char *mtp_path);
void q4_engine_close(q4_engine *e);
int q4_engine_pin_ja(q4_engine *e, FILE *fp);
/* Persist/restore the pinned expert set (guard-checked against the model). */
int q4_engine_warm_save(q4_engine *e, const char *path);
int q4_engine_warm_load(q4_engine *e, const char *path, FILE *fp);
/* Dump the routing hit histogram in warm-file format. */
int q4_engine_route_save(q4_engine *e, const char *path);
const q4_gguf *q4_engine_gguf(const q4_engine *e);
/* Prefill ids in 512-token chunks, then greedy-decode n_dec tokens.
 * Pins already on the expert cache are left in place. 0 on success. */
int q4_engine_bench(q4_engine *e, const int32_t *ids, int n, int n_dec, FILE *fp);
/* Teacher-forced NLL / perplexity / top-1 over ids (8-token windows). */
int q4_engine_ppl(q4_engine *e, const int32_t *ids, int n, FILE *fp,
                  FILE *dump);
void q4_engine_set_sample(q4_engine *e, float temp, int top_k, float top_p);
void q4_engine_set_progress(q4_engine *e,
                            void (*cb)(int prefill_i, int prefill_n, int gen_i,
                                       void *u),
                            void *u);
/* flag is polled during prefill and decode. Non-zero stops the turn.
 * The server sets it when the client socket closes. */
void q4_engine_set_cancel(q4_engine *e, volatile int *flag);
uint32_t q4_engine_ctx(const q4_engine *e);
const char *q4_engine_name(const q4_engine *e);
int q4_engine_last_prompt_tokens(const q4_engine *e);
int q4_engine_last_completion_tokens(const q4_engine *e);
const char *q4_engine_last_reasoning(const q4_engine *e);
int q4_engine_last_hit_limit(const q4_engine *e);
/* stream_cb kind: Q4_PIECE_CONTENT or Q4_PIECE_REASONING. out is content only. */
#define Q4_PIECE_CONTENT 0
#define Q4_PIECE_REASONING 1
int q4_engine_generate(q4_engine *e, const char *system, const char *user,
                       char *out, int max_out, int max_new,
                       void (*stream_cb)(const char *piece, int kind, void *u),
                       void *u);
int q4_engine_generate_msgs(q4_engine *e, const char *const *roles,
                            const char *const *contents, int nmsg, char *out,
                            int max_out, int max_new,
                            void (*stream_cb)(const char *piece, int kind, void *u),
                            void *u);
int q4_serve(q4_engine *e, const char *host, int port);

bool q4_hip_qsa_decode(const float *d_q, const void *d_kcache, const void *d_vcache,
                       const float *d_gate, float *d_out, uint32_t n_head,
                       uint32_t n_kvh, uint32_t hd, uint32_t n_kv, uint32_t max_kv,
                       float scale, int kv_f16);

/* Expert GPU arenas: 0 = main model, 1 = MTP draft head. The _a variants
 * address a specific arena; the unmarked names are arena 0. */
bool q4_hip_experts_alloc_a(int arena, uint32_t n_slots, uint64_t slot_bytes);
uint32_t q4_hip_experts_n_a(int arena);
bool q4_hip_experts_put_a(int arena, uint32_t slot, const uint8_t *host, uint64_t n);
bool q4_hip_experts_put_async_a(int arena, uint32_t slot, const uint8_t *host,
                                uint64_t n);
const uint8_t *q4_hip_experts_dev_a(int arena, uint32_t slot);
const uint8_t *q4_hip_experts_base_a(int arena);
uint64_t q4_hip_experts_stride_a(int arena);
/* Batched routed-expert eval: gate/up/(silu*mul)/down/wsum for n experts in
 * 5 launches. slots are already-resident L1 slot indices. */
/* Device argmax + softmax-max probability over d_x[0..n): *h_id = first
 * index of the max (host argmax parity), *h_p = p of that index under
 * softmax(x).  Copies only 8 bytes. */
bool q4_hip_argmax_prob(const float *d_x, uint32_t n, int32_t *h_id,
                        float *h_p);
bool q4_hip_moe_tok_ok(uint32_t gt, uint32_t ut, uint32_t dt,
                        uint32_t in_dim, uint32_t n_ff);
bool q4_hip_moe_exec_tok(const uint8_t *base, uint64_t stride,
                         const int32_t *d_slots, const float *d_wts,
                         const uint32_t *h_off, uint32_t n_tok, uint32_t gt,
                         uint64_t og, uint32_t ut, uint64_t ou, uint32_t dt,
                         uint64_t od, const float *d_x, float *d_h,
                         float *d_h2, float *d_y, uint32_t n_ff,
                         uint32_t in_dim, uint32_t n_embd);
bool q4_hip_moe_exec(const uint8_t *base, uint64_t stride,
                     const int32_t *slots, const float *wts, uint32_t n,
                     uint32_t gt, uint64_t og, uint32_t ut, uint64_t ou,
                     uint32_t dt, uint64_t od, const float *d_x, float *d_h,
                     float *d_h2, float *d_down, float *d_y, uint32_t n_ff,
                     uint32_t in_dim, uint32_t n_embd);
/* Multi-segment GEMV: <=4 weight tensors sharing one x in one launch. */
typedef struct {
    const uint8_t *w;
    float *out;
    uint32_t rows;
    uint32_t ty;        /* Q4_T_Q8_0 or Q4_T_F32 */
} q4_seg_t;
bool q4_hip_gemv_m(const q4_seg_t *segs, uint32_t nseg, const float *d_x,
                   uint32_t in_dim);
/* Elementwise fusions replacing two launches each. */
bool q4_hip_mul_sig(float *y, const float *a, const float *g, uint64_t n);
bool q4_hip_scale_silu(float *x, float s, uint64_t n);
bool q4_hip_silu_mul(float *g, const float *u, uint64_t n);
bool q4_hip_dot_sigmoid(const float *a, const float *b, float *out,
                        uint32_t n);
bool q4_hip_gdn_convshift(float *hist, const float *qkv, const float *w,
                          float *out, uint32_t ch, uint32_t ksz);
/* Whole GDN tail in one launch (conv+shift, q/k l2, step, rms*z gate).
 * Requires d_k == d_v == 128. */
bool q4_hip_gdn_tail(float *hist, const float *qkv, const float *cw,
                     uint32_t ch, uint32_t ksz, float *S, const float *beta,
                     const float *alpha, const float *A, const float *dt,
                     const float *nw, const float *z, float *core,
                     uint32_t n_vh, uint32_t n_kh, uint32_t d, float eps,
                     float qscale);

/* Device-position QSA variants used inside graph capture: pos is read from a
 * device counter pushed once per decode token via q4_hip_pos_push. */
void q4_hip_pos_push(int32_t pos);
int q4_hip_capturing(void);
int q4_hip_pos_dev_ok(void);
bool q4_hip_qsa_q_prep_d(const float *qg, const float *wn, float *q, float *g,
                         uint32_t n_head, uint32_t hd, uint32_t n_rot,
                         float base, float eps);
bool q4_hip_qsa_k_prep_d(float *k, const float *wn, uint32_t n_kvh,
                         uint32_t hd, uint32_t n_rot, float base, float eps);
bool q4_hip_kv_append_d(void *kcache, void *vcache, const float *k,
                        const float *v, uint32_t n, uint32_t bpt, int kind);
bool q4_hip_f32_to_f16_d(void *dst_base, uint32_t stride_elems,
                         const float *src, uint64_t n);
bool q4_hip_qsa_pool_d(const void *raw, float *pool, const float *wn,
                       uint32_t hd, uint32_t ratio, uint32_t n_rot,
                       float rope_base, float eps);
bool q4_hip_qsa_score_d(const float *q, const float *pool, float *score,
                        uint32_t stride, uint32_t n_head, uint32_t hd,
                        uint32_t ratio);
bool q4_hip_qsa_topk_alloc(void);
bool q4_hip_qsa_topk_d(const float *score, int32_t *sel, uint32_t stride,
                       uint32_t k_blk, uint32_t ratio, uint32_t width);

bool q4_hip_init(void);
void q4_hip_shutdown(void);
bool q4_hip_ok(void);
bool q4_hip_host_register(void *p, size_t n);
void *q4_hip_host_devptr(void *p);   /* zero-copy device alias, NULL if n/a */
void q4_hip_host_unregister(void *p);
/* Device GEMV from host W (copies W). Overwrites y. */
bool q4_hip_gemv(uint32_t ggml_type, const uint8_t *w, uint64_t nrows,
                 uint64_t ncols, const float *x, float *y, float scale);
bool q4_hip_gemv_dev(uint32_t ggml_type, const uint8_t *d_w, uint64_t nrows,
                     uint64_t ncols, const float *x, float *y, float scale);
/* All device pointers. No host copy. */
bool q4_hip_gemv_dd(uint32_t ggml_type, const uint8_t *d_w, uint64_t nrows,
                    uint64_t ncols, const float *d_x, float *d_y, float scale);
bool q4_hip_gemv_dd_n(uint32_t ggml_type, const uint8_t *d_w, uint64_t nrows,
                      uint64_t ncols, const float *d_x, float *d_y, float scale,
                      uint32_t n_batch);
/* Batched GEMV with row gather: token i reads x[which[i]]. which may be NULL. */
bool q4_hip_gemv_dd_n_which(uint32_t ggml_type, const uint8_t *d_w,
                            uint64_t nrows, uint64_t ncols, const float *d_x,
                            float *d_y, float scale, uint32_t n_batch,
                            const int32_t *which);
bool q4_hip_gateup_q4k_n_which(const uint8_t *d_gate, const uint8_t *d_up,
                               const float *d_x, float *d_h, uint32_t n_ff,
                               uint32_t n_embd, uint32_t n_tok,
                               const int32_t *which);
/* dy[tok] += sum_k rw * pd[rsrc]  (deterministic expert output reduce). */
bool q4_hip_expert_reduce(float *d_y, const float *d_pd, const int32_t *d_rsrc,
                          const float *d_rw, uint32_t n_tok, uint32_t topk,
                          uint32_t n_embd);
/* Batched prefill expert GEMM: one launch per tensor over a whole expert
 * group.  grid=(nrows, ne); entry e applies tokens atok[eb[e]..+ec[e]) (or
 * assignment-indexed rows of x when d_atok==NULL) to the expert pack at
 * wbase + d_slots[e]*stride + woff.  d_slots/d_eb/d_ec/d_atok are device
 * pointers; out rows are assignment-indexed. */
bool q4_hip_moe_pf(uint32_t type, const uint8_t *wbase, uint64_t stride,
                   uint64_t woff, const int32_t *d_slots, const uint32_t *d_eb,
                   const uint32_t *d_ec, const int32_t *d_atok,
                   const float *d_x, float *d_out, uint32_t nrows,
                   uint32_t ncols, uint32_t ne);
/* Fused two-tensor variant (gate+up share the token list). */
bool q4_hip_moe_pf2(uint32_t type, const uint8_t *wbase, uint64_t stride,
                    uint64_t woff0, uint64_t woff1, const int32_t *d_slots,
                    const uint32_t *d_eb, const uint32_t *d_ec,
                    const int32_t *d_atok, const float *d_x, float *d_out0,
                    float *d_out1, uint32_t nrows, uint32_t ncols,
                    uint32_t ne);
/* silu(g)*u fused, in place on g. */
bool q4_hip_moe_act(float *d_g, const float *d_u, uint64_t n);
/* WMMA variants of the batched prefill expert GEMMs (f16 LDS operands).
 * moe_wmma2 fuses gate+up and the silu(g)*u activation, writing d_h rows.
 * Both return false for unsupported types or Q4_WMMA=0 (callers fall back
 * to q4_hip_moe_pf/pf2 + moe_act). */
bool q4_hip_moe_wmma2(uint32_t type, const uint8_t *wbase, uint64_t stride,
                      uint64_t woff0, uint64_t woff1, const int32_t *d_slots,
                      const uint32_t *d_eb, const uint32_t *d_ec,
                      const int32_t *d_atok, const float *d_x, float *d_out,
                      uint32_t nrows, uint32_t ncols, uint32_t ne);
bool q4_hip_moe_wmma(uint32_t type, const uint8_t *wbase, uint64_t stride,
                     uint64_t woff, const int32_t *d_slots, const uint32_t *d_eb,
                     const uint32_t *d_ec, const int32_t *d_atok,
                     const float *d_x, float *d_out, uint32_t nrows,
                     uint32_t ncols, uint32_t ne);
/* dy[t] += sg[t] * dx[t] row-wise. */
bool q4_hip_axpy_rows(float *d_y, const float *d_x, const float *d_sg, uint32_t n,
                      uint32_t n_tok);
bool q4_hip_silu(float *d, uint64_t n);
bool q4_hip_sigmoid(float *d, uint64_t n);
bool q4_hip_scale(float *d, float s, uint64_t n);
bool q4_hip_mul(float *y, const float *a, const float *b, uint64_t n);
bool q4_hip_axpy(float *y, const float *x, float a, uint64_t n);
bool q4_hip_axpy_dalpha(float *y, const float *x, const float *d_alpha,
                        uint64_t n);
bool q4_hip_fill(float *d, float v, uint64_t n);
int q4_hip_graph_begin(void);
void *q4_hip_graph_end(void);
uint32_t q4_hip_graph_nodes(void *exec);
bool q4_hip_graph_launch(void *exec);
void q4_hip_graph_free(void *exec);
bool q4_hip_d2h_async(void *h, const void *d, uint64_t n);
bool q4_hip_h2d_async(void *d, const void *h, uint64_t n);
bool q4_hip_stream_sync(void);
/* Pinned-flag helpers: flag_bump is a stream-ordered "host data ready"
 * marker; cpx_wait idles the stream until the CPU expert pool reaches the
 * published seq.  Both replay-safe inside captured graphs. */
bool q4_hip_flag_bump(void *flag_d);
/* progress_d: a monotone pool-wide counter (cpuex done floor); the in-kernel
 * timeout only fires after LIMIT ticks with NO floor advance — slow waits
 * behind queued jobs stay alive, dead pipelines fail fast. */
bool q4_hip_cpx_wait(void *done_d, void *expect_d, void *progress_d);
double q4_hip_cpx_wait_prof(unsigned long long *calls);
unsigned long long q4_hip_cpx_wait_timeouts(void);
unsigned long long q4_hip_cpx_wait_lastseen(void);
/* Device-side routing: softmax+top-k + L1-slot split on the GPU; publishes
 * CPU-miss ids/wts through a pinned mailbox consumed by a host dispatcher
 * thread.  exec_dev runs the hit set reading slots/wts/count straight from
 * device memory — both are graphable. */
bool q4_hip_route_dev(const float *d_rlog, const int32_t *devtab,
                      int32_t *d_es, float *d_ew, int32_t *d_nhit,
                      void *mail, unsigned long long *d_seq,
                      unsigned long long *d_rseq,
                      unsigned long long *d_prof,
                      uint32_t n_exp, uint32_t topk);
bool q4_hip_moe_exec_dev(const uint8_t *base, uint64_t stride,
                         const int32_t *d_es, const float *d_ew,
                         const int32_t *d_nhit, uint32_t gt, uint64_t og,
                         uint32_t ut, uint64_t ou, uint32_t dt, uint64_t od,
                         const float *d_x, float *d_h, float *d_h2,
                         float *d_down, float *d_y, uint32_t n_ff,
                         uint32_t in_dim, uint32_t n_embd);
/* Hybrid-path staging: host writes slots/wts/count into mapped memory so the
 * hit-exec kernels can live inside the gB graph (they read the words at exec
 * time).  stage is the per-token host call; exec_staged is what gets captured
 * or launched eagerly as the fallback. */
bool q4_hip_moe_stage(const int32_t *slots, const float *wts, uint32_t n);
bool q4_hip_moe_exec_staged(const uint8_t *base, uint64_t stride,
                            uint32_t gt, uint64_t og, uint32_t ut, uint64_t ou,
                            uint32_t dt, uint64_t od, const float *d_x,
                            float *d_h, float *d_h2, float *d_down, float *d_y,
                            uint32_t n_ff, uint32_t in_dim, uint32_t n_embd);
/* Ordered writes/copies on the expert copy stream (device residency table —
 * stay ordered with the payload uploads riding the same stream). */
bool q4_hip_copy_i32(int32_t *dst, int32_t v);
bool q4_hip_copy_memset(void *dst, int v, size_t n);
bool q4_hip_copy_h2d(void *dst, const void *src, size_t n);
bool q4_hip_moe_pack(float *d_y, const float *d_x, const float *d_sgate,
                     float *d_dot, void *flag_d, uint32_t n_embd);
bool q4_hip_moe_final(float *res, const float *y, const float *hy,
                      const float *sh, const float *dot, const float *inj,
                      uint32_t n_embd, uint32_t hc);
bool q4_hip_mulsig_mean(const float *a, const float *g, float *mixed,
                        uint32_t n_embd, uint32_t hc);
bool q4_hip_gemv_dd_act(uint32_t ggml_type, const uint8_t *d_w,
                        uint64_t nrows, uint64_t ncols, const float *d_x,
                        float *d_y, float scale, float act);
void q4_hip_gemv_prof_reset(void);
void q4_hip_gemv_prof_report(void);
void q4_hip_mark_begin(void);   /* Q4_STEP_PROF: GPU-window event markers */
double q4_hip_mark_ms(void);
void q4_hip_mark_begin_slot(int slot);
double q4_hip_mark_ms_slot(int slot);
void q4_hip_mark_pt(int slot);      /* capture-safe boundary event (Q4_STEP_PROF) */
double q4_hip_mark_elapsed(int sa, int sb);
bool q4_hip_copy(float *d, const float *s, uint64_t n);
bool q4_hip_grouped_rms(const float *x, const float *w, float *y, uint32_t n_embd,
                        uint32_t hc, float eps);
bool q4_hip_grouped_rms_n(const float *x, const float *w, float *y, uint32_t n_embd,
                          uint32_t hc, float eps, uint32_t n_tok);
bool q4_hip_mean_hc(const float *gated, float *mixed, uint32_t n_embd, uint32_t hc);
bool q4_hip_mean_hc_n(const float *gated, float *mixed, uint32_t n_embd, uint32_t hc,
                      uint32_t n_tok);
bool q4_hip_hc_combine(float *res, const float *block, const float *inject,
                       uint32_t n_embd, uint32_t hc);
bool q4_hip_hc_combine_n(float *res, const float *block, const float *inject,
                         uint32_t n_embd, uint32_t hc, uint32_t n_tok);
bool q4_hip_repeat_hc(const float *emb, float *res, uint32_t n_embd, uint32_t hc);
bool q4_hip_dot(const float *a, const float *b, float *out, uint32_t n);
bool q4_hip_gdn_conv(const float *hist, const float *qkv, const float *w,
                     float *out, uint32_t ch, uint32_t ksz);
bool q4_hip_gdn_shift_hist(float *hist, const float *qkv, uint32_t ch,
                           uint32_t ksz);
bool q4_hip_l2_heads(float *x, uint32_t n_h, uint32_t d, float eps);
bool q4_hip_gdn_step(float *S, const float *q, const float *k, const float *v,
                     const float *beta, const float *alpha, const float *A,
                     const float *dt, float *y, uint32_t n_vh, uint32_t n_kh,
                     uint32_t d, float qscale);
bool q4_hip_rms_gate_heads(float *h, const float *w, const float *z, uint32_t n_h,
                           uint32_t d, float eps);
bool q4_hip_add(float *y, const float *a, const float *b, uint64_t n);
void *q4_hip_malloc(size_t n);
void q4_hip_free(void *p);
bool q4_hip_h2d(void *d, const void *h, size_t n);
bool q4_hip_d2h(void *h, const void *d, size_t n);
int q4_hip_clock_khz(void);
bool q4_hip_experts_alloc(uint32_t n_slots, uint64_t slot_bytes);
void q4_hip_experts_free(void);
uint32_t q4_hip_experts_n(void);
bool q4_hip_experts_put(uint32_t slot, const uint8_t *host, uint64_t n);
bool q4_hip_experts_put_async(uint32_t slot, const uint8_t *host, uint64_t n);
void q4_hip_copy_wait(void);
const uint8_t *q4_hip_experts_dev(uint32_t slot);
bool q4_hip_softmax_topk(const float *d_logits, uint32_t n, uint32_t k,
                         int32_t *h_ids, float *h_wts);
bool q4_hip_gateup_q4k(const uint8_t *d_gate, const uint8_t *d_up, const float *d_x,
                       float *d_h, uint32_t n_ff, uint32_t n_embd);
bool q4_hip_gateup_q4k_n(const uint8_t *d_gate, const uint8_t *d_up, const float *d_x,
                         float *d_h, uint32_t n_ff, uint32_t n_embd, uint32_t n_tok);
bool q4_hip_qsa_q_prep(const float *d_qg, const float *d_wn, float *d_q, float *d_g,
                      uint32_t n_head, uint32_t hd, uint32_t n_rot, int32_t pos,
                      float base, float eps);
bool q4_hip_qsa_q_prep_n(const float *d_qg, const float *d_wn, float *d_q, float *d_g,
                        uint32_t n_head, uint32_t hd, uint32_t n_rot, int32_t pos0,
                        float base, float eps, uint32_t n_tok);
bool q4_hip_qsa_k_prep(float *d_k, const float *d_wn, uint32_t n_kvh, uint32_t hd,
                      uint32_t n_rot, int32_t pos, float base, float eps);
bool q4_hip_qsa_k_prep_n(float *d_k, const float *d_wn, uint32_t n_kvh, uint32_t hd,
                        uint32_t n_rot, int32_t pos0, float base, float eps,
                        uint32_t n_tok);
bool q4_hip_qsa_decode_n(const float *d_q, const void *d_kcache, const void *d_vcache,
                         const float *d_gate, float *d_out, uint32_t n_head,
                         uint32_t n_kvh, uint32_t hd, uint32_t n_tok, uint32_t kv_base,
                         uint32_t max_kv, float scale, int kv_f16);
/* Attend to an explicit token list (row t is idx[t * width + j], -1 pad). */
bool q4_hip_qsa_decode_idx(const float *d_q, const void *d_kcache, const void *d_vcache,
                           const float *d_gate, float *d_out, const int32_t *idx,
                           uint32_t width, uint32_t n_head, uint32_t n_kvh,
                           uint32_t hd, uint32_t n_tok, uint32_t max_kv, float scale,
                           int kv_f16);
/* Pool complete indexer blocks [b0, b1) from raw fp16 keys. */
bool q4_hip_qsa_pool(const void *raw_f16, float *pool, const float *wn,
                     uint32_t hd, uint32_t ratio, uint32_t b0, uint32_t b1,
                     uint32_t n_rot, float rope_base, float eps);
/* score[t, b] = sum_h relu(dot(q[t,h], pool[b])) for visible blocks, else -inf. */
bool q4_hip_qsa_score(const float *q, const float *pool, float *score,
                      uint32_t n_tok, uint32_t n_blocks, uint32_t stride,
                      uint32_t n_head, uint32_t hd, uint32_t pos0, uint32_t ratio);
/* Expand the best blocks plus the causal tail into sel[t * width + j]. */
bool q4_hip_qsa_topk(const float *score, int32_t *sel, uint32_t n_tok,
                     uint32_t n_blocks, uint32_t stride, uint32_t k_blk,
                     uint32_t ratio, uint32_t pos0, uint32_t width);
bool q4_hip_f32_to_f16(void *dst_f16, const float *src, uint64_t n);
bool q4_hip_f32_to_q8(void *dst_q8, const float *src, uint64_t n);
/* Batched PLE layer helpers (prefill): per-(tok,stream) gate written over kn,
 * dilated causal conv over the batch (state rows cover pre-batch offsets),
 * and the ping-pong conv-state tail update. */
bool q4_hip_ple_gate(float *kn, const float *qn, const float *val,
                     uint32_t n_embd, uint32_t hc, uint32_t n_tok);
bool q4_hip_ple_conv(float *co, const float *cn, const float *state,
                     const float *kw, uint32_t hc_dim, uint32_t kern,
                     uint32_t dil, uint32_t hist, uint32_t n_tok);
bool q4_hip_ple_state(float *dst, const float *old, const float *cn,
                      uint32_t n_tok, uint32_t hc_dim, uint32_t H);
/* Whole-chunk GDN tail (prefill): conv+shift+l2+delta-step+rms-gate over
 * n_tok tokens in one launch per layer. */
bool q4_hip_gdn_scan(float *hist, const float *qkv, const float *cw,
                     uint32_t ch, uint32_t ksz, float *S, const float *beta,
                     const float *alpha, const float *A, const float *dt,
                     const float *nw, const float *z, float *core,
                     uint32_t n_tok, uint32_t n_vh, uint32_t n_kh, uint32_t d,
                     float eps, float qscale);
void q4_hip_sync(void);
void q4_hip_clear(void);

extern double q4_prof_moe_io;
extern double q4_prof_moe_d2h;
extern double q4_prof_moe_gemm;
extern double q4_prof_moe_cw;
extern double q4_prof_moe_xd2h;
extern double q4_prof_moe_topk;
extern double q4_prof_moe_sg;
extern double q4_prof_moe_join;
extern double q4_prof_moe_merg;
extern double q4_prof_moe_ng;    /* GPU-L1 routed uses (decode hostpart) */
extern double q4_prof_moe_nc;    /* CPU-pool routed uses */
extern double q4_prof_moe_ncall; /* hostpart calls */
extern double q4_prof_dec[16]; /* 0 embed 1 attn 2 moe 3 combine 4 head
                                   5 ple 6 hc-attn 7 qsa/gdn 8 comb+ffnmix */
extern double q4_prof_step[8]; /* host wall per decode phase (Q4_STEP_PROF) */
extern double q4_prof_gpu_ms; /* GPU busy window sum (Q4_STEP_PROF) */
extern double q4_prof_gpu_ph[8]; /* per-phase GPU event ms (Q4_STEP_PROF) */
extern double q4_prof_gpu_sec[8]; /* gA section ms: hc1 attn comb ffnmix moea
                                     + QSA: prep sel dec */

/* Set by the server's SIGINT/SIGTERM handler; long loops poll and bail out. */
#include <signal.h>
extern volatile sig_atomic_t q4_stop;

#endif
