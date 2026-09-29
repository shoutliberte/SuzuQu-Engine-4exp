#include "q4.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(const char *m) {
    fprintf(stderr, "FAIL %s\n", m);
    return 1;
}

int main(int argc, char **argv) {
    q4_install_crashdump();
    if (argc > 1 && strcmp(argv[1], "--cpxbench") == 0)
        return q4_cpx_bench();
    if (argc > 1 && strcmp(argv[1], "--crash") == 0)
        abort();   /* crash-handler smoke test */
    const char *path = argc > 1 ? argv[1] : NULL;
    if (!path) {
        uint64_t b = 0;
        if (!q4_parse_gib("9GB", &b) || b != 9ull * Q4_GIB) return fail("parse 9GB");
        if (q4_parse_gib("0GB", &b)) return fail("parse 0");
        {
            float W[6] = {1, 2, 3, 4, 5, 6};
            float x[3] = {1, 1, 1};
            float y[2] = {0, 0};
            if (!q4_gemv(Q4_T_F32, (const uint8_t *)W, 2, 3, x, y, 1.0f))
                return fail("gemv f32");
            if (fabsf(y[0] - 6.f) > 1e-4f || fabsf(y[1] - 15.f) > 1e-4f)
                return fail("gemv values");
        }
        {
            /* One super-block, x = 1. IQ2_S/IQ3_S grids at index 0 are
             * constant bytes, so a zero codeword has a closed-form dot. */
            uint8_t iq2[82 * 2];
            memset(iq2, 0, sizeof(iq2));
            uint16_t one = 0x3c00;
            memcpy(iq2, &one, 2);
            memcpy(iq2 + 82, &one, 2);
            memset(iq2 + 82 + 74, 0x11, 8); /* both scale nibbles = 1 */
            float x[256], y[2];
            for (int i = 0; i < 256; i++) x[i] = 1.f;
            y[0] = y[1] = 0.f;
            if (!q4_gemv(Q4_T_IQ2_S, iq2, 2, 256, x, y, 1.f))
                return fail("iq2 gemv");
            /* d=1, grid byte 8, scale 0 → 0.125*8=1; scale nibble 1 → 0.375*8=3 */
            if (fabsf(y[0] - 256.f) > 1e-2f || fabsf(y[1] - 768.f) > 1e-2f)
                return fail("iq2 values");
            uint8_t iq3[110];
            memset(iq3, 0, sizeof(iq3));
            memcpy(iq3, &one, 2);
            y[0] = 0.f;
            if (!q4_gemv(Q4_T_IQ3_S, iq3, 1, 256, x, y, 1.f))
                return fail("iq3 gemv");
            if (fabsf(y[0] - 256.f) > 1e-2f) return fail("iq3 values");
            uint8_t iq4[18];
            memset(iq4, 0, sizeof(iq4));
            memcpy(iq4, &one, 2);
            float x32[32], y4 = 0.f;
            for (int i = 0; i < 32; i++) x32[i] = 1.f;
            if (!q4_gemv(Q4_T_IQ4_NL, iq4, 1, 32, x32, &y4, 1.f))
                return fail("iq4 gemv");
            if (fabsf(y4 - (-127.f * 32.f)) > 1e-2f) return fail("iq4 values");
            uint8_t q6[210];
            memset(q6, 0, sizeof(q6));
            memset(q6 + 192, 1, 16);
            memcpy(q6 + 208, &one, 2);
            y[0] = 0.f;
            if (!q4_gemv(Q4_T_Q6_K, q6, 1, 256, x, y, 1.f))
                return fail("q6 gemv");
            if (fabsf(y[0] - (-32.f * 256.f)) > 1e-2f) return fail("q6 values");
            if (q4_hip_init()) {
                /* Batched IQ GEMM must match the CPU dot, including a which[] gather. */
                {
                    const int cases[3][3] = {{Q4_T_IQ2_S, 40, 512},
                                             {Q4_T_IQ3_S, 40, 512},
                                             {Q4_T_IQ4_NL, 40, 64}};
                    for (int ci = 0; ci < 3; ci++) {
                        uint32_t ty = (uint32_t)cases[ci][0];
                        int nrows = cases[ci][1], ncols = cases[ci][2], nbat = 40;
                        uint64_t rb = q4_row_bytes(ty, (uint64_t)ncols);
                        uint8_t *w = calloc((size_t)nrows * rb, 1);
                        float *xb = calloc((size_t)nbat * ncols, 4);
                        float *yref = calloc((size_t)nbat * nrows, 4);
                        float *ygot = calloc((size_t)nbat * nrows, 4);
                        int32_t which[40];
                        if (!w || !xb || !yref || !ygot || rb == 0) return fail("iq batch alloc");
                        for (int r = 0; r < nrows; r++) {
                            uint16_t d = (uint16_t)(0x3c00 + (r & 3));
                            memcpy(w + (size_t)r * rb, &d, 2);
                            w[(size_t)r * rb + 4] = (uint8_t)(r * 3);
                        }
                        for (int i = 0; i < nbat * ncols; i++)
                            xb[i] = ((i * 17) % 11) * 0.05f - 0.2f;
                        for (int t = 0; t < nbat; t++) which[t] = nbat - 1 - t;
                        for (int t = 0; t < nbat; t++)
                            if (!q4_gemv(ty, w, (uint64_t)nrows, (uint64_t)ncols,
                                         xb + (size_t)which[t] * ncols,
                                         yref + (size_t)t * nrows, 1.f))
                                return fail("iq batch cpu");
                        uint8_t *dw = q4_hip_malloc((size_t)nrows * rb);
                        float *dx = q4_hip_malloc((size_t)nbat * ncols * 4);
                        float *dy = q4_hip_malloc((size_t)nbat * nrows * 4);
                        int32_t *dwhich = q4_hip_malloc(sizeof(which));
                        if (!dw || !dx || !dy || !dwhich) return fail("iq batch gpu alloc");
                        if (!q4_hip_h2d(dw, w, (size_t)nrows * rb) ||
                            !q4_hip_h2d(dx, xb, (size_t)nbat * ncols * 4) ||
                            !q4_hip_h2d(dwhich, which, sizeof(which)) ||
                            !q4_hip_gemv_dd_n_which(ty, dw, (uint64_t)nrows,
                                                    (uint64_t)ncols, dx, dy, 1.f,
                                                    (uint32_t)nbat, dwhich) ||
                            !q4_hip_d2h(ygot, dy, (size_t)nbat * nrows * 4))
                            return fail("iq batch gpu");
                        for (int i = 0; i < nbat * nrows; i++) {
                            float e = fabsf(ygot[i] - yref[i]);
                            float lim = 2e-3f * (1.f + fabsf(yref[i]));
                            if (e > lim) {
                                fprintf(stderr, "iq batch ty %u i %d got %g ref %g\n",
                                        ty, i, ygot[i], yref[i]);
                                return fail("iq batch values");
                            }
                        }
                        q4_hip_free(dw);
                        q4_hip_free(dx);
                        q4_hip_free(dy);
                        q4_hip_free(dwhich);
                        free(w);
                        free(xb);
                        free(yref);
                        free(ygot);
                    }
                }
                /* QSA: online softmax must match a dense reference for f32 and f16 KV. */
                {
                    const int n_head = 2, n_kvh = 1, hd = 32, n_tok = 2, npos = 4;
                    const int kv_base = 0;
                    float scale = 1.f / sqrtf((float)hd);
                    float *q = calloc((size_t)n_tok * n_head * hd, 4);
                    float *k = calloc((size_t)npos * n_kvh * hd, 4);
                    float *v = calloc((size_t)npos * n_kvh * hd, 4);
                    float *gate = calloc((size_t)n_tok * n_head * hd, 4);
                    float *out = calloc((size_t)n_tok * n_head * hd, 4);
                    float *ref = calloc((size_t)n_tok * n_head * hd, 4);
                    if (!q || !k || !v || !gate || !out || !ref) return fail("qsa alloc");
                    for (int i = 0; i < npos * hd; i++) {
                        k[i] = ((i * 17) % 11) * 0.05f - 0.2f;
                        v[i] = ((i * 13) % 9) * 0.07f;
                    }
                    for (int i = 0; i < n_tok * n_head * hd; i++) {
                        q[i] = ((i * 3) % 7) * 0.1f;
                        gate[i] = ((i % 5) - 2) * 0.3f;
                    }
                    int gqa = n_head / n_kvh;
                    for (int t = 0; t < n_tok; t++) {
                        int n_kv = kv_base + t + 1;
                        for (int h = 0; h < n_head; h++) {
                            int kh = h / gqa;
                            float sc[8];
                            float m = -1e30f;
                            for (int p = 0; p < n_kv; p++) {
                                float dot = 0.f;
                                for (int d = 0; d < hd; d++)
                                    dot += q[(t * n_head + h) * hd + d] *
                                           k[(p * n_kvh + kh) * hd + d];
                                sc[p] = dot * scale;
                                if (sc[p] > m) m = sc[p];
                            }
                            float sum = 0.f;
                            for (int p = 0; p < n_kv; p++) {
                                sc[p] = expf(sc[p] - m);
                                sum += sc[p];
                            }
                            for (int d = 0; d < hd; d++) {
                                float a = 0.f;
                                for (int p = 0; p < n_kv; p++)
                                    a += sc[p] * v[(p * n_kvh + kh) * hd + d];
                                float gv = 1.f / (1.f + expf(-gate[(t * n_head + h) * hd + d]));
                                ref[(t * n_head + h) * hd + d] = a / sum * gv;
                            }
                        }
                    }
                    float *dq = q4_hip_malloc((size_t)n_tok * n_head * hd * 4);
                    float *dk = q4_hip_malloc((size_t)npos * hd * 4);
                    float *dv = q4_hip_malloc((size_t)npos * hd * 4);
                    float *dg = q4_hip_malloc((size_t)n_tok * n_head * hd * 4);
                    float *dout = q4_hip_malloc((size_t)n_tok * n_head * hd * 4);
                    void *dk16 = q4_hip_malloc((size_t)npos * hd * 2);
                    void *dv16 = q4_hip_malloc((size_t)npos * hd * 2);
                    if (!dq || !dk || !dv || !dg || !dout || !dk16 || !dv16)
                        return fail("qsa hip alloc");
                    q4_hip_h2d(dq, q, (size_t)n_tok * n_head * hd * 4);
                    q4_hip_h2d(dk, k, (size_t)npos * hd * 4);
                    q4_hip_h2d(dv, v, (size_t)npos * hd * 4);
                    q4_hip_h2d(dg, gate, (size_t)n_tok * n_head * hd * 4);
                    if (!q4_hip_qsa_decode_n(dq, dk, dv, dg, dout, n_head, n_kvh, hd,
                                             n_tok, kv_base, npos, scale, 0))
                        return fail("qsa f32 launch");
                    q4_hip_d2h(out, dout, (size_t)n_tok * n_head * hd * 4);
                    for (int i = 0; i < n_tok * n_head * hd; i++) {
                        if (fabsf(out[i] - ref[i]) > 1e-3f) {
                            fprintf(stderr, "qsa f32 [%d] got %g ref %g\n", i, out[i], ref[i]);
                            return fail("qsa f32");
                        }
                    }
                    if (!q4_hip_f32_to_f16(dk16, dk, (uint64_t)npos * hd) ||
                        !q4_hip_f32_to_f16(dv16, dv, (uint64_t)npos * hd) ||
                        !q4_hip_qsa_decode_n(dq, dk16, dv16, dg, dout, n_head, n_kvh, hd,
                                             n_tok, kv_base, npos, scale, 1))
                        return fail("qsa f16 launch");
                    q4_hip_d2h(out, dout, (size_t)n_tok * n_head * hd * 4);
                    for (int i = 0; i < n_tok * n_head * hd; i++) {
                        if (fabsf(out[i] - ref[i]) > 2e-2f) {
                            fprintf(stderr, "qsa f16 [%d] got %g ref %g\n", i, out[i], ref[i]);
                            return fail("qsa f16");
                        }
                    }
                    void *dk8 = q4_hip_malloc((size_t)npos * hd / 32 * 34);
                    void *dv8 = q4_hip_malloc((size_t)npos * hd / 32 * 34);
                    if (!dk8 || !dv8) return fail("qsa q8 alloc");
                    if (!q4_hip_f32_to_q8(dk8, dk, (uint64_t)npos * hd) ||
                        !q4_hip_f32_to_q8(dv8, dv, (uint64_t)npos * hd) ||
                        !q4_hip_qsa_decode_n(dq, dk8, dv8, dg, dout, n_head, n_kvh, hd,
                                             n_tok, kv_base, npos, scale, 2))
                        return fail("qsa q8 launch");
                    q4_hip_d2h(out, dout, (size_t)n_tok * n_head * hd * 4);
                    for (int i = 0; i < n_tok * n_head * hd; i++) {
                        if (fabsf(out[i] - ref[i]) > 8e-2f) {
                            fprintf(stderr, "qsa q8 [%d] got %g ref %g\n", i, out[i], ref[i]);
                            return fail("qsa q8");
                        }
                    }
                    q4_hip_free(dq); q4_hip_free(dk); q4_hip_free(dv);
                    q4_hip_free(dg); q4_hip_free(dout);
                    q4_hip_free(dk16); q4_hip_free(dv16);
                    q4_hip_free(dk8); q4_hip_free(dv8);
                    free(q); free(k); free(v); free(gate); free(out); free(ref);
                }
                float g0 = 0.f, g1 = 0.f, g3 = 0.f, g4 = 0.f, g6 = 0.f;
                float yg[2];
                if (!q4_hip_gemv(Q4_T_IQ2_S, iq2, 2, 256, x, yg, 1.f))
                    return fail("iq2 hip");
                g0 = yg[0];
                g1 = yg[1];
                if (fabsf(g0 - 256.f) > 0.5f || fabsf(g1 - 768.f) > 0.5f)
                    return fail("iq2 hip values");
                if (!q4_hip_gemv(Q4_T_IQ3_S, iq3, 1, 256, x, &g3, 1.f) ||
                    fabsf(g3 - 256.f) > 0.5f)
                    return fail("iq3 hip");
                if (!q4_hip_gemv(Q4_T_IQ4_NL, iq4, 1, 32, x32, &g4, 1.f) ||
                    fabsf(g4 - (-127.f * 32.f)) > 0.5f)
                    return fail("iq4 hip");
                if (!q4_hip_gemv(Q4_T_Q6_K, q6, 1, 256, x, &g6, 1.f) ||
                    fabsf(g6 - (-32.f * 256.f)) > 0.5f)
                    return fail("q6 hip");
                /* Batched path dequants each row once. Four identical tokens. */
                {
                    float xb[4 * 256];
                    for (int t = 0; t < 4; t++)
                        memcpy(xb + t * 256, x, 256 * sizeof(float));
                    uint8_t *dw = q4_hip_malloc(82ull * 2);
                    float *dx = q4_hip_malloc(4ull * 256 * 4);
                    float *dy = q4_hip_malloc(4ull * 2 * 4);
                    float yh[8];
                    if (!dw || !dx || !dy) return fail("iq2 batch alloc");
                    q4_hip_h2d(dw, iq2, 82ull * 2);
                    q4_hip_h2d(dx, xb, 4ull * 256 * 4);
                    if (!q4_hip_gemv_dd_n(Q4_T_IQ2_S, dw, 2, 256, dx, dy, 1.f, 4))
                        return fail("iq2 batch launch");
                    q4_hip_d2h(yh, dy, sizeof(yh));
                    for (int t = 0; t < 4; t++) {
                        if (fabsf(yh[t * 2] - 256.f) > 0.5f ||
                            fabsf(yh[t * 2 + 1] - 768.f) > 0.5f) {
                            fprintf(stderr, "iq2 batch t%d %g %g\n", t, yh[t * 2],
                                    yh[t * 2 + 1]);
                            return fail("iq2 batch");
                        }
                    }
                    q4_hip_free(dw);
                    q4_hip_free(dx);
                    q4_hip_free(dy);
                }
            }
        }
        {
            float logits[5] = {0.1f, 0.9f, 0.2f, 0.8f, 0.3f};
            int32_t ids[2];
            float w[2];
            q4_topk(logits, 5, 2, ids, w, false);
            if (ids[0] != 1 || ids[1] != 3) return fail("topk ids");
        }
        {
            const char *xml =
                "ok then\n"
                "<tool_call>\n"
                "<function=bash>\n"
                "<parameter=command>\n"
                "ls -la /tmp\n"
                "</parameter>\n"
                "<parameter=timeout>\n"
                "30\n"
                "</parameter>\n"
                "</function>\n"
                "</tool_call>\n";
            char *prefix = NULL, **names = NULL, **args = NULL;
            int n = 0;
            if (q4_qwen_parse_tool_calls(xml, &prefix, &names, &args, &n) != 1)
                return fail("parse xml n");
            if (!prefix || strcmp(prefix, "ok then") != 0) return fail("parse xml prefix");
            if (!names || strcmp(names[0], "bash") != 0) return fail("parse xml name");
            if (!args || !strstr(args[0], "\"command\":\"ls -la /tmp\""))
                return fail("parse xml command");
            if (!strstr(args[0], "\"timeout\":30")) return fail("parse xml timeout");
            q4_qwen_free_tool_calls(names, args, n);
            free(prefix);
            const char *hermes =
                "<tool_call>\n{\"name\":\"read\",\"arguments\":{\"path\":\"a.c\"}}\n"
                "</tool_call>";
            prefix = NULL; names = NULL; args = NULL; n = 0;
            if (q4_qwen_parse_tool_calls(hermes, &prefix, &names, &args, &n) != 1)
                return fail("parse hermes n");
            if (strcmp(names[0], "read") != 0) return fail("parse hermes name");
            if (!strstr(args[0], "\"path\":\"a.c\"")) return fail("parse hermes args");
            q4_qwen_free_tool_calls(names, args, n);
            free(prefix);
            char *pre = q4_qwen_tools_preamble(
                "[{\"type\":\"function\",\"function\":{\"name\":\"bash\"}}]", 1);
            if (!pre || !strstr(pre, "<tools>") || !strstr(pre, "\"name\":\"bash\"") ||
                !strstr(pre, "<tool_call>"))
                return fail("tools preamble");
            free(pre);
            {
                char fat[4096];
                memset(fat, 'x', 3000);
                fat[3000] = 0;
                char req[4500];
                snprintf(req, sizeof(req),
                         "[{\"type\":\"function\",\"function\":{\"name\":\"bash\","
                         "\"description\":\"%s\",\"parameters\":{\"type\":\"object\","
                         "\"properties\":{\"command\":{\"type\":\"string\","
                         "\"description\":\"run a shell command\"}},"
                         "\"required\":[\"command\"]}}}]",
                         fat);
                char *c = q4_qwen_compact_tools_json(req);
                if (!c || strstr(c, "description") || strstr(c, "xxx") ||
                    !strstr(c, "\"name\":\"bash\"") || !strstr(c, "\"command\""))
                    return fail("compact tools json");
                if (strlen(c) > 400) return fail("compact tools still fat");
                char *p2 = q4_qwen_tools_preamble(req, 1);
                if (!p2 || strstr(p2, fat) || strlen(p2) > 800)
                    return fail("preamble still fat");
                free(c);
                free(p2);
                char *ord = q4_qwen_compact_tools_json(
                    "[{\"function\":{\"name\":\"write\"}},{\"function\":{\"name\":\"bash\"}}]");
                if (!ord) return fail("compact sort");
                const char *wb = strstr(ord, "\"name\":\"write\"");
                const char *bb = strstr(ord, "\"name\":\"bash\"");
                if (!wb || !bb || bb > wb) return fail("compact sort order");
                free(ord);
            }
        }
        {
            const char *w = "こんにちは！";
            if (q4_utf8_safe_end(w, 3) != 3) return fail("utf8 end of こ");
            if (q4_utf8_safe_end(w, 4) != 3) return fail("utf8 cut inside ん");
            if (q4_utf8_safe_end(w, 6) != 6) return fail("utf8 end of こん");
            if (q4_utf8_safe_end(w, 15) != 15) return fail("utf8 whole greeting");
            if (q4_utf8_safe_end("abc", 2) != 2) return fail("utf8 ascii");
        }
        if (q4_cpx_selftest()) return fail("cpx q8 kernels");
        printf("ok unit (no model)\n");
        return 0;
    }
    q4_gguf g;
    if (!q4_gguf_open(&g, path)) return fail("open");
    if (g.n_tensors == 0) return fail("no tensors");
    if (g.bytes_total == 0) return fail("zero size");
    if (g.n_files == 0) return fail("no files");
    if (g.arch && strcmp(g.arch, "qwen4exp") == 0) {
        if (g.n_expert == 0) return fail("qwen4exp has no experts");
        if (g.bytes_routed == 0) return fail("no routed bytes");
        if (g.per_expert_bytes == 0) return fail("per-expert");
    }
    q4_machine m = q4_machine_this_pc(8192);
    q4_place p = q4_place_plan(&g, &m);
    {
        uint64_t sum = g.bytes_dense + g.bytes_shared + g.bytes_routed + g.bytes_ple;
        if (sum != g.bytes_total) {
            fprintf(stderr, "kind sum %.3f vs total %.3f GiB\n",
                    (double)sum / Q4_GIB, (double)g.bytes_total / Q4_GIB);
            q4_gguf_close(&g);
            return fail("kind partition");
        }
    }
    if (g.arch && strcmp(g.arch, "qwen4exp") == 0) {
        if (g.ple_n_heads == 0 || g.bytes_ple == 0) return fail("PLE meta");
        if (g.hc_mult != 4) return fail("hc_mult");
        if (g.n_vocab == 0) return fail("n_vocab");
        if (g.ssm_d_inner == 0 || g.indexer_top_k == 0) return fail("ssm/qsa kv");
        q4_ple *ple = q4_ple_open(&g);
        if (!ple) return fail("ple open");
        int32_t toks[4] = {1, 2, 3, 4};
        int32_t rows[4 * 16];
        if (!q4_ple_hash(ple, toks, 4, NULL, rows)) {
            q4_ple_close(ple);
            q4_gguf_close(&g);
            return fail("ple hash");
        }
        float *emb = malloc(16 * 160 * sizeof(float));
        if (!emb || !q4_ple_gather(ple, rows, 16, emb)) {
            free(emb);
            q4_ple_close(ple);
            q4_gguf_close(&g);
            return fail("ple gather");
        }
        free(emb);
        q4_ple_close(ple);
    }
    q4_gguf_print(&g, stdout);
    q4_place_print(&p, &g, stdout);
    if (g.tok_vocab && g.n_tok_vocab) {
        q4_tok *tok = q4_tok_open(&g);
        if (!tok) {
            q4_gguf_close(&g);
            return fail("tok open");
        }
        int32_t ids[64];
        int n = q4_tok_encode(tok, "こんにちは", ids, 64);
        char back[64];
        int dn = q4_tok_decode(tok, ids, n, back, (int)sizeof(back));
        if (n <= 0 || n > 8 || dn <= 0 || strcmp(back, "こんにちは") != 0) {
            fprintf(stderr, "tok ja n=%d dec='%s'\n", n, back);
            q4_tok_close(tok);
            q4_gguf_close(&g);
            return fail("tok encode ja");
        }
        printf("tok encode ja %d tokens  roundtrip ok\n", n);
        /* Qwen2 regex splits each digit; whole-string BPE would not. */
        int nd = q4_tok_encode(tok, "123", ids, 64);
        if (nd != 3) {
            fprintf(stderr, "tok digits n=%d (want 3)\n", nd);
            q4_tok_close(tok);
            q4_gguf_close(&g);
            return fail("tok encode digits");
        }
        int ns = q4_tok_encode(tok, "hello world", ids, 64);
        if (ns < 2) {
            fprintf(stderr, "tok hello world n=%d\n", ns);
            q4_tok_close(tok);
            q4_gguf_close(&g);
            return fail("tok encode words");
        }
        {
            putenv("Q4_THINK=1");
            const char *roles[] = {"user", "assistant", "user"};
            const char *texts[] = {"hi", "answer only", "next"};
            int32_t hist[256];
            int nh = q4_tok_apply_messages(tok, roles, texts, 3, hist, 256);
            int32_t te = q4_tok_id(tok, "</think>");
            int32_t th = q4_tok_id(tok, "<think>");
            int saw_close = 0, open_unclosed = 0;
            for (int i = 0; i < nh; i++) {
                if (hist[i] == te) saw_close = 1;
                if (hist[i] == th && !saw_close) open_unclosed = 1;
            }
            /* History answer must sit after a closed think; only the final
             * assistant-open may leave <think> unclosed. */
            int last_th = -1, last_te = -1;
            for (int i = 0; i < nh; i++) {
                if (hist[i] == th) last_th = i;
                if (hist[i] == te) last_te = i;
            }
            if (nh <= 0 || last_th < 0 || last_te < 0 || last_te >= last_th) {
                fprintf(stderr, "tok hist think nh=%d last_th=%d last_te=%d\n",
                        nh, last_th, last_te);
                q4_tok_close(tok);
                q4_gguf_close(&g);
                return fail("tok history think close");
            }
            (void)open_unclosed;
            printf("tok history think closed then reopen  %d tokens\n", nh);
            int32_t tc = q4_tok_id(tok, "<tool_call>");
            int32_t tce = q4_tok_id(tok, "</tool_call>");
            int32_t ids2[8];
            int n2 = q4_tok_encode(tok, "<tool_call>", ids2, 8);
            if (tc < 0 || tce < 0 || n2 != 1 || ids2[0] != tc) {
                fprintf(stderr, "tok tool_call id=%d enc_n=%d enc0=%d\n", tc, n2,
                        n2 > 0 ? ids2[0] : -1);
                q4_tok_close(tok);
                q4_gguf_close(&g);
                return fail("tok tool_call special");
            }
        }
        q4_tok_close(tok);
        printf("tok digits 3  hello world %d tokens\n", ns);
    }
    {
        uint64_t kv = q4_qsa_kv_bytes(&g, 65536);
        if (g.n_layer && kv == 0) {
            q4_gguf_close(&g);
            return fail("qsa kv 64k");
        }
        printf("QSA KV 64K %.2f GiB\n", (double)kv / Q4_GIB);
    }
    q4_gguf_close(&g);
    printf("ok model\n");
    return 0;
}
