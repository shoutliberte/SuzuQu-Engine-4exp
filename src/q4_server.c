#define _GNU_SOURCE
#include "q4.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t g_stop;

volatile sig_atomic_t q4_stop;

static void on_stop(int sig) {
    (void)sig;
    g_stop = 1;
    q4_stop = 1;
}

static void install_signals(void) {
    /* No SA_RESTART: accept()/prefill/generate must notice the interrupt. */
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_stop;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    signal(SIGPIPE, SIG_IGN);
}

static int send_all(int fd, const char *p, size_t n) {
    while (n) {
        ssize_t w = send(fd, p, n, MSG_NOSIGNAL);
        if (w < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        p += (size_t)w;
        n -= (size_t)w;
    }
    return 0;
}

static int send_str(int fd, const char *s) {
    return send_all(fd, s, strlen(s));
}

static char *json_escape(const char *s) {
    if (!s) s = "";
    size_t n = strlen(s);
    char *o = malloc(n * 6 + 1);
    if (!o) return NULL;
    size_t j = 0;
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') {
            o[j++] = '\\';
            o[j++] = (char)c;
        } else if (c == '\n') {
            o[j++] = '\\';
            o[j++] = 'n';
        } else if (c == '\r') {
            o[j++] = '\\';
            o[j++] = 'r';
        } else if (c == '\t') {
            o[j++] = '\\';
            o[j++] = 't';
        } else if (c < 0x20) {
            j += (size_t)sprintf(o + j, "\\u%04x", c);
        } else if (c < 0x80) {
            o[j++] = (char)c;
        } else {
            /* copy a whole UTF-8 sequence or skip a bad byte */
            int L = 1;
            if ((c & 0xe0) == 0xc0) L = 2;
            else if ((c & 0xf0) == 0xe0) L = 3;
            else if ((c & 0xf8) == 0xf0) L = 4;
            else continue;
            if (i + (size_t)L > n) break;
            int ok = 1;
            for (int k = 1; k < L; k++)
                if ((((unsigned char)s[i + (size_t)k]) & 0xc0) != 0x80) ok = 0;
            if (!ok) continue;
            memcpy(o + j, s + i, (size_t)L);
            j += (size_t)L;
            i += (size_t)L - 1;
        }
    }
    o[j] = 0;
    return o;
}

static const char *skip_ws(const char *p) {
    while (*p && isspace((unsigned char)*p)) p++;
    return p;
}

static char *parse_json_string(const char **pp) {
    const char *p = skip_ws(*pp);
    if (*p != '"') return NULL;
    p++;
    size_t cap = 64, n = 0;
    char *o = malloc(cap);
    if (!o) return NULL;
    while (*p && *p != '"') {
        if (n + 8 >= cap) {
            cap *= 2;
            char *q = realloc(o, cap);
            if (!q) {
                free(o);
                return NULL;
            }
            o = q;
        }
        if (*p == '\\' && p[1]) {
            p++;
            switch (*p) {
            case 'n': o[n++] = '\n'; break;
            case 'r': o[n++] = '\r'; break;
            case 't': o[n++] = '\t'; break;
            case '"':
            case '\\':
            case '/': o[n++] = *p; break;
            case 'u': {
                unsigned cp = 0;
                for (int i = 0; i < 4 && p[1]; i++) {
                    p++;
                    char c = *p;
                    cp <<= 4;
                    if (c >= '0' && c <= '9') cp |= (unsigned)(c - '0');
                    else if (c >= 'a' && c <= 'f') cp |= (unsigned)(c - 'a' + 10);
                    else if (c >= 'A' && c <= 'F') cp |= (unsigned)(c - 'A' + 10);
                }
                if (cp < 0x80) o[n++] = (char)cp;
                else if (cp < 0x800) {
                    o[n++] = (char)(0xc0 | (cp >> 6));
                    o[n++] = (char)(0x80 | (cp & 0x3f));
                } else {
                    o[n++] = (char)(0xe0 | (cp >> 12));
                    o[n++] = (char)(0x80 | ((cp >> 6) & 0x3f));
                    o[n++] = (char)(0x80 | (cp & 0x3f));
                }
                break;
            }
            default: o[n++] = *p; break;
            }
            p++;
        } else {
            o[n++] = *p++;
        }
    }
    if (*p == '"') p++;
    o[n] = 0;
    *pp = p;
    return o;
}

static const char *find_key(const char *json, const char *key) {
    if (!json || !key) return NULL;
    size_t klen = strlen(key);
    int depth = 0;
    int in_str = 0, esc = 0;
    for (const char *p = json; *p; p++) {
        if (in_str) {
            if (esc) esc = 0;
            else if (*p == '\\') esc = 1;
            else if (*p == '"') in_str = 0;
            continue;
        }
        if (*p == '"') {
            if (depth == 1 && p[1 + klen] && strncmp(p + 1, key, klen) == 0 &&
                p[1 + klen] == '"') {
                const char *c = skip_ws(p + 2 + klen);
                if (*c == ':') return skip_ws(c + 1);
            }
            in_str = 1;
            continue;
        }
        if (*p == '{' || *p == '[') depth++;
        else if (*p == '}' || *p == ']') {
            depth--;
            if (depth < 0) break;
        }
    }
    return NULL;
}

static char *json_str(const char *json, const char *key) {
    const char *p = find_key(json, key);
    if (!p) return NULL;
    return parse_json_string(&p);
}

static int json_bool(const char *json, const char *key, int def) {
    const char *p = find_key(json, key);
    if (!p) return def;
    if (!strncmp(p, "true", 4)) return 1;
    if (!strncmp(p, "false", 5)) return 0;
    return def;
}

static double json_num(const char *json, const char *key, double def) {
    const char *p = find_key(json, key);
    if (!p) return def;
    return strtod(p, NULL);
}

static const char *skip_json_item(const char *p) {
    p = skip_ws(p);
    if (*p != '{' && *p != '[') return p;
    int d = 0, ins = 0, esc = 0;
    do {
        if (ins) {
            if (esc) esc = 0;
            else if (*p == '\\') esc = 1;
            else if (*p == '"') ins = 0;
        } else if (*p == '"') ins = 1;
        else if (*p == '{' || *p == '[') d++;
        else if (*p == '}' || *p == ']') d--;
        p++;
    } while (*p && d > 0);
    return p;
}

static char *json_raw(const char *json, const char *key) {
    const char *p = find_key(json, key);
    if (!p) return NULL;
    p = skip_ws(p);
    if (*p == '"') {
        const char *s = p;
        p++;
        int esc = 0;
        while (*p && !(!esc && *p == '"')) {
            if (esc) esc = 0;
            else if (*p == '\\') esc = 1;
            p++;
        }
        if (*p == '"') p++;
        return strndup(s, (size_t)(p - s));
    }
    if (*p == '{' || *p == '[') {
        const char *e = skip_json_item(p);
        return strndup(p, (size_t)(e - p));
    }
    const char *s = p;
    while (*p && *p != ',' && *p != '}' && *p != ']' && !isspace((unsigned char)*p))
        p++;
    return strndup(s, (size_t)(p - s));
}

static int buf_add(char **o, size_t *n, size_t *cap, const char *s, size_t L) {
    if (*n + L + 1 >= *cap) {
        size_t nc = (*n + L + 256) * 2;
        char *q = realloc(*o, nc);
        if (!q) return -1;
        *o = q;
        *cap = nc;
    }
    memcpy(*o + *n, s, L);
    *n += L;
    (*o)[*n] = 0;
    return 0;
}

static int xml_params_from_obj(char **o, size_t *n, size_t *cap, const char *obj) {
    const char *p = skip_ws(obj);
    if (*p != '{') return 0;
    p++;
    while (*p && *p != '}') {
        p = skip_ws(p);
        if (*p == ',') {
            p++;
            continue;
        }
        if (*p != '"') break;
        char *key = parse_json_string(&p);
        p = skip_ws(p);
        if (*p == ':') p++;
        p = skip_ws(p);
        char *val = NULL;
        if (*p == '"') {
            val = parse_json_string(&p);
        } else if (*p == '{' || *p == '[') {
            const char *e = skip_json_item(p);
            val = strndup(p, (size_t)(e - p));
            p = e;
        } else {
            const char *s = p;
            while (*p && *p != ',' && *p != '}' && !isspace((unsigned char)*p)) p++;
            val = strndup(s, (size_t)(p - s));
        }
        if (key && val) {
            char line[256];
            int hl = snprintf(line, sizeof(line), "<parameter=%s>\n", key);
            if (hl > 0) buf_add(o, n, cap, line, (size_t)hl);
            buf_add(o, n, cap, val, strlen(val));
            buf_add(o, n, cap, "\n</parameter>\n", 14);
        }
        free(key);
        free(val);
    }
    return 0;
}

static char *openai_tool_calls_to_xml(const char *arr) {
    arr = skip_ws(arr);
    if (*arr != '[') return NULL;
    arr++;
    size_t cap = 512, n = 0;
    char *o = malloc(cap);
    if (!o) return NULL;
    o[0] = 0;
    while (*arr && *arr != ']') {
        arr = skip_ws(arr);
        if (*arr == ',') {
            arr++;
            continue;
        }
        if (*arr != '{') break;
        const char *obj = arr;
        arr = skip_json_item(arr);
        const char *fn = find_key(obj, "function");
        const char *src = (fn && *skip_ws(fn) == '{') ? skip_ws(fn) : obj;
        char *name = json_str(src, "name");
        if (!name || !name[0]) {
            free(name);
            continue;
        }
        char head[256];
        int hl = snprintf(head, sizeof(head), "<tool_call>\n<function=%s>\n", name);
        buf_add(&o, &n, &cap, head, (size_t)hl);
        const char *ap = find_key(src, "arguments");
        if (ap) {
            ap = skip_ws(ap);
            if (*ap == '"') {
                char *s = parse_json_string(&ap);
                if (s) {
                    xml_params_from_obj(&o, &n, &cap, s);
                    free(s);
                }
            } else if (*ap == '{') {
                xml_params_from_obj(&o, &n, &cap, ap);
            }
        }
        buf_add(&o, &n, &cap, "</function>\n</tool_call>\n", 25);
        free(name);
    }
    return o;
}

static char *content_from_value(const char **pp) {
    const char *p = skip_ws(*pp);
    if (*p == '"') {
        char *s = parse_json_string(&p);
        *pp = p;
        return s;
    }
    if (*p != '[') {
        *pp = p;
        return strdup("");
    }
    p++;
    size_t cap = 256, n = 0;
    char *acc = malloc(cap);
    if (!acc) return NULL;
    acc[0] = 0;
    while (*p && *p != ']') {
        p = skip_ws(p);
        if (*p == ',') {
            p++;
            continue;
        }
        if (*p == '{') {
            const char *obj = p;
            int d = 0, ins = 0, e = 0;
            do {
                if (ins) {
                    if (e) e = 0;
                    else if (*p == '\\') e = 1;
                    else if (*p == '"') ins = 0;
                } else if (*p == '"') ins = 1;
                else if (*p == '{') d++;
                else if (*p == '}') d--;
                p++;
            } while (*p && d > 0);
            char *t = json_str(obj, "text");
            if (!t) t = json_str(obj, "content");
            if (t) {
                size_t L = strlen(t);
                if (n + L + 1 >= cap) {
                    cap = (n + L + 256) * 2;
                    char *q = realloc(acc, cap);
                    if (!q) {
                        free(t);
                        free(acc);
                        return NULL;
                    }
                    acc = q;
                }
                memcpy(acc + n, t, L);
                n += L;
                acc[n] = 0;
                free(t);
            }
        } else if (*p == '"') {
            char *t = parse_json_string(&p);
            if (t) {
                size_t L = strlen(t);
                if (n + L + 1 >= cap) {
                    cap = (n + L + 256) * 2;
                    char *q = realloc(acc, cap);
                    if (!q) {
                        free(t);
                        free(acc);
                        return NULL;
                    }
                    acc = q;
                }
                memcpy(acc + n, t, L);
                n += L;
                acc[n] = 0;
                free(t);
            }
        } else {
            p++;
        }
    }
    if (*p == ']') p++;
    *pp = p;
    return acc;
}

static int parse_messages(const char *json, char ***roles_out, char ***texts_out) {
    const char *p = find_key(json, "messages");
    if (!p || *p != '[') return 0;
    p++;
    int n = 0, cap = 8;
    char **roles = calloc((size_t)cap, sizeof(char *));
    char **texts = calloc((size_t)cap, sizeof(char *));
    if (!roles || !texts) {
        free(roles);
        free(texts);
        return 0;
    }
    while (*p && *p != ']') {
        p = skip_ws(p);
        if (*p == ',') {
            p++;
            continue;
        }
        if (*p != '{') {
            p++;
            continue;
        }
        const char *obj = p;
        int d = 0, ins = 0, e = 0;
        do {
            if (ins) {
                if (e) e = 0;
                else if (*p == '\\') e = 1;
                else if (*p == '"') ins = 0;
            } else if (*p == '"') ins = 1;
            else if (*p == '{') d++;
            else if (*p == '}') d--;
            p++;
        } while (*p && d > 0);
        char *role = json_str(obj, "role");
        const char *cp = find_key(obj, "content");
        char *text = cp ? content_from_value(&cp) : strdup("");
        if (!text) text = strdup("");
        char *reason = json_str(obj, "reasoning_content");
        if (!reason) reason = json_str(obj, "reasoning");
        if (reason && reason[0] && role && strcmp(role, "assistant") == 0 &&
            !strstr(text, "</think>") && !strstr(text, "<think>")) {
            size_t nr = strlen(reason), nt = strlen(text);
            char *w = malloc(nr + nt + 32);
            if (w) {
                sprintf(w, "<think>\n%s\n</think>\n\n%s", reason, text);
                free(text);
                text = w;
            }
        }
        free(reason);
        if (role && strcmp(role, "assistant") == 0) {
            const char *tc = find_key(obj, "tool_calls");
            if (tc && *skip_ws(tc) == '[') {
                char *xml = openai_tool_calls_to_xml(skip_ws(tc));
                if (xml && xml[0]) {
                    size_t nt = strlen(text), nx = strlen(xml);
                    char *w = malloc(nt + nx + 4);
                    if (w) {
                        sprintf(w, "%s%s%s", text,
                                (nt && text[nt - 1] != '\n') ? "\n" : "", xml);
                        free(text);
                        text = w;
                    }
                }
                free(xml);
            }
        }
        if (n == cap) {
            cap *= 2;
            roles = realloc(roles, (size_t)cap * sizeof(char *));
            texts = realloc(texts, (size_t)cap * sizeof(char *));
        }
        roles[n] = role ? role : strdup("user");
        texts[n] = text;
        n++;
    }
    *roles_out = roles;
    *texts_out = texts;
    return n;
}

static void free_msgs(char **roles, char **texts, int n) {
    for (int i = 0; i < n; i++) {
        free(roles[i]);
        free(texts[i]);
    }
    free(roles);
    free(texts);
}

static int http_reply(int fd, int code, const char *ctype, const char *body,
                      int extra_sse) {
    char hdr[512];
    size_t n = body ? strlen(body) : 0;
    const char *st = code == 200 ? "OK" : code == 400 ? "Bad Request" : "Error";
    int hlen = snprintf(hdr, sizeof(hdr),
                        "HTTP/1.1 %d %s\r\n"
                        "Content-Type: %s\r\n"
                        "Content-Length: %zu\r\n"
                        "Access-Control-Allow-Origin: *\r\n"
                        "Access-Control-Allow-Headers: *\r\n"
                        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                        "Connection: close\r\n"
                        "%s"
                        "\r\n",
                        code, st, ctype, n, extra_sse ? "Cache-Control: no-cache\r\n" : "");
    if (send_all(fd, hdr, (size_t)hlen) < 0) return -1;
    if (n && send_all(fd, body, n) < 0) return -1;
    return 0;
}

static int http_begin_sse(int fd) {
    const char *hdr =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n"
        "\r\n";
    return send_str(fd, hdr);
}

typedef struct {
    int fd;
    char id[64];
    int started;
    int has_tools;
    int n_tools;
    int sent;     /* bytes of acc already streamed as content */
    int tool_at;  /* index of <tool_call> in acc, or -1 */
    char *acc;
    size_t acc_n, acc_cap;
    volatile int *gone; /* set when the client socket closes */
} sse_u;

static int sse_send(sse_u *u, const char *s) {
    if (!u || !s) return -1;
    if (u->gone && *u->gone) return -1;
    if (send_str(u->fd, s) < 0) {
        if (u->gone) *u->gone = 1;
        return -1;
    }
    return 0;
}

static void sse_start_role(sse_u *u) {
    if (!u || u->started) return;
    char buf[512];
    snprintf(buf, sizeof(buf),
             "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
             "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},"
             "\"finish_reason\":null}]}\n\n",
             u->id);
    if (sse_send(u, buf) < 0) return;
    u->started = 1;
}

static void sse_progress(int prefill_i, int prefill_n, int gen_i, void *vu) {
    sse_u *u = vu;
    if (!u) return;
    char buf[96];
    if (gen_i <= 0)
        snprintf(buf, sizeof(buf), ": prefill %d/%d\n\n", prefill_i, prefill_n);
    else
        snprintf(buf, sizeof(buf), ": decode %d\n\n", gen_i);
    sse_send(u, buf);
}

static void sse_delta_field(sse_u *u, const char *field, const char *piece) {
    if (!u || !piece || !piece[0]) return;
    char *esc = json_escape(piece);
    if (!esc) return;
    sse_start_role(u);
    size_t need = strlen(esc) + strlen(u->id) + strlen(field) + 192;
    char *buf = malloc(need);
    if (!buf) {
        free(esc);
        return;
    }
    snprintf(buf, need,
             "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
             "\"choices\":[{\"index\":0,\"delta\":{\"%s\":\"%s\"},"
             "\"finish_reason\":null}]}\n\n",
             u->id, field, esc);
    sse_send(u, buf);
    free(buf);
    free(esc);
}

static void sse_tool_call(sse_u *u, int index, const char *name, const char *args) {
    char *en = json_escape(name ? name : "");
    char *ea = json_escape(args ? args : "{}");
    if (!en || !ea) {
        free(en);
        free(ea);
        return;
    }
    sse_start_role(u);
    size_t need = strlen(en) + strlen(ea) + strlen(u->id) + 320;
    char *buf = malloc(need);
    if (!buf) {
        free(en);
        free(ea);
        return;
    }
    snprintf(buf, need,
             "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
             "\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":%d,"
             "\"id\":\"call_%d\",\"type\":\"function\",\"function\":{\"name\":\"%s\","
             "\"arguments\":\"%s\"}}]},\"finish_reason\":null}]}\n\n",
             u->id, index, index, en, ea);
    sse_send(u, buf);
    free(buf);
    free(en);
    free(ea);
    u->n_tools++;
}

static void sse_flush_tools(sse_u *u) {
    if (!u || !u->acc) return;
    char *prefix = NULL, **names = NULL, **args = NULL;
    int n = 0;
    q4_qwen_parse_tool_calls(u->acc, &prefix, &names, &args, &n);
    if (prefix && prefix[0]) {
        /* send only the part not already streamed */
        size_t pl = strlen(prefix);
        if ((int)pl > u->sent)
            sse_delta_field(u, "content", prefix + u->sent);
        u->sent = (int)pl;
    }
    for (int i = 0; i < n; i++) {
        fprintf(stderr, "q4: tool_call[%d] %s %s\n", i, names[i], args[i] ? args[i] : "{}");
        sse_tool_call(u, i, names[i], args[i]);
    }
    q4_qwen_free_tool_calls(names, args, n);
    free(prefix);
}

static void sse_piece(const char *piece, int kind, void *vu) {
    sse_u *u = vu;
    if (kind == Q4_PIECE_REASONING) {
        sse_delta_field(u, "reasoning_content", piece);
        return;
    }
    if (!u->has_tools) {
        sse_delta_field(u, "content", piece);
        return;
    }
    size_t L = strlen(piece);
    if (buf_add(&u->acc, &u->acc_n, &u->acc_cap, piece, L) < 0) return;
    if (u->tool_at < 0) {
        char *hit = strstr(u->acc, "<tool_call>");
        if (hit) u->tool_at = (int)(hit - u->acc);
        int keep = 12;
        int send_upto = u->tool_at >= 0 ? u->tool_at : (int)u->acc_n - keep;
        if (send_upto < u->sent) send_upto = u->sent;
        /* The 12-byte holdback was cutting inside こんにちは and the
         * unsent tail was then skipped, so the UI showed こん. */
        send_upto = q4_utf8_safe_end(u->acc, send_upto);
        if (send_upto < u->sent) send_upto = u->sent;
        if (send_upto > u->sent) {
            char save = u->acc[send_upto];
            u->acc[send_upto] = 0;
            sse_delta_field(u, "content", u->acc + u->sent);
            u->acc[send_upto] = save;
            u->sent = send_upto;
        }
    }
}

static const char *INDEX_HTML =
    "q4 OpenAI-compatible server\n"
    "OpenCode: http://127.0.0.1:25757  model q4/qwen3.8-flash-next\n"
    "API:      http://127.0.0.1:8090/v1/chat/completions\n";

static int handle_chat(q4_engine *e, int fd, const char *body) {
    char **roles = NULL, **texts = NULL;
    int nmsg = parse_messages(body, &roles, &texts);
    if (nmsg <= 0) {
        http_reply(fd, 400, "application/json",
                   "{\"error\":{\"message\":\"messages required\"}}", 0);
        return 0;
    }
    {
        char *choice = json_str(body, "tool_choice");
        int skip = choice && strcmp(choice, "none") == 0;
        free(choice);
        char *tools_raw = skip ? NULL : json_raw(body, "tools");
        char *pre = tools_raw ? q4_qwen_tools_preamble(tools_raw, q4_tok_think_enabled())
                              : NULL;
        if (pre)
            fprintf(stderr, "q4: tools json %zu B -> preamble %zu B\n",
                    strlen(tools_raw), strlen(pre));
        free(tools_raw);
        if (pre) {
            if (nmsg > 0 && roles[0] && strcmp(roles[0], "system") == 0) {
                size_t ns = strlen(pre), nt = strlen(texts[0]);
                char *w = malloc(ns + nt + 4);
                if (w) {
                    sprintf(w, "%s\n\n%s", pre, texts[0]);
                    free(texts[0]);
                    texts[0] = w;
                }
                free(pre);
            } else {
                char **r = realloc(roles, (size_t)(nmsg + 1) * sizeof(char *));
                char **t = realloc(texts, (size_t)(nmsg + 1) * sizeof(char *));
                memmove(r + 1, r, (size_t)nmsg * sizeof(char *));
                memmove(t + 1, t, (size_t)nmsg * sizeof(char *));
                r[0] = strdup("system");
                t[0] = pre;
                roles = r;
                texts = t;
                nmsg++;
            }
        }
    }
    int stream = json_bool(body, "stream", 0);
    int max_new = (int)json_num(body, "max_tokens", 0);
    if (max_new <= 0) max_new = (int)json_num(body, "max_completion_tokens", 0);
    int want_new = max_new;
    if (max_new <= 0) {
        const char *en = getenv("Q4_MAX_NEW");
        max_new = (en && en[0]) ? atoi(en) : 32768;
        want_new = max_new;
    }
    int ctx = (int)q4_engine_ctx(e);
    if (max_new < 1) max_new = 1;
    if (ctx > 128 && max_new > ctx - 64) max_new = ctx - 64;
    double temp = json_num(body, "temperature", -1);
    double top_p = json_num(body, "top_p", -1);
    if (temp >= 0 || top_p >= 0)
        q4_engine_set_sample(e, temp >= 0 ? (float)temp : 0.7f, 20,
                             top_p >= 0 ? (float)top_p : 0.8f);

    fprintf(stderr, "q4: chat nmsg=%d stream=%d max_new=%d (req %d ctx %d)\n", nmsg,
            stream, max_new, want_new, ctx);
    fflush(stderr);

    char id[64];
    snprintf(id, sizeof(id), "chatcmpl-%ld", (long)time(NULL));
    size_t out_cap = (size_t)max_new * 16u + 65536u;
    if (out_cap < (1u << 18)) out_cap = 1u << 18;
    if (out_cap > (4u << 20)) out_cap = 4u << 20;
    char *out = malloc(out_cap);
    if (!out) {
        free_msgs(roles, texts, nmsg);
        return -1;
    }
    out[0] = 0;
    int has_tools = nmsg > 0 && texts[0] && strstr(texts[0], "<tools>");
    if (stream) {
        if (http_begin_sse(fd) < 0) {
            free(out);
            free_msgs(roles, texts, nmsg);
            return 0;
        }
        volatile int gone = 0;
        sse_u u = {.fd = fd,
                   .started = 0,
                   .has_tools = has_tools,
                   .tool_at = -1,
                   .gone = &gone};
        snprintf(u.id, sizeof(u.id), "%s", id);
        sse_start_role(&u);
        q4_engine_set_cancel(e, &gone);
        q4_engine_set_progress(e, sse_progress, &u);
        int n = q4_engine_generate_msgs(e, (const char *const *)roles,
                                        (const char *const *)texts, nmsg, out,
                                        (int)out_cap, max_new, sse_piece, &u);
        q4_engine_set_cancel(e, NULL);
        q4_engine_set_progress(e, NULL, NULL);
        (void)n;
        if (!gone) {
            if (has_tools) {
                if (u.tool_at >= 0) sse_flush_tools(&u);
                else if (u.acc && (int)u.acc_n > u.sent)
                    sse_delta_field(&u, "content", u.acc + u.sent);
            }
            if (!u.started) sse_start_role(&u);
            const char *fr = u.n_tools ? "tool_calls"
                                       : (q4_engine_last_hit_limit(e) ? "length"
                                                                      : "stop");
            char fin[512];
            snprintf(fin, sizeof(fin),
                     "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
                     "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}]}\n\n"
                     "data: [DONE]\n\n",
                     id, fr);
            send_str(fd, fin);
        }
        free(u.acc);
    } else {
        int n = q4_engine_generate_msgs(e, (const char *const *)roles,
                                        (const char *const *)texts, nmsg, out,
                                        (int)out_cap, max_new, NULL, NULL);
        char *prefix = NULL, **tnames = NULL, **targs = NULL;
        int ncalls = 0;
        q4_qwen_parse_tool_calls(n > 0 ? out : "", &prefix, &tnames, &targs, &ncalls);
        char *esc = json_escape(prefix && prefix[0] ? prefix : "");
        const char *rs = q4_engine_last_reasoning(e);
        char *resc = json_escape(rs && rs[0] ? rs : "");
        const char *fr = ncalls ? "tool_calls"
                                : (q4_engine_last_hit_limit(e) ? "length" : "stop");
        char *tcj = NULL;
        if (ncalls) {
            size_t cap = 256, tn = 0;
            tcj = malloc(cap);
            buf_add(&tcj, &tn, &cap, "[", 1);
            for (int i = 0; i < ncalls; i++) {
                char *en = json_escape(tnames[i]);
                char *ea = json_escape(targs[i]);
                char item[80];
                if (i) buf_add(&tcj, &tn, &cap, ",", 1);
                snprintf(item, sizeof(item),
                         "{\"id\":\"call_%d\",\"type\":\"function\",\"function\":{", i);
                buf_add(&tcj, &tn, &cap, item, strlen(item));
                buf_add(&tcj, &tn, &cap, "\"name\":\"", 8);
                buf_add(&tcj, &tn, &cap, en, strlen(en));
                buf_add(&tcj, &tn, &cap, "\",\"arguments\":\"", 16);
                buf_add(&tcj, &tn, &cap, ea, strlen(ea));
                buf_add(&tcj, &tn, &cap, "\"}}", 3);
                free(en);
                free(ea);
            }
            buf_add(&tcj, &tn, &cap, "]", 1);
        }
        size_t blen = (esc ? strlen(esc) : 0) + (resc ? strlen(resc) : 0) +
                      (tcj ? strlen(tcj) : 0) + 768;
        char *bodyj = malloc(blen);
        if (tcj)
            snprintf(bodyj, blen,
                     "{\"id\":\"%s\",\"object\":\"chat.completion\",\"model\":\"%s\","
                     "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
                     "\"content\":\"%s\",\"reasoning_content\":\"%s\",\"tool_calls\":%s},"
                     "\"finish_reason\":\"%s\"}],"
                     "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,"
                     "\"total_tokens\":%d}}\n",
                     id, q4_engine_name(e), esc ? esc : "", resc ? resc : "", tcj, fr,
                     q4_engine_last_prompt_tokens(e),
                     q4_engine_last_completion_tokens(e),
                     q4_engine_last_prompt_tokens(e) +
                         q4_engine_last_completion_tokens(e));
        else
            snprintf(bodyj, blen,
                     "{\"id\":\"%s\",\"object\":\"chat.completion\",\"model\":\"%s\","
                     "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
                     "\"content\":\"%s\",\"reasoning_content\":\"%s\"},"
                     "\"finish_reason\":\"%s\"}],"
                     "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,"
                     "\"total_tokens\":%d}}\n",
                     id, q4_engine_name(e), esc ? esc : "", resc ? resc : "", fr,
                     q4_engine_last_prompt_tokens(e),
                     q4_engine_last_completion_tokens(e),
                     q4_engine_last_prompt_tokens(e) +
                         q4_engine_last_completion_tokens(e));
        http_reply(fd, 200, "application/json", bodyj, 0);
        free(bodyj);
        free(esc);
        free(resc);
        free(tcj);
        free(prefix);
        q4_qwen_free_tool_calls(tnames, targs, ncalls);
    }
    free(out);
    free_msgs(roles, texts, nmsg);
    return 0;
}

static int read_req(int fd, char **path, char **body) {
    *path = NULL;
    *body = NULL;
    size_t cap = 8192, n = 0;
    char *buf = malloc(cap);
    if (!buf) return -1;
    while (n < cap - 1) {
        ssize_t r = recv(fd, buf + n, cap - 1 - n, 0);
        if (r < 0) {
            if (errno == EINTR) continue;
            free(buf);
            return -1;
        }
        if (r == 0) break;
        n += (size_t)r;
        buf[n] = 0;
        if (strstr(buf, "\r\n\r\n")) break;
        if (n + 1 == cap) {
            cap *= 2;
            if (cap > 1u << 16) break;
            char *q = realloc(buf, cap);
            if (!q) {
                free(buf);
                return -1;
            }
            buf = q;
        }
    }
    buf[n] = 0;
    char *line_end = strstr(buf, "\r\n");
    if (!line_end) {
        free(buf);
        return -1;
    }
    *line_end = 0;
    char method[16] = {0}, pth[1024] = {0};
    sscanf(buf, "%15s %1023s", method, pth);
    *path = malloc(strlen(method) + strlen(pth) + 4);
    sprintf(*path, "%s %s", method, pth);
    char *hdrs = line_end + 2;
    char *sep = strstr(hdrs, "\r\n\r\n");
    size_t clen = 0;
    for (char *h = hdrs; h && h < (sep ? sep : hdrs + strlen(hdrs));) {
        if (!strncasecmp(h, "Content-Length:", 15)) clen = (size_t)strtoul(h + 15, NULL, 10);
        char *nl = strstr(h, "\r\n");
        if (!nl) break;
        h = nl + 2;
    }
    const char *got = sep ? sep + 4 : "";
    size_t have = n - (size_t)(got - buf);
    if (clen > 8u << 20) clen = 8u << 20;
    char *b = calloc(clen + 1, 1);
    if (!b) {
        free(buf);
        return -1;
    }
    if (have > clen) have = clen;
    memcpy(b, got, have);
        while (have < clen) {
        if (g_stop) { free(b); free(buf); return -1; }
        ssize_t r = recv(fd, b + have, clen - have, 0);
        if (r < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (r == 0) break;
        have += (size_t)r;
    }
    b[have] = 0;
    *body = b;
    free(buf);
    return 0;
}

int q4_serve(q4_engine *e, const char *host, int port) {
    if (!e) return 1;
    if (!host || !host[0]) host = "127.0.0.1";
    if (port <= 0) port = 8090;
    install_signals();
    int sfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sfd < 0) {
        perror("socket");
        return 1;
    }
    int yes = 1;
    setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        fprintf(stderr, "q4: bad host %s\n", host);
        close(sfd);
        return 1;
    }
    if (bind(sfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(sfd);
        return 1;
    }
    if (listen(sfd, 16) < 0) {
        perror("listen");
        close(sfd);
        return 1;
    }
    fprintf(stderr, "q4: serving http://%s:%d/\n", host, port);
    fprintf(stderr, "    OpenAI  POST /v1/chat/completions\n");
    fprintf(stderr, "    OpenCode baseURL http://%s:%d/v1\n", host, port);
    fprintf(stderr, "    UI      http://%s:%d/\n", host, port);
    while (!g_stop) {
        struct sockaddr_in cli;
        socklen_t cl = sizeof(cli);
        int fd = accept(sfd, (struct sockaddr *)&cli, &cl);
        if (fd < 0) {
            if (g_stop) break;
            if (errno == EINTR) continue;
            perror("accept");
            continue;
        }
        {
            int one = 1;
            setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        }
        char *path = NULL, *body = NULL;
        if (read_req(fd, &path, &body) < 0) {
            close(fd);
            continue;
        }
        const char *req = path ? path : "";
        if (!strncmp(req, "OPTIONS ", 8)) {
            http_reply(fd, 200, "text/plain", "", 0);
        } else if (!strcmp(req, "GET /") || !strcmp(req, "GET /index.html")) {
            http_reply(fd, 200, "text/plain; charset=utf-8", INDEX_HTML, 0);
        } else if (!strcmp(req, "GET /health") || !strcmp(req, "GET /v1/health")) {
            http_reply(fd, 200, "application/json", "{\"ok\":true}\n", 0);
        } else if (!strcmp(req, "GET /v1/models") || !strcmp(req, "GET /models")) {
            char buf[512];
            snprintf(buf, sizeof(buf),
                     "{\"object\":\"list\",\"data\":[{\"id\":\"%s\",\"object\":\"model\","
                     "\"owned_by\":\"q4\"}]}\n",
                     q4_engine_name(e));
            http_reply(fd, 200, "application/json", buf, 0);
        } else if (strstr(req, "POST ") &&
                   (strstr(req, "/v1/chat/completions") ||
                    strstr(req, "/chat/completions"))) {
            handle_chat(e, fd, body ? body : "{}");
        } else {
            http_reply(fd, 404, "application/json",
                       "{\"error\":{\"message\":\"not found\"}}\n", 0);
        }
        free(path);
        free(body);
        close(fd);
    }
    close(sfd);
    return 0;
}
