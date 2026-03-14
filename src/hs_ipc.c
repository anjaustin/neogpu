#include "hs_ipc.h"
#include "hs_nodes.h"
#include "hs_ml.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

enum {
    HS_IPC_MAGIC = 0x4E474950u, /* 'NGIP' */
    HS_IPC_VERSION = 1,
    HS_IPC_TYPE_REQ = 1,
    HS_IPC_TYPE_RESP = 2,

    HS_IPC_OK = 0,
    HS_IPC_ERR_BAD_MAGIC = 1,
    HS_IPC_ERR_BAD_VERSION = 2,
    HS_IPC_ERR_BAD_TYPE = 3,
    HS_IPC_ERR_BAD_LEN = 4,
    HS_IPC_ERR_UNSUPPORTED_OP = 5,
    HS_IPC_ERR_BAD_PAYLOAD = 6,
    HS_IPC_ERR_ENQUEUE_FAILED = 7,
    HS_IPC_ERR_TIMEOUT = 8,
    HS_IPC_ERR_INTERNAL = 9,
    HS_IPC_ERR_BAD_PATH = 10,
};

typedef struct {
    u32 magic;
    u16 version;
    u16 type;
    u32 len;
    u32 cid;
} __attribute__((packed)) HSIpcHdr;

typedef struct {
    u8 op;
    u8 flags;
    u16 reserved;
    u32 payload_len;
    /* payload follows */
} __attribute__((packed)) HSIpcReq;

typedef struct {
    u32 status;
    u32 result_op;
    u32 result_len;
    /* result bytes follow */
} __attribute__((packed)) HSIpcResp;

/* P0-3: Thread-safe global ML state with mutex protection */
static HSMLSystem g_ml_system;
static bool g_ml_initialized = false;
static pthread_mutex_t g_ml_mutex = PTHREAD_MUTEX_INITIALIZER;

/*
 * P0-1: Path validation for IPC file operations.
 * 
 * Security policy:
 * - Reject embedded null bytes
 * - Reject path traversal sequences (..)
 * - Reject absolute paths (starting with /)
 * - Reject paths with control characters
 * - Require .gguf extension for ML model files
 * 
 * Returns true if path is safe, false otherwise.
 */
static bool hs_ipc_path_is_safe(const char* path, u32 len) {
    if (!path || len == 0) return false;
    
    /* Reject absolute paths */
    if (path[0] == '/') return false;
    
    /* Scan for dangerous patterns */
    for (u32 i = 0; i < len; i++) {
        char c = path[i];
        
        /* Reject embedded null bytes */
        if (c == '\0') return false;
        
        /* Reject control characters (0x00-0x1F, 0x7F) */
        if (c < 0x20 || c == 0x7F) return false;
        
        /* Check for path traversal: .. at start, end, or surrounded by / */
        if (c == '.' && i + 1 < len && path[i + 1] == '.') {
            /* .. at start of path */
            if (i == 0) return false;
            /* /.. pattern */
            if (path[i - 1] == '/') return false;
            /* ../ pattern at start */
            if (i == 0 && i + 2 < len && path[i + 2] == '/') return false;
        }
    }
    
    /* Require .gguf extension for model files */
    if (len < 5) return false;
    const char* ext = path + len - 5;
    if (ext[0] != '.' || ext[1] != 'g' || ext[2] != 'g' || ext[3] != 'u' || ext[4] != 'f') {
        return false;
    }
    
    return true;
}

static int hs_ipc_handle_ml_op(u8 op, const u8* payload, u32 payload_len, u8* out, u32* out_len) {
    if (!out || !out_len) return -1;

    /* P0-3: Lock mutex for all ML operations */
    pthread_mutex_lock(&g_ml_mutex);

    if (!g_ml_initialized) {
        hs_ml_init(&g_ml_system);
        g_ml_initialized = true;
    }

    switch ((OpCode)op) {
        case OP_ML_LOAD: {
            if (payload_len == 0 || payload_len > HS_PAYLOAD_SIZE) {
                out[0] = 0;
                *out_len = 1;
                break;
            }

            char path[HS_PAYLOAD_SIZE + 1];
            memcpy(path, payload, payload_len);
            path[payload_len] = '\0';

            /* P0-1: Validate path before file access */
            if (!hs_ipc_path_is_safe(path, payload_len)) {
                out[0] = 0;
                *out_len = 1;
                break;
            }

            out[0] = (hs_ml_load_gguf(&g_ml_system, path) == 0) ? 1 : 0;
            *out_len = 1;
            break;
        }

        case OP_ML_UNLOAD:
            hs_ml_free(&g_ml_system);
            g_ml_initialized = false;
            out[0] = 1;
            *out_len = 1;
            break;

        case OP_ML_FORWARD: {
            if (payload_len < 8) {
                out[0] = 0;
                *out_len = 1;
                break;
            }

            u32 seq_len = 0;
            memcpy(&seq_len, payload, 4);
            u32 available_tokens = (payload_len - 4) / (u32)sizeof(u32);
            if (seq_len == 0 || seq_len > available_tokens || !g_ml_system.loaded || g_ml_system.vocab_size == 0) {
                out[0] = 0;
                *out_len = 1;
                break;
            }

            float* logits = malloc(g_ml_system.vocab_size * sizeof(float));
            if (!logits) {
                out[0] = 0;
                *out_len = 1;
                break;
            }

            if (hs_ml_forward(&g_ml_system, (const u32*)(payload + 4), seq_len, logits) == 0) {
                u32 max_logits = (HS_TOOLBUS_PAYLOAD_MAX - 1) / (u32)sizeof(float);
                u32 copy_len = g_ml_system.vocab_size;
                if (copy_len > max_logits) copy_len = max_logits;
                out[0] = 1;
                memcpy(out + 1, logits, (size_t)copy_len * sizeof(float));
                *out_len = 1 + copy_len * (u32)sizeof(float);
            } else {
                out[0] = 0;
                *out_len = 1;
            }
            free(logits);
            break;
        }

        case OP_ML_GENERATE: {
            if (payload_len < 16) {
                out[0] = 0;
                *out_len = 1;
                break;
            }

            u32 prompt_len = 0;
            u32 max_new = 0;
            float temperature = 0.0f;
            u32 top_k = 0;
            memcpy(&prompt_len, payload, 4);
            memcpy(&max_new, payload + 4, 4);
            memcpy(&temperature, payload + 8, 4);
            memcpy(&top_k, payload + 12, 4);

            if (prompt_len > payload_len - 16 || !g_ml_system.loaded) {
                out[0] = 0;
                *out_len = 1;
                break;
            }
            if (max_new > 256) max_new = 256;

            char prompt[HS_PAYLOAD_SIZE - 16 + 1];
            memcpy(prompt, payload + 16, prompt_len);
            prompt[prompt_len] = '\0';

            u32 output_cap = prompt_len + max_new + 1;
            u32* output = malloc((size_t)output_cap * sizeof(u32));
            if (!output) {
                out[0] = 0;
                *out_len = 1;
                break;
            }

            u32 num_gen = hs_ml_generate(&g_ml_system, prompt, max_new, temperature, top_k, output);
            if (num_gen == 0 && max_new != 0) {
                out[0] = 0;
                *out_len = 1;
                free(output);
                break;
            }

            u32 max_tokens = (HS_TOOLBUS_PAYLOAD_MAX - 5) / (u32)sizeof(u32);
            u32 copy_len = num_gen;
            if (copy_len > max_tokens) copy_len = max_tokens;
            out[0] = 1;
            memcpy(out + 1, &num_gen, 4);
            memcpy(out + 5, output, (size_t)copy_len * sizeof(u32));
            *out_len = 5 + copy_len * (u32)sizeof(u32);
            free(output);
            break;
        }

        default:
            out[0] = 0;
            *out_len = 1;
            break;
    }

    /* P0-3: Unlock mutex */
    pthread_mutex_unlock(&g_ml_mutex);

    return 0;
}

static bool hs_ipc_op_allowed(u8 op) {
    switch ((OpCode)op) {
        case OP_QUERY_STATS:
        case OP_QUERY_FABRIC:
        case OP_SET_RECORD_MASK:
        case OP_SET_CHAN_BUDGET:
        case OP_SET_BLOCK_POLICY:
        case OP_FENCE:
        case OP_ML_LOAD:
        case OP_ML_UNLOAD:
        case OP_ML_FORWARD:
        case OP_ML_GENERATE:
        case OP_ML_TOKENIZE:
        case OP_ML_DETOKENIZE:
            return true;
        default:
            return false;
    }
}

static bool hs_ipc_payload_valid(u8 op, u32 payload_len) {
    switch ((OpCode)op) {
        case OP_SET_RECORD_MASK:  return payload_len == 4;
        case OP_SET_CHAN_BUDGET:  return payload_len == 8;
        case OP_SET_BLOCK_POLICY: return payload_len == 2;
        case OP_QUERY_STATS:
        case OP_QUERY_FABRIC:
        case OP_FENCE:
        case OP_ML_UNLOAD:
        case OP_ML_TOKENIZE:
        case OP_ML_DETOKENIZE:
            return payload_len == 0;
        case OP_ML_LOAD:
            return payload_len > 0 && payload_len <= HS_PAYLOAD_SIZE;
        case OP_ML_FORWARD:
            return payload_len >= 8 && payload_len <= HS_PAYLOAD_SIZE && ((payload_len - 4) % sizeof(u32) == 0);
        case OP_ML_GENERATE:
            return payload_len >= 16 && payload_len <= HS_PAYLOAD_SIZE;
        default:
            return false;
    }
}

static int hs_write_full(int fd, const void* buf, size_t len) {
    const u8* p = (const u8*)buf;
    size_t off = 0;
    while (off < len) {
        ssize_t w = write(fd, p + off, len - off);
        if (w < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (w == 0) return -1;
        off += (size_t)w;
    }
    return 0;
}

static int hs_read_full(int fd, void* buf, size_t len) {
    u8* p = (u8*)buf;
    size_t off = 0;
    while (off < len) {
        ssize_t r = read(fd, p + off, len - off);
        if (r < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (r == 0) return -1;
        off += (size_t)r;
    }
    return 0;
}

static void hs_ipc_send_error(int fd, u32 cid, u32 status) {
    HSIpcHdr h = {
        .magic = HS_IPC_MAGIC,
        .version = HS_IPC_VERSION,
        .type = HS_IPC_TYPE_RESP,
        .len = (u32)sizeof(HSIpcResp),
        .cid = cid,
    };
    HSIpcResp r = {
        .status = status,
        .result_op = 0,
        .result_len = 0,
    };
    (void)hs_write_full(fd, &h, sizeof(h));
    (void)hs_write_full(fd, &r, sizeof(r));
}

static int hs_ipc_handle_req(HSIpcServer* srv, int cfd, const HSIpcHdr* hdr) {
    if (!srv || !hdr) return -1;

    if (hdr->len < sizeof(HSIpcReq) || hdr->len > (sizeof(HSIpcReq) + HS_PAYLOAD_SIZE)) {
        hs_ipc_send_error(cfd, hdr->cid, HS_IPC_ERR_BAD_LEN);
        return -1;
    }

    u8 body[sizeof(HSIpcReq) + HS_PAYLOAD_SIZE];
    if (hs_read_full(cfd, body, hdr->len) != 0) return -1;

    const HSIpcReq* req = (const HSIpcReq*)body;
    const u8* payload = body + sizeof(HSIpcReq);

    if (!hs_ipc_op_allowed(req->op)) {
        hs_ipc_send_error(cfd, hdr->cid, HS_IPC_ERR_UNSUPPORTED_OP);
        return 0;
    }

    if (!hs_ipc_payload_valid(req->op, req->payload_len)) {
        hs_ipc_send_error(cfd, hdr->cid, HS_IPC_ERR_BAD_PAYLOAD);
        return 0;
    }

    if (sizeof(HSIpcReq) + req->payload_len != hdr->len) {
        hs_ipc_send_error(cfd, hdr->cid, HS_IPC_ERR_BAD_LEN);
        return 0;
    }
    
    /* Handle ML ops directly without going through toolbus */
    switch ((OpCode)req->op) {
        case OP_ML_LOAD:
        case OP_ML_UNLOAD:
        case OP_ML_FORWARD:
        case OP_ML_GENERATE: {
            u8 ml_out[HS_TOOLBUS_PAYLOAD_MAX];
            u32 ml_out_len = 0;
            
            hs_ipc_handle_ml_op(req->op, payload, req->payload_len, ml_out, &ml_out_len);
            
            HSIpcHdr rh = {
                .magic = HS_IPC_MAGIC,
                .version = HS_IPC_VERSION,
                .type = HS_IPC_TYPE_RESP,
                .len = (u32)(sizeof(HSIpcResp) + ml_out_len),
                .cid = hdr->cid,
            };
            HSIpcResp rr = {
                .status = HS_IPC_OK,
                .result_op = req->op,
                .result_len = ml_out_len,
            };
            
            if (hs_write_full(cfd, &rh, sizeof(rh)) != 0) return -1;
            if (hs_write_full(cfd, &rr, sizeof(rr)) != 0) return -1;
            if (ml_out_len) {
                if (hs_write_full(cfd, ml_out, ml_out_len) != 0) return -1;
            }
            return 0;
        }
        default:
            break;  /* Continue with normal toolbus handling */
    }
    
    HSSystem* sys = srv->sys;
    if (!sys) {
        hs_ipc_send_error(cfd, hdr->cid, HS_IPC_ERR_INTERNAL);
        return 0;
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    u32 after_seq = hs_toolbus_seq(sys);

    Message m = {
        .to = NODE_SYSTEM,
        .from = NODE_CPU,
        .op = req->op,
        .flags = 0,
        .cid = hdr->cid,
        .tick = 0,
        .payload_idx = 0,
        .payload_len = 0,
        .channel = CHAN_RT,
    };
    if ((OpCode)req->op == OP_FENCE) {
        m.flags = req->flags;
    }

    bool ok = false;
    if (req->payload_len) ok = hs_send_with_payload(sys, &m, payload, req->payload_len);
    else ok = hs_send(sys, &m);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    long send_us = (t1.tv_sec - t0.tv_sec) * 1000000 + (t1.tv_nsec - t0.tv_nsec) / 1000;
    (void)send_us;  /* unused for now */

    if (!ok) {
        hs_ipc_send_error(cfd, hdr->cid, HS_IPC_ERR_ENQUEUE_FAILED);
        return 0;
    }

    u8 out[HS_TOOLBUS_PAYLOAD_MAX];
    u32 out_len = 0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (!hs_toolbus_wait(sys, after_seq, hdr->cid, req->op, out, &out_len, 250)) {
        hs_ipc_send_error(cfd, hdr->cid, HS_IPC_ERR_TIMEOUT);
        return 0;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    long wait_us = (t1.tv_sec - t0.tv_sec) * 1000000 + (t1.tv_nsec - t0.tv_nsec) / 1000;
    (void)wait_us;  /* unused for now */

    HSIpcHdr rh = {
        .magic = HS_IPC_MAGIC,
        .version = HS_IPC_VERSION,
        .type = HS_IPC_TYPE_RESP,
        .len = (u32)(sizeof(HSIpcResp) + out_len),
        .cid = hdr->cid,
    };
    HSIpcResp rr = {
        .status = HS_IPC_OK,
        .result_op = req->op,
        .result_len = out_len,
    };

    if (hs_write_full(cfd, &rh, sizeof(rh)) != 0) return -1;
    if (hs_write_full(cfd, &rr, sizeof(rr)) != 0) return -1;
    if (out_len) {
        if (hs_write_full(cfd, out, out_len) != 0) return -1;
    }
    return 0;
}

static void* hs_ipc_thread_main(void* arg) {
    HSIpcServer* srv = (HSIpcServer*)arg;
    if (!srv) return NULL;

    int fd;
    if (srv->mode == HS_IPC_MODE_TCP) {
        fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return NULL;
        srv->listen_fd = fd;

        int opt = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in tcp_addr;
        memset(&tcp_addr, 0, sizeof(tcp_addr));
        tcp_addr.sin_family = AF_INET;
        tcp_addr.sin_port = htons(srv->port);
        if (!srv->addr[0] || strcmp(srv->addr, "0.0.0.0") == 0) {
            tcp_addr.sin_addr.s_addr = INADDR_ANY;
        } else {
            inet_pton(AF_INET, srv->addr, &tcp_addr.sin_addr);
        }

        if (bind(fd, (struct sockaddr*)&tcp_addr, sizeof(tcp_addr)) != 0) {
            close(fd);
            srv->listen_fd = -1;
            return NULL;
        }

        if (listen(fd, 4) != 0) {
            close(fd);
            srv->listen_fd = -1;
            return NULL;
        }
    } else {
        fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd < 0) return NULL;
        srv->listen_fd = fd;

        struct sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, srv->path, sizeof(addr.sun_path) - 1);

        (void)unlink(srv->path);
        if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
            close(fd);
            srv->listen_fd = -1;
            return NULL;
        }
        (void)chmod(srv->path, 0600);
        if (listen(fd, 4) != 0) {
            close(fd);
            srv->listen_fd = -1;
            return NULL;
        }
    }

    atomic_store_explicit(&srv->running, true, memory_order_release);

    while (!atomic_load_explicit(&srv->stop, memory_order_acquire)) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(fd, &rfds);
        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 200000; /* 200ms */

        int rc = select(fd + 1, &rfds, NULL, NULL, &tv);
        if (rc < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (rc == 0) continue;
        if (!FD_ISSET(fd, &rfds)) continue;

        int cfd = accept(fd, NULL, NULL);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            break;
        }

        /* single connection, sequential request/response */
        for (;;) {
            HSIpcHdr h;
            if (hs_read_full(cfd, &h, sizeof(h)) != 0) break;

            if (h.magic != HS_IPC_MAGIC) {
                hs_ipc_send_error(cfd, h.cid, HS_IPC_ERR_BAD_MAGIC);
                break;
            }
            if (h.version != HS_IPC_VERSION) {
                hs_ipc_send_error(cfd, h.cid, HS_IPC_ERR_BAD_VERSION);
                break;
            }
            if (h.type != HS_IPC_TYPE_REQ) {
                hs_ipc_send_error(cfd, h.cid, HS_IPC_ERR_BAD_TYPE);
                break;
            }

            if (hs_ipc_handle_req(srv, cfd, &h) != 0) break;
        }

        close(cfd);
    }

    close(fd);
    srv->listen_fd = -1;
    if (srv->mode == HS_IPC_MODE_UNIX) {
        (void)unlink(srv->path);
    }
    atomic_store_explicit(&srv->running, false, memory_order_release);
    return NULL;
}

bool hs_ipc_start(HSIpcServer* srv, HSSystem* sys, const char* path) {
    if (!srv || !sys || !path) return false;
    memset(srv, 0, sizeof(*srv));
    srv->sys = sys;
    srv->mode = HS_IPC_MODE_UNIX;
    srv->listen_fd = -1;
    atomic_init(&srv->running, false);
    atomic_init(&srv->stop, false);
    strncpy(srv->path, path, sizeof(srv->path) - 1);

    int rc = pthread_create(&srv->thread, NULL, hs_ipc_thread_main, srv);
    if (rc != 0) return false;

    /* wait briefly for running */
    for (int i = 0; i < 50; i++) {
        if (atomic_load_explicit(&srv->running, memory_order_acquire)) return true;
        usleep(1000);
    }
    return atomic_load_explicit(&srv->running, memory_order_acquire);
}

bool hs_ipc_start_tcp(HSIpcServer* srv, HSSystem* sys, const char* addr, u16 port) {
    if (!srv || !sys) return false;
    memset(srv, 0, sizeof(*srv));
    srv->sys = sys;
    srv->mode = HS_IPC_MODE_TCP;
    srv->port = port ? port : HS_IPC_DEFAULT_PORT;
    if (addr) strncpy(srv->addr, addr, sizeof(srv->addr) - 1);
    srv->listen_fd = -1;
    atomic_init(&srv->running, false);
    atomic_init(&srv->stop, false);

    int rc = pthread_create(&srv->thread, NULL, hs_ipc_thread_main, srv);
    if (rc != 0) return false;

    /* wait briefly for running */
    for (int i = 0; i < 50; i++) {
        if (atomic_load_explicit(&srv->running, memory_order_acquire)) return true;
        usleep(1000);
    }
    return atomic_load_explicit(&srv->running, memory_order_acquire);
}

void hs_ipc_stop(HSIpcServer* srv) {
    if (!srv) return;
    atomic_store_explicit(&srv->stop, true, memory_order_release);
    if (srv->listen_fd >= 0) {
        shutdown(srv->listen_fd, SHUT_RDWR);
    }
    if (atomic_load_explicit(&srv->running, memory_order_acquire)) {
        pthread_join(srv->thread, NULL);
    }
    atomic_store_explicit(&srv->running, false, memory_order_release);
}
