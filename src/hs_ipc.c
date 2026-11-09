#include "hs_ipc.h"
#include "hs_nodes.h"

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
#include <poll.h>

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

static bool hs_ipc_op_allowed(u8 op) {
    switch ((OpCode)op) {
        case OP_QUERY_STATS:
        case OP_QUERY_FABRIC:
        case OP_SET_RECORD_MASK:
        case OP_SET_CHAN_BUDGET:
        case OP_SET_BLOCK_POLICY:
        case OP_FENCE:
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
            return payload_len == 0;
        default:
            return false;
    }
}

static int hs_write_full(int fd, const void* buf, size_t len) {
    const u8* p = (const u8*)buf;
    size_t off = 0;
    struct timespec start, now;
    clock_gettime(CLOCK_MONOTONIC, &start);
    while (off < len) {
        clock_gettime(CLOCK_MONOTONIC, &now);
        long remaining = 250 - ((now.tv_sec - start.tv_sec) * 1000 + (now.tv_nsec - start.tv_nsec) / 1000000);
        if (remaining <= 0) return -1;
        struct pollfd wait = {.fd = fd, .events = POLLOUT};
        int ready = poll(&wait, 1, (int)remaining);
        if (ready < 0 && errno == EINTR) continue;
        if (ready <= 0) return -1;
        ssize_t w = send(fd, p + off, len - off, MSG_NOSIGNAL | MSG_DONTWAIT);
        if (w < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
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
    struct timespec start, now;
    clock_gettime(CLOCK_MONOTONIC, &start);
    while (off < len) {
        clock_gettime(CLOCK_MONOTONIC, &now);
        long remaining = 250 - ((now.tv_sec - start.tv_sec) * 1000 + (now.tv_nsec - start.tv_nsec) / 1000000);
        if (remaining <= 0) return -1;
        struct pollfd wait = {.fd = fd, .events = POLLIN};
        int ready = poll(&wait, 1, (int)remaining);
        if (ready < 0 && errno == EINTR) continue;
        if (ready <= 0) return -1;
        ssize_t r = recv(fd, p + off, len - off, MSG_DONTWAIT);
        if (r < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
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
    ok = hs_try_send(sys, &m, req->payload_len ? payload : NULL, req->payload_len);

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

    int fd = srv->listen_fd;
    atomic_store_explicit(&srv->running, true, memory_order_release);
    while (!atomic_load_explicit(&srv->stop, memory_order_acquire)) {
        struct pollfd wait = {.fd = fd, .events = POLLIN};
        int ready = poll(&wait, 1, 100);
        if (ready < 0 && errno == EINTR) continue;
        if (ready < 0) break;
        if (ready == 0) continue;
        int client = accept(fd, NULL, NULL);
        if (client < 0) {
            if (errno == EINTR || errno == EAGAIN) continue;
            break;
        }
        if (srv->mode == HS_IPC_MODE_TCP) {
            int enabled = 1;
            setsockopt(client, IPPROTO_TCP, TCP_NODELAY, &enabled, sizeof(enabled));
        }
        pthread_mutex_lock(&srv->connection_lock);
        srv->client_fd = client;
        bool stopping = atomic_load_explicit(&srv->stop, memory_order_acquire);
        pthread_mutex_unlock(&srv->connection_lock);
        while (!stopping && !atomic_load_explicit(&srv->stop, memory_order_acquire)) {
            HSIpcHdr h;
            if (hs_read_full(client, &h, sizeof(h)) != 0) break;
            u32 error = h.magic != HS_IPC_MAGIC ? HS_IPC_ERR_BAD_MAGIC :
                        h.version != HS_IPC_VERSION ? HS_IPC_ERR_BAD_VERSION :
                        h.type != HS_IPC_TYPE_REQ ? HS_IPC_ERR_BAD_TYPE : 0;
            if (error) {
                hs_ipc_send_error(client, h.cid, error);
                break;
            }
            if (hs_ipc_handle_req(srv, client, &h) != 0) break;
        }
        pthread_mutex_lock(&srv->connection_lock);
        srv->client_fd = -1;
        close(client);
        pthread_mutex_unlock(&srv->connection_lock);
    }
    atomic_store_explicit(&srv->running, false, memory_order_release);
    return NULL;
}

static bool hs_ipc_start_common(HSIpcServer* srv) {
    int fd;
    if (srv->mode == HS_IPC_MODE_TCP) {
        fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return false;
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
            if (inet_pton(AF_INET, srv->addr, &tcp_addr.sin_addr) != 1) {
                close(fd);
                srv->listen_fd = -1;
                return false;
            }
        }

        if (bind(fd, (struct sockaddr*)&tcp_addr, sizeof(tcp_addr)) != 0) {
            close(fd);
            srv->listen_fd = -1;
            return false;
        }

        if (listen(fd, 4) != 0) {
            close(fd);
            srv->listen_fd = -1;
            return false;
        }
    } else {
        fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd < 0) return false;
        srv->listen_fd = fd;

        struct sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        size_t pathlen = strlen(srv->path);
        if (pathlen >= sizeof(addr.sun_path)) pathlen = sizeof(addr.sun_path) - 1;
        memcpy(addr.sun_path, srv->path, pathlen);
        addr.sun_path[pathlen] = '\0';

        /* Never remove a caller's existing file/socket to claim its name. */
        if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
            close(fd);
            srv->listen_fd = -1;
            return false;
        }
        if (chmod(srv->path, 0600) != 0) {
            close(fd);
            srv->listen_fd = -1;
            unlink(srv->path);
            return false;
        }
        if (listen(fd, 4) != 0) {
            close(fd);
            srv->listen_fd = -1;
            unlink(srv->path);
            return false;
        }
    }

    if (pthread_mutex_init(&srv->connection_lock, NULL) != 0) {
        close(fd);
        srv->listen_fd = -1;
        if (srv->mode == HS_IPC_MODE_UNIX) unlink(srv->path);
        return false;
    }
    srv->lock_inited = true;
    if (pthread_create(&srv->thread, NULL, hs_ipc_thread_main, srv) != 0) {
        close(fd);
        srv->listen_fd = -1;
        if (srv->mode == HS_IPC_MODE_UNIX) unlink(srv->path);
        pthread_mutex_destroy(&srv->connection_lock);
        srv->lock_inited = false;
        return false;
    }
    srv->thread_started = true;
    return true;
}

bool hs_ipc_start(HSIpcServer* srv, HSSystem* sys, const char* path) {
    if (!srv) return false;
    memset(srv, 0, sizeof(*srv));
    srv->sys = sys;
    srv->mode = HS_IPC_MODE_UNIX;
    srv->listen_fd = -1;
    srv->client_fd = -1;
    atomic_init(&srv->running, false);
    atomic_init(&srv->stop, false);
    if (!sys || !path || !path[0] || strlen(path) >= sizeof(srv->path)) return false;
    strncpy(srv->path, path, sizeof(srv->path) - 1);

    return hs_ipc_start_common(srv);
}

bool hs_ipc_start_tcp(HSIpcServer* srv, HSSystem* sys, const char* addr, u16 port) {
    if (!srv) return false;
    memset(srv, 0, sizeof(*srv));
    srv->sys = sys;
    srv->mode = HS_IPC_MODE_TCP;
    srv->port = port ? port : HS_IPC_DEFAULT_PORT;
    srv->listen_fd = -1;
    srv->client_fd = -1;
    atomic_init(&srv->running, false);
    atomic_init(&srv->stop, false);
    if (!sys || (addr && strlen(addr) >= sizeof(srv->addr))) return false;
    if (addr) memcpy(srv->addr, addr, strlen(addr) + 1);

    return hs_ipc_start_common(srv);
}

void hs_ipc_stop(HSIpcServer* srv) {
    if (!srv || !srv->thread_started) return;
    atomic_store_explicit(&srv->stop, true, memory_order_release);
    pthread_mutex_lock(&srv->connection_lock);
    if (srv->client_fd >= 0) shutdown(srv->client_fd, SHUT_RDWR);
    if (srv->listen_fd >= 0) {
        shutdown(srv->listen_fd, SHUT_RDWR);
    }
    pthread_mutex_unlock(&srv->connection_lock);
    pthread_join(srv->thread, NULL);
    srv->thread_started = false;
    close(srv->listen_fd);
    srv->listen_fd = -1;
    if (srv->mode == HS_IPC_MODE_UNIX) unlink(srv->path);
    pthread_mutex_destroy(&srv->connection_lock);
    srv->lock_inited = false;
    atomic_store_explicit(&srv->running, false, memory_order_release);
}
