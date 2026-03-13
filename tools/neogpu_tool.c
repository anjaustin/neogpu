#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "hs_core.h"
#include "hs_msg.h"

enum {
    NGIP_MAGIC = 0x4E474950u,
    NGIP_VERSION = 1,
    NGIP_TYPE_REQ = 1,
    NGIP_TYPE_RESP = 2,
};

typedef struct {
    u32 magic;
    u16 version;
    u16 type;
    u32 len;
    u32 cid;
} __attribute__((packed)) NgipHdr;

typedef struct {
    u8 op;
    u8 flags;
    u16 reserved;
    u32 payload_len;
} __attribute__((packed)) NgipReq;

typedef struct {
    u32 status;
    u32 result_op;
    u32 result_len;
} __attribute__((packed)) NgipResp;

static void usage(const char* argv0) {
    printf("neogpu_tool\n\n");
    printf("Usage:\n");
    printf("  %s --sock <path> <cmd> [args]\n\n", argv0 ? argv0 : "neogpu_tool");
    printf("Commands:\n");
    printf("  query-stats\n");
    printf("  query-fabric\n");
    printf("  set-record-mask <mask>\n");
    printf("  set-budget <ch> <budget>\n");
    printf("  set-block <ch> <0|1>\n");
    printf("  fence <ch> <cid>\n");
    printf("\nChannel ids: 1=RT 2=RENDER 3=TELEM\n");
    printf("Default socket: /tmp/neogpu.sock\n");
}

static int write_full(int fd, const void* buf, size_t len) {
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

static int read_full(int fd, void* buf, size_t len) {
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

static int connect_sock(const char* path) {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
    if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static int ngip_request(int fd, u8 op, u8 flags, u32 cid, const void* payload, u32 payload_len,
                        u8* out, u32* out_len, u32 out_cap) {
    if (!out || !out_len) return -1;
    *out_len = 0;

    NgipHdr h = {
        .magic = NGIP_MAGIC,
        .version = NGIP_VERSION,
        .type = NGIP_TYPE_REQ,
        .len = (u32)(sizeof(NgipReq) + payload_len),
        .cid = cid,
    };
    NgipReq r = {
        .op = op,
        .flags = flags,
        .reserved = 0,
        .payload_len = payload_len,
    };

    if (write_full(fd, &h, sizeof(h)) != 0) return -1;
    if (write_full(fd, &r, sizeof(r)) != 0) return -1;
    if (payload_len) {
        if (write_full(fd, payload, payload_len) != 0) return -1;
    }

    NgipHdr rh;
    if (read_full(fd, &rh, sizeof(rh)) != 0) return -1;
    if (rh.magic != NGIP_MAGIC || rh.version != NGIP_VERSION || rh.type != NGIP_TYPE_RESP) return -1;
    if (rh.len < sizeof(NgipResp)) return -1;

    NgipResp resp;
    if (read_full(fd, &resp, sizeof(resp)) != 0) return -1;
    u32 remain = rh.len - (u32)sizeof(resp);
    if (resp.result_len != remain) {
        /* protocol mismatch */
        if (remain) {
            /* drain */
            u8 tmp[256];
            while (remain) {
                u32 n = remain;
                if (n > sizeof(tmp)) n = (u32)sizeof(tmp);
                if (read_full(fd, tmp, n) != 0) break;
                remain -= n;
            }
        }
        return -1;
    }

    if (resp.status != 0) {
        if (remain) {
            u8 tmp[256];
            while (remain) {
                u32 n = remain;
                if (n > sizeof(tmp)) n = (u32)sizeof(tmp);
                if (read_full(fd, tmp, n) != 0) break;
                remain -= n;
            }
        }
        fprintf(stderr, "error: status=%u\n", (unsigned)resp.status);
        return -1;
    }

    if (resp.result_len > out_cap) return -1;
    if (resp.result_len) {
        if (read_full(fd, out, resp.result_len) != 0) return -1;
    }
    *out_len = resp.result_len;
    return 0;
}

static void print_stats(const u8* payload, u32 len) {
    u32 tick = 0, log_head = 0, record_mask = 0, prod = 0, flags = 0;
    u32 budgets[3] = {0}, dropped[4] = {0};
    if (!hs_unpack_result_system_stats(payload, len, &tick, &log_head, &record_mask, budgets, dropped, &prod, &flags)) {
        printf("stats: <decode failed>\n");
        return;
    }
    printf("stats: tick=%u log_head=%u prod=%u record_mask=0x%08x\n",
           (unsigned)tick, (unsigned)log_head, (unsigned)prod, (unsigned)record_mask);
    printf("budgets: rt=%u render=%u telem=%u\n",
           (unsigned)budgets[0], (unsigned)budgets[1], (unsigned)budgets[2]);
    printf("dropped: error_ex=%u queue_full=%u system_nonrt=%u result=%u\n",
           (unsigned)dropped[0], (unsigned)dropped[1], (unsigned)dropped[2], (unsigned)dropped[3]);
    printf("flags: recording=%u validate=%u block_rt=%u block_render=%u block_telem=%u\n",
           (unsigned)((flags >> 0) & 1u), (unsigned)((flags >> 1) & 1u),
           (unsigned)((flags >> 2) & 1u), (unsigned)((flags >> 3) & 1u), (unsigned)((flags >> 4) & 1u));
}

static void print_fabric(const u8* payload, u32 len) {
    u32 spsc_ok3[3] = {0}, spsc_full3[3] = {0}, mpsc_ok3[3] = {0}, submit_full3[3] = {0};
    u32 prod = 0, waiters = 0;
    if (!hs_unpack_result_fabric(payload, len, spsc_ok3, spsc_full3, mpsc_ok3, submit_full3, &prod, &waiters)) {
        printf("fabric: <decode failed>\n");
        return;
    }
    printf("fabric: prod=%u waiters=%u\n", (unsigned)prod, (unsigned)waiters);
    printf("spsc_ok:    rt=%u render=%u telem=%u\n", (unsigned)spsc_ok3[0], (unsigned)spsc_ok3[1], (unsigned)spsc_ok3[2]);
    printf("spsc_full:  rt=%u render=%u telem=%u\n", (unsigned)spsc_full3[0], (unsigned)spsc_full3[1], (unsigned)spsc_full3[2]);
    printf("mpsc_ok:    rt=%u render=%u telem=%u\n", (unsigned)mpsc_ok3[0], (unsigned)mpsc_ok3[1], (unsigned)mpsc_ok3[2]);
    printf("submit_full:rt=%u render=%u telem=%u\n", (unsigned)submit_full3[0], (unsigned)submit_full3[1], (unsigned)submit_full3[2]);
}

int main(int argc, char** argv) {
    const char* sock = "/tmp/neogpu.sock";
    int argi = 1;
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }

    if (argi < argc && strcmp(argv[argi], "--sock") == 0) {
        if (argi + 1 >= argc) {
            fprintf(stderr, "missing value for --sock\n");
            return 2;
        }
        sock = argv[argi + 1];
        argi += 2;
    }

    if (argi >= argc) {
        usage(argv[0]);
        return 2;
    }

    const char* cmd = argv[argi++];
    int fd = connect_sock(sock);
    if (fd < 0) {
        fprintf(stderr, "connect failed: %s\n", strerror(errno));
        return 1;
    }

    u8 out[HS_TOOLBUS_PAYLOAD_MAX];
    u32 out_len = 0;
    u32 cid = 1;

    if (strcmp(cmd, "query-stats") == 0) {
        if (ngip_request(fd, OP_QUERY_STATS, 0, cid, NULL, 0, out, &out_len, sizeof(out)) != 0) return 1;
        print_stats(out, out_len);

    } else if (strcmp(cmd, "query-fabric") == 0) {
        if (ngip_request(fd, OP_QUERY_FABRIC, 0, cid, NULL, 0, out, &out_len, sizeof(out)) != 0) return 1;
        print_fabric(out, out_len);

    } else if (strcmp(cmd, "set-record-mask") == 0) {
        if (argi >= argc) {
            fprintf(stderr, "missing mask\n");
            return 2;
        }
        u32 mask = (u32)strtoul(argv[argi++], NULL, 0);
        u8 p[4];
        hs_pack_set_record_mask(p, mask);
        if (ngip_request(fd, OP_SET_RECORD_MASK, 0, cid, p, sizeof(p), out, &out_len, sizeof(out)) != 0) return 1;
        u32 oldv = 0, newv = 0;
        if (hs_unpack_u32x2(out, out_len, &oldv, &newv)) {
            printf("set-record-mask: 0x%08x -> 0x%08x\n", (unsigned)oldv, (unsigned)newv);
        }

    } else if (strcmp(cmd, "set-budget") == 0) {
        if (argi + 1 >= argc) {
            fprintf(stderr, "missing args\n");
            return 2;
        }
        u8 ch = (u8)strtoul(argv[argi++], NULL, 0);
        u32 bud = (u32)strtoul(argv[argi++], NULL, 0);
        u8 p[8];
        hs_pack_set_chan_budget(p, ch, bud);
        if (ngip_request(fd, OP_SET_CHAN_BUDGET, 0, cid, p, sizeof(p), out, &out_len, sizeof(out)) != 0) return 1;
        u32 oldv = 0, newv = 0;
        if (hs_unpack_u32x2(out, out_len, &oldv, &newv)) {
            if (oldv == 0xFFFFFFFFu || newv == 0xFFFFFFFFu) printf("set-budget: invalid channel %u\n", (unsigned)ch);
            else printf("set-budget: ch=%u %u -> %u\n", (unsigned)ch, (unsigned)oldv, (unsigned)newv);
        }

    } else if (strcmp(cmd, "set-block") == 0) {
        if (argi + 1 >= argc) {
            fprintf(stderr, "missing args\n");
            return 2;
        }
        u8 ch = (u8)strtoul(argv[argi++], NULL, 0);
        u8 v = (u8)strtoul(argv[argi++], NULL, 0);
        u8 p[2];
        hs_pack_set_block_policy(p, ch, v);
        if (ngip_request(fd, OP_SET_BLOCK_POLICY, 0, cid, p, sizeof(p), out, &out_len, sizeof(out)) != 0) return 1;
        u32 oldv = 0, newv = 0;
        if (hs_unpack_u32x2(out, out_len, &oldv, &newv)) {
            if (oldv == 0xFFFFFFFFu || newv == 0xFFFFFFFFu) printf("set-block: invalid channel %u\n", (unsigned)ch);
            else printf("set-block: ch=%u %u -> %u\n", (unsigned)ch, (unsigned)oldv, (unsigned)newv);
        }

    } else if (strcmp(cmd, "fence") == 0) {
        if (argi + 1 >= argc) {
            fprintf(stderr, "missing args\n");
            return 2;
        }
        u8 ch = (u8)strtoul(argv[argi++], NULL, 0);
        cid = (u32)strtoul(argv[argi++], NULL, 0);
        if (ngip_request(fd, OP_FENCE, ch, cid, NULL, 0, out, &out_len, sizeof(out)) != 0) return 1;
        u32 tick = 0;
        u8 out_ch = 0;
        if (hs_unpack_result_fence(out, out_len, &tick, &out_ch)) {
            printf("fence: cid=%u tick=%u channel=%u\n", (unsigned)cid, (unsigned)tick, (unsigned)out_ch);
        }
    } else {
        usage(argv[0]);
        return 2;
    }

    close(fd);
    return 0;
}
