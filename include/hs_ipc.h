#ifndef HS_IPC_H
#define HS_IPC_H

#include "hs_core.h"

#define HS_IPC_DEFAULT_PORT 8765

typedef enum {
    HS_IPC_MODE_UNIX,
    HS_IPC_MODE_TCP,
} HSIpcMode;

/* Local IPC server for NODE_SYSTEM query/control ops. */
typedef struct {
    HSSystem* sys;
    pthread_t thread;
    int listen_fd;
    atomic_bool running;
    atomic_bool stop;
    HSIpcMode mode;
    char path[108]; /* sockaddr_un.sun_path limit */
    char addr[64];  /* TCP bind address */
    u16 port;       /* TCP port */
} HSIpcServer;

bool hs_ipc_start(HSIpcServer* srv, HSSystem* sys, const char* path);
bool hs_ipc_start_tcp(HSIpcServer* srv, HSSystem* sys, const char* addr, u16 port);
void hs_ipc_stop(HSIpcServer* srv);

#endif
