#ifndef HS_IPC_H
#define HS_IPC_H

#include "hs_core.h"

/* Local IPC server for NODE_SYSTEM query/control ops. */
typedef struct {
    HSSystem* sys;
    pthread_t thread;
    int listen_fd;
    atomic_bool running;
    atomic_bool stop;
    char path[108]; /* sockaddr_un.sun_path limit */
} HSIpcServer;

bool hs_ipc_start(HSIpcServer* srv, HSSystem* sys, const char* path);
void hs_ipc_stop(HSIpcServer* srv);

#endif
