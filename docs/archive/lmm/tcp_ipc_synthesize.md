# LMM: TCP/IP IPC Server (SYNTHESIZE)

## P0 Spec: TCP/IP Support for IPC Server

### 1. HSIpcServer Changes (hs_ipc.h)
```c
typedef enum {
    HS_IPC_MODE_UNIX,
    HS_IPC_MODE_TCP,
} HSIpcMode;

typedef struct {
    HSSystem* sys;
    pthread_t thread;
    int listen_fd;
    atomic_bool running;
    atomic_bool stop;
    char path[108];       // Unix path OR
    char addr[64];       // TCP addr
    u16 port;            // TCP port
    HSIpcMode mode;      // UNIX or TCP
} HSIpcServer;
```

### 2. hs_ipc_start() Signature Change
```c
bool hs_ipc_start(HSIpcServer* srv, HSSystem* sys, const char* path);  // Unix
bool hs_ipc_start_tcp(HSIpcServer* srv, HSSystem* sys, const char* addr, u16 port);  // TCP
```

### 3. hs_ipc_thread_main() Changes
- Accept both sockaddr_un and sockaddr_in
- Set SO_REUSEADDR for TCP
- Port 8765 default

### 4. CLI Changes (main.c)
- `--ipc-server <path>` - Unix socket (existing)
- `--ipc-port <port>` - TCP port (new)

### 5. neogpu_tool Changes
- `--sock <path>` - Unix socket (existing)
- `--host <addr> --port <port>` - TCP (new)

## Acceptance Criteria
- [ ] Unix socket mode works (backward compatible)
- [ ] TCP mode works: connect, query, disconnect
- [ ] Both modes don't crash
- [ ] All existing tests pass
