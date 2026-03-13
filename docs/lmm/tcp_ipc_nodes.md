# LMM: TCP/IP IPC Server (NODES)

## Current Architecture
- `hs_ipc_start()` takes a path, creates Unix socket
- Single-threaded event loop with `select()`
- Protocol: HSIpcHdr + HSIpcReq/HSIpcResp

## Abstraction Needed
Create address family selector:
```c
typedef enum {
    HS_IPC_TYPE_UNIX,
    HS_IPC_TYPE_TCP,
} HSIpcType;

typedef struct {
    HSIpcType type;
    union {
        char unix_path[108];
        char tcp_addr[64];
        u16 tcp_port;
    };
} HSIpcConfig;
```

## Implementation Plan
1. Add `HSIpcConfig` to `HSIpcServer`
2. Add `-ipc-port <port>` to CLI
3. Select socket type based on config
4. Rest of server code stays same (protocol agnostic)

## Port Selection
- Default: 8765 (NeoGPU -> NGPU -> 8765)
- Or use 0 for auto-assign?

## Security Considerations
- No TLS initially (local network only use case)
- Rate limiting in future
- Connection limit (max 4 concurrent like Unix backlog)
