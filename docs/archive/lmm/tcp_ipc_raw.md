# LMM: TCP/IP IPC Server (RAW)

## Current State
- Unix domain sockets only: `--ipc-server /path/to/sock`
- Local-only, requires filesystem access
- No remote tooling possible

## Need
- TCP listen socket for remote connections
- `-ipc-port <port>` flag
- Keep Unix socket option for local fast path

## Protocol
- Same NGIP protocol over TCP
- Same port number or separate?

## Threat Model
- TCP opens to network - need authentication?
- Rate limiting?
- Connection limits?

## Questions for NODES
- Same protocol or version bump?
- Default port?
- Security: TLS? Auth token?
