# LMM: TCP/IP IPC Server (REFLECT)

## Edge Cases

### 1. Port Already in Use
- TCP: bind() fails - report error, don't crash
- Unix: unlink() before bind

### 2. Firewall / Permission
- Low ports (<1024) may need root
- Use high port (8765) by default

### 3. TCP vs Unix Performance
- Unix: ~10K local IPC req/sec (from benchmarks)
- TCP: ~1-2K expected due to stack overhead
- Keep both options

### 4. Mixed Environment
- Can run both Unix and TCP simultaneously?
- Simpler: one or the other, not both

### 5. Connection Storms
- Max connections: 4 (like Unix listen backlog)
- select() handles multiple, sequential per-connection

### 6. Zombie Sockets
- TCP connections in TIME_WAIT
- Set SO_REUSEADDR

## Error Handling
- bind failure: return false from hs_ipc_start()
- connect failure: client-side, already handled
- Protocol errors: same as Unix (send error, close)

## Backward Compatibility
- Default to Unix socket if no port specified
- Same protocol version
- Existing neogpu_tool works with --sock
