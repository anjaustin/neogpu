# LMM: IPC Stress Tests (NODES)

## Architecture Analysis

### Server Thread Model
- Single-threaded event loop using `select()`
- One connection at a time (sequential request/response)
- `listen(fd, 4)` - backlog of 4 connections
- 200ms select timeout for graceful shutdown

### Toolbus Ring Buffer
- 256 slots (HS_TOOLBUS_SIZE)
- Overflow strategy: drop and count (`toolbus_dropped`)
- Wait uses condition variable with timeout (250ms default)
- Search is linear from head backwards

### Vulnerabilities Identified
1. **Flood**: Single connection can send faster than processing (no backpressure)
2. **Blocking wait**: Server blocks on toolbus_wait, cannot accept new connections
3. **No per-connection limits**: One client can consume all server time
4. **Toolbus overflow**: Old entries silently dropped, client times out

## Design Decisions Needed
- Add flood protection? (rate limit per connection)
- Add connection timeout?
- Add toolbus overflow error response to client?
- Consider non-blocking architecture?

## Test Implementation Plan
1. Flood test: Send N requests, count successes/timeouts
2. Bad断开 test: Close socket mid-request
3. Max payload test: Verify max size handled
4. Quick benchmark: Measure throughput under load
