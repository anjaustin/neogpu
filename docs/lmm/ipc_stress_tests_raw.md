# LMM: IPC Stress Tests (RAW)

## Threat Model
1. **Flood attack**: Client sends rapid IPC requests faster than server can process
2. **Connection exhaustion**: Multiple concurrent connections overwhelm server
3. **Payload bomb**: Client sends max-size payloads to stress memory
4. **Malformed requests**: Invalid headers, wrong sizes, malicious payloads
5. **Resource exhaustion**: Server runs out of toolbus slots, memory, or file descriptors

## Test Scenarios

### Flood Test
- Send 1000+ requests in tight loop without waiting for responses
- Measure: how many succeed, how many fail, server stability
- Expectation: server should handle gracefully with backpressure or error responses

### Concurrent Connections
- Spawn 10+ client threads connecting to same socket
- Each sends queries simultaneously
- Measure: throughput, error rates, server stability

### Large Payload Test
- Send requests with max payload size (HS_PAYLOAD_SIZE)
- Verify server handles without crashing

### Bad Connection Handling
- Abruptly close connections mid-request
- Send data after close
- Verify server doesn't crash

## Current State
- Server uses single-threaded accept loop (one connection at a time)
- Toolbus has 256-slot ring buffer with overflow tracking
- No explicit rate limiting

## Questions for NODES phase
- Should server reject new connections when busy?
- Should server have per-connection timeout?
- Should toolbus drop old entries or reject new when full?
