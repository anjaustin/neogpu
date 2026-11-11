# LMM: IPC Stress Tests (REFLECT)

## Edge Cases Considered

### 1. Connection Death During Request
- Client writes header but dies before payload
- Server: reads partial, returns error, closes connection
- Test: verify server doesn't crash

### 2. Toolbus Overflow
- Client sends many requests
- Server processing slower than requests
- Toolbus wraps, old entries overwritten
- Client times out waiting for result
- Current behavior: returns HS_IPC_ERR_TIMEOUT
- Risk: client doesn't know if dropped vs still pending

### 3. Zombie Connections
- Client connects but sends nothing
- Server waits in hs_read_full (blocking)
- Current: select() has 200ms timeout, but then loops back
- No connection timeout implemented

### 4. Payload Size Limits
- Max payload: HS_PAYLOAD_SIZE (4096?)
- Client sends hdr->len > actual data
- hs_read_full blocks forever - BAD
- Server should enforce hdr->len match actual read

### 5. Magic/Version Mismatch
- Client sends bad magic: server sends error, closes connection
- Test: verify clean disconnect

## Failure Mode Analysis

| Scenario | Current Behavior | Desired |
|----------|-----------------|---------|
| Flood | Timeouts | Rate limit or error |
| Toolbus full | Drop + timeout | Return specific error |
| Partial write | Block/hang | Timeout + close |
| Bad magic | Error + close | Same |
| Max payload | OK | Same |

## Red Team Test Cases
1. `test_ipc_flood`: Send 1000 requests, measure success rate
2. `test_ipc_disconnect_mid`: Close fd after header, verify no crash
3. `test_ipc_max_payload`: Send max size, verify handled
4. `test_ipc_bad_magic`: Send 0xDEADBEEF, verify error response
5. `test_ipc_timeout`: Don't respond, verify timeout after 250ms
