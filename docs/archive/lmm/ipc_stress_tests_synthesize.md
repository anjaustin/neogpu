# LMM: IPC Stress Tests (SYNTHESIZE)

## P0 Spec: IPC Red-Team Tests

### Test 1: Flood Test
- **Goal**: Measure server behavior under rapid request flood
- **Setup**: IPC server + stepper thread + flood client
- **Action**: Send 500 OP_QUERY_STATS requests in tight loop without waiting for responses
- **Expected**: 
  - Server should handle gracefully
  - Some requests may timeout (expected due to toolbus overflow)
  - No crashes, no hangs
- **Metrics**: success count, timeout count, total time

### Test 2: Mid-Connection Disconnect
- **Goal**: Verify server handles abrupt disconnect
- **Setup**: IPC server running
- **Action**: Connect, send partial data, close fd
- **Expected**: Server continues running, accepts new connections

### Test 3: Max Payload Test
- **Goal**: Verify server handles max-size payloads
- **Setup**: IPC server
- **Action**: Send OP_SET_RECORD_MASK with 4-byte payload (already max)
- **Expected**: Request succeeds, no crash

### Test 4: Bad Magic Rejection
- **Goal**: Verify error handling for bad protocol
- **Setup**: IPC server
- **Action**: Send header with magic=0xDEADBEEF
- **Expected**: Server sends error response, closes connection cleanly

### Test 5: Connection Timeout (baseline)
- **Goal**: Measure current timeout behavior
- **Setup**: IPC server
- **Action**: Connect but don't send anything, wait 300ms
- **Expected**: Server should still be responsive to new connections (already works via select)

## Implementation Notes
- Add tests to test_ipc() in src/main.c
- Use same test infrastructure (ipc_connect, ipc_write_full, ipc_read_full)
- Check toolbus_dropped counter after flood test
- All tests should pass - server should be resilient

## Acceptance Criteria
- [ ] Flood test: >0 successful requests, no crash
- [ ] Disconnect test: no crash, server alive
- [ ] Max payload: success, no crash
- [ ] Bad magic: error sent, connection closed
- [ ] All existing tests still pass
