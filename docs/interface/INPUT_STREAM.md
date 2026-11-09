# Input streaming contract

The evdev adapter owns one reader per device (up to eight) and one processing
thread. Readers serialize queue admission with a mutex and signal the consumer.
It is **not** a lock-free multi-device pipeline. The standalone
`hs_input_spsc_*` helpers are SPSC only; the stream's locked submit API supplies
multi-producer safety.

## Lifetime and state ownership

1. `hs_input_stream_init()`; check its return value.
2. Optionally add explicit evdev paths and enable hotplug before starting.
3. `hs_input_stream_start()` installs the hotplug watch before reporting success.
4. `hs_input_stream_get_state()` returns a mutex-protected copy in thread-local
   storage, valid until that calling thread's next `get_state()` call. It is
   not a pointer to a concurrently mutable worker buffer.
5. `hs_input_stream_stop()` joins readers before closing their descriptors.
6. `hs_input_stream_shutdown()` releases synchronization resources. Repeated
   stop/shutdown calls are safe after successful initialization.

Init/start/stop/shutdown and hotplug configuration must be externally serialized.
Submission and snapshot reads can be concurrent while the stream is alive.
Do not destroy a stream while another application thread is using its API.

Held key/button state survives snapshot reads and `swap_buffers()`. A release
event clears it. `process_events()` drains pending events synchronously under
the same lock; `swap_buffers()` is a compatibility alias, not a state reset.
Pause retains queued input; resume wakes the worker. The queue has 255 usable
slots. A rejected submission increments the atomic rejection counter once.

`hs_input_stream_submit()` admits synthetic events through exactly the same
queue/wakeup path as device readers. `make test` exercises four concurrent
producers, queue wrap, pause/resume, and held-state snapshots without devices.

## Device behavior and limitations

- Device descriptors are nonblocking; readers wait with `select()`.
- Added devices use `EVIOCGRAB`: do not point tests at a person's active keyboard.
- Hotplug handles create/delete events and reuses free descriptor slots.
- Removal resets aggregate held state, preventing stuck keys. The aggregate is
  last-event-wins, not a per-device key-reference-count model.
- Capability queries report failed ioctls rather than pretending success.
- Generic absolute-axis mapping assumes X/Y in 0..65535 and RX/RY in
  -32767..32767. Device-specific calibration is not implemented.
- `frame` counts processed event batches, not display frames. `elapsed` is that
  batch count divided by `HS_FPS`, not measured wall-clock runtime.
- This adapter does not automatically emit `NODE_INPUT` messages. The separate
  node handlers are an application-facing message interface.

## Verification

`make test` includes deterministic synthetic concurrency checks.
`tests/test_evdev` is an opt-in privileged integration test. It creates only its
own `/dev/uinput` virtual devices, exercises two real evdev reader threads and
hot-add/remove, then destroys those devices. It never grabs physical devices.

```bash
make tests/test_evdev
sudo ./tests/test_evdev
```
