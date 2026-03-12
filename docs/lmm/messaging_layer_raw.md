# Raw Thoughts: NeoGPU Messaging Layer

I like the messaging layer a lot. It is small, fast, and easy to reason about in the happy path: a fixed-size queue, a compact message header, and node handlers that drain inboxes. It feels like the right shape for an engine that wants deterministic behavior, capture/replay, and a stable API boundary.

But the more I look at how messages are encoded today, the more I see a coming scaling problem: the ABI is implicit and informal. Some ops treat `payload_idx` as an immediate value (shader id, buffer id, booleans). Other ops treat it as an index into a payload ring. Some ops put extra meaning into `flags`. Payloads are capped at 64 bytes, which is fine until you want larger structured arguments or richer commands. The current scheme works because it is small and mostly single-writer, but the moment multiple producers exist (async thread, asset loader, replay tools), ambiguity and lifetime bugs will appear.

Replay is the biggest red flag: it replays messages by reusing `payload_idx` references into a shared payload buffer. That means "recording" is not a self-contained capture, it is just a log of headers that assumes payload bytes stay valid. It is easy to accidentally invalidate that assumption by allocating more payloads between record and replay, or by changing the allocator.

If I rushed this, I would just keep adding more ad-hoc payload layouts and hope tests catch issues. That would be fast now and expensive later.

Risks/fears:
- We ship more opcodes without a schema and later cannot change the ABI without breaking everything.
- Replay becomes unreliable and stops being a trusted debugging tool.
- Payload truncation (64B) silently corrupts meaning for some commands.

Open questions:
- Do we want the message stream to be a stable external ABI (tooling, recorded traces), or only an internal interface?
- Should messages be "immediate only" whenever possible, reserving payloads for data blobs, or should every op have a typed payload struct?
- How should we handle >64B arguments: chained blocks, blob table, or indirect handles only?

Naive approach I want to avoid:
- Adding special cases ("this opcode uses flags+payload_idx+payload bytes") without a single authoritative spec.
