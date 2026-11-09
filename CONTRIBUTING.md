# Contributing to NeoGPU

Thanks for your interest. NeoGPU is a message-native GPU runtime for ARM/NEON
hardware, with a Raspberry Pi 4 as its primary target.

## Ground rules

These come straight from `AGENTS.md` and are the norms this project works by:

1. **One clean change over many speculative patches.** Prefer a focused commit
   that does one thing over a sweeping refactor that does five.
2. **Don't delete hand-written source during cleanup.** Archive instead. The
   [`docs/archive/`](docs/archive/) tree is the precedent.
3. **Report verified reality, not optimistic interpretation.** "Command success
   is not outcome success." If the change is visual or behavioral, verify what
   a user actually sees.
4. **Write down the uncertainty.** Before acting on shaky ground, note the
   success condition, the invariants that must not break, what is still
   uncertain, and what evidence would count as verification.
5. **Red-team optimizations.** Benchmark and profile before keeping a change.
   Document rejected ideas — the optimization log in
   [`docs/benchmarks/ATOMICS_PROFILE_2026-03-13.md`](docs/benchmarks/ATOMICS_PROFILE_2026-03-13.md)
   records both the wins and the things that didn't work.

## Getting set up

```bash
git clone https://github.com/anjaustin/neogpu.git
cd neogpu
make
```

You'll need an ARM toolchain. On a Raspberry Pi 4, the native toolchain works
directly. For cross-compiling:

```bash
make CC=aarch64-linux-gnu-gcc STRIP=aarch64-linux-gnu-strip build
```

Build dependencies on Debian/Ubuntu:

```bash
sudo apt install build-essential libgles2-mesa-dev libgbm-dev libdrm-dev libegl-dev
```

## Before you open a pull request

**Build the core.** This must always work — the core has no inference
dependencies:

```bash
make clean && make && make build-tests
```

**Run the headless suite.** No display or root needed:

```bash
./neogpu_demo
```

This exercises the message fabric, node handlers, IPC, thread safety,
overflow handling, and the producer benchmark.

The automated gate is `make test` (headless checks, P0 regressions, and real
Unix/TCP CLI contracts). Run `make sanitize` and `make tsan` for memory/undefined
behavior and race checks. `make check-sources` syntax-checks every C source;
`make benchmark` measures performance separately. Privileged display and virtual
evdev checks are opt-in with `sudo make test-hardware`.

**If you touched the graphics path**, verify it on real hardware — the DRM /
GBM backend cannot be meaningfully tested off-device:

```bash
sudo ./tests/test_08_raycast_orbit
```

## Where things live

| Path | What goes here |
|------|----------------|
| `include/` | Public headers. Inline helpers belong in headers. |
| `src/` | Core runtime: graphics and the message substrate. |
| `tests/` | Test and benchmark programs. |
| `tools/` | Standalone demos and utilities. |
| `docs/` | Documentation. See [`docs/README.md`](docs/README.md). |
| `docs/archive/` | Historical material. Keep it; don't delete it. |

Note that `src/` and `tests/` are source-of-truth. `make clean` removes build
artifacts only — if you add a new cleanup pattern, make sure it cannot match
a source file.

## Style

- Match the surrounding code. The project uses 4-space indent, `u8`/`u32`/`f32`
  fixed-width types, and `HS_`-prefixed public symbols.
- Keep the message ABI in sync with
  [`docs/interface/MESSAGE_ABI.md`](docs/interface/MESSAGE_ABI.md) when you add
  or change an opcode.
- The message fabric is the architectural core. Changes to channel semantics,
  QoS, or the lock-free queues need benchmarking and a red-team pass.

## Scope

NeoGPU focuses on the message-native runtime, renderer, input, capture/replay,
and tooling. The former BitNet/ML experiment lives in a separate repository;
model loaders, inference kernels, and ML demos do not belong in this tree.

## Reporting bugs

Open an issue with:

- What you did, what you expected, and what happened.
- Your hardware (Pi 4 revision, OS, compiler version).
- The output of `./neogpu_demo` if the failure is in the message layer.

## License

By contributing, you agree that your contributions are licensed under the
project's MIT license.
