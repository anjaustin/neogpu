# AGENTS.md

## Grounded Working State

- Do not paper over uncertainty with motion.
- When a task is fragile, visual, stateful, destructive, or performance-sensitive, sharpen first.
- Before acting on uncertain ground, write down:
  - the real success condition
  - the invariants that must not break
  - what is still uncertain
  - what evidence will count as verification
- Report verified reality, not optimistic interpretation.
- Command success is not outcome success.
- If the real artifact is visual or behavioral, verify the visible or behavioral result before claiming success.

## Repo Safety Rules

- Do not delete hand-written source files during cleanup.
- Treat `tests/*.c`, `tests/*.h`, `src/*.c`, `src/*.h`, `include/*.h`, `tools/*.c`, and `tools/*.h` as source-of-truth files unless a human explicitly asks for deletion.
- `make clean` and any ad-hoc cleanup commands must remove only build artifacts, never source files.
- Safe cleanup targets include:
  - `*.o`
  - `*.d`
  - built executables
  - generated capture/profile outputs like `gmon.out`
- Before changing cleanup rules, verify that patterns do not match source files.
- If a cleanup pattern could match both binaries and source files, do not use it; replace it with an explicit list of generated outputs.

## File Placement

- Do not write test programs, scratch files, or one-off diagnostics to .
- All test programs and diagnostic tools belong in  or  under the project root.
- This keeps the working tree self-contained, version-trackable, and safe from OS eviction.
- When writing a new test or probe to the Pi, SCP it directly into  or , compile in-tree, and add it to git if it is worth keeping.

## File Placement

- Do not write test programs, scratch files, or one-off diagnostics to /tmp/.
- All test programs and diagnostic tools belong in tests/ or tools/ under the project root.
- This keeps the working tree self-contained, version-trackable, and safe from OS eviction.
- When writing a new test or probe to the Pi, place it in ~/001/neogpu/tests/ or tools/, compile in-tree, and add to git if it is worth keeping.

## Execution Discipline

- Restore invariants before layering fixes on top of broken state.
- Prefer one clean change over multiple speculative patches.
- If a step fails, say it failed plainly and correct the actual cause.
- Do not treat reversibility as permission to be careless; undoing confusion is harder than undoing code.

## Tests

- Preserve test sources even if they are currently broken or incomplete.
- When rebuilding tests, prefer fixing build rules over recreating deleted sources.
- If a test must be reconstructed, document that fact clearly in commit messages or repo docs.
- If a test is user-visible, validate what appears on screen or in behavior, not only its exit status.

## Performance Work

- Red-team every optimization with benchmarks and profiling before keeping it.
- Revert benchmark regressions and document rejected ideas.
- Favor low-risk structural or arithmetic wins over clever heuristics that add coordination or branch complexity.
- Only keep optimizations that survive both measurement and correctness checks.
