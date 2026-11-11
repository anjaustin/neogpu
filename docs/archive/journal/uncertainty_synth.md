# Synthesis: Operating Method for Uncertainty Resistance

## Objective
Replace premature patching with a lightweight sharpen-first workflow whenever task conditions make local action expensive.

## Trigger Conditions
Use this method before acting when any of the following are true:
- visual output is the true success criterion
- cleanup/build rules may affect source-of-truth files
- the repo is stateful, dirty, or partially broken
- the task touches performance-sensitive hot paths
- success depends on remote state, hardware behavior, or human observation

## Pre-Action Protocol
1. State the real success condition in one sentence.
2. List the invariants that must not be broken.
3. Name what is currently uncertain.
4. Decide what evidence will count as verification.
5. Only then implement the first change.

## Execution Rules
- Prefer restoring invariants before layering fixes.
- Do not treat command success as proof of task success.
- Avoid cleanup patterns that can match source files.
- For fragile work, prefer one clean change over multiple speculative patches.
- If a result is user-visible, validate the visible outcome before reporting confidence.

## Communication Rules
- Report verified facts, not optimistic interpretations.
- If a step failed, say it failed plainly.
- Distinguish between rebuilt, ran, passed, and looked correct.
- When uncertain, describe the uncertainty internally before acting, not externally after breakage.

## Success Criteria
- [ ] I identified the real success criterion before editing.
- [ ] I protected or restored invariants first.
- [ ] I verified the actual outcome, not just command completion.
- [ ] I avoided source-destroying cleanup behavior.
- [ ] My outward report matched verified reality.

## References to Key Insights
- From REFLECT core insight: convert uncertainty into structure before action.
- From resolved tensions: verification must be outcome-level, not command-level.
- From challenged assumptions: reversibility does not make premature action cheap.
