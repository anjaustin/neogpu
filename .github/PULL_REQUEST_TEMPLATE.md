name: Pull request
description: Propose a change to NeoGPU
body:
  - type: textarea
    id: summary
    attributes:
      label: Summary
      description: What does this change, and why?
    validations:
      required: true

  - type: markdown
    attributes:
      value: |
        ### Verification

        Please check the boxes that apply, and include the evidence in your
        description. "Command success is not outcome success" — for visual or
        behavioral changes, show what a user actually sees.

  - type: checkboxes
    id: verification
    attributes:
      label: Verification performed
      options:
        - label: `make clean && make && make build-tests` succeeds
        - label: `./neogpu_demo` runs clean (headless suite)
        - label: Verified on real hardware (Pi 4) — describe below
        - label: `make ml` succeeds (if the experimental path is touched)
        - label: Benchmarked before and after (if performance is touched)
        - label: Documentation updated alongside the change

  - type: textarea
    id: hardware-verification
    attributes:
      label: Hardware verification details
      description: What you ran on the Pi 4, and what you observed. Include FPS or frame output where relevant.
      placeholder: |
        $ sudo ./tests/test_08_raycast_orbit
        ... 107 FPS ...
      render: text

  - type: textarea
    id: scope
    attributes:
      label: Scope notes
      description: |
        Does this touch the experimental ML path, the message ABI, or the
        public API surface? If so, list what a reviewer should scrutinize.
    validations:
      required: false

  - type: checkboxes
    id: checklist
    attributes:
      label: Checklist
      options:
        - label: No source files were deleted; anything removed was archived instead
        - label: The core build still has no ML dependencies
        - label: New build artifacts are gitignored (by explicit name, not a pattern that could match source)
        - label: `docs/interface/MESSAGE_ABI.md` reflects any opcode changes
