# Pre-freeze development log

- preregistration.md written before v2 implementation and any R9 rollout.
- First contract run `.tmp_teacher_revision_round9_unit_a`: 6 passed, 2 failed. Failure1: synthetic collision fixture omitted required `aim_x` emission field (KeyError in causal beam key). Corrected fixture to actual world emission schema. Failure2: no external geometric region with coverage8..376 caused empty min; v2 now returns no target and records bounded no-safe-candidate UNKNOWN instead of crashing or claiming infeasible. These fixes were before development evaluation and before confirmation freeze.
- First command skipped development because tests failed. No new-confirmation rollout executed.
