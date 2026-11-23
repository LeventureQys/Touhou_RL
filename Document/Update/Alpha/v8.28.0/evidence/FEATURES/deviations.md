# v8.28.0 FEATURES — deviations from v8.27.0's executor

`tools/rl/v8u0_features.py` is a **byte-level copy** of `tools/rl/v8t0_densereward.py`
(148,000 B, sha256 `877f4581b010712a…`) with the changes below.  The training
mathematics (BCE warm start, REINFORCE update, moving-average baseline, entropy
coefficient, Adam, grad-clip, the episode loop, the action masks and the death
rule) is **untouched**.

| # | deviation | why |
| --- | --- | --- |
| 1 | `OBS_DIM=106`, `EXPLICIT_DIM=3`, `FEATURE_DIM=109`; `features_of()` returns `(obs_106, obs_109, explicit)` and the **candidate head still consumes the 106-dim block** | the single factor: the bomb head's input; the frozen candidate head must not see a different observation |
| 2 | new `explicit_features()`: `f1 = bombs_left/3`, `f2 = max(budget-decisions_executed,0)/budget`, `f3 = clip(nearest_all_px,0,200)/200`, each clipped to `[0,1]`; `nearest_all_px` = `v800_bomb.nearest_bullet_distances` (the rule's own measurement) | prereg §1.1 |
| 3 | reward back to v8.24.0's sparse form `REWARD_BY_ARM = {"F": (1.0, 0.5)}` | comparability with the published 4.08 % |
| 4 | single arm: `ARMS=("F",)`, `WORK = BASE_WORK` (no `<arm>/` sub-directory) | there is exactly one factor |
| 5 | `--tag` prefix `v8u0-`, work root `.tmp_v8u0_features/`, schema prefix `v8u0-features-` | one unique tag per process |
| 6 | `--mode summary` is now an **alias** of `--mode verdict` (same function) | v8.27.0 had two code paths whose numbers could drift; here the published summary *is* the verdict file |
| 7 | `verdict` adds **P3 = median `first_bomb_decision` > 10** (`_first_bomb_block`, `_feature_block`, `_per_window_first_bomb`) | the round's key behavioural metric |
| 8 | `verdict` adds a **D3 ordering assertion** (`d3_gate.written_before_summary`: the gate's `created_at` must be ≤ the summary's) and answers `S-fork` when it is not | v8.27.0 was once misread as `S-fork` because a summary predated the gate file by 12 s |
| 9 | `verdict` splits bomb usage by outcome (`bomb_usage_survived` / `bomb_usage_never_survived`) | work order §7 |
| 10 | smoke adds `head_input_is_109`, `features_audited`, `features_cover_every_decision`, `features_in_unit_interval`, `first_bombs_logged` and prints the first 5 emissions' `f1/f2/f3` + `nearest` | work order's smoke requirement |
| 11 | v8t0's old single-arm `mode_summary` (with the 20 s P3 semantics) deleted | superseded by prereg §1.2 |
| 12 | `REF_*` tables kept read-only; `P2_REFERENCE` now points at v8.24 D0's 147/3,600 | prereg §1.2 P2 |
| 13 | published names drop the arm suffix: `bomb_head_seed{0,1,2}_d900.pt`, `seed{0,1,2}.jsonl`, `summary.json` | prereg §3 |

## Known gaps (recorded, not fixed)

1. The per-episode **training** jsonl has no `first_bomb_decision` field (v8t0's
   entry never had one and the training-record code was not touched), so
   `train_curve.first_bomb_decision_in_training` has `n = 0`.  P3 is measured on
   the 1,800 evaluation episodes by definition, and cross-checked against the BC
   clone probe (4 windows, all `[0,0,0,0]`) and the "k-th bomb" diagnostic.
2. `f3`'s "no bullets" branch is defined as `1.0` (the far end of the clip); the
   formula has no value there.  Zero evaluation episodes triggered it
   (`no_bullets_rows = 0`).
3. The prereg's verdict table has no row for "P1 holds but P3 does not"; that case
   is mapped to `S-features-partial` and spelled out in `stop_text`.  This round
   took the "P2 holds, P1 does not" row, so the gap was not exercised.
