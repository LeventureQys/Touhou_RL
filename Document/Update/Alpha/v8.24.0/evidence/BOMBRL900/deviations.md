# v8.24.0 BOMBRL900 — deviations from the prereg (and things a reader must know)

Prereg: `Document/Update/Alpha/v8.24.0/PREREGISTRATION_BOMBRL900_001.md` (FROZEN).
Runner: `tools/rl/v8p0_bombrl900.py` (a copy of v8.21.0's
`tools/rl/v8m0_bombrl.py` with the four changes the round authorises, see §1).

## 1. The four authorised changes (and nothing else in the protocol)

| # | v8.21.0 | v8.24.0 |
| --- | --- | --- |
| 1 | `TRAIN_DECISIONS = 600` (20 s) | **`TRAIN_DECISIONS = 900`** (30 s = 1,800 frames) |
| 2 | `R = survived(600) + 0.5·alive/1200` | **`R = survived(900) + 0.5·alive/1800`** |
| 3 | work dir `.tmp_v8m0_bombrl`, tags `v8m0-*` | **`.tmp_v8p0_bombrl900`, tags `v8p0-*`**, evidence `v8.24.0/evidence/BOMBRL900` |
| 4 | `RL` + `RL-argmax` + `R0-argmax` arms | **only `RL900`** (bomb sampling); no deterministic arm |

Everything else is the frozen v8.21.0 recipe: `BombHead(106→64→2)`, candidate head
`bombdag.pt` read out by **argmax**, BCE warm start (200 steps, batch 256, 50 %
positives, Adam 1e-2), REINFORCE with a moving-average baseline (0.95), entropy
0.01, Adam 3e-4, grad-clip 1.0, 300 episodes × 900 decisions × 3 seeds, eval at
300/600/900 decisions × 3 seeds × 4 windows × 300 episodes.

## 2. Deviations

1. **P1 also requires the one-sided Fisher p < 0.05.**  The prereg says
   "> 57.73 %, 单侧 Fisher + CP95" without naming an α; the bomb line's
   convention (v8.23.0 P1) is "gate on the rate **and** p < 0.05".  Both
   components are reported separately (`rate_gate`, `fisher_significant`), so a
   reader can apply either reading.
2. **The BCE warm start still uses the published 60/40/20 rule**
   (`v8l0_unified.rule_decision`), exactly as in v8.21.0.  The prereg keeps the
   warm start unchanged; 40/20/5 enters this round **only** as the published
   reference table of v8.23.0 (never re-run here).
3. **BC hyper-parameters not fixed by the prereg** (inherited from v8.21.0):
   Adam 1e-2 and balanced batches (50 % positives).  The rule's positive rate on
   this data is ~4 %, so the balanced batches are not neutral; they are the
   v8.21.0 setting that produced 90.26 % at 20 s.
4. **P4 is applied as a void rule, not only as a report**: a (seed, window) cell
   whose `distinct` (on `(alive_frames, bombs_used, bomb_emits,
   tuple(bomb_decisions))`) is < 10 is excluded from the pooled P1/P3 rate
   (`pooled_valid`), and the pooled rate over all rows is reported beside it.
5. **`distinct` in the pooled table is the prereg's 4-tuple with no window/seed
   in the key**, so identical rollouts in different windows collapse.  The
   per-window sum (`distinct_per_window_sum`) and the per-cell minimum are
   reported next to it so the collapse is visible.
6. **A mid-flight `--mode summary` dry run** (to smoke-test the new summary code)
   wrote `.tmp_v8p0_bombrl900/train_curve.json` from 47–51 training rows at
   17:06.  It was overwritten by the final summary; the final `train_curve.json`
   has 300 rows per seed.
7. **No `--episodes < 300` cell was evaluated**: `--mode eval` refuses anything
   but 300 episodes per window, so every cell in the table is n = 300.

## 3. Engine-process discipline (AGENTS.md, "> 6 engines → wait 5 min")

At **16:57:45** the census found **97 `th08-modern` processes**, all from the
*previous* round's directory (`.tmp_v8m0_bombrl/...`: 18 eval processes of
v8.21.0, 4 parked parents each).  None of them was touched (`pkill` is
forbidden and was not used).  By this round's own **preflight at 17:02:36** the
census was **0**, and the first engine of this round started at **17:02:40**
(4 min 55 s after the 97-engine reading).  The round therefore never ran on top
of the previous batch.

## 4. Audio (AGENTS.md §9.3)

Every mode that starts an engine goes through
`tools/rl/v8p0_bombrl900.py`, which sets `SDL_AUDIODRIVER=dummy`, pops
`DISPLAY`/`WAYLAND_DISPLAY`, and builds each private data dir with
`v400_make_cfg.py` (`musicMode=playSounds=musicVolume=sfxVolume=0`); the launcher
`.tmp_v8p0_bombrl900/launch/_run.sh` repeats the two env operations.  The
`--mode eval-smoke` payload re-inspects the cfg with `v400_make_cfg.py --inspect`
(four fields 0) and `--mode audio-audit` decodes every `th08.cfg` of this round.
