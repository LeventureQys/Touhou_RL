# v8.5.0 DISTILL-001 — distilling R5 into the 36-action policy

Prereg: [`PREREGISTRATION_DISTILL_001.md`](../../PREREGISTRATION_DISTILL_001.md)
(**FROZEN** 2026-11-22, before any v8.5.0 training run).
Executor: [`tools/rl/v850_distill.py`](../../../../../../tools/rl/v850_distill.py)
(written this round; every engine-touching loop is **imported** from
[`tools/rl/v830_bombpol.py`](../../../../../../tools/rl/v830_bombpol.py), never copied).
Status: **the rule targets, the three training runs and the 3 x 3 evaluation grid
ran to completion; every gate passed; all raw data published.**
**This file contains no statistical inference** (prereg §5): counts, means,
medians, quantiles and distributions only. P1–P4 belong to the analysis pass.

---

## 0. One-paragraph answer

The prereg's single factor was applied exactly as written: the **5,166** frozen
v8.3.0 states got an R5 target computed offline from their own `obs` — nearest
bullet distance, then `bombs_remaining` 3 -> 80 px, 2 -> 60 px, 1 -> 40 px,
0 -> never — and three `Policy(106, 64, 36)` checkpoints were trained on
`BCE(36 counterfactual labels) + 0.5 * BCE_bomb(R5)`, with the bomb logit of
candidate `c` defined as `logit(2c+1) - logit(2c)` (the 36 outputs reshaped to
`(18, 2)`). **The first thing the targets show is a hard limit of the frozen
batch: all 5,166 states have `bombs_remaining == 3`**, so R5's 60 px and 40 px
stages have **zero training rows** — the supervision that reached the bomb head
is the 80 px stage only, and it is positive on **3,972 / 5,166 = 76.89 %** of the
states. Training does exactly what that supervision asks: `BCE_bomb` falls
0.637 -> 0.062 and the bomb head's mean predicted probability is **0.792–0.796**
against the rule's 76.89 % positive rate (bomb-decision accuracy vs R5 98.0–98.2 %),
while the 36-way head stays where v8.3.0 left it (89.7 % accuracy). On the live
engine, at **300 / 600 / 900 decisions** the distilled policy walks
**3,596 / 3,600 (99.89 %)**, **558 / 3,600 (15.50 %)** and **101 / 3,600 (2.81 %)**
— against v8.3.0's 99.94 % / 19.44 % / 0.61 % and R5's 100 % / 77.05 % / 19.38 %.
So the raw counts are **worse than v8.3.0 at 600 decisions by 3.94 points, better
at 900 by 2.19 points, and 61.55 points short of the rule at 600**: the supervision
cut the bomb **press** rate from 54.5 % of decisions to 24.9–28.8 % and pushed the
**last honoured bomb** from median 185–189 (v8.3.0) to **206** decisions, but it
did **not** change the shape of the arm: the first bomb is still honoured on
decision 0 in **3,600 / 3,600** episodes and **all three bombs are still spent**
(exhaustion 99.89 / 100 / 100 %). In one line: **the rule was distilled into the bomb head, but
what the bomb head could learn from this batch was "bomb now at 3 bombs", not
"hold the second and third bombs" — because no training state ever has fewer than
three bombs.**

---

## 1. What was run

| item | value |
| --- | --- |
| states | **5,166** (`.tmp_v830_bombpol/labels_bomb36.jsonl`, sha256 `aea2c7ab…deed6e`, 49,602,266 bytes, **read-only**) |
| counterfactual labels | 36 per state, **185,976 total**, H = 120, re-used from v8.3.0 — **no re-collection, no engine** |
| new supervision | R5 (`staged`) on the bomb head: `bombs_remaining` 3 -> **80 px**, 2 -> **60 px**, 1 -> **40 px**, 0 -> never; test is v8.4.0's own **strict** `nearest_all_px < threshold` and `bombs_remaining > 0` |
| loss | `BCEWithLogits(36) + 0.5 * BCEWithLogits(bomb logit, rule_bomb)`, `bomb logit(c) = logit(2c+1) - logit(2c)`; **λ = 0.5 fixed** (prereg §1, no scan) |
| model | `Policy(106, 64, 36)` — `Linear(106,64) -> Tanh -> Linear(64,36)`, the frozen v431 class with its output widened (`v830_bombpol.policy36`, imported) |
| training | Adam 1e-3, batch 256, **200 epochs**, **3 seeds**, no early stopping, dev never read |
| evaluation | **3 budgets (300 / 600 / 900 decisions) x 3 seeds x 4 dev windows x 300 episodes**, `stochastic`, live engine |
| eval loop | `v830_bombpol.run_episode36` (imported): the sampled `bomb` bit is executed as `cand_to_mask(cand, bomb)` on the decision's first frame, plain candidate after |
| engine | `/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern` sha256 `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` (asserted) |
| harness | [`scripts/exec.sh`](scripts/exec.sh) + `commands.tsv` (every command, exit code, wall time) |

The 5,166 states are the four **train** windows (13–16) of the v4.5.0/v6.0.0
DAgger protocol; the evaluation windows are the four **dev** windows (11, 12, 17,
18). Nothing about the split, the decision grid (2 frames), the PARK frame
(8,663) or the episode loop changed.

---

## 2. Gates

| step | command | exit | wall |
| --- | --- | ---: | ---: |
| `preflight` | `v850_distill.py --mode preflight` | **0** | 2.42 s |
| `assert-start` | `v850_distill.py --mode assert-ckpt --phase start` | **0** | 0.85 s |
| `d3-fidelity` | `v432_fork_arm.py --mode fidelity --seeds 12 --learner frozen` | **0** | 6.96 s |
| `targets` | `v850_distill.py --mode targets` | **0** | 3.01 s |
| `train-s0/s1/s2` | `v850_distill.py --mode train --seed {0,1,2}` | **0 / 0 / 0** | 7.60 / 7.55 / 7.81 s |
| `train-summary` | `v850_distill.py --mode train-summary` | **0** | 0.10 s |
| `ckpt-manifest` | `v850_distill.py --mode ckpt-manifest` | **0** | 0.13 s |
| `eval-*` (9 cells, 9 in parallel) | `v850_distill.py --mode eval --decisions D --policy-seed S` | **0 x 9** | 922–1,509 s per cell |
| `diagnostics` | `v850_distill.py --mode diagnostics` | **0** | 4.51 s |
| `audio-audit` | `v850_distill.py --mode audio-audit` | **0** | 0.88 s |
| `process-audit` | `v850_distill.py --mode process-audit --data-dir-filter .tmp_v850_distill` | **0** | 0.17 s |
| `assert-final` | `v850_distill.py --mode assert-ckpt --phase final` | **0** | 1.25 s |
| `publish` | `v850_distill.py --mode publish` | **0** | 13.36 s |
| `verify` | `scripts/verify.py` (independent recomputation) | **0** | see [`verify.json`](verify.json) |

**D3 fidelity (5/5, [`fidelity_d3.json`](fidelity_d3.json))** —
`1_hello_role_shadow`, `2_debug_threads_1`, `3_prefix_2165_checkpoints`
(2,165/2,165, 0 mismatches, final frame 8,663), `4_r24_18_constants` (18/18
`first_death_offset` = R24's table), `5_park_frame_8663` — **all PASS**,
`gate all_pass=True wall=4.652s`, every private data dir with
`cfg muted=True`.

**Checkpoint assertions ([`ckpt_assertions.json`](ckpt_assertions.json) start /
[`ckpt_assertions_final.json`](ckpt_assertions_final.json) final)** — the three
frozen v4.5.0 `round2.pt` (the behaviour-policy source of the 5,166 states), the
engine and the frozen label file all match at both ends of the round:

| item | sha256 | match |
| --- | --- | --- |
| `seed0/round2.pt` | `4abee069d087872e0a2d726cc975850d760e90a705ddcbee60ea91268e5e61dc` | ✅ |
| `seed1/round2.pt` | `a93746bb5d123165e77e0271edfc1efcdb21931e9e2d706e5a22d613c5476ca4` | ✅ |
| `seed2/round2.pt` | `c69960fcbbd888e9b252d3fa685c22a6730f8dd637dcfc9edab45491d218fa01` | ✅ |
| `.tmp_v830_bombpol/labels_bomb36.jsonl` | `aea2c7abee220997caee5c58fe7b4bbad6b4cbeb5d8549d2fc5efad046deed6e` | ✅ |
| engine `th08-modern` | `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` | ✅ |

The imported sources are unchanged — each hashes to the value v8.3.0 pinned (and
v8.3.0's own final assertion recorded): `v830_bombpol.py`
`a8c6d306…c58e1c`, `v450_cf_label.py` `b7ab0ebe…d9947`, `v432_fork_arm.py`
`33235506…0cede`, `v431_arm_a.py` `96923930…45503`, `v431_dense_driver.py`
`1079d93a…fb4f25`, `touhou_rl/v420/contracts.py` `1357c69c…946c1`,
`touhou_rl/v400/gym/actions.py` `a67145b2…93eff`. `tools/rl/v800_bomb.py` is
`5d85bcb0…0d7bf` (the v8.4.0-patched file): this round **never imports it**, so it
is recorded in the assertion, not gated by it.

**Rule-target gate ([`rule_target_stats.json`](rule_target_stats.json))** — the
nearest-bullet distance is computed from the stored observation, so the round
first proves the stored bullet list **is** the one the frozen 106-dim vector was
built from: `v830_bombpol.build_arrays` re-derives every row's
`features_sha256_k12` and **5,166 / 5,166** rows reproduce it, with
**5,166 / 5,166** label vectors valid. In addition,
`nearest_is_first = 5,166` and `sorted_by_distance = 5,166` — the stored list is
the nearest bullets in distance order, so its minimum is the live minimum.

**Audio (AGENTS.md §9.3, [`audio_audit.json`](audio_audit.json))** — every engine
start in this round went through
`v432.build_private_data_dir` (muted `th08.cfg` generated by
`tools/rl/v400_make_cfg.py --out` and `--inspect`ed before the start) with
`SDL_AUDIODRIVER=dummy` and `DISPLAY`/`WAYLAND_DISPLAY` unset. The audit over
**this round's own** data dirs: `cfg_files 38, decoded 38, errors 0, muted 38,
audible 0`, `all_muted: true`.

**Process residue ([`process_residue.json`](process_residue.json),
[`process_residue_final.json`](process_residue_final.json))** — `pgrep -af
th08-modern` filtered by `--data-dir .tmp_v850_distill`: **0** of this round's
engines alive (while 48 engines of other rounds were counted and left alone).
No `pkill -f` was used at any point; the grid was waited for.

---

## 3. The rule targets — and the limit they exposed

[`rule_targets.jsonl`](rule_targets.jsonl) (5,166 rows, [`rule_target_stats.json`](rule_target_stats.json)),
sha256 `a7b788e18687ce7fdab1c99d2b645b44d2499dc35427597285982b825c9fdc83`,
1,794,237 bytes. Every row carries
`index/seed/round/window/decision/frame/nearest_px/bombs_remaining/threshold_px/rule_bomb/reason`
plus the provenance fields `bullets_stored`, `bullet_count`, `nearest_is_first`,
`sorted_by_distance`.

| item | value |
| --- | ---: |
| states | 5,166 |
| `rule_bomb = 1` | **3,972 (76.89 %)** |
| `rule_bomb = 0` | 1,194 (23.11 %) |

| `bombs_remaining` | states | R5 threshold | `rule_bomb = 1` | fraction | nearest px mean | nearest px median |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **3** | **5,166** | 80 px | 3,972 | 76.89 % | 56.8396 | 36.3091 |
| 2 | **0** | 60 px | — | — | — | — |
| 1 | **0** | 40 px | — | — | — | — |
| 0 | **0** | never | — | — | — | — |

`nearest_px` over all 5,166 states: mean 56.8396, median 36.3091, q1 22.9089,
q3 74.5013, min 3.6045, max 194.7099.

**This is the round's structural finding, and it is a property of the frozen
batch, not of the rule.** Every one of the 5,166 states was recorded by replaying
the v4.5.0 DAgger behaviour policy, which has **18 actions and no bomb action** —
so no recorded trajectory ever spent a bomb, and every state sits at
`bombs_remaining = 3`. The prereg's three-stage rule is therefore implemented in
full (the code carries 80/60/40 and would use them), but only its **80 px stage
is ever instantiated**: the 60 px and 40 px stages have zero training rows. What
the bomb head can learn from this batch is "at three bombs, bomb when the nearest
bullet is inside 80 px" — and on these states that is true **76.89 %** of the
time, i.e. nearly always.

---

## 4. Training

| seed | total loss e0 -> e199 | BCE36 e0 -> e199 | BCE_bomb e0 -> e199 | final BCE36 | final BCE_bomb | final total | 36-way acc | bomb-dec. acc vs R5 | ckpt sha256 | wall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 0 | 0.9579 -> 0.2677 | 0.6395 -> 0.2359 | 0.6368 -> 0.0636 | 0.2361 | 0.0619 | 0.2671 | 89.71 % | 98.01 % | `326760dc53d97588…` | 2.91 s |
| 1 | 0.9386 -> 0.2662 | 0.6252 -> 0.2334 | 0.6268 -> 0.0655 | 0.2363 | 0.0596 | 0.2661 | 89.70 % | 98.24 % | `32c38859cd5e4c2b…` | 2.99 s |
| 2 | 0.9563 -> 0.2685 | 0.6375 -> 0.2362 | 0.6377 -> 0.0645 | 0.2369 | 0.0623 | 0.2680 | 89.66 % | 98.01 % | `c997e4a288ae837c…` | 3.16 s |

The three seeds land within 0.05 % of each other on 36-way accuracy and within
0.001 on each final loss term — the same spread v8.3.0 reported. The two loss
parts behave as expected: the counterfactual term reproduces v8.3.0's
`~0.196`-class floor on a **different** loss scale (here 0.236 vs v8.3.0's 0.196,
because λ·BCE_bomb also moves the shared trunk), and the rule term converges to a
small residual (`0.060–0.062`) once the head has learned the 76.89 % base rate.

| `bombs_remaining` | states | R5 threshold | R5 positive rate | mean p(bomb) | share predicted bomb |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 5,166 | 80 px | 76.89 % | 0.7958 / 0.7920 / 0.7951 (seeds 0/1/2) | 78.63 % / 78.29 % / 78.59 % |

Mean p(bomb) is 0.987 on the `rule_bomb = 1` states and 0.149–0.160 on the
`rule_bomb = 0` states (seeds 0/1/2). The head tracks the rule's marginals on this
state set; it has no evidence about the other two stages, and **it cannot even
see them** as separate cases — the `bombs_remaining` feature (index 4 of the
106-dim vector) only ever takes the value 3/8 during training.

Checkpoints stay in the work dir and are registered:
[`ckpt_manifest.json`](ckpt_manifest.json) (`ckpt/seed{0,1,2}/policy36.pt`,
sha256 + bytes), and every train report and eval summary carries the same hash.
The three files are **kept** in `.tmp_v850_distill/ckpt/` — they are not copied
into the evidence tree (they are assets, not documentation).

---

## 5. Evaluation — 3 budgets x 3 seeds x 4 windows x 300 episodes

| budget | episodes | survived | full-rate | alive mean | alive median | q1 | q3 | min | max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | 3,600 | 3,596 | **99.89 %** | 599.9792 | 600.0000 | 600.0000 | 600.0000 | 581 | 600 |
| 600 | 3,600 | 558 | **15.50 %** | 937.0394 | 862.0000 | 834.0000 | 1,062.0000 | 580 | 1,200 |
| 900 | 3,600 | 101 | **2.81 %** | 981.3672 | 863.0000 | 834.7500 | 1,052.2500 | 580 | 1,800 |

| budget | seed | episodes | survived | full-rate | alive mean | alive median | bombs used |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | 0 | 1,200 | 1,199 | 99.92 % | 599.9850 | 600.0000 | 3,600 |
| 300 | 1 | 1,200 | 1,197 | 99.75 % | 599.9525 | 600.0000 | 3,600 |
| 300 | 2 | 1,200 | 1,200 | 100.00 % | 600.0000 | 600.0000 | 3,596 |
| 600 | 0 | 1,200 | 233 | 19.42 % | 944.7308 | 862.5000 | 3,600 |
| 600 | 1 | 1,200 | 143 | 11.92 % | 916.1483 | 854.0000 | 3,600 |
| 600 | 2 | 1,200 | 182 | 15.17 % | 950.2392 | 946.0000 | 3,600 |
| 900 | 0 | 1,200 | 48 | 4.00 % | 1,011.0000 | 864.5000 | 3,600 |
| 900 | 1 | 1,200 | 27 | 2.25 % | 943.2892 | 855.0000 | 3,600 |
| 900 | 2 | 1,200 | 26 | 2.17 % | 989.8125 | 945.0000 | 3,600 |

All 9 cells are `status: complete`, **0** invalid episodes, **0** dropped rows,
1,200 rows per cell. The four windows behave differently and the pattern is the
same as v8.3.0's: window **12** is the hardest (0/300 survivors at 900 decisions
for all three seeds) and window **17** the easiest (12–25 survivors).

| budget | s_w11 | s_w12 | s_w17 | s_w18 (survived / 300, seeds 0/1/2) |
| ---: | --- | --- | --- | --- |
| 300 | 300/300/300 | 299/297/300 | 300/300/300 | 300/300/300 |
| 600 | 42/16/20 | 11/2/6 | 84/72/80 | 96/53/76 |
| 900 | 5/3/3 | 0/0/0 | 25/12/15 | 18/12/8 |

| budget | alive-frames histogram |
| ---: | --- |
| 300 | `<600: 4, <900: 3,596` |
| 600 | `<600: 2, <900: 1,945, <1200: 3,042` |
| 900 | `<600: 4, <900: 1,929, <1200: 3,046, <1800: 3,499, >=1800: 101` |

"alive_frames" is frames executed before the first death (death frame excluded,
R24's rule); a survived episode walked the whole budget, i.e. 600 / 1,200 / 1,800
frames.

---

## 6. Beside v8.3.0 and R5 (no test, no interval)

| budget | **v8.5.0** full-rate | v8.3.0 | Δ vs v8.3.0 | R5 | Δ vs R5 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | **99.89 %** (3,596/3,600) | 99.94 % (3,598/3,600) | **−0.06 pp** | 100.00 % | −0.11 pp |
| 600 | **15.50 %** (558/3,600) | 19.44 % (700/3,600) | **−3.94 pp** | 77.05 % (77.1111 % published) | **−61.55 pp** |
| 900 | **2.81 %** (101/3,600) | 0.61 % (22/3,600) | **+2.19 pp** | 19.38 % (n = 970, quoted) | **−16.57 pp** |

| budget | alive mean (v8.5.0 / v8.3.0) | alive median (v8.5.0 / v8.3.0) | bombs/ep (v8.5.0 / v8.3.0) | exhaustion (v8.5.0 / v8.3.0) |
| ---: | ---: | ---: | ---: | ---: |
| 300 | 599.9792 / 599.9906 | 600.0000 / 600.0000 | 2.9989 / 3.0000 | 99.89 % / 100.00 % |
| 600 | 937.0394 / 990.7286 | 862.0000 / 989.5000 | 3.0000 / 3.0000 | 100.00 % / 100.00 % |
| 900 | 981.3672 / 1,029.8478 | 863.0000 / 992.0000 | 3.0000 / 3.0000 | 100.00 % / 100.00 % |

Read against prereg §2, on **raw counts only** (the Fisher test and the stop
condition are the analysis pass's, not this file's):

- **P1** (600 decisions, `> 19.4444 %`): raw **15.50 %** — **below** the bar, so
  the raw counts are **not** consistent with P1 holding.
- **P2** (gap to R5's 77.05 %, `>= 70 %` counts as distillation success):
  raw **15.50 %** — **61.55 pp short**; not met.
- **P3** (300 / 900 beside v8.3.0's 99.94 % / 0.61 % and R5's 100 % / 19.38 %):
  **99.89 %** and **2.81 %** — level with v8.3.0 at 300, **4.6x** v8.3.0's
  0.61 % at 900, still 16.57 pp below R5.
- **P4** (§7): descriptive only.

**Which stop condition the raw counts are consistent with.** P1 does not hold on
raw counts, which is the prereg's **S-fail** branch ("规则监督与反事实标签冲突").
This file does **not** declare it: the endpoint test (one-sided Fisher) and the
stop-condition call belong to the independent analysis pass. What this file says
is narrower and fully supported by the counts: **at 600 decisions the distilled
policy is 3.94 points *below* the counterfactual-only policy, and at 900 it is
2.19 points above it.**

**A note on the 900-decision reference.** R5's 900-decision number is **quoted
from the v8.5.0 prereg** (19.38 %, n = 970, "收尾中" when the prereg was frozen),
not recomputed here: v8.4.0's own R5 900-decision grid was **running on this same
box** for the first part of this round (§9) and had not published when the
prereg was frozen. Its 600-decision number is the published count
(2,776/3,600 = 77.1111 %, v8.4.0 RUN.md §0), which the prereg rounds to 77.05 %;
both are shown and neither is recomputed.

---

## 7. Bomb behaviour (P4, descriptive only)

| budget | episodes that pressed | bomb presses | honoured (spent) | ignored presses | bombs/ep mean | median | max | histogram | exhaustion | never bombed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 300 | 3,600 / 3,600 | 268,537 | **10,796** | 257,741 | 2.9989 | 3.0000 | 3 | 2: 4, 3: 3,596 | 99.89 % | 0 |
| 600 | 3,600 / 3,600 | 449,479 | **10,800** | 438,679 | 3.0000 | 3.0000 | 3 | 3: 3,600 | 100.00 % | 0 |
| 900 | 3,600 / 3,600 | 509,138 | **10,800** | 498,338 | 3.0000 | 3.0000 | 3 | 3: 3,600 | 100.00 % | 0 |

| budget | press rate (presses / executed decisions) | press rate (presses / budget decisions) | v8.3.0 (executed / budget) |
| ---: | ---: | ---: | ---: |
| 300 | **24.87 %** | 24.86 % | 54.54 % / 54.54 % |
| 600 | **26.61 %** | 20.81 % | 66.11 % / 54.65 % |
| 900 | **28.78 %** | 15.71 % | 67.54 % / 38.70 % |

**The effective bomb timing.** Every press is recorded with the counter it saw;
a press is **honoured** exactly when the counter seen by the next press (or the
episode's `bombs_end`) is one lower. That derivation is written into every
episode record's [`bomb_usage.json`](bomb_usage.json) block and is validated two
ways: it reproduces `bombs_used` **exactly** here (10,796 / 10,800 / 10,800,
**0** mismatched episodes) **and** on v8.3.0's published episodes (10,800 per
budget, **0** mismatched episodes — [`v830_honoured.json`](v830_honoured.json)).

| budget | arm | first honoured decision (mean / median / max) | last honoured decision (mean / median / max) | histogram of the last |
| ---: | --- | ---: | ---: | --- |
| 300 | **v8.5.0** | 0.0000 / 0 / 0 | **205.0753 / 206 / 282** | 100–199: 1,150, >=200: 2,450 |
| 300 | v8.3.0 | 0.0086 / 0 / 1 | 191.0725 / 189 / 264 | 100–199: 1,836, >=200: 1,764 |
| 600 | **v8.5.0** | 0.0003 / 0 / 1 | **205.2406 / 206 / 341** | 100–199: 1,166, >=200: 2,434 |
| 600 | v8.3.0 | 0.0117 / 0 / 2 | 190.7450 / 186 / 283 | 100–199: 1,893, >=200: 1,707 |
| 900 | **v8.5.0** | 0.0000 / 0 / 0 | **205.8450 / 206 / 352** | 100–199: 1,112, >=200: 2,488 |
| 900 | v8.3.0 | 0.0075 / 0 / 1 | 190.4281 / 185 / 260 | 100–199: 1,932, >=200: 1,668 |

What the numbers say, plainly:

- **the first bomb is honoured on decision 0 in 3,600 / 3,600 episodes** at 300
  and 900 decisions (3,599 / 3,600 at 600 — one episode at decision 1). R5 does
  the same for a mechanical reason v8.4.0 recorded: **every episode starts with a
  bullet 21.28 px from the player**, so the 80 px stage fires immediately. On this
  anchor the rule and the counterfactual labels agree, and the supervision cannot
  separate them.
- **all three bombs are still spent** (10,796–10,800 honoured, exhaustion
  99.89 / 100 / 100 %), exactly as in v8.3.0. The rule supervision did not teach
  the policy to hold one back — it could not, because the only stage it ever saw
  says "bomb" on 76.89 % of states and the second/third stage never appears.
- **what it did change is the spacing**: the last honoured bomb lands at median
  **206** (mean 205.1–205.8) against v8.3.0's median **185–189** (mean 190.4–191.1),
  i.e. **17–21 decisions later** and with a heavier upper tail (max 282–352 vs
  260–283). The press rate fell from 54.5–67.5 % of executed decisions to
  24.9–28.8 %, so most of the remaining presses are redundant held-key noise of
  the same kind v8.3.0 already had (98.2 % ignored there, 96–98 % here).
- consequence, measured: at 600 decisions this arm dies **earlier** than v8.3.0
  (median alive 862 vs 989.5) even though its last bomb lands later; at 900
  decisions it survives **4.6x** as often (101 vs 22 episodes) but is still far
  from the rule's 19.38 %.

---

## 8. Deviations and corrections (complete list)

1. **All 5,166 training states have `bombs_remaining == 3`, so R5's 60 px and
   40 px stages have zero training rows.** This is the round's most important
   limitation and it is a property of the frozen batch, not of the rule: the
   v4.5.0 DAgger behaviour policy has no bomb action (18 actions), so no recorded
   trajectory ever spent a bomb and every replayed state sits at 3 bombs. The
   prereg asked for the three-stage rule and the executor implements all three
   (80/60/40, 0 -> never); the data only ever instantiates the 80 px stage. The
   report therefore distinguishes the **rule** (three stages) from the
   **supervision actually delivered** (one stage, positive on 76.89 % of states),
   and no claim is made that the staged shape was distilled.
2. **`v8.4.0`'s R5 900-decision grid shared the box for the first part of this
   round.** When this round started (10:48) three `v800_bomb.py --bomb-rule staged
   --arm R5 --decisions 900` processes were running on the same 20 cores; they
   finished around 11:00–11:04, i.e. during this round's gate phase and the first
   minutes of the eval grid. Load average sat at **8–17** throughout. It explains
   the wall times (§2, 922–1,509 s per cell) and nothing else; R5's numbers are
   **quoted** from the prereg/published RUN.md, never recomputed here.
3. **The prereg quotes R5 600 = 77.05 %; v8.4.0's published count is 77.1111 %
   (2,776/3,600).** Both are reported side by side in §6 and neither is
   recomputed. R5's 300-decision 100 % is the 80 px script rule's published
   3,600/3,600 (v8.0.0 BOMB §4), which is the same arm because R5 also fires on
   decision 0.
4. **The honoured-bomb decisions are derived, not recorded.** v830's episode
   record stores presses, not honours; this round derives the honours from
   `(bomb_decisions, bombs_remaining_at_emit, bombs_end)`. The derivation is
   validated against the counter itself in both rounds — **0** episodes where the
   derived count differs from `bombs_used`, at every budget, on 123,600 episodes
   in total (this round) and 10,800 (v8.3.0). It is still a derived quantity and
   is labelled as one wherever it appears.
5. **`diagnostics`, `audio-audit`, `process-audit`, `ckpt-manifest`,
   `assert-ckpt` and `publish` each ran three times**, not once: the first pass
   wrote the artifacts, the second added the honoured-bomb block, the third added
   the v8.3.0 comparison (`v830_honoured.json`) and the audio `all_pass` alias.
   Each row in `commands.tsv` is one real invocation with its own exit code and
   wall time; only the **last** pass of each is the published artifact set. No
   cell was re-run and no episode was re-collected. The published `commands.tsv`
   is the snapshot closed immediately **before** the final `manifest` / `verify`
   pair, because that pair is what freezes the tree — those last two rows are in
   the work-dir copy (`.tmp_v850_distill/commands.tsv`), which is why this
   directory's log ends one pair short of the closing commands.
6. **`tools/rl/v800_bomb.py` is recorded but not gated.** v8.4.0 patched it after
   v8.3.0 closed (`f884a4be…` -> `5d85bcb0…`), so pinning its hash would fail for
   a reason unrelated to this round. It is listed in every assertion with its
   hash and `"gated": false`, and this round never imports it — `--mode equiv` of
   v8.3.0 was the only thing that did, and this round does not run it.
7. **`git status` on this checkout shows a whole-tree diff that predates the
   round** (deleted `Document/Dialogs/*`, ~equal `+`/`-` counts per file in
   `tools/rl/*`, which is the line-ending signature). This round adds
   `tools/rl/v850_distill.py`, `.tmp_v850_distill/**` and
   `Document/Update/Alpha/v8.5.0/**` and nothing else, and the evidence for the
   protected files is their hash equality with v8.3.0's pinned values (§2), not a
   clean `git status`. No `git add`/`git commit` was run.
8. **The 900-decision arm rests on 101 survivors in total** (48 / 27 / 26 per
   seed). Per prereg §5 the 12 (seed, window) cells per budget are **not**
   independent samples and are not aggregated as if they were; the per-cell table
   in §5 is published so the reader can see the spread (window 12 contributes
   **0** survivors at 900 decisions for all three seeds).
9. **The smoke run (`v850-smoke`) is in `commands.tsv` but not in the grid.** Two
   episodes at 300 decisions on window 11 before the grid, to check that the
   bomb bit reaches the engine and the counter drops; both walked 600 frames and
   spent 3 bombs. It is published as [`logs/v850-smoke.out`](logs/v850-smoke.out)
   and its data dir is inside the audited tree, but it is **not** part of any
   denominator in §5.
10. **The work dir is 73 MB of evidence plus the checkpoints.** The three trained
    `policy36.pt` stay in `.tmp_v850_distill/ckpt/` and are registered by sha256
    in [`ckpt_manifest.json`](ckpt_manifest.json) — they are assets, not
    documentation, and are deliberately not copied into the evidence tree.

---

## 9. The machine during this round

The box is shared. This round overlapped v8.4.0's R5 900-decision grid for its
gate phase and the first minutes of its eval phase, and then had the 20 cores to
itself with its own 9-cell grid (up to 87 `th08-modern` processes of its own, 4
parked parents per cell plus children):

| time (Asia/Shanghai) | what was running | load1 (1-min) |
| --- | --- | --- |
| 10:50 (round start) | v8.4.0 R5 d900 x3 + this round's gates | 4.31 falling from 17.51 |
| 10:54 | this round's 9 cells + v8.4.0's 3 | 13.78 |
| 10:59 | this round's 9 cells + v8.4.0's tail | 15.75 |
| 11:04 | this round's 9 cells only | 13.31 |
| 11:15 | this round's d600/d900 cells | 8.18 |
| 11:18 (grid done) | nothing of this round | — (0 own engines) |

The eval cells' wall times (922–1,509 s for 1,200 episodes each) are summed
per-process times of 9 **parallel** cells, not elapsed work; the grid's elapsed
time was 10:52:35 -> 11:17:44 (25.2 min).

---

## 10. What this round does **not** claim

Prereg §5, in effect and verbatim:

- **not** "学会了策略" — half of the supervision is a hand-written rule, and this
  round shows that only **one of its three stages** ever reached the model;
- the distilled result must **not** be mixed with pure-learning results (this one
  carries a rule prior; v8.3.0 does not);
- **not** "通关" — this is still a single anchored segment (frame 8,663, 300/600/
  900 decisions of a 2-frame decision grid);
- the three budgets are **not** comparable by their means (prereg §5), and this
  file does not compare them that way;
- the 12 (seed, window) cells per budget are **not** independent samples;
- **no statistics**: no Fisher, no permutation, no p-value, no Holm correction,
  no confidence interval, no "significant". P1/P2/P3 are **not** decided here and
  no stop condition (S-ok / S-partial / S-fail) is declared.

Beyond the prereg, three things this file also does not claim:

- **not** "the rule was distilled" — 60 px and 40 px were defined and implemented
  but never trained on, so the staged shape was not delivered to the model;
- **not** "the rule hurts" — the 600-decision shortfall is 3.94 points on raw
  counts, and the 900-decision arm is **better** than v8.3.0's; the honest
  statement is the pair, not either half;
- **not** "λ should be something else" — λ = 0.5 was fixed by the prereg and was
  not scanned; this round cannot say whether another λ would behave differently.

---

## 11. Files in this directory

The table below lists the primary artifacts; [`manifest.json`](manifest.json) is
**authoritative for all 124 published files** — it also covers `RUN.md` itself,
`publish.json`, `process_residue_final.json`, `verify.json` and the `logs/`,
`runs/` and `scripts/` subdirectories. [`verify.json`](verify.json) is the
independent recomputation pass: it re-derives all 5,166 rule targets from the
frozen label file with a **separate** implementation, re-counts every episode
against every summary and against `diagnostics.json`, re-hashes the three
checkpoints against their reports and the manifest, and re-hashes **every** file
the manifest lists. `verify.json` is written **after** `manifest.json` closes, so
it is by construction the one file the manifest cannot cover — the audit stamp
verifies the manifest, not the other way round.

| file | bytes | sha256 |
| --- | ---: | --- |
| `audio_audit.json` | 4,669 | `42c95d801a43ef14…` |
| `bomb_usage.json` | 31,074 | `56f1457d6911d06b…` |
| `ckpt_assertions.json` | 4,419 | `ebeba2a282f5fc00…` |
| `ckpt_assertions_final.json` | 4,419 | `9a6aec5496006fd8…` |
| `ckpt_manifest.json` | 1,715 | `4e84cb275c4f4281…` |
| `commands.tsv` | see `manifest.json` | one row per invocation of the table in §2 (this file cannot quote its own log's hash: the log grows as the round closes) |
| `diagnostics.json` | 40,410 | `2d321a9a8d5cefc6…` |
| `eval_d300_seed0.jsonl` | 6,330,256 | `6d8261e05836413c…` |
| `eval_d300_seed0.summary.json` | 5,998 | `0a1f8782b3c1af83…` |
| `eval_d300_seed1.jsonl` | 5,959,352 | `404161b75d8618a6…` |
| `eval_d300_seed1.summary.json` | 5,991 | `43f5436eab1f6861…` |
| `eval_d300_seed2.jsonl` | 5,446,991 | `afd2662d853c3cd4…` |
| `eval_d300_seed2.summary.json` | 5,980 | `cc8e4a5b8edec446…` |
| `eval_d600_seed0.jsonl` | 9,273,352 | `2facea1d18420008…` |
| `eval_d600_seed0.summary.json` | 6,055 | `34fe9545df40b79e…` |
| `eval_d600_seed1.jsonl` | 8,891,953 | `2cefb62f184198ff…` |
| `eval_d600_seed1.summary.json` | 6,078 | `13170b5cd8ec73fd…` |
| `eval_d600_seed2.jsonl` | 8,432,507 | `1108b6e68939c825…` |
| `eval_d600_seed2.summary.json` | 6,079 | `6f44b7e4dafaed20…` |
| `eval_d900_seed0.jsonl` | 10,356,076 | `3fa5fb0f7af03f38…` |
| `eval_d900_seed0.summary.json` | 6,065 | `35f17b706354554b…` |
| `eval_d900_seed1.jsonl` | 9,466,220 | `00bf021c39b362f4…` |
| `eval_d900_seed1.summary.json` | 6,060 | `dddbd521e62c289a…` |
| `eval_d900_seed2.jsonl` | 9,309,992 | `bbf9009f2ff702c4…` |
| `eval_d900_seed2.summary.json` | 6,069 | `390e247b6ba63b11…` |
| `fidelity_d3.json` | 26,626 | `ab63c2b161558680…` |
| `preflight.json` | 4,426 | `b45946184eae79b3…` |
| `process_residue.json` | 614 | `9dd12d284f8a3d25…` |
| `rule_target_stats.json` | 2,784 | `310564f0d779e29c…` |
| `rule_targets.jsonl` | 1,794,237 | `a7b788e18687ce7f…` |
| `train_seed0.json` | 12,178 | `36083caacf7c5f42…` |
| `train_seed1.json` | 12,186 | `7687b627dd2145f7…` |
| `train_seed2.json` | 12,165 | `fe21584fbf430933…` |
| `train_summary.json` | 3,065 | `e85dd0a0dbf84724…` |
| `v830_honoured.json` | 5,648 | `e892264c0bc8cee7…` |
| `scripts/verify.py` | 10,848 | `da08cbb7904c1a00…` |
| `scripts/tables.py` | 10,302 | `56842d2c4e538b07…` |
| `scripts/v830_honoured.py` | 7,346 | `b0ff26a530379b6d…` |
| `RUN.md` | this file | the manifest covers it |
| `manifest.json` | sha256 + bytes of every file above | re-generated **after** `RUN.md` was copied in |
