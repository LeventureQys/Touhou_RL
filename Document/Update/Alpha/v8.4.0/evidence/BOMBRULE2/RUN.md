# v8.4.0 BOMBRULE2-001 — six bomb rules at 20 s: raw data, publication, recompute

Prereg: [`PREREGISTRATION_BOMBRULE2_001.md`](../../PREREGISTRATION_BOMBRULE2_001.md)
(**FROZEN** 2026-11-22, before any v8.4.0 result; 79 lines, 3,703 bytes).
Runner: `tools/rl/v800_bomb.py`, sha256 `5d85bcb016a0eebb…c0d7bf` — published byte for byte as
[`refs/v800_bomb_used.py`](refs/v800_bomb_used.py); the pre-patch file it was diffed against is
[`ab/v800_bomb_pre.py`](ab/v800_bomb_pre.py), the unified diff
[`refs/v840_patch_vs_pre.diff`](refs/v840_patch_vs_pre.diff).

**Who wrote this file.** The round's executor finished the measurement (all 21 cells of raw data are
on disk) and then timed out during finalisation, so nothing was published. This `RUN.md` is written by
the **publishing pass** afterwards. Every number below is recomputed offline from the published copies
by [`scripts/publish_v840.py`](scripts/publish_v840.py) and re-verified against the runner's own records
by [`scripts/crosscheck_v840.py`](scripts/crosscheck_v840.py); **no number is copied from the executor's
prose**, and **no engine was started by this pass** (`diagnostics.json::engine_started = false`).
Where this file states what the executor measured, it cites the on-disk artefact, not a claim.

Status: **600-decision grid complete — 18/18 cells, 1,200 rows each, 0 invalid; all gates passed.
900-decision R5 cell complete — 3/3 seeds, 1,200 rows each (it finished *during* this publishing pass,
§2.2). The prereg's P1 test is run here (§4); no other test is claimed.**

---

## 0. One-paragraph answer

At the frozen 600-decision (20 s) budget, on the same 4 windows × 300 episodes × 3 seeds = 3,600
episodes per arm, the 80 px control **R0 reproduces v8.1.0's 71.1111 % exactly** (2,560/3,600; all
3,600 episode records identical to v8.1.0's A3 file on 11 behavioural fields, §8) and **two variants beat
it**: **`staged` (R5) 2,776/3,600 = 77.1111 %** (+6.0000 pp, one-sided Fisher p = 3.53e−09, Holm
1.77e−08) and **`double60` (R3) 2,726/3,600 = 75.7222 %** (+4.6111 pp, p = 5.32e−06, Holm 2.13e−05).
`conserve` (R1) is **worse** (2,428/3,600 = 67.4444 %, −3.6667 pp; two-sided Holm 8.17e−04), and the two
rules that restrict *when* the rule may look collapse: `delay200` (R2) **620/3,600 = 17.2222 %**
(−53.8889 pp) and `wall` (R4) **929/3,600 = 25.8056 %** (−45.3055 pp). The usage data say why: the parked
state already has its nearest bullet **18.3–36.4 px** away (one value per dev window, §5), so R0/R1/R5
fire on **decision 0** and spend **2.98 / 2.99 / 2.94 of the 3 bombs**, while R2 (0.49/ep, 9.19 %
exhausted) and R4 (1.01/ep, 29.56 %) **die with bombs in the pocket** — "save bombs for later" is the
wrong direction at 20 s. The two winners do not save bombs: they spend the same three and **place the
last one later** (median last press 458 for both vs R0's 433) by making the *later* bombs stricter.
At 900 decisions (30 s) R5 survives **634/3,600 = 17.6111 %** (seeds 225 / 409 / 0) against R0's
recorded **13.3333 %** — descriptive only (prereg §2 P3), and **seed 2 is still 0 %**: R5 buys the mean,
not the tail. All 334,517 presses of the grid were re-checked against their own rule clause with
**0 violations** (§7).

---

## 1. What was run

| arm | `--bomb-rule` | rule (prereg §1) | episodes | decisions | windows | seeds |
| --- | --- | --- | --- | --- | --- | --- |
| **R0** | `r80` | nearest bullet < 80 px (v8.1.0 A3 control) | 3,600 | 600 | 11, 12, 17, 18 | 0, 1, 2 |
| **R1** | `conserve` | 80 px, widened to **120 px** while `bombs_remaining == 1` | 3,600 | 600 | same | same |
| **R2** | `delay200` | **no bomb** while `decision < 200`, then 80 px | 3,600 | 600 | same | same |
| **R3** | `double60` | only when **>= 2 bullets** are within **60 px** | 3,600 | 600 | same | same |
| **R4** | `wall` | nearest **<= 80 px** **and** `x < 60` or `x > 340` | 3,600 | 600 | same | same |
| **R5** | `staged` | 3 bombs → 80 px, 2 → 60 px, 1 → 40 px | 3,600 | 600 | same | same |

- policy: the frozen v4.5.0 `S` checkpoint (`round2.pt`), `stochastic` (`softmax(z/1.0)` +
  `torch.multinomial`, one generator per (arm, seed) process); the three checkpoint sha256 are recorded
  in `ckpt_assertions_start.json` (`4abee069…`, `a93746bb…`, `c69960fc…`, all asserted at start);
- state arm only (no pixels): PARK at frame **8663**, `SPAWN`/`CHILD`/`QUIT`, **300 episodes per
  (seed, window)**, **2 frames per decision** (`decision_frames: 2` on all 21,600 rows);
- engine `/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`, sha256 `691bfd77…8ef6`;
- **the only code change** is `rule_gate()` resolving each v8.4.0 rule's per-decision threshold and
  gating; distance test, `bombs_remaining > 0` gate, the "first frame of the decision, later frames
  `bomb=0`" mask schedule and every metric are unchanged. The six pre-existing rules
  (`none`/`r40`/`r60`/`r80`/`lethal3`/`lethal5`) never enter `rule_gate`;
- 900-decision run: the best arm at 600 decisions (R5), same 4 windows × 300 episodes per seed.

### 1.1 Gates and their exit codes (recorded in `refs/commands/*.tsv` and `logs/`)

| step | command | exit | wall |
| --- | --- | --- | --- |
| ckpt/engine assertions (start) | `v800_bomb.py --mode assert-ckpt --phase start` | **0** | 0.622 s |
| **D3 fidelity** | `v432_fork_arm.py --mode fidelity --seeds 12 --learner frozen` | **0** | 8.108 s |
| **A/B before** (pre-patch runner, 60 episodes) | `v800_bomb.py --mode eval --bomb-rule r80 --decisions 600 --episodes 15` | **0** | 105.387 s |
| **A/B after** (live runner, same command) | same | **0** | 92.642 s |
| **A/B compare** | `scripts/ab_check.py` | **0** | `ab_check.json`: **60/60 episodes identical**, `all_pass: true` |
| smoke R1 / R2 / R3 / R4 / R5 (4 episodes × 600 decisions) | `v800_bomb.py --mode smoke --bomb-rule …` | **0 / 7 / 7 / 0 / 0** | 13.493 / 11.702 / 13.724 / 12.585 / 13.466 s |
| **18 d600 cells** | `v800_bomb.py --mode eval …` | **0 ×18** (one launch retried, §9.3) | 688.327–2019.128 s per cell |
| **3 d900 cells** | same, `--decisions 900` | **0** recorded for seed 2; **not recorded** for seeds 0/1 (§9.2) | runner's own clock 1382.501 / 2124.268 / 2185.910 s; the harness clock for seed 2 is 1451.188 s |
| ckpt/engine assertions (final) | `--mode assert-ckpt --phase final` | **not run** (executor timed out) | — |
| `finalize.sh` | `bash finalize.sh` | **not run** (no artefact of it exists) | — |

**D3 fidelity**: `fidelity_d3.json`, verdict `PASS` — 2,165/2,165 prefix checkpoints, 0 mismatches,
final frame 8663.
**A/B**: adding `--bomb-rule` did not move the existing rule: 60/60 episodes identical after normalising
away `wall_s`, `spawn_ms`, `handle`, `quit_reply`, `parent_alive`; the only extra fields are the seven
new record fields.
**Mute (AGENTS.md §9)**: every cell logs the cfg check at every window start. The 93
`cfg muted=` lines in `logs/*.out` (84 from the 21 eval cells, 8 from the A/B before/after runs, 1 from
the D3 fidelity harness run) are **93 muted=True, 0 muted=False**. No cell wrote an audible cfg, and this
pass started no engine at all.

---

## 2. Completeness, integrity, and the d900 timeline

### 2.1 The 21 data files

| file (18 × d600 + 3 × d900) | rows | expected | complete |
| --- | ---: | ---: | --- |
| `eval_R{0..5}_d600_seed{0,1,2}.jsonl` — 18 files | 1,200 each | 1,200 | **18/18 yes** |
| `eval_R5_d900_seed{0,1,2}.jsonl` — 3 files | 1,200 each | 1,200 | **3/3 yes** |

- 21 files, 21,600 d600 episodes + 3,600 d900 episodes; **0 unparsable lines, 0 blank lines, 0
  `invalid: true` rows**; `manifest.json` records bytes, sha256 and line count for every file;
- every copy is byte-identical to its source: `source_sha256 == copy_sha256` on all 21
  (`manifest.json → files[].source_sha256_equals_copy_sha256`), and the copy sha256 also equals the
  **runner's own** `out_sha256` in its `summary.json` (§2.3);
- the d600 rows are internally consistent: every row's `arm` / `policy_seed` / `decisions_cap` matches
  its file name (asserted in the script; a mismatch aborts the run), every (arm, seed) file has exactly
  1,200 rows (4 windows × 300) and every (arm, window) cell exactly 900 (3 seeds × 300), i.e. 300
  episodes per (seed, window).

### 2.2 The d900 cell finished **during** this publishing pass

This matters for anyone reading an earlier hand-off, so it is written down in order:

| time (2026-11-22, +08:00) | event |
| --- | --- |
| 10:35:41 | `run_d900.sh R5` launches **all three** d900 cells in parallel (`&`, one shell); the three `logs/eval-R5-d900-s*.out` are created in the same second |
| 10:59:52 | seed 2 completes, `status=complete survived=0/1200`, rc **0** recorded (`wave-d900-R5.out`, `commands-R5-d900-s2.tsv`) |
| before 11:07 | the executor is stopped after timing out; seeds 0/1 keep running **orphaned** (their wrapper is gone; no `commands-R5-d900-s{0,1}.tsv` row is ever written) |
| ~11:07 | the hand-off to this pass is written while the two files are **still growing**: 1,001 / 971 / 1,200 rows |
| ~11:08 | first freeze by this pass: 1,078 / 1,044 / 1,200 rows — **superseded, not preserved** (the final freeze overwrote it); only the hand-off's 1,001/971/1,200 prefix survives, as `d900_prefix_replica` |
| 11:12:57 | seed 0 completes: 1,200 rows, `status=complete survived=225/1200` |
| 11:14:02 | seed 1 completes: 1,200 rows, `status=complete survived=409/1200` |
| **11:15:25** | **final freeze published here: 3/3 files at 1,200 rows** (`diagnostics.json::frozen_at`) |

Consequences, stated plainly:

1. the published cell is **complete**; the "incomplete" reading belongs to the 11:02–11:14 window and is
   preserved as `diagnostics.json → d900_prefix_replica` (prefix = 1,001/971/1,200 rows → 504/3,172 =
   15.889 %, per seed 17.982 / 33.3677 / 0.00 %);
2. the figures carried in the hand-off (**15.01 %**, per-seed **17.4 / 33.2 / 0.0**) are a mid-run
   snapshot: the per-seed numbers reproduce approximately on that prefix, but **15.01 % is not
   reproducible from any slice** (504/3,172 = 15.889 %, and 504/3,600 = 14.000 %); they are therefore
   **not** used in any table here (§9.9);
3. seeds 0/1 have **no recorded exit code** — their wrapper died before it could write one. Completion is
   established by three independent facts: exactly 1,200 rows, the runner's own
   `status=complete …` final line, and its `summary.json` (`status: complete`, `out_sha256` equal to the
   published copy).

### 2.3 Three-source agreement (this is the integrity proof, not a claim)

[`refs/CROSSCHECK.json`](refs/CROSSCHECK.json) compares, for all **21** cells,
(A) the published copy, (B) the runner's `summary.json` (`out_sha256`, `out_bytes`, `status`,
`survived_total`, `episodes_total`, `bomb_emits_total`, `bombs_used_total`) and (C) the runner's final
log line (`status=complete survived=X/1200 emits=Y`):

| comparison | result |
| --- | --- |
| published sha256 == runner's `out_sha256` | **21/21** |
| published bytes == runner's `out_bytes` | **21/21** |
| runner `status` == `complete` | **21/21** |
| recomputed survived == runner's `survived_total` and mean log `survived` | **21/21** |
| recomputed episodes / bomb emits / bombs used == runner's totals | **21/21** |
| cells failing any check | **0** |

---

## 3. P1 at 600 decisions — raw counts (n = 3,600 per arm)

`alive_frames` = frames executed before the first death (death frame excluded); a full episode is
600 × 2 = **1,200**. Rate = survived / 3,600, exact fraction.

| arm | rule | survived | full rate | Δ vs R0 | alive mean | median | q1 | q3 | min | max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **R0** | `r80` | 2,560 | **71.1111 %** | — | 1155.3233 | 1200 | 1171 | 1200 | 778 | 1200 |
| **R1** | `conserve` | 2,428 | 67.4444 % | −3.6667 pp | 1148.7872 | 1200 | 1154 | 1200 | 791 | 1200 |
| **R2** | `delay200` | 620 | 17.2222 % | −53.8889 pp | 341.6542 | 109 | 73 | 311 | 57 | 1200 |
| **R3** | `double60` | 2,726 | 75.7222 % | +4.6111 pp | 1148.0469 | 1200 | 1200 | 1200 | 64 | 1200 |
| **R4** | `wall` | 929 | 25.8056 % | −45.3055 pp | 483.8481 | 110 | 74 | 1200 | 58 | 1200 |
| **R5** | `staged` | 2,776 | **77.1111 %** | **+6.0000 pp** | 1169.4875 | 1200 | 1200 | 1200 | 652 | 1200 |

### 3.1 Per seed (1,200 episodes per cell; the aggregate is *not* 3,600 i.i.d. draws)

| arm | seed 0 | seed 1 | seed 2 | spread (max−min) |
| --- | ---: | ---: | ---: | ---: |
| R0 | 1,157 (96.42 %) | 1,124 (93.67 %) | 279 (23.25 %) | 73.17 pp |
| R1 | 1,124 (93.67 %) | 1,071 (89.25 %) | 233 (19.42 %) | 74.25 pp |
| R2 | 277 (23.08 %) | 193 (16.08 %) | 150 (12.50 %) | 10.58 pp |
| R3 | 985 (82.08 %) | 920 (76.67 %) | **821 (68.42 %)** | **13.66 pp** |
| R4 | 516 (43.00 %) | 250 (20.83 %) | 163 (13.58 %) | 29.42 pp |
| R5 | **1,190 (99.17 %)** | **1,178 (98.17 %)** | 408 (34.00 %) | **65.17 pp** |

**R5 raises the ceiling (99.17 / 98.17 %) and R3 compresses the floor (68.42 % on the seed where
everything else collapses).** No arm lifts seed 2 above 34 %.

### 3.2 Per window (pooled over the 3 seeds; 900 episodes per cell)

| arm | w11 | w12 | w17 | w18 |
| --- | ---: | ---: | ---: | ---: |
| R0 | 676 (75.1 %) | 639 (71.0 %) | 564 (62.7 %) | 681 (75.7 %) |
| R1 | 671 (74.6 %) | 589 (65.4 %) | 507 (56.3 %) | 661 (73.4 %) |
| R2 | 42 (4.7 %) | 266 (29.6 %) | 286 (31.8 %) | 26 (2.9 %) |
| R3 | 678 (75.3 %) | 641 (71.2 %) | **723 (80.3 %)** | 684 (76.0 %) |
| R4 | 61 (6.8 %) | 458 (50.9 %) | 389 (43.2 %) | 21 (2.3 %) |
| R5 | **730 (81.1 %)** | **713 (79.2 %)** | 634 (70.4 %) | **699 (77.7 %)** |

Window identity is a fixed effect of the same size as the arm effect (R2 ranges 2.9–31.8 % across
windows), which is why the pooled 3,600 is reported as a design aggregate, never as an i.i.d. sample.

---

## 4. P1 — the prereg's test, run here

**Method.** For each variant vs R0, the 2×2 table is `[[survived_arm, failed_arm],
[survived_R0, failed_R0]]`. The p-values are **exact**: the hypergeometric pmf is summed in **rational
arithmetic** (`math.comb` + `fractions.Fraction`), with no floating point in either the p-value or the
Holm adjustment (`p × (m − i)`, monotone, capped at 1). `scipy.stats.fisher_exact` (scipy 1.18.1) is
computed on the same tables as an **independent cross-check**: the two agree to ≤ 1.7e−15 relative
wherever the double does not underflow. For R2 and R4 the one-sided and two-sided p are **below the
double range** (~1e−490 and ~1e−335); the exact values are printed instead of being replaced by 0.

| arm | survived / n | rate | Δ vs R0 | p (one-sided, "arm > R0") | **Holm, 5 one-sided** | p (one-sided, "arm < R0") | Holm, 5 two-sided |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R1 | 2,428 / 3,600 | 67.4444 % | −3.6667 pp | 9.99661e−01 | 1 (n.s.) | 4.08306e−04 | **8.16612e−04** |
| R2 | 620 / 3,600 | 17.2222 % | −53.8889 pp | 1 | 1 (n.s.) | 3.22750e−490 | **3.22750e−489** |
| R3 | 2,726 / 3,600 | 75.7222 % | +4.6111 pp | **5.32236e−06** | **2.12894e−05** | 0.999996 | 2.12894e−05 |
| R4 | 929 / 3,600 | 25.8056 % | −45.3055 pp | 1 | 1 (n.s.) | 1.40531e−335 | **1.12425e−334** |
| R5 | 2,776 / 3,600 | 77.1111 % | +6.0000 pp | **3.53114e−09** | **1.76557e−08** | 0.9999999997 | 2.11868e−08 |

**Verdict against prereg §2.1:**

- **S-better fires**: an arm exceeds R0's 71.1111 % **and** its Holm-adjusted one-sided p < 0.05 —
  R5 (+6.0000 pp, 1.77e−08) and R3 (+4.6111 pp, 2.13e−05). The prereg's action is "declare the timing
  optimisable, use it as the supervision source for v8.5.0 rule distillation, and re-measure at 900
  decisions" — the 900-decision run is §6.
- **S-null does not fire** (the arms are not all within ±3 pp) and **S-worse does not fire** (it requires
  *all* variants < R0 − 10 pp; two are above R0). The two failing arms (R2, R4) are far below the
  −10 pp line, but the branch is not declared, because it is written as a conjunction.
- **One honest caveat about direction.** The prereg registers a **one-sided (greater)** family, which by
  construction cannot establish that R1/R2/R4 are *worse* (their "greater" p is ≈ 1). The two-sided
  family (same tables, same Holm over 5) is therefore reported in the last column purely so that
  "significantly worse" can be stated with a test behind it. It is a **secondary, not preregistered**
  family, and is labelled as such in `diagnostics.json → p1_fisher_holm.secondary_family`.
- **Multiplicity scope.** Holm covers the 5 new comparisons against R0. It does **not** cover the
  selection that produced R0 itself (the incumbent 71.1111 % is the best of the v8.0.0/v8.1.0 rule
  family), nor any R5-vs-R3 comparison (none is registered, and none is run here). See §9.12.

---

## 5. P2 — bomb usage (descriptive only)

| arm | bombs used | per episode | distribution {0,1,2,3} | exhausted (`bombs_end == 0`) | first press (med / min–max) | last press (med / min–max) | last *honoured* press (med) |
| --- | ---: | ---: | --- | ---: | --- | --- | ---: |
| R0 | 10,740 | 2.9833 | 0 / 0 / 60 / 3,540 | 3,540 (98.33 %) | 0 / 0–0 | 433 / 260–599 | 432.5 |
| R1 | 10,770 | 2.9917 | 0 / 0 / 30 / 3,570 | 3,570 (99.17 %) | 0 / 0–0 | 420.5 / 247–598 | 420 |
| R2 | 1,747 | 0.4853 | 2,891 / 2 / 376 / 331 | 331 (9.19 %) | 200 / 200–201 | 441 / 206–599 | 440 |
| R3 | 9,328 | 2.5911 | 32 / 195 / 986 / 2,387 | 2,387 (66.31 %) | 0 / 0–71 | 458 / 5–599 | 456 |
| R4 | 3,624 | 1.0067 | 2,300 / 40 / 196 / 1,064 | 1,064 (29.56 %) | 49 / 15–365 | 418 / 23–599 | 417 |
| R5 | 10,581 | 2.9392 | 0 / 1 / 217 / 3,382 | 3,382 (93.94 %) | 0 / 0–0 | 458 / 6–599 | 458 |

Bomb *remaining* at the last frame (`bombs_remaining`), per arm, as the complement of the above:
R0 {0: 3,540, 1: 60}, R1 {0: 3,570, 1: 30}, R2 {0: 331, 1: 376, 2: 2, 3: 2,891},
R3 {0: 2,387, 1: 986, 2: 195, 3: 32}, R4 {0: 1,064, 1: 196, 2: 40, 3: 2,300},
R5 {0: 3,382, 1: 217, 2: 1}.

Presses (a press while a bomb is already active is ignored by the game, as in v8.0.0):

| arm | emits | honoured | ignored | episodes that ever pressed | doubled-bomb presses |
| --- | ---: | ---: | ---: | ---: | ---: |
| R0 | 76,198 | 10,740 | 65,458 | 3,600 | 0 |
| R1 | 104,870 | 10,769 | 94,101 | 3,600 | 1 |
| R2 | 25,629 | 1,680 | 23,949 | 709 | 67 |
| R3 | 51,959 | 9,238 | 42,721 | 3,568 | 90 |
| R4 | 27,525 | 3,621 | 23,904 | 1,300 | 3 |
| R5 | 48,336 | 10,581 | 37,755 | 3,600 | 0 |

What this says, plainly:

- **"Save the bombs" loses.** R2 (never bomb before decision 200) ends 2,891 of 3,600 episodes with all
  three bombs unused and survives 17.2 %; R4 (only near the side walls) ends 2,300 with none spent. Both
  spend ~1/6 to ~1/3 of R0's bombs and lose 45–54 pp.
- **The parked state is already lethal.** R0/R1/R5 press on **decision 0** in every episode, and the
  decision-0 presses report `emit_nearest_px` **24.9619 (w11) / 21.2839 (w12) / 36.3780 (w17) /
  18.2876 (w18)** — one value per dev window, identical across the three arms, i.e. the parked state's
  nearest bullet. The first bomb carries the episode. Waiting is not free: it is fatal.
- **Spend them, but make the *later* ones stricter — that wins.** R5 keeps the 1st bomb at 80 px and
  tightens to 60 px / 40 px for the 2nd / 3rd, moving the median last press from R0's 433 to **458**
  while spending the same 2.94 vs 2.98 bombs. R3 does the same with one knob (60 px **and** ≥ 2 bullets)
  and lands at 75.72 %.
- **R1 goes the wrong way.** Widening the last bomb to 120 px spends *more* (2.9917/ep, 99.17 %
  exhausted) and is worse than R0 (−3.67 pp): pressing the last bomb earlier is not free.
- **R0/R1/R5 all end 94–99 % of episodes at 0 bombs.** The difference between them is *where* the bombs
  go, not how many.

---

## 6. P3 — the best arm at 900 decisions (descriptive)

The cell is complete: 3 × 1,200 rows, 4 windows × 300 episodes per seed, same policy seeds. It is
**descriptive only** (prereg §2 P3); no test is run, and the comparison with R0's recorded 13.3333 % is
a point estimate against a **different round's** run (v8.1.0), not a paired test.

| arm | survived / 3,600 | rate | seed 0 | seed 1 | seed 2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **R5 @ 900** (this round) | **634** | **17.6111 %** | 225 (18.75 %) | 409 (34.08 %) | **0 (0.00 %)** |
| R0 @ 900 (v8.1.0's `eval_A3_d900_seed{k}`, recounted from that file) | 480 | 13.3333 % | 176 (14.67 %) | 304 (25.33 %) | 0 (0.00 %) |

| R5 @ 900 | bombs/ep | exhausted | first press (med) | last press (med) | alive mean | median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| seed 0 | 3.00 | 1,200/1,200 | 0 | 500.5 | 1593.1017 | 1610 |
| seed 1 | 3.00 | 1,200/1,200 | 0 | 505 | 1612.6050 | 1670 |
| seed 2 | 3.00 | 1,200/1,200 | 0 | 384 | 1145.7742 | 1156 |

Every d900 episode spends **all three** bombs (3,600 honoured presses per seed, 0 doubled); seed 2 dies
with the same 3.00 bombs/ep and a median last press at 384, i.e. the rule's budget is exhausted long
before the 900-decision wall. **R5 buys the mean, not the tail: 17.61 % pooled, 0 % on seed 2.**

---

## 7. Rule invariants — every one of the 334,517 presses re-checked

`bomb_audit.json` was **never produced** (the executor timed out before `finalize.sh`, §9.1), so this
pass re-implements the audit offline from the per-press fields the runner already wrote
(`trigger_threshold_px`, `emit_remaining_before`, `emit_nearest_px`, `emit_player_x`,
`emit_bullets_within_60`, `emit_decremented`). Full detail in
`diagnostics.json → arms_d600.<arm>.rule_invariants`.

| arm | presses checked | clause checked | violations |
| --- | ---: | --- | ---: |
| R0 | 76,198 | threshold == 80 and nearest ≤ 80 | **0** |
| R1 | 104,870 | threshold == 120 iff `bombs_remaining == 1`, else 80; nearest ≤ threshold | **0** |
| R2 | 25,629 | every press has `decision >= 200`; threshold 80; nearest ≤ 80 | **0** |
| R3 | 51,959 | every press has ≥ 2 bullets within 60 px; nearest ≤ 60 | **0** |
| R4 | 27,525 | every press has `x < 60` or `x > 340`; nearest ≤ 80 | **0** |
| R5 | 48,336 | threshold == 80 / 60 / 40 for `bombs_remaining` 3 / 2 / 1; nearest ≤ threshold | **0** |
| **total** | **334,517** | | **0** |

Two bookkeeping facts, reported rather than asserted:

- `bombs_used = honoured_presses + doubled_bomb_presses` holds **exactly** in all six arms (0/0, 1/1,
  67/67, 90/90, 3/3, 0/0), and no press ever spends more than one extra bomb. v8.0.0's stronger identity
  `bombs_used == honoured presses` **fails** for R1 (1), R2 (67), R3 (90) and R4 (3): a single honoured
  press can spend two bombs in the same decision. The cause is **not** established here — the doubled
  rows contain honoured presses at 0.35 px and (R3, w11 ep 180) at 57.89 px, so "near-death frame" does
  not explain all of them. Open item, not a claim.
- the runner's legacy smoke gate returns **7** for R2/R3 precisely because it asserts that identity; the
  rules' own semantics pass (§7).

---

## 8. The control is exact: R0 **is** v8.1.0's d600 A3 evaluation

`diagnostics.json → r0_reproduction_v810` compares all 1,200 rows of each R0 seed with v8.1.0's published
[`eval_A3_d600_seed{k}.jsonl`](../../../v8.1.0/evidence/BOMBFAR/) on 11 behavioural fields (`window`,
`episode`, `frames`, `alive_frames`, `first_death_offset`, `return_decisions`, `survived`,
`decisions_executed`, `bombs_used`, `bombs_end`, `bomb_decisions`):

| seed | episodes compared | rows missing in reference | field mismatches | identical |
| --- | ---: | ---: | ---: | --- |
| 0 | 1,200 | 0 | **0** | ✅ |
| 1 | 1,200 | 0 | **0** | ✅ |
| 2 | 1,200 | 0 | **0** | ✅ |
| **total** | **3,600** | 0 | **0** | **✅ `all_identical: true`** |

So the ±pp column in §3 is a comparison against a control that reproduces the v8.1.0 evaluation exactly,
not merely "a run with the same rule".

---

## 9. Deviations, corrections, and incomplete items (complete list)

1. **The executor timed out before finalisation; six expected artefacts do not exist.**
   `finalize.sh` was written but never produced `bomb_audit.json`, `audio_audit.json`,
   `process_residue.json`, `ckpt_assertions_final.json`, the runner's own `diagnostics.json`, or a
   `manifest.json`. This pass produced `diagnostics.json`, `manifest.json` and an offline rule audit (§7)
   instead; the other three are simply **absent** and are not simulated.
2. **The 900-decision cell outlived its launcher** (§2.2): all three seeds were launched at 10:35:41 by
   one `run_d900.sh`; seed 2 finished and recorded rc=0, seeds 0/1 were orphaned when the executor was
   stopped and completed at 11:12:57 / 11:14:02 with **no recorded exit code**. Completion is evidenced
   by 1,200 rows + the runner's `status=complete` line + its `summary.json`.
3. **One d600 cell failed at launch and was retried.** `eval-R4-d600-s0` died during parent start with
   `HarnessError: harness closed the control pipe` — **rc=4**, wall 7.792 s, started 10:12:25, finished
   10:12:32 (`logs/wave-d600.out`). It was relaunched at 10:12:46 and the published rows are the
   retry's (**rc=0**, 1168.603 s). `--out` truncates, so no partial row from the failed attempt can be
   in the file. The wave wrapper therefore exits **rc=1** even though 18/18 cells ended rc=0.
4. **The 900-decision arm was chosen before the 600-decision grid finished.** `run_d900.sh R5` was
   launched at 10:35:41; the last d600 cell finished at 10:46:04. The choice was made from preliminary
   counts. The final table confirms R5 is the maximum (77.1111 %), so the *choice* survives, but it was
   not a pre-registered determination and the prereg defines **no tie-break** if two arms had tied.
5. **The runner's `summary.json` names the wrong prereg.** Its `prereg` field is a hard-coded
   `Document/Update/Alpha/v8.0.0/PREREGISTRATION_BOMB_001.md §1` in all 21 cells. The governing prereg
   for this round is `PREREGISTRATION_BOMBRULE2_001.md`. Data unaffected; documentation conflict, listed
   in `SUMMARY_FOR_DECISION.md §6.4`.
6. **`--mode smoke`'s legacy gate returns 7 for R2 and R3** (§7, doubled bomb presses). The rules' own
   semantics are audited separately and pass; the runner's exit codes are published as measured
   (0 / 7 / 7 / 0 / 0).
7. **v8.0.0's `bombs_used == honoured presses` identity does not hold** for four arms (§7). The
   corrected identity holds exactly. Open item: the mechanism of a two-bomb press is not established.
8. **`--mode smoke`'s durable copy is a single shared path** (`<work>/smoke/bomb_smoke.json`), so five
   parallel smokes race on it. The canonical per-rule files are `smoke/bomb_smoke_R{1..5}.json`, which
   are the ones published and audited; the shared copy is ignored.
9. **The hand-off's d900 figures do not survive recompute** (§2.2): 1,001 / 971 / 1,200 rows, "15.01 %",
   per-seed 17.4 / 33.2 / 0.0. The prefix replica gives 504/3,172 = **15.889 %** and
   **17.982 / 33.3677 / 0.00 %**; 15.01 % corresponds to no slice of the published files (504/3,172 =
   15.889 %, 504/3,600 = 14.000 %). The published cell is complete: **634/3,600 = 17.6111 %**.
10. **The pre-patch runner was archived *after* the A/B before-run, not before it.** `ab/v800_bomb_pre.py`
    carries mtime 10:10:08 — after the before-run (10:07:40–10:09:25) and after the live file was patched
    (10:09:08). The gap is closed by hash and by behaviour: its sha256 is `f884a4be…b918`, exactly the
    `producer_sha256` that `ab_check.json` recorded for the before-run rows, and
    `ab/repro_before_check.json` re-runs that archived file for the same first 2 episodes and reproduces
    them **2/2 identical on 46 fields** (`all_pass: true`, rc=0, 13.81 s). So the archive really is the
    code that produced `ab/before/`, even though the copy was made late.
11. **Nothing else was touched.** No change to `touhou_rl/**`, `third_party/**`,
    `tools/rl/v432_fork_arm.py`, `tools/rl/v450_cf_label.py`, `tools/rl/v431_arm_a.py`, any earlier
    version directory or prereg; no deleted, moved or overwritten asset; no `git add` / `commit`. The
    only modified tracked-scope file of the round is `tools/rl/v800_bomb.py` (untracked, as in every
    previous round), plus `.tmp_v840_bomb/**` and this `Document/Update/Alpha/v8.4.0/**` tree. The
    publishing pass wrote **only** inside `Document/Update/Alpha/v8.4.0/` (evidence + two documents) and
    read `.tmp_v840_bomb/**` read-only.
12. **Preregistration gap (methodological, not a data defect):** the prereg compares five variants
    against an incumbent that was itself *selected as the best* of the previous rule family (v8.0.0's
    40/60/80 px, v8.1.0's A3 at 71.1111 %). Its Holm family covers the 5 new comparisons only, so the
    nominal α is optimistic with respect to that prior selection ("compare against the control's own
    best"); and there is **no registered R5-vs-R3 comparison**, although choosing a supervision source
    for v8.5.0 requires exactly that choice. Both gaps are inherited from the frozen text and are not
    repaired here — §4 reports only the registered family plus a labelled secondary one.

---

## 10. Published files, hashes, and how to reproduce

Package contents (`manifest.json` is authoritative; it lists every file with bytes, sha256 and line
count, and cannot contain its own hash):

| group | files | note |
| --- | ---: | --- |
| `eval_R{0..5}_d600_seed{0,1,2}.jsonl` | 18 | original names, byte-identical copies |
| `eval_R5_d900_seed{0,1,2}.jsonl` | 3 | original names, byte-identical copies (complete cell) |
| `diagnostics.json` | 1 | every number in this file, recomputed from the copies |
| `refs/` | see `refs/COPY_INDEX.json` | ckpt assertions, D3 fidelity, A/B check, rule check, patch diff, 27 commands TSVs, `CROSSCHECK.json`, `COPY_INDEX.json`, the runner used |
| `logs/` | 33 | every `.out` log of the round (waves, cells, smoke, A/B, D3) |
| `smoke/` | 5 | `bomb_smoke_R{1..5}.json` |
| `ab/` | 8 | 60-episode A/B before/after/repro rows + the pre-patch runner |
| `scripts/` | 9 | 4 written by this pass (`publish_v840.py`, `crosscheck_v840.py`, `manifest_v840.py`, `copy_refs_v840.py`) + 5 leftovers of the executor: `analyze.py`, `ab_check.py`, `exec.sh`, `publish.sh`, and a stale `__pycache__/analyze.cpython-310.pyc` |

**Note on the executor's leftovers.** `scripts/analyze.py` (28 KB) is the executor's own analyser. It
never produced an output file anywhere on disk, and it is **not** the source of `diagnostics.json` — every
number in this package comes from `scripts/publish_v840.py`. The leftovers are kept rather than deleted
(nothing is removed from `.tmp_v840_bomb/**` either) so that the round's tooling stays auditable.

Recorded hashes of the artefacts cited above:

| file | sha256 |
| --- | --- |
| `diagnostics.json` | `9fef417dae058beae369209ab3cb0bef4e07333add9b0b36f322c88dcfdcd7f5` |
| `refs/CROSSCHECK.json` | `a27e04475bf538f78c778a9d1c6f36a02e169fe7d8a273d6aa7ec8cf3caa462f` |
| `refs/COPY_INDEX.json` | `088f7969f05909f4eb059e1d3e7ea977a7ac5cd2f078c0dfbf7206d4d7a82281` |
| `refs/v800_bomb_used.py` (the runner) | `5d85bcb016a0eebbfb0fd101b2406c1841d8481468e057104b67c57120c0d7bf` |
| `manifest.json` | its sha256 is recorded in `SUMMARY_FOR_DECISION.md` / `README_INDEX.md` |

Reproduction (no engine, ~40 s, Windows or Linux):

```powershell
$py = "python"
$frozen = "2026-11-22T11:15:25+08:00"
& $py Document\Update\Alpha\v8.4.0\evidence\BOMBRULE2\scripts\publish_v840.py --no-copy --frozen-at=$frozen
& $py Document\Update\Alpha\v8.4.0\evidence\BOMBRULE2\scripts\crosscheck_v840.py
& $py Document\Update\Alpha\v8.4.0\evidence\BOMBRULE2\scripts\manifest_v840.py
```

`--no-copy` recomputes from the published copies only (it verifies sha256 against the sources and aborts
on any difference); `--frozen-at` pins the one field that would otherwise record "now", so the
regenerated file is **byte-identical** (verified: re-running the command above reproduces
`9fef417dae058beae369209ab3cb0bef4e07333add9b0b36f322c88dcfdcd7f5`, 175,407 bytes). Expected output:
`copied 21 files (sha256 verified)`, `d600 complete files: 18/18`,
`d900 rows: [1200, 1200, 1200] (complete: True)`, `cells checked: 21, failing: 0`.

---

## 11. What this file does not claim

- no claim that the policy "learned" anything — **all six arms are scripted rules, zero training**;
- no claim of a clear/game completion — single anchor segment (PARK frame 8663), 600 decisions (20 s);
- no claim that R5 > R3: it is **not tested** (not registered, not run; the point estimates are 77.11 % vs
  75.72 %, a 50-episode difference);
- no claim at 900 decisions beyond the descriptive table: one arm, one budget, R0's reference comes from
  a different round, and the prereg's P3 is descriptive;
- no claim that the 12 cells / 4 windows / 3 seeds are independent samples;
- no claim about seed 2's 0 % at 900 decisions beyond the measurement itself.
