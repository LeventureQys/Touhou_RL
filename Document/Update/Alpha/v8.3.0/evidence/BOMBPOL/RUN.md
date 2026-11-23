# v8.3.0 BOMBPOL-001 — a 36-action policy that decides for itself when to bomb

Prereg: [`PREREGISTRATION_BOMBPOL_001.md`](../../PREREGISTRATION_BOMBPOL_001.md)
(FROZEN 2026-11-22, before any v8.3.0 label or training run).
Executor: [`tools/rl/v830_bombpol.py`](../../../../../../tools/rl/v830_bombpol.py).
Status: **the label batch, the three training runs and the 3 x 3 evaluation grid
ran to completion; every gate passed; all raw data published.**
**This file contains no statistical inference** (prereg §5): counts, means,
medians, quantiles and distributions only. P1–P4 belong to the analysis pass.

---

## 0. One-paragraph answer

The action space went from 18 to **36** (`action = cand * 2 + bomb`) and every one
of the **5,166 states** of the frozen v6.0.0 batch was re-labelled along all 36
actions at H = 120 — **185,976 labels**. The bomb really fires: `bomb = 1` sends
`BTN_BOMB` on the label's first frame, the counter drops **2 frames later**, and
**109,638** bombs were spent across the 92,988 `bomb = 1` labels. The label
contrast is the whole point of the round: **54.58 % of the 92,988 `bomb = 0`
labels are safe vs 99.83 % of the 92,988 `bomb = 1` labels**, and on a
(state, candidate) pair the bomb keeps you alive **longer in 42,073, exactly as
long in 50,915 and shorter in 0** — in this batch bombing is never worse. The
18 `bomb = 0` paths reproduce the v6.0.0 labels **bit for bit** (5,166/5,166
rows: same frame, same `alive_frames`, same `safe`), so the 18 `bomb = 1` paths
are the only new information in the round. Training the 36-output policy is as
easy as the 18-output one (train accuracy 90.9–91.0 %, 3 seeds), and the policy
it learns is "press the bomb, almost always": mean predicted probability 0.998
on the `bomb = 1` half vs 0.55 on the `bomb = 0` half, and on the live engine it
presses the bomb on **54.5 %** of all decisions (589,050 presses in the
300-decision grid alone, of which the game ignores **98.2 %** because a bomb is
already active). At **300 decisions** that is enough: **3,598 / 3,600
(99.94 %)** episodes walk all 600 frames, against the no-bomb baseline's
13.44 % and the 80 px script rule's 100 % — P1 holds and P2's >= 95 % bar is met.
At **600** and **900** decisions it collapses to **19.44 %** and
**0.61 %** (the rule: 95.05 % / 64.51 %), and the bomb audit says exactly
why: the policy spends **all 3 bombs in 100 % of episodes** (first press at
decision 0 in 3,569 / 3,560 / 3,573 of 3,600), so it has none left for the second
half of the segment. In one line: **the labels teach it that bombing is safe, and
it learns to bomb — but not when to bomb, and not to save bombs.**

---

## 1. What was run

| item | value |
| --- | --- |
| states | **5,166** (the `.tmp_v600_obsp/labels_pipe.jsonl` batch, read-only) |
| state locator | `(seed, round, window, decision, frame)`; the recorded `behavior_action` of every decision was replayed and the reached state had to match |
| labels | **36 per state, 185,976 total**, each held for **H = 120 frames** |
| `bomb = 1` semantics | `cand_to_mask(cand, 1)` on the label's first frame, `cand_to_mask(cand, 0)` on frames 2..120 (prereg §1's "first frame must really press") |
| model | `Policy(106, 64, 36)` — `Linear(106,64) -> Tanh -> Linear(64,36)` |
| training | BCE, Adam 1e-3, batch 256, **200 epochs**, **3 seeds**, no early stopping, dev never read |
| evaluation | **3 budgets (300 / 600 / 900 decisions) x 3 seeds x 4 dev windows x 300 episodes**, `stochastic`, live engine |
| engine | `/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern` sha256 `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` (asserted) |
| harness | [`scripts/exec.sh`](scripts/exec.sh) + `commands.tsv` (every command, exit code, wall time) |

The 5,166 states are the four **train** windows (13–16) of the v4.5.0/v6.0.0
DAgger protocol: 3 seeds x 3 rounds x 4 windows = 36 recorded trajectories, and
every state was re-reached by replaying the recorded behaviour action, in order,
from a freshly parked parent (2,165 prefix hashes checked, 0 mismatches).

---

## 2. Gates

| step | command | exit | wall |
| --- | --- | --- | --- |
| `preflight` | `/root/v400-venv/bin/python /mnt/d/WorkShop/Touhou_RL/tools/rl/v830_bombpol.py --mode preflight -…` | **0** | 5.77 s |
| `assert-start` | `/root/v400-venv/bin/python /mnt/d/WorkShop/Touhou_RL/tools/rl/v830_bombpol.py --mode assert-ckpt…` | **0** | 2.03 s |
| `d3-fidelity` | `/root/v400-venv/bin/python /mnt/d/WorkShop/Touhou_RL/tools/rl/v432_fork_arm.py --mode fidelity -…` | **0** | 9.06 s |
| `selfcheck` | `/root/v400-venv/bin/python /mnt/d/WorkShop/Touhou_RL/tools/rl/v830_bombpol.py --mode selfcheck -…` | **0** | 14.33 s |
| `equiv` | `/root/v400-venv/bin/python /mnt/d/WorkShop/Touhou_RL/tools/rl/v830_bombpol.py --mode equiv --pol…` | **0** | 7.21 s |
| `merge` | `/root/v400-venv/bin/python /mnt/d/WorkShop/Touhou_RL/tools/rl/v830_bombpol.py --mode merge --out…` | **1** | 1.59 s |
| `merge` | `/root/v400-venv/bin/python /mnt/d/WorkShop/Touhou_RL/tools/rl/v830_bombpol.py --mode merge --out…` | **0** | 3.34 s |
| `train-s0` | `/root/v400-venv/bin/python /mnt/d/WorkShop/Touhou_RL/tools/rl/v830_bombpol.py --mode train --see…` | **0** | 6.96 s |
| `train-s1` | `/root/v400-venv/bin/python /mnt/d/WorkShop/Touhou_RL/tools/rl/v830_bombpol.py --mode train --see…` | **0** | 5.37 s |
| `train-s2` | `/root/v400-venv/bin/python /mnt/d/WorkShop/Touhou_RL/tools/rl/v830_bombpol.py --mode train --see…` | **0** | 5.37 s |
| `train-summary` | `/root/v400-venv/bin/python /mnt/d/WorkShop/Touhou_RL/tools/rl/v830_bombpol.py --mode train-summa…` | **0** | 0.12 s |
| `smoke` | `/root/v400-venv/bin/python /mnt/d/WorkShop/Touhou_RL/tools/rl/v830_bombpol.py --mode smoke --pol…` | **0** | 4.83 s |
| `diagnostics` | `/root/v400-venv/bin/python /mnt/d/WorkShop/Touhou_RL/tools/rl/v830_bombpol.py --mode diagnostics…` | **0** | 3.58 s |
| `audio-audit` | `/root/v400-venv/bin/python /mnt/d/WorkShop/Touhou_RL/tools/rl/v830_bombpol.py --mode audio-audit…` | **0** | 1.77 s |
| `process-audit` | `/root/v400-venv/bin/python /mnt/d/WorkShop/Touhou_RL/tools/rl/v830_bombpol.py --mode process-aud…` | **0** | 0.21 s |
| `assert-final` | `/root/v400-venv/bin/python /mnt/d/WorkShop/Touhou_RL/tools/rl/v830_bombpol.py --mode assert-ckpt…` | **0** | 1.20 s |
| `publish` | `/root/v400-venv/bin/python /mnt/d/WorkShop/Touhou_RL/tools/rl/v830_bombpol.py --mode publish --w…` | **1** | 2.04 s |
| `publish` | `/root/v400-venv/bin/python /mnt/d/WorkShop/Touhou_RL/tools/rl/v830_bombpol.py --mode publish --w…` | **0** | 20.75 s |
| `collect-*` (10 group runs, 9 in parallel) | `v830_bombpol.py --mode collect --seed S --round R` | **0,143,4** | 16401 s total, longest 2195 s |
| `eval-*` (9 cells, 9 in parallel) | `v830_bombpol.py --mode eval --decisions D --policy-seed S` | **0** | 11599 s total, longest 1587 s |

**D3 fidelity (5/5, [`fidelity_d3.json`](fidelity_d3.json))** —
`1_hello_role_shadow`, `2_debug_threads_1`, `3_prefix_2165_checkpoints`
(2,165/2,165, 0 mismatches, final frame 8663), `4_r24_18_constants` (18/18
`first_death_offset` = R24's table), `5_park_frame_8663` — all PASS.

**Checkpoint assertions ([`ckpt_assertions.json`](ckpt_assertions.json) start /
[`ckpt_assertions_final.json`](ckpt_assertions_final.json) final)** — the three
frozen v4.5.0 `round2.pt` (the end of the behaviour chain that produced the
5,166 states) and the mask table / runner / protected sources did not move:

| item | sha256 | match |
| --- | --- | --- |
| `seed0/round2.pt` | `4abee069d087872e0a2d726cc975850d760e90a705ddcbee60ea91268e5e61dc` | ✅ |
| `seed1/round2.pt` | `a93746bb5d123165e77e0271edfc1efcdb21931e9e2d706e5a22d613c5476ca4` | ✅ |
| `seed2/round2.pt` | `c69960fcbbd888e9b252d3fa685c22a6730f8dd637dcfc9edab45491d218fa01` | ✅ |
| `.tmp_v600_obsp/labels_pipe.jsonl` | `eeb2f9d6429dbc1430d48b1d18d0b1c127e851d86e476fa26a70124e94e04260` | ✅ |

**Labeller self-check ([`labeller_selfcheck.json`](labeller_selfcheck.json)) —
S-label, the two gates prereg §2.1 asks for:**

1. **18/18 at H = 600.** The bomb-aware labeller with `bomb = 0` on all 18
   candidates reproduces R24's 18 `first_death_offset` **bit for bit**
   (window 12, frame 8663).
2. **36/36.** One live state labelled along **all 36** actions: 36 labels
   present, **0 missing**, every `bomb = 1` path **really spent exactly 1 bomb**
   (18/18), no `bomb = 0` path spent any (0/18), the decrement happens at
   **frame 2** for every one of them — the same 2-frame latency v8.0.0 measured.
   On that state the bomb is also visible in the label itself: `bomb = 1` is safe
   **18/18** vs `bomb = 0` **15/18**, mean `alive_frames` 120.0 vs 113.1.

**Loop equivalence ([`equiv_check.json`](equiv_check.json))** — `run_episode36`
with the bomb bit forced to 0 is the v8.0.0 loop on the live engine: 4/4
episodes identical to `v800_bomb.run_episode_bomb --bomb-rule none` on every
compared field **and** on the full 18-candidate action sequence (33 / 97 / 99 /
65 decisions). The copy is not trusted on faith.

---

## 3. The labels — what bombing does to the label (prereg P4's raw material)

| item | bomb=0 (18 actions) | bomb=1 (18 actions) |
| --- | --- | --- |
| labels | 92,988 | 92,988 |
| safe labels | 50,753 | 92,826 |
| **safe rate** | **54.58%** | **99.83%** |
| alive_frames mean | 93.6791 | 119.7909 |
| alive_frames median | 120.0000 | 120.0000 |
| states where ALL of this arm is safe | 57 | 5157 |
| states where NONE is safe | 964 | 9 |
| bomb actions that ever spent a bomb (of 18) | 0 | 18 |
| bombs spent (total) | - | 109638 |

states: 5,166, rows with 36 labels: 5,166 (100.00%), H = [120], labels total: 185,976

per (state, candidate) pair, n = one (state, candidate) pair = 92988 pairs: bomb=1 keeps you alive **longer** on 42,073, **equal** on 50,915, **shorter** on 0; safe-rate delta = 45.25% points

| alive_frames bucket | bomb=0 | bomb=1 |
| --- | --- | --- |
| <1 | 162 | 162 |
| <12 | 3,744 | 162 |
| <24 | 7,524 | 162 |
| <48 | 15,084 | 162 |
| <60 | 19,106 | 162 |
| <120 | 42,235 | 162 |
| >=120 | 50,753 | 92,826 |

**The key contrast** (task item 3): the bomb makes a label **safer**
(99.83 % vs 54.58 %) and **never worse** on any (state, candidate) pair in this
batch. The mechanism is not subtle: `bomb = 1` at H = 120 buys invincibility for
most of the horizon, and 92,826 of 92,988 bomb labels survive all 120 frames
(vs 50,753 without). The 162 labels that die even with a bomb are the same 162
that die at frame 0 without one (`<1` bucket identical) — a lethal contact that
the bomb cannot pre-empt.

**The best fidelity evidence in the round** ([`labels_vs_v600.json`](labels_vs_v600.json)):
the 18 `bomb = 0` paths are **byte-identical** to the frozen v6.0.0 18-way
labels — all 5,166 keys present on both sides, 5,166/5,166 same `frame`,
5,166/5,166 same `alive_frames`, 5,166/5,166 same `safe`. So this round's
collection **is** the v6.0.0 experiment on the no-bomb half, and the only new
information is the 18 `bomb = 1` paths.

> What this means for learning, stated plainly: the label set is **not
> balanced** — "bomb" is nearly always safe, so a BCE-trained head can get ~91 %
> accuracy by learning "always bomb" and never learn *when*. That is a property
> of the labels (a bomb at H = 120 is cheap insurance), not of the optimiser, and
> it is exactly why §5's eval looks the way it does.

---

## 4. Training

| seed | states | labels | loss epoch 0 | loss epoch 199 | final BCE | train acc | mean p(bomb=0) | mean p(bomb=1) | ckpt sha256 | wall s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 5,166 | 185,976 | 0.6319 | 0.1963 | 0.1958 | 90.90% | 0.5529 | 0.9982 | `6e508206b9e951e0…` | 2.2 |
| 1 | 5,166 | 185,976 | 0.6173 | 0.1956 | 0.1952 | 90.92% | 0.5520 | 0.9982 | `a04366c305938982…` | 2.1 |
| 2 | 5,166 | 185,976 | 0.6302 | 0.1940 | 0.1936 | 90.99% | 0.5508 | 0.9983 | `5e9bcb60059d5961…` | 2.1 |

The three seeds land within 0.6 % of each other on train accuracy and 0.002 on
final BCE; the mean predicted probability is 0.55 on the `bomb = 0` half (the
base rate is 54.58 %) and **0.998** on the `bomb = 1` half (the base rate is
99.83 %). The model reproduces the label marginals; it does not sharpen them.

---

## 5. Evaluation — 3 budgets x 3 seeds x 4 windows x 300 episodes

| budget (decisions) | episodes | survived | full-rate | alive mean | alive median | q1 | q3 | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 300 | 3,600 | 3,598 | **99.94%** | 599.9906 | 600.0000 | 600.0000 | 600.0000 | 567 | 600 |
| 600 | 3,600 | 700 | **19.44%** | 990.7286 | 989.5000 | 845.7500 | 1155.0000 | 582 | 1200 |
| 900 | 3,600 | 22 | **0.61%** | 1029.8478 | 992.0000 | 848.0000 | 1158.0000 | 560 | 1800 |

| budget | seeds that bombed | bombs spent | bombs/ep mean | median | max | histogram {bombs/ep: episodes} | exhaustion | never bombed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 300 | 3,600 / 3,600 | 10,800 | 3.0000 | 3.0000 | 3 | 3: 3,600 | 100.00% | 0 |
| 600 | 3,600 / 3,600 | 10,800 | 3.0000 | 3.0000 | 3 | 3: 3,600 | 100.00% | 0 |
| 900 | 3,600 / 3,600 | 10,800 | 3.0000 | 3.0000 | 3 | 3: 3,600 | 100.00% | 0 |

| budget | first-bomb decision | mean | median | min | max |
| --- | --- | --- | --- | --- | --- |
| 300 | 0: 3,569, 1-4: 31, 5-9: 0, 10-24: 0, 25-49: 0, 50-99: 0, 100-199: 0, >=200: 0 | 0.0086 | 0.0000 | 0 | 1 |
| 600 | 0: 3,560, 1-4: 40, 5-9: 0, 10-24: 0, 25-49: 0, 50-99: 0, 100-199: 0, >=200: 0 | 0.0117 | 0.0000 | 0 | 2 |
| 900 | 0: 3,573, 1-4: 27, 5-9: 0, 10-24: 0, 25-49: 0, 50-99: 0, 100-199: 0, >=200: 0 | 0.0075 | 0.0000 | 0 | 1 |

| budget | seed | episodes | survived | full-rate | alive mean | alive median | bombs used |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 300 | 0 | 1,200 | 1,199 | 99.92% | 599.9992 | 600.0000 | 3,600 |
| 300 | 1 | 1,200 | 1,200 | 100.00% | 600.0000 | 600.0000 | 3,600 |
| 300 | 2 | 1,200 | 1,199 | 99.92% | 599.9725 | 600.0000 | 3,600 |
| 600 | 0 | 1,200 | 236 | 19.67% | 996.8383 | 1001.5000 | 3,600 |
| 600 | 1 | 1,200 | 266 | 22.17% | 999.2025 | 1001.0000 | 3,600 |
| 600 | 2 | 1,200 | 198 | 16.50% | 976.1450 | 972.5000 | 3,600 |
| 900 | 0 | 1,200 | 5 | 0.42% | 1038.1933 | 997.0000 | 3,600 |
| 900 | 1 | 1,200 | 16 | 1.33% | 1054.5183 | 1024.5000 | 3,600 |
| 900 | 2 | 1,200 | 1 | 0.08% | 996.8317 | 958.0000 | 3,600 |

| budget | seed/window | episodes | survived | alive mean | alive median | bombs used | exhausted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 300 | s0_w11 | 300 | 300 | 600.0000 | 600.0000 | 900 | 300 |
| 300 | s0_w12 | 300 | 299 | 599.9967 | 600.0000 | 900 | 300 |
| 300 | s0_w17 | 300 | 300 | 600.0000 | 600.0000 | 900 | 300 |
| 300 | s0_w18 | 300 | 300 | 600.0000 | 600.0000 | 900 | 300 |
| 300 | s1_w11 | 300 | 300 | 600.0000 | 600.0000 | 900 | 300 |
| 300 | s1_w12 | 300 | 300 | 600.0000 | 600.0000 | 900 | 300 |
| 300 | s1_w17 | 300 | 300 | 600.0000 | 600.0000 | 900 | 300 |
| 300 | s1_w18 | 300 | 300 | 600.0000 | 600.0000 | 900 | 300 |
| 300 | s2_w11 | 300 | 300 | 600.0000 | 600.0000 | 900 | 300 |
| 300 | s2_w12 | 300 | 300 | 600.0000 | 600.0000 | 900 | 300 |
| 300 | s2_w17 | 300 | 300 | 600.0000 | 600.0000 | 900 | 300 |
| 300 | s2_w18 | 300 | 299 | 599.8900 | 600.0000 | 900 | 300 |
| 600 | s0_w11 | 300 | 60 | 999.1100 | 1004.5000 | 900 | 300 |
| 600 | s0_w12 | 300 | 48 | 959.6467 | 954.5000 | 900 | 300 |
| 600 | s0_w17 | 300 | 74 | 1032.3867 | 1037.0000 | 900 | 300 |
| 600 | s0_w18 | 300 | 54 | 996.2100 | 985.0000 | 900 | 300 |
| 600 | s1_w11 | 300 | 66 | 1000.4767 | 988.5000 | 900 | 300 |
| 600 | s1_w12 | 300 | 54 | 966.8000 | 963.0000 | 900 | 300 |
| 600 | s1_w17 | 300 | 91 | 1037.5933 | 1075.5000 | 900 | 300 |
| 600 | s1_w18 | 300 | 55 | 991.9400 | 983.0000 | 900 | 300 |
| 600 | s2_w11 | 300 | 43 | 980.2333 | 970.5000 | 900 | 300 |
| 600 | s2_w12 | 300 | 46 | 937.1167 | 922.5000 | 900 | 300 |
| 600 | s2_w17 | 300 | 57 | 1003.9000 | 1023.5000 | 900 | 300 |
| 600 | s2_w18 | 300 | 52 | 983.3300 | 977.0000 | 900 | 300 |
| 900 | s0_w11 | 300 | 2 | 1022.8600 | 978.5000 | 900 | 300 |
| 900 | s0_w12 | 300 | 0 | 1001.6967 | 959.5000 | 900 | 300 |
| 900 | s0_w17 | 300 | 2 | 1112.3933 | 1067.0000 | 900 | 300 |
| 900 | s0_w18 | 300 | 1 | 1015.8233 | 973.5000 | 900 | 300 |
| 900 | s1_w11 | 300 | 4 | 1058.5000 | 1028.5000 | 900 | 300 |
| 900 | s1_w12 | 300 | 3 | 1011.3633 | 974.0000 | 900 | 300 |
| 900 | s1_w17 | 300 | 5 | 1108.1400 | 1084.0000 | 900 | 300 |
| 900 | s1_w18 | 300 | 4 | 1040.0700 | 991.5000 | 900 | 300 |
| 900 | s2_w11 | 300 | 1 | 996.9633 | 939.5000 | 900 | 300 |
| 900 | s2_w12 | 300 | 0 | 962.4333 | 945.0000 | 900 | 300 |
| 900 | s2_w17 | 300 | 0 | 1034.8900 | 1021.0000 | 900 | 300 |
| 900 | s2_w18 | 300 | 0 | 993.0400 | 951.0000 | 900 | 300 |

All 12 cells per budget are `status: complete`, 0 invalid episodes, 0 dropped
rows. "alive_frames" is frames executed before the first death (death frame
excluded, R24's rule); a survived episode walked the whole budget, i.e. 600 /
1,200 / 1,800 frames.

---

## 6. Beside the two frozen references (no test, no interval)

| arm | 300 decisions | 600 decisions | 900 decisions | source |
| --- | --- | --- | --- | --- |
| no bomb at all (v4.5.0 `S`, 18 actions) | 13.44% (484/3,600) | not run at this budget | not run at this budget | v8.0.0 BOMB RUN.md §4 |
| script rule, 80 px | 100.00% | 95.05% | 64.51% | v8.0.0 (300) + v8.3.0 prereg §0 (600/900, quoted) |
| **v8.3.0 36-action policy (this round)**  **99.94%** (3,598/3,600)  **19.44%** (700/3,600)  **0.61%** (22/3,600) | this round | |

Read against prereg §2:
**P1** (300 decisions, > 13.4444 %): raw count 3,598 / 3,600 =
99.94 % — consistent with P1 holding. **P2** (>= 95 % of the 80 px rule's
100 %): 99.94 % — the raw count is **at or above the 95 % bar**, the shortfall to the
rule is 0.06 points (2 episodes of 3,600). **P3** (600 / 900 beside the
script rule's 95.05 % / 64.51 %): 19.44 % / 0.61 % — far below, and the
reason is visible in the bomb audit (§7): the policy spends all 3 bombs
immediately, while the rule still has bombs left in the second half. **P4** is
§3 and §7.

**Which stop condition the raw counts are consistent with.** P1 holds and P2
holds *at the 300-decision budget*, so the raw data is consistent with
**S-learned** ("策略自己学会了放雷") — but only in the letter of P2, which is
defined at 300 decisions. P3 is where the learning shows its limit: the policy
walks 600 frames perfectly and then has nothing left, so at 600 and 900
decisions it is far *behind* the script rule. The prereg's S-learned next step
asks for exactly this pair of measurements; both are here (§5, §7).
**Bomb economy: the policy saves nothing** — 3.00 bombs per episode at every
budget, 100 % of episodes exhausted. The script rule at 300 decisions spends
1.36 / 1.59 / 1.75 bombs per episode (v8.0.0 A1 / A2 / A3) and still has bombs
left when the second half of a longer segment starts; that round never ran 600 or
900 decisions, so its bomb usage at those budgets is **not** on disk and is not
quoted here. This file does **not** declare a stop condition: the endpoint tests
are the analysis pass's job.

---

## 7. Bomb behaviour of the learned policy (prereg P4, descriptive only)

| budget | episodes that pressed bomb | bomb presses | admitted by the game | presses ignored | bombs spent | bombs/ep (mean / median / max) |
| --- | --- | --- | --- | --- | --- | --- |
| 300 | 3,600 / 3,600 | 589,050 | 10,800 | 578,250 | 10,800 | 3.0000 / 3.0000 / 3 |
| 600 | 3,600 / 3,600 | 1,180,473 | 10,800 | 1,169,673 | 10,800 | 3.0000 / 3.0000 / 3 |
| 900 | 3,600 / 3,600 | 1,253,912 | 10,800 | 1,243,112 | 10,800 | 3.0000 / 3.0000 / 3 |

| budget | bombs/ep histogram | episodes that ended on 0 bombs | exhaustion | episodes that never bombed |
| --- | --- | --- | --- | --- |
| 300 | 3: 3,600 | 3,600 | 100.00% | 0 |
| 600 | 3: 3,600 | 3,600 | 100.00% | 0 |
| 900 | 3: 3,600 | 3,600 | 100.00% | 0 |

| budget | first-bomb decision histogram | mean | median | min | max |
| --- | --- | --- | --- | --- | --- |
| 300 | 0: 3,569, 1-4: 31, 5-9: 0, 10-24: 0, 25-49: 0, 50-99: 0, 100-199: 0, >=200: 0 | 0.0086 | 0.0000 | 0 | 1 |
| 600 | 0: 3,560, 1-4: 40, 5-9: 0, 10-24: 0, 25-49: 0, 50-99: 0, 100-199: 0, >=200: 0 | 0.0117 | 0.0000 | 0 | 2 |
| 900 | 0: 3,573, 1-4: 27, 5-9: 0, 10-24: 0, 25-49: 0, 50-99: 0, 100-199: 0, >=200: 0 | 0.0075 | 0.0000 | 0 | 1 |

What the numbers say, plainly:

- **it presses the bomb on 54.5 % of all decisions** (589,050 / 1,080,000 in the
  300-decision grid; 1,180,473 and 1,253,912 at the longer budgets), and the game
  **ignores 98.2 % / 99.1 % / 99.1 %** of those presses because a bomb is already
  active. That is not a bug in the loop: it is what `cand_to_mask(cand, 1)`
  does when the bomb is already burning, and it is exactly the "wastes input"
  pattern v8.0.0 measured for its 80 px rule (34,891 presses / 4,880 honoured).
- **the first press is on decision 0** in 3,569 / 3,560 / 3,573 of 3,600 episodes
  (99.2 % / 98.9 % / 99.3 %); the latest first press in the whole grid is
  decision 2. The policy has learned "bomb at the start", i.e. the behaviour the
  labels reward, not a timing rule.
- **all three bombs are spent in 100 % of episodes** at every budget
  (`bombs_used = 3` for all 3,600 x 3 episodes), so **exhaustion is 100 %** and
  the policy ends every episode with an empty counter. The script rule at 300
  decisions spends 1.36 / 1.59 / 1.75 bombs on average (v8.0.0 A1–A3), and
  v8.0.0's own note applies here unchanged: most presses are *ignored input*, not
  wasted bombs.
- consequence, measured: the bombs are gone long before the segment is, so the
  600- and 900-decision budgets are walked by a policy with no bombs left —
  19.44 % and 0.61 % against the rule's 95.05 % and 64.51 % (§6).

---

## 8. Deviations and corrections (complete list)

1. **The collect was restarted once, at 09:16:32, and ~928 already-written rows
   were discarded.** The first version compared the live 25-field `STATE` dict
   with the 10-field subset the v6.0.0 row stores, so `state_equal_v600` was
   `False` **by construction** and measured nothing. The check now compares
   exactly the 10 stored keys and records `state_keys_compared` /
   `state_keys_differ`; the discarded partial files were deleted and their logs
   are kept in `logs/restart1/`. Nothing was reused from the discarded rows.
2. **One group (`s0 r0`) failed its first start and was re-run on a brand-new
   data dir.** `build_private_data_dir` never overwrites a file that already
   exists, so a private dir reused from the aborted attempt kept the
   `score.dat` that attempt's engine had written; the `(seed 13)` RESET hash of
   that dir (`ac06125b1138429b`) no longer matched the recorded one
   (`fed791a690b9b75a`) and `walk_prefix` refused to start (`FidelityError`,
   §2.3 first checkpoint). `retry_group.sh` re-ran the group with
   `--tag …-retry1` (fresh dir) → prefix 2,165/2,165, 0 mismatches. **No group
   was re-collected for a label reason.**
3. **`--mode merge` crashed after writing all 5,166 rows.** `mode_merge` ended
   with `partial.close()` on a `pathlib.Path` (the file object had already been
   closed by its `with` block):
   `AttributeError: 'PosixPath' object has no attribute 'close'` —
   `[merge] rc=1`, the 48 MB `labels_bomb36.jsonl.partial` was complete but not
   renamed. The stray call was removed and `merge` re-run (rc 0, 3.3 s); the rows
   were **not** collected again. The re-run added the state-locator block that
   proves the merged keys are exactly the v6.0.0 keys.
4. **`bomb = 1` label semantics.** The prereg pins only "the first frame really
   sends `BTN_BOMB`". The label sends the bomb bit on frame 1 and not on frames
   2..120, i.e. exactly what the evaluation loop sends for a decision (v8.0.0's
   own rule). One label therefore means "press the bomb now", not "hold the bomb
   for 2 s". Every row records `bomb_decrement_frame` so the press can be audited.
5. **`Policy(106, 64, 36)` is the frozen class with its output widened.**
   `v431_arm_a.Policy` hardcodes `N_CAND = 18` outputs and may not be edited this
   round, so the frozen class is instantiated and its last layer replaced by
   `Linear(64, 36)` (`policy36()`), which is the same shape and the same
   initialisation order.
6. **`run_episode36` is a copy of `v800_bomb.run_episode_bomb`** (that file's loop
   takes the bomb bit from a distance rule, not from a policy); the copy is
   proved equivalent by `--mode equiv` (§2).
7. **`tools/rl/v600_obs_pipe.py` is read, not imported.** Its `PINNED` table is
   stale (it pins `v450_cf_label.py` at `978c7cca…` and `v432_fork_arm.py` at
   `c018758b…`; both files have since moved), so importing it would abort on its
   own pin check. Its observation code is reached through
   `v431_arm_a.build_features` (identical at K = 12, proven by v6.0.0 on 240 live
   states) and its output file is read read-only.
8. **The 80 px rule's 600/900-decision numbers are quoted, not recomputed.**
   Prereg §0 lists 95.05 % and 64.51 % for v8.1.0 BOMBFAR; that round's raw run
   in `.tmp_v810_bomb` was interrupted (the largest cell on disk holds 1,085 of
   1,200 episodes and no complete summary exists), so only the 300-decision
   number (3,600/3,600, v8.0.0) is a published measurement.
9. **The box was shared for the whole round.** v8.1.0 and v8.2.0 evaluation grids
   were running on the same 20 cores; load average sat at **43–51** during
   collection and dropped to ~3–10 during the eval grid. `contention.tsv` is the
   per-minute record (states collected, load, engine process count) and
   [`scripts/tables.py contention`](scripts/tables.py) prints it. It explains the
   wall times, and nothing else.
10. **One forbidden-verb incident, disclosed.** To stop my own collection groups
    after the restart decision I ran `pkill -f 'v830_bombpol.py --mode collect'`
    once; it matched only my own processes (and the shell that issued it, which
    is why the call returned nothing), terminated none of them, and I then
    stopped them by exact PID (`stop_collect.sh`). No other round's process was
    touched, and `pkill` was not used again.
11. **`numbers.py` was renamed `tables.py`** before it could shadow the stdlib
    `numbers` module for anything run from this directory.
12. **Two non-zero exits in the gate table are this round's own bookkeeping, not
    hidden failures.** The `merge` row with `exit 1` is the `partial.close()`
    crash of item 3 (re-run: `exit 0`); the `publish` row with `exit 1` is a
    missing `scripts/` directory inside the evidence folder
    (`FileNotFoundError: …/scripts/exec.sh`; `mode_publish` now creates the
    parent of every target — re-run: `exit 0`). The `collect-*` row reports
    `0,143,4` because it also contains the two groups of the first attempt that
    were **deliberately stopped** (`143` = 128 + SIGTERM, item 1) and the one
    group that failed its data-dir reuse (`4`, item 2).
13. **`cmd_to_win` / wall times are contaminated by the shared box** (item 9):
    the label collection that takes ~35 min on an idle machine took **4 h 36 min**
    of summed group time (16,401 s over 10 group runs, longest 2,195 s) and the
    eval grid **3 h 13 min** of summed cell time (11,599 s over 9 cells, longest
    1,587 s). Those are summed wall times of parallel processes, not elapsed.
14. **`tools/rl/v800_bomb.py` was edited by *another* round during this one**
    (mtime 10:09:08; this round never wrote to it). The start-phase assertion
    (09:08:25) recorded the pinned `f884a4be1abebb2b…`; the final-phase assertion
    (10:28:54) records `5d85bcb016a0eebb…`. This round imports that file in
    **exactly one place** — `--mode equiv`, which ran at 09:09:02, i.e. **before**
    the edit — so every v8.3.0 measurement used the pinned version.
    `ckpt_assertions.json` and `ckpt_assertions_final.json` both ship, so the
    edit is visible rather than hidden; `verify.py` reports it as a warning with
    the two recorded hashes.

---

## 9. The machine during this round (contention record)

This round shared a 20-core box with v8.1.0 and v8.2.0 evaluation grids for its
whole label phase. `contention.tsv` records, **once per minute**, how many states
this round had labelled, the 1-/5-minute load average, how many `th08-modern`
processes existed on the box and how many of them were this round's. Sampled
every 15 minutes:

samples: 74 (one per minute), from 2026-11-22T09:14:34+08:00 to 2026-11-22T10:31:56+08:00; the table below is **every 15th sample** (≈15 min)

| timestamp | states collected | load1 | load5 | th08 processes | this round's |
| --- | --- | --- | --- | --- | --- |
| 2026-11-22T09:14:34+08:00 | 689 | 47.58 | 44.82 | 163 | 27 |
| 2026-11-22T09:30:27+08:00 | 2568 | 35.27 | 40.41 | 102 | 27 |
| 2026-11-22T09:46:24+08:00 | 4807 | 3.77 | 10.02 | 6 | 6 |
| 2026-11-22T10:02:16+08:00 | 5166 | 11.24 | 8.61 | 43 | 43 |
| 2026-11-22T10:18:14+08:00 | 5166 | 45.57 | 32.79 | 122 | 54 |
| 2026-11-22T10:31:56+08:00 | 5166 | 18.89 | 23.12 | 93 | 0 |

peak load1 **48.17** at 2026-11-22T09:15:36+08:00 with 837 states collected and 160 `th08-modern` processes on the box (26 of them this round's).

The plateaus are visible and they are not smooth: load **43–51** while the other
rounds ran their grids (09:14 → 09:47, the whole label collection), a trough at
**3.8** around 09:46 when they stopped, then **11 → 46 → 19** during the eval grid
as the next round's grid started and the box refilled. Nothing about a v8.3.0
number depends on this; it explains the wall times in §2 and nothing else
(deviation 9).

---

## 10. What this round does **not** claim

Prereg §5, in effect and verbatim:

- **not** "通关" — this is still a single anchored segment (frame 8663, 300/600/
  900 decisions of a 2-frame decision grid);
- the three budgets are **not comparable by their means** (prereg §5), and this
  file does not compare them that way;
- the policy's bomb behaviour and the 80 px script rule's behaviour are **not the
  same thing** (one is learned, one is written down), so "the policy is worse
  than the rule" is a statement about this comparison, not about the policy;
- the 12 (seed, window) cells per budget are **not** independent samples and are
  not aggregated as if they were;
- **no statistics**: no Fisher, no permutation, no p-value, no Holm correction,
  no confidence interval, no "significant". P1/P2/P3 are **not** decided here.

Beyond the prereg, two things this file also does not claim:

- **not** "bombing is free": the labels say a bomb at H = 120 is safe, not that
  spending the third bomb at decision 40 is wise. §5 and §7 show the opposite
  consequence at the longer budgets;
- **not** "the 36-action head is better than an 18-action head + a rule": this
  round never trained the latter.

---

## 11. Files in this directory

Every file below is listed with its byte count and sha256; `manifest.json` (which
also covers `RUN.md` itself) is the machine-readable copy of the same table, and
the `scripts/`, `logs/` and `runs/` subdirectories are in it as well.

| file | bytes | sha256 |
| --- | --- | --- |
| `audio_audit.json` | 8,561 | `019211283099f591…` |
| `bomb_usage.json` | 21,665 | `bfbea02668652925…` |
| `ckpt_assertions.json` | 3,570 | `e16aef436e627828…` |
| `ckpt_assertions_final.json` | 3,570 | `8600d003aa272d95…` |
| `commands.tsv` | 11,155 | `d501eb079aeb2618…` |
| `contention.tsv` | 3,500 | `56baf9a94ecdd6ec…` |
| `diagnostics.json` | 22,999 | `f575ca1495854349…` |
| `equiv_check.json` | 1,997 | `1e9e641ca7df0231…` |
| `eval_d300_seed0.jsonl` | 9,520,267 | `798cc24883c56d83…` |
| `eval_d300_seed0.summary.json` | 14,170 | `2982eb158ab9c3a2…` |
| `eval_d300_seed1.jsonl` | 9,044,805 | `8683894506e0852d…` |
| `eval_d300_seed1.summary.json` | 14,157 | `1ce61e54f8e38546…` |
| `eval_d300_seed2.jsonl` | 10,691,038 | `75185e500b29d198…` |
| `eval_d300_seed2.summary.json` | 14,178 | `cd2b2b983fb8535e…` |
| `eval_d600_seed0.jsonl` | 17,515,050 | `5e6d17accba9e8c5…` |
| `eval_d600_seed0.summary.json` | 14,249 | `c1735bda4572ba70…` |
| `eval_d600_seed1.jsonl` | 17,388,398 | `b001fbedd0809a2a…` |
| `eval_d600_seed1.summary.json` | 14,250 | `05e822eb3c77523e…` |
| `eval_d600_seed2.jsonl` | 18,549,828 | `7a8a9c03c48e6410…` |
| `eval_d600_seed2.summary.json` | 14,255 | `521435f669a6c541…` |
| `eval_d900_seed0.jsonl` | 18,515,000 | `c4bab471a72ba641…` |
| `eval_d900_seed0.summary.json` | 14,265 | `47764bf77a2d9a10…` |
| `eval_d900_seed1.jsonl` | 18,820,449 | `cc884c2bc951fff2…` |
| `eval_d900_seed1.summary.json` | 14,265 | `fe5fa3a61164a505…` |
| `eval_d900_seed2.jsonl` | 19,103,271 | `ae3c3327b5c8d0dc…` |
| `eval_d900_seed2.summary.json` | 14,244 | `d22003a0856ecb66…` |
| `fidelity_d3.json` | 26,596 | `a47e397a8bd8a7f0…` |
| `labeller_selfcheck.json` | 25,477 | `f7cd2156d16376be…` |
| `labels_bomb36.jsonl` | 49,602,266 | `aea2c7abee220997…` |
| `labels_bomb36_stats.json` | 7,372 | `57a224cb6d7c6904…` |
| `labels_vs_v600.json` | 803 | `c76cf040d8604bcc…` |
| `logs/assert-final.out` | 114 | `f97f942cccd8955d…` |
| `logs/assert-start.out` | 108 | `890f2c87d0e6a892…` |
| `logs/audio-audit.out` | 151 | `ffb1cbcb5ecab017…` |
| `logs/chain.log` | 171 | `be3e300a14baf36f…` |
| `logs/collect-driver.log` | 321 | `a34ddb48e0f934bc…` |
| `logs/collect-s0-r0-retry1.out` | 1,214 | `b2b21e60163cbfb3…` |
| `logs/collect-s0-r0.out` | 266 | `eae6f03227afb645…` |
| `logs/collect-s0-r1.out` | 1,414 | `2845f71f73cd013d…` |
| `logs/collect-s0-r2.out` | 1,407 | `ed4ef91944107c63…` |
| `logs/collect-s1-r0.out` | 1,452 | `8bdc5c98db0dcbe7…` |
| `logs/collect-s1-r1.out` | 1,732 | `a9a8f049744b0d4c…` |
| `logs/collect-s1-r2.out` | 1,528 | `07cd8f1015582254…` |
| `logs/collect-s2-r0.out` | 1,316 | `9882705788d04214…` |
| `logs/collect-s2-r1.out` | 1,486 | `f1dd961916c6f7bc…` |
| `logs/collect-s2-r2.out` | 1,612 | `ff48fd1bdedb9c30…` |
| `logs/d3-fidelity.out` | 2,727 | `a607c7e03792057f…` |
| `logs/diagnostics.out` | 471 | `07b653d3fae2da3e…` |
| `logs/equiv.out` | 413 | `bd2843753d9e1181…` |
| `logs/eval-d300-s0.out` | 2,759 | `5e8e6edf9ef8aad6…` |
| `logs/eval-d300-s1.out` | 2,756 | `b44b617eb4c10457…` |
| `logs/eval-d300-s2.out` | 2,756 | `cae22032ddc2f70e…` |
| `logs/eval-d600-s0.out` | 2,738 | `f09577b4cd894415…` |
| `logs/eval-d600-s1.out` | 2,741 | `0d5d04fca361ebd4…` |
| `logs/eval-d600-s2.out` | 2,736 | `7749da0c0dd538fa…` |
| `logs/eval-d900-s0.out` | 2,716 | `a0e27aa6eb76a1c4…` |
| `logs/eval-d900-s1.out` | 2,712 | `11ee9a18ed8babce…` |
| `logs/eval-d900-s2.out` | 2,711 | `fdb99c2300d8ea2c…` |
| `logs/grid.log` | 312 | `be968f064d604938…` |
| `logs/merge.out` | 447 | `d3ab778f291cd922…` |
| `logs/monitor.log` | 0 | `e3b0c44298fc1c14…` |
| `logs/preflight.out` | 126 | `0bc68b92070f286f…` |
| `logs/process-audit.out` | 121 | `927df93159e722b6…` |
| `logs/publish.out` | 0 | `e3b0c44298fc1c14…` |
| `logs/restart1/collect-s0-r0.out` | 445 | `7a2b621e26848ef2…` |
| `logs/restart1/collect-s0-r1.out` | 203 | `0d45028dc8f7d3f8…` |
| `logs/restart1/collect-s0-r2.out` | 451 | `4263c79867f5edf8…` |
| `logs/restart1/collect-s1-r0.out` | 244 | `9b733076978c842a…` |
| `logs/restart1/collect-s1-r1.out` | 203 | `33f78e1ac66adc14…` |
| `logs/restart1/collect-s1-r2.out` | 452 | `55046afefe3124cd…` |
| `logs/restart1/collect-s2-r0.out` | 491 | `21dfb3de4c07d7f4…` |
| `logs/restart1/collect-s2-r1.out` | 203 | `94e1e2e3fd8407ae…` |
| `logs/restart1/collect-s2-r2.out` | 203 | `45517fe5a3ed0da5…` |
| `logs/retry-s0r0.log` | 42 | `a0b7fb478b4b8593…` |
| `logs/selfcheck.out` | 1,425 | `81f7d8c7faae43d2…` |
| `logs/smoke.out` | 405 | `41a16ca2532d44c2…` |
| `logs/train-s0.out` | 264 | `c35ae0a7e17032df…` |
| `logs/train-s1.out` | 264 | `51a66153c74d7304…` |
| `logs/train-s2.out` | 264 | `4d762952a8b249f9…` |
| `logs/train-summary.out` | 98 | `f15f25c1b19a7305…` |
| `merge.json` | 11,592 | `60fbfa6b684cd02c…` |
| `preflight.json` | 9,568 | `68528026fc49550e…` |
| `process_residue.json` | 487 | `9307d4a41ad53bff…` |
| `runs/v830-collect-s0-r0-retry1/run.log` | 1,214 | `b2b21e60163cbfb3…` |
| `runs/v830-collect-s0-r0-retry1/w13-stderr.log` | 163 | `269756be14721050…` |
| `runs/v830-collect-s0-r0-retry1/w14-stderr.log` | 163 | `8344c5bdfd615a90…` |
| `runs/v830-collect-s0-r0-retry1/w15-stderr.log` | 163 | `08683880b63c57d7…` |
| `runs/v830-collect-s0-r0-retry1/w16-stderr.log` | 163 | `04da9df2f0299399…` |
| `runs/v830-collect-s0-r0/run.log` | 711 | `3d10667e3569e7b3…` |
| `runs/v830-collect-s0-r0/w13-stderr.log` | 163 | `8522be6ad1f82a42…` |
| `runs/v830-collect-s0-r0/w14-stderr.log` | 163 | `bc4b4f74d390d315…` |
| `runs/v830-collect-s0-r1/run.log` | 1,617 | `9576c6cc4648c08e…` |
| `runs/v830-collect-s0-r1/w13-stderr.log` | 163 | `bc06ad31a51e61f3…` |
| `runs/v830-collect-s0-r1/w14-stderr.log` | 163 | `7334039f00595428…` |
| `runs/v830-collect-s0-r1/w15-stderr.log` | 163 | `08683880b63c57d7…` |
| `runs/v830-collect-s0-r1/w16-stderr.log` | 163 | `25b8b977736f4fd5…` |
| `runs/v830-collect-s0-r2/run.log` | 1,858 | `9a9822f48ca23bfe…` |
| `runs/v830-collect-s0-r2/w13-stderr.log` | 163 | `9ef52b6d815f8773…` |
| `runs/v830-collect-s0-r2/w14-stderr.log` | 163 | `2ba133d3292ed196…` |
| `runs/v830-collect-s0-r2/w15-stderr.log` | 163 | `34336e23b7b554e9…` |
| `runs/v830-collect-s0-r2/w16-stderr.log` | 163 | `08683880b63c57d7…` |
| `runs/v830-collect-s1-r0/run.log` | 1,696 | `ee8742c223d7f666…` |
| `runs/v830-collect-s1-r0/w13-stderr.log` | 163 | `60fcab3b05be7a02…` |
| `runs/v830-collect-s1-r0/w14-stderr.log` | 163 | `2ba133d3292ed196…` |
| `runs/v830-collect-s1-r0/w15-stderr.log` | 163 | `04da9df2f0299399…` |
| `runs/v830-collect-s1-r0/w16-stderr.log` | 163 | `04da9df2f0299399…` |
| `runs/v830-collect-s1-r1/run.log` | 1,935 | `9d8279bcc5f83d8f…` |
| `runs/v830-collect-s1-r1/w13-stderr.log` | 163 | `bc06ad31a51e61f3…` |
| `runs/v830-collect-s1-r1/w14-stderr.log` | 163 | `269756be14721050…` |
| `runs/v830-collect-s1-r1/w15-stderr.log` | 163 | `04da9df2f0299399…` |
| `runs/v830-collect-s1-r1/w16-stderr.log` | 163 | `4efbecf959352831…` |
| `runs/v830-collect-s1-r2/run.log` | 1,980 | `e79eb1d6cd86fda2…` |
| `runs/v830-collect-s1-r2/w13-stderr.log` | 163 | `2ba133d3292ed196…` |
| `runs/v830-collect-s1-r2/w14-stderr.log` | 163 | `bc4b4f74d390d315…` |
| `runs/v830-collect-s1-r2/w15-stderr.log` | 163 | `72c3d163183ad656…` |
| `runs/v830-collect-s1-r2/w16-stderr.log` | 163 | `e7926619a2c51aa6…` |
| `runs/v830-collect-s2-r0/run.log` | 1,807 | `864f5a78f293b190…` |
| `runs/v830-collect-s2-r0/w13-stderr.log` | 163 | `25b8b977736f4fd5…` |
| `runs/v830-collect-s2-r0/w14-stderr.log` | 163 | `bc4b4f74d390d315…` |
| `runs/v830-collect-s2-r0/w15-stderr.log` | 163 | `85bd91248de42cf1…` |
| `runs/v830-collect-s2-r0/w16-stderr.log` | 163 | `8d4f96ee5d18e678…` |
| `runs/v830-collect-s2-r1/run.log` | 1,689 | `8a8cfd6b5e9c5302…` |
| `runs/v830-collect-s2-r1/w13-stderr.log` | 163 | `f29bedd8b4f7012b…` |
| `runs/v830-collect-s2-r1/w14-stderr.log` | 163 | `bc4b4f74d390d315…` |
| `runs/v830-collect-s2-r1/w15-stderr.log` | 163 | `04da9df2f0299399…` |
| `runs/v830-collect-s2-r1/w16-stderr.log` | 163 | `25b8b977736f4fd5…` |
| `runs/v830-collect-s2-r2/run.log` | 1,815 | `72001921aaeab192…` |
| `runs/v830-collect-s2-r2/w13-stderr.log` | 163 | `4352c68afe28d3dd…` |
| `runs/v830-collect-s2-r2/w14-stderr.log` | 163 | `08683880b63c57d7…` |
| `runs/v830-collect-s2-r2/w15-stderr.log` | 163 | `25b8b977736f4fd5…` |
| `runs/v830-collect-s2-r2/w16-stderr.log` | 163 | `8344c5bdfd615a90…` |
| `runs/v830-equiv/run.log` | 413 | `bd2843753d9e1181…` |
| `runs/v830-equiv/w12-stderr.log` | 163 | `bc4b4f74d390d315…` |
| `runs/v830-eval-d300-s0/run.log` | 2,759 | `5e8e6edf9ef8aad6…` |
| `runs/v830-eval-d300-s0/w11-stderr.log` | 163 | `04da9df2f0299399…` |
| `runs/v830-eval-d300-s0/w12-stderr.log` | 163 | `60fcab3b05be7a02…` |
| `runs/v830-eval-d300-s0/w17-stderr.log` | 163 | `60fcab3b05be7a02…` |
| `runs/v830-eval-d300-s0/w18-stderr.log` | 163 | `60fcab3b05be7a02…` |
| `runs/v830-eval-d300-s1/run.log` | 2,756 | `b44b617eb4c10457…` |
| `runs/v830-eval-d300-s1/w11-stderr.log` | 163 | `60fcab3b05be7a02…` |
| `runs/v830-eval-d300-s1/w12-stderr.log` | 163 | `60fcab3b05be7a02…` |
| `runs/v830-eval-d300-s1/w17-stderr.log` | 163 | `e7926619a2c51aa6…` |
| `runs/v830-eval-d300-s1/w18-stderr.log` | 163 | `25b8b977736f4fd5…` |
| `runs/v830-eval-d300-s2/run.log` | 2,756 | `cae22032ddc2f70e…` |
| `runs/v830-eval-d300-s2/w11-stderr.log` | 163 | `8d4f96ee5d18e678…` |
| `runs/v830-eval-d300-s2/w12-stderr.log` | 163 | `04da9df2f0299399…` |
| `runs/v830-eval-d300-s2/w17-stderr.log` | 163 | `60fcab3b05be7a02…` |
| `runs/v830-eval-d300-s2/w18-stderr.log` | 163 | `60fcab3b05be7a02…` |
| `runs/v830-eval-d600-s0/run.log` | 2,738 | `f09577b4cd894415…` |
| `runs/v830-eval-d600-s0/w11-stderr.log` | 163 | `25b8b977736f4fd5…` |
| `runs/v830-eval-d600-s0/w12-stderr.log` | 163 | `e7926619a2c51aa6…` |
| `runs/v830-eval-d600-s0/w17-stderr.log` | 163 | `25b8b977736f4fd5…` |
| `runs/v830-eval-d600-s0/w18-stderr.log` | 163 | `08683880b63c57d7…` |
| `runs/v830-eval-d600-s1/run.log` | 2,741 | `0d5d04fca361ebd4…` |
| `runs/v830-eval-d600-s1/w11-stderr.log` | 163 | `04da9df2f0299399…` |
| `runs/v830-eval-d600-s1/w12-stderr.log` | 163 | `04da9df2f0299399…` |
| `runs/v830-eval-d600-s1/w17-stderr.log` | 163 | `04da9df2f0299399…` |
| `runs/v830-eval-d600-s1/w18-stderr.log` | 163 | `bc4b4f74d390d315…` |
| `runs/v830-eval-d600-s2/run.log` | 2,736 | `7749da0c0dd538fa…` |
| `runs/v830-eval-d600-s2/w11-stderr.log` | 163 | `04da9df2f0299399…` |
| `runs/v830-eval-d600-s2/w12-stderr.log` | 163 | `25b8b977736f4fd5…` |
| `runs/v830-eval-d600-s2/w17-stderr.log` | 163 | `25b8b977736f4fd5…` |
| `runs/v830-eval-d600-s2/w18-stderr.log` | 163 | `0092f09babe0df1a…` |
| `runs/v830-eval-d900-s0/run.log` | 2,716 | `a0e27aa6eb76a1c4…` |
| `runs/v830-eval-d900-s0/w11-stderr.log` | 163 | `bc4b4f74d390d315…` |
| `runs/v830-eval-d900-s0/w12-stderr.log` | 163 | `04da9df2f0299399…` |
| `runs/v830-eval-d900-s0/w17-stderr.log` | 163 | `04da9df2f0299399…` |
| `runs/v830-eval-d900-s0/w18-stderr.log` | 163 | `bc4b4f74d390d315…` |
| `runs/v830-eval-d900-s1/run.log` | 2,712 | `11ee9a18ed8babce…` |
| `runs/v830-eval-d900-s1/w11-stderr.log` | 163 | `269756be14721050…` |
| `runs/v830-eval-d900-s1/w12-stderr.log` | 163 | `60fcab3b05be7a02…` |
| `runs/v830-eval-d900-s1/w17-stderr.log` | 163 | `25b8b977736f4fd5…` |
| `runs/v830-eval-d900-s1/w18-stderr.log` | 163 | `bc4b4f74d390d315…` |
| `runs/v830-eval-d900-s2/run.log` | 2,711 | `fdb99c2300d8ea2c…` |
| `runs/v830-eval-d900-s2/w11-stderr.log` | 163 | `60fcab3b05be7a02…` |
| `runs/v830-eval-d900-s2/w12-stderr.log` | 163 | `60fcab3b05be7a02…` |
| `runs/v830-eval-d900-s2/w17-stderr.log` | 163 | `08683880b63c57d7…` |
| `runs/v830-eval-d900-s2/w18-stderr.log` | 163 | `8d4f96ee5d18e678…` |
| `runs/v830-selfcheck/run.log` | 1,425 | `81f7d8c7faae43d2…` |
| `runs/v830-selfcheck/w12-stderr.log` | 163 | `bc4b4f74d390d315…` |
| `runs/v830-smoke/run.log` | 405 | `41a16ca2532d44c2…` |
| `runs/v830-smoke/w11-stderr.log` | 163 | `795ec69bd2077e10…` |
| `scripts/RUN.md.tmpl` | 17,193 | `807809a3c0960af5…` |
| `scripts/build_run.py` | 5,608 | `604bbdb1dde5b66e…` |
| `scripts/chain.sh` | 1,496 | `98bc0d38f85d5a64…` |
| `scripts/compare_v600.py` | 4,096 | `c1092eaf5ba048aa…` |
| `scripts/draft_check.py` | 1,610 | `610d83bc2b9fc433…` |
| `scripts/exec.sh` | 6,224 | `56d1ef27014e47e5…` |
| `scripts/grid_status.sh` | 656 | `c24704ec6a3b856c…` |
| `scripts/monitor.sh` | 959 | `e088ce92b8debe1c…` |
| `scripts/progress.sh` | 1,288 | `092b69393ddc79fd…` |
| `scripts/retry_group.sh` | 1,573 | `298401bba6016026…` |
| `scripts/show.py` | 7,388 | `185c3417551c192f…` |
| `scripts/status.sh` | 1,149 | `016ecfbd86107b39…` |
| `scripts/stop_collect.sh` | 1,023 | `3bc1a44a8ea9b35f…` |
| `scripts/tables.py` | 12,429 | `99c52dfe2329844f…` |
| `scripts/verify.py` | 6,321 | `89ea68941b8b87ff…` |
| `smoke.json` | 206,257 | `cfe93c9da085317f…` |
| `train_seed0.json` | 6,802 | `b4a9657e13bc1d7f…` |
| `train_seed1.json` | 6,798 | `d4d8cadcedac4a2f…` |
| `train_seed2.json` | 6,798 | `7f995f6bf357c5f5…` |
| `train_summary.json` | 2,463 | `693884b401e9c8ec…` |
| `verify.py` | 6,297 | `57fc24e0a61959b9…` |
| `RUN.md` | this file | it cannot list its own hash; the manifest below covers it |
| `manifest.json` | sha256 + bytes of every file above | re-generated **after** `RUN.md` was copied in, so it covers `RUN.md` too |

