# v8.9.0 RULEDIST2 — P0 forensics on the v8.6.0 batch

Prereg: [`PREREGISTRATION_RULEDIST2_001.md`](../../PREREGISTRATION_RULEDIST2_001.md)
(**FROZEN**, §2 P0). Executor: [`tools/rl/v890_ruledist.py`](../../../../../../tools/rl/v890_ruledist.py).
All numbers below are **counts, means, medians, quantiles** — no test, no interval,
no p-value.

**One-paragraph answer.** P0 does **not** hold on the v8.6.0 batch, so the round
stopped at **S-data**: the rule targets and their statistics are published and
**nothing was trained**. With R5's own input — `nearest_all_px`, which v8.6.0
recorded for every state in `row['behaviour']['nearest_all_px']` — the batch
carries **24 / 118 / 11** positive and **600 / 3,260 / 1,027** negative rule
targets at `bombs_remaining` 3 / 2 / 1, against the prereg's **≥ 50 positives and
≥ 50 negatives in every stage**. The 80 px and 40 px stages therefore fail; only
the 60 px stage passes. The cause is the **state distribution, not lost data**:
the behaviour policy that produced these states **is** R5, so the batch is made of
the states R5 *creates* (post-bomb, cleared screen, median nearest bullet
**220.0 px**) rather than the states R5 *fires in* (3.04 % of the batch). The
observations do lose `nearest_all_px` (they keep lethal bullets only), but that
field is recorded one level up and the rule target **is** recoverable offline;
recovering it does not rescue P0.

---

## 1. Data identity and method

| item | value |
| --- | --- |
| batch | `.tmp_v860_bombdag/labels_bombdag.jsonl` (**read-only**) |
| sha256 | `e1f47805efdd37bb92d15c51898aacbef0ca13995a200d7d2e087cd919543c68` |
| bytes / states | 35,189,092 / **5,040** |
| structure | 24 episodes = 3 seeds × 4 windows (13–16) × 2 episodes (12 × 300 states, 12 × 120 states) |
| `H` / labels | 120 / 36 per state (5,040 × 36, `label_vectors_valid` = 5,040) |
| `bombs_remaining` | 3: **624**, 2: **3,378**, 1: **1,038**, 0: 0 |
| behaviour | v4.5.0 `S` `round2.pt` (`4abee069…`/`a93746bb…`/`c69960fc…`) + **R5 `staged`** |
| feature reproduction | `features_sha256_k12` re-derived and equal on **5,040 / 5,040** rows |

Two distances are computed for every state:

* **`nearest_all_px`** — the minimum over **all** live bullets. This is what
  `v800_bomb.rule_gate('staged')` tests and what v8.6.0 stores per row in
  `row['behaviour']['nearest_all_px']` (present on **5,040 / 5,040** rows).
* **`nearest_obs_px`** — the minimum over `obs['hazards']['bullets']`. Both
  compact observations on this project keep **lethal bullets only**
  (`v450_cf_label.compact_observation`: `live = [b for b in hazards.bullets if
  b.lethal]`); v4.5.0 keeps 12, v6.0.0 keeps 64. This is the nearest **lethal**
  bullet and is an **upper bound** on `nearest_all_px`.

The rule is R5 as frozen: `bombs_remaining` 3 → 80 px, 2 → 60 px, 1 → 40 px,
0 → never; the test is **strictly** `distance < threshold`.

---

## 2. P0 — per stage, both definitions

| `bombs_remaining` | threshold | states | **pos (`nearest_all`)** | **neg** | pos rate | pos (obs-only) | neg (obs-only) | empty stored bullets | P0 verdict |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **3** | 80 px | 624 | **24** | 600 | 3.8462 % | 24 | 600 | 0 | **FAIL** (24 < 50) |
| **2** | 60 px | 3,378 | **118** | 3,260 | 3.4932 % | 10 | 3,368 | 372 | **PASS** |
| **1** | 40 px | 1,038 | **11** | 1,027 | 1.0597 % | 0 | 1,038 | 513 | **FAIL** (11 < 50) |
| all | — | 5,040 | **153** | 4,887 | 3.0357 % | 34 | 5,006 | 885 | **S-data** |

* P0 requires **≥ 50 positive and ≥ 50 negative** per stage (prereg §2). Negatives
  are satisfied everywhere; **positives are not** at the 80 px and 40 px stages.
* The obs-only column is an upper-bound-of-distance reading, so its positives are
  a **subset** of the `nearest_all` positives (34 ⊂ 153; 24 / 10 / 0). It is
  reported because the prereg's wording says "from `obs`"; it is **not** the rule
  target.
* **The 34 obs-only positives are exactly the 34 states v8.6.0 recorded as having
  spent a bomb** (`bomb_spent = True`): every recorded spend is a positive, and
  **no negative state spent a bomb**. That is the internal check that
  `nearest_all_px` — not the stored list — is the rule's input.
* Per (seed, window) the 153 positives split 2–20 each
  (13: 19/20/15, 14: 14/15/15, 15: 15/15/15, 16: 3/2/5 for seeds 0/1/2).

### 2.1 `nearest_all_px` quantiles

| subset | n | mean | median | q1 | q3 | min | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all 5,040 | 5,040 | 227.9751 | **220.0422** | 143.7987 | 289.3535 | 4.6041 | 572.7128 |
| `bombs=3` | 624 | 147.9903 | 154.2923 | 132.0202 | 182.0697 | 22.4827 | 195.0316 |
| `bombs=2` | 3,378 | 225.5968 | 245.7441 | 150.2069 | 291.4143 | 12.1391 | 564.2277 |
| `bombs=1` | 1,038 | 283.7982 | 239.2149 | 146.0968 | 412.4425 | 4.6041 | 572.7128 |

`nearest_obs_px` over the 4,155 non-empty rows: mean 225.9581, median 205.7532,
q1 147.9108, q3 287.0021, min 22.4827, max 478.2786.

For comparison, the v8.5.0 batch (v8.3.0's states, all at `bombs_remaining = 3`):
`nearest_obs_px` mean 56.8396, **median 36.3091** (v8.5.0 RUN.md §3). The v8.6.0
states are **~6× farther from the nearest bullet** — same anchor, same windows,
different behaviour (this one bombs).

---

## 3. What the observations store — v8.3.0 vs v8.6.0

| item | v8.3.0 (`labels_bomb36.jsonl`, 5,166) | v8.6.0 (`labels_bombdag.jsonl`, 5,040) |
| --- | --- | --- |
| `obs['hazards']['bullets']` length | **64** (63 in 18 rows) | **≤ 12**, distribution `0:885, 1:114, 2:51, 3:50, 4:49, 5:21, 6:37, 7:56, 8:12, 9:2, 10:188, 11:43, 12:3,532` |
| `selection.bullets_stored` | 64 (63 in 18), `enemies_stored` 2, `laser_stored` 0 | same three fields, `bullets_stored` = the length above, `enemies_stored` 2, `laser_stored` 0 |
| `hazards.bullet_count` (live count, all bullets) | not needed by the v8.5 comparison | median **87**, and **1** (mean 9.84, max 109) over the 885 empty rows |
| source of the observation | v6.0.0's `compact_observation` (cap **64**), copied from `.tmp_v600_obsp/labels_pipe.jsonl` | v4.5.0's `compact_observation` (cap **12**) |
| bullet filter | `[b for b in hazards.bullets if b.lethal]` | **identical** filter |
| order | sorted by distance to the player | sorted by distance to the player (**4,155 / 4,155** non-empty rows are non-decreasing) |
| `behaviour.nearest_all_px` | not recorded | **5,040 / 5,040 rows** |
| `behaviour.nearest_lethal_px` | not recorded | 5,040 / 5,040 rows |

### 3.1 Does "the 12 stored bullets are not the nearest 12" hold?

Partly — and the part that does **not** hold matters for the next round.

* **v8.3.0**: the minimum over the **first 12** stored bullets equals the minimum
  over all **64** on **5,166 / 5,166** rows (100 %). The list is a sorted
  nearest-first lethal list, so its prefix carries the nearest lethal bullet.
* **v8.6.0**: the minimum over the stored 12 equals the recorded
  **`nearest_lethal_px`** on **4,155 / 4,155** non-empty rows (100 %), and equals
  the recorded **`nearest_all_px`** on only **2,312 / 4,155** (55.64 %).
* So the 12 *are* the nearest 12 **lethal** bullets — the cap 64 → 12 does **not**
  drop the nearest lethal bullet. What the observation drops is **non-lethal**
  bullets (in both batches), and it is exactly those that make
  `nearest_all_px < nearest_lethal_px` on 1,843 of 4,155 rows (44.36 %).

**Consequence 1 (obs).** `nearest_all_px` — R5's input — **cannot be recomputed
from `obs['hazards']`**; the stored list is lethal-only. The median obs-only
minimum is 205.75 px against the true 220.04 px median, and the two disagree
row-by-row on 44 % of the non-empty rows.

**Consequence 2 (data, not obs).** It *can* be read from
`row['behaviour']['nearest_all_px']`, which v8.6.0 recorded at collection on every
row. Restoring that field costs no engine time and no re-collection — and doing
so gives the P0 table in §2, which still fails (24 / 118 / 11).

### 3.2 The 885 states with an empty stored bullet list

| item | value |
| --- | ---: |
| states | **885 / 5,040 = 17.56 %** |
| by `bombs_remaining` | 3: **0**, 2: **372**, 1: **513** |
| `hazards.bullet_count` on those rows | median **1**, mean 9.84, min 1, max 109 |
| `nearest_all_px` on those rows | median **304.2327**, min 91.0825, max 572.7128 |
| rule positives among them | **0** |

They are **not** silent skips: the row still carries `nearest_all_px`, and under
R5 none of them is within any threshold (min 91.08 px > 80 px). They are the
"screen cleared" frames a bomb leaves behind — i.e. the empty list is itself
evidence of the mechanism in §4.

---

## 4. Why the positives are rare — the mechanism

The batch's behaviour policy **is** R5. Every state in which R5 fires is followed
by a bomb, and a bomb clears the screen and grants invulnerability, so the next
decisions are recorded in the emptiness the bomb creates. The batch is therefore
dominated by the states R5 *creates*, not the states in which it *fires*:

| observation | value |
| --- | ---: |
| R5 positives (`nearest_all_px < threshold`) | **153 / 5,040 = 3.0357 %** |
| states where v8.6.0 recorded a bomb actually spent | **34** |
| of those, inside the positive set | **34 / 34** |
| negative states that spent a bomb | **0 / 4,887** |
| median `nearest_all_px` over the batch | **220.04 px** |
| median `nearest_all_px` over v8.3.0's (non-bombing) states, obs-only | **36.31 px** |

So the correct statement is **not** "the rule fires rarely on these states because
the observation is truncated". It is:

> **the rule fires 153 times in 5,040 states because the behaviour that generated
> the states obeys the rule: each firing removes the bullets (and the state) that
> would have produced the next firing. Sampling states with a rule-following
> policy under-samples the rule's own trigger points — and the tighter the rule,
> the stronger the effect (3.85 % at 80 px, 3.49 % at 60 px, 1.06 % at 40 px).**

The 34 recorded spends are a perfect subset of the 153 positives, which is the
strongest available check that the target definition used here (`nearest_all_px`,
strict `<`) is the rule the collector actually ran.

---

## 5. What this implies for the next round (recorded, not acted on here)

This round did **not** change the prereg's design and did **not** train. Three
facts the next prereg can build on:

1. **The rule target is already recorded** for v8.6.0
   (`behaviour.nearest_all_px`, 5,040/5,040). A re-collection is not required to
   *recover* it; a new collection is required only to *enrich* the positives.
2. **`obs` alone is not enough for R5** (lethal-only storage), so any future
   offline recomputation must read the recorded distance — or the observation
   format must gain `nearest_all_px` (it is one float; `behaviour` already has it).
3. **P0's 50-positives-per-stage bar cannot be met by sampling one trajectory per
   (seed, window) under the rule itself.** 24 / 118 / 11 positives at 3 / 2 / 1
   bombs means the design needs a positive-enriched sampler (states near
   trigger points, or triggered/not-triggered pairs) or a class-balanced loss
   (`pos_weight`) — either way, a design decision that belongs to the next
   prereg, not to this executor.

---

## 6. What this file does not claim

* It does **not** claim the v8.6.0 state batch is unusable — it claims P0 fails on
  it as frozen.
* It does **not** re-derive any v8.6.0 evaluation number, and it says nothing
  about v8.6.0's candidate-head results.
* It does **not** claim the 12-bullet cap degrades the **106-dim feature vector**:
  `build_arrays` reproduces `features_sha256_k12` on 5,040 / 5,040 rows, and the
  feature's own definition is "the 12 nearest of the stored lethal list", which is
  the 12 nearest lethal bullets in both batches.
* It does **not** run any test: no Fisher, no permutation, no interval, no
  "significant"; and it does not decide P1–P4.
