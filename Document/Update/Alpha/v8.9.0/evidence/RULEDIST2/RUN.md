# v8.9.0 RULEDIST2-001 — distilling R5 onto v8.6.0's bomb-state distribution

Prereg: [`PREREGISTRATION_RULEDIST2_001.md`](../../PREREGISTRATION_RULEDIST2_001.md)
(**FROZEN** 2026-11-22, before any v8.9.0 run).
Executor: [`tools/rl/v890_ruledist.py`](../../../../../../tools/rl/v890_ruledist.py)
(copied from v8.5.0's `v850_distill.py` and edited in three places only: the state
source, the P0 gate, and the two-distance forensic block; every engine-touching
loop is **imported** from `v830_bombpol.py`, never copied).
Status: **stopped at S-data (prereg §2).** The rule targets were computed on all
**5,040** v8.6.0 states, **P0 does not hold**, and nothing was trained and nothing
was evaluated. **No statistical inference** (prereg §5): counts, means, medians,
quantiles and distributions only.

---

## 0. One-paragraph answer

The prereg's single factor was executed up to its prerequisite gate: the **5,040**
v8.6.0 states (`bombs_remaining` 3: 624, 2: 3,378, 1: 1,038) got the R5 target
computed offline — `bombs_remaining` 3 → 80 px, 2 → 60 px, 1 → 40 px, 0 → never,
strictly `<` — using R5's own input, `nearest_all_px`, which v8.6.0 recorded on
**every** row in `row['behaviour']['nearest_all_px']`. **P0 is the round's
outcome: it fails.** The three stages carry **24 / 118 / 11** positives and
**600 / 3,260 / 1,027** negatives, and the prereg requires **≥ 50 positives and
≥ 50 negatives in every stage**; the 80 px and 40 px stages miss the positive bar
by 26 and 39. The cause is not a broken file: `features_sha256_k12` reproduces on
5,040/5,040 rows, all 5,040 rows carry `nearest_all_px`, and the 34 states v8.6.0
recorded as having **spent a bomb** are a perfect subset of the 153 positives
(0 negatives spent one). The cause is the **state distribution**: the behaviour
policy that generated the batch *is* R5, so the batch is made of the states a
bomb leaves behind — median nearest bullet **220.04 px**, 17.56 % of states with
**no lethal bullet at all** — rather than the states R5 fires in (3.04 % of the
batch). So this round publishes the rule targets and their per-stage positive /
negative counts and **stops**: per prereg §2.1 this is the **S-data** branch,
"只发布规则目标统计，不训练". The full forensic account — including what the
observations do and do not store — is in [`p0_forensics.md`](p0_forensics.md).

---

## 1. What was run

| step | command (all under `.tmp_v890_ruledist/scripts/exec.sh`) | exit | wall |
| --- | --- | ---: | ---: |
| `preflight` | `v890_ruledist.py --mode preflight` | **0** | 3.105 s |
| `assert-start` | `v890_ruledist.py --mode assert-ckpt --phase start` | **0** | 1.036 s |
| `targets` | `v890_ruledist.py --mode targets` | **0** | 3.346 s |
| `d3-fidelity` | `v432_fork_arm.py --mode fidelity --seeds 12 --learner frozen` | **0** | 7.458 s |
| `audio-audit` | `v890_ruledist.py --mode audio-audit` | **0** | 0.212 s |
| `process-audit` | `v890_ruledist.py --mode process-audit --data-dir-filter .tmp_v890_ruledist` | **0** | 0.166 s |
| `ckpt-manifest` | `v890_ruledist.py --mode ckpt-manifest` | **0** | 0.144 s |
| `assert-final` | `v890_ruledist.py --mode assert-ckpt --phase final` | see `commands.tsv` | see `commands.tsv` |
| `process-audit-final` | `--mode process-audit ... --out process_residue_final.json` | see `commands.tsv` | see `commands.tsv` |
| `publish` | `v890_ruledist.py --mode publish` | see `commands.tsv` | see `commands.tsv` |
| `verify` | `scripts/verify.py` (independent recomputation) | see `commands.tsv` / [`verify.json`](verify.json) | — |
| *(not run)* | `--mode train`, `--mode eval`, `--mode diagnostics` | — | S-data |

`commands.tsv` is the log of every invocation; the published copy is the snapshot
taken **inside** `publish`, so the `publish` and `verify` rows themselves appear
only in the work-dir copy (`.tmp_v890_ruledist/commands.tsv`). No command that
touched a live engine ran in any of the training/evaluation modes; the only
engine work in this round is the D3 fidelity gate (§2).

---

## 2. Gates

* **Preflight ([`preflight.json`](preflight.json))** — every pinned import and
  the state file match: `all_gated_match: true`, `states: 5,040`,
  `bytes_match: true`, `labels_per_state: [36]`, `H: [120]`,
  `windows: [13,14,15,16]`, `seeds: [0,1,2]`,
  `episodes: 24`, `states_without_round_field: 5,040`,
  `bombs_remaining_distribution: {1: 1,038, 2: 3,378, 3: 624}`,
  `feature_dim 106`, engine gate inside.
* **D3 fidelity (5/5, [`fidelity_d3.json`](fidelity_d3.json))** —
  `1_hello_role_shadow`, `2_debug_threads_1`, `3_prefix_2165_checkpoints`
  (2,165/2,165, 0 mismatches, final frame 8,663), `4_r24_18_constants` (18/18
  `first_death_offset` = R24's table), `5_park_frame_8663` — **all PASS**;
  `gate all_pass=True`, `18/18 match`, `deviations=[]`. Every private data dir
  carried a muted `th08.cfg` (generated with `v400_make_cfg.py --out` and
  `--inspect`ed by `v450.prepare_data_dir` before the start), with
  `SDL_AUDIODRIVER=dummy` and `DISPLAY`/`WAYLAND_DISPLAY` unset.
* **Audio (AGENTS.md §9.3, [`audio_audit.json`](audio_audit.json))** — over
  **this round's own** data dirs: `cfg_files 1, decoded 1, errors 0, muted 1,
  audible 0`, `all_muted: true`. (One dir: the D3 gate's. No other engine ran.)
* **Process residue ([`process_residue.json`](process_residue.json),
  [`process_residue_final.json`](process_residue_final.json))** — `pgrep -af
  th08-modern` filtered by `--data-dir .tmp_v890_ruledist`: **0** of this round's
  engines alive. 76 engines of v8.6.0's own evaluation grid were counted and left
  alone. **No `pkill -f` was used at any point.**
* **Checkpoint assertions ([`ckpt_assertions.json`](ckpt_assertions.json),
  [`ckpt_assertions_final.json`](ckpt_assertions_final.json))** — start and
  final, `all_pass: true` at both ends:

| item | sha256 | match |
| --- | --- | --- |
| engine `th08-modern` | `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` | ✅ |
| `seed0/round2.pt` | `4abee069d087872e0a2d726cc975850d760e90a705ddcbee60ea91268e5e61dc` | ✅ |
| `seed1/round2.pt` | `a93746bb5d123165e77e0271edfc1efcdb21931e9e2d706e5a22d613c5476ca4` | ✅ |
| `seed2/round2.pt` | `c69960fcbbd888e9b252d3fa685c22a6730f8dd637dcfc9edab45491d218fa01` | ✅ |
| `.tmp_v860_bombdag/labels_bombdag.jsonl` | `e1f47805efdd37bb92d15c51898aacbef0ca13995a200d7d2e087cd919543c68` | ✅ |

* **Imported sources unchanged** — every gated import still hashes to the value
  v8.3.0/v8.5.0 pinned: `v830_bombpol.py` `a8c6d306…c58e1c`,
  `v450_cf_label.py` `b7ab0ebe…d9947`, `v432_fork_arm.py` `33235506…0cede`,
  `v431_arm_a.py` `96923930…45503`, `v431_dense_driver.py` `1079d93a…fb4f25`,
  `touhou_rl/v420/contracts.py` `1357c69c…946c1`,
  `touhou_rl/v400/gym/actions.py` `a67145b2…93eff`. Recorded but never imported
  (never gated): `v800_bomb.py` `5d85bcb0…0d7bf`, `v860_bombdag.py`
  `3433ae74…47cf8a` (the state source, read only), `v850_distill.py`
  `65e02513…acdf2b` (the round this one is the single-factor successor of,
  copied not imported).

---

## 3. The rule targets — and the gate they fail

[`rule_targets_v86.jsonl`](rule_targets_v86.jsonl) (5,040 rows, 2,543,921 bytes,
sha256 `c013133c000c066aeecc9b97420d3022a6ce71630794e0037095adc51b4ccbae`),
[`rule_target_stats.json`](rule_target_stats.json) (9,716 bytes,
`b9dc09c4…fa12454`). Every row carries
`index/seed/window/episode/decision/frame/H/bombs_remaining/threshold_px/`
`nearest_all_px/nearest_obs_px/nearest_lethal_px_recorded/rule_bomb/rule_bomb_obs/`
`reason/reason_obs/bullets_stored/bullet_count/empty_bullets/nearest_is_first/`
`sorted_by_distance`.

| item | value |
| --- | ---: |
| states | 5,040 |
| `rule_bomb = 1` (`nearest_all_px`) | **153 (3.0357 %)** |
| `rule_bomb = 0` | 4,887 |
| `rule_bomb = 1` (obs-only, lethal minimum) | 34 (0.6746 %) |
| nearest bullet, overall median | **220.0422 px** |

| `bombs_remaining` | threshold | states | **positives** | **negatives** | pos rate | pos (obs-only) | empty stored bullets | **P0** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **3** | 80 px | 624 | **24** | 600 | 3.8462 % | 24 | 0 | **FAIL** |
| **2** | 60 px | 3,378 | **118** | 3,260 | 3.4932 % | 10 | 372 | **PASS** |
| **1** | 40 px | 1,038 | **11** | 1,027 | 1.0597 % | 0 | 513 | **FAIL** |

**The gate.** P0 requires ≥ 50 positives **and** ≥ 50 negatives per stage. The
negatives are satisfied on all three stages (600 / 3,260 / 1,027); the positives
are not, on two of the three (24 and 11). The prereg's S-data branch therefore
applies: **publish the rule-target statistics, do not train.** The executor
enforces this in code — `--mode train` refuses unless
`rule_target_stats.json → p0.all_pass` is true.

**Why `nearest_all_px` and not the stored bullets.** R5 tests `nearest_all_px`
(all bullets); both compact observations on this project keep **lethal bullets
only**, so the stored list's minimum is `nearest_lethal_px` and can only overstate
the distance. v8.6.0 records the true value per row, so the target is taken from
there; the obs-only count is published beside it as a bound (its positives are a
subset: 34 ⊂ 153). See [`p0_forensics.md`](p0_forensics.md) §3 for the row-level
evidence (4,155/4,155 stored minima equal `nearest_lethal_px`; 1,843/4,155 differ
from `nearest_all_px`) and §4 for the mechanism.

---

## 4. Training — not run (S-data)

No checkpoint was trained and none exists: [`ckpt_manifest.json`](ckpt_manifest.json)
lists **0** files under `.tmp_v890_ruledist/ckpt/`. `--mode train --seed {0,1,2}`
was **not invoked**; had it been, it would have raised `S-data: P0 is not
satisfied`. There is therefore no `train_seed{0,1,2}.json`, no `mean p(bomb)`, and
no train loss to report — the round's output is the targets and the gate.

The design was **not** modified to make training possible: λ = 0.5, Adam 1e-3,
batch 256, 200 epochs, 3 seeds, `Policy(106,64,36)`, the 36 counterfactual labels
and the `logit(2c+1) − logit(2c)` bomb logit are all identical to v8.5.0, and the
only change from v8.5.0 is the state source (prereg §1).

---

## 5. Evaluation — not run

No engine was driven for evaluation, so there are no `eval_d*` files, no
`diagnostics.json`, no `bomb_usage.json`, and no survival/`p(bomb)` numbers. The
3 × 3 × 4 × 300 grid belongs to a round whose P0 passes. The D3 gate (§2) is the
only engine work; it used its own data dirs and left 0 processes.

---

## 6. Deviations and corrections (complete list)

1. **The round stops at S-data.** Prereg §2's prerequisite fails on two of the
   three stages, so nothing was trained and nothing was evaluated. Everything the
   prereg's §4 file list would have produced for training/evaluation
   (`train_seed*.json`, `eval_d*.jsonl`, `bomb_usage.json`, `diagnostics.json`) is
   absent by design, not missing.
2. **Two recounts were needed, and the first two hypotheses were wrong.** The
   first pass (obs-only, silently skipping the 885 empty-bullet rows) gave
   0 / 10 / 24 positives and the reading "the rule almost never triggers"; the
   second read "the observation truncation dropped the nearest bullet, so the
   target cannot be computed offline". **Neither is correct as stated.**
   (a) The 885 empty rows are not skips: they carry `nearest_all_px` (median
   304.23 px) and none is a positive, but dropping them breaks the denominators —
   the obs-only **positives** are the same 24 / 10 / 0 by stage, while the
   **negatives** change from 600 / 2,996 / 525 to 600 / 3,368 / 1,038 on the full
   5,040 rows. (b) The stored 12 bullets **are** the nearest 12 **lethal**
   bullets — the minimum over them equals the recorded `nearest_lethal_px` on
   4,155/4,155 non-empty rows — so the truncation does not drop the nearest
   lethal bullet; what it drops is **non-lethal** bullets, which R5's
   `nearest_all_px` does include (the two differ on 1,843/4,155 rows). (c) The
   target therefore **is** recoverable offline, from
   `row['behaviour']['nearest_all_px']`; recovering it gives 24 / 118 / 11 and
   P0 still fails. The published stats carry **both** definitions so the reader
   can see the difference.
3. **`p0_forensics.md` is corrected relative to the request that asked for it.**
   The request asked for the evidence that "v8.6's 12 bullets are not the nearest
   12" and the conclusion "the rule target must be recorded at collection or
   recomputed from the live engine". What the data shows is narrower and is what
   the file says: the 12 are the nearest 12 **lethal** (so the lethal minimum is
   not lost), the observation **does** lose `nearest_all_px` (lethal-only filter,
   in both batches), that value **is already recorded** at row level for all
   5,040 v8.6.0 states, and P0 fails **even with it** because of the state
   distribution.
4. **The D3 gate was run although this round used no evaluation engine.** It is
   the checklist's step 2 and it certifies the engine binary; it cost 7.458 s and
   started its own engines in its own data dirs (audio muted, 0 residue). It is
   published as [`fidelity_d3.json`](fidelity_d3.json).
5. **The box was shared with v8.6.0's evaluation grid.** When this round started
   (11:52) **66** `th08-modern` processes of that grid were alive, load average
   ~11 on 20 cores; at the last process audit **76** were alive and **0** were
   this round's. They were counted and left alone — no `pkill -f`, no kill, no
   wait loop beyond the audits. This round's wall times (0.14–7.46 s) are its own
   process times and are not comparable to an idle box.
6. **`v890_ruledist.py` is a copy-and-edit of `v850_distill.py`**, which the task
   allows ("可从 v850_distill.py 复制改写"). `v850_distill.py` itself was not
   modified (`65e02513…acdf2b`, recorded in every assertion). The three edits are:
   the state source (path/sha/states/`episode` instead of `round`), the P0 gate
   (per-stage positive/negative counts, hard refusal to train), and the
   two-distance forensic block. The training and evaluation code paths are
   byte-for-byte the v8.5.0 design and were **never invoked**.
7. **`git add`/`git commit` were not run**; no existing version directory,
   preregistration, `touhou_rl/**`, `third_party/**` or `.tmp_v860_bombdag/**`
   asset was modified (the state file is read-only in every mode and is hashed at
   both ends of the round).
8. **`v860_bombdag.py` and `v850_distill.py` are recorded, not gated**, in every
   assertion: this round reads the former's output and copies the latter's source,
   and neither is imported at run time.
9. **The published `commands.tsv` is the snapshot taken inside `publish`** and
   therefore lacks the `publish` and `verify` rows; the work-dir copy has them.
   `verify.json` is written **after** `manifest.json` closes, so it is the one
   file the manifest cannot cover.

---

## 7. What this round does **not** claim

* **not** "the rule was distilled" — no training ran;
* **not** "v8.6.0's data is broken" — the labels are valid (5,040/5,040 feature
  hashes and 36-label vectors reproduce) and the rule input is recorded;
* **not** "R5 is rarely triggered in play" — the 3.04 % is a property of **this
  batch's sampling**, which is produced by a policy that obeys R5; the live
  trigger rate of the rule is a different quantity and is not measured here;
* **not** "the 12-bullet observation cap degrades the 106-dim features" — the
  features reproduce on 5,040/5,040 rows and their definition is unchanged;
* **not** "the candidate head's results are explained" — no v8.6.0 evaluation
  number is re-derived or interpreted here;
* **no statistics** — no Fisher, no permutation, no p-value, no Holm, no
  confidence interval, no "significant"; P1–P4 are **not** decided and no
  S-ok/S-partial/S-fail branch is declared (P0 fails first).

---

## 8. Files in this directory

[`manifest.json`](manifest.json) is authoritative for every published file.
`verify.json` is the independent recomputation (raw-file recount of P0, row
alignment of the target file, and a re-hash of every manifest entry); it is
written after the manifest closes and is therefore not covered by it.

| file | bytes | sha256 |
| --- | ---: | --- |
| `rule_targets_v86.jsonl` | 2,543,921 | `c013133c000c066a…` |
| `rule_target_stats.json` | 9,716 | `b9dc09c47eed9fa0…` |
| `p0_forensics.md` | 11,983 | `f63fce702fa8973b…` |
| `preflight.json` | 5,086 | `2c8a52df711f28b9…` |
| `ckpt_assertions.json` | 4,880 | `d7552281cc0ed860…` |
| `ckpt_manifest.json` | 1,070 | `56e9511fd705817d…` |
| `fidelity_d3.json` | 26,627 | `a52daf1044352586…` |
| `audio_audit.json` | 932 | `913a7c2174d5b78e…` |
| `process_residue.json` | 617 | `c7887ed36ef2db0f…` |
| `commands.tsv` | see `manifest.json` | one row per invocation |
| `RUN.md` | this file | covered by the manifest |
| `scripts/verify.py`, `scripts/p0_recount.py`, `scripts/exec.sh` | see `manifest.json` | — |
