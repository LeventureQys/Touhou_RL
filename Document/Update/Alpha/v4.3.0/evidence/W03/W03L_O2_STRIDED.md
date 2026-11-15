# W03L — a cross-tape strided re-sample of real O2 (6/9 → 9/9 `move` coverage)

Date: 2026-11-15
Repo: `D:\WorkShop\Touhou_RL` (source root; `/root/Touhou_RL` is a **stale checkout**,
used only for the engine binary)
Slice: W03L — **data collection**, not a new capability. It re-samples the same
audited clear tape with a stride so the O2 corpus stops being a *prefix*, and it
re-generates the manifest with truthful accounting.

> **Scope statement (read this first).** Nothing here is a capability, capability
> gate, or learning result. The deliverable is a corpus of **1,099 real O2
> decisions** whose label space covers **all 9 `move` values and 17 of the 18
> candidates**, together with a `v430-manifest/1` whose advertised numbers equal
> the numbers on disk. The BC run in §9 is a **plumbing smoke test on 1,099
> samples, 2 epochs, 8 optimizer steps**; it says nothing about learning.
> `shoot`/`bomb` remain constant by construction of the candidate space (§6).

---

## 1. Why a strided sample: the prefix was not representative

The audited tape's behaviour is concentrated in its opening block and diversifies
later, so a leading slice of the schedule is a bad sample of it. All four rows
below were measured from the tape itself (`slots.jsonl` sha
`f759de7d…`), counting the executed `(focus, move)` candidate of each decision:

| sample rule | decisions | `move` distinct | `cand` distinct |
| --- | --- | --- | --- |
| `schedule[:64]` (= `o2-64`) | 64 | 5/9 | 9/18 |
| `schedule[:1000]` (= `o2-1k`) | 1,000 | 6/9 | 10/18 |
| **`schedule[::6]` (this corpus)** | **1,099** | **9/9** | **17/18** |
| whole tape | 6,596 | 9/9 | 17/18 |

The 1,000-decision corpus is missing `move` **1, 5, 8**; the strided corpus misses
none. The only missing candidate is `cand = 9` (`focus=1, move=0`), which the tape
itself never executes.

**`o2-64/` and `o2-1k/` were not modified by this slice** (their newest file
timestamps are still 2026-11-15 12:11:14 and 12:21:33, before this capture ran).

---

## 2. Sampling semantics: filter the **capture**, never the tape

`capture_o2.py` gained exactly one optional parameter,
`--keep-every N` (alias `--stride N`, default `1`). Decision `i` is *saved* iff
`i % N == 0`, i.e. the sample is the `schedule[::N]` positions of the **same**
schedule. The schedule itself is replayed in full, decision for decision.

Why the tape must not be rewritten to every N-th row: `load_clear_schedule`
returns a *schedule*, and the capture executes it in order, each decision
advancing two game frames. Dropping five of every six rows would make the engine
execute the surviving actions **back to back**, producing a different trajectory
while still stamping the rows with the original `decision_frame` values. The
capture would look right and be wrong. So the filter sits at the encoding step:

* every decision is still **executed** (§4 shows the replay length is identical
  to `o2-1k`: 6,598 decisions attempt, 13,195 frames, `stage_clear`);
* only decisions whose index is not a stride position are skipped for encoding;
* `--collect` keeps its meaning = **saved samples**; `run.decisions` stays the
  **replayed decision** count, and both are written to `meta.json`
  (`run.replayed_decisions`, `run.saved_samples`).

Two consequences worth naming:

1. **1,100 positions → 1,099 samples.** Decision index `0` ends with only 3
   frames in the 4-frame stack (`short_stack`, pre-existing capture behaviour:
   the first decision has no 3 predecessor frames), so it is refused *before* the
   stride gate. The strided corpus is therefore `schedule[6::6]`, not
   `schedule[::6]`: 1,099 decisions, `first=6`, `last=6594`, step 6.
   Coverage is unaffected (dropping `cand=7` at index 0 leaves 9/9 moves and
   17/18 candidates).
2. **`--tail 0`.** With no synthesised tail the schedule is exactly
   `8 leading history + 6,588 recorded slots = 6,596` decisions, so
   `schedule[::6]` is the intended 1,100 positions. The replay still reaches the
   stage transition because the recorded slots end there.

---

## 3. Identity (what was actually run)

| item | value |
| --- | --- |
| engine | `/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern` |
| engine sha256 | `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` |
| tape (`slots.jsonl`) | `.tmp_v420_data/T04/dataset/d1-s0d1-n3300-seed12-shard/slots.jsonl` |
| tape sha256 | `f759de7dacc78343dec8825da5de14942e749b8a5a2c5319304e513d1618b7b9` |
| tape sha256 = ruler's frozen `CLEAR_SHARD_SHA256` | **yes** (`sha256_matches_frozen: true`; the capture refuses any other shard) |
| tape `meta.json` sha256 | `11d30d315bdc41b7495e63d8ffba02ba87367e5a0a0ca8f2726b9bac90e92bc1` |
| tape loader | `touhou_rl.v420.rl.load_clear_schedule` (read-only reuse) |
| schedule | 6,596 decisions = 8 leading + 6,588 slots; `--tail 0` |
| episode seed | 12 |
| **sampling** | `--keep-every 6` (alias `--stride 6`), `schedule[::6]` → 1,100 positions, **1,099 saved** |
| **private data dir** | `.tmp_v430_o2/engine-data-strided` (474 MB `cp -a` of `.tmp_v430_o2/engine-data`, 4.9 s, **kept and private to this run**) |
| shared data dir usage | **none** — `o2-1k`'s `.tmp_v430_o2/engine-data` was neither read nor written by this run |
| shm name | `w03l_o2_strided` / `w03l_o2_strided_dry` (private, so a concurrent capture cannot collide) |
| encoder | `touhou_rl/v430/observation.py::encode_o2`, `ENCODER_VERSION = o2-th08-encoder/1` |
| **alpha gate** | `alpha_mode="any"`, passed explicitly on the command line and recorded in `meta.json` (`alpha_mode`, `alpha_modes_recorded = ["any"]`) and in every `O2Observation.alpha_mode` |
| `o2_digest` | `cb1616e4cbfa928e691c2fb8497a84c813bded528064cf2b0aa1e30fa86b85b5` (identical to W03H/W03I — same encoder block) |
| legacy encoder used | **no** |
| output dir | `.tmp_v430_o2/o2-strided` |

`capture_o2.py` was edited by **two** slices: W03K added `assert_written_shapes`
(a generator self-check, now exercised as `meta.shape_guard`), and this slice
added the optional `--keep-every` parameter (§11). The additions do not
interact: the stride gate decides *what is encoded*, the shape guard checks that
every artefact holds one row per encoded decision.

---

## 4. What was written

Two runs, both `exit_code=0`, both `stop_reason = stage_clear`.

| run | rule | samples | decisions replayed | frames | `stage_clear` at | wall |
| --- | --- | --- | --- | --- | --- | --- |
| pre-flight (`.tmp_v430_o2/o2-strided/_dry-keep60`) | `--keep-every 60`, `--collect 110` | **109** (of 110 positions) | 6,598 | 13,195 | decision 6,597 | 179.9 s |
| **full** (`.tmp_v430_o2/o2-strided`) | `--keep-every 6`, `--collect 1100` | **1,099** (of 1,100 positions) | 6,598 | 13,195 | decision 6,597 | 223.4 s |
| cross-check: `o2-1k` (W03H, `--collect 1000`) | prefix | 1,000 | 6,598 | 13,195 | decision 6,597 | 172.1 s |

The full run and the historical prefix run advance the engine to the **same
decision, the same frame and the same stop reason**. That is the observable proof
that the stride changed only what was *saved*, not the trajectory that was
*replayed*.

`o2-strided/` top level (**572 MiB / 600,058,042 B** including the two report
JSONs; the manifest's `counts.bytes_total` = 600,030,621 B covers the eight
artefacts it lists):

| file | bytes | shape | sha256 |
| --- | ---: | --- | --- |
| `global_rgb.npy` | 253,209,728 | `(1099, 3, 240, 320)` | `ed70de270f5efc68f2934472b4615e77f934210779924d740edf55ca40faa4b8` |
| `play_gray.npy` | 189,063,296 | `(1099, 4, 224, 192)` | `cd9689750cad9db0ac7e0870507b5b6f130b8233ef8e71544302ecc14f626948` |
| `hud_gray.npy` | 78,776,448 | `(1099, 1, 448, 160)` | `7587a178a058824f215b4c864ba86c7b1f7c5877f9571434ac64d5510559c74a` |
| `executed_key_history.npy` | 61,672 | `(1099, 8, 7)` | `eab5673ea4c5b7ecc356bac4e5468eede3ddf656160e8249c8474fdc5a421123` |
| `history_valid.npy` | 8,920 | `(1099, 8)` | `b663dafecc16576640d8b168faacd044b8a2d89add59f8fa69c4ee9e7da70da7` |
| `elapsed_capture_s.npy` | 4,524 | `(1099,)` | `72ee2e9f878032c17d7b8bc39e925fa74092251abd2ffbbf65bf8960b4775a0f` |
| `labels.jsonl` | 262,705 | 1,099 lines | `405a400ffc181bd774db7572e45c9f4794cf88faf3d0bf50fb38db9a3ada0f6b` |
| `frames_rgba.npy` (evidence only) | 78,643,328 | `(64, 480, 640, 4)` | `43dfbbe5fb86f5191b6963fd93a48e68eb4c6f51653df6f12f7244d676eef721` |
| `manifest.json` | 8,360 | — | `5d8b405db0c6413bd4596f77c5f7a332d7d7c65be01757eebc8ee66dfdbbed8f` |
| `meta.json` | 8,870 | — | `3cc1ef56617a89825a1977026495e241b37cc19a689f926a198d0518ae3eec73` |
| `bc_smoke.json` | 7,941 | — | `245f4d5a7a03e61e33e258cfb7e4cc806880b14991156860b2fe0586f8da70a1` |
| `labels_provenance.json` | 2,250 | — | `396b1de99000ed62e961757fa35bf4787f586fe69a0367a8f9ab5ec9e13a4663` |

Whole tree (including the pre-flight capture `_dry-keep60/` = 51,720,942 B and
the two helper scripts in `_runner/` = 6,237 B): **651,785,221 B**.

### `frames_rgba.npy` policy

`frames_rgba.npy` holds the raw `(480,640,4)` readback of only the **first 64**
saved decisions (78.6 MB), not all 1,099 (which would be 1.3 GB). `meta.json`
says so in `blank_audit.raw_frames`
(`limit: 64`, `retained: 64`, `covers: "decisions [0,64)"`) and in
`blank_audit.frames_rgba_note`; `shape_guard` checks it as `<= labels` instead of
`== labels`. The alpha audit does **not** depend on it: it is measured per saved
decision during the replay (same frame, same plane).

---

## 5. Label-space statistics (measured on `labels.jsonl`, 1,099 rows)

| head | distinct | histogram |
| --- | ---: | --- |
| `move` | **9 / 9** | `{0:468, 1:28, 2:203, 3:172, 4:32, 5:21, 6:78, 7:76, 8:21}` |
| `focus` | 2 / 2 | `{0:922, 1:177}` |
| `cand` | **17 / 18** | `{0:468, 1:22, 2:118, 3:108, 4:28, 5:19, 6:68, 7:74, 8:17, 10:6, 11:85, 12:64, 13:4, 14:2, 15:10, 16:2, 17:4}` |
| `shoot` | 1 | `{1:1099}` |
| `bomb` | 1 | `{0:1099}` |

* `move` missing: **none**. `cand` missing: **`[9]`** (= `focus=1, move=0`), which
  the tape never executes either.
* Saved decisions are exactly `{6, 12, …, 6594}` (1,099 indices, step 6),
  `first_collected_decision = 6`, `last_collected_decision = 6594`,
  `first_frames = 14`, `last_frames = 13190`.
* **The same statistics computed straight off the tape at those same 1,099
  positions are identical, value for value** (`w03l_label_stats.py` prints both
  blocks side by side; the corpus and the tape histograms agree exactly, which is
  the label-side proof that the pixels are that trajectory's pixels at those
  decisions).

`shoot`/`bomb` are constant for instrument reasons, not capture defects: the
v4.2.0 candidate space is `(focus, move)`, `gym_actions.cand_to_mask` always ORs
`BTN_SHOOT`, and the ruler fixes `BOMB` to 0 (`meta.run.label_space_note`). A
four-head BC loss on this corpus can only be supervised on `move`/`focus`.

---

## 6. Provenance (the guard is on, and it fired)

`meta.run.label_provenance_counts = {"recorded_slot": 1098, "leading_history": 1}`,
`tail_repeat = 0`. The single `leading_history` label is decision 6, which is
inside the tape's 8-decision leading block (recorded by slot 0's
`history`/`history_frames`, not invented). Every other label has its own
`slots.jsonl` row.

Two decisions were **refused** and recorded in `meta.run.skipped_decisions`
rather than silently dropped:

| index | refusal | why |
| --- | --- | --- |
| 0 | `short_stack` (`frames_in_window: 3`) | pre-existing: the first decision has no 4 consecutive frames |
| 6596 | `label_without_tape_row`, `provenance: beyond_schedule` | past the last schedule entry; a label there has no tape row |

The `tail_repeat` / `beyond_schedule` guards therefore still reject: with
`--tail 0` there is no synthesised tail to collect, and the one beyond-schedule
decision was refused. The offline guard test (`.tmp_v430_o2/_selftest_guard.py`,
`--tail 64` schedule, no engine) still returns `guard_ok = True`, including
`first_tail_is_tail`, `last_schedule_is_tail` and `beyond_schedule_refused`.

Independent cross-check (`verify_labels_provenance.py`, exit 0): 1,098
`recorded_slot` labels cross-checked against the tape's own rows on
`cand`/`mask`/`decision_frame` — **0 mismatches**; `tail_repeat_labels = 0`.

---

## 7. Manifest accounting: the two `o2-1k` defects are structurally absent

### 7.1 Defect A — `counts.decisions` inflated 8×

`o2-1k` published `{"decisions": 8000, "episodes": 8, "shards": 8}` for a
1,000-decision corpus, because each of the eight artefacts was listed as its own
shard and each filled a full `decisions` column, and
`manifest._check_counts` demands `counts.decisions == Σ shard.decisions`
(`touhou_rl/v430/data/manifest.py:1531`).

This manifest uses **one logical shard** — one capture is one tape replay is one
episode — and lists the eight artefacts as *files* of that shard:

```json
"shards": [ { "path": "…/labels.jsonl", "decisions": 1099, "labels_valid": 1099,
              "episodes": [12], "observation_file_count": 6, … } ],
"counts": { "shards": 1, "decisions": 1099, "episodes": 1, "labels_valid": 1099,
            "files": 8, "observation_channels": 6, "evidence_files": 1,
            "saved_samples": 1099, "replayed_decisions": 6598,
            "bytes_total": 600030621 },
"files": [ …8 entries with path/size_bytes/sha256/digest_scope/role/shape… ],
```

So the library invariant `Σ shard.decisions == counts.decisions` **and** the
truth `counts.decisions == labels rows == global_rgb.shape[0]` hold at the same
time. The channel/file count lives in `counts.files` /
`counts.observation_channels` and never multiplies the decision count;
`counts_note` spells that out inside the document.

Self-check (three sources, fatal on disagreement):

* pre-write `sample_count_check` — `labels.jsonl` lines = 1,099, every
  per-decision `shape[0]` = 1,099, `meta.counts.collected` = 1,099 →
  `ok: true`;
* post-write `verify_written_manifest` — re-reads the document **from disk** and
  compares it with the artefacts: `three_way_equal: true`,
  `counts.episodes == 1`, `counts.shards == len(shards)`, `dataset_id` ends in
  `1099`, and a `files × samples` product is rejected by name
  (`problems: []`). A mismatch returns exit code **4** and leaves no claim
  behind.

### 7.2 Defect B — hard-coded `-64` in `dataset_id` and the disclaimer

`dataset_id` is now derived from the artefacts, not from a constant:

```
w03h-o2-capture-seed12-strided-1099
```

(`strided-` appears only when `--keep-every > 1`, so a stride-1 capture keeps its
historical `w03h-o2-capture-seed12-<n>` form; the `-64` literal is gone.) The
manifest `disclaimer` and the BC `disclaimer` are f-string-derived from the same
numbers (`1,099` samples, `6,598` replayed, `every 6-th`, `8` artefacts) — no
number in either string is typed in. `manifest.label_coverage` reports the label
space from the corpus' own rows with denominators from the action space
(`gym_actions.MOVES_PER_CAND = 9`, `N_CAND = 18`).

---

## 8. `validate_manifest` and `assert_o2_compatible`

```
validate_manifest(manifest, base_dir=repo_root)
  valid = True   complete = True   exit_code = 0
  problems = []  degradations = []
  checks_performed = [identity, shards, counts, ancestors, digest_scope, status]
  statistics = {"decisions": 1099, "episodes": 1, "labels_valid": 1099,
                "shards_listed": 1, "shards_readable": 1,
                "shards_by_split": {"dev": 1}, "ancestor_groups": 1,
                "ancestor_ids": 1, "bytes_listed": 262705}
assert_o2_compatible(manifest) -> Manifest named
                                 w03h-o2-capture-seed12-strided-1099  (no raise)
manifest sha256 (text/lf-normalized) = 5d8b405db0c6413bd4596f77c5f7a332d7d7c65be01757eebc8ee66dfdbbed8f
```

`not_checked` is reported, not hidden: the validator hashes every shard's bytes
but does not interpret them, and it did not compare `obs_digest`/`action_digest`/
`label_digest` against an externally supplied expected value (none was
supplied), nor run the unregistered-shard scan (it needs explicit `roots`).

`obs_shapes` are declared and compared against `observation.O2_SHAPES`:
`global_rgb (3,240,320)`, `play_gray (4,224,192)`, `hud_gray (1,448,160)`,
`executed_key_history (8,7)`.

---

## 9. BC smoke on the 1,099 real samples (plumbing only)

`lr=1e-4`, `seed=0`, `memory_mode="none"`, batch 16 (69 batches), 2 epochs,
`max_steps=8`:

| item | value |
| --- | --- |
| `n_samples` | 1,099 |
| `n_batches` | 69 |
| `n_steps` | 8 |
| first loss | 4.341777324676514 |
| last loss | 3.2010152339935303 |
| `model_supplied` | true (a real forward pass produced the logits) |
| parameters / trainable | 2,726,671 / 2,726,671 |

**This is not a learning result.** Eight optimizer steps on a corpus whose only
supervisable heads are `move`/`focus` cannot show learning; it shows that real
O2 pixels of all 1,099 decisions reach a `train_bc` loss.

---

## 10. Alpha / blank audit (all 1,099 decisions)

* `alpha_mode = "any"`, `alpha_modes_recorded = ["any"]`;
* measured per saved decision during the replay:
  `alpha_min_of_min = 18`, `alpha_max_of_max = 255`,
  `alpha_mean` min 187.63 / median 231.34 / max 249.58,
  frames with alpha all-255: **0**, frames with any alpha 0: **0**;
* `blank = 0`, `near_blank = 0` (out of 1,099), `global_rgb_mean` min 13.32 /
  median 37.67 / max 166.13 → no placeholder frames in the sample;
* orientation witness recorded (`top_left_rgb = [182,178,178]`,
  `bottom_right_rgb = [181,177,178]`, `uniform = false`); no extra flip applied,
  no un-premultiply, no upsample, no zero-fill, no channel synthesis anywhere in
  this pipeline.

---

## 11. Changes to `capture_o2.py` (line by line)

Only **additions**, all inert at the default `--keep-every 1`; the pre-existing
paths (`o2-64`, `o2-1k`) are untouched and re-running with `1` reproduces the old
prefix behaviour.

| lines | change |
| --- | --- |
| 21–41 | module docstring: new section “Stratified sampling across the whole tape” — states the filter-during-replay rule, why filtering the tape is wrong, the `--collect` = saved-samples meaning and the 6,596→1,100 mapping |
| 254–256 | `keep_every = int(args.keep_every)` + `if keep_every < 1: raise CaptureError(...)` in `capture()`, before any engine is started |
| 296–298 | new counter `keep_every_skipped = 0` (unsampled replayed decisions; counted, not listed, so `meta.json` stays small) |
| 419–428 | the stride gate: `if keep_every > 1 and index % keep_every: keep_every_skipped += 1; executed.append(key_row(mask)); continue` — placed **after** the provenance/short-stack checks (a refused label is still refused and counted) and **before** the `--collect` cap (which now counts saved samples) |
| 554–567 | `report["run"]` gains `keep_every`, `keep_every_alias`, `sample_rule`, `sample_rule_note`, `schedule_sample_positions`, `keep_every_skipped`, `replayed_decisions`, `saved_samples` |
| 591–593 | `report["counts"]` gains `saved_samples`, `replayed_decisions`, `keep_every_skipped` (the replayed-vs-saved split is recorded, never conflated) |
| 776–783 | new argparse option `--keep-every` / `--stride`, `dest="keep_every"`, `type=int`, `default=1` |
| 806 | `--keep-every` added to the `main()` minimum-value validation tuple |

Nothing else in the file changed by this slice (W03K's
`assert_written_shapes`, now at line 611, is separate and untouched; it is what
fills `meta.shape_guard`).

---

## 12. Reproduce

```bash
# 1. private data dir (the engine writes score.dat / th08.cfg there)
cd /mnt/d/WorkShop/Touhou_RL
cp -a .tmp_v430_o2/engine-data .tmp_v430_o2/engine-data-strided      # 474 MB, 4.9 s

# 2. pre-flight: 110 positions, no raw frames
bash .tmp_v430_o2/o2-strided/_runner/w03l_capture.sh \
  .tmp_v430_o2/o2-strided/_dry-keep60 60 110 0 w03l_o2_strided_dry \
  .tmp_v430_o2/o2-strided-dry.log                                    # exit 0, 109 samples, 202 s

# 3. full run: 1,100 positions, raw frames capped at 64
bash .tmp_v430_o2/o2-strided/_runner/w03l_capture.sh \
  .tmp_v430_o2/o2-strided 6 1100 64 w03l_o2_strided \
  .tmp_v430_o2/o2-strided.log                                        # exit 0, 1099 samples, 243 s

# 4. verification
python .tmp_v430_o2/o2-strided/_runner/w03l_label_stats.py --out-dir .tmp_v430_o2/o2-strided
python .tmp_v430_o2/verify_manifest_bc.py --out-dir .tmp_v430_o2/o2-strided
python .tmp_v430_o2/verify_labels_provenance.py .tmp_v430_o2/o2-strided
python .tmp_v430_o2/_selftest_guard.py
```

The runner wraps the capture in `timeout 300 xvfb-run -a -s "-screen 0
1024x768x24" /root/v400-venv/bin/python`, i.e. one engine per run (concurrency
1) with the 300 s fallback; the full run finished in 243 s of wall clock. The
expansion workers (`.tmp_v420_expand/run_all.sh`, `run_worker.sh` × 3 and the
three `th08-modern` instances they own) were **not** touched, and this capture
points at its own `--data-dir`, so no `score.dat`/`th08.cfg` race exists.

---

## 13. What this slice does *not* claim

* Not a capability, not a gate result, not evidence of learning.
* Not a new observation contract: `obs-v430-o2`, `o2-th08-encoder/1`, unchanged.
* Not a resampled or synthesised corpus: every pixel is a live
  `shm.pixels()` readback of the replayed tape; no un-premultiply, upsample,
  zero-fill or channel synthesis.
* Not a re-labelled corpus: every label is read back from the mask that was
  actually executed, and 1,098 of 1,099 labels were cross-checked against the
  tape's own rows with 0 mismatches.
* `shoot`/`bomb` diversity is **not** delivered here; it needs a tape whose
  teacher released shoot and used bombs.
