# W03H — real O2 capture + real-O2 → BC closed loop

Date: 2026-11-15
Repo: `D:\WorkShop\Touhou_RL` (source root; `/root/Touhou_RL` is a **stale checkout** and was used only for the engine binary)
Slice: W03H, the first slice in this project that records **real** `obs-v430-o2` pixels and trains BC on them.

> **Scope statement (read this first).** This slice proves that the *plumbing* works:
> real engine frames → `encode_o2` → O2 `.npy` → `v430-manifest/1` →
> `validate_manifest` / `assert_o2_compatible` → `train_bc`. The BC run below uses
> **64 samples for 2 epochs (8 optimizer steps)**; it is a **smoke run on 64
> samples and is NOT evidence of learning, capability, generalization or any gate
> result**. No capability claim of any kind is made here. Only the `move`/`focus`
> heads carry any learning signal on this corpus (§10).

---

## 1. Identity (what was actually run)

| item | value |
| --- | --- |
| engine | `/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern` |
| engine sha256 | `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` |
| tape (`slots.jsonl`) | `.tmp_v420_data/T04/dataset/d1-s0d1-n3300-seed12-shard/slots.jsonl` |
| tape sha256 | `f759de7dacc78343dec8825da5de14942e749b8a5a2c5319304e513d1618b7b9` |
| tape sha256 = ruler's frozen `CLEAR_SHARD_SHA256` | **yes** (`tools/rl/v420_eval_shoot.py:138`) — the capture pins this value and refuses any other shard |
| tape `meta.json` sha256 | `11d30d315bdc41b7495e63d8ffba02ba87367e5a0a0ca8f2726b9bac90e92bc1` |
| tape loader | `touhou_rl.v420.rl.load_clear_schedule` (read-only reuse, not copied) |
| tape slots / schedule decisions | 6,588 recorded slots → 6,660 decisions (8 leading + 6,588 + 64 tail) |
| episode seed | 12 |
| **private data dir** | `.tmp_v430_o2/engine-data` (474 MB `cp -a` of `.tmp_v420_data/T06/engine-data-gamesrc`, **kept**) |
| encoder | `touhou_rl/v430/observation.py::encode_o2`, `ENCODER_VERSION = o2-th08-encoder/1` |
| **alpha gate** | `alpha_mode="any"` (audited opt-in; recorded in `meta.json` and in every `O2Observation.alpha_mode`) |
| `o2_digest` | `cb1616e4cbfa928e691c2fb8497a84c813bded528064cf2b0aa1e30fa86b85b5` (unchanged by `alpha_mode`) |
| legacy encoder used | **no** — `obs_spec_v3.encode_planes_v3` is never called by the capture |
| instrument | `.tmp_v430_o2/capture_o2.py` (new file, git-ignored); no existing file was modified |

### Why the tape is `T04` and not `T02c`

The task named `.tmp_v420_data/T02c/rebuild/d1-s0d1-n3300-seed12-shard/2f`. That
path exists, but its slots store the executed candidate under `cand`, **not**
`executed_cand`, so `touhou_rl.v420.rl.load_clear_schedule` raises
`KeyError: 'executed_cand'` on it (measured — see `EXIT_CODES_W03H.txt` §1).
`.tmp_v420_data/T04/dataset/d1-s0d1-n3300-seed12-shard/slots.jsonl` is the layer
that carries `executed_cand`/`executed` **and** whose bytes hash to the ruler's
frozen `CLEAR_SHARD_SHA256`. The capture therefore uses T04 for the *schedule*
and reads **no** pre-existing `.npy` observation of any corpus — every pixel in
this dataset comes from a live `shm.pixels()` readback.

### The samples are inside the recorded range, not the tail

`--tail 64` only extends the schedule past the recording so a *full* replay can
reach the stage transition. The 64 collected decisions are decisions
**1 … 64** (`first_collected_decision = 1`, `last_collected_decision = 64`),
i.e. entirely inside the 6,588 recorded decisions. **No tail (synthesised
repetition) action is in the dataset.**

---

## 2. What a decision carries here, and the frame accounting

`decision_steps("consecutive", 2)` returns `units=2, want=1`: one decision is two
single-frame `STEP` requests and `shm.pixels()` is read after **each** frame, so
the capture feeds `deque(maxlen=4)` with **every** frame and hands `encode_o2` the
four **genuinely consecutive** frames ending at the decision's last frame
(oldest first). One-frame-per-decision sampling would have produced
`[t-6, t-4, t-2, t]`, which is not a consecutive stack.

* `encoders`/`renders`: **13,195** `pixels()` readbacks for 13,195 frames.
* first collected decision: index 1, stack = frames 4 (window filled).
* decision 0 is recorded as `skipped_decisions: [{decision: 0, reason:
  short_stack, frames_in_window: 3}]` — a 3-frame window is refused rather than
  zero-padded, exactly as the encoder requires.
* `invalid_frames = 0`, `warmup_frames = 0`, `zero-frame STEP` errors: none.
* `shm.pixels()` already returns **top-down** RGBA; the capture applies **no**
  extra flip. Orientation witness recorded in `meta.json`:
  `top_left_rgb=[215,213,213]`, `bottom_right_rgb=[215,213,213]`, `uniform=false`
  for the first stack's oldest frame.

---

## 3. The real engine run

| field | value |
| --- | --- |
| command | `bash .tmp_v430_o2/run_capture.sh <out> 13195 64 290` → `xvfb-run … python .tmp_v430_o2/capture_o2.py --max-frames 13195 --collect 64` |
| frames actually played | **13,195** |
| decisions executed | **6,598** |
| collected decisions | **64** |
| `stop_reason` | `stage_clear` (the stage counter advanced at the 13,195-frame budget — the replay reproduced the recorded clear) |
| wall time (engine loop) | **150.425 s** |
| process exit code | **0** |
| engine concurrency | 1 (the v4.2.0 expansion run was left untouched) |
| cost contention | `load ~2.8` throughout; no timeout was hit (150 s ≪ the 300 s fallback) |

---

## 4. `.tmp_v430_o2/o2-64/` — actual bytes and shapes

Directory total: **109,025,255 bytes** (sum of the seven artefacts below:
109,004,272 bytes, plus `meta.json`, `manifest.json`, `bc_smoke.json` and
`labels_provenance.json`).

| file | shape | dtype | bytes | sha256 |
| --- | --- | --- | --- | --- |
| `global_rgb.npy` | **(64, 3, 240, 320)** | uint8 | 14,745,728 | `0c2ccae67b843c5e6781c43e23050567c829f7405385ae7083f47398be1ce97b` |
| `play_gray.npy` | **(64, 4, 224, 192)** | uint8 | 11,010,176 | `27292db0272b9284662a3c01e8702f2ed0dfeaae1f32b94aa24ffd5db5d59bf4` |
| `hud_gray.npy` | **(64, 1, 448, 160)** | uint8 | 4,587,648 | `c6424dcf941b4b20c2f8c6fe7a876b1e7f69a48f86b2017b1857242c5709d294` |
| `executed_key_history.npy` | **(64, 8, 7)** | uint8 | 3,712 | `0e9d67c1740b1f8ea21ffe1a98068d000587453e0f48c70c698b9a22ad38ee99` |
| `history_valid.npy` | **(64, 8)** | bool | 640 | `56b18879fae33df061a5d0fd4ce51eb1a54a834009be35945b7c313b0ea504ba` |
| `elapsed_capture_s.npy` | **(64,)** | float32 | 384 | `7064fb4ebc0353b56f1530dcf3ef9059a77e6c03a3e34cf1e27eb1acee64d75a` |
| `frames_rgba.npy` | **(64, 480, 640, 4)** | uint8 | 78,643,328 | `2ece7c5b4ff6e7fbd6246ea8a755794a276f2d19223acf7f98e51ead9b058c94` |
| `labels.jsonl` | 64 lines | text/LF | 12,656 | `9ea9541d4309169407f09834a261e8e97ae98df9ce5ad7a397be842cbf920d7d` |
| `meta.json` | — | JSON | 6,215 | `843225095c98c9ffd42222e57d97b3563258323d204f3c94da4227bb76154e90` |
| `manifest.json` | — | JSON | 5,874 | `338d83d08bcc992cacaa0b4252d51218205928e6b0eeaba081abea9c0fea7fb7` |
| `bc_smoke.json` | — | JSON | 2,678 | `ab164ac21c2de388677d972c24d019cde036614424ac82bd10cc8fdd0612c559` |
| `labels_provenance.json` | — | JSON | 2,120 | `ca868c956ad1170acf85909834dcd817f87fe642ee1864d39888cd1a8185dfec` |

Composite digest over the seven artefacts (sha256 of the sorted `sha256 name`
lines): `1c85869a96832fd9413179d90d2c141dec12ccb6b64cdb2f2341e332a65c2dd2`.

`frames_rgba.npy` is the **unprocessed** `(480,640,4)` readback of each collected
decision's last game frame. It is **evidence, not a training input**: it exists so
a future ruling on premultiplied alpha can be re-derived **without another engine
run**. Training uses the `encode_o2` output only.

Determinism check: the 400-frame rehearsal produced `global_rgb.npy` with the
**same sha256** (`0c2ccae6…`) as the 13,195-frame run — the first 64 decisions are
identical, only `elapsed_capture_s.npy` (a wall-clock interval) differs.

---

## 5. Alpha: the finding this slice produced

The engine's `(480,640,4)` readback carries a **blended alpha plane for the whole
replay**, not only during the opening fade. Measured on 28 sampled frames over
the first 700 frames (`.tmp_v430_o2/probe_alpha2.py`):

| quantity | range over the 700 frames |
| --- | --- |
| `alpha_min` | 105 … 132 |
| `alpha_max` | 190 … 255 |
| `alpha_mean` | 187.8 … 241.34 |
| fraction of pixels with `alpha == 255` | **0.000000** (all 28 frames) |
| `alpha_mode(frame)` | **28/28 `FrameAlphaError`** |

So `encode_o2`'s default `alpha_mode="strict"` refuses **every** real engine frame
(`alpha plane holds intermediate values (202..219)`), and "wait for the fade to
end" is not a workaround. The audited `alpha_mode="any"` opt-in is what makes a
real capture possible; it is recorded per observation and in `meta.json`.

On the 64 collected decisions (`frames_rgba.npy`, `meta.json.blank_audit.alpha`):

| quantity | value |
| --- | --- |
| `alpha_min` (min over all 64 frames) | 111 |
| `alpha_max` (max over all 64 frames) | 255 |
| `alpha_mean` min / median / max | **187.6273 / 237.8625 / 243.6941** |
| frames whose alpha is all 255 | **0** |
| frames containing any alpha == 0 | **0** |

**Premultiplication:** the test is that a premultiplied buffer cannot hold a
colour channel above its own alpha. Only 0.03 %–0.12 % of pixels violate
`max(RGB) > alpha + 2` (103 / 362 / 338 out of 307,200 on three sampled frames),
which **supports premultiplied/alpha-weighted RGB** (a systematic darkening of
≈ `alpha/255` ≈ 0.91). **No un-premultiplication was applied** — the O2 arrays
hold exactly what the encoder produced from the raw readback, which is also what
the legacy v4.2.0 corpus did (same buffer, same BT.601 formula). Whether to
un-premultiply is a **specification decision**, deliberately left open and made
re-derivable by keeping `frames_rgba.npy`.

## 6. Blank-frame audit (requested)

`encode_o2`'s "uniform placeholder" refusal inspects only the **alpha** plane, so
an opaque all-black frame passes the encoder. The capture therefore measures
blankness itself and records `is_blank` / `global_rgb_mean` / `alpha_mean` **per
sample** in `labels.jsonl`:

```
blank      = (global_rgb.reshape(64,-1).sum(axis=1) == 0).sum()   -> 0   (0.0000 %)
near_blank = (global_rgb.reshape(64,-1).mean(axis=1) < 1.0).sum() -> 0   (0.0000 %)
global_rgb.mean() min / median / max = 12.9097 / 16.9853 / 201.7676
```

`is_blank = 0/64`, `near_blank = 0/64` — **no blank frames are in this dataset**
and the 10 % warning threshold is not approached. (The max 201.77 is the opening
fade-in frame at decision 1, which is a real bright frame, not a placeholder; the
capture also refuses uniform-RGB frames before they can enter a stack — none
occurred, `invalid_frames = 0`.)

---

## 7. Task B — manifest and O2 guard (measured)

`manifest.json` is built by `.tmp_v430_o2/verify_manifest_bc.py` with
`schema = v430-manifest/1`, `obs_contract = "obs-v430-o2"`, `status = complete`,
one shard entry per artefact (`labels.jsonl` → `text/lf-normalized`, every `.npy`
→ `binary/raw`), and `obs_shapes` taken from `observation.O2_SHAPES`.

```
validate_manifest(manifest, base_dir=REPO_ROOT)
  valid = True      complete = True      exit_code = 0      problems = []   degradations = []
  checks_performed = [identity, shards, counts, ancestors, digest_scope, status]
  statistics = {ancestor_groups: 1, ancestor_ids: 8, bytes_listed: 109004272,
                decisions: 512, episodes: 8, labels_valid: 512,
                shards_by_split: {dev: 8}, shards_listed: 8, shards_readable: 8}

assert_o2_compatible(manifest)
  -> returned the same Manifest (dataset_id = "w03h-o2-capture-seed12-64")
  -> DID NOT RAISE
```

`not_checked` is reported, not hidden, and is quoted verbatim in
`bc_smoke.json`: the validator hashes the bytes but does not interpret them; no
expected `obs_digest`/`action_digest`/`label_digest`/`split_registry_sha` was
supplied to compare against; the unregistered-shard scan needs explicit roots and
therefore did not run. **`valid=True` means "the manifest is internally consistent
and its bytes are the ones it names" — not "this is good training data".**

---

## 8. Task C — BC smoke run on the real O2 samples

```
train_bc(model, batches, epochs=2, lr=1e-4, seed=0, memory_mode="none", max_steps=8)
```

| field | value |
| --- | --- |
| samples | **64** real O2 decisions (8 batches × 8) |
| `n_steps` | **8** |
| first loss | **4.271554** |
| last loss | **3.305507** |
| `model_supplied` | **True** (the logits are `ActorPolicy`'s: batch dim 64/8) |
| model | `ActorPolicy` |
| parameters | **2,726,671** (all trainable) |
| lr / seed / memory_mode | 1e-4 / 0 / `none` |
| model init | `torch.manual_seed(0)` before `build_actor()` |
| artifact | `.tmp_v430_o2/o2-64/bc_smoke.json` (2,678 bytes) |

Reproducibility: `train_bc`'s `seed` drives only its own generator; the actor's
initial parameters come from the global torch RNG. The verifier therefore calls
`torch.manual_seed(0)` before `build_actor()` and records that, and **two
consecutive runs produced bit-identical losses** (4.271553993225098 →
3.305506706237793). Before this was added the same command produced
4.4415/4.4426 → 3.4201 depending on the un-seeded init; that first pair is
superseded and not cited.

**This is a 64-sample, 2-epoch, 8-optimizer-step smoke run. It shows that real O2
pixels reach the four-head BC loss and that the loss moves; it is NOT evidence of
learning, capability, or any gate.** Note that only the `move`/`focus` terms carry
a learning signal on this corpus — see §10.

---

## 9. Label provenance (audited — 0 % of labels lack a tape row)

The audited tape's **first slot records `decision_frame = 16`**, together with
`history_frames = [0,2,4,6,8,10,12,14]` and `history = [7,7,7,7,7,7,7,7]`. The
eight decisions before frame 16 exist **only** in that slot's `history`, not as
slot rows. The capture collects decisions **1 … 64**, so its first seven labels
(frames 2,4,…,14) are those recorded leading actions.

`.tmp_v430_o2/verify_labels_provenance.py` classifies every label against the
tape's own bytes and writes `labels_provenance.json`:

| provenance | count | where the label comes from |
| --- | --- | --- |
| `recorded_slot` | **57** | a `slots.jsonl` row (`executed_cand` / `mask` / `decision_frame`) |
| `leading_history` | **7** | slot 0's `history` / `history_frames` (recorded, not synthesised) |
| `tail_repeat` | **0** | — none collected (the synthesised tail is outside the window) |

Cross-check on the 57 slot-backed labels: `cand`, `mask` and `decision_frame`
match the tape **57/57, 0 mismatches**. So **0 of the 64 labels is without tape
provenance**; the 7 leading ones are recorded by the tape in its first slot rather
than as their own rows.

**New fail-closed guard** (`.tmp_v430_o2/capture_o2.py`, effective on the next
capture; verified offline by `_selftest_guard.py`, no engine needed): every label
is now tagged with `tape_provenance`, and a decision classified `tail_repeat`
(label has no recorded tape row) or `beyond_schedule` is **refused as a sample**
unless `--allow-tail-labels` is passed. Offline test result:
`schedule_length=6660`, `leading=8`, `recorded_frames=6588`, `tail=64`,
`decision 0..7 → leading_history`, `decision 8..6595 → recorded_slot`,
`first tail → tail_repeat`, `beyond schedule → beyond_schedule`,
`no_tail_in_collected_window = True`. The 64 already-captured labels predate the
tag; their provenance is established by the audit above, so no re-capture is
required for this evidence.

---

## 10. Which heads are actually supervised (important for any reader of §8)

Measured distribution over the 64 labels:

| head | distinct values | verdict |
| --- | --- | --- |
| `move` | 5 (`{0,2,3,6,7}`) | **supervised** |
| `focus` | 2 (`{0,1}`) | **supervised** |
| `shoot` | **1** (`{1}`) | constant |
| `bomb` | **1** (`{0}`) | constant |

**Mechanism (an instrument property, not a capture defect):** the v4.2.0
candidate space is `(focus, move)` — `gym_actions.cand_to_mask` *always* ORs
`BTN_SHOOT` into the mask, and the replay ruler fixes `BOMB` to 0. A
tape-derived O2 corpus therefore cannot contain shoot/bomb variation.

**Consequence:** in a four-head BC loss on this corpus the `shoot` and `bomb`
terms are **constant fitting with zero learning signal**, so the §8 losses are
only meaningful for `move`/`focus`. Any future training on tape-derived O2 data
should **exclude or mark those two heads unsupervised** (e.g. a per-head loss
mask) rather than read the four-head sum as four supervised tasks.

**Inference:** shoot/bomb diversity requires a **tape whose teacher released
shoot and used bombs**. No such tape is replayed in this slice; this dataset must
not be used to claim anything about shoot/bomb behaviour.

---

## 11. Faults met and how they were handled

1. **`KeyError: 'executed_cand'`** on the `T02c` rebuild shard — the rebuild layer
   stores `cand`. Resolved by using the `T04` dataset shard, whose bytes equal the
   ruler's frozen `CLEAR_SHARD_SHA256` (a *stronger* provenance than the task's
   path). No loader file was modified.
2. **`KeyError: 'lives'`** — the wire `STATE` reply names it `lives_remaining`.
   Resolved by using the ruler's own `_readings` mapping instead of guessing keys.
3. **`FrameAlphaError` on real frames** — the finding in §5. Not worked around;
   the encoder gained an audited `alpha_mode="any"` opt-in (W03I) and the capture
   records which gate it used.
4. Rehearsals were run at 40 frames / 4 samples and 400 frames / 64 samples before
   the 13,195-frame run, so each failure cost seconds rather than a full replay.
   All three exit codes and the two failed rehearsals are in
   `EXIT_CODES_W03H.txt`.

Nothing was fabricated: every array in `o2-64/` came from a live engine readback,
no array was resampled from a legacy corpus, and no legacy `encode_planes_v3`
output exists anywhere in this dataset.

---

## 12. Draft `ASSET_SYNC` entry (NOT applied — `ASSET_SYNC.*` untouched)

```markdown
- `.tmp_v430_o2/` — W03H real O2 capture (2026-11-15). **Local asset, git-ignored,
  never uploaded.**
  - `.tmp_v430_o2/o2-64/` — **109,025,255 bytes**, the real `obs-v430-o2` dataset:
    - `global_rgb.npy` (64,3,240,320) uint8 — 14,745,728 B — sha256 0c2ccae67b843c5e6781c43e23050567c829f7405385ae7083f47398be1ce97b
    - `play_gray.npy` (64,4,224,192) uint8 — 11,010,176 B — sha256 27292db0272b9284662a3c01e8702f2ed0dfeaae1f32b94aa24ffd5db5d59bf4
    - `hud_gray.npy` (64,1,448,160) uint8 — 4,587,648 B — sha256 c6424dcf941b4b20c2f8c6fe7a876b1e7f69a48f86b2017b1857242c5709d294
    - `executed_key_history.npy` (64,8,7) uint8 — 3,712 B — sha256 0e9d67c1740b1f8ea21ffe1a98068d000587453e0f48c70c698b9a22ad38ee99
    - `history_valid.npy` (64,8) bool — 640 B — sha256 56b18879fae33df061a5d0fd4ce51eb1a54a834009be35945b7c313b0ea504ba
    - `elapsed_capture_s.npy` (64,) float32 — 384 B — sha256 7064fb4ebc0353b56f1530dcf3ef9059a77e6c03a3e34cf1e27eb1acee64d75a
    - `frames_rgba.npy` (64,480,640,4) uint8 — 78,643,328 B — sha256 2ece7c5b4ff6e7fbd6246ea8a755794a276f2d19223acf7f98e51ead9b058c94
      (raw unprocessed readback; evidence for a future premultiplication ruling,
      **not** a training input)
    - `labels.jsonl` 64 lines — 12,656 B — sha256 9ea9541d4309169407f09834a261e8e97ae98df9ce5ad7a397be842cbf920d7d
    - composite digest (sorted `sha256 name` lines): 1c85869a96832fd9413179d90d2c141dec12ccb6b64cdb2f2341e332a65c2dd2
  - `.tmp_v430_o2/engine-data/` — 496,968,817 bytes, the **private** data dir the
    engine wrote `score.dat`/`th08.cfg` into (`cp -a` of
    `.tmp_v420_data/T06/engine-data-gamesrc`, kept for reproducibility).
  - provenance: engine `691bfd77…`, tape `f759de7d…` (= frozen `CLEAR_SHARD_SHA256`),
    `obs_contract="obs-v430-o2"`, `alpha_mode="any"`, 13,195 frames / 6,598
    decisions replayed, 64 decisions collected.
  - whole `.tmp_v430_o2/` tree: **619,754,789 bytes**.
  - verdict: **64-sample smoke dataset; not a capability dataset, not a gate result.**
```

---

## 13. What this slice does NOT establish

* No capability, no gate, no evaluation. 64 samples cannot be a training set.
* `valid=True` is a manifest-consistency result, not a data-quality result.
* The premultiplication question is **open** and deliberately not acted on.
* **`shoot` and `bomb` are constant in this corpus** (§10), so nothing about those
  two heads can be learned or claimed from it.
* The two legacy-corpus claims in this project (28.8 GB, S1 66,143) are untouched
  and were not re-validated here.
