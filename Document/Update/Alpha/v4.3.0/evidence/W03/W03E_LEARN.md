# W03E — BC training orchestration (synthetic batches only)

**合成测试，不是游戏/能力证据。** （Synthetic unit tests — not game or capability
evidence. No engine was started, no real corpus was read, no model was trained to
play, and nothing here measures TH08 capability.)

Owner slice: `learn.py` in `touhou_rl/v430/`, declared by `PLAN_SPEC.json` `bc`
(`optimizer=AdamW`, `learning_rate=1e-4`, `weight_decay=1e-4`,
`missing_action_labels=mask`) and `micro` (arms `D-state` / `D-pixel` /
`D-pixel-memory`), interpreted through `DEVELOPMENT_GUIDE.md` §7.4 (independent
supervised term, each target normalised by its own valid labels) and §8
(tactics/BC training semantics).

Deliberately small slice: **only the orchestration**. The four real arms, PPO, the
value warm-up and the §7.1 burn-in/prefix recompute are explicitly out of scope
and marked `TODO(W03E)` in the module docstring.

## 1. What was added

| File | Change |
| --- | --- |
| `touhou_rl/v430/learn.py` | **new** — loss, loop, checkpoints, fail-closed entry point |
| `tests/test_v430_learn.py` | **new** — 17 synthetic tests |
| `Document/Update/Alpha/v4.3.0/evidence/W03/W03E_LEARN.md` | **new** — this file |
| `Document/Update/Alpha/v4.3.0/evidence/W03/EXIT_CODES_W03E.txt` | **new** — commands + exit codes |

Nothing else was created, deleted or edited. `contracts`, `observation`,
`sequence`, `session`, `evaluate`, `actions`, `models/**`, `data/**`, `tools/rl/**`,
`configs/**`, `registries/**`, `PLAN_SPEC.json`, the other `tests/test_v430_*.py`
files and the earlier W03 evidence files were **read only**.

## 2. API and semantics

### `bc_loss(head_logits, targets) -> BCLoss`

Four cross entropies, **added**:

```
total = mean_CE(move) + mean_CE(focus) + mean_CE(shoot) + mean_CE(bomb)
```

Each head is averaged over its own samples; the total is **not** divided by four (a
silent `/4` would shrink every gradient and make the number impossible to
hand-check). `total` stays differentiable; `per_head` and `n_samples` are returned
alongside. Accepts an `actor.HeadLogits` or a four-key mapping, labels either as a
four-key mapping of 1-D integer columns or as an `(N, 4)` integer array in
`contracts.HEAD_ORDER` order.

Hand check (one sample, natural log):

| head | logits | label | expected CE | measured |
| --- | --- | --- | --- | --- |
| move | `(0,0,2,0,0,0,0,0,0)` | 0 | `ln(8+e²)` = 2.7336566139 | 2.7336566448 |
| focus | `(0,1)` | 0 | `ln(1+e)` = 1.3132616875 | 1.3132616282 |
| shoot | `(1,0)` | 1 | `ln(1+e)` = 1.3132616875 | 1.3132616282 |
| bomb | `(0,0)` | 1 | `ln 2` = 0.6931471806 | 0.6931471825 |
| **total** | | | **6.0533271695** | **6.0533270836** |

Abs differences ≤ 5.94e-08 (tolerance 1e-6); the quarter value 1.5133317924 is
asserted *not* to match, so an averaged loss cannot pass.

### `train_bc(model, batches, *, epochs, lr, seed, optimizer=None, memory_mode="none", max_steps=None, weight_decay=1e-4, shuffle_batches=False) -> TrainingReport`

* one record per optimizer step: `{"step", "epoch", "loss", "per_head_loss", "n_samples"}` (exactly these keys);
* deterministic: same `seed` + same batches + same starting weights ⇒ identical per-step losses and `torch.equal` weights; the only randomness is an explicit `torch.Generator` used when `shuffle_batches=True` (no global RNG, no implicit shuffle);
* `max_steps` stops across epochs (a cap smaller than one epoch's step count works too);
* fail-closed: empty/non-sequence `batches`, `epochs<=0`, `lr<=0`, `max_steps<=0`, bad `memory_mode`, a malformed batch, a wrong key set, a label count ≠ batch size, a non-integer label, an out-of-range label, a non-`nn.Module` model, and an optimizer that owns none of the model's parameters all raise `LearnError` — before any weight moves;
* `memory_mode="none"` (D-pixel) **never passes `hidden`** to the actor; `memory_mode="gru"` (D-pixel-memory) steps one observation sequence and carries the `(1,B,256)` state, back-propagating through the carried state. The mode used is in every report.

Measured determinism (seed 99, epochs 2, lr 1e-3, B=2, identical initial weights):

```
run 1: 4.034379959106445  4.381588459014893  3.952694892883301
run 2: 4.034379959106445  4.381588459014893  3.952694892883301
weights: torch.equal over all 60 parameter tensors -> True
```

### `save_checkpoint(model, path, *, extra=None)` / `load_checkpoint(path, *, map_location="cpu", expected_o2_digest=None)`

The payload records `state_dict` (CPU clones), a rebuildable `model_config`
(`actor.MODEL_SPEC`, head order/sizes, `joint_count==72`, `O2_SHAPES`,
`normalize_divisor`), `torch.__version__`, `contracts.o2_digest()` and
`contracts.SPEC_REVISION`.

* save opens the file with mode `"xb"`: an existing path is refused **and** a race cannot overwrite it; a missing parent directory is refused instead of created;
* load verifies, in order: schema, presence of every identity field, `o2_digest` vs the **current** encoder, optional `expected_o2_digest`, `spec_revision`, `model_config.model_spec` vs the running `MODEL_SPEC`, then `load_state_dict(strict=True)`;
* the `o2_digest` comparison is **mandatory** even when the argument is omitted, and `expected_o2_digest` only adds a constraint — naming a foreign digest explicitly still refuses (proven by a test);
* round-trip: `torch.equal` for the four head logits **and** the GRU state on the same input.

### `train_from_manifest(manifest, *, split_config, ..., run=True) -> dict`

Order is the contract: `validate_manifest` (manifest valid/complete) → `validate_splits` (frozen registry passes) → training. `valid=False` or `passed=False` raises `LearnError` naming the validator's own problem codes, before an optimizer exists. The returned dict is built with `contracts.new_result` and passed through `contracts.assert_result_schema`; it carries both validation reports, the training report (with the actual `memory_mode`), `o2_digest`, `spec_revision`, `batch_source="synthetic"`, `synthetic_data_only=True`, `model_supplied`, and `research_result.status` = `"inconclusive"` (a completed training step is not a research pass) or `"not_run"` for `run=False`. `capability_gate` stays `not_run`.

## 3. Requirement → test map (17 tests)

| # | Requirement | Test(s) |
| --- | --- | --- |
| 1 | `bc_loss` = sum of four CEs, hand-checked | `test_bc_loss_is_the_sum_of_four_cross_entropies`, `test_bc_loss_normalises_each_head_over_its_own_samples` |
| 2 | determinism (same seed identical, different seed differs) | `test_same_seed_reproduces_every_step_and_the_weights`, `test_a_different_seed_changes_the_run` |
| 3 | checkpoint round-trip | `test_checkpoint_roundtrip_keeps_the_logits_bitwise_identical` |
| 4 | refuse to overwrite | `test_save_checkpoint_refuses_to_overwrite` |
| 5 | `o2_digest` mismatch refused (and not bypassable) | `test_load_checkpoint_rejects_a_foreign_o2_digest` |
| 6 | invalid manifest refused before training | `test_invalid_manifest_is_refused_before_training` |
| 7 | train/dev crossing refused before training | `test_train_dev_crossing_is_refused_before_training` |
| 8 | out-of-range labels refused | `test_out_of_range_labels_are_refused`, `test_label_count_mismatch_is_refused` |
| 9 | empty batches / `epochs<=0` / `lr<=0` refused | `test_empty_batches_epochs_and_lr_are_refused` |
| 10 | `memory_mode` recorded, both modes | `test_memory_mode_is_recorded_for_none_and_gru`, `test_train_from_manifest_records_the_memory_mode_and_the_result_schema` |
| 11 | `max_steps` effective | `test_max_steps_stops_the_loop` |
| 12 | purity (no v400/v420/gym) | `test_learn_does_not_import_engine_v400_v420_or_gym` |
| 13 | synthetic-scope header | `test_file_header_declares_the_synthetic_scope` |

Extra coverage beyond the thirteen: `test_bc_loss_normalises_each_head_over_its_own_samples` (two identical samples give the same loss as one; `ln 9 + 3 ln 2`), `test_label_count_mismatch_is_refused` (missing head, wrong batch keys, short column), `test_train_from_manifest_...` (the `assert_result_schema` result and the `run=False` validate-only form).

## 4. Regression evidence

```
python -m compileall -q touhou_rl/v430/learn.py                       exit 0
python -m pytest ... -q tests/test_v430_learn.py                      17 passed in 8.24s   exit 0
python -m pytest ... -q tests/test_v430_models.py tests/test_v430_data.py tests/test_v430_sequence.py
                                                                      120 passed in 3.74s  exit 0
```

## 5. TODOs and remaining gaps (not claimed as done)

* `TODO(W03E)`: PPO, the critic/value warm-up (§7.4) and the §7.1 burn-in / exact-prefix hidden recompute are **not** implemented here.
* The three `micro` arms (`D-state` / `D-pixel` / `D-pixel-memory`) are represented only by the `memory_mode` switch on synthetic batches; no real corpus, no frozen config, no run identity, no closed-loop evaluation exists yet.
* No gradient-norm / direction-conflict logging yet (§7.4 asks for it), and no per-head valid-label masking beyond the whole-label-column boundary check.
* `train_from_manifest` builds `build_actor()` from torch's default generator when `model is None`; a reproducible run must pass an explicitly seeded model (recorded as `model_supplied`). No seeding of the global RNG happens inside `learn.py`.
* Two tests read the repository's frozen `configs/v430/splits.json` (unmodified) because a fully valid synthetic split registry would duplicate the lineage suite's fixture; the crossing case is a deep copy with one dev ancestor appended to train.
* Bitwise checkpoint round-trip is demonstrated on this host's single `torch` build; the recorded `torch_version` is the value to compare before claiming it across environments.
