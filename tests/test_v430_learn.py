"""v4.3.0 W03E — behaviour tests for the BC training orchestration.

**合成测试，不是游戏/能力证据。**  （Synthetic test fixture — not game or capability
evidence.  No engine was started, no real corpus was read, no model was trained to
play.  Every tensor below is generated in-process, and the only real files touched
are this repository's own frozen split registry, read **unmodified**, and files
created inside ``tmp_path``.)

Each test states the decision the module must make (accept / refuse) and, where the
guide requires it, that the refusal names the offending field, label or code — not
merely that "an exception happened".  A green run proves the orchestration's
decisions; it does **not** say anything about TH08 capability, dataset quality or a
trained policy.

Coverage map (W03E, thirteen requirements):

1. ``bc_loss`` is the **sum** of four per-head cross entropies, checked against a
   hand calculation (:func:`test_bc_loss_is_the_sum_of_four_cross_entropies`);
2. determinism: same seed + same batches -> bitwise identical losses and weights,
   a different seed -> a different run
   (:func:`test_same_seed_reproduces_every_step_and_the_weights`,
   :func:`test_a_different_seed_changes_the_run`);
3. checkpoint round-trip is bitwise identical
   (:func:`test_checkpoint_roundtrip_keeps_the_logits_bitwise_identical`);
4. an existing checkpoint is never overwritten
   (:func:`test_save_checkpoint_refuses_to_overwrite`);
5. an ``o2_digest`` mismatch is refused
   (:func:`test_load_checkpoint_rejects_a_foreign_o2_digest`);
6. an invalid manifest is refused **before** training
   (:func:`test_invalid_manifest_is_refused_before_training`);
7. a train/dev crossing is refused **before** training
   (:func:`test_train_dev_crossing_is_refused_before_training`);
8. out-of-range labels (and a label/batch-size mismatch) are refused
   (:func:`test_out_of_range_labels_are_refused`,
   :func:`test_label_count_mismatch_is_refused`);
9. empty batches / ``epochs<=0`` / ``lr<=0`` are refused
   (:func:`test_empty_batches_epochs_and_lr_are_refused`);
10. the ``memory_mode`` actually used is recorded, for both modes
    (:func:`test_memory_mode_is_recorded_for_none_and_gru`,
    :func:`test_train_from_manifest_records_the_memory_mode_and_the_result_schema`);
11. ``max_steps`` stops the loop
    (:func:`test_max_steps_stops_the_loop`);
12. purity: ``learn.py`` imports neither ``touhou_rl.v400`` / ``v420`` nor ``gym``
    (:func:`test_learn_does_not_import_engine_v400_v420_or_gym`);
13. this header (:func:`test_file_header_declares_the_synthetic_scope`).
"""

from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping

import pytest
import torch

from touhou_rl.v430 import contracts, learn, observation
from touhou_rl.v430.data import lineage
from touhou_rl.v430.data.lineage import (
    DIGEST_SCOPE_BINARY,
    DIGEST_SCOPE_TEXT,
    ancestor_id,
    binary_raw_digest,
)
from touhou_rl.v430.data.manifest import (
    LEGACY_OBS_CONTRACTS,
    MANIFEST_SCHEMA,
    ManifestError,
    load_manifest,
    validate_manifest,
)
from touhou_rl.v430.models import actor


REPO = Path(__file__).resolve().parents[1]
#: The repository's frozen split registry, **read only** (never edited here).
REAL_SPLITS = REPO / "configs" / "v430" / "splits.json"
LEARN_SOURCE = REPO / "touhou_rl" / "v430" / "learn.py"
TEST_SOURCE = Path(__file__).resolve()

FAMILY_TRAIN = "th08-stage1-normal-team0"
FAMILY_DEV = "th08-stage2-normal-team0"
TRAIN_ANCESTOR = ancestor_id("human_replay", "th08-r2", "ep-0001", FAMILY_TRAIN)
DEV_ANCESTOR = ancestor_id("human_replay", "th08-r2", "ep-0007", FAMILY_DEV)


# ---------------------------------------------------------------------------
# synthetic observations and labels (no corpus, no engine)
# ---------------------------------------------------------------------------
def _generator(salt: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(10_000 + salt)
    return generator


def _observation(batch: int = 2, *, salt: int = 0) -> dict[str, torch.Tensor]:
    """One batched O2 observation of real frozen shapes, filled deterministically."""

    generator = _generator(salt)
    shapes = observation.O2_SHAPES

    def image(name: str) -> torch.Tensor:
        shape = (batch,) + tuple(int(dim) for dim in shapes[name])
        return torch.randint(0, 256, shape, generator=generator).to(torch.uint8)

    return {
        "global_rgb": image("global_rgb"),
        "play_gray": image("play_gray"),
        "hud_gray": image("hud_gray"),
        "executed_key_history": torch.randint(
            0, 2, (batch, 8, 7), generator=generator).to(torch.uint8),
        "history_valid": torch.zeros((batch, 8), dtype=torch.bool),
        "elapsed_capture_s": torch.full((batch,), 0.016, dtype=torch.float32),
    }


def _labels(batch: int = 2, *, salt: int = 0) -> dict[str, torch.Tensor]:
    """In-range integer labels for the four heads, in contract order."""

    generator = _generator(1_000 + salt)
    return {
        head: torch.randint(0, int(contracts.HEAD_SIZES[head]), (batch,),
                            generator=generator, dtype=torch.int64)
        for head in contracts.HEAD_ORDER
    }


def _batch(batch: int = 2, *, salt: int = 0) -> dict[str, Any]:
    return {"observations": _observation(batch, salt=salt),
            "targets": _labels(batch, salt=salt)}


def _sequence_batch(steps: int = 2, batch: int = 2, *, salt: int = 0) -> dict[str, Any]:
    return {
        "observations": [_observation(batch, salt=salt + step)
                         for step in range(steps)],
        "targets": [_labels(batch, salt=salt + step) for step in range(steps)],
    }


def _fresh_actor(seed: int = 1234) -> actor.ActorPolicy:
    """Two calls with the same seed give bitwise identical initial weights."""

    torch.manual_seed(seed)
    return actor.build_actor()


# ---------------------------------------------------------------------------
# a *synthetic* manifest that really is valid (its own files live in tmp_path)
# ---------------------------------------------------------------------------
def _hex(label: str) -> str:
    return binary_raw_digest(label.encode("utf-8"))


def _identity(label: str, scope: str = DIGEST_SCOPE_TEXT) -> dict[str, str]:
    return {"sha256": _hex(label), "digest_scope": scope}


def _shard(path: str, payload: bytes, *, split: str, ancestors: list[str],
           episodes: list[str], decisions: int, labels_valid: int) -> dict[str, Any]:
    return {
        "path": path,
        "size_bytes": len(payload),
        "sha256": binary_raw_digest(payload),
        "digest_scope": DIGEST_SCOPE_BINARY,
        "episodes": list(episodes),
        "ancestors": list(ancestors),
        "frame_range": [0, 120],
        "decisions": decisions,
        "labels_valid": labels_valid,
        "duplicates": {"episode_repeats": 0, "frame_overlap": 0},
        "split": split,
        "status": "completed",
        "immutable": True,
    }


def _counts(shards: list[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "shards": len(shards),
        "decisions": sum(int(item["decisions"]) for item in shards),
        "labels_valid": sum(int(item["labels_valid"]) for item in shards),
        "episodes": sum(len(item["episodes"]) for item in shards),
    }


def _document(shards: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "dataset_id": "th08-w03e-synthetic",
        "parent_ids": [],
        "code_snapshot": {"git_commit": "0" * 40, "dirty": False},
        "engine_sha": _identity("w03e-engine", DIGEST_SCOPE_BINARY),
        "game_data_manifest_sha": _identity("w03e-game-data"),
        "obs_digest": {"sha256": contracts.o2_digest(), "digest_scope": DIGEST_SCOPE_TEXT},
        "action_digest": _identity("w03e-actions"),
        "label_digest": _identity("w03e-labels"),
        # Deliberately *not* the real registry's sha: this manifest never claims to
        # belong to the frozen registry, and ``validate_manifest`` is called here
        # without ``split_config``, so the binding stays in ``not_checked``.
        "split_registry_sha": _identity("w03e-synthetic-registry"),
        "obs_contract": "obs-v430-o2",
        "shards": shards,
        "counts": _counts(shards),
        "status": "complete",
    }


@pytest.fixture()
def dataset(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    """A valid two-shard manifest document plus its data root."""

    root = tmp_path / "data"
    (root / "train").mkdir(parents=True)
    (root / "dev").mkdir(parents=True)
    train_payload = b"TH08-W03E-TRAIN\x00\x01\x02" * 7
    dev_payload = b"TH08-W03E-DEV\x03\x04" * 5
    (root / "train" / "s1.bin").write_bytes(train_payload)
    (root / "dev" / "s1.bin").write_bytes(dev_payload)
    shards = [
        _shard("train/s1.bin", train_payload, split="train",
               ancestors=[TRAIN_ANCESTOR], episodes=["ep-0001"],
               decisions=10, labels_valid=9),
        _shard("dev/s1.bin", dev_payload, split="dev",
               ancestors=[DEV_ANCESTOR], episodes=["ep-0007"],
               decisions=6, labels_valid=6),
    ]
    return _document(shards), root


def _write_manifest(doc: Mapping[str, Any], directory: Path) -> Path:
    path = directory / "manifest.json"
    path.write_bytes(json.dumps(doc, indent=2, sort_keys=True,
                                ensure_ascii=True).encode("utf-8"))
    return path


# ---------------------------------------------------------------------------
# 1. bc_loss = sum of four cross entropies, checked by hand
# ---------------------------------------------------------------------------
def test_bc_loss_is_the_sum_of_four_cross_entropies() -> None:
    """``total == CE(move) + CE(focus) + CE(shoot) + CE(bomb)``, not their mean.

    Hand calculation for one sample (natural log):

    * ``move`` logits ``(0,0,2,0,0,0,0,0,0)``, label 0
      -> ``CE = ln(8 + e^2) = 2.7336566139...``
    * ``focus`` logits ``(0,1)``, label 0 -> ``CE = ln(1 + e) = 1.3132616875...``
      (the correct class has logit 0, the other 1)
    * ``shoot`` logits ``(1,0)``, label 1 -> ``CE = ln(1 + e) = 1.3132616875...``
    * ``bomb`` logits ``(0,0)``, label 1 -> ``CE = ln 2 = 0.6931471805...``

    Total ``= 6.0533271695...``; one quarter of it would be ``1.5133317924...``, so a
    silent ``/4`` cannot pass this assertion.
    """

    logits = {
        "move": torch.tensor([[0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        "focus": torch.tensor([[0.0, 1.0]]),
        "shoot": torch.tensor([[1.0, 0.0]]),
        "bomb": torch.tensor([[0.0, 0.0]]),
    }
    targets = {
        "move": torch.tensor([0]),
        "focus": torch.tensor([0]),
        "shoot": torch.tensor([1]),
        "bomb": torch.tensor([1]),
    }

    expected_move = math.log(8.0 + math.e ** 2)      # 2.7336566139...
    expected_binary = math.log(1.0 + math.e)         # 1.3132616875...
    expected_bomb = math.log(2.0)                    # 0.6931471805...
    expected_total = (expected_move + expected_binary + expected_binary
                      + expected_bomb)               # 6.0533271695...

    loss = learn.bc_loss(logits, targets)

    assert loss.n_samples == 1
    assert abs(loss.per_head["move"].item() - expected_move) < 1e-6
    assert abs(loss.per_head["focus"].item() - expected_binary) < 1e-6
    assert abs(loss.per_head["shoot"].item() - expected_binary) < 1e-6
    assert abs(loss.per_head["bomb"].item() - expected_bomb) < 1e-6
    assert abs(loss.item() - expected_total) < 1e-6
    assert abs(loss.item() - expected_total / 4.0) > 1e-3, "the loss was averaged"
    assert abs(loss.item() - sum(value.item() for value in loss.per_head.values())) \
        < 1e-12
    # A HeadLogits bundle is accepted exactly like the raw mapping.
    bundle = actor.HeadLogits(**{head: value for head, value in logits.items()})
    assert abs(learn.bc_loss(bundle, targets).item() - expected_total) < 1e-6
    # The loss is differentiable: a BC step can actually move the weights.
    assert loss.total.requires_grad is False  # built from tensors without grad
    grad_logits = {head: value.clone().requires_grad_(True)
                   for head, value in logits.items()}
    differentiable = learn.bc_loss(grad_logits, targets)
    differentiable.total.backward()
    assert grad_logits["move"].grad is not None
    assert float(grad_logits["move"].grad.abs().sum()) > 0.0


def test_bc_loss_normalises_each_head_over_its_own_samples() -> None:
    """Two identical samples give the same loss as one (mean, then added)."""

    logits = {"move": torch.zeros(2, 9), "focus": torch.zeros(2, 2),
              "shoot": torch.zeros(2, 2), "bomb": torch.zeros(2, 2)}
    targets = {"move": torch.tensor([1, 1]), "focus": torch.tensor([0, 0]),
               "shoot": torch.tensor([0, 0]), "bomb": torch.tensor([0, 0])}
    batched = learn.bc_loss(logits, targets)
    single = learn.bc_loss(
        {head: value[:1] for head, value in logits.items()},
        {head: value[:1] for head, value in targets.items()})
    assert abs(batched.item() - single.item()) < 1e-6
    assert abs(batched.item() - (math.log(9.0) + 3 * math.log(2.0))) < 1e-6


# ---------------------------------------------------------------------------
# 2. determinism
# ---------------------------------------------------------------------------
def test_same_seed_reproduces_every_step_and_the_weights() -> None:
    """Same seed + same batches + same initial weights -> bitwise equal runs."""

    batches = [_batch(salt=0), _batch(salt=1)]

    first_model = _fresh_actor(seed=7)
    first = learn.train_bc(first_model, batches, epochs=2, lr=1e-3, seed=99,
                           memory_mode="none")
    second_model = _fresh_actor(seed=7)
    second = learn.train_bc(second_model, batches, epochs=2, lr=1e-3, seed=99,
                            memory_mode="none")

    assert first.n_steps == 4 and second.n_steps == 4
    first_losses = list(first.losses)
    second_losses = list(second.losses)
    assert first_losses == second_losses, "per-step losses are not bitwise equal"
    # The requested proof: the first three losses, side by side.
    assert first_losses[:3] == second_losses[:3]
    for step_first, step_second in zip(first.steps, second.steps):
        assert step_first["per_head_loss"] == step_second["per_head_loss"]
        assert step_first["n_samples"] == step_second["n_samples"]

    assert first_model.state_dict().keys() == second_model.state_dict().keys()
    for name, tensor in first_model.state_dict().items():
        assert torch.equal(tensor, second_model.state_dict()[name]), name

    # The recorded step shape is exactly the contract of the task.
    for step in first.steps:
        assert set(step) == {"step", "epoch", "loss", "per_head_loss", "n_samples"}
    assert [step["step"] for step in first.steps] == [1, 2, 3, 4]
    assert [step["epoch"] for step in first.steps] == [0, 0, 1, 1]
    assert first.as_dict()["memory_mode"] == "none"


def test_a_different_seed_changes_the_run() -> None:
    """A different ``seed`` (and a different init) must change the trajectory."""

    batches = [_batch(salt=0), _batch(salt=1), _batch(salt=2)]
    baseline = learn.train_bc(_fresh_actor(seed=7), batches, epochs=1, lr=1e-3,
                              seed=1, memory_mode="none", shuffle_batches=True)
    # Same initial weights, different shuffle seed -> a different batch order.
    reseeded = learn.train_bc(_fresh_actor(seed=7), batches, epochs=1, lr=1e-3,
                              seed=2, memory_mode="none", shuffle_batches=True)
    assert list(baseline.losses) != list(reseeded.losses)
    # A different initialisation must not land on the same trajectory either.
    other_init = learn.train_bc(_fresh_actor(seed=8), batches, epochs=1, lr=1e-3,
                                seed=1, memory_mode="none", shuffle_batches=True)
    assert list(other_init.losses) != list(baseline.losses)


# ---------------------------------------------------------------------------
# 3. checkpoint round-trip
# ---------------------------------------------------------------------------
def test_checkpoint_roundtrip_keeps_the_logits_bitwise_identical(
        tmp_path: Path) -> None:
    model = _fresh_actor(seed=11)
    observation_batch = _observation(batch=2, salt=3)
    with torch.no_grad():
        before, hidden_before = model(observation_batch)

    target = tmp_path / "bc.ckpt"
    returned = learn.save_checkpoint(model, target, extra={"note": "synthetic"})
    assert returned == str(target)
    assert target.is_file()

    loaded = learn.load_checkpoint(target)
    assert isinstance(loaded.model, actor.ActorPolicy)
    with torch.no_grad():
        after, hidden_after = loaded.model(observation_batch)

    for head in contracts.HEAD_ORDER:
        assert torch.equal(before[head], after[head]), head
    assert torch.equal(hidden_before, hidden_after)

    # The identity the checkpoint must carry.
    assert loaded.o2_digest == contracts.o2_digest()
    assert loaded.spec_revision == contracts.SPEC_REVISION
    assert loaded.torch_version == str(torch.__version__)
    assert loaded.extra == {"note": "synthetic"}
    assert loaded.model_config["model_spec"] == actor.MODEL_SPEC
    assert loaded.model_config["head_sizes"] == dict(contracts.HEAD_SIZES)
    assert loaded.model_config["joint_count"] == contracts.JOINT_COUNT == 72
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict[name]), name


def test_save_checkpoint_refuses_to_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "keep.ckpt"
    learn.save_checkpoint(_fresh_actor(seed=12), target)
    before = target.read_bytes()

    with pytest.raises(learn.LearnError, match="refusing to overwrite"):
        learn.save_checkpoint(_fresh_actor(seed=13), target)
    assert target.read_bytes() == before, "the existing checkpoint was modified"

    # A missing parent directory is refused instead of created silently.
    with pytest.raises(learn.LearnError, match="parent directory"):
        learn.save_checkpoint(_fresh_actor(seed=12), tmp_path / "absent" / "x.ckpt")


def test_load_checkpoint_rejects_a_foreign_o2_digest(tmp_path: Path) -> None:
    model = _fresh_actor(seed=14)
    foreign = copy.deepcopy(learn.checkpoint_model_config())
    payload = {
        "schema": learn.CHECKPOINT_SCHEMA,
        "model_config": foreign,
        "state_dict": {name: tensor.detach().clone()
                       for name, tensor in model.state_dict().items()},
        "torch_version": str(torch.__version__),
        "o2_digest": "0" * 64,
        "spec_revision": contracts.SPEC_REVISION,
        "extra": {},
    }
    target = tmp_path / "foreign.ckpt"
    torch.save(payload, target)

    with pytest.raises(learn.LearnError, match="o2_digest"):
        learn.load_checkpoint(target)
    # The same file with the current digest loads, so the refusal is the digest.
    payload["o2_digest"] = contracts.o2_digest()
    torch.save(payload, target)
    assert learn.load_checkpoint(target).o2_digest == contracts.o2_digest()

    with pytest.raises(learn.LearnError, match="o2_digest"):
        learn.load_checkpoint(target, expected_o2_digest="f" * 64)
    # There is no way to *bypass* the mandatory current-encoder comparison: even
    # naming the foreign digest explicitly still refuses (the current `o2_digest`
    # check runs first, `expected_o2_digest` only adds a constraint).
    payload["o2_digest"] = "0" * 64
    torch.save(payload, target)
    with pytest.raises(learn.LearnError, match="current observation encoder"):
        learn.load_checkpoint(target, expected_o2_digest="0" * 64)
    with pytest.raises(learn.LearnError, match="does not exist"):
        learn.load_checkpoint(tmp_path / "nothing.ckpt")


# ---------------------------------------------------------------------------
# 4./5. the two gates come before training
# ---------------------------------------------------------------------------
def test_invalid_manifest_is_refused_before_training(
        dataset: tuple[dict[str, Any], Path], tmp_path: Path) -> None:
    doc, root = dataset
    broken = copy.deepcopy(doc)
    broken["shards"][0]["sha256"] = "0" * 64
    path = _write_manifest(broken, tmp_path)

    report = validate_manifest(load_manifest(path), base_dir=root)
    assert report["valid"] is False
    codes = {problem["code"] for problem in report["problems"]}
    assert codes, "the manifest validator found no problem"

    with pytest.raises(learn.LearnError, match="manifest is not valid") as excinfo:
        learn.train_from_manifest(path, split_config=REAL_SPLITS, batches=(),
                                  base_dir=root)
    message = str(excinfo.value)
    assert any(code in message for code in codes), message
    # A missing manifest is a refusal, not a default.
    with pytest.raises(learn.LearnError):
        learn.train_from_manifest(tmp_path / "absent.json", split_config=REAL_SPLITS,
                                  batches=())
    # A structurally empty manifest is refused too (no silent empty dataset).
    with pytest.raises(learn.LearnError, match="manifest is not valid"):
        learn.train_from_manifest({}, split_config=REAL_SPLITS, batches=())


def test_train_dev_crossing_is_refused_before_training(
        dataset: tuple[dict[str, Any], Path], tmp_path: Path) -> None:
    doc, root = dataset
    path = _write_manifest(doc, tmp_path)
    assert validate_manifest(load_manifest(path), base_dir=root)["valid"] is True

    real = json.loads(REAL_SPLITS.read_text(encoding="utf-8"))
    assert lineage.validate_splits(REAL_SPLITS)["passed"] is True

    crossed = copy.deepcopy(real)
    dev_ids = list(crossed["splits"]["dev"]["ancestor_ids"])
    assert dev_ids, "the frozen registry must have a dev ancestor to cross"
    crossed["splits"]["train"]["ancestor_ids"] = list(
        crossed["splits"]["train"]["ancestor_ids"]) + [dev_ids[0]]

    crossing_report = lineage.validate_splits(crossed)
    assert crossing_report["passed"] is False
    crossing_codes = {problem["code"] for problem in crossing_report["problems"]}
    assert crossing_codes & {"X_TRAIN_OVERLAP", "A_MULTI_SPLIT"}, crossing_codes

    with pytest.raises(learn.LearnError, match="split registry did not pass") as excinfo:
        learn.train_from_manifest(path, split_config=crossed, batches=[_batch()],
                                  base_dir=root)
    message = str(excinfo.value)
    assert any(code in message for code in crossing_codes), message


# ---------------------------------------------------------------------------
# 6. labels out of range / wrong count
# ---------------------------------------------------------------------------
def test_out_of_range_labels_are_refused() -> None:
    logits = {"move": torch.zeros(2, 9), "focus": torch.zeros(2, 2),
              "shoot": torch.zeros(2, 2), "bomb": torch.zeros(2, 2)}
    bad_move = {"move": torch.tensor([0, 9]), "focus": torch.tensor([0, 1]),
                "shoot": torch.tensor([0, 1]), "bomb": torch.tensor([0, 1])}
    with pytest.raises(learn.LearnError, match="valid 0..8"):
        learn.bc_loss(logits, bad_move)

    bad_bomb = {"move": torch.tensor([0, 1]), "focus": torch.tensor([0, 1]),
                "shoot": torch.tensor([0, 1]), "bomb": torch.tensor([0, 2])}
    with pytest.raises(learn.LearnError, match="valid 0..1"):
        learn.bc_loss(logits, bad_bomb)

    negative = {"move": torch.tensor([-1, 0]), "focus": torch.tensor([0, 1]),
                "shoot": torch.tensor([0, 1]), "bomb": torch.tensor([0, 1])}
    with pytest.raises(learn.LearnError):
        learn.bc_loss(logits, negative)

    # Training refuses the same batch (the loss is not bypassed by the loop).
    with pytest.raises(learn.LearnError, match="valid 0..8"):
        learn.train_bc(_fresh_actor(seed=21),
                       [{"observations": _observation(batch=2),
                         "targets": bad_move}],
                       epochs=1, lr=1e-3, seed=1)

    # A float label is not silently truncated to an index.
    float_labels = {"move": torch.tensor([0.0, 1.0]), "focus": torch.tensor([0, 1]),
                    "shoot": torch.tensor([0, 1]), "bomb": torch.tensor([0, 1])}
    with pytest.raises(learn.LearnError, match="not integral"):
        learn.bc_loss(logits, float_labels)


def test_label_count_mismatch_is_refused() -> None:
    logits = {"move": torch.zeros(2, 9), "focus": torch.zeros(2, 2),
              "shoot": torch.zeros(2, 2), "bomb": torch.zeros(2, 2)}
    short = {"move": torch.tensor([0]), "focus": torch.tensor([0, 1]),
             "shoot": torch.tensor([0, 1]), "bomb": torch.tensor([0, 1])}
    with pytest.raises(learn.LearnError, match="label"):
        learn.bc_loss(logits, short)
    # A missing head is not zero-filled either.
    with pytest.raises(learn.LearnError, match="four heads"):
        learn.bc_loss(logits, {"move": torch.tensor([0, 1]),
                                "focus": torch.tensor([0, 1]),
                                "shoot": torch.tensor([0, 1])})
    # A batch with the wrong key set is refused (never silently ignored).
    with pytest.raises(learn.LearnError, match="training batch"):
        learn.train_bc(_fresh_actor(seed=22),
                       [{"observation": _observation(batch=1),
                         "targets": _labels(batch=1)}],
                       epochs=1, lr=1e-3, seed=1)


# ---------------------------------------------------------------------------
# 7. empty batches / epochs / lr
# ---------------------------------------------------------------------------
def test_empty_batches_epochs_and_lr_are_refused() -> None:
    model = _fresh_actor(seed=31)
    batches = [_batch(batch=1)]
    with pytest.raises(learn.LearnError, match="batches is empty"):
        learn.train_bc(model, [], epochs=1, lr=1e-3, seed=0)
    with pytest.raises(learn.LearnError, match="batches"):
        learn.train_bc(model, {"observations": {}, "targets": {}}, epochs=1,
                       lr=1e-3, seed=0)
    for epochs in (0, -1):
        with pytest.raises(learn.LearnError, match="epochs"):
            learn.train_bc(model, batches, epochs=epochs, lr=1e-3, seed=0)
    for lr in (0.0, -1e-4):
        with pytest.raises(learn.LearnError, match="lr"):
            learn.train_bc(model, batches, epochs=1, lr=lr, seed=0)
    with pytest.raises(learn.LearnError, match="max_steps"):
        learn.train_bc(model, batches, epochs=1, lr=1e-3, seed=0, max_steps=0)
    with pytest.raises(learn.LearnError, match="memory_mode"):
        learn.train_bc(model, batches, epochs=1, lr=1e-3, seed=0,
                       memory_mode="transformer")
    # Nothing above may have moved a weight.
    reference = _fresh_actor(seed=31)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, reference.state_dict()[name]), name


# ---------------------------------------------------------------------------
# 8. memory mode is recorded honestly
# ---------------------------------------------------------------------------
def test_memory_mode_is_recorded_for_none_and_gru() -> None:
    # D-pixel: one observation, no hidden state passed to the actor.
    pixel = learn.train_bc(_fresh_actor(seed=41), [_batch(batch=2, salt=2)],
                           epochs=1, lr=1e-3, seed=3, memory_mode="none")
    assert pixel.memory_mode == "none"
    assert pixel.as_dict()["memory_mode"] == "none"
    assert pixel.n_steps == 1
    assert pixel.steps[0]["n_samples"] == 2

    # D-pixel-memory: a sequence of steps, the GRU state carried across it.
    sequence = learn.train_bc(_fresh_actor(seed=41),
                              [_sequence_batch(steps=3, batch=2, salt=2)],
                              epochs=1, lr=1e-3, seed=3, memory_mode="gru")
    assert sequence.memory_mode == "gru"
    assert sequence.as_dict()["memory_mode"] == "gru"
    assert sequence.n_steps == 1
    assert sequence.steps[0]["n_samples"] == 6, "three steps of two samples"

    # The mode is not just a label: the GRU state must really be carried.  Reproduce
    # the recorded step loss by hand, stepping the same initial weights with an
    # explicit hidden state and averaging the three per-step BC losses.
    batch = _sequence_batch(steps=3, batch=2, salt=2)
    model = _fresh_actor(seed=41)
    report = learn.train_bc(model, [batch], epochs=1, lr=1e-3, seed=3,
                            memory_mode="gru")
    carried = 0.0
    stateless = 0.0
    with torch.no_grad():
        replay = _fresh_actor(seed=41)
        hidden = None
        for step_observations, step_labels in zip(batch["observations"],
                                                  batch["targets"]):
            logits, hidden = replay(step_observations, hidden)
            carried += learn.bc_loss(logits, step_labels).item()
        replay = _fresh_actor(seed=41)
        for step_observations, step_labels in zip(batch["observations"],
                                                  batch["targets"]):
            logits, _zero = replay(step_observations)  # a fresh zero state each step
            stateless += learn.bc_loss(logits, step_labels).item()
    assert abs(report.steps[0]["loss"] - carried / 3.0) < 1e-6, \
        "the recorded loss is not the carried-state mean over the sequence"
    assert abs(stateless / 3.0 - carried / 3.0) > 1e-6, \
        "the hidden state made no difference, so 'gru' is only a label"

    # Mismatched mode and data shape is refused, not coerced.
    with pytest.raises(learn.LearnError, match="sequence"):
        learn.train_bc(_fresh_actor(seed=41), [_sequence_batch(steps=2)],
                       epochs=1, lr=1e-3, seed=3, memory_mode="none")
    with pytest.raises(learn.LearnError, match="sequence"):
        learn.train_bc(_fresh_actor(seed=41), [_batch(batch=2)],
                       epochs=1, lr=1e-3, seed=3, memory_mode="gru")


def test_train_from_manifest_records_the_memory_mode_and_the_result_schema(
        dataset: tuple[dict[str, Any], Path], tmp_path: Path) -> None:
    doc, root = dataset
    path = _write_manifest(doc, tmp_path)

    result = learn.train_from_manifest(
        path, split_config=REAL_SPLITS, batches=[_batch(batch=1, salt=4)],
        epochs=1, lr=1e-3, seed=5, memory_mode="none",
        base_dir=root, run=True)
    validated = contracts.assert_result_schema(result)
    assert validated["execution_status"] == "ok"
    assert validated["protocol_validity"] == "valid"
    assert validated["capability_gate"] == "not_run", "training is not a capability"
    assert validated["command_class"] == "training"
    research = validated["research_result"]
    assert research["status"] == "inconclusive", "training is not a research pass"
    assert research["memory_mode"] == "none"
    assert research["batch_source"] == "synthetic"
    assert research["manifest_validation"]["valid"] is True
    assert research["splits_validation"]["passed"] is True
    assert research["training"]["n_steps"] == 1
    assert research["training"]["memory_mode"] == "none"
    assert research["training"]["steps"][0]["n_samples"] == 1
    assert validated["o2_digest"] == contracts.o2_digest()
    assert validated["spec_revision"] == contracts.SPEC_REVISION
    assert validated["synthetic_data_only"] is True

    # ``run=False`` validates only: the result must say so, not pretend to train.
    dry = learn.train_from_manifest(path, split_config=REAL_SPLITS, run=False,
                                    memory_mode="gru", base_dir=root)
    dry = contracts.assert_result_schema(dry)
    assert dry["research_result"]["status"] == "not_run"
    assert dry["research_result"]["training"]["status"] == "not_run"
    assert dry["research_result"]["memory_mode"] == "gru"
    assert dry["research_result"]["training"]["n_steps"] == 0


# ---------------------------------------------------------------------------
# 9. max_steps
# ---------------------------------------------------------------------------
def test_max_steps_stops_the_loop() -> None:
    batches = [_batch(batch=1, salt=6)]
    report = learn.train_bc(_fresh_actor(seed=51), batches, epochs=200, lr=1e-3,
                            seed=2, max_steps=100)
    assert report.n_steps == 100
    assert [step["step"] for step in report.steps] == list(range(1, 101))
    # One batch per epoch, so 100 steps use epochs 0..99 and stop before 200.
    assert {step["epoch"] for step in report.steps} == set(range(100))
    assert report.steps[-1]["epoch"] == 99
    assert report.max_steps == 100

    # Two batches per epoch (6 steps available), capped at 5: the loop stops mid-run.
    two = [_batch(batch=1, salt=6), _batch(batch=1, salt=7)]
    short = learn.train_bc(_fresh_actor(seed=51), two, epochs=3, lr=1e-3,
                           seed=2, max_steps=5)
    assert short.n_steps == 5
    assert short.steps[-1]["epoch"] == 2
    uncapped = learn.train_bc(_fresh_actor(seed=51), two, epochs=3, lr=1e-3, seed=2)
    assert uncapped.n_steps == 6


# ---------------------------------------------------------------------------
# 12./13. purity and the synthetic-scope header
# ---------------------------------------------------------------------------
def test_learn_does_not_import_engine_v400_v420_or_gym() -> None:
    tree = ast.parse(LEARN_SOURCE.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    forbidden = ("touhou_rl.v400", "touhou_rl.v420", "gym", "gymnasium",
                 "third_party")
    for name in sorted(modules):
        for bad in forbidden:
            assert not (name == bad or name.startswith(bad + ".")), \
                f"learn.py imports {name!r}, which this slice forbids"
    # The allowed dependency set is really used (the check above is not vacuous).
    assert "torch" in modules
    assert "touhou_rl.v430" in modules
    assert "touhou_rl.v430.models" in modules
    assert "touhou_rl.v430.data" in modules


def test_file_header_declares_the_synthetic_scope() -> None:
    header = TEST_SOURCE.read_text(encoding="utf-8")[:4000]
    assert "合成测试，不是游戏/能力证据" in header
    assert "synthetic" in header.lower()

    learn_doc = LEARN_SOURCE.read_text(encoding="utf-8")[:8000]
    assert "synthetic" in learn_doc.lower()
    assert "no engine" in learn_doc.lower()
    assert "synthetic" in (learn.__doc__ or "").lower()


# ---------------------------------------------------------------------------
# W03G: the ``train-bc`` CLI subcommand (synthetic batches only)
# ---------------------------------------------------------------------------
CLI_PATH = REPO / "tools" / "rl" / "v430.py"
CLI_MODULE_NAME = "v430_cli_under_test_w03g"


def _cli_module():
    """Load ``tools/rl/v430.py`` as a module (it is a script, not a package)."""

    import importlib.util
    import sys as _sys

    module = _sys.modules.get(CLI_MODULE_NAME)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(CLI_MODULE_NAME, CLI_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _sys.modules[CLI_MODULE_NAME] = module
    return module


def _batches_spec(directory: Path, *, payload: Mapping[str, Any] | None = None) -> Path:
    """Write a minimal synthetic-batch specification; ``payload`` overrides keys."""

    document: dict[str, Any] = {
        "schema": "v430-synthetic-batches/1",
        "n_batches": 1,
        "batch_size": 1,
        "seed": 424242,
        "steps": 1,
        "elapsed_capture_s": 0.016,
    }
    if payload is not None:
        document.update(dict(payload))
    path = directory / "batches.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _run_cli(argv: list[str]) -> int:
    """Run the CLI in-process and return the exit code ``main()`` returns."""

    return int(_cli_module().main(argv))


def _train_bc_argv(spec: Path, out: Path, **overrides: Any) -> list[str]:
    argv = ["train-bc", "--batches", str(spec), "--epochs", "1", "--lr", "1e-3",
            "--seed", "7", "--max-steps", "1", "--out", str(out)]
    for flag, value in overrides.items():
        if value is True:
            argv.append(f"--{flag.replace('_', '-')}")
        elif value is not None:
            argv.extend([f"--{flag.replace('_', '-')}", str(value)])
    return argv


def test_train_bc_writes_a_schema_valid_training_result(tmp_path: Path) -> None:
    """A legal run exits 0 and the document is a *training* result, not a capability."""

    spec = _batches_spec(tmp_path)
    out = tmp_path / "train-bc-001.json"
    assert _run_cli(_train_bc_argv(spec, out)) == 0
    assert out.is_file()
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["command_class"] == "training"
    assert result["capability_gate"] == "not_run"
    assert result["dataset_kind"] == "synthetic"
    assert result["execution_status"] == "ok"
    assert result["protocol_validity"] == "valid"
    assert result["exit_code"] == 0
    assert result["memory_mode"] == "none"
    assert result["n_steps"] == 1
    assert result["epochs"] == 1
    assert result["lr"] == 1e-3
    assert result["seed"] == 7
    assert result["shuffle_batches"] is False
    assert result["max_steps"] == 1
    assert result["o2_digest"] == contracts.o2_digest()
    assert result["parameter_report"]["total"] == result["trainable_parameter_count"]
    assert result["parameter_report"]["total"] > 0
    assert len(result["losses"]) == result["n_steps"]
    assert result["batches"]["observation_dtype"] == "uint8"
    # The written document is exactly what the contract accepts.
    contracts.assert_result_schema(result)


def test_train_bc_same_seed_reproduces_the_loss_sequence(tmp_path: Path) -> None:
    """Two runs of one seed give bitwise identical batches and losses."""

    spec = _batches_spec(tmp_path, payload={"n_batches": 2, "batch_size": 2})
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    argv = ["train-bc", "--batches", str(spec), "--epochs", "1", "--lr", "1e-3",
            "--seed", "7", "--max-steps", "2"]
    assert _run_cli(argv + ["--out", str(first)]) == 0
    assert _run_cli(argv + ["--out", str(second)]) == 0
    a = json.loads(first.read_text(encoding="utf-8"))
    b = json.loads(second.read_text(encoding="utf-8"))
    assert a["losses"] == b["losses"]
    assert a["steps"] == b["steps"]
    assert len(a["losses"]) == 2
    assert a["batches"]["signature_sha256"] == b["batches"]["signature_sha256"]


def test_train_bc_refuses_an_existing_out_and_leaves_it_byte_identical(
        tmp_path: Path) -> None:
    """``--out`` is create-exclusive: exit 2, and the file is not touched."""

    spec = _batches_spec(tmp_path)
    out = tmp_path / "existing.json"
    sentinel = b'{"sentinel": "do not touch"}\n'
    out.write_bytes(sentinel)
    assert _run_cli(_train_bc_argv(spec, out)) == 2
    assert out.read_bytes() == sentinel


def test_train_bc_refuses_an_unregistered_memory_mode(tmp_path: Path) -> None:
    """``--memory-mode`` must be one of ``learn.MEMORY_MODES`` (exit 2)."""

    spec = _batches_spec(tmp_path)
    out = tmp_path / "out.json"
    argv = _train_bc_argv(spec, out, memory_mode="lstm")
    assert _run_cli(argv) == 2
    assert not out.exists()


def test_train_bc_refuses_an_invalid_batches_spec(tmp_path: Path) -> None:
    """A missing required field and a non-O2 shape are both refused (exit 2)."""

    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps({"batch_size": 1, "seed": 1}), encoding="utf-8")
    out_missing = tmp_path / "missing-out.json"
    assert _run_cli(_train_bc_argv(missing, out_missing)) == 2
    assert not out_missing.exists()

    wrong_shape = tmp_path / "shape.json"
    wrong_shape.write_text(json.dumps({
        "n_batches": 1, "batch_size": 1, "seed": 1,
        "observation_shapes": {"play_gray": [4, 224, 128]},
    }), encoding="utf-8")
    out_shape = tmp_path / "shape-out.json"
    assert _run_cli(_train_bc_argv(wrong_shape, out_shape)) == 2
    assert not out_shape.exists()


def test_a_training_result_cannot_promote_itself_to_a_capability(tmp_path: Path) -> None:
    """The anti-forgery gate: ``command_class='training'`` + ``pass`` is refused.

    The forgery is built as favourably as possible: the capability claim carries a
    *complete, well-formed* ``capability_judgement`` (judge identifier + 64-hex
    evidence digest), so the only thing left that can refuse it is the training
    command class itself.  The control case below (same forged document, class
    ``evaluation``) passes the schema, which proves the refusal is caused by
    ``command_class`` and not by a malformed judgement.
    """

    spec = _batches_spec(tmp_path)
    out = tmp_path / "result.json"
    assert _run_cli(_train_bc_argv(spec, out)) == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["capability_gate"] == "not_run"

    bare = copy.deepcopy(result)
    bare["capability_gate"] = "pass"
    with pytest.raises(contracts.ContractError) as bare_exc:
        contracts.assert_result_schema(bare)
    assert "capability_judgement" in str(bare_exc.value)

    forged = copy.deepcopy(bare)
    forged["capability_judgement"] = {"judge": "w03g-forged-judge",
                                      "sha256": "0" * 64}
    with pytest.raises(contracts.ContractError) as forged_exc:
        contracts.assert_result_schema(forged)
    message = str(forged_exc.value)
    assert "training exit 0 is not a capability" in message
    assert "capability_gate='pass'" in message

    # The result constructor cannot be talked into it either.
    with pytest.raises(contracts.ContractError):
        contracts.new_result(execution_status="ok", protocol_validity="valid",
                             capability_gate="pass")

    control = copy.deepcopy(forged)
    control["command_class"] = "evaluation"
    assert contracts.assert_result_schema(control)["capability_gate"] == "pass"


def test_train_bc_refuses_an_existing_checkpoint_and_writes_a_new_one(
        tmp_path: Path) -> None:
    """``--checkpoint-out`` is create-exclusive; a fresh path is recorded in the result."""

    spec = _batches_spec(tmp_path)
    existing = tmp_path / "existing.pt"
    sentinel = b"not a checkpoint, do not overwrite"
    existing.write_bytes(sentinel)
    out = tmp_path / "existing-out.json"
    assert _run_cli(_train_bc_argv(spec, out, checkpoint_out=str(existing))) == 2
    assert existing.read_bytes() == sentinel
    assert not out.exists()

    fresh = tmp_path / "fresh.pt"
    good = tmp_path / "fresh-out.json"
    assert _run_cli(_train_bc_argv(spec, good, checkpoint_out=str(fresh))) == 0
    assert fresh.is_file()
    assert fresh.stat().st_size > 0
    result = json.loads(good.read_text(encoding="utf-8"))
    assert result["checkpoint"]["path"].endswith("fresh.pt")
    assert len(result["checkpoint"]["sha256"]) == 64
    assert result["checkpoint"]["o2_digest"] == contracts.o2_digest()
    checkpoint = learn.load_checkpoint(fresh, expected_o2_digest=contracts.o2_digest())
    assert checkpoint.extra["dataset_kind"] == "synthetic"
    assert checkpoint.extra["command"] == "train-bc"


# ---------------------------------------------------------------------------
# W04B: the O2 guard is wired into the training path, and ``batch_source`` is a
# caller declaration instead of a hard-coded "synthetic"
# ---------------------------------------------------------------------------
LEGACY_OBS_CONTRACT = "obs-v420-2f"


def _legacy_document(doc: Mapping[str, Any]) -> dict[str, Any]:
    """The same, still-``valid`` manifest, relabelled to the legacy contract.

    The relabelling is the whole corpus: nothing else about the manifest changes,
    which is exactly the case a ``valid`` report cannot see.  ``obs-v420-2f`` is a
    *registered* contract (:data:`LEGACY_OBS_CONTRACTS`), so
    :func:`validate_manifest` accepts it — the O2 guard is the only thing left
    that can refuse it.
    """

    assert LEGACY_OBS_CONTRACT in LEGACY_OBS_CONTRACTS, (
        "the fixture's legacy contract is no longer registered; pick a real one")
    legacy = copy.deepcopy(dict(doc))
    legacy["obs_contract"] = LEGACY_OBS_CONTRACT
    return legacy


def test_train_from_manifest_refuses_a_legacy_obs_contract(
        dataset: tuple[dict[str, Any], Path], tmp_path: Path) -> None:
    """The manifest gate alone lets a legacy corpus through; the O2 guard does not.

    Also pins that the refusal happens **before any training step**: the model
    passed in comes out bitwise unchanged.
    """

    doc, root = dataset
    # The O2 manifest itself is valid (the fixture is the baseline)...
    o2_path = _write_manifest(doc, tmp_path)
    assert validate_manifest(load_manifest(o2_path), base_dir=root)["valid"] is True

    # ...and so is the relabelled one: ``obs_contract`` is bound by name only, so
    # the manifest validator is not the guard this test relies on.
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    legacy_path = _write_manifest(_legacy_document(doc), legacy_dir)
    legacy_report = validate_manifest(load_manifest(legacy_path), base_dir=root)
    assert legacy_report["valid"] is True, (
        "if the manifest validator already refused the legacy contract, this test "
        "would not prove the O2 guard is wired in")

    model = _fresh_actor(seed=77)
    before = {name: tensor.detach().clone()
              for name, tensor in model.state_dict().items()}

    # The refusal is the manifest layer's own exception type, propagated as-is
    # (it is a RuntimeError but deliberately *not* a LearnError).
    with pytest.raises(ManifestError) as excinfo:
        learn.train_from_manifest(
            legacy_path, split_config=REAL_SPLITS, model=model,
            batches=[_batch(batch=1, salt=8)], base_dir=root, run=True)
    message = str(excinfo.value)
    assert not isinstance(excinfo.value, learn.LearnError)
    assert LEGACY_OBS_CONTRACT in message, message
    assert "obs-v430-o2" in message, message
    # No step ran: every weight is bitwise identical.
    after = model.state_dict()
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_dry_run_also_refuses_a_legacy_obs_contract(
        dataset: tuple[dict[str, Any], Path], tmp_path: Path) -> None:
    """``run=False`` must not report ``protocol_validity="valid"`` for a legacy corpus.

    The dry path returns a result whose ``protocol_validity`` is ``"valid"``; if
    the guard were skipped there, that verdict would be the misleading artefact
    this slice exists to remove.  So the dry run must raise too — there is no
    result to misread.
    """

    doc, root = dataset
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    legacy_path = _write_manifest(_legacy_document(doc), legacy_dir)

    with pytest.raises(ManifestError) as excinfo:
        learn.train_from_manifest(legacy_path, split_config=REAL_SPLITS,
                                  run=False, base_dir=root)
    assert LEGACY_OBS_CONTRACT in str(excinfo.value)
    assert not isinstance(excinfo.value, learn.LearnError)

    # The O2 dry run still works and is still honest about not having trained.
    dry = learn.train_from_manifest(_write_manifest(doc, tmp_path),
                                    split_config=REAL_SPLITS, run=False,
                                    base_dir=root)
    assert dry["protocol_validity"] == "valid"
    assert dry["research_result"]["status"] == "not_run"
    assert dry["research_result"]["training"]["n_steps"] == 0


def test_batch_source_defaults_to_synthetic(
        dataset: tuple[dict[str, Any], Path], tmp_path: Path) -> None:
    """The default stays ``"synthetic"`` — and says so as a declaration."""

    doc, root = dataset
    path = _write_manifest(doc, tmp_path)

    result = learn.train_from_manifest(
        path, split_config=REAL_SPLITS, batches=[_batch(batch=1, salt=9)],
        epochs=1, lr=1e-3, seed=2, base_dir=root)
    validated = contracts.assert_result_schema(result)
    assert validated["batch_source"] == "synthetic"
    assert validated["synthetic_data_only"] is True
    research = validated["research_result"]
    assert research["batch_source"] == "synthetic"
    assert research["synthetic_data_only"] is True
    assert research["scope"] == "bc-orchestration/synthetic-batches"
    # Recorded as a declaration, never as a measurement.
    assert validated["batch_source_verified"] is False
    assert research["batch_source_verified"] is False
    assert "NOT verified" in validated["batch_source_note"]


def test_batch_source_captured_o2_is_not_called_synthetic(
        dataset: tuple[dict[str, Any], Path], tmp_path: Path) -> None:
    """``captured_o2`` flips the derived flag and the scope, and stays unverified."""

    doc, root = dataset
    path = _write_manifest(doc, tmp_path)

    result = learn.train_from_manifest(
        path, split_config=REAL_SPLITS, batches=[_batch(batch=1, salt=10)],
        epochs=1, lr=1e-3, seed=2, batch_source="captured_o2", base_dir=root)
    validated = contracts.assert_result_schema(result)
    assert validated["batch_source"] == "captured_o2"
    assert validated["synthetic_data_only"] is False
    research = validated["research_result"]
    assert research["batch_source"] == "captured_o2"
    assert research["synthetic_data_only"] is False
    assert research["scope"] == "bc-orchestration/captured-o2-batches"
    # The declaration is *not* a verification: the module cannot inspect the
    # batches, so it must never claim it did.
    assert validated["batch_source_verified"] is False
    assert "captured_o2" in validated["batch_source_note"]
    assert "not verified" in validated["batch_source_note"].lower()
    # Everything else about the run is unchanged by the declaration.
    assert validated["execution_status"] == "ok"
    assert validated["research_result"]["training"]["n_steps"] == 1

    # ``run=False`` carries the same declaration (a dry run can classify batches).
    dry = learn.train_from_manifest(path, split_config=REAL_SPLITS, run=False,
                                    batch_source="captured_o2", base_dir=root)
    assert dry["batch_source"] == "captured_o2"
    assert dry["synthetic_data_only"] is False


def test_an_unregistered_batch_source_is_refused(
        dataset: tuple[dict[str, Any], Path], tmp_path: Path) -> None:
    """``"real"`` is not a registered source: refuse, do not guess."""

    doc, root = dataset
    path = _write_manifest(doc, tmp_path)

    with pytest.raises(learn.LearnError, match="batch_source") as excinfo:
        learn.train_from_manifest(
            path, split_config=REAL_SPLITS, batches=[_batch()],
            batch_source="real", base_dir=root)
    message = str(excinfo.value)
    assert "synthetic" in message and "captured_o2" in message, message

    # Refused before anything else happens: no batches needed, no manifest read.
    with pytest.raises(learn.LearnError, match="batch_source"):
        learn.train_from_manifest(path, split_config=REAL_SPLITS, run=False,
                                  batch_source="", base_dir=root)
    with pytest.raises(learn.LearnError, match="batch_source"):
        learn.train_from_manifest(tmp_path / "absent.json", split_config=REAL_SPLITS,
                                  run=False, batch_source="Captured_O2")
