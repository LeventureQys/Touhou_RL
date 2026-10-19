"""Compare state checkpoints on a verified, reproducible validation sample (C6).

What changed in v1.0.4 and why
------------------------------
The v1.0.3 tool re-derived the split with ``split_indices`` instead of *verifying*
the dataset manifest, hard-coded ``depth=1`` (a depth-3 checkpoint was fed a single
frame), silently took the first 128 validation indices per stage even for a
"full validation" question, and printed ``mae`` already rounded to two decimals.
A-03 is a ratio rule (``B <= 1.25 * A``), so rounding before the comparison can
flip the verdict; and a re-derived split is not the split the model was trained
under - it looks identical until the dataset changes.

This version:

* takes ``--manifest`` (A defaults to ``<dataset>/manifest.json``; B must pass the
  S3-1 read-only evaluation manifest) and **verifies** shard bytes, the frame
  total and the split hash before any frame is read.  A missing B manifest is
  ``v104:DATA`` - it never falls back to re-splitting;
* adds ``--mode full-val|stage-sample`` (default ``stage-sample`` keeps the old
  sampling), ``--per-stage 128`` and reports how many frames each stage really had;
* takes the frame stack from the checkpoint's ``config.depth`` and builds the
  history with the dataset's C2 contract (``legacy`` for old checkpoints), never a
  hand-rolled reshape;
* returns the unrounded ``mae_raw`` (and per-stage ``mae_raw``) for threshold
  maths, keeps the rounded ``mae``/``median``/``p90`` for display, and records
  ``selected_index_sha256``, ``sampling_mode``, ``per_stage``, ``depth``, the
  manifest hash and the checkpoint hash so a reviewer can recompute every number;
* A-03 uses ``--mode stage-sample`` on A and B with the *same checkpoint bytes*
  and compares ``B_mae_raw <= 1.25 * A_mae_raw`` (``--baseline`` feeds A's earlier
  output).  A-01 uses ``--mode full-val`` on A only; a full-val mean must never be
  compared with B's stage-sample mean.

Error format is the shared ``v104:<CODE>: <detail>`` (CONFIG/DATA/CHECKPOINT); the
CLI prints a known error to stderr with the same prefix and exits 2, an unknown
exception keeps its traceback and a non-zero exit.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

from touhou_rl.pipeline.manifest import (
    DatasetManifest,
    ManifestError,
    file_digest,
    index_digest,
    split_indices,
)
from touhou_rl.vision import regions_to_batch
from touhou_rl.vision_data import VisionDataset
from touhou_rl.vision_history import (
    HISTORY_LEGACY,
    normalize_history_contract,
    v104_error,
)
from touhou_rl.vmodels import load_vision_checkpoint, vision_arch

ROOT = Path(__file__).resolve().parents[2]

SAMPLING_MODES = ("full-val", "stage-sample")
DEFAULT_MODE = "stage-sample"
DEFAULT_PER_STAGE = 128
DEFAULT_SEED = 7
DEFAULT_VAL_FRACTION = 0.1
A03_RATIO = 1.25
A03_RULE = "B_mae_raw <= 1.25 * A_mae_raw (stage-sample, same checkpoint bytes)"
#: Position error, unchanged since the v1.0.3 pilot:
#: ``(|dx| * 384 + |dy| * 448) / 2`` in native px.
POS_SCALE_X = 384
POS_SCALE_Y = 448

V104_CODES = frozenset(
    {"CONFIG", "DATA", "CACHE", "TEACHER", "CHECKPOINT", "COVERAGE", "BUDGET", "EVIDENCE"}
)
_V104_PREFIX = re.compile(r"^v104:([A-Z]+): ")


def v104_error_code(message: object) -> str | None:
    """``"DATA"`` for ``"v104:DATA: ..."``, else ``None``."""
    match = _V104_PREFIX.match(str(message))
    if match and match.group(1) in V104_CODES:
        return match.group(1)
    return None


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
def load_manifest(dataset_dir: Path | str, manifest_path: Path | str | None) -> tuple[DatasetManifest, Path]:
    """Load *and verify* the manifest that describes ``dataset_dir``.

    ``manifest_path=None`` means "the dataset's own manifest" (A).  B must pass the
    S3-1 evaluation manifest explicitly; if it is absent the tool refuses instead
    of calling ``split_indices`` again, because a re-derived split is not evidence
    of anything.
    """
    dataset = Path(dataset_dir)
    if not dataset.is_dir():
        raise v104_error("DATA", f"dataset directory not found: {dataset}")
    explicit = Path(manifest_path) if manifest_path else None
    path = explicit if explicit is not None else dataset / "manifest.json"
    if explicit is not None and not explicit.is_file():
        raise v104_error(
            "DATA",
            f"evaluation manifest not found: {explicit}; refusing to re-derive the split",
        )
    if not path.is_file():
        raise v104_error(
            "DATA",
            f"no manifest at {path}; pass --manifest with the read-only evaluation manifest "
            "(refusing to re-derive the split)",
        )
    try:
        manifest = DatasetManifest.load(path)
    except (ManifestError, KeyError, ValueError) as exc:
        raise v104_error("DATA", f"manifest {path} is unusable: {exc}") from exc
    if manifest.modality != "vision":
        raise v104_error(
            "DATA", f"manifest {path} modality {manifest.modality!r} is not 'vision'"
        )
    try:
        # deep=True: shard bytes and SHA-256, total frames, extra files on disk and
        # the split reproducibility check all have to hold before sampling.
        manifest.verify(dataset, deep=True)
    except ManifestError as exc:
        raise v104_error("DATA", f"manifest {path} does not verify against {dataset}: {exc}") from exc
    return manifest, path


def verify_split(manifest: DatasetManifest) -> list[int]:
    """The validation indices recorded by the manifest, ascending, or ``v104:DATA``."""
    splits = dict(manifest.splits or {})
    for key in ("seed", "val_fraction", "val_index_sha256"):
        if key not in splits:
            raise v104_error("DATA", f"manifest splits are missing {key!r}")
    try:
        _train, val = split_indices(
            int(manifest.total_frames), int(splits["seed"]), float(splits["val_fraction"])
        )
    except (TypeError, ValueError) as exc:
        raise v104_error("DATA", f"manifest split is unusable: {exc}") from exc
    digest = index_digest(val)
    if digest != splits["val_index_sha256"]:
        raise v104_error(
            "DATA",
            f"split reproducibility check failed: recomputed {digest} != "
            f"manifest {splits['val_index_sha256']}",
        )
    if "val_frames" in splits and int(splits["val_frames"]) != len(val):
        raise v104_error(
            "DATA", f"manifest val_frames {splits['val_frames']} != recomputed {len(val)}"
        )
    return val


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def select_indices(
    dataset: VisionDataset, val: list[int], mode: str, per_stage: int
) -> tuple[list[int], dict]:
    """Pick the evaluation indices; the same rule for A and B.

    * ``full-val``: every validation index, ascending.
    * ``stage-sample``: per stage, the first ``per_stage`` validation indices in
      **ascending** index order (the old tool did the same but reported nothing
      about stages that had fewer frames, and never for full validation).

    Returns ``(selected, per_stage_report)``; the report says how many frames each
    stage offered and how many were taken, so "128" is never mistaken for a
    guarantee.
    """
    if mode not in SAMPLING_MODES:
        raise v104_error("CONFIG", f"unknown sampling mode {mode!r}; expected {list(SAMPLING_MODES)}")
    limit = int(per_stage)
    if limit < 1:
        raise v104_error("CONFIG", f"--per-stage must be >= 1, got {limit}")
    if not val:
        raise v104_error("DATA", "the manifest validation split is empty")
    stages = dataset.stage_of(val).tolist()
    ordered = sorted(zip(val, stages))
    report: dict[str, dict] = {}
    if mode == "full-val":
        selected = [int(index) for index, _stage in ordered]
        for stage in sorted({int(stage) for _index, stage in ordered}):
            frames = sum(1 for _index, value in ordered if int(value) == stage)
            report[str(stage)] = {"available": frames, "selected": frames}
        return selected, report
    selected = []
    for stage in sorted({int(stage) for _index, stage in ordered}):
        rows = [int(index) for index, value in ordered if int(value) == stage]
        take = rows[:limit]
        report[str(stage)] = {"available": len(rows), "selected": len(take)}
        selected.extend(take)
    if not selected:
        raise v104_error("DATA", "stage sampling selected no frames")
    return selected, report


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def checkpoint_history_contract(extra: dict) -> tuple[str, bool]:
    """``(contract, legacy)`` from the checkpoint's ``extra``.

    Old checkpoints have no ``history_contract`` and keep the v1.0.3 ``legacy``
    clamp; the row says so (``legacy_history``) instead of pretending the causal
    segment rule was used.
    """
    if not isinstance(extra, dict) or extra.get("history_contract") is None:
        return HISTORY_LEGACY, True
    return normalize_history_contract(extra["history_contract"]), False


def _position_error(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return (
        np.abs(pred[:, 0] - truth[:, 0]) * POS_SCALE_X
        + np.abs(pred[:, 1] - truth[:, 1]) * POS_SCALE_Y
    ) / 2.0


def evaluate_checkpoint(
    checkpoint: Path | str,
    dataset_dir: Path | str,
    *,
    manifest_path: Path | str | None = None,
    mode: str = DEFAULT_MODE,
    per_stage: int = DEFAULT_PER_STAGE,
    batch: int = 64,
    device: str = "cpu",
    label: str | None = None,
) -> dict:
    """Evaluate one checkpoint on the manifest's validation sample.

    Returns one row with the unrounded ``mae_raw`` (so a ratio can be computed
    without rounding error), the rounded display values, the selected index hash
    and the identity of everything the number depends on.  ``_errors`` carries the
    per-frame errors for callers that want them; the CLI drops it before writing
    JSON.
    """
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise v104_error("CHECKPOINT", f"checkpoint not found: {checkpoint}")
    model, extra = load_vision_checkpoint(checkpoint, map_location=device)
    arch = vision_arch(model)
    depth = int(getattr(model.config, "depth", 0) or 0)
    if depth < 1:
        raise v104_error("CHECKPOINT", f"{checkpoint}: config.depth {depth!r} is not positive")
    history_contract, legacy_history = checkpoint_history_contract(extra)
    manifest, manifest_file = load_manifest(dataset_dir, manifest_path)
    val = verify_split(manifest)

    torch.set_num_threads(1)
    sizes = {name: tuple(int(v) for v in size) for name, size in model.config.region_sizes.items()}
    batch_size = int(batch)
    if batch_size < 1:
        raise v104_error("CONFIG", f"--batch must be >= 1, got {batch_size}")

    with VisionDataset(Path(dataset_dir), history_contract=history_contract) as dataset:
        if len(dataset) != int(manifest.total_frames):
            raise v104_error(
                "DATA",
                f"dataset has {len(dataset)} frames but the manifest says {manifest.total_frames}",
            )
        selected, selection = select_indices(dataset, val, mode, per_stage)
        errors = np.zeros(len(selected), dtype=np.float64)
        stage_of = np.zeros(len(selected), dtype=np.int64)
        for start in range(0, len(selected), batch_size):
            chunk = selected[start : start + batch_size]
            # C2 history, oldest -> newest, composed by the dataset (never reshaped
            # into depth 1): the model's first conv expects 3 * depth channels.
            history = dataset.stack_indices(chunk, depth)
            flat = history.reshape(-1).tolist()
            frames = dataset.region_frames(flat, sizes)
            stacked = {
                name: array.reshape(len(chunk), depth, sizes[name][0], sizes[name][1], 3)
                for name, array in frames.items()
            }
            with torch.no_grad():
                state = model.decode_state(regions_to_batch(stacked, device))
            pred = state["self_feat"][:, :2].detach().cpu().numpy().astype(np.float64)
            labels = dataset.labels(chunk)
            truth = np.asarray(labels["self_feat"], dtype=np.float64)[:, :2]
            if not np.isfinite(pred).all():
                raise v104_error(
                    "DATA",
                    f"{checkpoint}: the model produced non-finite positions on indices "
                    f"{chunk[:3]}; a non-finite prediction is a failure, not a zero",
                )
            if not np.isfinite(truth).all():
                raise v104_error(
                    "DATA", f"{dataset_dir}: non-finite position labels on indices {chunk[:3]}"
                )
            errors[start : start + len(chunk)] = _position_error(pred, truth)
            stage_of[start : start + len(chunk)] = np.asarray(labels["meta"], dtype=np.float64)[
                :, 1
            ].astype(np.int64)

    if not np.isfinite(errors).all():
        raise v104_error("DATA", "non-finite position errors; refusing to summarise")
    per_stage_node: dict[str, dict] = {}
    for stage in sorted(set(int(value) for value in stage_of.tolist())):
        mask = stage_of == stage
        values = errors[mask]
        per_stage_node[str(stage)] = {
            "frames": int(mask.sum()),
            "available": int(selection.get(str(stage), {}).get("available", int(mask.sum()))),
            "selected": int(selection.get(str(stage), {}).get("selected", int(mask.sum()))),
            "mae": round(float(values.mean()), 2),
            "mae_raw": float(values.mean()),
            "p90": round(float(np.percentile(values, 90)), 1),
        }
    mae_raw = float(errors.mean())
    return {
        "label": label,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_digest(checkpoint),
        "arch": arch,
        "depth": depth,
        "history_contract": history_contract,
        "legacy_history": bool(legacy_history),
        "playfield_hw": [int(v) for v in sizes["playfield"]],
        "parameters": int(model.num_parameters()),
        "dataset": str(dataset_dir),
        "manifest": str(manifest_file),
        "manifest_sha256": file_digest(manifest_file),
        "val_index_sha256": str(manifest.splits["val_index_sha256"]),
        "val_frames": int(len(val)),
        "sampling_mode": mode,
        "per_stage_limit": int(per_stage),
        "per_stage": per_stage_node,
        "selection": selection,
        "frames": int(len(selected)),
        "selected_index_sha256": index_digest(selected),
        # Machine-readable selection: the rule is deterministic, but shipping the
        # exact indices means a reviewer can re-hash them instead of trusting the
        # phrase "first 128 ascending".
        "selected_indices": [int(index) for index in selected],
        # Unrounded on purpose: A-03 compares ratios, so rounding here would let a
        # 1.249 look like 1.25.
        "mae_raw": mae_raw,
        "mae": round(mae_raw, 2),
        "median": round(float(np.median(errors)), 2),
        "p90": round(float(np.percentile(errors, 90)), 1),
        "under_10px_share": round(float((errors <= 10).mean()), 3),
        "tail_gt30_share": round(float((errors > 30).mean()), 3),
        "_errors": errors,
    }


def evaluate(checkpoint: str, dataset_dir: str, per_stage: int = DEFAULT_PER_STAGE, stride: int = 1) -> dict:
    """Backward-compatible wrapper: stage-sample with the dataset's own manifest.

    ``stride`` is accepted and ignored (the v1.0.3 tool never used it either); new
    callers should use :func:`evaluate_checkpoint`, which exposes the mode.
    """
    return evaluate_checkpoint(checkpoint, dataset_dir, per_stage=per_stage, mode=DEFAULT_MODE)


# --------------------------------------------------------------------------- #
# A-03 ratio against a previous run of the same checkpoint on A
# --------------------------------------------------------------------------- #
def load_baseline(path: Path | str) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        raise v104_error("DATA", f"baseline output not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise v104_error("DATA", f"baseline output is unreadable: {path} ({exc})") from exc
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise v104_error("DATA", f"baseline output has no rows: {path}")
    return [row for row in rows if isinstance(row, dict)]


def a03_ratio(rows: list[dict], baseline_rows: list[dict]) -> list[dict]:
    """``B_mae_raw / A_mae_raw`` for every row that has a matching baseline row.

    The checkpoint bytes must match (A-03 is "the same final policy.pt on two
    datasets"), the sampling mode must be ``stage-sample`` and the baseline must
    come from a different dataset - comparing full-val with stage-sample is the
    specific mistake the design doc calls out.
    """
    verdicts: list[dict] = []
    for row in rows:
        if row.get("sampling_mode") != "stage-sample":
            verdicts.append(
                {
                    "checkpoint_sha256": row.get("checkpoint_sha256"),
                    "status": "blocked",
                    "reason": f"A-03 requires stage-sample, got {row.get('sampling_mode')!r}",
                }
            )
            continue
        matches = [
            base
            for base in baseline_rows
            if base.get("checkpoint_sha256") == row.get("checkpoint_sha256")
            and base.get("dataset") != row.get("dataset")
            and base.get("sampling_mode") == "stage-sample"
        ]
        if not matches:
            verdicts.append(
                {
                    "checkpoint_sha256": row.get("checkpoint_sha256"),
                    "status": "blocked",
                    "reason": (
                        "no baseline row with the same checkpoint_sha256 on another dataset "
                        "with sampling_mode=stage-sample"
                    ),
                }
            )
            continue
        base = matches[0]
        base_mae = base.get("mae_raw")
        if not isinstance(base_mae, (int, float)) or base_mae <= 0:
            verdicts.append(
                {
                    "checkpoint_sha256": row.get("checkpoint_sha256"),
                    "status": "blocked",
                    "reason": f"baseline mae_raw is unusable: {base_mae!r}",
                }
            )
            continue
        ratio = float(row["mae_raw"]) / float(base_mae)
        verdicts.append(
            {
                "checkpoint_sha256": row.get("checkpoint_sha256"),
                "rule": A03_RULE,
                "a_dataset": base.get("dataset"),
                "a_mae_raw": float(base_mae),
                "a_indices_sha256": base.get("selected_index_sha256"),
                "a_frames": base.get("frames"),
                "b_dataset": row.get("dataset"),
                "b_mae_raw": float(row["mae_raw"]),
                "b_indices_sha256": row.get("selected_index_sha256"),
                "b_frames": row.get("frames"),
                "ratio": ratio,
                "threshold": A03_RATIO,
                "pass": bool(ratio <= A03_RATIO),
            }
        )
    return verdicts


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_row(row: dict) -> None:
    print(
        f"[{row['label']}] arch={row['arch']} depth={row['depth']} "
        f"history={row['history_contract']}"
        f"{' (legacy)' if row['legacy_history'] else ''} "
        f"mode={row['sampling_mode']} per_stage={row['per_stage_limit']} "
        f"frames={row['frames']}/{row['val_frames']}",
        flush=True,
    )
    print(
        f"    mae_raw={row['mae_raw']:.6f} mae={row['mae']} median={row['median']} "
        f"p90={row['p90']} <=10px={row['under_10px_share']:.1%} "
        f"tail>30px={row['tail_gt30_share']:.1%}",
        flush=True,
    )
    print(
        f"    manifest={row['manifest']} sha={row['manifest_sha256'][:16]}... "
        f"val_sha={row['val_index_sha256'][:16]}... "
        f"selected_sha={row['selected_index_sha256'][:16]}... "
        f"checkpoint_sha={row['checkpoint_sha256'][:16]}...",
        flush=True,
    )


def _print_table(rows: list[dict]) -> None:
    stages = sorted({stage for row in rows for stage in row["per_stage"]}, key=lambda value: int(value))
    header = "| stage | " + " | ".join(str(row["label"]) for row in rows) + " |"
    print(header)
    print("| --- |" + " --- |" * len(rows))
    for stage in stages:
        cells = []
        for row in rows:
            node = row["per_stage"].get(stage)
            cells.append("n/a" if node is None else f"{node['mae']:.2f} ({node['frames']})")
        print(f"| {stage} | " + " | ".join(cells) + " |")
    print(
        "| frames | " + " | ".join(str(row["frames"]) for row in rows) + " |",
    )


def _parse_checkpoints(specs: list[str], parser: argparse.ArgumentParser) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for spec in specs:
        if "=" not in spec:
            parser.error(f"--checkpoint expects LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        if not label or not path:
            parser.error(f"--checkpoint expects LABEL=PATH, got {spec!r}")
        pairs.append((label, path))
    return pairs


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    checkpoints = _parse_checkpoints(args.checkpoint, parser)
    if int(args.per_stage) < 1:
        raise v104_error("CONFIG", f"--per-stage must be >= 1, got {args.per_stage}")
    baseline_rows = load_baseline(args.baseline) if args.baseline else []
    rows = []
    for label, path in checkpoints:
        row = evaluate_checkpoint(
            path,
            args.dataset,
            manifest_path=args.manifest or None,
            mode=args.mode,
            per_stage=args.per_stage,
            batch=args.batch,
            label=label,
        )
        _print_row(row)
        rows.append(row)
    _print_table(rows)
    verdicts = a03_ratio(rows, baseline_rows) if baseline_rows else []
    for verdict in verdicts:
        if verdict.get("status") == "blocked":
            print(f"A-03 blocked: {verdict['reason']}", flush=True)
        else:
            print(
                f"A-03 {'pass' if verdict['pass'] else 'fail'} "
                f"ratio={verdict['ratio']:.6f} (threshold <= {verdict['threshold']}) "
                f"a_mae_raw={verdict['a_mae_raw']:.6f} b_mae_raw={verdict['b_mae_raw']:.6f}",
                flush=True,
            )
    if args.out:
        payload = []
        for row in rows:
            clean = {key: value for key, value in row.items() if key != "_errors"}
            payload.append(clean)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"written: {args.out}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare state checkpoints on a verified manifest sample (v1.0.4)"
    )
    parser.add_argument("--dataset", required=True, help="dataset directory (vision shards)")
    parser.add_argument(
        "--manifest",
        default="",
        help="manifest JSON; default = <dataset>/manifest.json (B must pass the S3-1 evaluation manifest)",
    )
    parser.add_argument(
        "--mode",
        choices=list(SAMPLING_MODES),
        default=DEFAULT_MODE,
        help="full-val = every validation index (A-01); stage-sample = first --per-stage per stage (A-03)",
    )
    parser.add_argument("--per-stage", type=int, default=DEFAULT_PER_STAGE)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--out", default="", help="write the rows (with mae_raw) as JSON")
    parser.add_argument(
        "--baseline",
        default="",
        help="earlier output JSON for the A side; enables the A-03 ratio against the same checkpoint",
    )
    args = parser.parse_args(argv)
    try:
        return _run(args, parser)
    except ValueError as exc:
        code = v104_error_code(exc)
        if code is None:
            raise
        print(str(exc), file=sys.stderr)
        return 2
    except (OSError, ManifestError) as exc:
        print(f"v104:DATA: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
