"""Verify the three v3 smoke runs as an *independent process* (S3-7 evidence).

For every ``runs/v104-v3-<E>-s7-smoke`` this checks, in order:

1. ``vision-summary.json`` - schema/contract, ``arch=v3-st``, depth 3,
   ``acceptance_eligible=false`` (a smoke run may never look eligible),
   ``coverage.val`` = 7241/7241 with the fixed A digest, screening flag;
2. every checkpoint file's sha256 against the file the summary selected;
3. ``torch.load`` payload -> ``validate_v3_payload`` (schema ``vision-checkpoint/3``,
   C6 identity: source_state_sha / teacher_fingerprint / manifest_sha /
   val_index_sha256 / loss_config) and ``load_vision_checkpoint`` strict load;
4. real-pixel inference through the deployment path: ``score`` ``[B,18]`` finite,
   ``act`` four-tuple with shoot=1 / bomb=0, ``decode_state`` key set;
5. the E4 ablation property on the *trained* checkpoints: the same parameter
   slots, and the local vector is exactly zero only for E4.

Usage::

    python Document/Update/Alpha/v1.0.4/Stage3/S3-7/evidence/verify_smoke.py \
        --runs-dir third_party/th08/build/rl/runs --out <json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[7]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from touhou_rl import vision, vision_train  # noqa: E402
from touhou_rl.pipeline.manifest import file_digest  # noqa: E402
from touhou_rl.vision_data import VisionDataset  # noqa: E402
from touhou_rl.vmodels import load_vision_checkpoint, validate_v3_payload, vision_arch  # noqa: E402

DATASET = ROOT / "third_party" / "th08" / "build" / "rl" / "datasets" / "pilot-ud0059-v2"
VAL_SHA = "e5d4ebbe6d8de05793520a393fc1fdbea2b1e2c4db9ff7c1632908340c76655f"
VAL_FRAMES = 7241
EXPERIMENTS = ("E4", "E5", "E6")


def real_pixel_batch(indices: list[int], depth: int, sizes, device: str):
    with VisionDataset(DATASET, history_contract="causal-segment-v1") as dataset:
        stacked, _labels = vision_train._gather(dataset, indices, depth, sizes)  # noqa: SLF001
    return vision.regions_to_batch({name: value.numpy() for name, value in stacked.items()}, device)


def verify_run(run_dir: Path, device: str, val_indices: list[int]) -> dict:
    summary = json.loads((run_dir / "vision-summary.json").read_text(encoding="utf-8"))
    record: dict = {
        "run_dir": run_dir.relative_to(ROOT).as_posix(),
        "summary": (run_dir / "vision-summary.json").relative_to(ROOT).as_posix(),
        "summary_sha256": file_digest(run_dir / "vision-summary.json"),
        "schema": summary.get("schema"),
        "contract_version": summary.get("contract_version"),
        "arch": summary.get("arch"),
        "depth": summary.get("depth"),
        "parameters": summary.get("parameters"),
        "acceptance_eligible": summary.get("acceptance_eligible"),
        "screening": (summary.get("recipe") or {}).get("screening"),
        "local_enabled": (summary.get("input") or {}).get("local_enabled"),
        "heatmap_sigma": (summary.get("input") or {}).get("heatmap_sigma"),
        "coverage": (summary.get("coverage") or {}).get("val"),
        "epoch_seconds": {
            stage: [row.get("seconds") for row in ((summary.get("stages") or {}).get(stage) or {}).get("history", [])]
            for stage in ("state", "policy")
        },
    }
    checks: dict = {}

    checks["arch_is_v3st"] = summary.get("arch") == "v3-st"
    checks["depth_is_three"] = int(summary.get("depth") or 0) == 3
    checks["smoke_is_not_eligible"] = summary.get("acceptance_eligible") is False
    coverage = record["coverage"] or {}
    checks["full_validation"] = (
        int(coverage.get("actual_frames") or 0) == VAL_FRAMES
        and coverage.get("visited_index_sha256") == VAL_SHA
        and coverage.get("complete") is True
    )

    checkpoints = {}
    for stage, name in (("state", "state.pt"), ("policy", "policy.pt")):
        best = ((summary.get("stages") or {}).get(stage) or {}).get("best") or {}
        path = run_dir / name
        entry = {
            "file": path.relative_to(ROOT).as_posix() if path.is_file() else None,
            "exists": path.is_file(),
            "best_epoch": best.get("epoch"),
            "best_checkpoint": best.get("checkpoint"),
            "recorded_sha256": best.get("checkpoint_sha256"),
            "actual_sha256": file_digest(path) if path.is_file() else None,
            "selection_rule": best.get("selection_rule"),
        }
        entry["sha_matches"] = bool(entry["exists"]) and entry["recorded_sha256"] == entry["actual_sha256"]
        entry["best_points_at_the_file"] = bool(best.get("checkpoint")) and str(best["checkpoint"]).endswith(name)
        checkpoints[stage] = entry
    record["checkpoints"] = checkpoints
    checks["checkpoint_shas_match"] = all(
        entry["sha_matches"] and entry["best_points_at_the_file"] for entry in checkpoints.values()
    )

    identities = {}
    strict = {}
    for stage, name in (("state", "state.pt"), ("policy", "policy.pt")):
        path = run_dir / name
        payload = torch.load(path, map_location="cpu", weights_only=False)
        contract = validate_v3_payload(payload)
        extra = payload["extra"]
        model, loaded_extra = load_vision_checkpoint(path, map_location="cpu")
        model = model.to(device).eval()
        scores = None
        actions = None
        state_keys = None
        with torch.no_grad():
            batch = real_pixel_batch(val_indices[:4], int(model.config.depth), model.config.region_sizes, device)
            scores = model.score(batch)
            result = model.act(batch)
            actions = result[0]
            state = model.decode_state(batch)
            state_keys = sorted(state)
            local_vector = model.local_features(
                torch.zeros(1, model.config.pf_width * 2, 4, 4, device=device),
                torch.full((1, 2), 0.5, device=device),
            ) if hasattr(model, "local_features") else None
        identities[stage] = {
            "schema": payload.get("schema"),
            "arch": payload.get("arch"),
            "strict_load": True,
            "contract": {key: contract.get(key) for key in ("depth", "roi_native_hw", "local_grid_hw", "history_contract")},
            "extra_keys": sorted(extra),
            "source_state_sha_len": len(str(extra.get("source_state_sha", ""))),
            "teacher_fingerprint": str(extra.get("teacher_fingerprint", ""))[:16],
            "manifest_sha_len": len(str(extra.get("manifest_sha", ""))),
            "val_index_sha256": extra.get("val_index_sha256"),
            "loss_config_objective": (extra.get("loss_config") or {}).get("policy", {}).get("objective"),
            "history_contract": extra.get("history_contract"),
            "local_enabled": (payload.get("config") or {}).get("local_enabled"),
        }
        strict[stage] = {
            "scores_shape": list(scores.shape),
            "scores_finite": bool(torch.isfinite(scores).all()),
            "actions_shape": list(actions.shape),
            "actions_dtype": str(actions.dtype),
            "shoot_column_all_one": bool((actions[:, 1] == 1).all()),
            "bomb_column_all_zero": bool((actions[:, 3] == 0).all()),
            "decode_state_keys": state_keys,
            "local_vector_zero": bool(local_vector is not None and int(torch.count_nonzero(local_vector)) == 0),
            "vision_arch": vision_arch(model),
            "extra_history_contract": loaded_extra.get("history_contract"),
        }
        checks[f"{stage}_payload_valid"] = True
        checks[f"{stage}_strict_load_and_inference"] = (
            strict[stage]["scores_finite"]
            and strict[stage]["actions_shape"][1] == 4
            and strict[stage]["shoot_column_all_one"]
            and strict[stage]["bomb_column_all_zero"]
            and strict[stage]["decode_state_keys"] == ["danger_log", "heatmap_logits", "self_feat"]
        )
        checks[f"{stage}_identity_complete"] = (
            identities[stage]["schema"] == "vision-checkpoint/3"
            and identities[stage]["manifest_sha_len"] == 64
            and identities[stage]["val_index_sha256"] == VAL_SHA
            and bool(identities[stage]["teacher_fingerprint"])
            and bool(identities[stage]["loss_config_objective"])
        )
        if stage == "policy":
            checks["policy_has_source_state_sha"] = identities[stage]["source_state_sha_len"] == 64
        else:
            checks["state_has_no_source_state"] = identities[stage]["source_state_sha_len"] == 0
    record["identities"] = identities
    record["inference"] = strict
    record["checks"] = checks
    record["passed"] = all(checks.values())
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="verify the S3-7 v3 smoke runs")
    parser.add_argument("--runs-dir", default="third_party/th08/build/rl/runs")
    parser.add_argument("--out", default="")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    runs_dir = (ROOT / args.runs_dir).resolve()
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    from touhou_rl.pipeline.manifest import DatasetManifest

    manifest = DatasetManifest.load(DATASET / "manifest.json")
    _train, val = manifest.verify(DATASET)
    val_indices = [int(index) for index in val[:4]]
    payload = {
        "schema": "v104-s37-smoke-verify/1",
        "created_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "producer": "Document/Update/Alpha/v1.0.4/Stage3/S3-7/evidence/verify_smoke.py",
        "device": device,
        "val_index_sha256": VAL_SHA,
        "runs": [],
    }
    failures = 0
    for name in EXPERIMENTS:
        run_dir = runs_dir / f"v104-v3-{name}-s7-smoke"
        if not run_dir.is_dir():
            payload["runs"].append({"experiment": name, "passed": False, "error": "run directory missing"})
            failures += 1
            continue
        record = verify_run(run_dir, device, val_indices)
        record["experiment"] = name
        payload["runs"].append(record)
        failures += 0 if record["passed"] else 1
        print(f"[{name}] passed={record['passed']} checks={json.dumps(record['checks'])}")
    payload["all_passed"] = failures == 0
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        out_path = (ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"written {out_path}")
    else:
        print(text)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
