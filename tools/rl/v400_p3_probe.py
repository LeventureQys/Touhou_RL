"""``v400_p3_probe`` -- T2a perception probe: collect, train, decide (EXP-P3-010).

Subcommands, each with a ``--dry-run`` that touches nothing:

    collect   replay D0's recorded per-frame masks at native resolution
    train     train the four pre-registered input heads (resumable per input)
    locator   train + evaluate the pixel ship locator (the gate on input D)
    dloc      score input D with locator-placed windows (``D+loc``)
    merge     assemble ``train-all.json`` from per-input results + the decision
    report    apply the pre-registered selection rule to a finished summary

The tool never writes to ``.tmp_v400_d0`` (D0 is read-only input), never edits
obs-v1/obs-v2, and never starts the game unless a subcommand asks for it.

WSL (the engine is a Linux ELF, so the collector must run there)::

    cd /mnt/d/WorkShop/Touhou_RL && env -u DISPLAY xvfb-run -a \
        /root/v400-venv/bin/python tools/rl/v400_p3_probe.py collect \
        --seeds 1-4 --binary third_party/th08-rl/build/rl_hazard2/th08-modern \
        --shard-root .tmp_v400_d0/d0/shards --data-template /root/v400-data/game \
        --data-root /root/v400-data/p3diag --out-dir /root/v400-data/p3diag

Training runs on either side (the GPU is visible from both); the collector is
WSL-only.  A run that is interrupted after some inputs keeps ``train-A.json`` /
``train-B.json``; ``train --inputs C D`` then finishes the rest and ``merge``
combines them, so nothing has to be retrained to resume.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):  # allow `python tools/rl/v400_p3_probe.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_FAILED = 3
EXIT_STOP_LOSS = 4

#: D0's engine, pinned by EXP-P1-014/P1-018.  A different binary would move the
#: pixels, so the collector checks the digest before it replays anything.
DEFAULT_ENGINE_SHA256 = "f172399f61b75baa8e62be46041ef756a710759f439e7fb07111f4747c78264e"
DEFAULT_BINARY = "third_party/th08-rl/build/rl_hazard2/th08-modern"
DEFAULT_SHARDS = ".tmp_v400_d0/d0/shards"
DEFAULT_DATA_TEMPLATE = "/root/v400-data/game"
DEFAULT_DATA_ROOT = "/root/v400-data/p3diag"
DEFAULT_OUT = "/root/v400-data/p3diag"
REPORT_DIR = ".tmp_v400_p3diag"


def _log(message: str) -> None:
    print(message, flush=True)


def parse_seeds(text: str) -> list[int]:
    """``"1-4,7"`` -> ``[1, 2, 3, 4, 7]``."""

    seeds: list[int] = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            low, high = chunk.split("-", 1)
            seeds.extend(range(int(low), int(high) + 1))
        else:
            seeds.append(int(chunk))
    if not seeds:
        raise argparse.ArgumentTypeError(f"no seeds in {text!r}")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError(f"duplicate seeds in {text!r}")
    return seeds


def shard_dir_for(root: Path, seed: int, *, stage: int | None = None) -> Path:
    """D0's shard layout: ``<root>/dual-s{stage}d{difficulty}seed{seed}``."""

    candidates = sorted(Path(root).glob(f"dual-*seed{seed}"))
    if not candidates:
        raise FileNotFoundError(f"no D0 shard for seed {seed} under {root}")
    if stage is not None:
        narrowed = [path for path in candidates if f"dual-s{int(stage)}d" in path.name]
        if narrowed:
            candidates = narrowed
    if len(candidates) > 1:
        raise RuntimeError(
            f"{len(candidates)} D0 shards match seed {seed}: {[p.name for p in candidates]}")
    return candidates[0]


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------
def cmd_collect(args: argparse.Namespace) -> int:
    from touhou_rl.v400.student.probe.collect import (
        CollectError, HashMismatch, collect_seed)

    seeds = args.seeds
    binary = Path(args.binary)
    out_dir = Path(args.out_dir)
    if args.dry_run:
        _log(json.dumps({
            "dry_run": True,
            "seeds": seeds,
            "binary": str(binary),
            "binary_exists": binary.is_file(),
            "data_template": str(args.data_template),
            "data_template_exists": Path(args.data_template).is_dir(),
            "out_dir": str(out_dir),
            "shards": [str(shard_dir_for(Path(args.shard_root), seed, stage=args.stage))
                       for seed in seeds],
            "expected_sha256": args.expected_sha,
            "concurrency": int(args.concurrency),
        }, indent=2, ensure_ascii=False))
        return EXIT_OK

    reports = []
    failures = []
    for seed in seeds:
        shard_dir = shard_dir_for(Path(args.shard_root), seed, stage=args.stage)
        data_dir = Path(args.data_root) / f"env-s{seed}"
        try:
            report = collect_seed(
                seed=seed,
                shard_dir=shard_dir,
                binary=binary,
                data_template=Path(args.data_template),
                data_dir=data_dir,
                out_dir=out_dir,
                expected_sha256=args.expected_sha,
                limit=args.limit,
                timeout=float(args.timeout),
                log=_log,
            )
        except HashMismatch as exc:
            failures.append({"seed": seed, "kind": "hash_mismatch",
                             "detail": str(exc)[:4000]})
            _log(f"[probe] seed {seed}: HASH MISMATCH, seed not published")
            continue
        except (CollectError, FileNotFoundError, RuntimeError) as exc:
            failures.append({"seed": seed, "kind": type(exc).__name__,
                             "detail": str(exc)[:2000]})
            _log(f"[probe] seed {seed}: {type(exc).__name__}: {exc}")
            continue
        reports.append(report)
        _log(f"[probe] seed {seed}: {report['decisions_filled']} decisions, "
             f"{report['wall_s']:.1f}s, "
             f"{report['frames_per_s']:.1f} frames/s, "
             f"positives {report['positives_total']}")
    summary = {
        "kind": "v400-p3diag-collect-batch-1",
        "seeds": [int(seed) for seed in seeds],
        "collected": reports,
        "failures": failures,
        "decisions_total": sum(int(row["decisions_filled"]) for row in reports),
        "status": "ok" if not failures else "partial",
    }
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"collect-{seeds[0]}-{seeds[-1]}.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    _log(json.dumps({"collected": len(reports), "failures": len(failures),
                     "decisions": summary["decisions_total"], "report": str(path)},
                    ensure_ascii=False))
    return EXIT_OK if not failures else EXIT_FAILED


# ---------------------------------------------------------------------------
# shared helpers for the resumable pipeline
# ---------------------------------------------------------------------------
def _load_first(directories: Sequence[Path], name: str) -> dict[str, Any] | None:
    """First ``name`` found in ``directories`` (search order preserved)."""

    for directory in directories:
        candidate = Path(directory) / name
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return None


def _micro_summary(payload: dict[str, Any]) -> dict[str, Any]:
    values = [float(run["test"]["micro"]["f1"]) for run in payload["runs"]]
    mean = sum(values) / len(values)
    if len(values) > 1:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        sd = variance ** 0.5
    else:
        sd = 0.0
    return {
        "mean": float(mean),
        "sd": float(sd),
        "min": float(min(values)),
        "max": float(max(values)),
        "range": float(max(values) - min(values)),
        "values": [round(float(value), 6) for value in values],
        "precision": [round(float(run["test"]["micro"]["precision"]), 6)
                      for run in payload["runs"]],
        "recall": [round(float(run["test"]["micro"]["recall"]), 6)
                   for run in payload["runs"]],
    }


def _run_locator(*, run_dir: Path, report_dir: Path, models_dir: Path,
                 seeds: Sequence[int], epochs: int, batch: int, lr: float,
                 limit: int | None, device: str | None) -> dict[str, Any]:
    """Train + evaluate one locator per training seed; the seed-0 one is primary."""

    import torch

    from touhou_rl.v400.student.probe.train import (
        TEST_SPLIT, TRAIN_SPLIT, evaluate_locator, train_locator, write_json)

    report_dir = Path(report_dir)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    per_seed: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        trained = train_locator(train_dir=Path(run_dir), seeds=TRAIN_SPLIT,
                                train_seed=int(seed), epochs=int(epochs),
                                batch=int(batch), lr=float(lr), limit=limit,
                                device=device, log=_log)
        evaluation = evaluate_locator(trained["model"], test_dir=Path(run_dir),
                                      seeds=TEST_SPLIT, device=device)
        evaluation["locator_seed"] = int(seed)
        evaluation["train"] = trained["report"]
        write_json(report_dir / f"locator-{int(seed)}.json", evaluation)
        target = models_dir / f"locator_{int(seed)}.pt"
        torch.save({"state_dict": trained["model"].state_dict(),
                    "locator_seed": int(seed), "report": trained["report"]}, target)
        for shard in trained["shards"]:
            shard.close()
        _log(f"[probe] locator {seed}: holdout median {evaluation['median_px']:.3f} px, "
             f"p90 {evaluation['p90_px']:.3f} px, share<=4px "
             f"{evaluation['share_within_4px']:.3f} (gate {evaluation['gate_px']} px, "
             f"passed={evaluation['gate_passed']})")
        per_seed[int(seed)] = evaluation
    primary = int(seeds[0])
    summary = {
        "kind": "v400-p3diag-locator-1",
        "primary_seed": primary,
        "median_px": per_seed[primary]["median_px"],
        "gate_px": per_seed[primary]["gate_px"],
        "gate_passed": per_seed[primary]["gate_passed"],
        "all_seeds_pass": all(bool(ev["gate_passed"]) for ev in per_seed.values()),
        "samples": per_seed[primary]["samples"],
        "test_seeds": per_seed[primary]["test_seeds"],
        "per_seed": {
            str(seed): {key: ev[key] for key in (
                "median_px", "mean_px", "p90_px", "max_px", "share_within_4px",
                "samples", "gate_passed", "per_seed_median_px", "train")}
            for seed, ev in per_seed.items()
        },
        "checkpoints": {str(seed): str(models_dir / f"locator_{seed}.pt")
                        for seed in per_seed},
    }
    write_json(report_dir / "locator.json", summary)
    _log(f"[probe] locator gate: median {summary['median_px']:.3f} px -> "
         f"{'PASS' if summary['gate_passed'] else 'FAIL'}"
         f"{'' if summary['all_seeds_pass'] else ' (a non-primary seed failed)'}")
    return summary


def _run_dloc(*, run_dir: Path, report_dir: Path, models_dir: Path,
              locator_models: Path, seeds: Sequence[int],
              device: str | None) -> dict[str, Any]:
    """Score input D with locator-placed windows, pairing seeds 1:1."""

    import torch

    from touhou_rl.v400.student.probe.net import PlayerLocator, ProbeModel
    from touhou_rl.v400.student.probe.train import (
        TEST_SPLIT, TrainError, _device, evaluate_probe, locator_anchor_cells,
        write_json)

    dev = _device(device)
    payload = {"kind": "v400-p3diag-dloc-1", "input": "D", "anchors": "locator",
               "pairs": []}
    for train_seed in seeds:
        probe_path = Path(models_dir) / f"probe_D_{int(train_seed)}.pt"
        locator_path = Path(locator_models) / f"locator_{int(train_seed)}.pt"
        if not locator_path.is_file():
            locator_path = Path(locator_models) / f"locator_{int(seeds[0])}.pt"
        if not probe_path.is_file():
            raise TrainError(f"missing D checkpoint {probe_path}; train input D first")
        if not locator_path.is_file():
            raise TrainError(f"missing locator checkpoint {locator_path}")
        probe_checkpoint = torch.load(probe_path, map_location=dev,
                                      weights_only=False)
        model = ProbeModel("D").to(dev)
        model.load_state_dict(probe_checkpoint["state_dict"])
        locator_checkpoint = torch.load(locator_path, map_location=dev,
                                        weights_only=False)
        locator = PlayerLocator().to(dev)
        locator.load_state_dict(locator_checkpoint["state_dict"])
        cells = locator_anchor_cells(locator, test_dir=Path(run_dir),
                                     seeds=TEST_SPLIT, device=device)
        evaluation = evaluate_probe(model, "D", test_dir=Path(run_dir),
                                    seeds=TEST_SPLIT, device=device,
                                    anchor_cells=cells)
        report_path = Path(report_dir) / f"locator-{int(locator_checkpoint.get('locator_seed', 0))}.json"
        locator_report = (json.loads(report_path.read_text(encoding="utf-8"))
                          if report_path.is_file() else None)
        pair = {
            "train_seed": int(train_seed),
            "locator_seed": int(locator_checkpoint.get("locator_seed",
                                                       int(seeds[0]))),
            "locator_median_px": (None if locator_report is None
                                  else locator_report["median_px"]),
            "test": evaluation,
        }
        payload["pairs"].append(pair)
        _log(f"[probe] D+loc train_seed {train_seed} locator {pair['locator_seed']}: "
             f"F1 {evaluation['micro']['f1']:.4f} "
             f"(P {evaluation['micro']['precision']:.4f} "
             f"R {evaluation['micro']['recall']:.4f})")
    values = [float(pair["test"]["micro"]["f1"]) for pair in payload["pairs"]]
    mean = sum(values) / len(values)
    sd = ((sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5
          if len(values) > 1 else 0.0)
    payload["micro_f1"] = {"mean": round(mean, 6), "sd": round(sd, 6),
                           "values": [round(value, 6) for value in values]}
    write_json(Path(report_dir) / "train-D-loc.json", payload)
    return payload


def _summary_payload(results: dict[str, dict[str, Any]],
                     locator: dict[str, Any] | None,
                     dloc: dict[str, Any] | None, *, run_dir: Path,
                     complete: bool, baselines: bool = True) -> dict[str, Any]:
    from touhou_rl.v400.student.probe.train import (
        TEST_SPLIT, naive_baselines, select_input)

    averaged = {name: {"micro": {"f1": payload["micro_f1"]["mean"]}}
                for name, payload in results.items()}
    decision = select_input(averaged, locator=locator) if complete else None
    baseline = naive_baselines(test_dir=run_dir, seeds=TEST_SPLIT) if baselines else None
    return {
        "kind": "v400-p3diag-train-1",
        "complete": bool(complete),
        "run_dir": str(run_dir),
        "results": results,
        "micro_f1": {name: _micro_summary(payload)
                     for name, payload in results.items()},
        "locator": locator,
        "dloc": dloc,
        "decision": decision,
        "baselines": baseline,
    }


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def cmd_train(args: argparse.Namespace) -> int:
    from touhou_rl.v400.student.probe.train import (
        TEST_SPLIT, TRAIN_SEEDS, TRAIN_SPLIT, evaluate_probe, train_probe,
        write_json)

    run_dir = Path(args.run_dir)
    results: dict[str, dict[str, Any]] = {}
    if args.dry_run:
        from touhou_rl.v400.student.probe.train import SeedShard

        plan = {}
        for name in args.inputs:
            plan[name] = []
            for seed in TRAIN_SEEDS:
                plan[name].append({"train_seed": seed})
        missing = [
            str(run_dir / f"native_s{seed}.npy")
            for seed in TRAIN_SPLIT + TEST_SPLIT
            if not (run_dir / f"native_s{seed}.npy").is_file()
        ]
        _log(json.dumps({
            "dry_run": True, "run_dir": str(run_dir), "plan": plan,
            "epochs": args.epochs, "seeds": {
                "train": list(TRAIN_SPLIT), "test": list(TEST_SPLIT)},
            "missing_native": missing, "device": args.device,
        }, indent=2, ensure_ascii=False))
        return EXIT_OK

    for name in args.inputs:
        runs = []
        for train_seed in (args.train_seeds if args.train_seeds else TRAIN_SEEDS):
            trained = train_probe(
                name, train_dir=run_dir, seeds=TRAIN_SPLIT,
                train_seed=int(train_seed), epochs=int(args.epochs),
                batch=int(args.batch), lr=float(args.lr), limit=args.limit,
                device=args.device, log=_log)
            evaluation = evaluate_probe(
                trained["model"], name, test_dir=run_dir,
                seeds=TEST_SPLIT, device=args.device)
            runs.append({"train": trained["report"], "test": evaluation})
            _log(f"[probe] input {name} train_seed {train_seed}: "
                 f"F1 {evaluation['micro']['f1']:.4f} "
                 f"(P {evaluation['micro']['precision']:.4f} "
                 f"R {evaluation['micro']['recall']:.4f})")
            if args.save_models:
                import torch

                target = Path(args.save_models) / f"probe_{name}_{train_seed}.pt"
                target.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": trained["model"].state_dict(),
                            "input": name, "train_seed": int(train_seed),
                            "report": trained["report"]}, target)
            for shard in trained["shards"]:
                shard.close()
        f1_values = [run["test"]["micro"]["f1"] for run in runs]
        results[name] = {
            "input": name,
            "runs": runs,
            "micro_f1": {
                "mean": float(sum(f1_values) / len(f1_values)),
                "min": float(min(f1_values)),
                "max": float(max(f1_values)),
                "values": [round(float(value), 6) for value in f1_values],
            },
        }
        write_json(Path(args.report_dir) / f"train-{name}.json", results[name])

    locator: dict[str, Any] | None = None
    if args.locator:
        models_dir = Path(args.save_models) if args.save_models else run_dir / "models"
        seeds = args.locator_seeds if args.locator_seeds else [
            int(args.train_seeds[0]) if args.train_seeds else 0]
        locator = _run_locator(run_dir=run_dir, report_dir=Path(args.report_dir),
                               models_dir=models_dir, seeds=seeds,
                               epochs=int(args.epochs), batch=int(args.batch),
                               lr=float(args.lr), limit=args.limit,
                               device=args.device)

    dloc: dict[str, Any] | None = None
    if "D" in args.inputs and args.dloc and args.save_models:
        locator_models = (Path(args.locator_models) if args.locator_models
                          else Path(args.save_models))
        try:
            dloc = _run_dloc(run_dir=run_dir, report_dir=Path(args.report_dir),
                             models_dir=Path(args.save_models),
                             locator_models=locator_models,
                             seeds=args.train_seeds if args.train_seeds else TRAIN_SEEDS,
                             device=args.device)
        except Exception as exc:  # noqa: BLE001 - D+loc is an extra, never a gate
            _log(f"[probe] D+loc skipped: {type(exc).__name__}: {exc}")

    complete = sorted(str(name) for name in args.inputs) == ["A", "B", "C", "D"]
    payload = _summary_payload(results, locator, dloc, run_dir=run_dir,
                               complete=complete)
    if complete:
        path = Path(args.report_dir) / "train-all.json"
    else:
        path = Path(args.report_dir) / (
            "train-partial-" + "".join(sorted(str(name) for name in args.inputs))
            + ".json")
    write_json(path, payload)
    _log(json.dumps({"complete": complete, "decision": payload["decision"],
                     "baselines": payload["baselines"], "report": str(path)},
                    indent=2, ensure_ascii=False))
    if payload["decision"] and payload["decision"]["stop_loss"]:
        _log("[probe] STOP LOSS: every input is below F1 0.90; ask the owner")
        return EXIT_STOP_LOSS
    return EXIT_OK


# ---------------------------------------------------------------------------
# locator / dloc / merge
# ---------------------------------------------------------------------------
def cmd_locator(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    models_dir = Path(args.save_models) if args.save_models else run_dir / "models"
    if args.dry_run:
        _log(json.dumps({
            "dry_run": True, "run_dir": str(run_dir),
            "locator_seeds": list(args.locator_seeds),
            "models_dir": str(models_dir),
            "epochs": int(args.epochs), "batch": int(args.batch),
            "lr": float(args.lr), "device": args.device,
        }, indent=2, ensure_ascii=False))
        return EXIT_OK
    summary = _run_locator(run_dir=run_dir, report_dir=Path(args.report_dir),
                           models_dir=models_dir, seeds=args.locator_seeds,
                           epochs=int(args.epochs), batch=int(args.batch),
                           lr=float(args.lr), limit=args.limit,
                           device=args.device)
    return EXIT_OK if summary["gate_passed"] else EXIT_FAILED


def cmd_dloc(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    models_dir = Path(args.models) if args.models else run_dir / "models"
    locator_models = (Path(args.locator_models) if args.locator_models
                      else models_dir)
    if args.dry_run:
        _log(json.dumps({
            "dry_run": True, "run_dir": str(run_dir), "models_dir": str(models_dir),
            "locator_models": str(locator_models),
            "train_seeds": list(args.train_seeds), "device": args.device,
            "checkpoints": [str(models_dir / f"probe_D_{seed}.pt")
                            for seed in args.train_seeds],
        }, indent=2, ensure_ascii=False))
        return EXIT_OK
    payload = _run_dloc(run_dir=run_dir, report_dir=Path(args.report_dir),
                        models_dir=models_dir, locator_models=locator_models,
                        seeds=args.train_seeds, device=args.device)
    _log(json.dumps({"D+loc": payload["micro_f1"]}, indent=2, ensure_ascii=False))
    return EXIT_OK


def cmd_merge(args: argparse.Namespace) -> int:
    from touhou_rl.v400.student.probe.train import write_json

    search = [Path(directory) for directory in
              (args.search_dir if args.search_dir else [args.report_dir])]
    results: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for name in args.inputs:
        payload = _load_first(search, f"train-{name}.json")
        if payload is None:
            missing.append(name)
        else:
            results[name] = payload
    if missing:
        _log(f"[probe] merge needs train-<X>.json for {missing}; searched "
             f"{[str(d) for d in search]}")
        return EXIT_FAILED
    locator = _load_first(search, "locator.json")
    dloc = _load_first(search, "train-D-loc.json")
    if locator is None:
        _log("[probe] no locator.json: D will be disqualified "
             "('locator error not measured')")
    else:
        _log(f"[probe] locator median {locator['median_px']:.3f} px, "
             f"gate_passed={locator['gate_passed']}")
    if dloc is not None:
        _log(f"[probe] D+loc F1 mean {dloc['micro_f1']['mean']:.4f} "
             f"(n={len(dloc['pairs'])})")
    complete = sorted(str(name) for name in args.inputs) == ["A", "B", "C", "D"]
    if not complete:
        _log(f"[probe] merge refuses a partial set {sorted(results)}")
        return EXIT_FAILED
    payload = _summary_payload(results, locator, dloc, run_dir=Path(args.run_dir),
                               complete=complete,
                               baselines=not args.no_baselines)
    path = Path(args.report_dir) / "train-all.json"
    write_json(path, payload)
    _log(json.dumps({"report": str(path), "decision": payload["decision"]},
                    indent=2, ensure_ascii=False))
    return EXIT_STOP_LOSS if (payload["decision"]
                              and payload["decision"]["stop_loss"]) else EXIT_OK


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def cmd_report(args: argparse.Namespace) -> int:
    from touhou_rl.v400.student.probe.train import TrainError, select_input

    path = Path(args.summary)
    if not path.is_file():
        _log(f"[probe] no summary at {path}; run `train` first")
        return EXIT_FAILED
    payload = json.loads(path.read_text(encoding="utf-8"))
    averaged = {name: {"micro": {"f1": entry["micro_f1"]["mean"]}}
                for name, entry in payload.get("results", {}).items()}
    decision = select_input(averaged, locator=payload.get("locator"))
    table = {
        "f1": decision["f1"],
        "locator_median_px": decision["locator_median_px"],
        "chosen": decision["chosen"],
        "stop_loss": decision["stop_loss"],
        "resolution_not_the_bottleneck": decision["resolution_not_the_bottleneck"],
        "baselines": payload.get("baselines"),
    }
    _log(json.dumps(table, indent=2, ensure_ascii=False))
    if args.write:
        target = Path(args.write)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"decision": decision, "table": table},
                                     indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
    return EXIT_STOP_LOSS if decision["stop_loss"] else EXIT_OK


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="v400_p3_probe",
        description="T2a perception probe (EXP-P3-010): resolution F1 table.")
    subparsers = parser.add_subparsers(dest="command")

    collect = subparsers.add_parser("collect", help="replay D0 at native resolution")
    collect.add_argument("--seeds", type=parse_seeds, default=[1, 2, 3, 4])
    collect.add_argument("--binary", default=DEFAULT_BINARY)
    collect.add_argument("--expected-sha", default=DEFAULT_ENGINE_SHA256)
    collect.add_argument("--shard-root", default=DEFAULT_SHARDS)
    collect.add_argument("--data-template", default=DEFAULT_DATA_TEMPLATE)
    collect.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    collect.add_argument("--out-dir", default=DEFAULT_OUT)
    collect.add_argument("--report-dir", default=REPORT_DIR)
    collect.add_argument("--stage", type=int, default=0)
    collect.add_argument("--limit", type=int, default=None,
                         help="only the first N decisions (smoke runs)")
    collect.add_argument("--timeout", type=float, default=300.0)
    collect.add_argument("--concurrency", type=int, default=1,
                         help="informational: this process runs seeds serially")
    collect.add_argument("--dry-run", action="store_true")
    collect.set_defaults(func=cmd_collect)

    train = subparsers.add_parser("train", help="train + evaluate the four inputs")
    train.add_argument("--run-dir", default=DEFAULT_OUT)
    train.add_argument("--report-dir", default=REPORT_DIR)
    train.add_argument("--inputs", nargs="+", default=["A", "B", "C", "D"])
    train.add_argument("--train-seeds", nargs="+", type=int, default=None)
    train.add_argument("--epochs", type=int, default=6)
    train.add_argument("--batch", type=int, default=256)
    train.add_argument("--lr", type=float, default=3e-3)
    train.add_argument("--limit", type=int, default=None)
    train.add_argument("--device", default=None)
    train.add_argument("--save-models", default=None)
    train.add_argument("--locator", action="store_true", default=True)
    train.add_argument("--no-locator", dest="locator", action="store_false")
    train.add_argument("--locator-seeds", nargs="+", type=int, default=None,
                       help="training seeds for the locator (default: the first "
                            "--train-seeds value, i.e. 0)")
    train.add_argument("--locator-models", default=None,
                       help="where D+loc looks for locator_<seed>.pt "
                            "(default: --save-models)")
    train.add_argument("--dloc", action="store_true", default=True,
                       help="score D with locator-placed windows after training")
    train.add_argument("--no-dloc", dest="dloc", action="store_false")
    train.add_argument("--dry-run", action="store_true")
    train.set_defaults(func=cmd_train)

    locator = subparsers.add_parser(
        "locator", help="train + evaluate the pixel ship locator (D's gate)")
    locator.add_argument("--run-dir", default=DEFAULT_OUT)
    locator.add_argument("--report-dir", default=REPORT_DIR)
    locator.add_argument("--save-models", default=None)
    locator.add_argument("--locator-seeds", nargs="+", type=int, default=[0, 1, 2])
    locator.add_argument("--epochs", type=int, default=6)
    locator.add_argument("--batch", type=int, default=256)
    locator.add_argument("--lr", type=float, default=3e-3)
    locator.add_argument("--limit", type=int, default=None)
    locator.add_argument("--device", default=None)
    locator.add_argument("--dry-run", action="store_true")
    locator.set_defaults(func=cmd_locator)

    dloc = subparsers.add_parser(
        "dloc", help="score input D with locator-placed windows (D+loc)")
    dloc.add_argument("--run-dir", default=DEFAULT_OUT)
    dloc.add_argument("--report-dir", default=REPORT_DIR)
    dloc.add_argument("--models", default=None,
                      help="directory holding probe_D_<seed>.pt "
                           "(default: <run-dir>/models)")
    dloc.add_argument("--locator-models", default=None,
                      help="directory holding locator_<seed>.pt (default: --models)")
    dloc.add_argument("--train-seeds", nargs="+", type=int, default=[0, 1, 2])
    dloc.add_argument("--device", default=None)
    dloc.add_argument("--dry-run", action="store_true")
    dloc.set_defaults(func=cmd_dloc)

    merge = subparsers.add_parser(
        "merge", help="combine per-input results and apply the pre-registered rule")
    merge.add_argument("--run-dir", default=DEFAULT_OUT)
    merge.add_argument("--report-dir", default=REPORT_DIR)
    merge.add_argument("--search-dir", nargs="+", default=None,
                       help="directories to read train-<X>.json from, in order "
                            "(default: --report-dir)")
    merge.add_argument("--inputs", nargs="+", default=["A", "B", "C", "D"])
    merge.add_argument("--no-baselines", action="store_true")
    merge.set_defaults(func=cmd_merge)

    report = subparsers.add_parser("report", help="apply the pre-registered rule")
    report.add_argument("--summary", default=str(Path(REPORT_DIR) / "train-all.json"))
    report.add_argument("--write", default=None)
    report.set_defaults(func=cmd_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return EXIT_USAGE
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
