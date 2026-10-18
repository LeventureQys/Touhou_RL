"""Command line for the three pipeline stages.

    python -m touhou_rl.pipeline collect --modality vision --replay ... --out ...
    python -m touhou_rl.pipeline process --raw ... [--out ...]
    python -m touhou_rl.pipeline train   --manifest ... --model vision --out ...
    python -m touhou_rl.pipeline inspect --manifest ...      # what did I actually build?

Each subcommand prints its report as JSON, so a shell pipeline can chain them and
a report can quote real numbers instead of remembered ones.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _collect(args: argparse.Namespace) -> dict:
    from .collect import CollectSpec, run_collect

    spec = CollectSpec(
        modality=args.modality,
        out_dir=Path(args.out),
        source=args.source,
        replay=Path(args.replay) if args.replay else None,
        every=args.every,
        stage=args.stage,
        episodes=args.episodes,
        max_seconds=args.max_seconds,
        difficulty=args.difficulty,
        team=args.team,
        tag=args.tag,
        overwrite=args.overwrite,
        extra={"epsilon": args.epsilon} if args.source == "teacher" else {},
    )
    return run_collect(spec)


def _process(args: argparse.Namespace) -> dict:
    from .process import ProcessSpec, run_process

    spec = ProcessSpec(
        raw_dir=Path(args.raw),
        out_dir=Path(args.out) if args.out else None,
        modality=args.modality or None,
        compute_teacher=not args.no_teacher,
        seed=args.seed,
        val_fraction=args.val_fraction,
        min_frames=args.min_frames,
    )
    return run_process(spec)


def _train(args: argparse.Namespace) -> dict:
    from .train import TrainSpec, run_train

    spec = TrainSpec(
        manifest=Path(args.manifest),
        out_dir=Path(args.out),
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        depth=args.depth,
        width=args.width,
        trunk_hidden=args.trunk_hidden,
        vision_stage=args.vision_stage,
        init_from=Path(args.init_from) if args.init_from else None,
        max_batches_per_epoch=args.max_batches_per_epoch,
    )
    return run_train(spec)


def _inspect(args: argparse.Namespace) -> dict:
    from .manifest import DatasetManifest

    manifest = DatasetManifest.load(Path(args.manifest))
    payload = manifest.summary()
    try:
        train, val = manifest.verify(Path(manifest.root), deep=args.deep)
        payload["verified"] = True
        payload["verified_split"] = {"train": len(train), "val": len(val)}
    except Exception as exc:  # report, do not raise: inspect is a diagnostic
        payload["verified"] = False
        payload["verification_error"] = f"{type(exc).__name__}: {exc}"
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="touhou_rl.pipeline", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="stage 1: game -> raw shards")
    collect.add_argument("--modality", default="vision", choices=["privileged", "vision"])
    collect.add_argument("--source", default="replay", choices=["replay", "teacher"])
    collect.add_argument("--replay", default="")
    collect.add_argument("--out", required=True)
    collect.add_argument("--every", type=int, default=2, help="vision: record every N-th frame")
    collect.add_argument("--stage", type=int, default=-1, help="replay stage index (-1 = auto)")
    collect.add_argument("--episodes", type=int, default=4, help="teacher source: episodes")
    collect.add_argument("--max-seconds", type=float, default=600.0)
    collect.add_argument("--difficulty", default="lunatic")
    collect.add_argument("--team", type=int, default=0)
    collect.add_argument("--tag", default="collect")
    collect.add_argument("--epsilon", type=float, default=0.05, help="teacher source: perturbation")
    collect.add_argument("--overwrite", action="store_true")
    collect.set_defaults(func=_collect)

    process = sub.add_parser("process", help="stage 2: validate, label, manifest")
    process.add_argument("--raw", required=True)
    process.add_argument("--out", default="")
    process.add_argument("--modality", default="", choices=["", "privileged", "vision"])
    process.add_argument("--no-teacher", action="store_true", help="do not compute teacher values")
    process.add_argument("--seed", type=int, default=7)
    process.add_argument("--val-fraction", type=float, default=0.1)
    process.add_argument("--min-frames", type=int, default=64)
    process.set_defaults(func=_process)

    train = sub.add_parser("train", help="stage 3: manifest -> checkpoint + metrics")
    train.add_argument("--manifest", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--model", default="vision", choices=["privileged_bc", "privileged_value", "vision"])
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--device", default="cuda")
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--depth", type=int, default=3)
    train.add_argument("--width", type=int, default=32)
    train.add_argument("--trunk-hidden", type=int, default=256)
    train.add_argument("--vision-stage", default="both", choices=["state", "policy", "both"])
    train.add_argument("--init-from", default="")
    train.add_argument("--max-batches-per-epoch", type=int, default=0)
    train.set_defaults(func=_train)

    inspect = sub.add_parser("inspect", help="diagnostic: does the manifest still match the data?")
    inspect.add_argument("--manifest", required=True)
    inspect.add_argument("--deep", action="store_true", help="re-hash every shard")
    inspect.set_defaults(func=_inspect)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.func(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
