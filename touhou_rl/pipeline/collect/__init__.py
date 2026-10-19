"""Pipeline stage 1 - collect: turn a game into raw shards.

One entry point for the four source x modality combinations that exist today, so
downstream stages never have to know which script produced a directory:

===================  =================  =======================================
source               modality           implementation
===================  =================  =======================================
``replay``           ``privileged``     ``capture_dataset.capture_replay``
``replay``           ``vision``         ``vision_capture.capture_vision``
``teacher``          ``privileged``     ``collect_pilot.collect``
``teacher``          ``vision``         not implemented (see README)
===================  =================  =======================================

The stage writes ``collect-report.json`` next to the shards.  It deliberately does
**not** compute teacher values or splits - that is the process stage's job, so a
capture can be re-processed with a different teacher without re-running the game.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

MODALITIES = ("privileged", "vision")
SOURCES = ("replay", "teacher")


@dataclass
class CollectSpec:
    modality: str
    out_dir: Path
    source: str = "replay"
    # replay source
    replay: Path | None = None
    every: int = 2  # vision only: record every N-th logic frame
    stage: int = -1  # -1 = the replay's own first stage
    # teacher source
    episodes: int = 4
    # shared
    max_seconds: float = 600.0
    difficulty: str = "lunatic"
    team: int = 0
    tag: str = "collect"
    overwrite: bool = False
    verbose: bool = True
    extra: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.modality not in MODALITIES:
            raise ValueError(f"modality must be one of {MODALITIES}")
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}")
        if self.source == "replay" and self.replay is None:
            raise ValueError("source=replay requires a replay file")
        if self.source == "teacher" and self.modality == "vision":
            raise NotImplementedError(
                "vision capture from the teacher is not implemented yet: the teacher "
                "acts through the environment, and vision_capture currently only reads "
                "frames during replay playback. Use modality=privileged for teacher "
                "capture (collect_pilot) until this lands."
            )
        if self.every < 1:
            raise ValueError("every must be >= 1")


def _difficulty_code(name: str) -> int:
    from ... import spec

    table = {"easy": spec.EASY, "normal": spec.NORMAL, "hard": spec.HARD, "lunatic": spec.LUNATIC}
    if name not in table:
        raise ValueError(f"unknown difficulty {name!r}")
    return table[name]


def run_collect(spec: CollectSpec) -> dict:
    """Run one capture; returns the report that was written to disk."""
    spec.validate()
    out_dir = Path(spec.out_dir)
    if out_dir.exists() and any(out_dir.glob("*.npz")) and not spec.overwrite:
        raise FileExistsError(f"{out_dir} already holds shards; pass overwrite=True")
    started = time.time()
    report: dict

    if spec.modality == "privileged" and spec.source == "replay":
        from ...capture_dataset import capture_replay

        report = capture_replay(
            Path(spec.replay),
            out_dir,
            tag=spec.tag,
            stage=None if spec.stage < 0 else spec.stage,
            max_seconds=spec.max_seconds,
            overwrite=spec.overwrite,
            verbose=spec.verbose,
        ).as_dict()
    elif spec.modality == "privileged" and spec.source == "teacher":
        from ...collect_pilot import CollectConfig, collect

        report = collect(
            CollectConfig(
                out_dir=out_dir,
                episodes=spec.episodes,
                max_steps=20000,
                difficulty=_difficulty_code(spec.difficulty),
                team=spec.team,
                tag=spec.tag,
                max_minutes=spec.max_seconds / 60.0,
                overwrite=spec.overwrite,
                epsilon=float(spec.extra.get("epsilon", 0.05)),
            )
        )
    elif spec.modality == "vision":
        from ...vision_capture import capture_vision

        report = capture_vision(
            Path(spec.replay),
            out_dir,
            tag=spec.tag,
            every=spec.every,
            max_seconds=spec.max_seconds,
            stage=None if spec.stage < 0 else spec.stage,
            allow_append=spec.overwrite,
            freeze_limit=int(spec.extra.get("freeze_limit", 5)),
            verbose=spec.verbose,
        )
    else:  # pragma: no cover - guarded by validate()
        raise NotImplementedError(spec.modality)

    payload = {
        "stage": "collect",
        "modality": spec.modality,
        "source": spec.source,
        "out_dir": str(out_dir),
        "difficulty": spec.difficulty,
        "team": spec.team,
        "extra": spec.extra,
        "elapsed_seconds": round(time.time() - started, 1),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "report": report,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "collect-report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload
