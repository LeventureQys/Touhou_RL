"""Replay a recorded game trace through the v3.7.0 probe (D8 readiness, offline).

Why this exists
---------------
The live probe needs the game.  The next best thing is the *recorded* game state:
the trace files already contain the player position and the bullet pool per
decision, so the v3.7.0 observation + forecast + controller can be exercised on
real bullet patterns without launching TH08.

What it does **not** prove
--------------------------
Same-frame synchronisation, input timing and the actual input path still need the
game: a replay cannot tell whether ``read_synced`` would have agreed with the
engine in real time.  Everything produced here is labelled
``source: "recorded-game-trace"`` and ``live: false``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import contracts, forecast
from .contracts import (
    Completeness,
    EscapeObservation,
    HazardObs,
    MotionModel,
    PlayerObs,
    Source,
    HURTBOX_HALF_PX,
)

TRACE_V350 = "v350-trace/1"
TRACE_V370 = "v370-trace/1"


@dataclass
class RecordedSource:
    """Hands out one observation per recorded decision, plus its hazard field."""

    trace_path: Path | str
    sidecar_path: Path | str | None = None
    limit: int | None = None
    start: int = 0
    index: int = 0
    _records: list[dict] = field(default_factory=list, init=False)
    _sidecar: dict = field(default_factory=dict, init=False)
    _schema: str = field(default="", init=False)
    _base: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.trace_path = Path(self.trace_path)
        with self.trace_path.open("r", encoding="utf-8") as handle:
            self._records = [json.loads(line) for line in handle if line.strip()]
        if self.start:
            self._records = self._records[self.start :]
            self._base = int(self.start)
        if self.limit:
            self._records = self._records[: self.limit]
        if self._records:
            self._schema = self._records[0].get("schema", "")
        sidecar = Path(self.sidecar_path) if self.sidecar_path else self._guess_sidecar()
        if sidecar is not None and sidecar.is_file():
            data = np.load(sidecar)
            self._sidecar = {key: np.asarray(data[key]) for key in data.files}

    def _guess_sidecar(self) -> Path | None:
        stem = self.trace_path.stem
        for candidate in (
            self.trace_path.with_suffix(".npz"),
            self.trace_path.parent.parent / "bullets" / f"{stem}.npz",
            self.trace_path.parent / "bullets" / f"{stem}.npz",
        ):
            if candidate.is_file():
                return candidate
        return None

    # -- hazards ---------------------------------------------------------
    def hazards(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        record = self._records[min(self.index, len(self._records) - 1)]
        absolute = self._base + self.index
        if self._schema == TRACE_V350 and "offsets" in self._sidecar:
            offsets = self._sidecar["offsets"]
            start = int(offsets[min(absolute, offsets.size - 2)])
            end = int(offsets[min(absolute + 1, offsets.size - 1)])
            positions = self._sidecar["positions"][start:end]
            velocities = self._sidecar["velocities"][start:end]
            half = self._sidecar["half"][start:end]
            halves = np.stack([half, half], axis=1)
            return positions.astype(np.float64), velocities.astype(np.float64), halves.astype(np.float64)
        # v370 sidecar layout
        if "offsets" in self._sidecar:
            index = self._base + int(record.get("sidecar_index", self.index))
            offsets = self._sidecar["offsets"]
            start = int(offsets[min(index, offsets.size - 2)])
            end = int(offsets[min(index + 1, offsets.size - 1)])
            return (
                self._sidecar["positions"][start:end].astype(np.float64),
                self._sidecar["velocities"][start:end].astype(np.float64),
                self._sidecar["halves"][start:end].astype(np.float64),
            )
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 2))

    def field(self) -> forecast.DangerField:
        positions, velocities, halves = self.hazards()
        count = positions.shape[0]
        return forecast.DangerField(
            frame_base=int(self.frame_id()),
            positions=positions,
            velocities=velocities,
            halves=halves,
            spawn=np.full(count, -10**9, dtype=np.int64),
            despawn=np.full(count, 10**9, dtype=np.int64),
            config=forecast.ForecastConfig(),
        ).compressed()

    # -- observation -----------------------------------------------------
    def frame_id(self) -> int:
        record = self._records[min(self.index, len(self._records) - 1)]
        if self._schema == TRACE_V350:
            return int(record.get("clock", {}).get("gameplay_frame", record.get("frame_post", 0)))
        observation = record.get("observation") or {}
        return int(observation.get("frame_id", record.get("frame", 0)))

    def read(self) -> EscapeObservation | None:
        if self.done():
            return None
        record = self._records[self.index]
        positions, velocities, halves = self.hazards()
        if self._schema == TRACE_V350:
            player = record.get("player") or {}
            if "x" not in player or "y" not in player:
                return None
            x, y = float(player["x"]), float(player["y"])
            focus = int(player.get("focus_mode", 0)) & 1
            complete = bool(record.get("complete", True))
            missing = tuple(record.get("incomplete_reasons") or ())
        else:
            observation = record.get("observation") or {}
            player = observation.get("player") or {}
            if "x" not in player:
                return None
            x, y = float(player["x"]), float(player["y"])
            focus = int(player.get("focus", 0)) & 1
            complete = observation.get("completeness") == "complete"
            missing = tuple(observation.get("missing_reasons") or ())
        hazards = tuple(
            HazardObs(
                position=(float(positions[i, 0]), float(positions[i, 1])),
                velocity=(float(velocities[i, 0]), float(velocities[i, 1])),
                half=(float(halves[i, 0]), float(halves[i, 1])),
                motion_model=MotionModel.LINEAR,
                confidence=1.0,
                valid_until=self.frame_id() + 300,
            )
            for i in range(positions.shape[0])
        )
        return EscapeObservation(
            schema="v370-escape/observation",
            episode_id=record.get("episode", self.trace_path.stem),
            decision_id=int(record.get("decision", self.index + 1)),
            frame_id=self.frame_id(),
            source=Source.PRIVILEGED if record.get("schema") == TRACE_V350 else Source.SYNTHETIC,
            player=PlayerObs(x=x, y=y, focus=focus, hurtbox_half=HURTBOX_HALF_PX),
            frameskip=int(record.get("frameskip", 2)),
            observed_frame_age=0,
            action_latency_frames=1,
            completeness=Completeness.COMPLETE if complete else Completeness.PARTIAL,
            coordinate_contract=dict(contracts.COORDINATE_CONTRACT),
            hazards=hazards,
            bullet_pool_count=int(record.get("bullets", {}).get("count", len(hazards))),
            missing_reasons=missing,
        )

    def advance(self) -> None:
        self.index += 1

    def done(self) -> bool:
        return self.index >= len(self._records)
