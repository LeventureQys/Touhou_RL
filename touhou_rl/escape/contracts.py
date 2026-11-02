"""Frozen v3.7.0 records: units, observation/forecast/route/decision, JSON schema.

Why a generated schema instead of a hand-written one
----------------------------------------------------
The requirement (FR-01, FR-11) is that a *missing* read and an *empty* pool stay
distinguishable, and that every decision carries units, reasons and timing.  A
hand-maintained ``*.json`` schema drifts from the dataclasses on the first
rename.  Here the schema is derived from the record definitions, and the file
under ``Document/Update/Alpha/v3.7.0/contract/`` is a build artifact: a test
fails when the committed copy differs from :func:`contract_schema`.

Units and coordinates are not free-form: :data:`COORDINATE_CONTRACT` is part of
every observation and is re-checked by :func:`validate_document`, so an encoder
that silently switches from pixels to normalised coordinates is rejected instead
of producing a plausible-looking plan.

Everything here is pure: no memory reads, no game launch, no learning.
"""

from __future__ import annotations

import hashlib
import json
import types
from dataclasses import MISSING, dataclass, field, fields
from enum import Enum
from typing import Any, Union, get_args, get_origin, get_type_hints

from .. import spec

CONTRACT_VERSION = "v370-escape/1"

OBSERVATION_SCHEMA = "v370-escape/observation"
FORECAST_SCHEMA = "v370-escape/forecast"
ROUTE_SCHEMA = "v370-escape/route"
DECISION_SCHEMA = "v370-escape/decision"

#: The real game's ``Player::size`` is read from the shot file as
#: ``hurtboxSize / 2`` (Player.cpp:1576-1577); the measured value is 0.825 px in
#: *both* focus states, because focusing changes speed, not the hitbox.
HURTBOX_HALF_PX = 0.825

#: Measured on the live game by ``tools/probes/calibrate_player.py``
#: (``third_party/th08/build/rl/probe-v103/player-calibration.json``).
#: Per-axis values; a diagonal moves both axes by the diagonal value, matching
#: ``Player.cpp:766-789`` (``focusedDiagonalSpeed``/``normalDiagonalSpeed``).
MOVEMENT_MODEL = {
    "axis": 4.0,
    "diagonal": 2.8284,
    "focus_axis": 2.0,
    "focus_diagonal": 1.4142,
}

#: ``Player.cpp:856-871`` clamps the *centre* to ``topLeft + [0, size]``; the
#: measured clamp agrees to the float (8..376, 16..432).
PLAYER_CENTER_CLAMP = {
    "x": [spec.MOVE_AREA_X, spec.MOVE_AREA_X + spec.MOVE_AREA_W],
    "y": [spec.MOVE_AREA_Y, spec.MOVE_AREA_Y + spec.MOVE_AREA_H],
}

#: Everything a consumer may assume about a v3.7.0 observation.  Copied into
#: each record and compared byte-for-byte by the validator: a change here is a
#: contract change, not a local tweak.
COORDINATE_CONTRACT: dict[str, Any] = {
    "length_unit": "px",
    "time_unit": "game_frame",
    "frame_rate": 60,
    "origin": "screen top-left; the playfield starts at (PLAYFIELD_X, PLAYFIELD_Y)",
    "axes": "+x right, +y down",
    "playfield_xywh": [spec.PLAYFIELD_X, spec.PLAYFIELD_Y, spec.PLAYFIELD_W, spec.PLAYFIELD_H],
    "move_area_xywh": [spec.MOVE_AREA_X, spec.MOVE_AREA_Y, spec.MOVE_AREA_W, spec.MOVE_AREA_H],
    "player_center_clamp": PLAYER_CENTER_CLAMP,
    "player_speed_px_per_frame": MOVEMENT_MODEL,
    "hurtbox_half_px": HURTBOX_HALF_PX,
    "bullet_velocity_unit": "px per frame at Supervisor.framerateMultiplier == 1",
    "bullet_motion_model": "position += velocity once per logic frame (BulletManager.cpp:865)",
    "collision_model": "axis-aligned box (bullet collisionSize) vs axis-aligned player hurtbox",
    "action_space": "[move, shoot, focus, bomb] = MultiDiscrete(9, 2, 2, 2)",
}

N_FIRST_ACTIONS = 18  # focus(2) x move(9); shoot/bomb are not part of routing


class Source(str, Enum):
    """Where a record came from -- never inferred, always written down."""

    SYNTHETIC = "synthetic"
    PRIVILEGED = "privileged"
    HUMAN = "human"
    STUDENT = "student"


class Completeness(str, Enum):
    """Whether the observation is usable as a claim about the whole pool."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    UNREADABLE = "unreadable"


class MotionModel(str, Enum):
    LINEAR = "linear"
    UNKNOWN = "unknown"


class FirstActionStatus(str, Enum):
    """Per first action: was a route found, refused with evidence, or unknown?"""

    FEASIBLE = "feasible"
    CERTIFIED_BLOCKED = "certified_blocked"
    UNKNOWN = "unknown"
    IMMEDIATE_COLLISION = "immediate_collision"


class SolutionStatus(str, Enum):
    """FR-10: a failed search is not a proof of no route."""

    FEASIBLE = "feasible"
    NO_SOLUTION_CERTIFIED = "no_solution_certified"
    BUDGET_EXHAUSTED = "budget_exhausted"
    MODEL_UNKNOWN = "model_unknown"


class Intent(str, Enum):
    LOCAL = "local"
    PREPARE_EXIT = "prepare_exit"
    EVACUATE = "evacuate"
    HOLD = "hold"
    REPLAN = "replan"
    EMERGENCY = "emergency"
    UNKNOWN = "unknown"


class RejectReason(str, Enum):
    CERTIFIED_BLOCKED = "certified_blocked"
    IMMEDIATE_COLLISION = "immediate_collision"
    PREFER_SAFER_ROUTE = "prefer_safer_route"
    INTENT_HOLD = "intent_hold"
    BUDGET = "budget"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Records (all immutable; JSON form is ``asdict`` with tuples turned into lists)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlayerObs:
    x: float = field(metadata={"description": "player centre x, px, screen coordinates"})
    y: float = field(metadata={"description": "player centre y, px, screen coordinates"})
    focus: int = field(metadata={"minimum": 0, "maximum": 1})
    hurtbox_half: float = field(metadata={"minimum": 0.0})
    alive: bool = field(default=True)
    invulnerable: bool = field(default=False)
    speed_model: str = field(default="border-team-lunatic-1.00d")


@dataclass(frozen=True)
class HazardObs:
    """One active hazard entity as the privileged snapshot sees it."""

    position: tuple[float, float]
    velocity: tuple[float, float]
    half: tuple[float, float] = field(metadata={"description": "collision half extents, px"})
    motion_model: MotionModel = MotionModel.LINEAR
    confidence: float = field(default=1.0, metadata={"minimum": 0.0, "maximum": 1.0})
    valid_until: int = field(default=0, metadata={"minimum": 0})
    collision_disabled: bool = False
    track_id: int | None = None


@dataclass(frozen=True)
class EscapeObservation:
    schema: str = field(metadata={"const": OBSERVATION_SCHEMA})
    episode_id: str
    decision_id: int = field(metadata={"minimum": 0})
    frame_id: int = field(metadata={"minimum": 0})
    source: Source
    player: PlayerObs
    frameskip: int = field(metadata={"minimum": 1})
    observed_frame_age: int = field(metadata={"minimum": 0})
    action_latency_frames: int = field(metadata={"minimum": 0})
    completeness: Completeness
    coordinate_contract: dict[str, Any]
    #: ``None`` means "the pool could not be read"; ``()`` means "read, empty".
    hazards: tuple[HazardObs, ...] | None = None
    previous_action: tuple[int, int, int, int] | None = None
    missing_reasons: tuple[str, ...] = ()
    unsupported_hazards: tuple[str, ...] = ()
    bullet_pool_count: int | None = field(default=None, metadata={"minimum": 0})
    #: Remaining bombs.  Live runs need this: without it every decision sees 0
    #: and the bomb arbiter can never fire (measured: the subject never bombed
    #: and died where the baseline bombed and survived).
    bombs: int = field(default=0, metadata={"minimum": 0})


@dataclass(frozen=True)
class DangerForecast:
    """Time-indexed occupancy.  Never a permanent union of future positions."""

    schema: str = field(metadata={"const": FORECAST_SCHEMA})
    model_version: str
    time_nodes_frames: tuple[int, ...] = field(metadata={"minItems": 1})
    coverage_shape: tuple[int, int]
    coverage: tuple[tuple[float, ...], ...]
    uncertainty: tuple[float, ...]
    valid_until: int = field(metadata={"minimum": 0})
    calibrated: bool = False
    dropped_entities: int = field(default=0, metadata={"minimum": 0})
    drop_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class CorridorBox:
    frame: int = field(metadata={"minimum": 0})
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class PositionKnot:
    """One route knot: where the player is (or is planned to be) at ``frame``."""

    frame: int = field(metadata={"minimum": 0})
    x: float
    y: float


@dataclass(frozen=True)
class RoutePlan:
    schema: str = field(metadata={"const": ROUTE_SCHEMA})
    route_id: str
    solution_status: SolutionStatus
    first_action_status: tuple[FirstActionStatus, ...] = field(
        metadata={"minItems": N_FIRST_ACTIONS, "maxItems": N_FIRST_ACTIONS}
    )
    action_sequence: tuple[tuple[int, int, int, int], ...] = ()
    position_time_knots: tuple[PositionKnot, ...] = ()
    corridor_by_time: tuple[CorridorBox, ...] = ()
    first_action_reason: tuple[str, ...] = ()
    min_clearance: float | None = None
    exit_eta: int | None = field(default=None, metadata={"minimum": 0})
    closure_eta: int | None = field(default=None, metadata={"minimum": 0})
    slack: int | None = None
    continuation_window: int = field(default=0, metadata={"minimum": 0})
    confidence: float = field(default=0.0, metadata={"minimum": 0.0, "maximum": 1.0})
    valid_until: int = field(default=0, metadata={"minimum": 0})
    alternatives: tuple[str, ...] = ()
    node_budget: int = field(default=0, metadata={"minimum": 0})
    elapsed_ms: float = field(default=0.0, metadata={"minimum": 0.0})


@dataclass(frozen=True)
class RejectedAction:
    action: tuple[int, int, int, int]
    reason: RejectReason
    evidence: str = ""
    min_clearance: float | None = None


@dataclass(frozen=True)
class ControlDecision:
    schema: str = field(metadata={"const": DECISION_SCHEMA})
    decision_id: int = field(metadata={"minimum": 0})
    intent: Intent
    accepted_action: tuple[int, int, int, int]
    reason: str
    route_id: str | None = None
    rejected_actions: tuple[RejectedAction, ...] = ()
    emergency_override: bool = False
    fallback: str | None = None
    bomb_reason: str | None = None
    deadline_miss: bool = False
    timing_breakdown: dict[str, float] = field(default_factory=dict)


RECORDS: dict[str, type] = {
    OBSERVATION_SCHEMA: EscapeObservation,
    FORECAST_SCHEMA: DangerForecast,
    ROUTE_SCHEMA: RoutePlan,
    DECISION_SCHEMA: ControlDecision,
}

# Helper records that only appear nested; they still get a ``$defs`` entry.
NESTED_RECORDS: tuple[type, ...] = (
    PlayerObs,
    HazardObs,
    CorridorBox,
    PositionKnot,
    RejectedAction,
)


class ContractError(ValueError):
    """Raised when a document violates the frozen contract."""


# ---------------------------------------------------------------------------
# Type -> schema, derived from the annotations and the field metadata
# ---------------------------------------------------------------------------
def _is_union(tp: Any) -> bool:
    return get_origin(tp) in (Union, types.UnionType)


def _schema_for_type(tp: Any) -> dict[str, Any]:
    if isinstance(tp, type) and issubclass(tp, Enum):
        return {"type": "string", "enum": [member.value for member in tp]}
    if _is_union(tp):
        args = [arg for arg in get_args(tp) if arg is not type(None)]
        if len(args) != 1:
            raise TypeError(f"only Optional unions are supported, got {tp!r}")
        return {"oneOf": [_schema_for_type(args[0]), {"type": "null"}]}
    if get_origin(tp) is tuple:
        args = get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            return {"type": "array", "items": _schema_for_type(args[0])}
        if not args:
            return {"type": "array"}
        first = _schema_for_type(args[0])
        if any(_schema_for_type(arg) != first for arg in args[1:]):
            raise TypeError(f"heterogeneous tuples are not supported, got {tp!r}")
        return {"type": "array", "items": first, "minItems": len(args), "maxItems": len(args)}
    if get_origin(tp) is dict or tp is dict:
        args = get_args(tp)
        value_type = args[1] if len(args) == 2 else float
        return {"type": "object", "additionalProperties": _schema_for_type(value_type)}
    if get_origin(tp) is list or tp is list:
        args = get_args(tp)
        item = args[0] if args else float
        return {"type": "array", "items": _schema_for_type(item)}
    if isinstance(tp, type) and hasattr(tp, "__dataclass_fields__"):
        return {"$ref": f"#/$defs/{tp.__name__}"}
    mapping = {int: {"type": "integer"}, float: {"type": "number"}, str: {"type": "string"}, bool: {"type": "boolean"}}
    if tp in mapping:
        return dict(mapping[tp])
    if tp is Any:
        return {}
    raise TypeError(f"no schema mapping for {tp!r}")


def _field_schema(record: type, the_field) -> dict[str, Any]:
    hints = get_type_hints(record)
    schema = _schema_for_type(hints[the_field.name])
    for key in ("minimum", "maximum", "minItems", "maxItems", "const", "description"):
        if key in the_field.metadata:
            schema[key] = the_field.metadata[key]
    return schema


def _record_schema(record: type) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for the_field in fields(record):
        properties[the_field.name] = _field_schema(record, the_field)
        if the_field.default is MISSING and the_field.default_factory is MISSING:
            required.append(the_field.name)
    doc: dict[str, Any] = {"type": "object", "additionalProperties": False, "properties": properties}
    if required:
        doc["required"] = required
    return doc


def contract_schema() -> dict[str, Any]:
    """The JSON Schema that ships as ``contract/escape-schema.json``."""

    defs: dict[str, Any] = {}
    for record in (*RECORDS.values(), *NESTED_RECORDS):
        defs[record.__name__] = _record_schema(record)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": CONTRACT_VERSION,
        "title": "v3.7.0 escape records (generated from touhou_rl/escape/contracts.py)",
        "$defs": defs,
        "oneOf": [{"$ref": f"#/$defs/{record.__name__}"} for record in RECORDS.values()],
    }


# ---------------------------------------------------------------------------
# Validation (deliberately dependency-free: the repo has no ``jsonschema``)
# ---------------------------------------------------------------------------
def _type_name(tp: Any) -> str:
    if isinstance(tp, type) and issubclass(tp, Enum):
        return tp.__name__
    return getattr(tp, "__name__", str(tp))


def _check_scalar(tp: Any, value: Any, path: str, meta: dict, issues: list[str]) -> None:
    if isinstance(tp, type) and issubclass(tp, Enum):
        values = {member.value for member in tp}
        if value not in values:
            issues.append(f"{path}: {value!r} not in {sorted(values)}")
        return
    if tp is int:
        # ``bool`` is an ``int`` subclass; an accidental True must not pass as 1.
        if not isinstance(value, int) or isinstance(value, bool):
            issues.append(f"{path}: expected an integer, got {value!r}")
            return
    elif tp is float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            issues.append(f"{path}: expected a number, got {value!r}")
            return
    elif tp is str:
        if not isinstance(value, str):
            issues.append(f"{path}: expected a string, got {value!r}")
            return
    elif tp is bool:
        if not isinstance(value, bool):
            issues.append(f"{path}: expected a boolean, got {value!r}")
            return
    if "const" in meta and value != meta["const"]:
        issues.append(f"{path}: expected {meta['const']!r}, got {value!r}")


def _check_value(tp: Any, value: Any, path: str, meta: dict, issues: list[str]) -> None:
    if value is None:
        if not _is_union(tp):
            issues.append(f"{path}: null is not allowed for {_type_name(tp)}")
        return
    if _is_union(tp):
        args = [arg for arg in get_args(tp) if arg is not type(None)]
        if len(args) != 1:
            issues.append(f"{path}: unsupported union {tp!r}")
            return
        _check_value(args[0], value, path, meta, issues)
        return
    if isinstance(tp, type) and issubclass(tp, Enum):
        _check_scalar(tp, value, path, meta, issues)
        return
    if get_origin(tp) is tuple:
        if not isinstance(value, (list, tuple)):
            issues.append(f"{path}: expected an array, got {type(value).__name__}")
            return
        args = get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            for index, item in enumerate(value):
                _check_value(args[0], item, f"{path}[{index}]", {}, issues)
            return
        if len(value) != len(args):
            issues.append(f"{path}: expected {len(args)} items, got {len(value)}")
            return
        for index, (arg, item) in enumerate(zip(args, value)):
            _check_value(arg, item, f"{path}[{index}]", {}, issues)
        return
    origin = get_origin(tp)
    if origin is dict or tp is dict:
        if not isinstance(value, dict):
            issues.append(f"{path}: expected an object, got {type(value).__name__}")
            return
        args = get_args(tp)
        value_type = args[1] if len(args) == 2 else float
        for key, item in value.items():
            if not isinstance(key, str):
                issues.append(f"{path}: object keys must be strings, got {key!r}")
                continue
            _check_value(value_type, item, f"{path}.{key}", {}, issues)
        return
    if origin is list or tp is list:
        args = get_args(tp)
        item_type = args[0] if args else float
        if not isinstance(value, (list, tuple)):
            issues.append(f"{path}: expected an array, got {type(value).__name__}")
            return
        for index, item in enumerate(value):
            _check_value(item_type, item, f"{path}[{index}]", {}, issues)
        return
    if isinstance(tp, type) and hasattr(tp, "__dataclass_fields__"):
        _check_record(tp, value, path, issues)
        return
    _check_scalar(tp, value, path, meta, issues)


def _check_record(record: type, doc: Any, path: str, issues: list[str]) -> None:
    if not isinstance(doc, dict):
        issues.append(f"{path}: expected an object, got {type(doc).__name__}")
        return
    hints = get_type_hints(record)
    known = {the_field.name: the_field for the_field in fields(record)}
    for extra in sorted(set(doc) - set(known)):
        issues.append(f"{path}.{extra}: unexpected field (schema has no such key)")
    for name, the_field in known.items():
        meta = dict(the_field.metadata)
        sub_path = f"{path}.{name}"
        if name not in doc:
            if the_field.default is MISSING and the_field.default_factory is MISSING:
                issues.append(f"{sub_path}: missing required field")
            continue
        _check_value(hints[name], doc[name], sub_path, meta, issues)
        value = doc[name]
        if value is None:
            continue
        if "minimum" in meta and isinstance(value, (int, float)) and value < meta["minimum"]:
            issues.append(f"{sub_path}: {value} below minimum {meta['minimum']}")
        if "maximum" in meta and isinstance(value, (int, float)) and value > meta["maximum"]:
            issues.append(f"{sub_path}: {value} above maximum {meta['maximum']}")
        if "minItems" in meta and isinstance(value, (list, tuple)) and len(value) < meta["minItems"]:
            issues.append(f"{sub_path}: {len(value)} items, minimum {meta['minItems']}")
        if "maxItems" in meta and isinstance(value, (list, tuple)) and len(value) > meta["maxItems"]:
            issues.append(f"{sub_path}: {len(value)} items, maximum {meta['maxItems']}")


def _semantic_issues(record: type, doc: dict) -> list[str]:
    issues: list[str] = []
    if record is EscapeObservation:
        if doc.get("coordinate_contract") != COORDINATE_CONTRACT:
            issues.append(
                "coordinate_contract: differs from the frozen units (px / game_frame / +y down, "
                "playfield at (32, 16))"
            )
        hazards = doc.get("hazards")
        if hazards is None:
            if doc.get("completeness") == Completeness.COMPLETE.value:
                issues.append("completeness: an unreadable hazard pool cannot be 'complete'")
            if not doc.get("missing_reasons"):
                issues.append("missing_reasons: an unreadable pool must name its reason")
        else:
            count = doc.get("bullet_pool_count")
            if count is not None and len(hazards) > count:
                issues.append(
                    f"hazards: {len(hazards)} entries exceed the engine pool count {count}"
                )
    if record is DangerForecast:
        times = doc.get("time_nodes_frames") or []
        if any(t <= 0 for t in times):
            issues.append("time_nodes_frames: nodes are offsets and must be > 0")
        if any(b <= a for a, b in zip(times, times[1:])):
            issues.append("time_nodes_frames: nodes must be strictly increasing")
        shape = doc.get("coverage_shape") or [0, 0]
        expected = int(shape[0]) * int(shape[1]) if len(shape) == 2 else -1
        for index, row in enumerate(doc.get("coverage") or []):
            if len(row) != expected:
                issues.append(f"coverage[{index}]: {len(row)} cells, expected {expected}")
        if len(doc.get("coverage") or []) != len(times):
            issues.append("coverage: one row per time node")
        if len(doc.get("uncertainty") or []) != len(times):
            issues.append("uncertainty: one value per time node")
    if record is RoutePlan:
        if doc.get("solution_status") == SolutionStatus.FEASIBLE.value:
            if not doc.get("action_sequence"):
                issues.append("action_sequence: a feasible route needs at least one action")
            if doc.get("min_clearance") is None:
                issues.append("min_clearance: a feasible route must report its clearance")
        reasons = doc.get("first_action_reason") or []
        if reasons and len(reasons) != N_FIRST_ACTIONS:
            issues.append(f"first_action_reason: must be empty or have {N_FIRST_ACTIONS} entries")
    if record is ControlDecision:
        if doc.get("deadline_miss") and doc.get("emergency_override"):
            issues.append(
                "deadline_miss + emergency_override: a miss must not be reported as a planned override"
            )
    return issues


def record_for(doc: dict) -> type:
    if not isinstance(doc, dict):
        raise ContractError(f"expected a JSON object, got {type(doc).__name__}")
    schema = doc.get("schema")
    if schema not in RECORDS:
        raise ContractError(f"unknown schema {schema!r}; expected one of {sorted(RECORDS)}")
    return RECORDS[schema]


def validate_document(doc: dict) -> list[str]:
    """Return every contract violation, or ``[]``.  Never raises for bad data."""

    try:
        record = record_for(doc)
    except ContractError as exc:
        return [str(exc)]
    issues: list[str] = []
    _check_record(record, doc, record.__name__, issues)
    issues.extend(_semantic_issues(record, doc))
    return issues


def assert_valid(doc: dict) -> None:
    issues = validate_document(doc)
    if issues:
        raise ContractError("; ".join(issues))


def to_json_dict(record: Any) -> dict:
    """``asdict`` with tuples converted to lists so the result is JSON-native."""

    def convert(value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    from dataclasses import asdict

    return convert(asdict(record))


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def document_digest(doc: dict) -> str:
    return hashlib.sha256(canonical_json(doc).encode("utf-8")).hexdigest()
