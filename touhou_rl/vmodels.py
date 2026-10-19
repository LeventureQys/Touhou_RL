"""Vision models: one CNN trunk feeding a privileged-state decoder and a policy.

Design decisions worth defending
--------------------------------
**One trunk, two heads.**  The trunk must answer two questions about the same
picture: "what is on screen" (privileged state) and "what should I press"
(candidate values).  Training the state head first gives the policy head a trunk
that already sees bullets instead of one that has to invent geometry from a
sparse action signal - which matters because at 30 decisions/s the action signal
is the expensive one, while privileged labels are free (we read them).

**Candidate-value policy, not four action heads.**  This project measured that
cloning a planner's argmax collapses (ties become contradictory labels, the
average of "left" and "right" is "stand still").  The privileged teacher outputs
a value per candidate; the vision policy regresses those values and takes
``argmax``.  Ties stay ties.

**No recurrence, explicit stack.**  Motion has to come from somewhere.  A
``depth``-frame stack is passed as extra input channels; that keeps the model
stateless and the training/eval paths identical, at the cost of 3x input size for
depth 3.  It is a deliberate simplification - a GRU could be added behind the
same interface later.

Shapes (from :mod:`touhou_rl.vision`): input ``[B, depth, 128, 112, 3]`` uint8
scaled to [0, 1].
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
import torch.nn as nn

from . import obs, spec, vision
from .vision_history import HISTORY_CONTRACTS, v104_error as _v104_error

N_CANDIDATES = 18  # 9 directions x 2 focus states, same set the teacher scores

# --- checkpoint contract (v1.0.4 C6) ---------------------------------------
#: Every architecture this build can *name*.  ``vision_arch`` reports the actual
#: type that was loaded; an unknown type is an error, never a silent "v1".
VISION_ARCH_V1 = "v1"
VISION_ARCH_V2 = "v2"
VISION_ARCH_V3 = "v3-st"
KNOWN_VISION_ARCHES: tuple[str, ...] = (VISION_ARCH_V1, VISION_ARCH_V2, VISION_ARCH_V3)

#: v3 checkpoints carry an explicit schema; v1/v2 keep the v1.0.3 payload shape
#: (``kind``/``arch``/``state_dict``/``config``/``frame``/``extra``) and pre-v1.0.3
#: payloads have no ``arch`` at all and stay v1.
V3_CHECKPOINT_SCHEMA = "vision-checkpoint/3"

#: Identity a v3 checkpoint must record (C6).  ``source_state_sha`` is allowed to
#: be ``""`` only for the first state-pretraining round ("no initialisation
#: source"); a policy-stage checkpoint must carry the real sha.
V3_REQUIRED_EXTRA: tuple[str, ...] = (
    "source_state_sha",
    "teacher_fingerprint",
    "manifest_sha",
    "val_index_sha256",
    "loss_config",
)

#: Contract fields a v3 config has to declare (C5/C6).  Kept as data so the
#: loader, the report and the tests all read one definition.
V3_CONTRACT: dict = {
    "depth": 3,
    "channels": vision.VISION_CHANNELS,  # RGB3
    "normalization": "rgb255-v1",
    "candidate_order_version": "focus9-move-v1",
    "score_semantics": "preference_logits",
    "roi_native_hw": (128, 128),
    "local_grid_hw": (16, 16),
    "align_corners": False,
}
V3_REGION_SIZES: dict[str, tuple[int, int]] = {
    "playfield": (224, 192),
    "hud_score": (34, 216),
    "hud_status": (126, 216),
}

#: ``frame`` is a historical field in every checkpoint: for v3 the real ROI comes
#: from ``config.region_sizes`` and ``frame`` must not be read as the model input.
FRAME_LEGACY_H = vision.VISION_H
FRAME_LEGACY_W = vision.VISION_W

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def is_sha256(value: object) -> bool:
    """``True`` for a 64-char lowercase hex sha256 string."""
    return isinstance(value, str) and bool(_SHA256_RE.match(value))


def _sha256_field(value: object, field: str) -> str:
    if not is_sha256(value):
        raise _v104_error(
            "CHECKPOINT", f"{field} must be a 64-char lowercase hex sha256, got {value!r}"
        )
    return str(value)


def _v3_classes():
    """``(config_cls, model_cls)`` for v3, or ``(None, None)`` when blocked.

    The v3 model is owned by S3-5 and only exists after the G-v3 gate.  Looking it
    up lazily keeps this module importable before that gate; callers then get an
    explicit ``v104:CHECKPOINT`` instead of an ``ImportError`` or a fake pass.
    """
    try:
        from .vmodels3 import VisionSpatialTemporalConfig, VisionSpatialTemporalPolicy
    except ImportError:
        return None, None
    return VisionSpatialTemporalConfig, VisionSpatialTemporalPolicy


def vision_arch(model) -> str:
    """The architecture actually loaded, or ``v104:CHECKPOINT`` for anything else.

    Called on the *model object*, not on the payload: a pre-v1.0.3 payload has no
    ``arch`` key and has to keep loading as v1, while a payload that claims an
    unknown architecture is rejected at load time.  Before v1.0.4 every non-v2
    model was labelled v1, which would silently mislabel a v3 deployment.
    """
    _config_cls, v3_model = _v3_classes()
    if v3_model is not None and isinstance(model, v3_model):
        return VISION_ARCH_V3
    if isinstance(model, VisionScorer):
        return VISION_ARCH_V1
    from .vmodels2 import VisionScorerV2  # local: vmodels2 imports this module

    if isinstance(model, VisionScorerV2):
        return VISION_ARCH_V2
    raise _v104_error(
        "CHECKPOINT",
        f"unknown vision model type {type(model).__name__!r}; "
        f"expected one of {list(KNOWN_VISION_ARCHES)}",
    )


def _model_config(config) -> dict:
    """The config as a plain dict (dataclass instances are the normal case)."""
    if isinstance(config, dict):
        return dict(config)
    values = getattr(config, "__dict__", None)
    if isinstance(values, dict):
        return dict(values)
    raise _v104_error("CHECKPOINT", f"cannot serialise config of type {type(config).__name__!r}")


def _as_hw(value, field: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise _v104_error("CHECKPOINT", f"{field} must be an (H, W) pair, got {value!r}")
    try:
        height, width = int(value[0]), int(value[1])
    except (TypeError, ValueError) as exc:
        raise _v104_error("CHECKPOINT", f"{field} must be integer (H, W), got {value!r}") from exc
    return height, width


def _v3_region_sizes(config: dict) -> dict[str, tuple[int, int]]:
    """``region_sizes`` (or ``model_regions`` / ``*_hw``) of a v3 config, normalised."""
    sizes = config.get("region_sizes")
    if sizes is None and isinstance(config.get("model_regions"), (list, tuple)):
        sizes = {
            str(row[0]): (row[1], row[2])
            for row in config["model_regions"]
            if isinstance(row, (list, tuple)) and len(row) == 3
        }
    if sizes is None:
        # v2-style configs expose ``region_sizes`` as a property and store the three
        # ``*_hw`` fields; accept that layout too so a v3 model following the
        # sibling convention is not rejected for a serialisation detail.
        candidate = {name: config.get(f"{name}_hw") for name in V3_REGION_SIZES}
        if all(value is not None for value in candidate.values()):
            sizes = candidate
    if not isinstance(sizes, dict):
        raise _v104_error(
            "CHECKPOINT", "v3 config must declare region_sizes (playfield/hud_score/hud_status)"
        )
    return {str(name): _as_hw(value, f"region_sizes[{name!r}]") for name, value in sizes.items()}


def validate_v3_payload(payload: dict) -> dict:
    """Validate a ``vision-checkpoint/3`` payload and return its contract record.

    Every check is a C6 requirement: schema/kind/arch, the five identity keys in
    ``extra``, RGB3, depth 3, ``rgb255-v1`` normalisation, ``focus9-move-v1``
    candidate order, the history contract, the native/local grids,
    ``align_corners=False``, ``score_semantics=preference_logits``, the three ROI
    sizes and a single spatial head.  A malformed v3 file must fail *before* the
    game starts, so this runs inside :func:`load_vision_checkpoint`.
    """
    if not isinstance(payload, dict):
        raise _v104_error("CHECKPOINT", f"checkpoint payload is {type(payload).__name__}, not a mapping")
    for key in ("schema", "kind", "arch", "config", "state_dict", "extra"):
        if key not in payload:
            raise _v104_error("CHECKPOINT", f"v3 checkpoint is missing the {key!r} key")
    if payload["schema"] != V3_CHECKPOINT_SCHEMA:
        raise _v104_error(
            "CHECKPOINT", f"v3 checkpoint schema {payload['schema']!r} != {V3_CHECKPOINT_SCHEMA!r}"
        )
    if payload["kind"] != "vision_scorer":
        raise _v104_error("CHECKPOINT", f"v3 checkpoint kind {payload['kind']!r} != 'vision_scorer'")
    if payload["arch"] != VISION_ARCH_V3:
        raise _v104_error("CHECKPOINT", f"v3 checkpoint arch {payload['arch']!r} != {VISION_ARCH_V3!r}")
    config = payload["config"]
    if not isinstance(config, dict):
        raise _v104_error("CHECKPOINT", "v3 checkpoint config must be a mapping")
    extra = payload["extra"]
    if not isinstance(extra, dict):
        raise _v104_error("CHECKPOINT", "v3 checkpoint extra must be a mapping")
    if not dict(payload["state_dict"]):
        raise _v104_error("CHECKPOINT", "v3 checkpoint state_dict is empty")

    # --- extra identity ---------------------------------------------------
    missing = [key for key in V3_REQUIRED_EXTRA if key not in extra]
    if missing:
        raise _v104_error(
            "CHECKPOINT", f"v3 checkpoint extra is missing {missing}; required {list(V3_REQUIRED_EXTRA)}"
        )
    source_sha = extra["source_state_sha"]
    if not isinstance(source_sha, str):
        raise _v104_error(
            "CHECKPOINT", f"extra.source_state_sha must be a string, got {type(source_sha).__name__}"
        )
    if source_sha:
        _sha256_field(source_sha, "extra.source_state_sha")
    stage = str(extra.get("stage", ""))
    if not source_sha and "policy" in stage.lower():
        raise _v104_error(
            "CHECKPOINT",
            f"policy-stage v3 checkpoint must record a real source_state_sha, got '' (stage={stage!r})",
        )
    for field in ("manifest_sha", "val_index_sha256"):
        _sha256_field(extra[field], f"extra.{field}")
    if not isinstance(extra["teacher_fingerprint"], (str, dict)) or not extra["teacher_fingerprint"]:
        raise _v104_error("CHECKPOINT", "extra.teacher_fingerprint must be a non-empty string or mapping")
    if not isinstance(extra["loss_config"], dict) or not extra["loss_config"]:
        raise _v104_error("CHECKPOINT", "extra.loss_config must be a non-empty mapping")

    # --- model contract ---------------------------------------------------
    depth = config.get("depth")
    if depth is None or int(depth) != int(V3_CONTRACT["depth"]):
        raise _v104_error(
            "CHECKPOINT", f"v3 depth {depth!r} != {V3_CONTRACT['depth']} (this round is depth 3 only)"
        )
    if int(depth) != int(extra.get("depth", depth)):
        raise _v104_error(
            "CHECKPOINT", f"v3 config depth {depth!r} != extra.depth {extra.get('depth')!r}"
        )
    channels = config.get("channels", config.get("in_channels", V3_CONTRACT["channels"]))
    if int(channels) != int(V3_CONTRACT["channels"]):
        raise _v104_error("CHECKPOINT", f"v3 channels {channels!r} != 3 (RGB only)")
    normalization = config.get("normalization", V3_CONTRACT["normalization"])
    if normalization != V3_CONTRACT["normalization"]:
        raise _v104_error(
            "CHECKPOINT", f"v3 normalization {normalization!r} != {V3_CONTRACT['normalization']!r}"
        )
    order = config.get("candidate_order_version")
    if order != V3_CONTRACT["candidate_order_version"]:
        raise _v104_error(
            "CHECKPOINT",
            f"v3 candidate_order_version {order!r} != {V3_CONTRACT['candidate_order_version']!r}",
        )
    semantics = config.get("score_semantics")
    if semantics != V3_CONTRACT["score_semantics"]:
        raise _v104_error(
            "CHECKPOINT", f"v3 score_semantics {semantics!r} != {V3_CONTRACT['score_semantics']!r}"
        )
    history_contract = config.get("history_contract")
    if history_contract not in HISTORY_CONTRACTS:
        raise _v104_error(
            "CHECKPOINT",
            f"v3 history_contract {history_contract!r} not in {list(HISTORY_CONTRACTS)}",
        )
    roi = _as_hw(config.get("roi_native_hw"), "config.roi_native_hw")
    if roi != V3_CONTRACT["roi_native_hw"]:
        raise _v104_error(
            "CHECKPOINT", f"v3 roi_native_hw {roi} != {V3_CONTRACT['roi_native_hw']}"
        )
    grid = _as_hw(config.get("local_grid_hw"), "config.local_grid_hw")
    if grid != V3_CONTRACT["local_grid_hw"]:
        raise _v104_error(
            "CHECKPOINT", f"v3 local_grid_hw {grid} != {V3_CONTRACT['local_grid_hw']}"
        )
    if config.get("align_corners") is not False:
        raise _v104_error(
            "CHECKPOINT", f"v3 align_corners {config.get('align_corners')!r} must be False"
        )
    if "state_head" in config and config["state_head"] is not True:
        raise _v104_error("CHECKPOINT", "v3 state_head must be enabled (single spatial head)")
    if "local_enabled" in config and not isinstance(config["local_enabled"], bool):
        raise _v104_error("CHECKPOINT", "v3 local_enabled must be a bool")
    sizes = _v3_region_sizes(config)
    if sizes != V3_REGION_SIZES:
        raise _v104_error(
            "CHECKPOINT", f"v3 region_sizes {sizes} != {V3_REGION_SIZES}"
        )
    frame = payload.get("frame")
    if isinstance(frame, dict) and "C" in frame and int(frame["C"]) != int(V3_CONTRACT["channels"]):
        raise _v104_error("CHECKPOINT", f"v3 frame.C {frame['C']!r} != 3")
    return {
        "schema": V3_CHECKPOINT_SCHEMA,
        "arch": VISION_ARCH_V3,
        "depth": int(depth),
        **{key: value for key, value in V3_CONTRACT.items() if key != "depth"},
        "history_contract": str(history_contract),
        "region_sizes": {name: [int(h), int(w)] for name, (h, w) in sizes.items()},
        "legacy_frame": dict(frame) if isinstance(frame, dict) else {},
        "identity": {
            key: (extra[key] if isinstance(extra[key], (str, int, float)) else extra[key])
            for key in V3_REQUIRED_EXTRA
        },
    }


def vision_contract_node(model, extra: dict | None = None) -> dict:
    """Contract metadata for ``eval-report.json``; only v3 calls this (C6).

    v1/v2 reports keep their existing key set - this node exists so a v3
    deployment records *why* its pixels may be trusted, not just its shape.
    """
    arch = vision_arch(model)
    if arch != VISION_ARCH_V3:
        raise _v104_error(
            "CHECKPOINT", f"vision_contract_node is v3-only; model arch is {arch!r}"
        )
    contract = validate_v3_payload(
        {
            "schema": V3_CHECKPOINT_SCHEMA,
            "kind": "vision_scorer",
            "arch": arch,
            "config": _model_config(model.config),
            "state_dict": model.state_dict(),
            "extra": dict(extra or {}),
        }
    )
    contract.pop("state_dict", None)
    return contract


@dataclass
class VisionNetConfig:
    depth: int = 3
    width: int = 32  # channels of the first conv block
    trunk_hidden: int = 256
    candidate_hidden: int = 64
    state_hidden: int = 256
    state_head: bool = True
    danger_cells: int = obs.DANGER_ROWS * obs.DANGER_COLS
    # Model input size per screen region.  The archive keeps native resolution
    # (playfield 448x384, hud_score 34x216, hud_status 126x216); the model trades
    # that down for compute, but the *HUD stays legible* because those regions are
    # already small.  Tuple order is (H, W).
    model_regions: tuple[tuple[str, int, int], ...] = (
        ("playfield", 224, 192),
        ("hud_score", 34, 216),
        ("hud_status", 126, 216),
    )

    @property
    def region_names(self) -> tuple[str, ...]:
        return tuple(name for name, _h, _w in self.model_regions)

    @property
    def region_sizes(self) -> dict[str, tuple[int, int]]:
        return {name: (height, width) for name, height, width in self.model_regions}


class VisionEncoder(nn.Module):
    """Per-region CNN with shared weights, then a trunk.

    Weights are shared across regions on purpose: a region is just "an image", and
    sharing keeps the parameter count small enough that a handful of small HUD
    streams cost almost nothing.  Each region is pooled to a fixed 2x2 before
    concatenation so the trunk is independent of the input sizes.
    """

    def __init__(self, config: VisionNetConfig | None = None):
        super().__init__()
        self.config = config or VisionNetConfig()
        width = self.config.width
        self.body = nn.Sequential(
            nn.Conv2d(vision.VISION_CHANNELS * self.config.depth, width, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 3, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 3, width * 4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 4 * 4 * len(self.config.region_names), self.config.trunk_hidden),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _as_batched(region: torch.Tensor) -> tuple[torch.Tensor, int]:
        if region.ndim == 4:  # (depth, H, W, C) -> (1, depth, H, W, C)
            region = region.unsqueeze(0)
        batch, depth, height, width, channels = region.shape
        return region.permute(0, 1, 4, 2, 3).reshape(batch, depth * channels, height, width), batch

    def forward(self, regions: dict[str, torch.Tensor]) -> torch.Tensor:
        missing = set(self.config.region_names) - set(regions)
        if missing:
            raise KeyError(f"missing regions {sorted(missing)}; got {sorted(regions)}")
        pooled = []
        for name in self.config.region_names:
            x, _batch = self._as_batched(regions[name])
            features = self.body(x)
            pooled.append(torch.nn.functional.adaptive_avg_pool2d(features, (2, 2)).flatten(1))
        return self.trunk(torch.cat(pooled, dim=-1))


class VisionStateDecoder(nn.Module):
    """Privileged-state head: features -> self_feat (24) + danger map (128)."""

    def __init__(self, config: VisionNetConfig | None = None):
        super().__init__()
        self.config = config or VisionNetConfig()
        self.net = nn.Sequential(
            nn.Linear(self.config.trunk_hidden, self.config.state_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.state_hidden, obs.N_SELF + self.config.danger_cells),
        )
        self.n_self = obs.N_SELF
        self.danger_cells = self.config.danger_cells

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.net(features)
        return {
            "self_feat": out[:, : self.n_self],
            # Danger is a *count* of bullets per cell (0..~30 after the /8 scaling
            # used elsewhere); predicting log1p keeps the loss from being
            # dominated by the few crowded cells.
            "danger_log": out[:, self.n_self :],
        }


class VisionScorer(nn.Module):
    """Candidate-conditioned policy over pixels.

    ``score(frames)`` -> [B, 18]; ``act`` takes the argmax.  The candidate set is
    the same 18 (focus, direction) pairs the privileged teacher scores, so a
    teacher value array can be used as the regression target verbatim.
    """

    def __init__(self, config: VisionNetConfig | None = None):
        super().__init__()
        self.config = config or VisionNetConfig()
        self.encoder = VisionEncoder(self.config)
        self.candidate_embed = nn.Embedding(N_CANDIDATES, self.config.candidate_hidden)
        self.score_head = nn.Sequential(
            nn.Linear(self.config.trunk_hidden + self.config.candidate_hidden, self.config.candidate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.candidate_hidden, 1),
        )
        self.state_decoder = VisionStateDecoder(self.config) if self.config.state_head else None

    def features(self, regions: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder(regions)

    def score(self, regions: dict[str, torch.Tensor], features: torch.Tensor | None = None) -> torch.Tensor:
        features = self.features(regions) if features is None else features
        index = torch.arange(N_CANDIDATES, device=features.device)
        embedded = self.candidate_embed(index)
        fused = torch.cat(
            [
                features.unsqueeze(1).expand(-1, N_CANDIDATES, -1),
                embedded.unsqueeze(0).expand(features.size(0), -1, -1),
            ],
            dim=-1,
        )
        return self.score_head(fused).squeeze(-1)

    def decode_state(self, regions: dict[str, torch.Tensor], features: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if self.state_decoder is None:
            raise RuntimeError("this model was built with state_head=False")
        features = self.features(regions) if features is None else features
        return self.state_decoder(features)

    def act(self, regions: dict[str, torch.Tensor], deterministic: bool = True):
        """Greedy action tuple ``[move, shoot, focus, bomb]`` from pixels.

        Shooting is always on and bombing is left to a rule (the teacher's value
        only ranks movement/focus), which mirrors how the privileged scorer acts.
        """
        values = self.score(regions)
        best = values.argmax(dim=-1)
        move = best % spec.ACTION_DIMS[0]
        focus = best // spec.ACTION_DIMS[0]
        ones = torch.ones_like(move)
        zeros = torch.zeros_like(move)
        actions = torch.stack([move, ones, focus, zeros], dim=-1)
        zero = torch.zeros(values.size(0), dtype=values.dtype, device=values.device)
        return actions, zero, values.max(dim=-1).values, zero

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def save_vision_checkpoint(path, model: VisionScorer, extra: dict | None = None) -> None:
    """Serialise a vision checkpoint; the arch is taken from the model type.

    v1/v2 keep the v1.0.3 payload shape verbatim (a v2 config is never read as a
    v1 one because ``arch`` says which), while v3 writes the explicit
    ``vision-checkpoint/3`` payload and validates its identity *before* the file
    is written: a checkpoint whose ``extra`` is missing the teacher/manifest/val
    fingerprints cannot be saved.
    """
    arch = vision_arch(model)
    config = _model_config(model.config)
    frame = {
        "H": FRAME_LEGACY_H,
        "W": FRAME_LEGACY_W,
        "C": vision.VISION_CHANNELS,
        "depth": int(model.config.depth),
    }
    if arch == VISION_ARCH_V3:
        payload = {
            "schema": V3_CHECKPOINT_SCHEMA,
            "kind": "vision_scorer",
            "arch": arch,
            "config": config,
            # ``frame`` is the historical H128/W112 marker only: the real input
            # sizes are ``config.region_sizes`` (C6).
            "frame": frame,
            "state_dict": model.state_dict(),
            "extra": dict(extra or {}),
        }
        validate_v3_payload(payload)
        torch.save(payload, path)
        return
    torch.save(
        {
            "kind": "vision_scorer",
            # Inferred from the model type, so callers keep the v1 signature and
            # a v2 config (which has different fields) is never read as a v1 one.
            "arch": arch,
            "state_dict": model.state_dict(),
            "config": config,
            "frame": frame,
            "extra": extra or {},
        },
        path,
    )


def _v3_config_from_payload(config_cls, config: dict):
    """Build ``VisionSpatialTemporalConfig`` from a stored mapping (lists -> tuples)."""
    values = dict(config)
    for field in ("roi_native_hw", "local_grid_hw"):
        if values.get(field) is not None:
            values[field] = tuple(values[field])
    if isinstance(values.get("region_sizes"), dict):
        values["region_sizes"] = {
            name: tuple(size) for name, size in values["region_sizes"].items()
        }
    if isinstance(values.get("model_regions"), (list, tuple)):
        values["model_regions"] = tuple(
            tuple(row) if isinstance(row, (list, tuple)) else row for row in values["model_regions"]
        )
    try:
        return config_cls(**values)
    except TypeError as exc:
        raise _v104_error(
            "CHECKPOINT", f"v3 config does not match VisionSpatialTemporalConfig: {exc}"
        ) from exc


def load_vision_checkpoint(path, map_location: str = "cpu") -> tuple[nn.Module, dict]:
    """Load a v1/v2/v3 vision checkpoint with strict weights.

    Dispatch is explicit: a payload without ``arch`` is a pre-v1.0.3 v1 file, an
    unknown ``arch`` is a ``v104:CHECKPOINT`` error (never silently v1), and a
    ``v3-st`` payload is validated field by field.  The v3 branch is *blocked*
    while ``touhou_rl/vmodels3.py`` does not exist (G-v3 not approved): loading
    such a file reports that instead of pretending it worked.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise _v104_error(
            "CHECKPOINT", f"{path}: checkpoint payload is {type(payload).__name__}, not a mapping"
        )
    if payload.get("kind") != "vision_scorer":
        raise _v104_error(
            "CHECKPOINT",
            f"{path}: not a vision checkpoint (kind={payload.get('kind')!r}, expected 'vision_scorer')",
        )
    # Checkpoints written before v1.0.3 carry no arch key; they are all v1.
    arch = payload.get("arch") or VISION_ARCH_V1
    if arch not in KNOWN_VISION_ARCHES:
        raise _v104_error(
            "CHECKPOINT", f"{path}: unknown vision arch {arch!r}; expected one of {list(KNOWN_VISION_ARCHES)}"
        )
    if arch == VISION_ARCH_V3:
        contract = validate_v3_payload(payload)
        config_cls, model_cls = _v3_classes()
        if model_cls is None:
            raise _v104_error(
                "CHECKPOINT",
                f"{path}: v3 branch blocked - touhou_rl/vmodels3.py is not implemented and "
                "G-v3 is not approved; refusing to load a v3-st checkpoint",
            )
        config = _v3_config_from_payload(config_cls, payload["config"])
        model = model_cls(config)
        model.load_state_dict(payload["state_dict"], strict=True)
        return model, payload.get("extra", {})
    frame = payload.get("frame", {})
    if frame.get("H") != vision.VISION_H or frame.get("W") != vision.VISION_W:
        raise _v104_error(
            "CHECKPOINT",
            f"{path}: checkpoint frame size {frame.get('H')}x{frame.get('W')} does not match "
            f"this build's {vision.VISION_H}x{vision.VISION_W}",
        )
    config = payload.get("config")
    if not isinstance(config, dict):
        raise _v104_error("CHECKPOINT", f"{path}: checkpoint has no config mapping")
    if "state_dict" not in payload:
        raise _v104_error("CHECKPOINT", f"{path}: checkpoint has no state_dict")
    if arch == VISION_ARCH_V1:
        model = VisionScorer(VisionNetConfig(**config))
    else:
        from .vmodels2 import VisionNetConfigV2, VisionScorerV2

        config = dict(config)
        # json/torch.save may hand the (H, W) pairs back as lists; the config
        # properties assume tuples.
        for field in ("playfield_hw", "hud_score_hw", "hud_status_hw"):
            if config.get(field) is not None:
                config[field] = tuple(config[field])
        model = VisionScorerV2(VisionNetConfigV2(**config))
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload.get("extra", {})
