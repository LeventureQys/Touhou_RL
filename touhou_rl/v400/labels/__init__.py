"""P2 label generator: perception labels derived from engine state, never pixels.

The training ground renders a frame and exports its state inside the *same*
engine frame, so pixel ``k`` and state ``k`` are aligned by construction.  Every
label in this package is a function of that exported state only:

* ``player``          — ``PLAYER`` (position, hurtbox, focus, resources);
* ``hud``             — ``STATE`` fields (the truth the HUD is drawing from);
* ``bullet_occupancy``— ``HAZARDS`` lethal AABBs on a fixed playfield grid;
* ``future_danger``   — the P1-validated ``MODEL_ENGINE_PHASE`` geometry applied
  to recorded future hazard frames, with the ship held still;
* ``events``          — v2: real ``EVENTS`` bits for hit/dying/effect/death/bomb
  (``event_source="EVENTS"``) plus state proxies for game_over/extend
  (``event_source="proxy"``); ``stage_clear`` is unavailable (``None``).
  The v1 proxy-only generator is kept as :func:`event_labels` for EXP-P2-001.

No function in this package reads a pixel.  That is the point of P2 §3: the
pixel branch must *learn* these labels, so the labels cannot be derived from
pixels or the exercise would be circular.
"""

from __future__ import annotations

from touhou_rl.v400.labels.danger import (
    DEFAULT_HORIZONS,
    FutureDanger,
    evaluate_future_danger,
)
from touhou_rl.v400.labels.events import (
    EVENT_BITS,
    EVENT_CLASSES,
    EVENT_SOURCES,
    PROXY_CRITERIA,
    decode_event_flags,
    event_labels,
    event_labels_v2,
)
from touhou_rl.v400.labels.hud import HUD_LAYOUT, hud_label
from touhou_rl.v400.labels.occupancy import (
    GridOccupancy,
    decode_occupancy,
    encode_occupancy,
    hazard_occupancy,
)
from touhou_rl.v400.labels.player import player_label
from touhou_rl.v400.labels.spec import (
    GRID_CELL,
    GRID_HEIGHT,
    GRID_WIDTH,
    HORIZON_FRAMES,
    LABEL_VERSION,
    LABEL_VERSION_V1,
    PLAYFIELD_HEIGHT,
    PLAYFIELD_WIDTH,
    SCREEN_ORIGIN,
    playfield_to_screen,
    screen_to_playfield,
)
from touhou_rl.v400.labels.state import (
    FIELD_NAMES,
    PLAYER_STATE_ALIVE,
    PLAYER_STATE_DYING,
    PLAYER_STATE_INVULNERABLE,
    PLAYER_STATE_NAMES,
    PLAYER_STATE_SPAWNING,
    decode_state,
    f32_from_bits,
    player_state_name,
    state_from_reply,
)

__all__ = [
    "DEFAULT_HORIZONS",
    "EVENT_BITS",
    "EVENT_CLASSES",
    "EVENT_SOURCES",
    "FIELD_NAMES",
    "FutureDanger",
    "GRID_CELL",
    "GRID_HEIGHT",
    "GRID_WIDTH",
    "GridOccupancy",
    "HORIZON_FRAMES",
    "HUD_LAYOUT",
    "LABEL_VERSION",
    "LABEL_VERSION_V1",
    "PLAYER_STATE_ALIVE",
    "PLAYER_STATE_DYING",
    "PLAYER_STATE_INVULNERABLE",
    "PLAYER_STATE_NAMES",
    "PLAYER_STATE_SPAWNING",
    "PLAYFIELD_HEIGHT",
    "PLAYFIELD_WIDTH",
    "PROXY_CRITERIA",
    "SCREEN_ORIGIN",
    "decode_event_flags",
    "decode_occupancy",
    "decode_state",
    "encode_occupancy",
    "evaluate_future_danger",
    "event_labels",
    "event_labels_v2",
    "f32_from_bits",
    "hazard_occupancy",
    "hud_label",
    "playfield_to_screen",
    "player_label",
    "player_state_name",
    "screen_to_playfield",
    "state_from_reply",
]
