"""HUD scalar labels: the state values the HUD is drawing (P2 §3.2).

The HUD is a *picture* of state, so the label is the state value, not an OCR
guess.  Two facts about this build are treated as data, not hidden:

* ``displayScore`` (what the score row actually renders) is **not exported** by
  the harness.  Only ``globals.score`` (``STATE f4``) is.  The label therefore
  carries ``score=globals.score`` and an explicit
  ``score_display_unavailable`` marker; an audit can measure the lag of the
  rendered roll against it (``audit.hud_lag``) but cannot label the render
  itself.
* ``nextPointItemExtendThreshold`` and ``lastSpellTimeOrbThreshold`` are the
  denominators of the point/time rows and are also not exported, so
  ``point_total`` / ``time_total`` are ``None`` (unavailable), never ``0``.

Magnitude semantics: ``lives_remaining`` and ``bombs_remaining`` are floats on
the wire; the engine draws the ``i32`` conversion, which truncates toward zero,
so the label truncates the same way instead of rounding.
"""

from __future__ import annotations

from typing import Any, Mapping

from touhou_rl.v400.labels.spec import HUD_LAYOUT

#: State key -> HUD field.  A key missing from the decoded state means the
#: engine did not report the value, and the corresponding label is ``None``.
HUD_STATE_KEYS: dict[str, str] = {
    "score": "score",
    "graze": "graze",
    "point": "point_items_collected",
    "time": "time_orbs",
    "lives": "lives_remaining",
    "bombs": "bombs_remaining",
    "power": "player_power",
}

#: Fields whose value is known to be missing from the export (not "zero").
UNAVAILABLE_FIELDS: dict[str, str] = {
    "display_score": (
        "globals.displayScore is not exported by rlharness STATE "
        "(rlharness.cpp TrackedFields); the score row renders it, the label "
        "can only carry globals.score"
    ),
    "point_total": (
        "globals.nextPointItemExtendThreshold is not exported; the point row's "
        "'/total' half cannot be labelled"
    ),
    "time_total": (
        "globals.lastSpellTimeOrbThreshold is not exported; the time row's "
        "'/total' half cannot be labelled"
    ),
}


def _integer(value: Any) -> int | None:
    """``i32`` conversion of a wire float, or ``None`` when the value is absent."""

    if value is None:
        return None
    return int(float(value))


def hud_label(state: Mapping[str, Any]) -> dict[str, Any]:
    """Build the HUD label dict from a decoded state mapping.

    ``state`` may be a full :func:`touhou_rl.v400.labels.state.decode_state`
    dict or any mapping with the same keys.  Every missing key becomes ``None``
    and is listed in ``missing``; nothing is fabricated.
    """

    label: dict[str, Any] = {}
    missing: list[str] = []
    for target, key in HUD_STATE_KEYS.items():
        value = state.get(key)
        if target in ("lives", "bombs", "power"):
            converted = _integer(value)
        else:
            converted = None if value is None else int(value)
        label[target] = converted
        if converted is None:
            missing.append(key)

    # The two "/ total" values are not exported at all, so they are missing by
    # construction and are listed even on a complete state.
    label["point_total"] = None
    label["time_total"] = None
    label["display_score"] = None
    label["unavailable"] = sorted(UNAVAILABLE_FIELDS)
    label["missing"] = missing
    label["complete"] = not missing
    label["sources"] = {
        "score": "STATE f4 = globals.score (displayScore not exported)",
        "graze": "STATE f5 = globals.graze",
        "point": "STATE f17 = globals.pointItemsCollected",
        "time": "STATE f18 = globals.currentTimeOrbs (GetTimeOrbs)",
        "lives": "STATE f7 = FloatBits(globals.livesRemaining), truncated",
        "bombs": "STATE f8 = FloatBits(globals.bombsRemaining), truncated",
        "power": "STATE f9 = FloatBits(globals.playerPower), truncated",
    }
    label["hud_layout"] = {
        name: {
            "x": spec["x"],
            "y": spec["y"],
            "format": spec["format"],
            "source": spec["source"],
        }
        for name, spec in HUD_LAYOUT.items()
    }
    label["text"] = _display_text(label)
    return label


def _display_text(label: Mapping[str, Any]) -> dict[str, str | None]:
    """The strings the HUD would draw for these values (for the overlay)."""

    score = label.get("score")
    power = label.get("power")
    text: dict[str, str | None] = {
        "score": None if score is None else f"{int(score):09d}",
        "lives": None if label.get("lives") is None else f"{int(label['lives'])}",
        "bombs": None if label.get("bombs") is None else f"{int(label['bombs'])}",
        "power": (
            None
            if power is None
            else ("MAX" if int(power) >= 128 else f"{int(power)}")
        ),
        "graze": None if label.get("graze") is None else f"{int(label['graze'])}",
    }
    point = label.get("point")
    text["point"] = None if point is None else f"{int(point)}/?"
    time_orbs = label.get("time")
    text["time"] = None if time_orbs is None else f"{int(time_orbs)}/?"
    return text
