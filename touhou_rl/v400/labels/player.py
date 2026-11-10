"""Player labels from the ``PLAYER`` reply (P2 §3.3 first bullet).

The label is the engine's own numbers: position, the exact hurtbox the lethal
test uses, focus, power, lives, bombs and ``playerState``.  The hurtbox is
never rebuilt from ``position +/- size``: the engine exports the bounds it
tests, and inventing them would hide exactly the kind of half-width mistake
EXP-P1-004 was written to catch.
"""

from __future__ import annotations

from typing import Any

from touhou_rl.v400.labels.spec import playfield_to_screen
from touhou_rl.v400.labels.state import player_state_name


def player_label(snapshot: Any) -> dict[str, Any]:
    """Turn one ``PLAYER`` snapshot (duck-typed) into the player label dict."""

    missing: list[str] = []
    position = getattr(snapshot, "position", None)
    if position is None:
        missing.append("position")
        x = y = None
    else:
        x, y = float(position[0]), float(position[1])

    hurt_min = getattr(snapshot, "hurtbox_min", None)
    hurt_max = getattr(snapshot, "hurtbox_max", None)
    if hurt_min is None or hurt_max is None:
        missing.extend(
            name
            for name, value in (("hurtbox_min", hurt_min), ("hurtbox_max", hurt_max))
            if value is None
        )
        hurtbox = None
        hurtbox_size = None
    else:
        hurtbox = {
            "min": [float(hurt_min[0]), float(hurt_min[1])],
            "max": [float(hurt_max[0]), float(hurt_max[1])],
        }
        hurtbox_size = [
            float(hurt_max[0]) - float(hurt_min[0]),
            float(hurt_max[1]) - float(hurt_min[1]),
        ]

    state_value = getattr(snapshot, "player_state", None)
    if state_value is None:
        missing.append("player_state")

    def take(name: str) -> Any:
        value = getattr(snapshot, name, None)
        if value is None:
            missing.append(name)
        return value

    lives = take("lives")
    bombs = take("bombs")
    power = take("power")
    focus = take("focus")

    screen = None
    if x is not None and y is not None:
        screen = list(playfield_to_screen(x, y))

    return {
        "x": x,
        "y": y,
        "screen": screen,
        "hurtbox": hurtbox,
        "hurtbox_size": hurtbox_size,
        "focus": None if focus is None else bool(int(focus)),
        "is_youkai": getattr(snapshot, "is_youkai", None),
        "power": None if power is None else float(power),
        "lives": None if lives is None else float(lives),
        "bombs": None if bombs is None else float(bombs),
        "player_state": state_value,
        "player_state_name": player_state_name(state_value),
        "invuln_state": getattr(snapshot, "invuln_state", None),
        "speeds": {
            "normal_axial": getattr(snapshot, "speed_normal_axial", None),
            "normal_diagonal": getattr(snapshot, "speed_normal_diagonal", None),
            "focused_axial": getattr(snapshot, "speed_focused_axial", None),
            "focused_diagonal": getattr(snapshot, "speed_focused_diagonal", None),
        },
        "missing": sorted(set(missing)),
        "complete": not missing,
    }
