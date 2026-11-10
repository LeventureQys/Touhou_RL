"""Human-inspection overlays for the P2 labels (12-P2 §2.2, P2-WP1 acceptance).

The overlay is the only place pixels and labels meet, and they meet as *ink on
top of the picture*: the labels are already fixed by the state exports, so
drawing them cannot change them.  A reviewer looks at the result to confirm

* the self-ship crosshair and hurtbox sit exactly on the rendered ship,
* the occupancy cells sit on the rendered bullets,
* the HUD label text matches the HUD row next to it,
* the playfield frame sits at ``(32, 16)``.

Requires Pillow; the collector does not import this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from touhou_rl.v400.labels.occupancy import decode_occupancy, occupied_cell_rects
from touhou_rl.v400.labels.spec import (
    GRID_CELL,
    HUD_LAYOUT,
    PLAYFIELD_HEIGHT,
    PLAYFIELD_WIDTH,
    SCREEN_HEIGHT,
    SCREEN_ORIGIN,
    SCREEN_WIDTH,
    playfield_rect_on_screen,
    playfield_to_screen,
)

GRID_RGBA = (60, 255, 90, 46)
GRID_LINE_RGBA = (120, 255, 150, 18)
BULLET_RGBA = (40, 255, 60, 120)
ENEMY_RGBA = (255, 170, 40, 120)
LASER_RGBA = (255, 60, 255, 120)
PLAYFIELD_RGBA = (255, 255, 255, 180)
POINT_RGBA = (255, 40, 40, 255)
HURTBOX_RGBA = (255, 255, 255, 230)
HUD_ANCHOR_RGBA = (0, 220, 255, 255)


def _font(size: int = 11) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no size argument
        return ImageFont.load_default()


def load_png(path: str | Path) -> np.ndarray:
    """Read one lossless PNG back as ``uint8[H, W, 4]``."""

    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA"), dtype=np.uint8)


def render_overlay(
    pixels: np.ndarray,
    record: dict[str, Any],
    *,
    origin: tuple[int, int] = SCREEN_ORIGIN,
) -> Image.Image:
    """Draw the label overlay for one frame onto its own pixels."""

    base = Image.fromarray(np.ascontiguousarray(pixels), mode="RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font = _font(11)
    small = _font(10)

    x0, y0, x1, y1 = playfield_rect_on_screen(origin)
    draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=PLAYFIELD_RGBA, width=1)

    occupancy = record.get("occupancy")
    if occupancy:
        grid = decode_occupancy(occupancy)
        # Faint grid so "the occupancy grid" is visible even where empty.
        for col in range(0, grid.width + 1, 8):
            sx = x0 + col * GRID_CELL
            draw.line([(sx, y0), (sx, y1 - 1)], fill=GRID_LINE_RGBA)
        for row in range(0, grid.height + 1, 8):
            sy = y0 + row * GRID_CELL
            draw.line([(x0, sy), (x1 - 1, sy)], fill=GRID_LINE_RGBA)

        other = grid.occupied & ~grid.bullet_occupied
        for cx0, cy0, cx1, cy1 in occupied_cell_rects(grid, grid.bullet_occupied):
            draw.rectangle(
                [x0 + cx0, y0 + cy0, x0 + cx1 - 1, y0 + cy1 - 1],
                fill=BULLET_RGBA,
            )
        for cx0, cy0, cx1, cy1 in occupied_cell_rects(grid, other):
            draw.rectangle(
                [x0 + cx0, y0 + cy0, x0 + cx1 - 1, y0 + cy1 - 1],
                fill=ENEMY_RGBA,
            )
        if occupancy.get("laser_cells"):
            for cx0, cy0, cx1, cy1 in occupied_cell_rects(grid):
                draw.rectangle(
                    [x0 + cx0, y0 + cy0, x0 + cx1 - 1, y0 + cy1 - 1],
                    outline=LASER_RGBA,
                )

    player = record.get("player") or {}
    px, py = player.get("x"), player.get("y")
    if px is not None and py is not None:
        sx, sy = playfield_to_screen(px, py, origin)
        draw.line([(sx - 7, sy), (sx + 7, sy)], fill=POINT_RGBA, width=1)
        draw.line([(sx, sy - 7), (sx, sy + 7)], fill=POINT_RGBA, width=1)
        draw.ellipse([sx - 1, sy - 1, sx + 1, sy + 1], fill=POINT_RGBA)
        hurtbox = player.get("hurtbox") or {}
        if hurtbox.get("min") and hurtbox.get("max"):
            hx0, hy0 = playfield_to_screen(hurtbox["min"][0], hurtbox["min"][1], origin)
            hx1, hy1 = playfield_to_screen(hurtbox["max"][0], hurtbox["max"][1], origin)
            draw.rectangle(
                [hx0 - 1, hy0 - 1, hx1 + 1, hy1 + 1], outline=HURTBOX_RGBA, width=1
            )

    # HUD anchors (where the label values are supposed to be drawn).
    for name, spec in HUD_LAYOUT.items():
        draw.ellipse(
            [spec["x"] - 2, spec["y"] + 4, spec["x"] + 2, spec["y"] + 8],
            outline=HUD_ANCHOR_RGBA,
        )

    hud = record.get("hud") or {}
    lines = ["HUD label = state truth (12-P2 §3.2)"]
    text = hud.get("text") or {}
    lines.append(f"score {text.get('score', '-')}")
    lines.append(
        f"lives {text.get('lives', '-')}   bombs {text.get('bombs', '-')}"
    )
    lines.append(f"power {text.get('power', '-')}   graze {text.get('graze', '-')}")
    lines.append(f"point {text.get('point', '-')}   time {text.get('time', '-')}")
    danger = record.get("future_danger") or {}
    would = danger.get("would_die")
    first = danger.get("first_hit_offset")
    gap = danger.get("min_gap")
    lines.append(
        "danger16 "
        + ("None" if would is None else ("YES" if would else "no"))
        + (f" @+{first}f" if first else "")
        + ("" if gap is None else f" gap {gap:.2f}px")
    )
    events = record.get("events") or {}
    death = (events.get("death") or {}).get("value")
    lines.append(
        "death_proxy " + ("None" if death is None else ("YES" if death else "no"))
        + "  (event_source=proxy)"
    )
    panel_x, panel_y = 470, 214
    line_h = 13
    panel_h = line_h * len(lines) + 8
    draw.rectangle(
        [panel_x - 4, panel_y - 4, SCREEN_WIDTH - 1, panel_y + panel_h],
        fill=(0, 0, 0, 170),
    )
    for index, line in enumerate(lines):
        draw.text((panel_x, panel_y + index * line_h), line, fill=(230, 230, 230, 255), font=small)

    # Footer: identity of the frame.
    footer = (
        f"run {record.get('run')} frame {record.get('frame')} "
        f"stage {record.get('stage')} diff {record.get('difficulty')} "
        f"policy {record.get('policy')} mask {record.get('mask')}"
    )
    draw.rectangle([0, SCREEN_HEIGHT - 14, SCREEN_WIDTH - 1, SCREEN_HEIGHT - 1], fill=(0, 0, 0, 190))
    draw.text((4, SCREEN_HEIGHT - 13), footer, fill=(220, 220, 120, 255), font=small)
    draw.text(
        (4, y0 - 13),
        f"screen = playfield + {origin}   playfield {PLAYFIELD_WIDTH}x{PLAYFIELD_HEIGHT}",
        fill=(255, 255, 180, 255),
        font=small,
    )

    return Image.alpha_composite(base, layer)


def render_montage(
    overlays: Sequence[Image.Image],
    *,
    columns: int = 5,
    scale: float = 0.5,
) -> Image.Image:
    """Tile the overlays into one contact sheet (default 50% size)."""

    if not overlays:
        raise ValueError("no overlays to tile")
    columns = max(1, int(columns))
    rows = (len(overlays) + columns - 1) // columns
    tw = max(1, int(round(overlays[0].size[0] * scale)))
    th = max(1, int(round(overlays[0].size[1] * scale)))
    sheet = Image.new("RGB", (columns * tw, rows * th), (16, 16, 16))
    for index, image in enumerate(overlays):
        thumb = image.convert("RGB").resize((tw, th), Image.LANCZOS)
        sheet.paste(thumb, ((index % columns) * tw, (index // columns) * th))
    return sheet


def crop_band(pixels: np.ndarray, band: Iterable[int]) -> np.ndarray:
    """Crop one HUD band out of a frame (used by the lag audit)."""

    x0, y0, x1, y1 = band
    return pixels[y0:y1, x0:x1]
