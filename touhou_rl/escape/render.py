"""Offline layered replay for v3.7.0 traces (D6).

Two views of the same record, for two different questions:

* :func:`render_text` — the decision table.  What did the controller intend, which
  route did it pick, did a plan exist, when does the exit close.  This is what a
  reviewer reads first.
* :func:`render_png` — one frame, layered: hazards (red boxes), the coarse
  closure map at that instant (dark overlay), the planned route (blue), the
  actually executed trajectory (green) and the current player position (yellow).

Truth and prediction are drawn in different layers/colours on purpose: a pretty
overlay must never be mistaken for evidence that the forecast was right.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import spec
from . import contracts, trace as trace_module

PLAYFIELD = (spec.PLAYFIELD_X, spec.PLAYFIELD_Y, spec.PLAYFIELD_W, spec.PLAYFIELD_H)


def render_text(trace: list[dict], *, max_rows: int = 40) -> str:
    if not trace:
        return "(empty trace)\n"
    first = trace[0]
    lines = [
        f"episode: {first['episode']}",
        f"decisions: {len(trace)}",
        "",
        "| # | frame | intent | route | first action | plan | closure | min clr | fallback |",
        "|---:|---:|---|---|---|---:|---:|---:|---|",
    ]
    rows = trace[-max_rows:]
    for record in rows:
        decision = record["decision_record"]
        plan = record.get("plan") or {}
        forecast = record.get("forecast") or {}
        action = decision.get("accepted_action") or [0, 0, 0, 0]
        lines.append(
            "| {decision} | {frame} | {intent} | {route} | {action} | {status} | {closure} | {clearance} | {fallback} |".format(
                decision=record["decision"],
                frame=record["observation"].get("frame_id"),
                intent=decision.get("intent"),
                route=decision.get("route_id") or "-",
                action=f"[{action[0]},{action[1]},{action[2]},{action[3]}]",
                status=forecast.get("plan_status") or "-",
                closure=plan.get("closure_eta"),
                clearance=plan.get("min_clearance"),
                fallback=decision.get("fallback") or "-",
            )
        )
    summary = trace_module.attribute(trace)
    lines += [
        "",
        f"classification: {summary['classification']}",
        f"hit_frame: {summary['hit_frame']}",
        f"window intents: {summary['window_intents']}",
    ]
    return "\n".join(lines) + "\n"


def render_png(
    trace: list[dict],
    sidecar: dict,
    out_path: Path | str,
    *,
    decision: int | None = None,
    scale: int = 2,
) -> Path:
    from PIL import Image, ImageDraw

    if not trace:
        raise ValueError("empty trace")
    index = (decision if decision is not None else len(trace) - 1) - 1
    index = max(0, min(index, len(trace) - 1))
    record = trace[index]
    width = int(PLAYFIELD[2] * scale)
    height = int(PLAYFIELD[3] * scale)
    image = Image.new("RGBA", (width, height), (250, 250, 245, 255))
    draw = ImageDraw.Draw(image)

    # Hazards at this decision from the sidecar.
    start = int(sidecar["offsets"][index])
    end = int(sidecar["offsets"][index + 1])
    positions = sidecar["positions"][start:end]
    halves = sidecar["halves"][start:end]
    for position, half in zip(positions, halves):
        x0 = (position[0] - half[0] - PLAYFIELD[0]) * scale
        y0 = (position[1] - half[1] - PLAYFIELD[1]) * scale
        x1 = (position[0] + half[0] - PLAYFIELD[0]) * scale
        y1 = (position[1] + half[1] - PLAYFIELD[1]) * scale
        draw.rectangle([x0, y0, x1, y1], fill=(200, 40, 40, 200), outline=(140, 0, 0, 220))

    # Planned route (blue).
    plan = record.get("plan") or {}
    knots = plan.get("position_time_knots") or []
    if knots:
        points = [((knot["x"] - PLAYFIELD[0]) * scale, (knot["y"] - PLAYFIELD[1]) * scale) for knot in knots]
        if len(points) > 1:
            draw.line(points, fill=(30, 80, 220, 230), width=max(2, scale))
        for point in points[:: max(1, len(points) // 6)]:
            draw.ellipse([point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2], fill=(30, 80, 220, 230))

    # Executed trajectory so far (green).
    played = [
        (
            (row["observation"]["player"]["x"] - PLAYFIELD[0]) * scale,
            (row["observation"]["player"]["y"] - PLAYFIELD[1]) * scale,
        )
        for row in trace[: index + 1]
    ]
    if len(played) > 1:
        draw.line(played, fill=(20, 150, 60, 220), width=max(2, scale))

    # Player now (yellow).
    player = record["observation"]["player"]
    px = (player["x"] - PLAYFIELD[0]) * scale
    py = (player["y"] - PLAYFIELD[1]) * scale
    draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=(250, 200, 20, 255), outline=(120, 90, 0, 255))

    decision_record = record["decision_record"]
    text = (
        f"{record['episode']}  decision {record['decision']}  frame {record['observation'].get('frame_id')}\n"
        f"intent={decision_record['intent']}  route={decision_record.get('route_id')}  "
        f"plan={(record.get('forecast') or {}).get('plan_status')}"
    )
    draw.text((6, 6), text, fill=(10, 10, 10, 255))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(out)
    return out


def render_episode(trace_path: Path | str, sidecar_path: Path | str, out_dir: Path | str, *, decision: int | None = None) -> dict:
    trace = trace_module.load_trace(trace_path)
    sidecar = trace_module.load_sidecar(sidecar_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episode = trace[0]["episode"]
    text_path = out_dir / f"{episode}.md"
    text_path.write_text(render_text(trace), encoding="utf-8")
    png_path = render_png(trace, sidecar, out_dir / f"{episode}.png", decision=decision)
    return {
        "episode": episode,
        "text": str(text_path),
        "png": str(png_path),
        "attribution": trace_module.attribute(trace),
    }
