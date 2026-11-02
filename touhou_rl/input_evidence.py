"""Read-only live-input evidence. Observed GUI masks are not engine hooks."""
from __future__ import annotations
import time
from . import spec

CONTROL_MASK = (spec.BTN_UP | spec.BTN_DOWN | spec.BTN_LEFT | spec.BTN_RIGHT |
                spec.BTN_SHOOT | spec.BTN_FOCUS | spec.BTN_BOMB)


def stamp(state):
    return {"stage": int(state.stage), "frame": int(state.gameplay_frame)}


def sample(reader, capture_pool=False):
    before = reader.read_state(with_bullets=False, with_title=False)
    # A stable frame counter alone is insufficient: the engine may increment it
    # before updating Player. Let that frame's update settle, then re-read.
    time.sleep(.002)
    after = reader.read_state(with_bullets=False, with_title=False)
    pool=None
    if capture_pool:
        field=reader.read_bullets();active=field.active_mask()
        pool=dict(slot_id=__import__('numpy').flatnonzero(active).astype(int).tolist(),
                  position=field.position[active].astype(float).tolist(),velocity=field.velocity[active].astype(float).tolist(),
                  half=__import__('numpy').maximum(field.half_size[active].max(axis=1,initial=.5),.5).astype(float).tolist(),
                  half_xy=field.half_size[active].astype(float).tolist())
        end=reader.read_state(with_bullets=False,with_title=False)
        stable=stamp(before)==stamp(after)==stamp(end)
    else: stable=stamp(before)==stamp(after)
    return after, dict(**stamp(after), bullets=pool, gui_input_current=int(after.gui_input_current),
                      player_position=[float(after.player_x), float(after.player_y)],
                      deaths=int(after.deaths), player_state=int(getattr(after, 'player_state', -1)),
                      read_stable=stable)


def observe_hold(reader, keyboard, keys, requested_buttons, frames, timeout=.8, capture_pool=False):
    """Set keys and observe frameskip *after first readback*, not stale last step.

    First observed match is an upper-bound detection timestamp, not the exact
    instant the engine consumed input. Missing intermediate frames invalidate
    contiguous evidence; never invent observations for gaps.
    """
    before, _ = sample(reader, capture_pool=capture_pool)
    keyboard.set_keys(keys)
    issue_after, first = sample(reader, capture_pool=capture_pool)
    observations, effective, latest = [], None, issue_after
    first_match = None
    deadline = time.perf_counter() + timeout
    while True:
        state, record = (latest, first) if not observations else sample(reader, capture_pool=capture_pool)
        latest = state
        if not observations or (record['stage'], record['frame']) != (observations[-1]['stage'], observations[-1]['frame']):
            observations.append(record)
        elif record['read_stable']:
            observations[-1] = record
        matched = record['read_stable'] and (record['gui_input_current'] & CONTROL_MASK) == (requested_buttons & CONTROL_MASK)
        if matched:
            if first_match is None:
                first_match = {"stage": record['stage'], "frame": record['frame']}
            if effective is None and record['stage'] == first_match['stage'] and record['frame'] >= first_match['frame'] + 1:
                effective = {"stage": record['stage'], "frame": record['frame']}
        elif effective is None and record['read_stable']:
            first_match = None
        if state.supervisor_state != spec.STATE_GAME_MANAGER or state.stage != before.stage:
            break
        if effective is not None and state.gameplay_frame - effective['frame'] >= frames and record['read_stable']:
            break
        if time.perf_counter() >= deadline:
            break
        time.sleep(0)
    selected = [r for r in observations if effective is not None and r['stage'] == effective['stage'] and r['frame'] >= effective['frame']]
    contiguous = bool(selected) and all(r['read_stable'] for r in selected) and all(b['frame'] == a['frame'] + 1 for a,b in zip(selected,selected[1:]))
    matching = bool(selected) and all((r['gui_input_current'] & CONTROL_MASK) == (requested_buttons & CONTROL_MASK) for r in selected)
    positive = effective is not None and latest.stage == effective['stage'] and latest.gameplay_frame > effective['frame']
    evidence = dict(schema='v200-input-observation/1', requested_buttons=int(requested_buttons),
                    issue_before=stamp(before), issue_after=stamp(issue_after), observations=observations,
                    first_match_observed=first_match, effective_first_observed=effective, end=stamp(latest), coverage_contiguous=contiguous,
                    timing_valid=bool(contiguous and matching and positive),
                    semantics='GUI-mask observed then one-frame settling; not exact engine-consumption instant')
    return latest, evidence
