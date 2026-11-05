"""Offline, fail-closed F2 replay-demonstration contract (not a safety oracle).

F2 is an aimed stream, not a gap. Only the human's *actual*, decoder-aligned
18-way candidate is supervised. This module neither collects data nor trains,
nor establishes G3. SHA bindings establish identity, not reviewer honesty: the
independent pixel annotation and its reviewer still require external review.
All coordinates are capture-image pixels; tolerances are supplied explicitly.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
from collections.abc import Mapping

from .v380.replay import decode_records, align_masks, ReplayError
from .v380.live import candidate_from_mask

SCHEMA = "f2-replay-demonstration/1"
HEADS = ("option", "target", "termination", "slack", "local")
NULL_TARGETS = ("deadline", "route", "gap", "gap_center", "relative_dx")


def evidence_sha(value):
    """Canonical SHA for JSON evidence, excluding no fields implicitly."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _sha(x):
    return isinstance(x, str) and re.fullmatch(r"[0-9a-f]{64}", x) is not None


def _int(x):
    return type(x) is int and x >= 0


def _number(x):
    return type(x) in (float, int) and math.isfinite(x)


def _point(x, width, height):
    return (isinstance(x, (list, tuple)) and len(x) == 2
            and all(_number(v) for v in x)
            and 0 <= x[0] < width and 0 <= x[1] < height)


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def aim_miss(origin, velocity, aim):
    """Perpendicular distance from ``aim`` to the ray through ``origin``.

    This -- and not an angle -- is the quantity an F2 claim is about: a shot is
    aimed at the player if its straight path passes through the player's hitbox.
    ``|cross|/|aim-origin|`` is the *cross-track speed* of the velocity against
    the aim direction: it shares units with the bullet speed, so a tolerance of
    the same order as the speed accepts almost any direction (a 2.5 px/frame
    bound against a 2-4 px/frame bullet allows tens of degrees of error).
    ``|cross|/|velocity|`` has units of pixels and is what the reviewer claims.

    Returns ``None`` for a degenerate (zero) velocity; the caller must reject it.
    """
    speed = math.hypot(velocity[0], velocity[1])
    if not speed > 0:
        return None
    dx, dy = aim[0] - origin[0], aim[1] - origin[1]
    return abs(dx * velocity[1] - dy * velocity[0]) / speed


def aim_ahead(origin, velocity, aim):
    """True when the annotated player is in front of the emission, not behind."""
    return ((aim[0] - origin[0]) * velocity[0]
            + (aim[1] - origin[1]) * velocity[1]) > 0


def audit_observation(observation):
    """Return errors for causal image-bound observations, never teacher inputs."""
    errors = []
    if not isinstance(observation, Mapping):
        return ["observation must be a mapping"]
    o = observation
    frame = o.get("frame")
    if not _int(frame):
        errors.append("frame must be a nonnegative integer")
    through = o.get("observed_through")
    if not _int(through) or not _int(frame) or through > frame:
        errors.append("observed_through must be <= frame")
    if o.get("units") != "image_px":
        errors.append("units must be image_px")
    size = o.get("image_size")
    if not isinstance(size, (list, tuple)) or len(size) != 2 or not all(_int(v) and v > 0 for v in size):
        return errors + ["image_size must be positive integer [width,height]"]
    w, h = size
    if not _point(o.get("player_px"), w, h):
        errors.append("invalid player_px")
    source = o.get("source", {})
    if not isinstance(source, Mapping):
        source = {}
    for key in ("capture", "proof"):
        item = source.get(key, {})
        if not isinstance(item, Mapping) or not item.get("id") or not _sha(item.get("sha256")):
            errors.append(f"missing {key} identity/SHA")
    images = o.get("images", [])
    image_frames = {}
    if not isinstance(images, list):
        images = []
    for image in images:
        if not isinstance(image, Mapping):
            errors.append("invalid image binding")
            continue
        f = image.get("frame")
        if not _int(f) or not _int(through) or f > through or not _sha(image.get("sha256")) or f in image_frames:
            errors.append("invalid/future/duplicate image frame or SHA")
        else:
            image_frames[f] = image["sha256"]
    if frame not in image_frames:
        errors.append("current image SHA is required")
    tracks = o.get("bullet_tracks")
    if not isinstance(tracks, list) or not tracks:
        return errors + ["visible bullet tracks required"]
    ids = set()
    for track in tracks:
        if not isinstance(track, Mapping) or not isinstance(track.get("id"), str) or not track["id"] or track["id"] in ids:
            errors.append("invalid/duplicate track id")
            continue
        ids.add(track["id"])
        samples = track.get("samples")
        if not isinstance(samples, list) or not samples:
            errors.append("track needs visible samples")
            continue
        previous = -1
        for sample in samples:
            if not isinstance(sample, Mapping):
                errors.append("invalid track sample")
                continue
            f = sample.get("frame")
            if (not _int(f) or f <= previous or f not in image_frames
                    or sample.get("image_sha256") != image_frames.get(f)
                    or sample.get("visible") is not True or not _point(sample.get("xy"), w, h)):
                errors.append("invalid/future/nonvisible track sample or image binding")
            if _int(f):
                previous = f
    return errors


def validate_event_annotation(annotation, observation, tolerances=None):
    """Check structured independent pixel evidence against privileged tracks.

    Returns errors, not a trust score. A literal ``verified`` or ``family=F2``
    has no authority. Annotation carries exact observation digest, independent
    reviewer identity, pixel samples, separately sourced teacher samples, and
    repeated aimed-emission evidence. No future pixels/teacher points accepted.

    The aim claim is a *miss distance* in pixels (``aim_miss_px``): the
    perpendicular distance from the annotated player position to the privileged
    bullet path must fit inside the player hitbox. The earlier criterion compared
    a cross-track *speed* with a speed-sized tolerance, which accepts tens of
    degrees of error and therefore proves nothing; it is gone. The aim frame is
    bounded by ``aim_lookback_frames`` so that a review cannot hunt through the
    past for a player position that happens to sit near a stray bullet, and every
    emission must declare the ``miss_px`` it measured.
    """
    errors = audit_observation(observation)
    if errors:
        return errors
    if not isinstance(annotation, Mapping):
        return ["independent visual annotation missing; requires_review"]
    a, o = annotation, observation
    try:
        bound = evidence_sha(o)
    except (ValueError, TypeError):
        return ["observation is not finite JSON"]
    if a.get("observation_sha256") != bound:
        errors.append("annotation observation binding mismatch")
    reviewer = a.get("reviewer", {})
    teacher = a.get("teacher", {})
    if not isinstance(reviewer, Mapping) or not isinstance(teacher, Mapping):
        return errors + ["independent reviewer/teacher metadata missing"]
    if (not reviewer.get("id") or reviewer.get("method") != "independent_pixel_annotation"
            or not teacher.get("producer_id") or reviewer.get("id") == teacher.get("producer_id")
            or not _sha(teacher.get("artifact_sha256"))):
        errors.append("independent pixel reviewer and distinct teacher provenance required")
    t = tolerances if tolerances is not None else a.get("tolerances", {})
    if not isinstance(t, Mapping) or any(
            not _number(t.get(k)) or t[k] < 0
            for k in ("position_px", "aim_cross_px_per_frame", "aim_miss_px")):
        return errors + ["explicit finite nonnegative tolerances required"]
    # The aim frame must be the emission frame or at most a couple of stage
    # frames earlier.  Searching a wider past window for "the player position
    # whose distance to this line is smallest" manufactures aimed shots out of
    # stray bullets: with seven candidates the minimum miss falls well below the
    # tolerance for trajectories that were never aimed anywhere.  The bound is
    # declared, so a review cannot silently widen it.
    lookback = t.get("aim_lookback_frames")
    if not _int(lookback):
        return errors + ["explicit nonnegative aim lookback required"]
    # A row belongs to an aimed *stream* only while the cited emissions are part
    # of the current stream.  Without this bound a row could be justified by a
    # volley fired seconds earlier whose bullets happen to still be on screen,
    # i.e. by a scene that no longer shows aimed fire.
    recency = t.get("emission_recency_frames")
    if not _int(recency):
        return errors + ["explicit nonnegative emission recency required"]
    if a.get("family") != "F2" or a.get("event_type") != "aimed_stream":
        errors.append("only independently annotated F2 aimed_stream is applicable")
    w, h = o["image_size"]
    pixels = a.get("pixel_tracks")
    privileged = teacher.get("tracks")
    # Require complete sample coverage; no verified-string shortcut.
    if not isinstance(pixels, list) or not isinstance(privileged, list):
        return errors + ["pixel/privileged track samples missing; requires_review"]
    expected = {(tr["id"], s["frame"]): s for tr in o["bullet_tracks"] for s in tr["samples"]}
    images_by_frame = {im["frame"]: im["sha256"] for im in o["images"]}

    def indexed(tracks, name):
        # The pixel side must cover exactly the observed samples.  The teacher
        # side must *reach back to the emission frame*, which is often a few
        # frames before the first sample a sprite is separable from its muzzle
        # flash, so it may carry samples the observation does not, as long as
        # they are at frames the observation cites as images.
        out = {}
        for tr in tracks:
            if not isinstance(tr, Mapping) or not isinstance(tr.get("samples"), list):
                errors.append(f"invalid {name} track")
                continue
            for s in tr["samples"]:
                if not isinstance(s, Mapping) or not isinstance(tr.get("id"), str) or not _int(s.get("frame")):
                    errors.append(f"invalid {name} sample")
                    continue
                key = (tr["id"], s["frame"])
                allowed = key in expected if name == "pixel" else s["frame"] in images_by_frame
                if key in out or not allowed or not _point(s.get("xy"), w, h):
                    errors.append(f"invalid/future/duplicate {name} sample")
                else:
                    out[key] = s
        return out

    p, q = indexed(pixels, "pixel"), indexed(privileged, "teacher")
    if set(p) != set(expected):
        errors.append("visual coverage must exactly match observed tracks")
    teacher_frames = {}
    for track_id, frame in q:
        teacher_frames.setdefault(track_id, set()).add(frame)
    for track_id in {key[0] for key in expected}:
        if not teacher_frames.get(track_id):
            errors.append("teacher track missing for an observed track")
        elif not {key[1] for key in expected if key[0] == track_id} <= teacher_frames[track_id]:
            errors.append("teacher track must cover every observed pixel sample")
    for key in set(p) & set(q) & set(expected):
        if p[key].get("image_sha256") != expected[key]["image_sha256"]:
            errors.append("pixel annotation image SHA mismatch")
        if (_distance(p[key]["xy"], q[key]["xy"]) > t["position_px"]
                or _distance(p[key]["xy"], expected[key]["xy"]) > t["position_px"]):
            errors.append("pixel/teacher/observation position tolerance exceeded")
    emissions = a.get("emissions")
    if not isinstance(emissions, list) or len(emissions) < 2:
        return errors + ["repeated visible aimed emissions required; requires_review"]
    frames, emitters = [], set()
    for e in emissions:
        if not isinstance(e, Mapping):
            errors.append("invalid emission evidence")
            continue
        f, aim_f = e.get("frame"), e.get("aim_player_frame")
        track_id = e.get("track_id")
        if (not _int(f) or not _int(aim_f) or aim_f > f
                or not isinstance(track_id, str) or f not in teacher_frames.get(track_id, ())):
            errors.append("emission must bind the teacher track at its emission frame")
            continue
        frames.append(f)
        if not isinstance(e.get("emitter_id"), str) or not e["emitter_id"]:
            errors.append("emitter identity required")
        else:
            emitters.add(e["emitter_id"])
        origin, aim, velocity = e.get("origin_px"), e.get("aim_player_px"), e.get("velocity_px_per_frame")
        images = {im["frame"]: im["sha256"] for im in o["images"]}
        if (not _point(origin, w, h) or not _point(aim, w, h)
                or aim_f not in images or e.get("aim_image_sha256") != images.get(aim_f)
                or not isinstance(velocity, (list, tuple)) or len(velocity) != 2
                or not all(_number(v) for v in velocity)):
            errors.append("invalid image-bound aim geometry/velocity units")
            continue
        motion = sorted((key[1], s["xy"]) for key, s in p.items() if key[0] == track_id and key[1] >= f)
        if len(motion) < 2:
            errors.append("emission needs at least two visible motion samples")
            continue
        dt = motion[1][0] - motion[0][0]
        derived = [(motion[1][1][i] - motion[0][1][i]) / dt for i in (0, 1)]
        if (_distance(origin, motion[0][1]) > t["position_px"]
                or _distance(velocity, derived) > t["aim_cross_px_per_frame"]):
            errors.append("emission origin/velocity disagree with independent pixel motion")
        player_sample = e.get("player_pixel_annotation", {})
        if (not isinstance(player_sample, Mapping) or player_sample.get("frame") != aim_f
                or player_sample.get("image_sha256") != images.get(aim_f)
                or not _point(player_sample.get("xy"), w, h)
                or _distance(player_sample["xy"], aim) > t["position_px"]):
            errors.append("independent image-bound player aim annotation required")
        elif aim_f == o["frame"] and _distance(player_sample["xy"], o["player_px"]) > t["position_px"]:
            errors.append("player annotation disagrees with observation")
        # Aim is proven by *privileged* geometry and checked as a miss distance.
        # Pixel positions carry measurement noise, so a velocity derived from two
        # measured samples is not an aim source; the teacher's exact positions
        # are.  ``miss`` then answers the actual question: does the shot's path
        # pass through the player's hitbox?
        teacher_motion = sorted((key[1], s["xy"]) for key, s in q.items()
                                if key[0] == track_id and key[1] >= f)
        if len(teacher_motion) < 2 or teacher_motion[0][0] != f:
            errors.append("teacher track must start at the emission frame with two samples")
            continue
        (ta, t_origin), (tb, t_second) = teacher_motion[0], teacher_motion[1]
        teacher_velocity = [(t_second[i] - t_origin[i]) / (tb - ta) for i in (0, 1)]
        miss = aim_miss(t_origin, teacher_velocity, aim)
        if miss is None:
            errors.append("privileged teacher velocity is degenerate")
            continue
        if aim_f > f or f - aim_f > lookback:
            errors.append("aim frame is outside the declared physical lookback")
        if f > o["frame"] or o["frame"] - f > recency:
            errors.append("emission is outside the declared current-stream recency")
        if not aim_ahead(t_origin, teacher_velocity, aim):
            errors.append("emission does not advance toward the annotated player")
        if miss > t["aim_miss_px"]:
            errors.append("privileged aim path misses the player hitbox beyond tolerance")
        # The declared number is recomputed from the same privileged geometry, so
        # the review cannot claim a tight miss it did not measure.
        declared = e.get("miss_px")
        if not _number(declared) or abs(declared - miss) > .01:
            errors.append("declared aim miss does not match recomputed privileged geometry")
        # The pixel-derived velocity may disagree only within the measurement
        # tolerance; it is a consistency check, never the aim proof.
        if _distance(velocity, teacher_velocity) > t["aim_cross_px_per_frame"]:
            errors.append("declared velocity disagrees with privileged teacher motion")
    if len(set(frames)) < 2 or len(emitters) != 1:
        errors.append("aimed stream requires distinct emissions from one emitter")
    return errors


def _replay_candidate(replay, frame):
    if not isinstance(replay, Mapping) or replay.get("source") != "replay_decoder_alignment":
        raise ValueError("actual action requires replay_decoder_alignment, not inputReadback")
    if type(replay.get("extended")) is not bool:
        raise ValueError("explicit replay layout required")
    if not _sha(replay.get("replay_sha256")) or not _sha(replay.get("payload_sha256")):
        raise ValueError("replay and decoded stage payload SHA required")
    # The replay cursor runs on the *logic* clock, not the stage clock:
    # ``GameManager::stageActiveFrames`` stops during a dialogue
    # (``Player.cpp:1072``) while ``PlaybackInputAndFps`` consumes one record per
    # logic frame (``ReplayManager.cpp:329``), so a stage-frame index silently
    # drifts by the accumulated stall. The row therefore names the record index
    # explicitly and carries the two input words the game itself consumed; both
    # are re-derived from the payload here instead of being trusted.
    index = replay.get("record_index")
    if (not _int(index) or index < 1 or replay.get("logic_frame") != index
            or replay.get("observation_frame") != frame
            or not _int(replay.get("consumed_mask"))):
        raise ValueError("unaligned replay action")
    try:
        payload = bytes.fromhex(replay["payload_hex"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("invalid stage payload") from exc
    if hashlib.sha256(payload).hexdigest() != replay["payload_sha256"]:
        raise ValueError("stage payload SHA mismatch")
    # Unknown high bits are preserved, not ignored: the candidate check below
    # rejects any record that carries them.  Decoding strictly here would turn a
    # single unknown-flag frame anywhere in the stage into "no label at all" for
    # every unrelated frame.
    records = decode_records(payload, extended=replay["extended"], allow_unknown_flags=True)
    if align_masks(records, start_frame=index - 1, count=1)[0] != replay["consumed_mask"]:
        raise ValueError("replay record is not the input this frame consumed")
    if index >= 2 and replay.get("previous_mask") is not None and (
            align_masks(records, start_frame=index - 2, count=1)[0] != replay["previous_mask"]):
        raise ValueError("replay record is not the second input word this frame consumed")
    mask = align_masks(records, start_frame=index, count=1)[0]
    if mask & ~0xF5:  # candidate_from_mask otherwise ignores menu/unknown bits.
        raise ValueError("record has buttons outside actual 18-way candidate set")
    candidate, reason = candidate_from_mask(mask)
    if candidate is None:
        raise ValueError(reason)
    return candidate


def build_label(observation, annotation, replay):
    """Create one row; missing evidence yields unknown and no pseudo action."""
    row = dict(schema=SCHEMA, method="replay_demonstration", observation=deepcopy(observation),
               event_annotation=deepcopy(annotation), replay=deepcopy(replay),
               action_positive=[False] * 18, action_valid=False,
               label_status="unknown", unknown_reason=None, requires_review=True)
    for name in NULL_TARGETS:
        row[name] = None
        row[name + "_valid"] = False
        row[name + "_status"] = "not_applicable"
    for name in HEADS:
        row[name + "_valid"] = False
    errors = validate_event_annotation(annotation, observation)
    candidate = None
    if not errors:
        try:
            candidate = _replay_candidate(replay, observation["frame"])
        except (ValueError, ReplayError, TypeError, KeyError) as exc:
            errors.append(str(exc))
    if errors:
        row["unknown_reason"] = "; ".join(errors)
    else:
        row.update(label_status="demonstration", action_valid=True, requires_review=False)
        row["action_positive"][candidate] = True
    return row


def validate_label(row) -> list[str]:
    """Validate schema and re-audit evidence; never trust a cached valid flag."""
    if not isinstance(row, Mapping):
        return ["row must be a mapping"]
    errors = []
    if row.get("schema") != SCHEMA or row.get("method") != "replay_demonstration":
        errors.append("wrong schema/method")
    for name in NULL_TARGETS:
        if name not in row or row[name] is not None or row.get(name + "_valid") is not False or row.get(name + "_status") != "not_applicable":
            errors.append(f"{name} must be null, invalid and not_applicable")
    for name in HEADS:
        if row.get(name + "_valid") is not False:
            errors.append(f"{name} supervision is not applicable")
    positive = row.get("action_positive")
    wellformed = isinstance(positive, list) and len(positive) == 18 and all(type(x) is bool for x in positive)
    if not wellformed:
        errors.append("action_positive must contain exactly 18 booleans")
    if type(row.get("action_valid")) is not bool:
        errors.append("action_valid must be bool")
    status = row.get("label_status")
    if status == "unknown":
        if row.get("action_valid") is not False or (wellformed and any(positive)):
            errors.append("unknown cannot carry a pseudo action")
        if not isinstance(row.get("unknown_reason"), str) or not row["unknown_reason"].strip():
            errors.append("unknown_reason required")
    elif status == "demonstration":
        if row.get("action_valid") is not True or not wellformed or sum(positive) != 1 or row.get("unknown_reason") is not None:
            errors.append("demonstration must have exactly one actual action")
        evidence_errors = validate_event_annotation(row.get("event_annotation"), row.get("observation"))
        errors.extend(evidence_errors)
        if not evidence_errors:
            try:
                c = _replay_candidate(row.get("replay"), row["observation"]["frame"])
                if wellformed and not positive[c]:
                    errors.append("action differs from decoder-aligned actual candidate")
            except (ValueError, TypeError, KeyError) as exc:
                errors.append(str(exc))
    else:
        errors.append("label_status must be unknown or demonstration")
    return errors


def label_to_tensors(rows, *, device=None):
    """Flat [N] masks/[N,18] actions accepted by tactical_policy_loss.

    Reshape to [B,T] in a sequence collator if needed. Every auxiliary mask,
    including local KL, stays false; unknown rows have zero gradient on all heads.
    Invalid rows raise instead of being silently repaired.
    """
    import torch
    rows = list(rows)
    for i, row in enumerate(rows):
        errors = validate_label(row)
        if errors:
            raise ValueError(f"row {i}: " + "; ".join(errors))
    result = {"action_positive": torch.tensor([r["action_positive"] for r in rows], dtype=torch.bool, device=device).reshape(-1, 18),
              "action_valid": torch.tensor([r["action_valid"] for r in rows], dtype=torch.bool, device=device)}
    for name in HEADS:
        result[name + "_valid"] = torch.zeros(len(rows), dtype=torch.bool, device=device)
    return result
