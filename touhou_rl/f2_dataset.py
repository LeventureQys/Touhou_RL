"""Frozen-plan F2 behaviour cloning, deliberately NOT the original G4 task.

Plan schema ``v376-f2-bc-plan/1``::

  {frozen: true, preregistered_at: ISO8601, config: CONFIG,
   config_sha256: evidence_sha(CONFIG), sources: [SOURCE], events: [EVENT]}

CONFIG is exactly DEFAULT_CONFIG. SOURCE has id and content-addressed relative
refs exact_manifest, replay, collect_config, teacher_artifact. collect_config is
an independently frozen capture provenance JSON with source='replay',
frame_proof='exact', replay_sha256, exact_manifest_sha256, config_sha256 (the
preregistered collection_config digest), collection_config. EVENT has event_id,
source_id, stage (decoder stage index), split, template_group, episode_id,
derived_group, and review (hashed JSON ref). Review JSON has frozen=true and rows
[{observation, annotation}], using f2_labels' independent-pixel-review contract.
All scored F2 rows must already have real annotations: this module never invents
review. Explicit context rows use status='context_only', annotation=null and a
nonempty unknown_reason; they remain zero-masked. Every event must also have a
real reviewed demonstration (after burnin16 for train/validation). Noncandidate
actual replay actions are retained unknown, never silently converted to hold.
The native collect report may wrap identity fields in capture_identity; the
proof must bind collectionProvenance {replaySha256,configSha256} at capture start.

Two clocks exist and they are not interchangeable. ``observation['frame']`` and
the PNG keys are the captured *stage* clock (``stageActiveFrames``); the replay
input record index and the row time base are the *logic* clock
(``gameplayFrameCounter``). ``stageActiveFrames`` stops while a dialogue is
present while the replay keeps consuming one record per logic frame, so a
stage-frame index silently goes stale after the first stage dialogue. Every row
therefore binds its record index to the logic clock and re-derives the two input
words the game itself consumed for that held frame; a mismatch fails the build.

Refs are {path,sha256}; relative paths cannot escape their containing root.
The original plan and upstream bytes remain required for every deep re-audit.
SHA provenance is not an external timestamp authority or proof of human honesty.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path, PureWindowsPath
import re

import numpy as np
import torch

from . import exact_audit, frame_proof, f2_labels, vision
from .v380.replay import decode_records, align_masks
from .v380.live import candidate_from_mask

SCHEMA = 'v376-f2-bc-events/1'
PLAN_SCHEMA = 'v376-f2-bc-plan/1'
SPLITS = ('train', 'validation', 'test')
SIZES = {'playfield': (224, 192), 'hud_score': (34, 216), 'hud_status': (126, 216)}
DEFAULT_CONFIG = dict(depth=3, region_sizes={k: list(v) for k, v in SIZES.items()},
                      history_offsets=[-4, -2, 0], min_event_rows=48,
                      objective='f2_behavior_cloning', original_G4=False,
                      split_unit='replay_and_association_groups', coordinate_targets=False)


class F2DatasetError(ValueError):
    pass


def _require(ok, message):
    if not ok:
        raise F2DatasetError(message)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def _inside(root, text):
    _require(isinstance(text, str) and text.strip(), 'missing relative path')
    p = Path(text)
    w = PureWindowsPath(text)
    _require(not p.is_absolute() and not w.is_absolute() and not w.drive
             and '\\' not in text and ':' not in text, 'absolute/Windows path refused')
    root = Path(root).resolve()
    target = (root / p).resolve()
    _require(target.is_relative_to(root), 'path escapes root')
    return target


def _asset(root, ref):
    _require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'hashed ref required')
    target = _inside(root, ref['path'])
    _require(target.is_file() and file_sha(target) == ref['sha256'], 'asset SHA mismatch: ' + str(ref['path']))
    return target


def _ref(root, path):
    return dict(path=Path(path).relative_to(root).as_posix(), sha256=file_sha(path))


def _stage_payload(path, stage_index):
    """Use the existing TH08 codec, never accept supplied payload_hex as truth."""
    script = Path(__file__).resolve().parents[1] / 'third_party/th08/scripts/replay_file.py'
    spec = importlib.util.spec_from_file_location('_f2_th08_replay_file', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = module.decode(Path(path).read_bytes())
    info = module.inspect(path)
    stages = [s for s in info['stages'] if s['index'] == stage_index]
    _require(len(stages) == 1, 'stage not uniquely present in actual replay')
    stage = stages[0]
    extended = info['extended']
    start = stage['offset'] + 0x24
    end = start + stage['inputRecords'] * (6 if extended else 2)
    _require(0 <= start < end <= len(data), 'stage payload extent invalid')
    return data[start:end], extended


def _candidate(mask):
    if mask & ~0xF5:
        return None
    return candidate_from_mask(mask)[0]


def _load_plan(path):
    path = Path(path).resolve()
    plan = _json(path)
    _require(plan.get('schema') == PLAN_SCHEMA and plan.get('frozen') is True, 'frozen preregistered plan required')
    _require(plan.get('config') == DEFAULT_CONFIG, 'unsupported preregistered config')
    _require(plan.get('config_sha256') == f2_labels.evidence_sha(plan['config']), 'plan config digest mismatch')
    try:
        prereg = datetime.fromisoformat(plan['preregistered_at'])
    except (ValueError, KeyError, TypeError) as exc:
        raise F2DatasetError('preregistered_at required') from exc
    _require(prereg.tzinfo is not None, 'preregistration timestamp timezone required')
    sources = plan.get('sources', [])
    events = plan.get('events', [])
    _require(len(sources) >= 3 and events, 'at least three independent replay sources required')
    ids, replay_hashes, source_map = set(), set(), {}
    for source in sources:
        sid = source.get('id')
        _require(isinstance(sid, str) and sid and sid not in ids, 'duplicate/missing source id')
        ids.add(sid)
        replay = _asset(path.parent, source['replay'])
        replay_hashes.add(file_sha(replay))
        source_map[sid] = source
    _require(len(replay_hashes) >= 3, 'at least three independent replay hashes required; never split one replay')
    groups, eids, splits = {}, set(), set()
    for event in events:
        eid, split = event.get('event_id'), event.get('split')
        _require(isinstance(eid, str) and re.fullmatch(r'[A-Za-z0-9_-]+', eid) and eid not in eids,
                 'safe unique event_id required')
        eids.add(eid)
        _require(split in SPLITS, 'invalid fixed split')
        splits.add(split)
        source = source_map.get(event.get('source_id'))
        _require(source is not None, 'unknown source_id')
        keys = [('replay', source['replay']['sha256'])]
        for name in ('template_group', 'episode_id', 'derived_group'):
            value = event.get(name)
            _require(isinstance(value, str) and value, 'missing association group: ' + name)
            keys.append((name, value))
        for key in keys:
            _require(key not in groups or groups[key] == split, 'cross-split replay/association leakage: ' + repr(key))
            groups[key] = split
    _require(splits == set(SPLITS), 'all three preregistered splits required')
    return plan


def _load_source(root, source, *, deep):
    manifest = _asset(root, source['exact_manifest'])
    capture = manifest.parent
    _require(manifest.name == frame_proof.PROOF_MANIFEST_NAME, 'exact manifest filename mismatch')
    # Do not trust upstream passed/status strings: run the real verifier.
    exact_audit.audit_exact_dataset(capture, deep=deep, expect_exact=True, require_proof=True)
    proof = _json(manifest)
    replay = _asset(root, source['replay'])
    collect_path = _asset(root, source['collect_config'])
    collect_document = _json(collect_path)
    collect = collect_document.get('capture_identity', collect_document)
    _require(collect.get('source') == 'replay' and collect.get('frame_proof') == 'exact', 'capture collection source/config mismatch')
    _require(collect.get('replay_sha256') == file_sha(replay)
             and collect.get('exact_manifest_sha256') == file_sha(manifest), 'collection replay/capture identity mismatch')
    _require(isinstance(source.get('collection_config_sha256'), str)
             and collect.get('config_sha256') == source['collection_config_sha256']
             and f2_labels.evidence_sha(collect.get('collection_config')) == source['collection_config_sha256'],
             'preregistered collection config mismatch')
    provenance = proof.get('collectionProvenance', {})
    _require(provenance.get('replaySha256') == file_sha(replay)
             and provenance.get('configSha256') == source['collection_config_sha256'],
             'capture-start collectionProvenance missing or mismatched')
    teacher = _asset(root, source['teacher_artifact'])
    _require(teacher.parent == capture and teacher.name == 'audit-state.jsonl', 'same-hold full teacher artifact required')
    _require(proof.get('auditState', {}).get('sha256') == file_sha(teacher), 'proof does not bind full teacher artifact')
    rows_path = _asset(capture, dict(path=proof['proofRows']['path'], sha256=proof['proofRows']['sha256']))
    rows, teachers, ambiguous = {}, {}, {}
    for line in rows_path.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        state = row['stateSummary']
        key = (state['stage'], state['stageFrames'])
        # A paused stage clock can accompany advancing logic frames. None of
        # those samples uniquely identifies the actual replay input; never pick
        # first/last. Keep the raw capture intact and exclude the entire group.
        if key in ambiguous:
            ambiguous[key].append(row['logicFrame'])
        elif key in rows:
            ambiguous[key] = [rows.pop(key)['logicFrame'], row['logicFrame']]
        else:
            rows[key] = row
    for line in teacher.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        _require(row['logicFrame'] not in teachers, 'duplicate teacher logic frame')
        teachers[row['logicFrame']] = row
    lag = [row['logicFrame'] - int(row['stateSummary']['stageFrames']) for row in rows.values()]
    distinct_lag = sorted(set(lag))
    return dict(manifest=manifest, capture=capture, proof=proof, replay=replay,
                teacher=teacher, rows=rows, teachers=teachers, proof_rows=rows_path,
                alignment_diagnostics=dict(ambiguous_stage_frames=[
                    dict(stage=stage, stage_frame=frame, logic_frames=logic_frames)
                    for (stage, frame), logic_frames in sorted(ambiguous.items())],
                    excluded_samples=sum(map(len, ambiguous.values())), unique_stage_frames=len(rows),
                    # How far the captured stage clock has fallen behind the
                    # replay logic clock. Zero means the two clocks agree for
                    # every kept sample; anything else is the accumulated stage
                    # clock stall (dialogue) that a stage-frame index would carry
                    # into every later label.
                    stage_clock_lag=dict(minimum=distinct_lag[0] if distinct_lag else None,
                                         maximum=distinct_lag[-1] if distinct_lag else None,
                                         values=distinct_lag,
                                         samples_not_at_minimum=sum(1 for value in lag if distinct_lag and value != distinct_lag[0]))))


def _verify_teacher(observation, annotation, upstream, stage):
    """Rebind independent review to actual full same-hold upstream samples."""
    teacher = annotation['teacher']
    _require(teacher.get('artifact_sha256') == file_sha(upstream['teacher']), 'review teacher artifact SHA mismatch')
    tolerance = (annotation.get('tolerances') or {}).get('aim_miss_px')
    _require(type(tolerance) in (int, float) and tolerance >= 0, 'declared aim miss tolerance required')
    # Raw audit state uses game coordinates. TH08's native playfield origin is
    # (32,16); annotations explicitly use full 640x480 image pixels.
    for emission in annotation['emissions']:
        proof = upstream['rows'].get((stage, emission['aim_player_frame']))
        _require(proof is not None, 'aim frame lacks exact upstream evidence')
        raw = upstream['teachers'].get(proof['logicFrame'])
        _require(raw is not None and raw.get('ticket') == proof['ticket'], 'aim player state is not same hold')
        state = raw.get('state', {})
        actual = np.asarray([state.get('player_x', float('nan')) + 32,
                             state.get('player_y', float('nan')) + 16])
        _require(float(np.linalg.norm(actual - emission['aim_player_px'])) <= annotation['tolerances']['position_px'],
                 'review aim player geometry differs from actual upstream')
        # Recompute the aim claim straight from the raw pool instead of trusting
        # the annotation's own teacher copy: origin and velocity are the game's
        # exact values for this held frame, and the miss distance is the
        # geometric quantity the review claims to have measured.
        track_id = emission['track_id']
        _require(isinstance(track_id, str) and track_id.startswith('slot:'),
                 'explicit frame-local slot identity required')
        emission_proof = upstream['rows'].get((stage, emission['frame']))
        _require(emission_proof is not None, 'emission frame lacks exact upstream evidence')
        emission_raw = upstream['teachers'].get(emission_proof['logicFrame'])
        _require(emission_raw is not None and emission_raw.get('ticket') == emission_proof['ticket'],
                 'emission state is not same hold')
        slot = int(track_id[5:])
        bullets = [b for b in emission_raw.get('bullets', [])
                   if b.get('slot') == slot and not b.get('disabled')]
        _require(len(bullets) == 1, 'emission slot missing/inactive in actual full pool')
        bullet = bullets[0]
        _require(isinstance(bullet.get('velocity'), (list, tuple)) and len(bullet['velocity']) == 2,
                 'emission bullet velocity missing from actual full pool')
        origin = [bullet['position'][0] + 32, bullet['position'][1] + 16]
        miss = f2_labels.aim_miss(origin, bullet['velocity'], emission['aim_player_px'])
        _require(miss is not None and miss <= tolerance,
                 'privileged aim path misses the player hitbox beyond tolerance')
    for track in teacher['tracks']:
        for sample in track['samples']:
            proof = upstream['rows'].get((stage, sample['frame']))
            _require(proof is not None, 'teacher sample lacks exact proof')
            raw = upstream['teachers'].get(proof['logicFrame'])
            _require(raw is not None and raw.get('ticket') == proof['ticket'], 'teacher is not the same held ticket')
            _require(raw.get('schema') == 'th08-exact-audit-state/1'
                     and raw.get('bulletsStatus') == 'available' and raw.get('readComplete') is True
                     and len(raw.get('bullets', [])) == raw.get('activeCount'), 'teacher full bullet pool required')
            # Slots are not lifecycle IDs. A review must explicitly use slot:N;
            # this checks frame-local identity only, never invents generations.
            _require(track['id'].startswith('slot:'), 'explicit frame-local slot identity required')
            slot = int(track['id'][5:])
            bullets = [b for b in raw.get('bullets', []) if b.get('slot') == slot]
            _require(len(bullets) == 1, 'teacher slot missing/inactive in actual full pool')
            bullet = bullets[0]
            _require(sample['xy'] == [bullet['position'][0] + 32, bullet['position'][1] + 16], 'teacher sample differs from raw upstream bullet')


def _event_payload(root, spec, source, upstream):
    stage = spec.get('stage')
    _require(type(stage) is int and stage >= 0, 'explicit decoder stage index required')
    payload, extended = _stage_payload(upstream['replay'], stage)
    records = decode_records(payload, extended=extended, allow_unknown_flags=True)
    review_path = _asset(root, spec['review'])
    review = _json(review_path)
    _require(review.get('frozen') is True and isinstance(review.get('rows'), list), 'frozen human review rows required')
    reviewed = review['rows']
    _require(len(reviewed) >= DEFAULT_CONFIG['min_event_rows'], 'each event requires at least 48 reviewed rows')
    frames = [r['observation']['frame'] for r in reviewed]
    _require(all(type(f) is int and f >= 4 for f in frames) and all(b > a for a, b in zip(frames, frames[1:])),
             'strictly increasing stage frames with complete past required')
    pixels, input_frames, labels, steps = {k: [] for k in SIZES}, {}, [], []
    history = dict(previous_candidate=[], previous_option=[], previous_hold=[], frame_delta=[],
                   previous_target=[], target_valid=[], history_valid=[])
    for item, frame in zip(reviewed, frames):
        observation, annotation = item['observation'], item.get('annotation')
        _require(not f2_labels.audit_observation(observation), 'invalid causal observation')
        # Explicitly unreviewed context may be retained with zero masks, but a
        # supplied F2 claim must fully verify; forged evidence is not unknown.
        reviewed_f2 = item.get('status') != 'context_only'
        if not reviewed_f2:
            _require(annotation is None and isinstance(item.get('unknown_reason'), str)
                     and bool(item['unknown_reason'].strip()), 'explicit context_only requires null annotation and reason')
        if reviewed_f2:
            _require(isinstance(annotation, dict), 'missing human annotation; use explicit context_only')
            _require(not f2_labels.validate_event_annotation(annotation, observation), 'invalid independent F2 human review')
        _require(observation['source']['capture'] == {'id': source['id'], 'sha256': file_sha(upstream['manifest'])}
                 and observation['source']['proof'] == {'id': frame_proof.PROOF_ROWS_NAME, 'sha256': file_sha(upstream['proof_rows'])},
                 'review capture/proof identity mismatch')
        _require(observation.get('image_size') == [640, 480], 'full native PNG coordinates required')
        current_proof = upstream['rows'].get((stage, frame))
        _require(current_proof is not None, 'current reviewed frame lacks exact proof')
        current_raw = upstream['teachers'].get(current_proof['logicFrame'])
        _require(current_raw is not None and current_raw.get('ticket') == current_proof['ticket'], 'current review lacks same-hold state')
        player = current_raw.get('state') or {}
        _require(observation['player_px'] == [player.get('player_x', float('nan')) + 32,
                                             player.get('player_y', float('nan')) + 16],
                 'review player differs from same-hold upstream state')
        for image in observation['images']:
            proof = upstream['rows'].get((stage, image['frame']))
            _require(proof is not None and image['sha256'] == proof['imageSha256'], 'review image SHA/proof binding mismatch')
            image_path = _inside(upstream['capture'], proof['imagePath'])
            _require(file_sha(image_path) == image['sha256'], 'review PNG bytes changed')
            content = frame_proof.pixel_sha256(frame_proof.decode_png_pixels(image_path))
            _require(content == proof['pixelSha256'], 'review PNG content changed')
            input_frames[image['frame']] = content
        if reviewed_f2:
            _verify_teacher(observation, annotation, upstream, stage)
        # The replay cursor runs on the logic clock. ``stageActiveFrames`` stops
        # while a dialogue is present (Player.cpp:1072) but ``PlaybackInputAndFps``
        # still consumes one record per logic frame (ReplayManager.cpp:329), so a
        # stage-frame index becomes stale by the accumulated stall once the stage
        # has shown a dialogue (measured: 149 records on train, 393 on
        # validation). Bind the index to the logic clock and *prove* it with the
        # two input words the game itself published for this held frame instead of
        # assuming any offset.
        logic = current_proof['logicFrame']
        _require(type(logic) is int and logic >= 2, 'logic clock cannot be verified against the replay')
        consumed = align_masks(records, start_frame=logic - 1, count=1)[0]
        prior = align_masks(records, start_frame=logic - 2, count=1)[0]
        _require(player.get('gui_input_current') == consumed,
                 'decoded replay record is not the input this frame consumed')
        _require(player.get('gui_input_previous') == prior,
                 'decoded replay record is not the previous input this frame consumed')
        replay = dict(source='replay_decoder_alignment', extended=extended,
                      logic_frame=logic, record_index=logic, observation_frame=frame,
                      consumed_mask=consumed, previous_mask=prior,
                      replay_sha256=file_sha(upstream['replay']),
                      payload_sha256=hashlib.sha256(payload).hexdigest(), payload_hex=payload.hex())
        label = f2_labels.build_label(observation, annotation if reviewed_f2 else None, replay)
        labels.append(label)
        native_regions = {k: [] for k in SIZES}
        for f in (frame - 4, frame - 2, frame):
            proof = upstream['rows'].get((stage, f))
            _require(proof is not None, 'missing complete current/past exact PNG')
            image = _inside(upstream['capture'], proof['imagePath'])
            _require(image.is_file() and file_sha(image) == proof['imageSha256'], 'input PNG SHA mismatch')
            array = frame_proof.decode_png_pixels(image)
            content = frame_proof.pixel_sha256(array)
            _require(content == proof['pixelSha256'], 'input pixel content mismatch')
            input_frames[f] = content
            for name, region in vision.extract_regions(array).items():
                native_regions[name].append(vision.resize_area(region, *SIZES[name]))
        for name in SIZES:
            pixels[name].append(np.stack(native_regions[name]))
        # Input history only sees the decoder record the current frame already
        # consumed; never this frame's own label, targets or future action.
        mask = consumed
        candidate = _candidate(mask)
        hold = 1
        if candidate is not None:
            while logic - hold - 1 >= 0 and records[logic - hold - 1].mask == mask:
                hold += 1
        valid = candidate is not None
        history['previous_candidate'].append(candidate if valid else 18)
        history['previous_option'].append(3)
        history['previous_hold'].append(float(hold) if valid else 0.)
        history['frame_delta'].append(1. if valid else 0.)
        history['previous_target'].append([0., 0.])
        history['target_valid'].append(False)
        history['history_valid'].append(valid)
        # ``frame`` is the decoder logic clock (replay time base); ``stage_frame``
        # is the capture clock that keys the PNGs. They are not interchangeable.
        steps.append(dict(frame=logic, stage_frame=frame, logic_frame=logic,
                          capture_interval=[frame, frame], alignment_status='verified',
                          pool_complete=True, history_observed_through=logic - 1,
                          previous_actual_candidate=candidate,
                          previous_effective_frame=logic - 1 if valid else None,
                          previous_actual_hold=hold if valid else None,
                          previous_actual_option=3, previous_actual_target=[0., 0.],
                          previous_target_valid=False))
    _require(any(r['action_valid'] for r in labels), 'event has no independently reviewed demonstration')
    if spec['split'] in ('train', 'validation'):
        _require(any(r['action_valid'] for r in labels[16:]), 'event has no supervised scored rows after burnin16')
    pixel_tensors = vision.regions_to_batch({k: np.stack(v) for k, v in pixels.items()})
    dtypes = dict(previous_candidate=torch.long, previous_option=torch.long, previous_hold=torch.float32,
                  frame_delta=torch.float32, previous_target=torch.float32, target_valid=torch.bool, history_valid=torch.bool)
    history = {k: torch.tensor(v, dtype=dtypes[k]) for k, v in history.items()}
    return dict(pixels=pixel_tensors, history=history, labels=f2_labels.label_to_tensors(labels),
                label_evidence=labels, steps=steps,
                input_pixel_hashes=[dict(frame=f, sha256=h) for f, h in sorted(input_frames.items())])


def _same_tensors(actual, expected):
    return (isinstance(actual, dict) and set(actual) == set(expected)
            and all(isinstance(actual[k], torch.Tensor) and actual[k].dtype == expected[k].dtype
                    and actual[k].shape == expected[k].shape and torch.equal(actual[k], expected[k]) for k in expected))


def _materialize(plan_path, *, deep):
    plan_path = Path(plan_path).resolve()
    plan = _load_plan(plan_path)
    sources = {s['id']: s for s in plan['sources']}
    upstream = {sid: _load_source(plan_path.parent, s, deep=deep) for sid, s in sources.items()}
    seen, results = {}, []
    for spec in plan['events']:
        result = _event_payload(plan_path.parent, spec, sources[spec['source_id']], upstream[spec['source_id']])
        result['alignment_diagnostics'] = upstream[spec['source_id']].get('alignment_diagnostics', {})
        for row in result['input_pixel_hashes']:
            sha = row['sha256']
            _require(sha not in seen or seen[sha] == spec['split'], 'cross-split per-frame pixel content duplicate')
            seen[sha] = spec['split']
        results.append((spec, result))
    return plan, results


def build_dataset(plan_path, out, *, deep=True):
    """Build three fixed splits. Structural-only builds can never pass audit."""
    plan_path = Path(plan_path).resolve()
    plan, results = _materialize(plan_path, deep=deep)
    out = Path(out).resolve()
    _require(not out.exists(), 'output already exists; never overwrite frozen dataset')
    out.mkdir(parents=True)
    refs = []
    for spec, result in results:
        directory = out / spec['event_id']
        directory.mkdir()
        for name in ('pixels', 'history', 'labels'):
            torch.save(result[name], directory / (name + '.pt'))
        _write_json(directory / 'label-evidence.json', result['label_evidence'])
        event = {k: spec[k] for k in ('event_id', 'source_id', 'stage', 'split', 'template_group', 'episode_id', 'derived_group')}
        event.update(schema=SCHEMA, family='F2', rows=len(result['steps']), steps=result['steps'],
                     inputs={k: _ref(directory, directory / (k + '.pt')) for k in ('pixels', 'history')},
                     labels=_ref(directory, directory / 'labels.pt'),
                     label_evidence=_ref(directory, directory / 'label-evidence.json'),
                     input_pixel_hashes=result['input_pixel_hashes'])
        _write_json(directory / 'event.json', event)
        refs.append(_ref(out, directory / 'event.json'))
    manifest = dict(schema=SCHEMA, objective='f2_behavior_cloning', original_G4=False,
                    coordinate_targets='not_applicable', upstream_deep=deep is True,
                    plan=dict(path=str(plan_path), sha256=file_sha(plan_path)),
                    config_sha256=plan['config_sha256'], events=refs,
                    alignment_diagnostics={s['source_id']: r.get('alignment_diagnostics', {}) for s, r in results},
                    split_groups={split: sorted({s['template_group'] for s in plan['events'] if s['split'] == split}) for split in SPLITS},
                    adapter=dict(target_schema='v373-paired-events/1', objective='f2_behavior_cloning', original_G4=False))
    _write_json(out / 'manifest.json', manifest)
    return out / 'manifest.json'


def audit_dataset(manifest):
    """Offline deep audit of ALL splits. Not a training data loader.

    Training consumers must separately load only train/validation tensor assets;
    this isolated auditor intentionally recomputes sealed test evidence as well.
    """
    manifest = Path(manifest).resolve()
    errors, diagnostics = [], {}
    try:
        obj = _json(manifest)
        _require(obj.get('schema') == SCHEMA and obj.get('objective') == 'f2_behavior_cloning'
                 and obj.get('original_G4') is False, 'not an F2 BC manifest')
        _require(obj.get('upstream_deep') is True, 'structural-only fixture build cannot certify real data')
        plan_path = Path(obj['plan']['path'])
        _require(plan_path.is_absolute() and file_sha(plan_path) == obj['plan']['sha256'], 'original frozen plan SHA mismatch')
        plan, results = _materialize(plan_path, deep=True)
        diagnostics = {s['source_id']: r.get('alignment_diagnostics', {}) for s, r in results}
        _require(obj.get('alignment_diagnostics') == diagnostics, 'alignment diagnostics differ from actual upstream')
        _require(obj.get('config_sha256') == plan['config_sha256'], 'manifest config mismatch')
        expected_groups = {s: sorted({e['template_group'] for e in plan['events'] if e['split'] == s}) for s in SPLITS}
        _require(obj.get('split_groups') == expected_groups, 'split_groups differs from preregistered plan')
        _require(len(obj['events']) == len(results), 'event count mismatch')
        all_input_paths, all_label_paths = set(), set()
        for ref, (spec, expected) in zip(obj['events'], results):
            event_path = _asset(manifest.parent, ref)
            event = _json(event_path)
            for key in ('event_id', 'source_id', 'stage', 'split', 'template_group', 'episode_id', 'derived_group'):
                _require(event.get(key) == spec[key], 'event metadata differs from frozen plan: ' + key)
            _require(event.get('schema') == SCHEMA and event.get('family') == 'F2', 'event schema/family mismatch')
            _require(event.get('rows') == len(expected['steps']) and event.get('steps') == expected['steps']
                     and event.get('input_pixel_hashes') == expected['input_pixel_hashes'], 'event alignment/pixel evidence mismatch')
            _require(set(event['inputs']) == {'pixels', 'history'}, 'undeclared input channel')
            for section in ('pixels', 'history', 'labels'):
                asset_ref = event['labels'] if section == 'labels' else event['inputs'][section]
                target = _asset(event_path.parent, asset_ref)
                _require(target.is_relative_to(manifest.parent), 'asset escapes dataset root')
                (all_label_paths if section == 'labels' else all_input_paths).add(target)
                loaded = torch.load(target, map_location='cpu', weights_only=True)
                _require(_same_tensors(loaded, expected[section]), 'recomputed ' + section + ' content mismatch')
            evidence_path = _asset(event_path.parent, event['label_evidence'])
            _require(evidence_path.is_relative_to(manifest.parent), 'label evidence escapes dataset root')
            all_label_paths.add(evidence_path)
            _require(_json(evidence_path) == expected['label_evidence'], 'label evidence differs from actual upstream')
        _require(not all_input_paths & all_label_paths, 'inputs and labels must be physically isolated')
        # Hard links are aliases too; a distinct string is not physical isolation.
        _require(not any(a.samefile(b) for a in all_input_paths for b in all_label_paths), 'input/label hardlink alias')
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, IndexError) as exc:
        errors.append(str(exc))
    return dict(passed=not errors, errors=errors, schema=SCHEMA,
                manifest_sha256=file_sha(manifest) if manifest.is_file() else None,
                alignment_diagnostics=diagnostics,
                objective='f2_behavior_cloning', original_G4=False)


def legacy_training_view(manifest):
    """Return audited adapter metadata, never rewrite the schema or G4 status."""
    audit = audit_dataset(manifest)
    _require(audit['passed'], '; '.join(audit['errors']))
    return _json(manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    build = sub.add_parser('build')
    build.add_argument('plan')
    build.add_argument('out')
    audit = sub.add_parser('audit')
    audit.add_argument('manifest')
    args = parser.parse_args(argv)
    if args.mode == 'build':
        print(build_dataset(args.plan, args.out))
        return 0
    result = audit_dataset(args.manifest)
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
