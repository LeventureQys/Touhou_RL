"""Frozen-protocol evaluation for the v3.7.6 F2 behaviour-cloning run.

What it does *not* do: it never selects a checkpoint, never changes a tolerance,
and never reports a number the frozen protocol did not ask for.  Validation
metrics are produced for the checkpoint the trainer already selected by
validation loss.  The test split is not cached and is not read here at all: the
sealed test evaluator is a separate invocation that refuses to overwrite output.

The baselines matter more than the headline number.  Six supervised rows cannot
distinguish "learned something" from "predict the previous action", and a few
hundred rows from one stage do not fix that on their own, so the report always
prints the hold-last, majority-class and uniform-random baselines beside the
model, plus class-balanced and action-change accuracy, a per-event macro average
and a playfield-zeroed ablation that exposes a policy reading the HUD instead of
the bullets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from touhou_rl import tactical_sequence_train as seq  # noqa: E402
from touhou_rl.tactical_data import file_sha  # noqa: E402
from touhou_rl.tactical_policy import TacticalConfig, TacticalPolicy  # noqa: E402

PROTOCOL_SCHEMAS = ('v376-f2-preregistration/3', 'v376-f2-eval-protocol/1')
RUN_SCHEMA = 'v373-sequence-research-run/1'


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def event_meta(manifest):
    """Event JSONs (steps, split, asset refs) keyed by id; tensors untouched."""
    obj = _json(manifest)
    meta = {}
    for ref in obj['events']:
        path = (manifest.parent / ref['path']).resolve()
        event = _json(path)
        event['_dir'] = path.parent
        meta[event['event_id']] = event
    return meta


def scored_rows(event):
    """Exactly the rows the trainer scores: burnin16 then non-overlapping 32."""
    rows = []
    for _event, start in seq.windows([dict(features=event['steps'])]):
        rows.extend(range(start, start + 32))
    return torch.tensor(sorted(set(rows)), dtype=torch.long)


def truth(event, rows):
    labels = event['labels']
    positive = labels['action_positive'][rows]
    action = positive.to(torch.int64).argmax(-1)
    valid = labels['action_valid'][rows] & positive.any(-1)
    previous = []
    for row in rows.tolist():
        value = event['steps'][row].get('previous_actual_candidate')
        previous.append(-1 if value is None else int(value))
    return action, valid, torch.tensor(previous, dtype=torch.long)


def metrics(action, valid, previous, predicted, majority=None):
    """Protocol metrics over supervised rows only."""
    action, previous, predicted = (x[valid] for x in (action, previous, predicted))
    total = int(action.numel())
    if total == 0:
        return dict(supervised_rows=0)
    correct = predicted == action
    classes = sorted(set(action.tolist()))
    recalls = [float(correct[action == value].float().mean()) for value in classes]
    defined = previous >= 0
    changed = defined & (action != previous)
    hold_correct = predicted[defined] == previous[defined]
    result = dict(
        supervised_rows=total,
        accuracy=float(correct.float().mean()),
        class_balanced_accuracy=float(sum(recalls) / len(recalls)),
        classes_present=len(classes),
        action_changes=int(changed.sum()),
        action_change_accuracy=float(correct[changed].float().mean()) if bool(changed.any()) else None,
        hold_last_defined_rows=int(defined.sum()),
        hold_last_accuracy=float(hold_correct.float().mean()) if int(defined.sum()) else None,
        hold_last_accuracy_all_rows=float(hold_correct.sum()) / total,
        hold_last_note='an undefined previous action (unknown history) counts as wrong in the all-rows column',
        uniform_random_accuracy=1.0 / 18.0,
        predicted_histogram={int(k): int(v) for k, v in zip(*torch.unique(predicted, return_counts=True))},
        true_histogram={int(k): int(v) for k, v in zip(*torch.unique(action, return_counts=True))},
    )
    if majority is not None:
        result['majority_class'] = int(majority)
        result['majority_class_accuracy'] = float((action == majority).float().mean())
    return result


def predict(model, features, base_logits, history, device, rows=None):
    with torch.no_grad():
        out = model.forward_features(features[None].to(device), base_logits[None].to(device),
                                     {k: v[None].to(device) for k, v in history.items()})
    logits = out['policy_logits'][0]
    if rows is not None:
        logits = logits[rows]
    return logits.argmax(-1).cpu()


def _training_actions(data, meta):
    actions = []
    for event in data['train']:
        rows = scored_rows(meta[event['event_id']])
        action, valid, _previous = truth(meta[event['event_id']], rows)
        actions.extend(action[valid].tolist())
    return actions


def playfield_ablation(backbone, model, manifest, events, meta, device):
    """Re-run with the playfield zeroed; a HUD shortcut shows up as no drop."""
    collected = dict(action=[], valid=[], previous=[], predicted=[])
    with torch.no_grad():
        for event in events:
            info = meta[event['event_id']]
            path = (info['_dir'] / info['inputs']['pixels']['path']).resolve()
            pixels = {k: v.clone() for k, v in
                      torch.load(path, map_location='cpu', weights_only=True).items()}
            if 'playfield' not in pixels:
                raise SystemExit('playfield region missing from the dataset')
            pixels['playfield'] = torch.zeros_like(pixels['playfield'])
            features = []
            for offset in range(0, len(info['steps']), 16):
                batch = {k: v[offset:offset + 16].to(device) for k, v in pixels.items()}
                features.append(backbone.forward_core(batch)['features'].cpu())
            features = torch.cat(features)
            base_logits = backbone.policy_head(features.to(device)).cpu()
            rows = scored_rows(info)
            action, valid, previous = truth(info, rows)
            collected['action'].append(action)
            collected['valid'].append(valid)
            collected['previous'].append(previous)
            collected['predicted'].append(predict(model, features, base_logits, event['history'], device, rows))
    return metrics(*(torch.cat(collected[k]) for k in ('action', 'valid', 'previous', 'predicted')))


def evaluate(manifest, cache_index, baseline, run, protocol_path, out, *, ablation=True):
    manifest, cache_index, baseline, run = map(Path, (manifest, cache_index, baseline, run))
    protocol = _json(protocol_path)
    if protocol.get('schema') not in PROTOCOL_SCHEMAS or protocol.get('frozen') is not True:
        raise SystemExit('frozen preregistration required')
    bounds = protocol.get('binds') or {}
    for name, path in (('manifest', manifest), ('cache_index', cache_index), ('baseline', baseline)):
        ref = bounds.get(name)
        if ref is not None and file_sha(path) != ref['sha256']:
            raise SystemExit(f'{name} SHA differs from the frozen protocol')
    if Path(out).exists():
        raise SystemExit('evaluation output already exists; a frozen report is written once')
    report = _json(run / 'report.json')
    if report.get('schema') != RUN_SCHEMA:
        raise SystemExit('trained run report required')
    expected = protocol['training']
    for key in ('arm', 'seed', 'epochs', 'max_updates', 'learning_rate', 'burnin', 'sequence', 'device'):
        if expected[key] != report['config'][key]:
            raise SystemExit(f'trained run config {key}={report["config"][key]} differs from the frozen protocol {expected[key]}')
    if report.get('test_used') is not False:
        raise SystemExit('training report must show test_used=false')
    backbone, data, provenance = seq.prepare(manifest, cache_index, baseline)
    meta = event_meta(manifest)
    device = torch.device(provenance['compute_protocol']['device'])
    model = TacticalPolicy(backbone, TacticalConfig(arm=report['config']['arm'],
                                                   feature_dim=backbone.config.trunk_hidden))
    heads = torch.load(run / 'research-heads.pt', map_location='cpu', weights_only=True)
    model.heads.load_state_dict(heads['heads'])
    model.eval().to(device)
    train_actions = _training_actions(data, meta)
    if not train_actions:
        raise SystemExit('no supervised train rows: the majority baseline is undefined')
    majority = max(sorted(set(train_actions)), key=train_actions.count)
    per_event, pooled = {}, dict(action=[], valid=[], previous=[], predicted=[])
    for event in data['validation']:
        rows = scored_rows(meta[event['event_id']])
        action, valid, previous = truth(meta[event['event_id']], rows)
        predicted = predict(model, event['features'], event['base_logits'], event['history'], device, rows)
        per_event[event['event_id']] = metrics(action, valid, previous, predicted, majority)
        pooled['action'].append(action)
        pooled['valid'].append(valid)
        pooled['previous'].append(previous)
        pooled['predicted'].append(predicted)
    tensors = [torch.cat(pooled[k]) for k in ('action', 'valid', 'previous', 'predicted')]
    summary = metrics(*tensors, majority=majority)
    macro = [m['accuracy'] for m in per_event.values() if m.get('supervised_rows')]
    summary['per_event_macro_accuracy'] = float(sum(macro) / len(macro)) if macro else None
    summary['train_supervised_rows'] = len(train_actions)
    result = dict(
        schema='v376-f2-eval-report/1', split='validation',
        objective='f2_behavior_cloning', original_G4=False,
        protocol=dict(path=str(protocol_path), sha256=file_sha(protocol_path), frozen=protocol['frozen']),
        manifest_sha256=file_sha(manifest), cache_index_sha256=file_sha(cache_index),
        baseline_sha256=file_sha(baseline), run_report_sha256=file_sha(run / 'report.json'),
        heads_sha256=file_sha(run / 'research-heads.pt'),
        config=report['config'], summary=summary, per_event=per_event,
        limitation=protocol['evaluation']['claim_limitation'],
        test_sealed=True)
    if ablation:
        result['playfield_zeroed'] = playfield_ablation(backbone, model, manifest,
                                                        data['validation'], meta, device)
    _write_new(out, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--cache-index', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--no-ablation', action='store_true')
    args = parser.parse_args(argv)
    result = evaluate(args.manifest, args.cache_index, args.baseline, args.run,
                      args.protocol, args.out, ablation=not args.no_ablation)
    print(json.dumps(result['summary'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
