"""Freeze a ``v376-f2-bc-plan/1`` plan from a preregistration and its reviews.

The plan is the object the dataset builder is not allowed to invent: it binds
each event to a review file by SHA, each review to a capture by SHA, and each
capture to a replay and a collection config by SHA.  This tool only assembles and
re-checks those bindings; it never writes a review, never chooses a window and
never edits a tolerance.

Run it once per round into a new filename.  An existing plan is refused.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from touhou_rl import f2_dataset, f2_labels  # noqa: E402
from touhou_rl.f2_dataset import file_sha  # noqa: E402


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _ref(root, path):
    path = Path(path).resolve()
    return dict(path=path.relative_to(root).as_posix(), sha256=file_sha(path))


def build(preregistration, review_root, out, *, events_per_source=3):
    preregistration = Path(preregistration).resolve()
    review_root = Path(review_root).resolve()
    out = Path(out).resolve()
    if out.exists():
        raise SystemExit('plan already exists; a frozen plan is written once')
    root = ROOT.resolve()
    config = json.loads(json.dumps(f2_dataset.DEFAULT_CONFIG))
    spec = _json(preregistration)
    if spec.get('frozen') is not True:
        raise SystemExit('frozen preregistration required')
    sources, events, report = [], [], []
    for source in spec['sources']:
        role = source['role']
        capture = (root / source['capture']).resolve()
        manifest = capture / 'frame-proof.json'
        collect = capture / 'collect-report.json'
        teacher = capture / 'audit-state.jsonl'
        replay = (root / source['replay']).resolve()
        if file_sha(replay) != source['sha256']:
            raise SystemExit(f'{role}: replay SHA differs from the preregistration')
        collective = _json(collect).get('capture_identity', _json(collect))
        collection_sha = collective.get('config_sha256')
        if collective.get('replay_sha256') != file_sha(replay) or collective.get('exact_manifest_sha256') != file_sha(manifest):
            raise SystemExit(f'{role}: collection config does not bind this replay/capture')
        sources.append(dict(
            id=f'f2-{role}-st2',
            replay=_ref(root, replay),
            exact_manifest=_ref(root, manifest),
            collect_config=_ref(root, collect),
            teacher_artifact=_ref(root, teacher),
            collection_config_sha256=collection_sha))
        for index in range(1, events_per_source + 1):
            review = review_root / f'{role}-st2' / f'e{index}-review.json'
            document = _json(review)
            event_id = f'f2-{role}-st2-e{index}'
            if document.get('frozen') is not True:
                raise SystemExit(f'{event_id}: review must be frozen')
            if document.get('event_id') != event_id or document.get('source_id') != f'f2-{role}-st2':
                raise SystemExit(f'{event_id}: review identity differs from the plan')
            if document.get('capture_manifest_sha256') != file_sha(manifest):
                raise SystemExit(f'{event_id}: review capture manifest SHA differs from the capture')
            if document.get('teacher_artifact_sha256') != file_sha(teacher):
                raise SystemExit(f'{event_id}: review teacher artifact SHA differs from the capture')
            rows = document.get('rows') or []
            if len(rows) < config['min_event_rows']:
                raise SystemExit(f'{event_id}: fewer than {config["min_event_rows"]} rows')
            events.append(dict(
                event_id=event_id, source_id=f'f2-{role}-st2', stage=source['stage'], split=role,
                template_group=f'st2-volley-{role}-e{index}',
                episode_id=f'{replay.stem}-stage2-e{index}',
                derived_group=f'none-{role}-st2-e{index}',
                review=_ref(root, review)))
            annotated = sum(1 for r in rows if r.get('status') != 'context_only')
            report.append(dict(event_id=event_id, rows=len(rows), annotated=annotated,
                               context_only=len(rows) - annotated,
                               stage_frames=[rows[0]['observation']['frame'], rows[-1]['observation']['frame']],
                               position_tolerance=document['tolerances']['position_px'],
                               review_sha256=file_sha(review)))
    plan = dict(schema=f2_dataset.PLAN_SCHEMA, frozen=True,
                preregistered_at=spec['preregistered_at'],
                note='Row set fixed by the frozen window rule; aim is a hitbox miss distance; every row of every window was attempted.',
                preregistration=dict(path=preregistration.relative_to(root).as_posix(),
                                     sha256=file_sha(preregistration)),
                config=config, config_sha256=f2_labels.evidence_sha(config),
                sources=sources, events=events)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
    return dict(plan=str(out), sha256=file_sha(out), events=report,
                sources=[dict(id=s['id'], replay=s['replay']['sha256'],
                              exact_manifest=s['exact_manifest']['sha256']) for s in sources])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preregistration', type=Path, required=True)
    parser.add_argument('--review-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--events-per-source', type=int, default=3)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.preregistration, args.review_root, args.out,
                           events_per_source=args.events_per_source), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
