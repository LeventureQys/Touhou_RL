"""Strict, atomic, diagnostic chunk-safety cache. Never consumes legacy risk labels."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from .chunk_safety import ChunkSafetyConfig, predict

SCHEMA = 'v200-chunk-safety-cache/1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def local(root, name):
    path = (root / name).resolve()
    if path.parent != root.resolve():
        raise ValueError('source must be a direct child')
    return path


def audit_source(rollout):
    """Verify report hashes and bidirectional trace/shard identity before labelling."""
    rollout = Path(rollout)
    report = json.loads((rollout / 'student-rollout-report.json').read_text(encoding='utf-8'))
    if report.get('training_approved') is not False:
        raise ValueError('requires explicitly diagnostic source')
    if not report.get('rollout_id'):
        raise ValueError('missing rollout identity')
    rows, sources, seen_ep = [], {}, set()
    for ep in report['episodes']:
        if ep['episode'] in seen_ep:
            raise ValueError('duplicate episode')
        seen_ep.add(ep['episode'])
        path = local(rollout, ep['trace'])
        sources[path.name] = sha(path)
        if sources[path.name] != ep['trace_sha256']:
            raise ValueError('trace SHA mismatch')
        trace = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        if len(trace) != ep['decisions']:
            raise ValueError('trace length mismatch')
        for i, row in enumerate(trace):
            if row['episode'] != ep['episode'] or row['decision'] != i or row['rollout_id'] != report['rollout_id']:
                raise ValueError('trace order/identity mismatch')
        rows.extend(trace)
    aligned = [r for r in rows if r.get('sample_index') is not None]
    if [r['sample_index'] for r in aligned] != list(range(report['samples'])):
        raise ValueError('sample order mismatch')
    cursor, names = 0, set()
    for item in report['shards']:
        path = local(rollout, item['name'])
        if path.name in names:
            raise ValueError('duplicate shard')
        names.add(path.name)
        sources[path.name] = sha(path)
        if sources[path.name] != item['sha256']:
            raise ValueError('shard SHA mismatch')
        with np.load(path, allow_pickle=False) as z:
            required = {'meta', 'decision_meta', 'executed_action', 'action'}
            if not required <= set(z.files):
                raise ValueError('missing shard alignment arrays')
            n = len(z['meta'])
            if any(len(z[k]) != n for k in required):
                raise ValueError('shard length mismatch')
            if z['meta'].shape != (n, 8) or z['decision_meta'].shape != (n, 4) or z['executed_action'].shape != (n, 4) or z['action'].shape != (n, 4):
                raise ValueError('shard shape mismatch')
            for j in range(n):
                if cursor >= len(aligned):
                    raise ValueError('extra shard row')
                r = aligned[cursor]
                checks = [(z['decision_meta'][j], [r[k] for k in ('episode','decision','frame_delta','stage_delta')]),
                          (z['meta'][j, :2], [r['frame'], r['stage']]),
                          (z['executed_action'][j], r['executed_action']), (z['action'][j], r['teacher_action'])]
                if any(not np.array_equal(a, b) for a, b in checks):
                    raise ValueError(f'trace/shard mismatch at sample {cursor}')
                if not r.get('aligned') or not r.get('fresh_pixels'):
                    raise ValueError('unaligned sample in shard')
                for key in ('executed_action', 'teacher_action'):
                    a = r[key]
                    if len(a) != 4 or any(type(x) is not int for x in a) or not (0 <= a[0] < 9) or any(x not in (0,1) for x in a[1:]):
                        raise ValueError('invalid action mapping')
                cursor += 1
    if {p.name for p in rollout.glob('student-state-*.npz')} - names:
        raise ValueError('undeclared student shard')
    if cursor != len(aligned):
        raise ValueError('missing shard rows')
    return report, rows, aligned, sources


def build(rollout, out, config=None):
    rollout, out = Path(rollout), Path(out)
    if out.exists():
        raise FileExistsError(out)
    config = config or ChunkSafetyConfig()
    config.validate()
    report, rows, aligned, sources = audit_source(rollout)
    if report['frameskip'] != config.frameskip:
        raise ValueError('frameskip mismatch')
    labels = []
    for row in aligned:
        snapshot = row.get('raw_safety_snapshot')
        if snapshot is not None and (snapshot.get('frame') != row['frame'] or snapshot.get('stage') != row['stage']):
            raise ValueError('snapshot stamp mismatch')
        result = predict(snapshot or {}, config)
        result.update(sample_index=np.int64(row['sample_index']), episode=np.int64(row['episode']), decision=np.int64(row['decision']))
        labels.append(result)
    if not labels:
        raise ValueError('no aligned samples')
    arrays = {k: np.stack([r[k] for r in labels]) for k in labels[0]}
    teacher_sources = {name: sha(Path(__file__).with_name(name)) for name in ('chunk_safety.py', 'build_chunk_safety_cache.py', 'spec.py')}
    identity = dict(schema=SCHEMA, source=str(rollout.resolve()), source_report_sha256=sha(rollout/'student-rollout-report.json'),
                    source_files=sources, rollout_id=report['rollout_id'], checkpoint_sha256=report['checkpoint_sha256'],
                    behavior_teacher_source_sha256=report['teacher_source_sha256'], behavior_teacher_config_digest=digest(report['teacher_config']),
                    teacher_sources=teacher_sources, config=asdict(config), config_digest=digest(asdict(config)),
                    candidate_mapping='focus*9+move', durations=[1,3,5], rows=len(labels), training_approved=False,
                    semantics='forced hold, linear existing bullets; model prediction only; world risk always unknown',
                    model_valid_cells=int(arrays['model_valid'].sum()), world_risk_valid_cells=int(arrays['world_risk_valid'].sum()),
                    missing_raw_snapshots=sum(r.get('raw_safety_snapshot') is None for r in aligned))
    out.parent.mkdir(parents=True, exist_ok=True)
    # Same-volume temporary directory. No partial directory is ever published.
    temporary = Path(tempfile.mkdtemp(prefix=out.name+'.pending-', dir=out.parent))
    np.savez(temporary/'chunk-arrays.npz', **arrays)
    serial = []
    for row in labels:
        # JSON null for unknown floating targets; no non-standard NaN JSON.
        serial.append({k: (np.where(np.isfinite(v), v, None).tolist() if np.issubdtype(np.asarray(v).dtype, np.floating) else np.asarray(v).tolist()) for k,v in row.items()})
    (temporary/'chunk-rows.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in serial), encoding='utf-8')
    identity['outputs'] = {name: sha(temporary/name) for name in ('chunk-arrays.npz', 'chunk-rows.jsonl')}
    (temporary/'chunk-identity.json').write_text(json.dumps(identity, indent=2, ensure_ascii=False), encoding='utf-8')
    validate(temporary, verify_source=True)
    # Recheck the source just before publish (including report identity).
    if sha(rollout/'student-rollout-report.json') != identity['source_report_sha256'] or any(sha(rollout/n) != h for n,h in sources.items()):
        raise ValueError('source changed during build')
    os.rename(temporary, out)
    return identity


def validate(cache, verify_source=True):
    cache = Path(cache)
    identity = json.loads((cache/'chunk-identity.json').read_text(encoding='utf-8'))
    if identity['schema'] != SCHEMA or identity['candidate_mapping'] != 'focus*9+move' or identity['durations'] != [1,3,5]:
        raise ValueError('cache mapping/schema mismatch')
    if digest(identity['config']) != identity['config_digest']:
        raise ValueError('config digest mismatch')
    for name,h in identity['teacher_sources'].items():
        if sha(Path(__file__).with_name(name)) != h:
            raise ValueError('teacher source changed')
    for name,h in identity['outputs'].items():
        if sha(local(cache,name)) != h:
            raise ValueError('output SHA mismatch')
    rows = [json.loads(s) for s in (cache/'chunk-rows.jsonl').read_text(encoding='utf-8').splitlines()]
    n = identity['rows']
    if len(rows) != n:
        raise ValueError('output length mismatch')
    shapes = {'clearance':(n,18,3),'model_valid':(n,18,3),'collision_predicted':(n,18,3),'world_risk_valid':(n,18,3),
              'duration_class':(n,18),'duration_valid':(n,18),'sample_index':(n,), 'episode':(n,), 'decision':(n,)}
    with np.load(cache/'chunk-arrays.npz', allow_pickle=False) as z:
        if set(z.files) != set(shapes):
            raise ValueError('missing/extra target arrays')
        for k,shape in shapes.items():
            if z[k].shape != shape:
                raise ValueError('target shape mismatch')
            for i,r in enumerate(rows):
                expected = np.asarray(r[k], dtype=z[k].dtype)
                if not np.array_equal(z[k][i],expected,equal_nan=True):
                    raise ValueError('JSONL/array mismatch')
        if not np.array_equal(z['sample_index'], np.arange(n)) or np.any(z['world_risk_valid']):
            raise ValueError('invalid identity/world mask')
        for k in ('model_valid','collision_predicted','world_risk_valid','duration_valid'):
            if not np.isin(z[k], [0,1]).all():
                raise ValueError('invalid mask')
        if not np.isin(z['duration_class'], [-1,0,1,2]).all():
            raise ValueError('invalid duration class')
    if verify_source:
        source = Path(identity['source'])
        if sha(source/'student-rollout-report.json') != identity['source_report_sha256']:
            raise ValueError('source report changed')
        report,_,aligned,sources = audit_source(source)
        if sources != identity['source_files'] or len(aligned) != n:
            raise ValueError('source identity mismatch')
        for r,a in zip(rows, aligned):
            if any(r[k] != a[k] for k in ('sample_index','episode','decision')):
                raise ValueError('source/output alignment mismatch')
        with np.load(cache/'chunk-arrays.npz', allow_pickle=False) as z:
            config = ChunkSafetyConfig(**identity['config'])
            for i,a in enumerate(aligned):
                expected = predict(a.get('raw_safety_snapshot') or {}, config)
                for k,v in expected.items():
                    if not np.array_equal(z[k][i], v, equal_nan=True):
                        raise ValueError('teacher/output target mismatch')
    return identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rollout', required=True); p.add_argument('--out', required=True)
    a=p.parse_args(); print(json.dumps(build(a.rollout,a.out),ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
