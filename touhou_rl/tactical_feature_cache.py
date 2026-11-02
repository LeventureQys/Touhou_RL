"""Build research feature caches, never train or certify real-image provenance.

Production gates and byte-integrity audit precede baseline loading and output.
Test assets are hashed by the auditor, never deserialized or cached. SOURCE is
an interface identifier, NOT independent proof of real capture or freshness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from .tactical_data import audit_manifest, file_sha
from .tactical_train import validate_training_gate
from .tactical_sequence_train import (
    CACHE_SCHEMA, REPORT_SCHEMA, SOURCE, _asset, _finite_float, load_backbone,
    _compute_protocol, _fp32_mode, _pixel_mapping, _source_hashes,
)


def _events(manifest, obj):
    root = manifest.parent.resolve()
    events, groups, assets = {}, {}, []
    for ref in obj['events']:
        path = _asset(root, ref)
        event = json.loads(path.read_text(encoding='utf-8'))
        assets.append((path, ref['sha256']))
        for section in (event['inputs']['pixels'], event['inputs']['history'], event['labels']):
            # Actual refs are relative to the event JSON, not the manifest.
            asset = _asset(root, section, relative_to=path.parent)
            section['path'] = str(asset.relative_to(root))
            assets.append((asset, section['sha256']))
        eid, group, split = event['event_id'], event['template_group'], event['split']
        if eid in events or (group in groups and groups[group] != split):
            raise ValueError('event/template leakage')
        if split not in ('train', 'validation', 'test') or group not in obj['split_groups'][split]:
            raise ValueError('event group not in declared split')
        if split != 'test' and len(event['steps']) < 48:
            raise ValueError('event needs at least burnin16+sequence32 rows')
        events[eid], groups[group] = event, split
    if set(groups.values()) != {'train', 'validation', 'test'} or any(
        set(obj['split_groups'][s]) != {g for g, split in groups.items() if split == s}
        for s in ('train', 'validation', 'test')
    ):
        raise ValueError('declared/actual split groups differ')
    return events, assets


def build(manifest, baseline, out, *, device='cpu', batch_size=16):
    """Create a NEW cache directory; no bypass, synthetic mode, AMP or test cache.

    Only train/validation pixel mappings are torch.load'ed (weights_only=True).
    History and labels, including test assets, receive byte-hash audit only.
    A failed build may leave partial event files, but publishes index.json last.
    It must be retried in another new directory, never overwrite an old cache.
    """
    manifest, baseline, out = map(Path, (manifest, baseline, out))
    if type(batch_size) is not int or batch_size not in (16, 32):
        raise ValueError('batch_size must be 16 or 32')
    protocol = _compute_protocol(device, batch_size)
    device = torch.device(protocol['device'])
    if out.exists():
        raise FileExistsError('new output directory required')
    obj = validate_training_gate(manifest)
    audit = audit_manifest(manifest)
    manifest_sha = file_sha(manifest)
    if not audit.get('passed') or audit.get('manifest_sha256') != manifest_sha:
        raise ValueError('manifest audit rejected')
    events, assets = _events(manifest, obj)
    baseline_sha = file_sha(baseline)
    backbone = load_backbone(baseline).to(device=device, dtype=torch.float32).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    frozen = {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}
    source_hashes = protocol['source_sha256']
    source_paths = [Path(__file__).with_name(name) for name in source_hashes]
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    refs, summaries, rows = [], [], 0
    with _fp32_mode(device), torch.inference_mode():
        for eid, event in sorted(events.items()):
            if event['split'] == 'test':
                continue
            tick = time.perf_counter()
            pixel_ref = event['inputs']['pixels']
            pixels = torch.load(_asset(manifest.parent, pixel_ref), map_location='cpu', weights_only=True)
            t = len(event['steps'])
            _pixel_mapping(pixels, t)
            features, logits = [], []
            for offset in range(0, t, batch_size):
                batch = {k: x[offset:offset+batch_size].to(device) for k, x in pixels.items()}
                core = backbone.forward_core(batch)
                n = min(batch_size, t-offset)
                f, l = core['features'], core['policy_logits']
                _finite_float(f, (n, backbone.config.trunk_hidden), 'features')
                _finite_float(l, (n, 18), 'base_logits')
                if not torch.allclose(backbone.policy_head(f), l, atol=1e-6, rtol=1e-5):
                    raise ValueError('forward_core logits disagree with policy_head')
                features.append(f.cpu().clone())
                logits.append(l.cpu().clone())
            cache = dict(schema=CACHE_SCHEMA, event_id=eid, baseline_sha256=baseline_sha,
                         pixel_sha256=pixel_ref['sha256'],
                         history_sha256=event['inputs']['history']['sha256'], feature_source=SOURCE,
                         features=torch.cat(features), base_logits=torch.cat(logits))
            # Numeric filenames cannot escape output via an arbitrary event_id.
            path = out / f'event-{len(refs):06d}.pt'
            torch.save(cache, path)
            ref = dict(event_id=eid, path=path.name, sha256=file_sha(path))
            refs.append(ref)
            rows += t
            summaries.append(dict(**ref, split=event['split'], rows=t,
                                  pixel_sha256=pixel_ref['sha256'],
                                  history_sha256=cache['history_sha256'], seconds=time.perf_counter()-tick))
            del pixels, features, logits, cache
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter()-start
    if any(not torch.equal(v.detach().cpu(), frozen[k]) for k, v in backbone.state_dict().items()):
        raise ValueError('frozen backbone changed')
    if file_sha(manifest) != manifest_sha or file_sha(baseline) != baseline_sha or any(
        file_sha(p) != sha for p, sha in assets
    ) or any(file_sha(p) != source_hashes[p.name] for p in source_paths):
        raise ValueError('manifest/baseline/source/assets changed during build')
    report = dict(schema=REPORT_SCHEMA, compute_protocol=protocol, manifest_sha256=manifest_sha,
                  baseline_sha256=baseline_sha, source_sha256=source_hashes, feature_source=SOURCE,
                  device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                  torch_version=str(torch.__version__), batch_size=batch_size, precision='float32',
                  autocast=False, tf32=False, backbone_frozen=True, rows=rows, seconds=elapsed,
                  rows_per_second=rows/elapsed, events=summaries,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                  test_tensor_loaded=False, test_cache_created=False,
                  real_image_source_independently_certified=False,
                  warning='Gates and hashes do not independently certify real-image capture, freshness or teacher quality.')
    report_path = out/'report.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    index = dict(schema=CACHE_SCHEMA, compute_protocol=protocol, manifest_sha256=manifest_sha, baseline_sha256=baseline_sha,
                 feature_source=SOURCE, source_sha256=source_hashes, events=refs,
                 report=dict(path=report_path.name, sha256=file_sha(report_path)))
    (out/'index.json').write_text(json.dumps(index, indent=2), encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, choices=(16, 32), default=16)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.manifest, args.baseline, args.out,
                           device=args.device, batch_size=args.batch_size), indent=2))


if __name__ == '__main__':
    main()
