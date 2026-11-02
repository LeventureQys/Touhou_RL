"""P4 sequence research interface, NOT a deployable candidate or certification.

Cache schema v373-event-feature-cache/2: JSON index binds manifest_sha256 and
baseline_sha256 and a hashed /2 build report with a required compute_protocol;
/1 indices (including CPU) are rejected, never implicitly upgraded. Protocol binds
canonical device/name, batch16/32, FP32/no autocast/no TF32, torch version and source
hashes. Each weights-only .pt is
{schema,event_id,baseline_sha256,pixel_sha256,history_sha256,feature_source,
 features: float32[T,F],base_logits: float32[T,18]}. feature_source must equal
'frozen-backbone.forward_core/real_pixels'. Pixel assets are weights-only region
mappings [T,depth,H,W,C]; history assets contain the exact TacticalPolicy history
fields [T,...]. Label assets use tactical_policy_loss's independent mask schema.
Every train/validation feature is recomputed from hashed pixels before training.
Test metadata is audited by the existing auditor; no test tensor is loaded here.
No cache-builder, real-pixel certification or deployment export is provided.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import torch
from .tactical_train import validate_training_gate
from .tactical_data import audit_manifest, file_sha
from .tactical_policy import TacticalConfig, TacticalPolicy, bos_history, tactical_policy_loss

CACHE_SCHEMA = 'v373-event-feature-cache/2'
REPORT_SCHEMA = 'v373-feature-cache-build/2'
SOURCE = 'frozen-backbone.forward_core/real_pixels'


@dataclass(frozen=True)
class SequenceConfig:
    arm: str = 'N1'
    seed: int = 7
    epochs: int = 30
    max_updates: int = 1000
    learning_rate: float = 1e-3
    burnin: int = 16
    sequence: int = 32
    device: str = 'cpu'

    def __post_init__(self):
        if self.arm not in {'N0', 'N1', 'B1'}:
            raise ValueError('invalid arm')
        if type(self.seed) is not int or type(self.epochs) is not int or not 1 <= self.epochs <= 30:
            raise ValueError('seed integer and epochs 1..30 required')
        if type(self.max_updates) is not int or self.max_updates < 1:
            raise ValueError('positive shared update budget required')
        if not 0 < self.learning_rate <= .01 or (self.burnin, self.sequence) != (16, 32):
            raise ValueError('fixed burnin16/sequence32 and finite bounded learning rate required')


def _asset(root, ref, relative_to=None):
    path = ((relative_to or root) / ref['path']).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('asset escapes dataset root')
    if file_sha(path) != ref['sha256']:
        raise ValueError('asset SHA mismatch')
    return path


def _tensor_asset(root, ref):
    return torch.load(_asset(root, ref), map_location='cpu', weights_only=True)


def load_backbone(path):
    from .vmodels import load_vision_checkpoint, vision_arch
    model, _ = load_vision_checkpoint(path, map_location='cpu')
    if vision_arch(model) != 'v3-st':
        raise ValueError('v3-st backbone required')
    return model


def _finite_float(x, shape, name):
    if not isinstance(x, torch.Tensor) or x.dtype != torch.float32 or tuple(x.shape) != tuple(shape) or not torch.isfinite(x).all():
        raise ValueError(f'{name}: finite float32 {shape} required')



@contextmanager
def _fp32_mode(device):
    """Local shared helper: no import of the builder (avoids a cycle)."""
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.autocast(device_type=device.type, enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


def _source_hashes():
    names = ('tactical_feature_cache.py', 'tactical_sequence_train.py', 'tactical_train.py',
             'tactical_data.py', 'vmodels.py', 'vmodels3.py')
    return {name: file_sha(Path(__file__).with_name(name)) for name in names}


def _compute_protocol(device, batch_size):
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda') or (device.type == 'cpu' and device.index is not None):
        raise ValueError('CPU or CUDA required')
    if type(batch_size) is not int or batch_size not in (16, 32):
        raise ValueError('batch_size must be 16 or 32')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('cache protocol requires CUDA; CPU fallback forbidden')
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    return dict(device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                batch_size=batch_size, precision='float32',
                autocast=False, tf32=False, feature_source=SOURCE,
                source_sha256=_source_hashes(), torch_version=str(torch.__version__))


def _cache_protocol(index, root):
    """Require hashed report and current source/protocol; /1 is never compatible."""
    ref = index.get('report')
    if not isinstance(ref, dict):
        raise ValueError('hashed cache report required')
    report = json.loads(_asset(root, ref).read_text(encoding='utf-8'))
    protocol = index.get('compute_protocol')
    if not isinstance(protocol, dict):
        raise ValueError('cache compute protocol required')
    try:
        expected = _compute_protocol(protocol['device'], protocol['batch_size'])
    except (KeyError, TypeError, RuntimeError) as exc:
        raise ValueError('cache compute protocol rejected') from exc
    if protocol != expected or report.get('compute_protocol') != protocol:
        raise ValueError('cache compute protocol/source mismatch')
    if report.get('schema') != REPORT_SCHEMA or any(
        report.get(k) != index.get(k) for k in
        ('manifest_sha256', 'baseline_sha256', 'feature_source', 'source_sha256')
    ) or index.get('feature_source') != SOURCE or index.get('source_sha256') != expected['source_sha256']:
        raise ValueError('cache report binding rejected')
    if any(report.get(k) != protocol[k] for k in
           ('device', 'device_name', 'batch_size', 'precision', 'autocast', 'tf32', 'torch_version')):
        raise ValueError('cache report protocol mismatch')
    return protocol


def _pixel_mapping(pixels, rows):
    if not isinstance(pixels, dict) or not pixels or any(
        not isinstance(x, torch.Tensor) or x.ndim != 5 or x.shape[0] != rows
        or x.dtype != torch.float32 or not torch.isfinite(x).all()
        for x in pixels.values()
    ):
        raise ValueError('finite float32 preprocessed pixel region mapping [T,depth,H,W,C] required')


def prepare(manifest, cache_index, baseline):
    """Fail closed before optimizer/output; no bypass flags or synthetic mode."""
    manifest, cache_index, baseline = map(Path, (manifest, cache_index, baseline))
    obj = validate_training_gate(manifest)
    audit = audit_manifest(manifest)
    if not audit.get('passed') or audit.get('manifest_sha256') != file_sha(manifest):
        raise ValueError('manifest audit rejected')
    index = json.loads(cache_index.read_text(encoding='utf-8'))
    sha = file_sha(baseline)
    if index.get('schema') != CACHE_SCHEMA or index.get('baseline_sha256') != sha or index.get('manifest_sha256') != file_sha(manifest):
        raise ValueError('cache schema/baseline/manifest binding rejected')
    protocol = _cache_protocol(index, cache_index.parent)
    device = torch.device(protocol['device'])
    events, groups = {}, {}
    for ref in obj['events']:
        event_path = _asset(manifest.parent, ref)
        event = json.loads(event_path.read_text(encoding='utf-8'))
        # Normalize refs to manifest root after resolving against the event file.
        for section in (event['inputs']['pixels'], event['inputs']['history'], event['labels']):
            asset = (event_path.parent / section['path']).resolve()
            if not asset.is_relative_to(manifest.parent.resolve()):
                raise ValueError('event asset escapes dataset root')
            section['path'] = str(asset.relative_to(manifest.parent.resolve()))
        eid, group, split = event['event_id'], event['template_group'], event['split']
        if eid in events or (group in groups and groups[group] != split):
            raise ValueError('event/template leakage')
        if group not in obj['split_groups'][split]:
            raise ValueError('event group not in declared split')
        events[eid], groups[group] = event, split
    if set(groups.values()) != {'train', 'validation', 'test'}:
        raise ValueError('three grouped splits required')
    if any(set(obj['split_groups'][s]) != {g for g, split in groups.items() if split == s}
           for s in ('train', 'validation', 'test')):
        raise ValueError('declared/actual split groups differ')
    refs = {}
    for ref in index.get('events', []):
        if ref['event_id'] in refs or ref['event_id'] not in events:
            raise ValueError('duplicate/unknown cache event')
        refs[ref['event_id']] = ref
    wanted = {eid for eid, e in events.items() if e['split'] != 'test'}
    if not wanted <= refs.keys():
        raise ValueError('train/validation caches missing')
    backbone = load_backbone(baseline).to(device=device, dtype=torch.float32).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    data = {'train': [], 'validation': []}
    summaries = []
    for eid in sorted(wanted):
        e = events[eid]
        cache = _tensor_asset(cache_index.parent, refs[eid])
        expected = dict(schema=CACHE_SCHEMA, event_id=eid, baseline_sha256=sha,
                        pixel_sha256=e['inputs']['pixels']['sha256'],
                        history_sha256=e['inputs']['history']['sha256'], feature_source=SOURCE)
        if not isinstance(cache, dict) or set(cache) != set(expected) | {'features', 'base_logits'} or any(cache.get(k) != v for k, v in expected.items()):
            raise ValueError('cache real pixel provenance rejected')
        t = len(e['steps'])
        features, logits = cache['features'], cache['base_logits']
        _finite_float(features, (t, backbone.config.trunk_hidden), 'features')
        _finite_float(logits, (t, 18), 'base_logits')
        if t < 48:
            raise ValueError('event needs at least burnin16+sequence32 rows')
        pixels = _tensor_asset(manifest.parent, e['inputs']['pixels'])
        _pixel_mapping(pixels, t)
        recomputed_features, recomputed_logits = [], []
        with _fp32_mode(device), torch.inference_mode():
            for offset in range(0, t, protocol['batch_size']):
                end = min(t, offset + protocol['batch_size'])
                core = backbone.forward_core({k: x[offset:end].to(device) for k, x in pixels.items()})
                f, l = core['features'], core['policy_logits']
                _finite_float(f, (end-offset, backbone.config.trunk_hidden), 'features')
                _finite_float(l, (end-offset, 18), 'base_logits')
                cached_f, cached_l = features[offset:end].to(device), logits[offset:end].to(device)
                if not torch.allclose(backbone.policy_head(cached_f), cached_l, atol=1e-6, rtol=1e-5):
                    raise ValueError('cache base logits disagree with backbone.policy_head')
                if not torch.allclose(f, cached_f, atol=1e-6, rtol=1e-5):
                    raise ValueError('cache features disagree with actual pixels')
                if not torch.allclose(l, cached_l, atol=1e-6, rtol=1e-5):
                    raise ValueError('cache base logits disagree with actual pixels')
                recomputed_features.append(f.cpu().clone())
                recomputed_logits.append(l.cpu().clone())
        # Normal CPU tensors (not inference tensors) are consumed by head training.
        features = torch.cat(recomputed_features).clone()
        logits = torch.cat(recomputed_logits).clone()
        history = _tensor_asset(manifest.parent, e['inputs']['history'])
        required = bos_history(features[None])
        if not isinstance(history, dict) or set(history) != set(required):
            raise ValueError('actual history fields required')
        for k, sample in required.items():
            x = history[k]
            if not isinstance(x, torch.Tensor) or x.shape != sample.shape[1:] or x.dtype != sample.dtype:
                raise ValueError('history shape/dtype mismatch')
        for i, row in enumerate(e['steps']):
            actual = row.get('previous_actual_candidate')
            if bool(history['history_valid'][i]) != (actual is not None):
                raise ValueError('history must reflect acknowledged actual execution')
            if actual is not None and (history['previous_candidate'][i].item() != actual or history['previous_hold'][i].item() != row['previous_actual_hold'] or history['frame_delta'][i].item() != row['frame']-row['previous_effective_frame']):
                raise ValueError('cached actual history mismatch')
            # Option/target cannot be inferred from teacher labels.
            if actual is not None:
                for key in ('previous_actual_option', 'previous_actual_target', 'previous_target_valid'):
                    if key not in row:
                        raise ValueError('actual option/target evidence missing')
                if history['previous_option'][i].item() != row['previous_actual_option'] or bool(history['target_valid'][i]) != row['previous_target_valid'] or history['previous_target'][i].tolist() != row['previous_actual_target']:
                    raise ValueError('cached actual option/target mismatch')
        labels = _tensor_asset(manifest.parent, e['labels'])
        if not isinstance(labels, dict) or any(not isinstance(x, torch.Tensor) or x.shape[:1] != (t,) for x in labels.values()):
            raise ValueError('labels require independent event row tensors')
        data[e['split']].append(dict(event_id=eid, features=features, base_logits=logits, history=history, labels=labels))
        summaries.append(dict(event_id=eid, split=e['split'], rows=t, pixels_sha256=expected['pixel_sha256'], cache_sha256=refs[eid]['sha256'], source=SOURCE))
    return backbone.cpu(), data, dict(compute_protocol=protocol, cache_report_sha256=index['report']['sha256'], baseline_sha256=sha, manifest_sha256=file_sha(manifest), cache_index_sha256=file_sha(cache_index), sources=summaries)


def windows(events):
    """Nonoverlapping scored chunks, each with preceding 16 actual-history rows.

    First 16 rows and incomplete tails are deliberately unscored in this MVP.
    No padding and no state/history crosses an event boundary.
    """
    return [(e, start) for e in events for start in range(16, len(e['features'])-31, 32)]


def window_loss(model, event, start, device):
    def part(x, a, b):
        return x[a:b][None].to(device)
    def forward(a, b, hidden=None):
        return model.forward_features(part(event['features'], a, b), part(event['base_logits'], a, b),
            {k: part(v, a, b) for k, v in event['history'].items()}, hidden=hidden)
    with torch.no_grad():
        hidden = forward(start-16, start)['hidden'].detach()
    out = forward(start, start+32, hidden)
    labels = {k: part(v, start, start+32) for k, v in event['labels'].items()}
    return tactical_policy_loss(out, labels)


def train(manifest, cache_index, baseline, out, config=SequenceConfig()):
    """Uniform event-window/update/epoch budget for N0/N1/B1, validation-only selection.

    Writes research-only head weights, not the deployable tactical policy schema.
    A successful run still does not establish teacher credibility or deployment safety.
    """
    out = Path(out)
    if out.exists():
        raise FileExistsError('new output directory required')
    backbone, data, provenance = prepare(manifest, cache_index, baseline)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if str(config.device).startswith('cuda'):
        torch.cuda.manual_seed_all(config.seed)
    model = TacticalPolicy(backbone, TacticalConfig(arm=config.arm, feature_dim=backbone.config.trunk_hidden)).to(config.device)
    frozen = {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}
    params = [p for p in model.parameters() if p.requires_grad]
    # Validate even burn-in/tail rows before an optimizer can mutate any head.
    model.eval()
    with torch.no_grad():
        for events in data.values():
            for event in events:
                output = model.forward_features(event['features'][None].to(config.device),
                    event['base_logits'][None].to(config.device),
                    {k: v[None].to(config.device) for k, v in event['history'].items()})
                tactical_policy_loss(output, {k: v[None].to(config.device) for k, v in event['labels'].items()})
    optimizer = torch.optim.Adam(params, lr=config.learning_rate)
    training, validation = windows(data['train']), windows(data['validation'])
    if not training or not validation:
        raise ValueError('train and validation windows required')
    updates, logs, best, best_state = 0, [], float('inf'), None
    rng = random.Random(config.seed)
    for epoch in range(config.epochs):
        model.train()
        order = list(training)
        rng.shuffle(order)
        skipped = 0
        for event, start in order:
            if updates >= config.max_updates:
                break
            optimizer.zero_grad(set_to_none=True)
            result = window_loss(model, event, start, config.device)
            if not result['has_supervision']:
                skipped += 1
                continue
            result['loss'].backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            logs.append(dict(epoch=epoch, step=updates, event_id=event['event_id'], start=start, loss=float(result['loss'].detach()), counts=result['counts'], grad_norm=float(norm)))
        model.eval()
        with torch.no_grad():
            scores = [window_loss(model, e, s, config.device) for e, s in validation]
        supervised = [float(x['loss']) for x in scores if x['has_supervision']]
        if not supervised:
            raise ValueError('validation has no supervision; cannot select')
        score = sum(supervised)/len(supervised)
        logs.append(dict(epoch=epoch, validation_loss=score, skipped_unknown=skipped, updates=updates))
        if updates and score < best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.heads.state_dict().items()}
        if updates >= config.max_updates:
            break
    if any(not torch.equal(v.detach().cpu(), frozen[k]) for k, v in backbone.state_dict().items()):
        raise RuntimeError('frozen backbone changed')
    if file_sha(baseline) != provenance['baseline_sha256'] or file_sha(manifest) != provenance['manifest_sha256'] or file_sha(cache_index) != provenance['cache_index_sha256']:
        raise ValueError('source changed during run')
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='v373-sequence-research-run/1', config=asdict(config), **provenance,
                  updates=updates, validation_loss=best if best_state is not None else None,
                  deployable=False, test_used=False, backbone_unchanged=True,
                  limitations='Interface MVP; independent pixel/teacher certification still required. Not a deployed candidate.')
    if best_state is not None:
        with (out/'research-heads.pt').open('xb') as stream:
            torch.save(dict(schema='v373-sequence-research-only/1', heads=best_state, report=report), stream)
    (out/'log.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in logs), encoding='utf-8')
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    hashes = {p.name: file_sha(p) for p in out.iterdir() if p.is_file()}
    (out/'sha256.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'cache-index', 'baseline', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--arm', choices=('N0', 'N1', 'B1'), default='N1')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--max-updates', type=int, default=1000)
    p.add_argument('--device', default='cpu')
    a = p.parse_args(argv)
    print(json.dumps(train(a.manifest, a.cache_index, a.baseline, a.out,
        SequenceConfig(arm=a.arm, seed=a.seed, epochs=a.epochs, max_updates=a.max_updates, device=a.device)), indent=2))


if __name__ == '__main__':
    main()
