"""CPU-only synthetic unit fixtures; monkeypatched gates are NOT real eligibility.

No deployable export, live game, production manifest or real training is used.
"""
import json
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from touhou_rl import tactical_sequence_train as seq
from touhou_rl.tactical_policy import bos_history


class SyntheticBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(trunk_hidden=4)
        self.policy_head = nn.Linear(4, 18)

    def forward_core(self, regions):
        features = regions['synthetic'].flatten(1)
        return dict(features=features, policy_logits=self.policy_head(features))


def dump_json(path, obj):
    path.write_text(json.dumps(obj), encoding='utf-8')
    return dict(path=path.name, sha256=seq.file_sha(path))


def dump_tensor(path, obj):
    torch.save(obj, path)
    return dict(path=path.name, sha256=seq.file_sha(path))


def write_index(path, obj):
    protocol = seq._compute_protocol('cpu', 16)
    obj.update(feature_source=seq.SOURCE, source_sha256=protocol['source_sha256'], compute_protocol=protocol)
    report = dict(schema=seq.REPORT_SCHEMA, **protocol,
                  compute_protocol=protocol, manifest_sha256=obj['manifest_sha256'],
                  baseline_sha256=obj['baseline_sha256'])
    obj['report'] = dump_json(path.parent/'build-report.json', report)
    return dump_json(path, obj)


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    """Only this fixture substitutes gates; baseline bytes explicitly synthetic."""
    torch.manual_seed(19)
    backbone = SyntheticBackbone()
    baseline = tmp_path/'synthetic-baseline.bin'
    baseline.write_bytes(b'SYNTHETIC UNIT FIXTURE NOT REAL BASELINE')
    manifest = tmp_path/'manifest.json'
    index = tmp_path/'index.json'
    events, caches, paths = [], [], {}
    for split in ('train', 'validation', 'test'):
        eid = 'synthetic-'+split
        features = torch.randn(48, 4)
        regions = {'synthetic': features.reshape(48, 1, 1, 1, 4)}
        pixels = dump_tensor(tmp_path/(eid+'-pixels.pt'), regions)
        history = {k: v[0] for k, v in bos_history(features[None]).items()}
        hist = dump_tensor(tmp_path/(eid+'-history.pt'), history)
        labels = {'action_positive': torch.ones(48, 18, dtype=torch.bool),
                  'action_valid': torch.ones(48, dtype=torch.bool),
                  'option_valid': torch.zeros(48, dtype=torch.bool)}
        lab = dump_tensor(tmp_path/(eid+'-labels.pt'), labels)
        event = dict(event_id=eid, template_group=eid, split=split,
                     inputs=dict(pixels=pixels, history=hist), labels=lab,
                     steps=[dict(frame=i, capture_interval=[i, i], alignment_status='verified',
                                 pool_complete=True) for i in range(48)])
        event_path = tmp_path/(eid+'.json')
        events.append(dump_json(event_path, event))
        cache = dict(schema=seq.CACHE_SCHEMA, event_id=eid,
                     baseline_sha256=seq.file_sha(baseline), pixel_sha256=pixels['sha256'],
                     history_sha256=hist['sha256'], feature_source=seq.SOURCE,
                     features=features, base_logits=backbone.policy_head(features).detach())
        cache_path = tmp_path/(eid+'-cache.pt')
        caches.append(dict(event_id=eid, **dump_tensor(cache_path, cache)))
        paths[split] = dict(cache=cache_path, event=event_path, history=tmp_path/hist['path'], labels=tmp_path/lab['path'])
    dump_json(manifest, dict(schema='v373-paired-events/1', events=events,
                           split_groups={s: ['synthetic-'+s] for s in ('train', 'validation', 'test')}))
    write_index(index, dict(schema=seq.CACHE_SCHEMA, baseline_sha256=seq.file_sha(baseline),
                         manifest_sha256=seq.file_sha(manifest), events=caches))
    monkeypatch.setattr(seq, 'validate_training_gate', lambda p: json.loads(p.read_text()))
    monkeypatch.setattr(seq, 'audit_manifest', lambda p: dict(passed=True, manifest_sha256=seq.file_sha(p)))
    monkeypatch.setattr(seq, 'load_backbone', lambda p: backbone)
    return SimpleNamespace(root=tmp_path, baseline=baseline, manifest=manifest, index=index, paths=paths, backbone=backbone)


def rehash_cache(f, split):
    obj = json.loads(f.index.read_text())
    for ref in obj['events']:
        if ref['event_id'] == 'synthetic-'+split:
            ref['sha256'] = seq.file_sha(f.paths[split]['cache'])
    dump_json(f.index, obj)


def test_real_unqualified_cache_rejected_before_backbone(tmp_path, monkeypatch):
    manifest = tmp_path/'unqualified-real.json'
    dump_json(manifest, dict(schema='v373-paired-events/1', input_kind='real_pixels', teacher_gate='failed'))
    monkeypatch.setattr(seq, 'load_backbone', lambda p: pytest.fail('must not touch baseline'))
    with pytest.raises(ValueError, match='training gate rejected'):
        seq.prepare(manifest, tmp_path/'nonexistent-cache', tmp_path/'nonexistent-baseline')


def test_audit_failure_precedes_cache(synthetic, monkeypatch):
    f = synthetic
    monkeypatch.setattr(seq, 'audit_manifest', lambda p: dict(passed=False))
    with pytest.raises(ValueError, match='audit rejected'):
        seq.prepare(f.manifest, f.root/'missing', f.baseline)


@pytest.mark.parametrize('field,value,match', [
    ('feature_source', 'fabricated', 'provenance'),
    ('baseline_sha256', '0'*64, 'provenance'),
    ('features', torch.zeros(48, 4, dtype=torch.float64), 'float32'),
    ('base_logits', torch.zeros(48, 18), 'policy_head'),
])
def test_bad_cache_rejected(synthetic, field, value, match):
    f = synthetic
    cache = torch.load(f.paths['train']['cache'], weights_only=True)
    cache[field] = value
    torch.save(cache, f.paths['train']['cache'])
    rehash_cache(f, 'train')
    with pytest.raises(ValueError, match=match):
        seq.prepare(f.manifest, f.index, f.baseline)


def test_wrong_features_even_matching_logits_rejected(synthetic):
    f = synthetic
    cache = torch.load(f.paths['train']['cache'], weights_only=True)
    cache['features'] += 1
    cache['base_logits'] = f.backbone.policy_head(cache['features']).detach()
    torch.save(cache, f.paths['train']['cache'])
    rehash_cache(f, 'train')
    with pytest.raises(ValueError, match='actual pixels'):
        seq.prepare(f.manifest, f.index, f.baseline)


def test_test_tensors_not_loaded(synthetic, monkeypatch):
    f = synthetic
    original = seq._tensor_asset
    def watched(root, ref):
        assert 'synthetic-test' not in ref['path']
        return original(root, ref)
    monkeypatch.setattr(seq, '_tensor_asset', watched)
    _, data, info = seq.prepare(f.manifest, f.index, f.baseline)
    assert set(data) == {'train', 'validation'}
    assert len(info['sources']) == 2
    assert len(seq.windows(data['train'])) == 1
    assert seq.windows(data['train'])[0][1] == 16


@pytest.mark.parametrize('arm', ['N0', 'N1', 'B1'])
def test_synthetic_cpu_budget_logs_sha_not_deployable(synthetic, arm):
    f = synthetic
    out = f.root/('unit-only-'+arm)
    frozen = {k: v.clone() for k, v in f.backbone.state_dict().items()}
    report = seq.train(f.manifest, f.index, f.baseline, out,
                       seq.SequenceConfig(arm=arm, epochs=1, max_updates=1))
    assert report['updates'] == 1 and not report['deployable'] and not report['test_used']
    artifact = torch.load(out/'research-heads.pt', weights_only=True)
    assert artifact['schema'] == 'v373-sequence-research-only/1'
    assert all(torch.equal(v, frozen[k]) for k, v in f.backbone.state_dict().items())
    hashes = json.loads((out/'sha256.json').read_text())
    assert all(seq.file_sha(out/k) == v for k, v in hashes.items())
    with pytest.raises(FileExistsError):
        seq.train(f.manifest, f.index, f.baseline, out)


def test_unknown_batch_never_steps_or_exports_weights(synthetic, monkeypatch):
    f = synthetic
    labels = torch.load(f.paths['train']['labels'], weights_only=True)
    labels['action_valid'].zero_()
    torch.save(labels, f.paths['train']['labels'])
    e = json.loads(f.paths['train']['event'].read_text())
    e['labels']['sha256'] = seq.file_sha(f.paths['train']['labels'])
    dump_json(f.paths['train']['event'], e)
    m = json.loads(f.manifest.read_text())
    m['events'][0]['sha256'] = seq.file_sha(f.paths['train']['event'])
    dump_json(f.manifest, m)
    idx = json.loads(f.index.read_text())
    idx['manifest_sha256'] = seq.file_sha(f.manifest)
    write_index(f.index, idx)
    monkeypatch.setattr(torch.optim.Adam, 'step', lambda *a, **k: pytest.fail('unknown batch stepped'))
    out = f.root/'unknown-unit'
    report = seq.train(f.manifest, f.index, f.baseline, out, seq.SequenceConfig(epochs=1))
    assert report['updates'] == 0 and not (out/'research-heads.pt').exists()


def test_independent_masks_action_positive_set(synthetic):
    f = synthetic
    backbone, data, _ = seq.prepare(f.manifest, f.index, f.baseline)
    model = seq.TacticalPolicy(backbone, seq.TacticalConfig(feature_dim=4))
    e = data['train'][0]
    e['labels'] = dict(action_positive=torch.ones(48, 18, dtype=torch.bool),
                       action_valid=torch.ones(48, dtype=torch.bool),
                       target=torch.full((48, 2), float('nan')), target_valid=torch.zeros(48, dtype=torch.bool),
                       option_positive=torch.ones(48, 3, dtype=torch.bool), option_valid=torch.arange(48) == 20)
    loss = seq.window_loss(model, e, 16, 'cpu')
    assert loss['counts']['action'] == 32 and loss['counts']['option'] == 1
    assert loss['counts']['target'] == 0 and torch.isfinite(loss['loss'])
    assert abs(float(loss['action_loss'].detach())) < 1e-5


def test_nested_event_relative_assets(synthetic):
    f = synthetic
    sub = f.root/'events'
    sub.mkdir()
    event = json.loads(f.paths['train']['event'].read_text())
    for ref in (event['inputs']['pixels'], event['inputs']['history'], event['labels']):
        ref['path'] = '../'+ref['path']
    nested = dump_json(sub/'train.json', event)
    nested['path'] = 'events/train.json'
    m = json.loads(f.manifest.read_text())
    m['events'][0] = nested
    dump_json(f.manifest, m)
    idx = json.loads(f.index.read_text())
    idx['manifest_sha256'] = seq.file_sha(f.manifest)
    write_index(f.index, idx)
    _, data, _ = seq.prepare(f.manifest, f.index, f.baseline)
    assert len(data['train']) == 1


def test_split_declaration_drift_rejected(synthetic):
    f = synthetic
    m = json.loads(f.manifest.read_text())
    m['split_groups']['train'].append('nonexistent-group')
    dump_json(f.manifest, m)
    idx = json.loads(f.index.read_text())
    idx['manifest_sha256'] = seq.file_sha(f.manifest)
    write_index(f.index, idx)
    with pytest.raises(ValueError, match='declared/actual'):
        seq.prepare(f.manifest, f.index, f.baseline)


def test_fabricated_actual_history_rejected(synthetic):
    f = synthetic
    history = torch.load(f.paths['train']['history'], weights_only=True)
    history['history_valid'][20] = True
    torch.save(history, f.paths['train']['history'])
    event = json.loads(f.paths['train']['event'].read_text())
    event['inputs']['history']['sha256'] = seq.file_sha(f.paths['train']['history'])
    dump_json(f.paths['train']['event'], event)
    m = json.loads(f.manifest.read_text())
    m['events'][0]['sha256'] = seq.file_sha(f.paths['train']['event'])
    dump_json(f.manifest, m)
    idx = json.loads(f.index.read_text())
    idx['manifest_sha256'] = seq.file_sha(f.manifest)
    write_index(f.index, idx)
    cache = torch.load(f.paths['train']['cache'], weights_only=True)
    cache['history_sha256'] = event['inputs']['history']['sha256']
    torch.save(cache, f.paths['train']['cache'])
    rehash_cache(f, 'train')
    with pytest.raises(ValueError, match='acknowledged actual'):
        seq.prepare(f.manifest, f.index, f.baseline)


@pytest.mark.parametrize('kwargs', [dict(epochs=31), dict(sequence=16), dict(burnin=0), dict(max_updates=0), dict(learning_rate=float('nan'))])
def test_config_rejects_changed_budget(kwargs):
    with pytest.raises(ValueError):
        seq.SequenceConfig(**kwargs)


@pytest.mark.parametrize('mutation', ['old_schema', 'missing_protocol', 'missing_report',
                                     'bad_report_hash', 'report_protocol', 'source',
                                     'precision', 'device_name', 'torch_version', 'batch'])
def test_protocol_fail_closed_before_backbone(synthetic, monkeypatch, mutation):
    f = synthetic
    idx = json.loads(f.index.read_text())
    if mutation == 'old_schema':
        idx['schema'] = 'v373-event-feature-cache/1'
    elif mutation == 'missing_protocol':
        idx.pop('compute_protocol')
    elif mutation == 'missing_report':
        idx.pop('report')
    elif mutation == 'bad_report_hash':
        idx['report']['sha256'] = '0'*64
    elif mutation == 'report_protocol':
        report_path = f.root/idx['report']['path']
        report = json.loads(report_path.read_text())
        report['compute_protocol']['batch_size'] = 32
        idx['report'] = dump_json(report_path, report)
    elif mutation == 'source':
        idx['compute_protocol']['source_sha256']['vmodels3.py'] = '0'*64
    else:
        key, value = {'precision': ('precision', 'float16'),
                      'device_name': ('device_name', 'other CPU'),
                      'torch_version': ('torch_version', 'other'),
                      'batch': ('batch_size', 1)}[mutation]
        idx['compute_protocol'][key] = value
    dump_json(f.index, idx)
    monkeypatch.setattr(seq, 'load_backbone', lambda p: pytest.fail('backbone loaded'))
    with pytest.raises(ValueError):
        seq.prepare(f.manifest, f.index, f.baseline)


@pytest.mark.parametrize('fail', [False, True])
def test_prepare_fp32_flags_restored(synthetic, monkeypatch, fail):
    f = synthetic
    forward = f.backbone.forward_core
    batches = []
    def watched(regions):
        assert not torch.is_autocast_enabled('cpu')
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        batches.append(len(regions['synthetic']))
        if fail:
            raise RuntimeError('intentional forward failure')
        return forward(regions)
    monkeypatch.setattr(f.backbone, 'forward_core', watched)
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        with torch.autocast('cpu', dtype=torch.bfloat16):
            if fail:
                with pytest.raises(RuntimeError, match='intentional'):
                    seq.prepare(f.manifest, f.index, f.baseline)
            else:
                backbone, data, _ = seq.prepare(f.manifest, f.index, f.baseline)
                assert next(backbone.parameters()).device.type == 'cpu'
                assert batches == [16]*6
                assert not data['train'][0]['features'].is_inference()
            assert torch.is_autocast_enabled('cpu')
        assert torch.backends.cuda.matmul.allow_tf32
        assert torch.backends.cudnn.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old
