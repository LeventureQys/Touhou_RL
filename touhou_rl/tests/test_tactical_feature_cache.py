"""Synthetic CPU-only builder checks; gate monkeypatches exist ONLY in tests."""
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from touhou_rl import tactical_feature_cache as fc
from touhou_rl import tactical_sequence_train as seq
from touhou_rl.tactical_policy import bos_history


class TinySyntheticBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(trunk_hidden=4)
        self.policy_head = nn.Linear(4, 18)
        self.batches = []

    def forward_core(self, regions):
        assert not self.training and not torch.is_grad_enabled()
        assert all(not p.requires_grad for p in self.parameters())
        assert not torch.is_autocast_enabled('cpu')
        features = regions['synthetic'].flatten(1).float()
        self.batches.append(len(features))
        return dict(features=features, policy_logits=self.policy_head(features))


def dump(path, obj, tensor=False):
    if tensor:
        torch.save(obj, path)
    else:
        path.write_text(json.dumps(obj), encoding='utf-8')
    return dict(path=path.name, sha256=fc.file_sha(path))


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    torch.manual_seed(32)
    backbone = TinySyntheticBackbone()
    baseline = tmp_path/'SYNTHETIC-baseline.bin'
    baseline.write_bytes(b'UNIT TEST ONLY; NOT A REAL BASELINE')
    refs, paths = [], {}
    for split in ('train', 'validation', 'test'):
        sub = tmp_path/split
        sub.mkdir()
        f = torch.randn(48, 4)
        pixels = dump(sub/'pixels.pt', {'synthetic': f.reshape(48, 1, 1, 1, 4)}, True)
        history = dump(sub/'history.pt', {k: v[0] for k, v in bos_history(f[None]).items()}, True)
        labels = dump(sub/'labels.pt', {'action_valid': torch.ones(48, dtype=torch.bool)}, True)
        event = dict(event_id='synthetic-'+split, template_group=split, split=split,
                     inputs=dict(pixels=pixels, history=history), labels=labels,
                     steps=[dict(frame=i, capture_interval=[i, i], alignment_status='verified', pool_complete=True)
                            for i in range(48)])
        ref = dump(sub/'event.json', event)
        ref['path'] = split+'/'+ref['path']
        refs.append(ref)
        paths[split] = sub/'event.json'
    manifest = tmp_path/'manifest.json'
    dump(manifest, dict(schema='v373-paired-events/1', events=refs,
                        split_groups={s: [s] for s in paths}))
    # No production bypass: synthetic manifests omit the production gate fields.
    gate = lambda p: json.loads(p.read_text(encoding='utf-8'))
    monkeypatch.setattr(fc, 'validate_training_gate', gate)
    monkeypatch.setattr(seq, 'validate_training_gate', gate)
    monkeypatch.setattr(fc, 'load_backbone', lambda p: backbone)
    monkeypatch.setattr(seq, 'load_backbone', lambda p: backbone)
    return SimpleNamespace(root=tmp_path, manifest=manifest, baseline=baseline, backbone=backbone,
                           paths=paths, out=tmp_path/'cache')


@pytest.mark.parametrize('batch_size,expected', [(16, [16]*6), (32, [32, 16]*2)])
def test_builder_sequence_prepare_no_test_load(synthetic, monkeypatch, batch_size, expected):
    f = synthetic
    frozen = {k: v.clone() for k, v in f.backbone.state_dict().items()}
    original = torch.load
    loaded = []
    def watched(path, *args, **kwargs):
        assert 'test' not in str(path.relative_to(f.root)).split('/')[0].split('\\')[0]
        assert kwargs.get('weights_only') is True
        loaded.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(torch, 'load', watched)
    report = fc.build(f.manifest, f.baseline, f.out, batch_size=batch_size)
    assert f.backbone.batches == expected
    assert len(loaded) == 2  # Only train/validation pixels, no history/labels.
    assert report['precision'] == 'float32' and not report['autocast'] and not report['tf32']
    assert not report['real_image_source_independently_certified']
    assert report['rows'] == 96 and report['rows_per_second'] > 0
    assert not report['test_tensor_loaded'] and not report['test_cache_created']
    assert all(torch.equal(v, frozen[k]) for k, v in f.backbone.state_dict().items())
    index = json.loads((f.out/'index.json').read_text())
    assert len(index['events']) == 2
    for ref in index['events']:
        path = f.out/ref['path']
        assert fc.file_sha(path) == ref['sha256']
        cache = original(path, weights_only=True)
        assert set(cache) == {'schema', 'event_id', 'baseline_sha256', 'pixel_sha256',
                              'history_sha256', 'feature_source', 'features', 'base_logits'}
        assert cache['features'].dtype == cache['base_logits'].dtype == torch.float32
    _, data, provenance = seq.prepare(f.manifest, f.out/'index.json', f.baseline)
    assert len(data['train']) == len(data['validation']) == 1
    assert len(provenance['sources']) == 2
    assert f.backbone.batches == expected + expected
    assert index['compute_protocol'] == report['compute_protocol']
    assert provenance['compute_protocol'] == index['compute_protocol']
    assert all(e['features'].device.type == e['base_logits'].device.type == 'cpu'
               for split in data.values() for e in split)
    with pytest.raises(FileExistsError):
        fc.build(f.manifest, f.baseline, f.out)


def test_real_unqualified_gate_before_baseline_or_output(tmp_path, monkeypatch):
    manifest = tmp_path/'real-calibration.json'
    dump(manifest, dict(schema='v373-paired-events/1', input_kind='real_pixels', teacher_gate='failed'))
    monkeypatch.setattr(fc, 'load_backbone', lambda p: pytest.fail('baseline touched'))
    with pytest.raises(ValueError, match='training gate rejected'):
        fc.build(manifest, tmp_path/'absent-baseline', tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_audit_before_baseline_or_output(synthetic, monkeypatch):
    f = synthetic
    monkeypatch.setattr(fc, 'audit_manifest', lambda p: dict(passed=False))
    monkeypatch.setattr(fc, 'load_backbone', lambda p: pytest.fail('baseline touched'))
    with pytest.raises(ValueError, match='audit rejected'):
        fc.build(f.manifest, f.root/'missing', f.out)
    assert not f.out.exists()


def test_test_pixel_hash_corruption_rejected_without_tensor_load(synthetic, monkeypatch):
    f = synthetic
    (f.root/'test'/'pixels.pt').write_bytes(b'corrupt test bytes')
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('tensor loaded before audit'))
    with pytest.raises(ValueError, match='audit rejected'):
        fc.build(f.manifest, f.baseline, f.out)
    assert not f.out.exists()


@pytest.mark.parametrize('batch_size', [0, 1, 64, True])
def test_bad_batch_size(synthetic, batch_size):
    with pytest.raises(ValueError, match='16 or 32'):
        fc.build(synthetic.manifest, synthetic.baseline, synthetic.out, batch_size=batch_size)


def test_split_drift_rejected_before_backbone(synthetic, monkeypatch):
    f = synthetic
    obj = json.loads(f.manifest.read_text())
    obj['split_groups']['train'].append('ghost')
    dump(f.manifest, obj)
    monkeypatch.setattr(fc, 'load_backbone', lambda p: pytest.fail('baseline touched'))
    with pytest.raises(ValueError, match='declared/actual'):
        fc.build(f.manifest, f.baseline, f.out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_synthetic_cuda_batch32_prepare_readable(synthetic):
    f = synthetic
    report = fc.build(f.manifest, f.baseline, f.out, device='cuda', batch_size=32)
    assert report['device'] == 'cuda:0' and report['peak_allocated_bytes'] > 0
    assert f.backbone.batches == [32, 16, 32, 16]
    _, data, _ = seq.prepare(f.manifest, f.out/'index.json', f.baseline)
    assert f.backbone.batches == [32, 16, 32, 16]*2
    assert len(data['train']) == len(data['validation']) == 1
    assert data['train'][0]['features'].device.type == 'cpu'


def test_cuda_protocol_never_falls_back_to_cpu(synthetic, monkeypatch):
    f = synthetic
    fc.build(f.manifest, f.baseline, f.out)
    idx = json.loads((f.out/'index.json').read_text())
    idx['compute_protocol']['device'] = 'cuda:0'
    dump(f.out/'index.json', idx)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(seq, 'load_backbone', lambda p: pytest.fail('CPU fallback'))
    with pytest.raises(ValueError, match='CUDA'):
        seq.prepare(f.manifest, f.out/'index.json', f.baseline)


def test_ambient_autocast_disabled_and_restored(synthetic):
    f = synthetic
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with torch.autocast('cpu', dtype=torch.bfloat16):
            assert torch.is_autocast_enabled('cpu')
            fc.build(f.manifest, f.baseline, f.out)
            assert torch.is_autocast_enabled('cpu')
        assert torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def test_escape_rejected_before_output(synthetic):
    f = synthetic
    obj = json.loads(f.manifest.read_text())
    obj['events'][0]['path'] = '../outside.json'
    dump(f.manifest, obj)
    with pytest.raises(ValueError, match='audit rejected'):
        fc.build(f.manifest, f.baseline, f.out)
    assert not f.out.exists()


def test_baseline_mutation_prevents_index_publication(synthetic, monkeypatch):
    f = synthetic
    forward = f.backbone.forward_core
    def mutate(regions):
        f.baseline.write_bytes(b'CHANGED WHILE BUILDING')
        return forward(regions)
    monkeypatch.setattr(f.backbone, 'forward_core', mutate)
    with pytest.raises(ValueError, match='changed during build'):
        fc.build(f.manifest, f.baseline, f.out)
    assert not (f.out/'index.json').exists()


@pytest.mark.parametrize('dtype', [torch.float64, torch.uint8])
def test_pixel_dtype_rejected_no_index(synthetic, dtype):
    f = synthetic
    path = f.root/'train'/'pixels.pt'
    ref = dump(path, {'synthetic': torch.zeros(48, 1, 1, 1, 4, dtype=dtype)}, True)
    event = json.loads(f.paths['train'].read_text())
    event['inputs']['pixels'] = ref
    eref = dump(f.paths['train'], event)
    obj = json.loads(f.manifest.read_text())
    obj['events'][0]['sha256'] = eref['sha256']
    dump(f.manifest, obj)
    with pytest.raises(ValueError, match='pixel region mapping'):
        fc.build(f.manifest, f.baseline, f.out)
    assert not (f.out/'index.json').exists()
