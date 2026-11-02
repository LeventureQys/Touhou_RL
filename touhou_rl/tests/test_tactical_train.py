"""Formal-data prerequisites must reject engineering and leaky manifests."""
import json
import pytest
from touhou_rl.tactical_train import validate_training_gate


def test_gate_rejects_engineering_features(tmp_path):
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps({'schema':'v373-engineering-optimization/1'}),encoding='utf-8')
    with pytest.raises(ValueError,match='training gate rejected'):
        validate_training_gate(path)


def test_gate_rejects_leaky_groups(tmp_path):
    obj={'schema':'v373-paired-events/1','teacher_gate':'passed',
         'pixel_alignment_gate':'passed','data_audit_gate':'passed','input_kind':'real_pixels',
         'split_groups':{'train':['same'],'validation':['same'],'test':['test']}}
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps(obj),encoding='utf-8')
    with pytest.raises(ValueError,match='split leakage'):
        validate_training_gate(path)


def test_gate_requires_all_prerequisites(tmp_path):
    obj={'schema':'v373-paired-events/1','teacher_gate':'passed',
         'pixel_alignment_gate':'pending','data_audit_gate':'passed','input_kind':'real_pixels',
         'split_groups':{'train':['train'],'validation':['validation'],'test':['test']}}
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps(obj),encoding='utf-8')
    with pytest.raises(ValueError,match='pixel_alignment_gate'):
        validate_training_gate(path)
