import json
import pytest
from tools.rl.v373_synthetic_learning import run


def test_real_manifest_cannot_use_synthetic_learning_entry(tmp_path):
    records=tmp_path/'records.json'
    records.write_text(json.dumps({'schema':'v373-paired-events/1','teacher_gate':'passed'}),encoding='utf-8')
    with pytest.raises(ValueError,match='explicitly synthetic'):
        run(records,tmp_path/'nonexistent-baseline.pt',tmp_path/'out',steps=1,device='cpu')
    assert not (tmp_path/'out').exists()


def test_synthetic_diagnostic_rejects_real_flag(tmp_path):
    records=tmp_path/'records.json'
    records.write_text(json.dumps({'schema':'v373-synthetic-rendered-diagnostic/1','real_pixel_training':True}),encoding='utf-8')
    with pytest.raises(ValueError):run(records,tmp_path/'no.pt',tmp_path/'out',1,'cpu')
