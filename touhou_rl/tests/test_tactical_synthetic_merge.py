import json
import pytest
from tools.rl.v373_merge_synthetic import merge


def test_validation_cannot_become_augmentation(tmp_path):
    original=tmp_path/'original.json';augmentation=tmp_path/'aug.json'
    schema='v373-synthetic-rendered-diagnostic/1'
    original.write_text(json.dumps({'schema':schema,'events':[
        {'group':'train','split':'train'},{'group':'val','split':'validation'}]}),encoding='utf-8')
    augmentation.write_text(json.dumps({'schema':schema,'events':[
        {'group':'val','parent_group':'val','split':'train','augmentation':True}]}),encoding='utf-8')
    with pytest.raises(ValueError,match='original train'):
        merge(original,augmentation,tmp_path/'out')
    assert not (tmp_path/'out').exists()
