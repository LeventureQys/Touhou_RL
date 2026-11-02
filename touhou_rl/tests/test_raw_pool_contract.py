import pytest
from touhou_rl.raw_pool_contract import validate_pool

def pool():return dict(slot_id=[0],position=[[1.,2.]],velocity=[[0.,1.]],half=[1.])
def test_pool_strict_parallel_arrays():
    assert validate_pool(pool())
    p=pool();p['velocity']=[]
    with pytest.raises(ValueError):validate_pool(p)
    p=pool();p['half']=[float('nan')]
    with pytest.raises(ValueError):validate_pool(p)
    p=pool();p['slot_id']=[-1]
    with pytest.raises(ValueError):validate_pool(p)
