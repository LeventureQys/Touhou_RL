import torch
from tools.rl.v373_alias_audit import tensor_digest,conflict_summary


def test_digest_uses_values_not_storage_or_future():
    a={'x':torch.tensor([[1.,2.],[3.,4.]])}
    b={'x':a['x'].clone()}
    assert tensor_digest(a,0)==tensor_digest(b,0)
    b['x'][1]=100
    assert tensor_digest(a,0)==tensor_digest(b,0)
    b['x'][0,0]=5
    assert tensor_digest(a,0)!=tensor_digest(b,0)


def test_conflict_bound_uses_majority_not_forced_zero():
    result=conflict_summary({'same':[{'action':0},{'action':0},{'action':3}],
                             'other':[{'action':7}]})
    assert result['conflicting_groups']==1
    assert result['rows_in_conflicting_groups']==3
    assert result['deterministic_exact_input_upper_bound']==.75
