"""Strict raw-pool structure verifier; additive audit without changing teacher."""
import numpy as np

def validate_pool(pool):
    if not isinstance(pool,dict):raise ValueError('pool missing')
    required=('position','velocity','half','slot_id')
    if any(k not in pool for k in required):raise ValueError('pool fields missing')
    n=len(pool['half']);pos=np.asarray(pool['position'],float).reshape(-1,2);vel=np.asarray(pool['velocity'],float).reshape(-1,2)
    half=np.asarray(pool['half'],float);slots=pool['slot_id']
    if pos.shape!=(n,2) or vel.shape!=(n,2) or half.shape!=(n,) or len(slots)!=n:raise ValueError('pool lengths mismatch')
    if not all(np.isfinite(x).all() for x in (pos,vel,half)) or (half<0).any():raise ValueError('invalid pool geometry')
    if any(type(s) is not int or s<0 for s in slots) or len(set(slots))!=n:raise ValueError('invalid pool slot identity')
    return True
