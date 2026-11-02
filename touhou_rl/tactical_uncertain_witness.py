"""Positive-only route checks over supplied alignment hypotheses.

Enumerated hypotheses are not automatically a complete uncertainty set. A robust
positive is useful only conditional on externally certified hypothesis coverage.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import hypot,isfinite


@dataclass(frozen=True)
class LinearHazard:
    x: float
    y: float
    vx: float
    vy: float
    radius: float

    def __post_init__(self):
        if not all(isfinite(v) for v in (self.x,self.y,self.vx,self.vy,self.radius)) or self.radius<0:
            raise ValueError('finite hazard and nonnegative radius required')


def witness_over_hypotheses(route,hazards,hypotheses,*,player_radius=2.,coverage_certified=False,model_known=True):
    """route is positions at every logical frame including initial; hazards at t=0.

    hypotheses: dicts time_shift/int, player_dx/dy/float, complete_pool/bool.
    Linear motion is checked with swept relative line distance each frame, not
    endpoints alone. This does not model future births/acceleration/lasers.
    Unknown is never turned into an unsafe-action negative label.
    """
    if len(route)<2 or not hazards or not hypotheses or not model_known:
        return {'status':'unknown','reason':'missing_route_hazards_hypotheses_or_model','positive':False}
    if player_radius<0 or not isfinite(player_radius):raise ValueError('invalid player radius')
    if any(len(p)!=2 or not all(isfinite(v) for v in p) for p in route):raise ValueError('finite route required')
    results=[]
    for h in hypotheses:
        if not h.get('complete_pool'):
            return {'status':'unknown','reason':'incomplete_hazard_pool','positive':False}
        shift=h['time_shift'];dx=h.get('player_dx',0.);dy=h.get('player_dy',0.)
        if type(shift) is not int or not all(isfinite(v) for v in (dx,dy)):raise ValueError('invalid alignment hypothesis')
        collision=None
        for i,(a,b) in enumerate(zip(route,route[1:])):
            for j,hazard in enumerate(hazards):
                # Relative player-minus-bullet vector across complete segment.
                rx=a[0]+dx-(hazard.x+hazard.vx*(i+shift))
                ry=a[1]+dy-(hazard.y+hazard.vy*(i+shift))
                vx=(b[0]-a[0])-hazard.vx;vy=(b[1]-a[1])-hazard.vy
                vv=vx*vx+vy*vy
                t=max(0.,min(1.,-(rx*vx+ry*vy)/vv)) if vv else 0.
                clearance=hypot(rx+t*vx,ry+t*vy)-(player_radius+hazard.radius)
                if clearance<=0:
                    collision={'segment':i,'hazard':j,'relative_fraction':t,'clearance':clearance};break
            if collision:break
        results.append({'hypothesis':dict(h),'collision':collision})
    if any(r['collision'] is not None for r in results):
        return {'status':'rejected_witness','reason':'this_route_not_robust_not_universal_infeasibility',
                'positive':False,'results':results}
    if not coverage_certified:
        return {'status':'unknown','reason':'enumerated_alignment_coverage_not_certified','positive':False,'results':results}
    return {'status':'conditional_positive','reason':'full_route_clear_for_certified_linear_hypotheses',
            'positive':True,'results':results,'scope':'known existing linear hazards only; no unseen births'}
