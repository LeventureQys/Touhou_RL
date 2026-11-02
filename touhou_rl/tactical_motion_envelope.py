"""Frozen narrow hardware motion envelope; exploratory, not a safety certificate."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MotionEnvelope:
    x_min: float = 8.
    x_max: float = 376.
    y_min: float = 16.
    y_max: float = 432.
    normal_speed: float = 4.
    focus_speed: float = 2.

    def advance(self,xy,direction,frames,*,focus=False):
        """Idealized committed input only; caller separately accounts input lag."""
        if type(frames) is not int or frames<0:
            raise ValueError('nonnegative integral logical frames required')
        if len(xy)!=2 or len(direction)!=2 or not all(math.isfinite(v) for v in (*xy,*direction)):
            raise ValueError('finite two dimensional values required')
        if any(v not in (-1,0,1) for v in direction):
            raise ValueError('discrete direction required')
        dx,dy=direction
        scale=(self.focus_speed if focus else self.normal_speed)*frames/(math.hypot(dx,dy) or 1)
        return (min(self.x_max,max(self.x_min,xy[0]+dx*scale)),
                min(self.y_max,max(self.y_min,xy[1]+dy*scale)))


def old_action_carryover(envelope,xy,previous,new,hold_frames,lag_interval,*,focus=False):
    """Set of endpoint hypotheses for integer effective lag, not a safe reachable set.

    Lag can consume the entire window. No negative exposure, no linear speed
    averaging that would silently replace old direction with a slower new one.
    """
    lo,hi=lag_interval
    if any(type(v) is not int for v in (lo,hi,hold_frames)) or not 0<=lo<=hi or hold_frames<0:
        raise ValueError('ordered nonnegative integer lag interval required')
    points=[]
    for lag in range(lo,hi+1):
        consumed=min(hold_frames,lag)
        first=envelope.advance(xy,previous,consumed,focus=focus)
        points.append(envelope.advance(first,new,hold_frames-consumed,focus=focus))
    return sorted(set(points))
