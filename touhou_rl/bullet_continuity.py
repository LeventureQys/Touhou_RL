"""Conservative *observed* pool continuity, never a true engine generation ID."""
import numpy as np

class ObservedBulletTracker:
    def __init__(self):
        self.last_stamp=None; self.previous={}; self.generations={}
    def update(self, raw, stage, frame):
        slots=np.asarray(raw['slot_id'],dtype=np.int64)
        positions=np.asarray(raw['position'],dtype=float).reshape(-1,2)
        velocities=np.asarray(raw['velocity'],dtype=float).reshape(-1,2)
        consecutive=self.last_stamp is not None and self.last_stamp==(stage,frame-1)
        current={}; generation=[]; valid=[]; residual=[]
        for i,slot0 in enumerate(slots):
            slot=int(slot0); old=self.previous.get(slot)
            error=float(np.linalg.norm(positions[i]-(old[0]+old[1]))) if consecutive and old is not None else None
            continuous=error is not None and error <= .5
            if not continuous: self.generations[slot]=self.generations.get(slot,-1)+1
            generation.append(self.generations[slot]);valid.append(bool(continuous));residual.append(error)
            current[slot]=(positions[i].copy(),velocities[i].copy())
        self.last_stamp=(stage,frame);self.previous=current
        return dict(observed_generation=generation,observed_continuity_valid=valid,
                    linear_residual_px=residual,engine_generation_known=False,
                    semantics='observed consecutive pool continuity, slot reuse between samples remains unknowable')
