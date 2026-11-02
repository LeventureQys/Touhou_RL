from types import SimpleNamespace
from touhou_rl.input_evidence import observe_hold, CONTROL_MASK
from touhou_rl import spec

class Reader:
    def __init__(self, skip=False): self.i=0; self.skip=skip
    def read_state(self, **kw):
        f=self.i//2; self.i+=1
        return SimpleNamespace(stage=1,gameplay_frame=f*(2 if self.skip else 1),gui_input_current=spec.BTN_RIGHT,
                               player_x=100.+f*4,player_y=100.,deaths=0,supervisor_state=spec.STATE_GAME_MANAGER)
class Keyboard:
    def set_keys(self, keys): self.keys=keys

def test_observed_readback_and_hold():
    k=Keyboard(); state,e=observe_hold(Reader(),k,{'right'},spec.BTN_RIGHT,2)
    assert k.keys=={'right'}
    assert e['timing_valid'] and e['coverage_contiguous']
    assert e['end']['frame']-e['effective_first_observed']['frame']==2
    assert len(e['observations'])==4
    assert e['effective_first_observed']['frame'] == e['first_match_observed']['frame'] + 1

def test_skipped_frames_not_fabricated():
    _,e=observe_hold(Reader(True),Keyboard(),{'right'},spec.BTN_RIGHT,2)
    assert not e['coverage_contiguous'] and not e['timing_valid']
