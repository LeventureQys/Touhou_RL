import torch
from touhou_rl import tactical_synthetic_dagger as m
from touhou_rl.tactical_teacher_validation import make_world
from touhou_rl.tactical_synthetic_closed_loop import run_event

class Stay:
    def __call__(self,*args):
        logits=torch.zeros(18);logits[9]=1
        return logits,None


def test_export_r10_assets_actual_mixed_pixels_and_refs(tmp_path):
    import numpy as np
    from touhou_rl.tactical_scenarios import initial_state
    s=make_world(m.train_specs()[0]['spec'])
    trace=run_event(s,'N1','horizontal_masked',Stay(),None)
    rescue=m.rescue_event(s,trace)
    data=tmp_path/'records';data.mkdir()
    e=m.export_event(data,s,rescue,trace,{'collection_checkpoint_sha':'frozen-test'})
    assert e['split']=='train' and e['augmentation'] and e['parent_group']==e['group']==s.template_group
    assert e['event_id']!=s.event_id and e['parent_event_id']==s.event_id
    for key in ('pixels','history','labels','raw','rollout'):
        assert m.sha(data/e[key]['path'])==e[key]['sha256']
    raw=np.load(data/e['raw']['path'])
    assert raw['rgb'].shape==(161,480,640,3)
    states=[initial_state(s)]
    for a in rescue['actions']:
        for _ in range(2):states.append(m.logical_step(s,states[-1],a))
    assert np.array_equal(raw['rgb'][rescue['rescue_start_frame']],m.render_state(s,states[rescue['rescue_start_frame']]))
    history=torch.load(data/e['history']['path'],weights_only=True)
    labels=torch.load(data/e['labels']['path'],weights_only=True)
    pixels=torch.load(data/e['pixels']['path'],weights_only=True)
    assert pixels['playfield'].shape[0:2]==(80,3)
    assert history['previous_candidate'][1:rescue['prefix_rows']+1].tolist()==[9]*rescue['prefix_rows']
    assert not labels['action_valid'][:rescue['prefix_rows']].any()
    assert int(labels['action_valid'].sum())==e['supervised_rows']
    for k in pixels:assert torch.equal(pixels[k][0,0],pixels[k][0,1])
