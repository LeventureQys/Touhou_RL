"""Independent Phase3 saved-model recompute and pixel runtime/reset checks."""
from pathlib import Path
import json,time
import numpy as np
import torch
from .temporal_policy import load_temporal_checkpoint,TemporalRunner
from .phase3_experiment import infer,metric
from .phase3_sequences import build_sequence_index,previous_history
from .build_chunk_safety_cache import audit_source,validate,sha
from .vmodels import load_vision_checkpoint
from .vision_data import VisionDataset
from .vision_train import _gather
from . import vision


def verify(root,source,cache):
    root,source,cache=map(Path,(root,source,cache));p=json.loads((root/'preregistration.json').read_text());r=json.loads((root/'experiment/phase3-report.json').read_text());device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    validate(cache,verify_source=True);report,rows,aligned,files=audit_source(source)
    with np.load(cache/'chunk-arrays.npz') as z:ca={k:z[k].copy() for k in z.files}
    with np.load(root/'experiment/student-features.npz') as z:features=z['features'].copy();bullets=z['bullets'].copy()
    episode_map={e:k for k,eps in p['episodes'].items() for e in eps};segments=build_sequence_index(rows,episode_map,p['temporal_data']['max_frame_gap'])
    clean=[]
    for segment in segments:
        current=[]
        for index in segment:
            if ca['model_valid'][index].all():current.append(index)
            elif current:clean.append(current);current=[]
        if current:clean.append(current)
    segments=clean;history=previous_history(rows,segments)
    pc=np.full(len(features),18,np.int64);pd=np.full(len(features),3,np.int64)
    for i,(c,d) in history.items():pc[i]=c;pd[i]=d
    ids=sorted(i for seg in segments for i in seg if ca['episode'][i] in p['episodes']['test'] and ca['model_valid'][i].all())
    base,_=load_vision_checkpoint(Path(p['baseline_checkpoint']));base.to(device).eval()
    with torch.inference_mode():base_choice=base.policy_head(torch.as_tensor(features,device=device)).argmax(1).cpu().numpy()
    results={}
    for label,name,reset in [('temporal','temporal-policy.pt',False),('reset_control','reset-control.pt',True)]:
        model=load_temporal_checkpoint(root/'experiment'/name,map_location='cpu').to(device).eval()
        scores,duration=infer(model,features,segments,pc,pd,device,reset)
        actual=metric(scores,duration,ca['clearance'][:,:,1],bullets,ids,ca['episode'],ca['clearance'],base_choice);saved=r['metrics'][label]['test']
        results[label]=dict(frames=len(ids),candidate_equal=actual['per_frame']['candidate']==saved['per_frame']['candidate'],
                           duration_equal=actual['per_frame']['duration_class']==saved['per_frame']['duration_class'],
                           max_regret_diff=float(np.max(np.abs(np.asarray(actual['per_frame']['regret'])-saved['per_frame']['regret']))),
                           checkpoint_sha256=sha(root/'experiment'/name))
        if label=='temporal':chosen_model=model
    ds=VisionDataset(source,history_contract='causal-segment-v1');frames,_=_gather(ds,[ids[0]],3,chosen_model.vision_model.config.region_sizes);batch=vision.regions_to_batch(frames,device);ds.close()
    runner=TemporalRunner(chosen_model);times=[]
    with torch.inference_mode():
        for _ in range(10):c,d,_=runner.step(batch,reset=True)
        for _ in range(100):
            if device.type=='cuda':torch.cuda.synchronize()
            start=time.perf_counter();c,d,_=runner.step(batch);runner.commit(c,d)
            if device.type=='cuda':torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000)
        runner.reset();c0,d0,o0=runner.step(batch);runner.commit(c0,d0);runner.step(batch);runner.reset();c1,d1,o1=runner.step(batch)
        reset_ok=torch.equal(c0,c1) and torch.equal(d0,d1) and torch.allclose(o0['hidden'],o1['hidden'],atol=1e-6)
    latency=dict(schema='v200-phase3-latency-reset/1',report_sha256=sha(root/'experiment/phase3-report.json'),checkpoint_sha256=sha(root/'experiment/temporal-policy.pt'),device=str(device),iterations=100,pixel_end_to_end_p50_ms=float(np.percentile(times,50)),pixel_end_to_end_p95_ms=float(np.percentile(times,95)),reset_isolation_passed=bool(reset_ok),
                 input='preloaded uint8 ROI batch, includes frozen visual encoder and recurrent/heads; excludes OS grab/env step',scope='model inference budget only')
    (root/'latency-and-reset.json').write_text(json.dumps(latency,indent=2))
    result=dict(schema='v200-phase3-independent-recompute/1',protocol_sha256=sha(root/'preregistration.json'),report_sha256=sha(root/'experiment/phase3-report.json'),results=results,
                passed=all(x['candidate_equal'] and x['duration_equal'] and x['max_regret_diff']<1e-5 for x in results.values()),training_executed=False)
    (root/'independent-recompute.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));print(json.dumps(latency,indent=2));return result
if __name__=='__main__':
    import argparse;pa=argparse.ArgumentParser();pa.add_argument('--root',required=True);pa.add_argument('--source',required=True);pa.add_argument('--cache',required=True);a=pa.parse_args();verify(a.root,a.source,a.cache)
