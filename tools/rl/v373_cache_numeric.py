"""Real-backbone numeric/throughput diagnostic on captured pixels, NO labels/updates."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from touhou_rl import vision
from touhou_rl.vmodels import load_vision_checkpoint
from touhou_rl.tactical_feature_cache import _fp32_mode
from touhou_rl.tactical_data import file_sha


def run(source,baseline,out):
    out=Path(out)
    if out.exists():raise FileExistsError('new numeric report required')
    torch.set_num_threads(4)
    frames=np.load(source,allow_pickle=False)['rgb']
    model,_=load_vision_checkpoint(baseline,map_location='cpu')
    indices=np.linspace(2,len(frames)-1,32,dtype=int)
    arrays={k:[] for k in model.config.region_names}
    for i in indices:
        histories=[vision.extract_regions(frames[j]) for j in (i-2,i-1,i)]
        for name,size in model.config.region_sizes.items():
            arrays[name].append(np.stack([vision.resize_area(h[name],*size) for h in histories]))
    tensors=vision.regions_to_batch({k:np.stack(v) for k,v in arrays.items()})
    def forward(device,batch):
        nonlocal model
        model=model.to(device).eval()
        ff=[];ll=[]
        if device=='cuda':torch.cuda.synchronize()
        begin=time.perf_counter()
        with _fp32_mode(torch.device(device)),torch.inference_mode():
            for start in range(0,32,batch):
                result=model.forward_core({k:v[start:start+batch].to(device) for k,v in tensors.items()})
                ff.append(result['features'].cpu());ll.append(result['policy_logits'].cpu())
        if device=='cuda':torch.cuda.synchronize()
        return torch.cat(ff),torch.cat(ll),time.perf_counter()-begin
    ref_f,ref_l,ref_s=forward('cpu',1)
    rows=[]
    for device,batch in [('cpu',16),('cuda',1),('cuda',16),('cuda',32)]:
        f,l,seconds=forward(device,batch)
        repeat_f,repeat_l,repeat_seconds=forward(device,batch)
        rows.append({'device':device,'batch':batch,'seconds':seconds,'rows_per_second':32/seconds,
            'repeat_seconds':repeat_seconds,'repeat_feature_max_abs':float((repeat_f-f).abs().max()),
            'repeat_logit_max_abs':float((repeat_l-l).abs().max()),
            'repeat_features_allclose':torch.allclose(repeat_f,f,atol=1e-6,rtol=1e-5),
            'repeat_logits_allclose':torch.allclose(repeat_l,l,atol=1e-6,rtol=1e-5),
            'feature_max_abs':float((f-ref_f).abs().max()),'logit_max_abs':float((l-ref_l).abs().max()),
            'features_allclose_seq_tolerance':torch.allclose(f,ref_f,atol=1e-6,rtol=1e-5),
            'logits_allclose_seq_tolerance':torch.allclose(l,ref_l,atol=1e-6,rtol=1e-5)})
    report={'schema':'v373-real-backbone-cache-numeric/1','source_sha256':file_sha(source),
        'baseline_sha256':file_sha(baseline),'rows':32,'indices':indices.tolist(),
        'history':'causal adjacent captures for compute diagnostic only; not logical-frame history certificate',
        'precision':'FP32 no autocast no TF32','cpu_single_seconds':ref_s,'comparisons':rows,
        'training_updates':0,'formal_cache_created':False,'pixel_alignment_verified':False,
        'warning':'Cold short-run wall times; not complete training/capture throughput. No gate bypass or event cache.'}
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','baseline','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.source,a.baseline,a.out)


if __name__=='__main__':main()
