"""Controlled recurrent preference/duration experiment on frozen pixel features."""
from pathlib import Path
import argparse,copy,hashlib,json,time
import numpy as np
import torch
from .temporal_policy import TemporalPolicy,TemporalConfig,save_temporal_checkpoint
from .phase3_sequences import build_sequence_index,previous_history,chunk_targets
from .phase3_features import bank
from .phase3_losses import rank_per_frame,duration_loss,weighted_mean
from .vmodels import load_vision_checkpoint
from .vision_data import VisionDataset
from .build_chunk_safety_cache import audit_source,validate,sha
from .phase2_chunk_experiment import _stratified_a,_bootstrap
from .pipeline.manifest import DatasetManifest,index_digest
from .vision_train import teacher_identity,_fingerprint,teacher_cache_path,_inspect_teacher_cache
from . import vision_metrics


def frozen_digest(model):
    h=hashlib.sha256()
    for name,value in model.vision_model.state_dict().items():h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def infer(model,features,segments,pc,pd,device,reset_each=False):
    n=len(features);score=np.full((n,18),np.nan,np.float32);duration=np.full((n,18,3),np.nan,np.float32)
    model.eval()
    with torch.inference_mode():
        for segment in segments:
            hidden=None
            for start in range(0,len(segment),64):
                ids=segment[start:start+64];reset=torch.zeros((1,len(ids)),dtype=torch.bool,device=device)
                reset[0,0]=start==0
                if reset_each:reset[:]=True
                out=model.forward_features(torch.as_tensor(features[ids][None],device=device),torch.as_tensor(pc[ids][None],device=device),
                  torch.as_tensor(pd[ids][None],device=device),hidden,reset=reset)
                hidden=out['hidden'];score[ids]=out['candidate_logits'][0].cpu().numpy();duration[ids]=out['duration_logits'][0].cpu().numpy()
    return score,duration


def metric(scores,duration,values,bullets,ids,episodes,clearance=None,baseline_candidate=None):
    ids=np.asarray(ids,int);p=scores[ids];target=values[ids];b=bullets[ids]
    if not len(ids) or not np.isfinite(p).all():raise ValueError('empty/nonfinite temporal evaluation')
    result=vision_metrics.policy_metrics(p,target,b);chosen=p.argmax(1);regret=target.max(1)-target[np.arange(len(ids)),chosen]
    result.update(frames=len(ids),index_sha256=index_digest(ids.tolist()),max_candidate_fraction=float(np.bincount(chosen,minlength=18).max()/len(ids)))
    per=dict(candidate=chosen.tolist(),regret=regret.tolist(),nearest_bullet_px=np.nan_to_num(vision_metrics._nearest_bullet_px(b),posinf=1e9).tolist(),episode=episodes[ids].tolist())
    if clearance is not None:
        dl=duration[ids];dc=dl[np.arange(len(ids)),chosen].argmax(1);c=clearance[ids];actual=c[np.arange(len(ids)),chosen,dc]
        baseline=baseline_candidate[ids] if baseline_candidate is not None else chosen
        same_duration_baseline=c[np.arange(len(ids)),baseline,dc]
        result['chunk']=dict(duration_counts=np.bincount(dc,minlength=3).tolist(),duration_fraction=(np.bincount(dc,minlength=3)/len(ids)).tolist(),
          mean_duration_decisions=float(np.mean(np.array([1,3,5])[dc])),chosen_clearance=float(actual.mean()),chosen_collision_rate=float(np.mean(actual<=0)),
          duration_matched_baseline_clearance=float(same_duration_baseline.mean()),duration_matched_baseline_collision_rate=float(np.mean(same_duration_baseline<=0)),
          duration_matched_clearance_gain=float(np.mean(actual-same_duration_baseline)),
          fixed3_clearance=float(c[np.arange(len(ids)),chosen,1].mean()),
          semantics='linear existing-pool proxy, not world safety or survival')
        per.update(duration_class=dc.tolist(),chosen_chunk_clearance=actual.tolist(),duration_matched_clearance_gain=(actual-same_duration_baseline).tolist())
    result['per_frame']=per;return result


def train(model,features,segments,pc,pd,targets,duration_targets,duration_valid,nearest,a_features,a_values,device,exp,reset_each=False):
    if exp.get('risk_gate_enabled',False):
        train_ids=np.unique(np.concatenate(segments));risk_optimizer=torch.optim.Adam(model.risk_head.parameters(),lr=.001)
        train_x=torch.as_tensor(features[train_ids],device=device);risk_y=torch.as_tensor((nearest[train_ids]<30).astype(np.float32),device=device)
        model.train()
        for _ in range(200):
            risk_optimizer.zero_grad(set_to_none=True);pred=model.risk_head(train_x).squeeze(-1)
            risk_loss=torch.nn.functional.binary_cross_entropy_with_logits(pred,risk_y,pos_weight=torch.tensor(10.,device=device))
            risk_loss.backward();risk_optimizer.step()
    before=frozen_digest(model);params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=exp['learning_rate'],weight_decay=1e-4)
    rng=np.random.default_rng(exp['seed']);torch.manual_seed(exp['seed']);step=0;seen=0
    # Class weights are computed only from actual train-target masks, not test.
    ids=np.concatenate(segments);counts=np.bincount(duration_targets[ids][duration_valid[ids]],minlength=3)
    weights=np.sqrt(max(1,counts.sum())/np.maximum(counts,1));weights=np.minimum(weights,10.);weights=weights/weights.mean()
    class_weights=torch.as_tensor(weights,dtype=torch.float32,device=device)
    trainable_segments=[s for s in segments if len(s)>exp.get('burn_in',4)]
    for epoch in range(exp['epochs']):
        model.train();order=rng.permutation(len(trainable_segments))
        for sid in order:
            segment=trainable_segments[sid];hidden=None
            for start in range(0,len(segment),16):
                selected=segment[start:start+16]
                x=torch.as_tensor(features[selected][None],device=device);c=torch.as_tensor(pc[selected][None],device=device);d=torch.as_tensor(pd[selected][None],device=device)
                reset=torch.zeros((1,len(selected)),dtype=torch.bool,device=device);reset[0,0]=start==0
                if reset_each:reset[:]=True
                optimizer.zero_grad(set_to_none=True)
                out=model.forward_features(x,c,d,hidden,reset=reset);hidden=out['hidden'].detach()
                loss_mask=torch.arange(start,start+len(selected),device=device)[None]>=exp.get('burn_in',4)
                valid=torch.as_tensor(duration_valid[selected][None],device=device)
                value=torch.as_tensor(targets[selected][None],device=device)
                w=torch.as_tensor(np.where(nearest[selected]<60,exp['danger_weight'],1.)[None],device=device,dtype=torch.float32)
                rank=weighted_mean(rank_per_frame(out['candidate_logits'],value),loss_mask,w)
                if exp.get('candidate_correction_ce',False):
                    with torch.no_grad():
                        baseline_logits=model.vision_model.policy_head(x)
                        baseline_choice=baseline_logits.argmax(-1)
                        baseline_value=value.gather(-1,baseline_choice[...,None]).squeeze(-1)
                        correction=value.argmax(-1)
                        target_choice=torch.where(value.max(-1).values-baseline_value>.5,correction,baseline_choice)
                    ce=torch.nn.functional.cross_entropy(out['candidate_logits'].reshape(-1,18),target_choice.reshape(-1),reduction='none').reshape(loss_mask.shape)
                    rank=weighted_mean(ce,loss_mask,w)
                dt=torch.as_tensor(np.maximum(duration_targets[selected],0)[None],device=device)
                dur=duration_loss(out['duration_logits'],dt,valid & loss_mask[...,None],class_weights)
                # Preserve A training logits and its old ranking; reset A hidden
                # per independent sample so no random frame sequence is invented.
                count=3*len(selected);a_idx=(np.arange(count)+(seen*3))%len(a_features)
                ax=torch.as_tensor(a_features[a_idx,None],device=device);bos_c=torch.full((count,1),18,device=device,dtype=torch.long);bos_d=torch.full((count,1),3,device=device,dtype=torch.long)
                ao=model.forward_features(ax,bos_c,bos_d,reset=torch.ones((count,1),device=device,dtype=torch.bool))
                av=torch.as_tensor(a_values[a_idx,None],device=device)
                a_rank=rank_per_frame(ao['candidate_logits'],av).mean()
                with torch.no_grad():anchor=model.vision_model.policy_head(ax)
                preserve=torch.nn.functional.mse_loss(ao['candidate_logits'],anchor)
                loss=.25*rank+.75*a_rank+exp['anchor_weight']*preserve+exp['duration_loss_weight']*dur
                if exp.get('risk_gate_enabled',False):
                    risk_target=torch.as_tensor((nearest[selected]<30).astype(np.float32)[None],device=device)
                    risk_raw=torch.nn.functional.binary_cross_entropy_with_logits(out['risk_logits'],risk_target,reduction='none',pos_weight=torch.tensor(10.,device=device))
                    loss=loss+weighted_mean(risk_raw,loss_mask)
                if exp.get('student_danger_anchor_weight',0.)>0:
                    with torch.no_grad():student_anchor=model.vision_model.policy_head(x)
                    critical=torch.as_tensor(nearest[selected]<15,device=device)[None]&loss_mask
                    anchor_loss=weighted_mean(((out['candidate_logits']-student_anchor)**2).mean(-1),critical)
                    loss=loss+exp['student_danger_anchor_weight']*anchor_loss
                if not torch.isfinite(loss):raise ValueError('nonfinite temporal loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step();step+=1;seen+=len(selected)
        print(f'[phase3] epoch {epoch+1} steps {step} recurrent_frames {seen}',flush=True)
    after=frozen_digest(model)
    if before!=after:raise RuntimeError('frozen pixel model changed')
    return dict(steps=step,student_frame_exposures=seen,A_frame_exposures=3*seen,duration_train_class_counts=counts.tolist(),duration_class_weights=weights.tolist(),frozen_before=before,frozen_after=after)


def run(protocol,source,cache,out,device_name='cuda'):
    protocol,source,cache,out=map(Path,(protocol,source,cache,out))
    if out.exists():raise FileExistsError(out)
    p=json.loads(protocol.read_text());exp=p['experiment'];device=torch.device(device_name if torch.cuda.is_available() else 'cpu')
    seal_path=protocol.parent/'training-code-identity.json'
    seal=json.loads(seal_path.read_text())
    if seal.get('protocol_sha256')!=sha(protocol):raise ValueError('training code seal protocol mismatch')
    for name,value in seal['sources'].items():
        if sha(Path(__file__).with_name(name))!=value:raise ValueError('training code changed after seal: '+name)
    for name,value in p.get('source_code_sha256',{}).items():
        if sha(Path(__file__).with_name(name))!=value:raise ValueError('sealed source mismatch '+name)
    baseline=Path(p['baseline_checkpoint'])
    if sha(baseline)!=p['baseline_sha256']:raise ValueError('Phase2 baseline mismatch')
    validate(cache,verify_source=True);report,rows,aligned,files=audit_source(source)
    if report['protocol_sha256']!=sha(protocol) or report['checkpoint_sha256']!=p['baseline_sha256']:raise ValueError('Phase3 source identity mismatch')
    with np.load(cache/'chunk-arrays.npz') as z:ca={k:z[k].copy() for k in z.files}
    episode_map={ep:split for split,eps in p['episodes'].items() for ep in eps}
    segments=build_sequence_index(rows,episode_map,p['temporal_data']['max_frame_gap']);history=previous_history(rows,segments)
    pc=np.full(len(aligned),18,np.int64);pd=np.full(len(aligned),3,np.int64)
    for i,(a,d) in history.items():pc[i]=a;pd[i]=d
    eligible=ca['model_valid'].all((1,2))
    clean_segments=[]
    for segment in segments:
        current=[]
        for index in segment:
            if eligible[index]:current.append(index)
            elif current:clean_segments.append(current);current=[]
        if current:clean_segments.append(current)
    segments=clean_segments
    history=previous_history(rows,segments);pc[:]=18;pd[:]=3
    for i,(a,d) in history.items():pc[i]=a;pd[i]=d
    by_split={key:[s for s in segments if episode_map[ca['episode'][s[0]]]==key] for key in ('train','dev','test')}
    by_split={key:[s for s in value if s] for key,value in by_split.items()}
    eval_ids={key:sorted(i for s in value for i in s) for key,value in by_split.items()}
    if len(eval_ids['test'])<p['gates']['min_test_frames']:raise ValueError('insufficient Phase3 test samples')
    vision_model,extra=load_vision_checkpoint(baseline,map_location='cpu');vision_model.to(device).eval()
    out.mkdir(parents=True);ds=VisionDataset(source,history_contract='causal-segment-v1')
    all_ids=list(range(len(ds)));features,bullets=bank(ds,vision_model,all_ids,device,out/'student-features.npz',dict(source_report_sha256=sha(source/'student-rollout-report.json'),baseline_sha256=sha(baseline)))
    ds.close();target=ca['clearance'][:,:,1].astype(np.float32);chunk=chunk_targets(ca['clearance'],ca['model_valid'],p['experiment']['duration_safety_margin_px'])
    nearest=vision_metrics._nearest_bullet_px(bullets)
    adir=Path('third_party/th08/build/rl/datasets/pilot-ud0059-v2');manifest=DatasetManifest.load(adir/'manifest.json');a_train,a_val=manifest.verify(adir)
    if sha(adir/'manifest.json')!=p['A_manifest_sha256'] or index_digest(a_val)!=p['A_val_index_sha256']:raise ValueError('fixed A identity mismatch')
    a=VisionDataset(adir,history_contract='causal-segment-v1');a_subset=_stratified_a(a_train,min(len(a_train),3*len(eval_ids['train'])),a)
    a_ids=sorted(set(a_subset+a_val));a_feature,a_bullets=bank(a,vision_model,a_ids,device,out/'A-features.npz',dict(manifest_sha256=sha(adir/'manifest.json'),baseline_sha256=sha(baseline),indices_sha256=index_digest(a_ids)))
    a_lookup={x:i for i,x in enumerate(a_ids)};a_train_local=[a_lookup[x] for x in a_subset];a_val_local=[a_lookup[x] for x in a_val]
    ti=teacher_identity(a,adir,manifest=manifest,val_index_sha256=index_digest(a_val));fp=_fingerprint(ti);a_teacher,reason=_inspect_teacher_cache(teacher_cache_path(adir,fp),fp,len(a))
    if a_teacher is None:raise ValueError('verified A teacher cache missing '+reason)
    a_target=a_teacher[a_ids].astype(np.float32);a.close()
    torch.manual_seed(exp['seed']);model=TemporalPolicy(vision_model,TemporalConfig(risk_gate_enabled=exp.get('risk_gate_enabled',False),risk_gate_threshold=exp.get('risk_gate_threshold',.1))).to(device);control=copy.deepcopy(model)
    with torch.inference_mode():base_scores=vision_model.policy_head(torch.as_tensor(features,device=device)).cpu().numpy();base_A=vision_model.policy_head(torch.as_tensor(a_feature,device=device)).cpu().numpy()
    base_dur=np.zeros((len(features),18,3),np.float32);base_dur[:,:,1]=1
    baseline_candidates=base_scores.argmax(1)
    results={};training={};frozen={}
    # Treat missing label rows only as boundaries, never concatenate across gaps.
    for label,policy,reset_each in [('temporal',model,False),('reset_control',control,True)]:
        training[label]=train(policy,features,by_split['train'],pc,pd,target,chunk['best_duration'],chunk['duration_valid'],nearest,
          a_feature[a_train_local],a_target[a_train_local],device,{**exp,'burn_in':p['temporal_data']['burn_in']},reset_each)
        scores,dur=infer(policy,features,segments,pc,pd,device,reset_each)
        results[label]={key:metric(scores,dur,target,bullets,eval_ids[key],ca['episode'],ca['clearance'],baseline_candidates) for key in ('dev','test')}
        ascores,adur=infer(policy,a_feature,[[i] for i in a_val_local],np.full(len(a_feature),18),np.full(len(a_feature),3),device,True)
        results[label]['A']=metric(ascores,adur,a_target,a_bullets,a_val_local,np.zeros(len(a_feature),int))
    results['baseline']={key:metric(base_scores,base_dur,target,bullets,eval_ids[key],ca['episode'],ca['clearance'],baseline_candidates) for key in ('dev','test')}
    results['baseline']['A']=metric(base_A,np.zeros((len(a_feature),18,3)),a_target,a_bullets,a_val_local,np.zeros(len(a_feature),int))
    bootstrap={}
    for key in ('dev','test'):
        ep=np.asarray(results['baseline'][key]['per_frame']['episode']);distance=np.asarray(results['baseline'][key]['per_frame']['nearest_bullet_px']);danger=distance<15
        before=np.asarray(results['baseline'][key]['per_frame']['regret']);after=np.asarray(results['temporal'][key]['per_frame']['regret'])
        bootstrap[key]=dict(overall=_bootstrap(after-before,ep,draws=2000),danger=_bootstrap((after-before)[danger],ep[danger],draws=2000),danger_count=int(danger.sum()))
    save_temporal_checkpoint(out/'temporal-policy.pt',model,baseline,metadata=dict(protocol_sha256=sha(protocol),cache_sha256=sha(cache/'chunk-identity.json')))
    save_temporal_checkpoint(out/'reset-control.pt',control,baseline,metadata=dict(protocol_sha256=sha(protocol),reset_each_step=True))
    result=dict(schema='v200-phase3-experiment/1',protocol_sha256=sha(protocol),source_report_sha256=sha(source/'student-rollout-report.json'),cache_identity_sha256=sha(cache/'chunk-identity.json'),baseline_sha256=sha(baseline),
      splits={key:dict(episodes=p['episodes'][key],frames=len(eval_ids[key]),segments=len(by_split[key]),index_sha256=index_digest(eval_ids[key])) for key in eval_ids},
      training=training,metrics=results,bootstrap=bootstrap,history=dict(source='only previous actual executed actions/durations',episode_split_crossings=0,segments=len(segments)),
      B_used=False,phase4_executed=False,scope='finite linear existing-pool model proxy; no survival/world safety claim')
    (out/'phase3-report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='metrics'},indent=2));return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--source',required=True);parser.add_argument('--cache',required=True);parser.add_argument('--out',required=True);parser.add_argument('--device',default='cuda');args=parser.parse_args();run(args.protocol,args.source,args.cache,args.out,args.device)
