"""Phase 2 episode-heldout, provenance-checked head-only policy pilot.

No ordinary VisionDataset manifest split is used: A's original validation split
is reused, student episode 1 trains, episode 2 is a disjoint diagnostic holdout.
Training and evaluation take only pixel stacks; privilege enters teacher labels.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from . import vision, vision_losses, vision_metrics
from .pipeline.manifest import DatasetManifest, file_digest, index_digest
from .vision_data import VisionDataset
from .vision_train import _gather, _compute_teacher_values
from .vmodels import load_vision_checkpoint, vision_arch, VISION_ARCH_V3
from .stage0_diagnostics import summarize_actions

BASE_SHA = "7ebc32f61cd92d94fdcc56096d79fad5463d6f7f0cad68ca00b5aae703d3ad21"


def _values(dataset, source):
    """Teacher is recomputed *on each dataset*; no mixed cache or A overwrite."""
    return _compute_teacher_values(dataset, verbose=False)


def _model_digest(model):
    import hashlib
    h=hashlib.sha256()
    for name,param in model.named_parameters():
        if not name.startswith("policy_head."):
            h.update(name.encode()); h.update(param.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _action_summary(model,dataset,indices,device):
    acts=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(indices),64):
            batch=indices[start:start+64]; frames,_=_gather(dataset,batch,3,model.config.region_sizes)
            score=model.score(vision.regions_to_batch(frames,device)).cpu().numpy()
            best=score.argmax(axis=1); acts.extend([[int(i%9),1,int(i//9),0] for i in best])
    return summarize_actions(acts)


def _evaluate(model, dataset, indices, values, device, batch_size=64):
    predictions=[]; targets=[]; bullets=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(indices),batch_size):
            batch=indices[start:start+batch_size]
            frames,labels=_gather(dataset,batch,3,model.config.region_sizes)
            score=model.score(vision.regions_to_batch(frames,device))
            predictions.append(score.cpu().numpy())
            targets.append(values[batch].astype(np.float32))
            bullets.append(labels["bullets"])
    if not predictions: raise ValueError("empty evaluation set")
    result=vision_metrics.policy_metrics(np.concatenate(predictions),np.concatenate(targets),np.concatenate(bullets))
    result["frames"]=len(indices)
    result["index_sha256"]=index_digest(sorted(indices))
    return result


def run(*, a_dir:Path, student_dir:Path, baseline:Path, out_dir:Path, seed=7, epochs=1, batch_size=32, device_name="cuda", relabel_cache:Path|None=None, train_episodes=(1,2,3,4), holdout_episodes=(5,6)):
    if out_dir.exists() and any(out_dir.iterdir()): raise FileExistsError(out_dir)
    if not train_episodes or not holdout_episodes or set(train_episodes)&set(holdout_episodes):
        raise ValueError("student train and holdout episode groups must be disjoint")
    if file_digest(baseline)!=BASE_SHA: raise ValueError("unexpected baseline checkpoint")
    student_report=json.loads((student_dir/"student-rollout-report.json").read_text(encoding="utf-8"))
    if student_report["phase1_collection_status"]!="collection-ready-for-distribution-audit" or student_report["training_approved"] is not False: raise ValueError("student source must be approved diagnostic Phase1 round")
    if student_report["checkpoint_sha256"]!=BASE_SHA: raise ValueError("student policy identity mismatch")
    for row in student_report["shards"]:
        if file_digest(student_dir/row["name"])!=row["sha256"]: raise ValueError("student shard SHA mismatch")
    cache_identity=None
    if relabel_cache is not None:
        cache_identity=json.loads((relabel_cache/"relabel-identity.json").read_text(encoding="utf-8"))
        if cache_identity.get("schema")!="v200-student-relabel-cache/2": raise ValueError("relabel cache schema mismatch")
        if cache_identity.get("rollout_report_sha256")!=file_digest(student_dir/"student-rollout-report.json"):
            raise ValueError("relabel cache source report mismatch")
        if cache_identity.get("teacher_source_sha256")!=student_report["teacher_source_sha256"]:
            raise ValueError("relabel teacher identity mismatch")
        if cache_identity.get("cache_sha256")!=file_digest(relabel_cache/"relabel-arrays.npz"):
            raise ValueError("relabel cache payload SHA mismatch")
        if cache_identity.get("student_shards_sha256")!=[x["sha256"] for x in student_report["shards"]]:
            raise ValueError("relabel cache shard identity mismatch")
    manifest=DatasetManifest.load(a_dir/"manifest.json")
    a_train,a_val=manifest.verify(a_dir)
    if len(a_val)!=7241: raise ValueError("fixed A validation set missing")
    device=torch.device(device_name if torch.cuda.is_available() else "cpu")
    model,extra=load_vision_checkpoint(baseline,map_location=device)
    if vision_arch(model)!=VISION_ARCH_V3 or model.config.depth!=3: raise ValueError("baseline v3-st depth3 required")
    model.to(device).eval()
    a=VisionDataset(a_dir,history_contract="causal-segment-v1")
    s=VisionDataset(student_dir,history_contract="causal-segment-v1")
    try:
        student_rows=[]
        for path,start,size in s.offsets:
            with np.load(path,allow_pickle=False) as d:
                if "decision_meta" not in d or "death_window_valid" not in d: raise ValueError("student provenance/mask missing")
                for local in range(size):
                    episode=int(d["decision_meta"][local,0]); valid=int(d["death_window_valid"][local])
                    student_rows.append((start+local,episode,valid))
        s_train=[i for i,e,v in student_rows if e in train_episodes and v==1]
        s_val=[i for i,e,v in student_rows if e in holdout_episodes and v==1]
        if min(len(s_train),len(s_val))<100: raise ValueError("episode holdout too small")
        if set(s_train)&set(s_val): raise ValueError("student episode train/holdout overlap")
        if len(a_train)<3*len(s_train): raise ValueError("A train set too small for 3:1 mix")
        if cache_identity is not None:
            with np.load(relabel_cache/"relabel-arrays.npz",allow_pickle=False) as cache:
                if len(cache["sample_index"])!=len(s) or not np.array_equal(cache["sample_index"],np.arange(len(s))):
                    raise ValueError("relabel cache sample indexes mismatch")
                cache_valid=cache["future_death_valid"].astype(bool)
                source_valid=np.asarray([bool(v) for _i,_e,v in student_rows])
                if np.any(cache_valid & ~source_valid):
                    raise ValueError("relabel cache claims validity absent in source shard")
                s_train=[i for i in s_train if cache_valid[i]]
                s_val=[i for i in s_val if cache_valid[i]]
                if min(len(s_train),len(s_val))<100: raise ValueError("valid relabel holdout too small")
        # Fixed A first N train indices gives 3:1 against episode-1 student only.
        a_subset=sorted(a_train)[:3*len(s_train)]
        if set(a_subset)&set(a_val): raise ValueError("A train/val overlap")
        a_values=_values(a,a_dir); s_values=_values(s,student_dir)
        base_a=_evaluate(model,a,a_val,a_values,device)
        base_s=_evaluate(model,s,s_val,s_values,device)
        base_actions=_action_summary(model,s,s_val,device)
        frozen_sha_before=_model_digest(model)
        model.train()
        for name,param in model.named_parameters(): param.requires_grad_(name.startswith("policy_head."))
        optimizer=torch.optim.AdamW(model.policy_head.parameters(),lr=3e-4,weight_decay=1e-4)
        rng=np.random.default_rng(seed)
        steps=0
        for epoch in range(epochs):
            rng.shuffle(a_subset); rng.shuffle(s_train)
            # Each step draws equal-sized student/A minibatches in 3:1 ratio.
            for start in range(0,len(s_train),max(1,batch_size//4)):
                sub=s_train[start:start+max(1,batch_size//4)]
                human=a_subset[3*start:3*start+3*len(sub)]
                if not human: break
                optimizer.zero_grad(set_to_none=True)
                for data,idx,labels,scale in ((a,human,a_values,0.75),(s,sub,s_values,0.25)):
                    frames,_=_gather(data,idx,3,model.config.region_sizes)
                    with torch.no_grad(): feat=model.features(vision.regions_to_batch(frames,device))
                    scores=model.policy_head(feat.detach())
                    target=torch.as_tensor(labels[idx].astype(np.float32),device=device)
                    loss,_=vision_losses.policy_loss(scores,target,"rank",rank_delta_px=0.5,rank_weight=1.0,regression_weight=1.0)
                    (scale*loss).backward()
                optimizer.step();steps+=1
        after_a=_evaluate(model,a,a_val,a_values,device)
        after_s=_evaluate(model,s,s_val,s_values,device)
        after_actions=_action_summary(model,s,s_val,device)
        frozen_sha_after=_model_digest(model)
        if frozen_sha_after != frozen_sha_before:
            raise RuntimeError("encoder/perception changed during policy-head-only training")
        out_dir.mkdir(parents=True)
        checkpoint=out_dir/"head-only.pt"
        torch.save({"schema":"v200-phase2-head-only/1","baseline_sha256":BASE_SHA,"state_dict":model.state_dict(),"config":model.config.__dict__,"source_A_manifest_sha256":file_digest(a_dir/"manifest.json"),"student_report_sha256":file_digest(student_dir/"student-rollout-report.json")},checkpoint)
        report={"schema":"v200-phase2-head-only-report/1","checkpoint":str(checkpoint),"checkpoint_sha256":file_digest(checkpoint),"baseline":str(baseline),"baseline_sha256":BASE_SHA,"A_manifest_sha256":file_digest(a_dir/"manifest.json"),"student_report_sha256":file_digest(student_dir/"student-rollout-report.json"),"student_shards_sha256":[x["sha256"] for x in student_report["shards"]],"train":{"A_frames":len(a_subset),"student_frames":len(s_train),"ratio":len(a_subset)/len(s_train),"epochs":epochs,"steps":steps,"seed":seed,"policy_head_only":True,"human_CE":0},"holdout":{"A_frames":len(a_val),"A_val_index_sha256":index_digest(a_val),"student_episodes":list(holdout_episodes),"student_frames":len(s_val),"student_indices_sha256":index_digest(s_val),"student_valid_only":True},"relabel_cache_identity_sha256":file_digest(relabel_cache/"relabel-identity.json") if relabel_cache else None,"auxiliary_labels_used_for_policy_loss":False,"teacher_value_semantics":"AvoidPilot 18 candidate clearance recomputed from source privileged labels; observational risk/duration cache used for validity/identity only","student_train_episodes":list(train_episodes),"before":{"A":base_a,"student":base_s},"after":{"A":after_a,"student":after_s},"student_actions":{"before":base_actions,"after":after_actions},"frozen_parameter_sha256":{"before":frozen_sha_before,"after":frozen_sha_after},"B_used":False,"real_game_executed":False}
        (out_dir/"phase2-report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        return report
    finally:
        a.close();s.close()

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument("--A",required=True);parser.add_argument("--student",required=True);parser.add_argument("--baseline",required=True);parser.add_argument("--out",required=True);parser.add_argument("--epochs",type=int,default=1);parser.add_argument("--device",default="cuda")
    parser.add_argument("--relabel-cache",required=True)
    parser.add_argument("--train-episodes",default="1,2,3,4")
    parser.add_argument("--holdout-episodes",default="5,6")
    x=parser.parse_args(argv)
    train_episodes=tuple(int(i) for i in x.train_episodes.split(","))
    holdout_episodes=tuple(int(i) for i in x.holdout_episodes.split(","))
    result=run(a_dir=Path(x.A),student_dir=Path(x.student),baseline=Path(x.baseline),out_dir=Path(x.out),epochs=x.epochs,device_name=x.device,relabel_cache=Path(x.relabel_cache),train_episodes=train_episodes,holdout_episodes=holdout_episodes)
    print(json.dumps({"before":result["before"],"after":result["after"],"checkpoint_sha256":result["checkpoint_sha256"]},ensure_ascii=False,indent=2))
    return 0
if __name__=="__main__":raise SystemExit(main())
