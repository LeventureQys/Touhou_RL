"""Single-pass Phase-2 head-only chunk-safety experiment.

This module deliberately keeps the deployed ``v3-st`` architecture unchanged.  The
chunk cache's duration=3 clearance is used as an 18-candidate target; masks are
per-candidate model-valid masks and are never derived from future death.
"""
from __future__ import annotations

import argparse, hashlib, json, math
from pathlib import Path
from typing import Iterable
import numpy as np
import torch

from . import vision, vision_losses, vision_metrics
from .build_chunk_safety_cache import validate as validate_chunk_cache
from .pipeline.manifest import DatasetManifest, file_digest, index_digest
from .vision_data import VisionDataset
from .vision_train import _gather, _compute_teacher_values, teacher_identity, _fingerprint, teacher_cache_path, _inspect_teacher_cache
from .vmodels import load_vision_checkpoint, save_vision_checkpoint, vision_arch, VISION_ARCH_V3

BASE_SHA = "7ebc32f61cd92d94fdcc56096d79fad5463d6f7f0cad68ca00b5aae703d3ad21"
SCHEMA = "v200-phase2-chunk-experiment/1"


def _sha(path: Path) -> str:
    return file_digest(Path(path))


def _digest_state(model, *, include_policy: bool = True) -> str:
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        if not include_policy and name.startswith("policy_head."):
            continue
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def _load_protocol(path: Path) -> dict:
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(protocol, dict): raise ValueError("protocol must be a JSON object")
    root = protocol.get("experiment", protocol)
    splits = root.get("splits") if isinstance(root, dict) else None
    if splits is None and isinstance(protocol.get("episodes"), dict):
        ep = protocol["episodes"]
        splits = {"student_train": ep.get("train", []), "student_dev": ep.get("dev", []), "student_test": ep.get("test", [])}
        root = dict(root) if isinstance(root, dict) else {}
        root["splits"] = splits
    if not isinstance(splits, dict):
        raise ValueError("protocol must declare experiment.splits or top-level episodes")
    required = ("student_train", "student_dev", "student_test")
    if any(k not in splits for k in required): raise ValueError("protocol missing student train/dev/test")
    for key in required:
        if not isinstance(splits[key], (list, tuple)) or not splits[key]: raise ValueError(f"empty split {key}")
    if set(splits["student_train"]) & (set(splits["student_dev"]) | set(splits["student_test"])) or set(splits["student_dev"]) & set(splits["student_test"]):
        raise ValueError("student episode splits overlap")
    if "A_val" not in splits:
        splits["A_val"] = []
    if not isinstance(protocol.get("experiment"), dict): protocol["experiment"] = {}
    protocol["experiment"]["splits"] = splits
    return protocol


def _student_rows(dataset: VisionDataset, cache_arrays: dict) -> dict[int, list[int]]:
    episodes = np.asarray(cache_arrays["episode"]).reshape(-1)
    indices = np.asarray(cache_arrays["sample_index"]).reshape(-1)
    if len(episodes) != len(dataset) or not np.array_equal(indices, np.arange(len(dataset))):
        raise ValueError("chunk cache sample identity does not match student dataset")
    out: dict[int, list[int]] = {}
    for i, ep in enumerate(episodes.tolist()): out.setdefault(int(ep), []).append(i)
    return out


def _stratified_a(indices: Iterable[int], n: int, dataset: VisionDataset) -> list[int]:
    """Deterministic stage/shard-stratified sample (never first-N)."""
    candidates = list(map(int, indices)); n = int(n)
    if n > len(candidates): raise ValueError("A train set too small")
    labels = dataset.labels(candidates)["meta"]
    groups: dict[tuple[int, int], list[int]] = {}
    for i, row in zip(candidates, labels): groups.setdefault((int(row[1]), int(i // 128)), []).append(i)
    order = sorted(groups)
    if not order: raise ValueError('A train set has no strata')
    picks=[]; cursors={key:0 for key in order}
    while len(picks)<n:
        progressed=False
        for key in order:
            if cursors[key]<len(groups[key]):
                picks.append(groups[key][cursors[key]]);cursors[key]+=1;progressed=True
                if len(picks)==n:break
        if not progressed:raise ValueError('insufficient unique A samples')
    return picks


def _student_target(cache: dict, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    # duration axis is explicitly [1,3,5]; use only duration=3 (column 1).
    clearance = np.asarray(cache["clearance"])[indices, :, 1].astype(np.float32)
    mask = np.asarray(cache["model_valid"])[indices, :, 1].astype(bool)
    if clearance.shape != mask.shape or np.any(mask & ~np.isfinite(clearance)): raise ValueError("invalid student clearance/mask")
    return clearance, mask


def _loss_masked(scores: torch.Tensor, target: np.ndarray, mask: np.ndarray, device: torch.device, sample_weight=None) -> torch.Tensor:
    valid_rows = mask.all(axis=1)
    if not bool(valid_rows.all()):
        raise ValueError("student duration=3 requires complete 18-candidate model_valid rows")
    t = torch.as_tensor(target, dtype=torch.float32, device=device)
    if sample_weight is not None:
        pairs=torch.triu_indices(18,18,offset=1,device=device)
        gap=t[:,pairs[0]]-t[:,pairs[1]];keep=gap.abs()>.5
        score_gap=scores[:,pairs[0]]-scores[:,pairs[1]]
        per=(torch.nn.functional.softplus(-gap.sign()*score_gap)*keep).sum(1)/keep.sum(1).clamp(min=1)
        w=torch.as_tensor(sample_weight,dtype=scores.dtype,device=device)
        return (per*w).sum()/w.sum().clamp(min=1)
    loss, _ = vision_losses.policy_loss(scores, t, "rank", rank_delta_px=.5, rank_weight=1., regression_weight=0.)
    return loss


def _train_steps(model, a, s, a_subset, student_train, a_values, student_targets, student_mask, device, *, include_student: bool, seed: int, batch_size: int, learning_rate: float=3e-4, anchor_weight: float=0., danger_weight: float=1., epochs: int=1) -> tuple[int, str, str]:
    import copy
    anchor_head=copy.deepcopy(model.policy_head).eval() if anchor_weight>0 else None
    if anchor_head is not None:
        for parameter in anchor_head.parameters():parameter.requires_grad_(False)
    frozen_before = _digest_state(model, include_policy=False)
    optimizer = torch.optim.AdamW(model.policy_head.parameters(), lr=learning_rate, weight_decay=1e-4)
    policy_ids = {id(p) for p in model.policy_head.parameters()}
    for p in model.parameters(): p.requires_grad_(id(p) in policy_ids)
    torch.manual_seed(seed)
    model.eval(); model.policy_head.train(); rng = np.random.default_rng(seed); steps=0
    order = rng.permutation(len(student_train))
    def danger_weights(data,ids):
        if danger_weight==1:return None
        bullets=data.labels(ids)['bullets']
        nearest=vision_metrics._nearest_bullet_px(bullets)
        return np.where(nearest<60,danger_weight,1.).astype(np.float32)
    for epoch in range(epochs):
      order = rng.permutation(len(student_train))
      for start in range(0, len(student_train), max(1, batch_size//4)):
        sub = [student_train[i] for i in order[start:start+max(1,batch_size//4)]]
        human = a_subset[3*start:3*start+3*len(sub)]
        if len(human) != 3*len(sub): continue
        optimizer.zero_grad(set_to_none=True)
        frames, _ = _gather(a, human, 3, model.config.region_sizes)
        with torch.no_grad(): feat=model.features(vision.regions_to_batch(frames, device))
        scores=model.policy_head(feat.detach())
        a_loss=.75 * _loss_masked(scores, a_values[human], np.ones_like(a_values[human], dtype=bool), device, danger_weights(a,human))
        if anchor_head is not None:
            with torch.no_grad(): anchor_scores=anchor_head(feat.detach())
            a_loss=a_loss + anchor_weight*torch.nn.functional.mse_loss(scores,anchor_scores)
        a_loss.backward()
        if include_student:
            frames, _ = _gather(s, sub, 3, model.config.region_sizes); feat=model.features(vision.regions_to_batch(frames, device)).detach(); scores=model.policy_head(feat)
            (.25 * _loss_masked(scores, student_targets[sub], student_mask[sub], device, danger_weights(s,sub))).backward()
        else:
            extra = a_subset[(3*len(student_train)+start):(3*len(student_train)+start+len(sub))]
            if len(extra) != len(sub): raise ValueError("A-only control extra samples missing")
            frames, _ = _gather(a, extra, 3, model.config.region_sizes); feat=model.features(vision.regions_to_batch(frames, device)).detach(); scores=model.policy_head(feat)
            (.25 * _loss_masked(scores, a_values[extra], np.ones_like(a_values[extra], dtype=bool), device, danger_weights(a,extra))).backward()
        optimizer.step(); steps += 1
    frozen_after = _digest_state(model, include_policy=False)
    if frozen_before != frozen_after: raise RuntimeError("non-policy weights/buffers changed")
    return steps, frozen_before, frozen_after


def _eval(model, dataset, indices, targets, device):
    preds, bullets = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), 64):
            batch = indices[start:start+64]; frames, labels = _gather(dataset, batch, 3, model.config.region_sizes)
            preds.append(model.score(vision.regions_to_batch(frames, device)).cpu().numpy()); bullets.append(labels["bullets"])
    p = np.concatenate(preds); b = np.concatenate(bullets)
    result = vision_metrics.policy_metrics(p, targets, b); chosen = p.argmax(1); result["frames"] = len(indices)
    chosen_clear=targets[np.arange(len(targets)),chosen]
    result['per_frame']={'candidate':chosen.tolist(),'regret':(targets.max(1)-chosen_clear).tolist(),
                         'best_clearance':targets.max(1).tolist(),'chosen_clearance':chosen_clear.tolist(),
                         'nearest_bullet_px':np.nan_to_num(vision_metrics._nearest_bullet_px(b),posinf=1e9).tolist()}
    result['max_single_candidate_fraction']=float(np.bincount(chosen,minlength=18).max()/len(chosen))
    result["chosen_clearance_px"] = float(np.mean(targets[np.arange(len(targets)), chosen])) if len(targets) else 0.
    result["chosen_collision_rate"] = float(np.mean(targets[np.arange(len(targets)), chosen] <= 0)) if len(targets) else 0.
    return result


def _bootstrap(values: np.ndarray, episodes: np.ndarray, seed: int = 7, draws: int = 400) -> dict:
    uniq = np.unique(episodes); rng = np.random.default_rng(seed); means=[]
    if len(values):
        for _ in range(draws):
            chosen = rng.choice(uniq, len(uniq), replace=True); means.append(float(np.mean(np.concatenate([values[episodes == e] for e in chosen]))))
    if not means: return {"episodes": int(len(uniq)), "mean": 0., "lo95": 0., "hi95": 0.}
    return {"episodes": int(len(uniq)), "mean": float(np.mean(values)), "lo95": float(np.percentile(means,2.5)), "hi95": float(np.percentile(means,97.5))}


def run(*, baseline: Path, a_dir: Path, student: Path, cache: Path, protocol: Path, out: Path, seed: int = 7, batch_size: int = 32, device_name: str = "cpu") -> dict:
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    protocol_obj = _load_protocol(protocol)
    experiment=protocol_obj.get('experiment',{})
    if experiment.get('epochs',1) not in (1,2) or seed!=experiment.get('seed',7) or batch_size!=experiment.get('batch_size',32):
        raise ValueError('runtime seed/batch/epochs disagree with sealed protocol')
    for name, expected in protocol_obj.get('source_code_sha256',{}).items():
        if _sha(Path(__file__).with_name(name))!=expected:
            raise ValueError(f'sealed source code changed: {name}')
    # Calibration is a hard prerequisite: absence, malformed JSON, or pass != true
    # must stop before loading/training the policy.  Accept the canonical gate file
    # beside the student rollout or cache for portable experiment bundles.
    gate_candidates = [Path(protocol).parent / "calibration-gate.json", Path(student) / "calibration-gate.json", Path(cache) / "calibration-gate.json"]
    gate_path = next((p for p in gate_candidates if p.exists()), None)
    if gate_path is None:
        raise ValueError("calibration gate missing; refusing Phase-2 training")
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("calibration gate unreadable; refusing Phase-2 training") from exc
    if gate.get("protocol_sha256") and gate["protocol_sha256"] != _sha(protocol):
        raise ValueError("calibration gate protocol identity mismatch")
    if gate.get("passed") is not True:
        raise ValueError("calibration gate did not pass; refusing Phase-2 training")
    if gate.get("pass") is False:
        raise ValueError("calibration gate did not pass; refusing Phase-2 training")
    if gate.get('protocol_sha256') != _sha(protocol):
        raise ValueError('calibration gate protocol identity missing/mismatch')
    residual = gate.get("position_residual_p95_px")
    if residual is None:
        attempts=gate.get('attempts', [])
        residual=attempts[-1].get('residual_p95') if attempts and attempts[-1].get('passed') is True else None
    if residual is None or not math.isfinite(float(residual)) or float(residual) > 3.0:
        raise ValueError("calibration position residual exceeds preregistered gate")
    if gate.get('validated_against_measured_clearance') is True:
        from .verify_clearance_evidence import verify
        verify(Path(protocol).parent)
    if _sha(baseline) != BASE_SHA: raise ValueError("unexpected baseline checkpoint")
    gate_obj = json.loads(Path(protocol).read_text(encoding="utf-8"))
    if gate_obj.get("checkpoint_sha256") != BASE_SHA: raise ValueError("protocol checkpoint identity mismatch")
    if gate_obj.get("A_manifest_sha256") and _sha(a_dir / "manifest.json") != gate_obj["A_manifest_sha256"]: raise ValueError("A manifest identity mismatch")
    validate_chunk_cache(cache, verify_source=True)
    source_report=json.loads((student/'student-rollout-report.json').read_text(encoding='utf-8'))
    if source_report.get('behavior_policy')!='v3-st-pixel-student' or source_report.get('protocol_sha256')!=_sha(protocol):
        raise ValueError('student source behavior/protocol identity mismatch')
    with np.load(cache / "chunk-arrays.npz", allow_pickle=False) as z: ca = {k: z[k] for k in z.files}
    manifest = DatasetManifest.load(a_dir / "manifest.json"); a_train, a_val = manifest.verify(a_dir)
    if len(a_val) != 7241: raise ValueError("fixed A_val mismatch")
    if gate_obj.get("A_val_index_sha256") and index_digest(a_val) != gate_obj["A_val_index_sha256"]: raise ValueError("A_val index identity mismatch")
    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    model, baseline_extra = load_vision_checkpoint(baseline, map_location=device)
    model.to(device).eval()
    if vision_arch(model) != VISION_ARCH_V3 or model.config.depth != 3: raise ValueError("baseline v3 depth3 required")
    a = VisionDataset(a_dir, history_contract="causal-segment-v1"); s = VisionDataset(student, history_contract="causal-segment-v1")
    try:
        ep_rows = _student_rows(s, ca); splits = protocol_obj.get("experiment", protocol_obj)["splits"]
        for shard,start,size in s.offsets:
            with np.load(shard,allow_pickle=False) as z:
                meta=z['meta'];episode=z['decision_meta'][:,0]
                for j in range(1,size):
                    if episode[j]!=episode[j-1] and meta[j,1]==meta[j-1,1] and meta[j,7]==meta[j-1,7] and meta[j,0]-meta[j-1,0]==2:
                        raise ValueError('causal-segment history could cross episode boundary')
        print('[phase2] student episode/history isolation verified',flush=True)
        split_idx = {k: [i for ep in splits[k] for i in ep_rows.get(int(ep), [])] for k in ("student_train", "student_dev", "student_test")}
        if min(map(len, split_idx.values())) == 0: raise ValueError("protocol split has no cache rows")
        student_targets, student_mask = _student_target(ca, list(range(len(s))))
        split_idx = {key:[i for i in ids if student_mask[i].all()] for key,ids in split_idx.items()}
        from .build_chunk_safety_cache import audit_source
        _,trace_rows,aligned_rows,_=audit_source(student)
        by_index={r['sample_index']:r for r in aligned_rows}
        split_idx={key:[i for i in ids if not by_index[i].get('terminated') and not by_index[i].get('truncated') and by_index[i].get('stage_delta',0)==0] for key,ids in split_idx.items()}
        # The target is simulated from the current snapshot only (no observed
        # future data); purge applies to actual history, not imaginary futures.
        history_proof={'contract':'causal-segment-v1','episode_crossings':0,'split_crossings':0,
                       'target_future_source':'current snapshot linear extrapolation only; no future trace labels'}
        exp=protocol_obj['experiment']
        for key,minimum in [('student_train',exp.get('min_train_samples',200)),('student_dev',exp.get('min_dev_samples',100)),('student_test',exp.get('min_test_samples',100))]:
            if len(split_idx[key])<minimum: raise ValueError(f'{key} insufficient independently model-valid samples')
        student_train = split_idx["student_train"]; n_student = len(student_train); a_subset = _stratified_a(a_train, 4*n_student, a)
        a_identity=teacher_identity(a,a_dir,manifest=manifest,val_index_sha256=index_digest(a_val))
        a_fingerprint=_fingerprint(a_identity)
        a_cache_path=teacher_cache_path(a_dir,a_fingerprint)
        a_values,a_cache_reason=_inspect_teacher_cache(a_cache_path,a_fingerprint,len(a))
        if a_values is None:
            print(f'[phase2] A teacher recompute: {a_cache_reason}',flush=True)
            a_values = _compute_teacher_values(a, verbose=True)
        else: print(f'[phase2] verified A teacher cache hit {a_cache_path}',flush=True)
        base_a = _eval(model, a, a_val, a_values[a_val], device)
        base_s = {k: _eval(model, s, split_idx[k], student_targets[split_idx[k]], device) for k in ("student_dev", "student_test")}
        # Train independent mixed and A-only models with identical optimizer steps.
        import copy
        mixed = model
        control = copy.deepcopy(model)
        mixed_steps, mixed_frozen_before, mixed_frozen_after = _train_steps(mixed, a, s, a_subset, student_train, a_values, student_targets, student_mask, device, include_student=True, seed=seed, batch_size=batch_size, learning_rate=protocol_obj['experiment'].get('learning_rate',3e-4),anchor_weight=protocol_obj['experiment'].get('anchor_weight',0.),danger_weight=protocol_obj['experiment'].get('danger_weight',1.),epochs=protocol_obj['experiment'].get('epochs',1))
        control_steps, control_frozen_before, control_frozen_after = _train_steps(control, a, s, a_subset, student_train, a_values, student_targets, student_mask, device, include_student=False, seed=seed, batch_size=batch_size, learning_rate=protocol_obj['experiment'].get('learning_rate',3e-4),anchor_weight=protocol_obj['experiment'].get('anchor_weight',0.),danger_weight=protocol_obj['experiment'].get('danger_weight',1.),epochs=protocol_obj['experiment'].get('epochs',1))
        if mixed_steps != control_steps: raise RuntimeError("mixed/control optimizer step mismatch")
        after_a = _eval(mixed, a, a_val, a_values[a_val], device)
        after_s={k:_eval(mixed,s,split_idx[k],student_targets[split_idx[k]],device) for k in ("student_dev","student_test")}
        control_a = _eval(control, a, a_val, a_values[a_val], device)
        control_s={k:_eval(control,s,split_idx[k],student_targets[split_idx[k]],device) for k in ("student_dev","student_test")}
        frozen_before, frozen_after, steps = mixed_frozen_before, mixed_frozen_after, mixed_steps
        if frozen_before != frozen_after or control_frozen_before != control_frozen_after: raise RuntimeError("non-policy weights/buffers changed")
        bootstrap={}
        for key in ('student_dev','student_test'):
            ids=split_idx[key]; ep=ca['episode'][ids]
            before=np.asarray(base_s[key]['per_frame']['regret'])
            mixed_reg=np.asarray(after_s[key]['per_frame']['regret'])
            control_reg=np.asarray(control_s[key]['per_frame']['regret'])
            best=np.asarray(base_s[key]['per_frame']['nearest_bullet_px']); danger=(best>=0)&(best<15)
            bootstrap[key]={}
            for label,mask in [('overall',np.ones(len(ids),bool)),('danger_0_15',danger)]:
                bootstrap[key][label]={'frames':int(mask.sum()),
                    'mixed_minus_baseline':_bootstrap((mixed_reg-before)[mask],ep[mask],draws=protocol_obj['experiment'].get('bootstrap_episode_replicates',2000)),
                    'mixed_minus_control':_bootstrap((mixed_reg-control_reg)[mask],ep[mask],draws=protocol_obj['experiment'].get('bootstrap_episode_replicates',2000))}
        behavior={}
        for key in ('student_dev','student_test'):
            ids=split_idx[key]; valid_pairs=[]
            for j in range(1,len(ids)):
                left,right=by_index[ids[j-1]],by_index[ids[j]]
                if left['episode']==right['episode'] and left['stage']==right['stage'] and right['decision']==left['decision']+1 and right['frame']>left['frame'] and left['deaths_before']==left['deaths_after']==right['deaths_before'] and not left.get('terminated') and not left.get('truncated'):
                    valid_pairs.append((j-1,j))
            behavior[key]={}
            for label,metric in [('baseline',base_s[key]),('mixed',after_s[key]),('A_only_control',control_s[key])]:
                chosen=metric['per_frame']['candidate']
                behavior[key][label]={'valid_adjacent_pairs':len(valid_pairs),'candidate_flip':float(np.mean([chosen[a]!=chosen[b] for a,b in valid_pairs])) if valid_pairs else None,
                    'max_single_candidate_fraction':metric['max_single_candidate_fraction']}
        phase2_extra={**baseline_extra,'source_state_sha':BASE_SHA,'teacher_fingerprint':_sha(cache/'chunk-identity.json'),
                      'manifest_sha':_sha(a_dir/'manifest.json'),'val_index_sha256':index_digest(a_val),
                      'loss_config':{'policy_loss':'rank','human_action_weight':0,'learning_rate':protocol_obj['experiment'].get('learning_rate',3e-4),
                                     'phase2_student_report_sha256':_sha(student/'student-rollout-report.json'),'target_duration_decisions':3},
                      'phase2_schema':SCHEMA,'baseline_sha256':BASE_SHA,'protocol_sha256':_sha(protocol)}
        out.mkdir(parents=True,exist_ok=True); ckpt=out/"policy.pt"; save_vision_checkpoint(ckpt, mixed, extra=phase2_extra)
        save_vision_checkpoint(out/'A-only-control.pt',control,extra={**phase2_extra,'phase2_behavior':'A-only-control'})
        report={"schema":SCHEMA,"baseline_sha256":BASE_SHA,"baseline_checkpoint_sha256":_sha(baseline),"protocol_sha256":_sha(protocol),"calibration_gate_sha256":_sha(gate_path),"cache_identity_sha256":_sha(cache/"chunk-identity.json"),"cache_validated":True,"target":"chunk clearance duration=3 column 1","candidate_mapping":"focus*9+move","input_depth":3,"history_contract":"causal-segment-v1","history_isolation":history_proof,"A_val_index_sha256":index_digest(a_val),"A_manifest_sha256":_sha(a_dir/'manifest.json'),"splits":{k:{"episodes":splits[k],"frames":len(split_idx[k]),"index_sha256":index_digest(split_idx[k])} for k in split_idx},"train":{"A_frames":3*n_student,"control_A_frames":4*n_student,"student_frames":n_student,"ratio":3,"optimizer_steps":steps,"control_optimizer_steps":control_steps,"epochs":protocol_obj['experiment'].get('epochs',1),"learning_rate":protocol_obj['experiment'].get('learning_rate',3e-4),"anchor_weight":protocol_obj['experiment'].get('anchor_weight',0.),"danger_weight":protocol_obj['experiment'].get('danger_weight',1.),"danger_weight_radius_px":60,"seed":seed,"batch_size":batch_size,"stratified_A":True,"mask":"model_valid duration=3 only; future_death not used"},"before":{"A":base_a,"student":base_s},"after":{"mixed":{"A":after_a,"student":after_s},"A_only_control":{"A":control_a,"student":control_s}},"frozen_state_sha256":{"mixed":{"before":frozen_before,"after":frozen_after},"control":{"before":control_frozen_before,"after":control_frozen_after}},"executed_clearance_calibration":{"validated_against_measured_clearance":gate.get('validated_against_measured_clearance') is True,"gate_sha256":_sha(gate_path),"clearance_error_p95_px":gate.get('clearance_error_p95_px'),"false_safe_count":gate.get('false_safe_count')},"bootstrap":bootstrap,"student_behavior":behavior,"checkpoint":str(ckpt),"checkpoint_sha256":_sha(ckpt),"B_used":False,"real_game_executed":False}
        (out/"phase2-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"); return report
    finally: a.close(); s.close()


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--baseline", required=True); p.add_argument("--A", required=True); p.add_argument("--student", required=True); p.add_argument("--cache", required=True); p.add_argument("--protocol", required=True); p.add_argument("--out", required=True); p.add_argument("--device", default="cpu"); x=p.parse_args(argv)
    print(json.dumps(run(baseline=Path(x.baseline),a_dir=Path(x.A),student=Path(x.student),cache=Path(x.cache),protocol=Path(x.protocol),out=Path(x.out),device_name=x.device), indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
