"""v3.7.3 training gates. Engineering smoke is NOT gameplay training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import torch


def validate_training_gate(manifest):
    """Fail closed before touching weights; v3.7.3 real data requires explicit gates."""
    obj = json.loads(Path(manifest).read_text(encoding='utf-8'))
    required = {'schema': 'v373-paired-events/1', 'teacher_gate': 'passed',
                'pixel_alignment_gate': 'passed', 'data_audit_gate': 'passed',
                'input_kind': 'real_pixels'}
    for key, expected in required.items():
        if obj.get(key) != expected:
            raise ValueError(f'training gate rejected: {key} must be {expected!r}')
    groups = obj.get('split_groups', {})
    if set(groups) != {'train', 'validation', 'test'} or any(not groups[s] for s in groups):
        raise ValueError('nonempty independent train/validation/test groups required')
    if any(len(set(v)) != len(v) for v in groups.values()):
        raise ValueError('duplicate split groups')
    if any(set(groups[a]) & set(groups[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('split leakage')
    return obj


def engineering_smoke(baseline, out, *, steps=100, device='cuda', seed=7):
    """Optimize fabricated feature labels solely to verify CUDA/backprop/frozen SHA.

    No scenario labels, pixel dataset or capability claim is produced. This is
    intentionally not T21, which requires eight credible real training events.
    """
    from .tactical_policy import TacticalConfig, TacticalPolicy
    from .vmodels import load_vision_checkpoint
    from .macro_train import file_sha
    out = Path(out)
    if out.exists():
        raise FileExistsError('use a new smoke output directory')
    if steps < 1 or steps > 2000:
        raise ValueError('engineering smoke requires 1..2000 steps')
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    backbone, _ = load_vision_checkpoint(baseline, map_location='cpu')
    model = TacticalPolicy(backbone, TacticalConfig(arm='N1')).to(device).train()
    frozen = {n:p.detach().clone() for n,p in model.vision_model.named_parameters()}
    features = torch.randn(4, 32, 256, device=device)
    with torch.no_grad():
        base_logits = backbone.policy_head(features).detach()
    # Easily separable non-physical label for testing the optimizer, never a teacher.
    target = (features[..., 0] > 0).long()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    begin = time.perf_counter()
    losses = []
    for step in range(steps):
        output = model.forward_features(features, base_logits)
        logits = output['policy_logits']
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1,18), target.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 25 == 0:
            print(json.dumps({'step':step,'loss':losses[-1]}), flush=True)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    elapsed = time.perf_counter()-begin
    with torch.no_grad():
        result = model.forward_features(features, base_logits)
        accuracy = float((result['policy_logits'].argmax(-1)==target).float().mean())
    unchanged = all(torch.equal(p, frozen[n]) for n,p in model.vision_model.named_parameters())
    if not unchanged:
        raise AssertionError('frozen backbone changed')
    out.mkdir(parents=True)
    # Deliberately distinct schema: cannot masquerade as deployable trained checkpoint.
    torch.save({'schema':'v373-engineering-smoke-only/1',
        'policy_state_dict':model.state_dict(), 'baseline_sha256':file_sha(baseline),
        'warning':'fabricated feature labels; not gameplay trained; must not deploy'}, out/'engineering-only.pt')
    report = {'schema':'v373-engineering-optimization/1', 'seed':seed,
        'steps':steps, 'device':device, 'batch_sequences':4, 'sequence_length':32,
        'amp':False, 'seconds':elapsed, 'steps_per_second':steps/elapsed,
        'first_loss':losses[0], 'last_loss':losses[-1], 'fabricated_label_accuracy':accuracy,
        'backbone_unchanged':unchanged, 'baseline_sha256':file_sha(baseline),
        'trainable_parameters':sum(p.numel() for p in parameters),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated() if device.startswith('cuda') else None,
        'real_pixel_training':False, 'T21_passed':False, 'capability':'unvalidated',
        'warning':'Engineering CUDA/optimizer smoke only; no reliable event dataset yet.', 'losses':losses}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='losses'},indent=2))
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    smoke=sub.add_parser('engineering-smoke',help='fabricated-feature optimizer check; NOT T21')
    smoke.add_argument('--baseline',type=Path,required=True)
    smoke.add_argument('--out',type=Path,required=True)
    smoke.add_argument('--steps',type=int,default=100)
    smoke.add_argument('--device',default='cuda')
    smoke.add_argument('--seed',type=int,default=7)
    gate=sub.add_parser('check-gate')
    gate.add_argument('--manifest',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='engineering-smoke':
        engineering_smoke(args.baseline,args.out,steps=args.steps,device=args.device,seed=args.seed)
    else:
        validate_training_gate(args.manifest)
        print('Data prerequisites validated; full event trainer must also validate content hashes.')


if __name__=='__main__':
    main()
