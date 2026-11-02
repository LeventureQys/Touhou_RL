"""v3.7.3 staged CLI: only implemented operations are exposed."""
from __future__ import annotations
import argparse
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    audit=sub.add_parser('audit', help='freeze actual baseline identity and optionally benchmark CUDA')
    audit.add_argument('--benchmark',action='store_true')
    smoke=sub.add_parser('engineering-smoke', help='fabricated-feature optimization; NOT formal gameplay training')
    smoke.add_argument('--baseline',type=Path,required=True)
    smoke.add_argument('--out',type=Path,required=True)
    smoke.add_argument('--steps',type=int,default=100)
    smoke.add_argument('--device',default='cuda')
    smoke.add_argument('--seed',type=int,default=7)
    gate=sub.add_parser('check-training-gate',help='reject unapproved event data before any optimizer steps')
    gate.add_argument('--manifest',type=Path,required=True)
    data=sub.add_parser('data-audit',help='audit causal event structure, SHA and split leakage')
    data.add_argument('--manifest',type=Path,required=True)
    data.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv)
    if args.command=='audit':
        from tools.rl.v373_audit import run
        run(args.benchmark)
    elif args.command=='engineering-smoke':
        from touhou_rl.tactical_train import engineering_smoke
        engineering_smoke(args.baseline,args.out,steps=args.steps,device=args.device,seed=args.seed)
    elif args.command=='data-audit':
        import json
        from touhou_rl.tactical_data import audit_manifest
        result=audit_manifest(args.manifest)
        with args.out.open('x',encoding='utf-8') as f:
            json.dump(result,f,indent=2)
        print(json.dumps(result,indent=2))
        if not result['passed']:
            raise SystemExit(2)
    elif args.command=='check-training-gate':
        from touhou_rl.tactical_train import validate_training_gate
        validate_training_gate(args.manifest)
        print('Prerequisites validated; content hashes still require verification by full event trainer.')


if __name__=='__main__':
    main()
