"""Reproducible v3.7.3 baseline/hardware audit; never opens the game."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'Document/Update/Alpha/v3.7.3'
BASE = ROOT / 'third_party/th08/build/rl/runs/v104-v3-E5-s7/policy.pt'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def command(args):
    # File redirection avoids nested captured pipes on restricted Windows hosts.
    target = OUT / 'evidence' / ('audit-' + Path(args[0]).stem + '.txt')
    with target.open('w', encoding='utf-8') as f:
        result = subprocess.run(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    return {'exit_code': result.returncode, 'output': str(target.relative_to(ROOT))}


def run(benchmark=False):
    from touhou_rl.vmodels import load_vision_checkpoint, vision_arch
    (OUT / 'evidence').mkdir(parents=True, exist_ok=True)
    (OUT / 'configs').mkdir(exist_ok=True)
    model, extra = load_vision_checkpoint(BASE, map_location='cpu')
    config = asdict(model.config)
    baseline_sha = sha(BASE)
    environment = {'schema': 'v373-environment/1', 'python': sys.version,
        'executable': sys.executable, 'platform': platform.platform(),
        'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(), 'cuda_available': torch.cuda.is_available(),
        'baseline': {'path': str(BASE.relative_to(ROOT)), 'sha256': baseline_sha,
                     'arch': vision_arch(model), 'config': config,
                     'parameters': sum(p.numel() for p in model.parameters()),
                     'checkpoint_metadata_keys': sorted(extra)},
        'git': command(['git', 'status', '--short']),
        'nvidia_smi': command(['nvidia-smi']),
        'live_calibration': 'not_run', 'pixel_alignment': 'not_verified'}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        environment['gpu'] = {'name': p.name, 'total_memory_bytes': p.total_memory,
                              'capability': list(torch.cuda.get_device_capability(0))}
    if benchmark:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA benchmark requires CUDA')
        torch.set_num_threads(4)
        model = model.cuda().eval()
        rows = []
        for batch in (1, 4, 8, 16, 32):
            regions = {name: torch.rand(batch, model.config.depth, *size, 3, device='cuda')
                       for name, size in model.config.region_sizes.items()}
            for precision in ('float32', 'float16'):
                torch.cuda.reset_peak_memory_stats()
                samples = []
                with torch.inference_mode():
                    for i in range(15):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        with torch.autocast('cuda', dtype=torch.float16, enabled=precision == 'float16'):
                            result = model.forward_core(regions)
                        torch.cuda.synchronize()
                        if i >= 5:
                            samples.append((time.perf_counter()-start)*1000)
                samples.sort()
                rows.append({'batch': batch, 'precision': precision,
                    'median_ms': samples[len(samples)//2], 'p95_ms': samples[-1],
                    'images_per_second': batch * 1000 / (sum(samples)/len(samples)),
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'finite': bool(torch.isfinite(result['policy_logits']).all())})
                print(json.dumps(rows[-1]), flush=True)
            del regions
        environment['benchmark'] = {'kind': 'synthetic_tensor_real_backbone_forward_not_game_latency',
            'warmup': 5, 'measured_iterations': 10, 'cpu_threads': 4, 'rows': rows}
    base = {'schema': 'v373-base/1', 'baseline': str(BASE.relative_to(ROOT)),
            'baseline_sha256': baseline_sha, 'arch': vision_arch(model),
            'control_period_frames': 2, 'duration_cycles': 1, 'shoot': True,
            'bomb': False, 'ema': False, 'switch_penalty': 0,
            'arms': ['B0', 'B1', 'N0', 'N1'], 'seeds': [7, 17, 27],
            'default_deployment_changed': False, 'live_gate': 'pending_calibration',
            'pixel_alignment_unknown': 'exclude_from_exact_geometric_labels'}
    (OUT/'configs/base-v1.json').write_text(json.dumps(base, indent=2), encoding='utf-8')
    (OUT/'evidence/environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    paths = [BASE, ROOT/'touhou_rl/vmodels3.py', ROOT/'touhou_rl/escape/harness.py',
             ROOT/'touhou_rl/macro_policy.py', ROOT/'touhou_rl/temporal_policy.py', Path(__file__)]
    (OUT/'evidence/source-sha256.json').write_text(json.dumps(
        {str(p.relative_to(ROOT)): sha(p) for p in paths}, indent=2), encoding='utf-8')
    print(json.dumps(environment, indent=2))
    return environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    run(args.benchmark)


if __name__ == '__main__':
    main()
