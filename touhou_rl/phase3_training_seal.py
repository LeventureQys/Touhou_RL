"""Bind implementation before any Phase3 optimizer/test metrics are run."""
from pathlib import Path
import json
from .build_chunk_safety_cache import sha
NAMES=('temporal_policy.py','phase3_sequences.py','phase3_features.py','phase3_losses.py','phase3_experiment.py','phase3_executor.py','phase3_verify.py','phase3_final_gate.py')
def seal(root):
    root=Path(root);target=root/'training-code-identity.json'
    if target.exists():raise FileExistsError(target)
    result=dict(schema='v200-phase3-training-code-seal/1',protocol_sha256=sha(root/'preregistration.json'),
                sources={n:sha(Path(__file__).with_name(n)) for n in NAMES},status='sealed-before-first-optimizer-and-Phase3-test-metrics')
    target.write_text(json.dumps(result,indent=2),encoding='utf-8');return result
if __name__=='__main__':
    import argparse;p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();print(json.dumps(seal(a.root),indent=2))
