"""Read-only train/dev eligibility diagnostics; intentionally no test metrics."""
from pathlib import Path
import json
from collections import Counter

def summarize(source,protocol):
    source,protocol=Path(source),Path(protocol);p=json.loads(protocol.read_text());out={}
    for split in ('train','dev'):
        phases=Counter(); n=0; aligned=0
        for ep in p['episodes'][split]:
            file=source/f'episode-{ep:03d}-decisions.jsonl'
            if not file.exists():continue
            rows=[json.loads(l) for l in file.read_text().splitlines()]
            n+=len(rows);aligned+=sum(r.get('sample_index') is not None for r in rows)
            phases.update(str(r.get('raw_safety_snapshot',{}).get('player_state')) for r in rows)
        out[split]=dict(decisions=n,aligned=aligned,phases=dict(phases))
    return dict(schema='v200-train-dev-eligibility/1',test_read=False,splits=out)
