"""Protocol artifact inventory, excluding pixel/data payloads from final cards."""
from pathlib import Path
import json
from .build_chunk_safety_cache import sha

def inventory(root):
    root=Path(root);files={}
    for pattern in ('*.json','*.md','head-only-controlled/*.json','head-only-controlled/*.pt','student-chunk-cache/chunk-identity.json'):
        for p in root.glob(pattern):files[str(p.relative_to(root))]=dict(bytes=p.stat().st_size,sha256=sha(p))
    return dict(schema='v200-phase2-artifact-inventory/1',root=str(root.resolve()),files=files)
