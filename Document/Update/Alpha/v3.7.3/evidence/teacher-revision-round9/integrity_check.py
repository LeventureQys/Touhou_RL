"""Post-result integrity audit; does not select actions or modify frozen policy."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
from touhou_rl.tactical_teacher_validation import ValidationWorld, verify
from touhou_rl.tactical_teacher_v2 import hashes
out=Path(__file__).resolve().parent
manifest=json.loads((out/'confirmation-manifest.json').read_text())
assert manifest['source_sha256']==hashes()
result={}; failures=[]; verified=0; svgs=0
for phase in ('development','confirmation'):
    rows=[]
    for path in sorted((out/phase).glob('*.json')):
        r=json.loads(path.read_text())
        if 'scenario' not in r: continue
        s=ValidationWorld(**r['scenario'])
        for arm in ('v2','old_teacher','reference'):
            m=r[arm]
            if m:
                assert len(m['actions'])==80 and len(m['trace'])==81
                assert verify(s,m['actions'])==m['verdict']
                if arm=='reference':
                    assert m['verdict']=='feasible'; verified+=1
            if arm!='reference' and m['verdict']!='feasible':
                failures.append(dict(phase=phase,event_id=s.event_id,arm=arm,verdict=m['verdict'],first_collision_frame=m['first_collision_frame']))
        ET.parse(path.with_suffix('.svg')); svgs+=1; rows.append(r)
    for split in sorted({r['scenario']['split'] for r in rows}):
        result[split]={}
        for family in ('F1','F2','F3'):
            rr=[r for r in rows if r['scenario']['split']==split and r['scenario']['family']==family]
            result[split][family]={'instances':len(rr)}
            for arm in ('v2','old_teacher','reference'):
                result[split][family][arm]={key:sum(r[arm][key] for r in rr) for key in ('survived','reached_goal','corridor_occupancy_bullet_frames','final_horizontal_space_px')}
                result[split][family][arm]['verified']=sum(r[arm]['verdict']=='feasible' for r in rr)
(out/'detailed-summary.json').write_text(json.dumps(result,indent=2))
(out/'all-arm-failures.json').write_text(json.dumps(failures,indent=2))
(out/'integrity-audit.json').write_text(json.dumps(dict(reference_full_replays=verified,xml_svgs_parsed=svgs,all_three_arm_verdicts_rechecked=True,source_equal_to_frozen=hashes()==manifest['source_sha256'],source_sha256_after=hashes(),executed_R8_test=0),indent=2))
print(json.dumps(result,indent=2))
