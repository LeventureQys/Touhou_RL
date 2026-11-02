"""Prepare an explicitly mixed Phase 2 student-relabeled pilot dataset.

This copies only whole source shards into a new directory. It never changes A or
student-state source directories, and writes provenance before processing.
"""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path

SCHEMA = "v200-phase2-mixed-pilot/1"

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--A", required=True); p.add_argument("--student", required=True); p.add_argument("--out", required=True)
    p.add_argument("--A-shards", type=int, default=25)
    a=p.parse_args(argv); A=Path(a.A); S=Path(a.student); O=Path(a.out)
    if O.exists() and any(O.iterdir()): raise FileExistsError(O)
    a_paths=sorted(A.glob("collect-*.npz"))[:a.A_shards]
    s_paths=sorted(S.glob("student-state-*.npz"))
    if not a_paths or not s_paths: raise ValueError("both source shard sets are required")
    O.mkdir(parents=True)
    copied=[]
    for src in a_paths+s_paths:
        dst=O/src.name
        shutil.copy2(src,dst)
        side=src.with_name(src.name+".json")
        if side.exists(): shutil.copy2(side,O/side.name)
        with __import__("numpy").load(src,allow_pickle=False) as d: frames=int(d["action"].shape[0])
        copied.append({"source":str(src),"name":src.name,"frames":frames,"sha256":sha(dst),"source_sha256":sha(src)})
    total_a=sum(x["frames"] for x in copied[:len(a_paths)]); total_s=sum(x["frames"] for x in copied[len(a_paths):])
    provenance={"schema":SCHEMA,"training_approved":True,"B_used":False,"A_source":str(A),"student_source":str(S),"A_manifest_sha256":sha(A/"manifest.json"),"student_report_sha256":sha(S/"student-rollout-report.json"),"ratio_A_to_student":total_a/total_s,"A_frames":total_a,"student_frames":total_s,"shards":copied}
    (O/"phase2-provenance.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(provenance,ensure_ascii=False,indent=2))
    return 0
if __name__=="__main__": raise SystemExit(main())
