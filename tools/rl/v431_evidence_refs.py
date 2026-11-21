"""v4.3.1 recheck R46: every evidence reference resolves, and no evidence is orphaned.

The report's K/C/N tables and the command table point at evidence groups (`R1`..`R45`). A
reference that does not resolve is a citation to nothing -- the same class of problem as C14,
but at the file level instead of the number level.

This checks BOTH directions:

  forward  every `R<n>` named in the report has a non-empty `evidence/R<n>/` directory;
  reverse  every `evidence/R<n>/` directory is named somewhere in the report (an orphan
           directory means evidence was produced but never cited, which is also a defect).

Writes only Document/Update/Alpha/v4.3.1/evidence/R46/.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
REPORT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "ROUTE_RECHECK_2026-11-15.md"
EV = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence"
OUT = EV / "R46"

REF = re.compile(r"\bR(\d{1,2})\b")

# three groups are PROCESS STEPS, not evidence: their output is the assertion self-check or the
# report's own sections, so no directory is owed. Naming them here is what keeps the check honest
# -- otherwise "missing" would be silenced by a blanket exemption.
PROCESS_GROUPS = {
    13: ("报告自检", "python tools/rl/v431_verify_report.py 的断言数即输出"),
    17: ("判据算术自测", "python tools/rl/v431_seg001_gate.py --selftest 的 24 项即输出"),
    40: ("未复现/未执行清单复核", "输出是本报告 §6.3 / §7.2 与断言本身"),
}


def main() -> int:
    text = REPORT.read_text(encoding="utf-8")
    referenced = sorted({int(m.group(1)) for m in REF.finditer(text)})

    dirs = {}
    for d in sorted(EV.iterdir()):
        m = re.fullmatch(r"R(\d+)", d.name)
        if m and d.is_dir():
            files = [p for p in d.rglob("*") if p.is_file()]
            dirs[int(m.group(1))] = {"dir": d.name, "files": len(files),
                                     "bytes": sum(p.stat().st_size for p in files)}

    forward = []
    for n in referenced:
        info = dirs.get(n)
        forward.append({
            "group": f"R{n}",
            "dir_exists": info is not None,
            "files": info["files"] if info else 0,
            "resolves": bool(info and info["files"] > 0),
        })
    for r in forward:
        n = int(r["group"][1:])
        r["process_step"] = n in PROCESS_GROUPS
        if r["process_step"]:
            r["why_no_dir"] = PROCESS_GROUPS[n][1]
    missing = [r["group"] for r in forward
               if not r["resolves"] and not r["process_step"]]

    reverse = [{"group": f"R{n}", "files": v["files"], "bytes": v["bytes"],
                "cited": n in referenced}
               for n, v in sorted(dirs.items())]
    orphans = [r["group"] for r in reverse if not r["cited"]]

    out = {
        "schema": "v431-recheck-r46/1",
        "question": ("does every evidence reference in the report resolve, and is every evidence "
                     "directory cited?"),
        "counts": {
            "distinct_groups_referenced": len(referenced),
            "groups_with_a_nonempty_dir": sum(1 for r in forward if r["resolves"]),
            "process_steps_without_dir": sum(1 for r in forward if r["process_step"]),
            "missing_references": len(missing),
            "evidence_dirs": len(dirs),
            "orphan_dirs": len(orphans),
        },
        "missing_references": missing,
        "orphan_dirs": orphans,
        "forward": forward,
        "reverse": reverse,
        "verdict": {"all_references_resolve": not missing,
                    "no_orphans": not orphans,
                    "reading": None},
    }
    v = out["verdict"]
    if not missing and not orphans:
        v["reading"] = ("all {} cited evidence groups resolve: {} have a non-empty directory and "
                        "{} are declared process steps with no file owed; all {} evidence "
                        "directories are cited".format(
                            len(referenced), len(referenced) - len(PROCESS_GROUPS),
                            len(PROCESS_GROUPS), len(dirs)))
    else:
        v["reading"] = (f"{len(missing)} cited groups have no evidence on disk ({missing}); "
                        f"{len(orphans)} evidence directories are never cited ({orphans})")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evidence_refs.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"distinct R-groups referenced : {len(referenced)}")
    print(f"  with non-empty evidence dir: {out['counts']['groups_with_a_nonempty_dir']}")
    print(f"  missing                    : {missing or 'none'}")
    print(f"  declared process steps     : {sorted(PROCESS_GROUPS)}")
    print()
    print(f"evidence dirs on disk        : {len(dirs)}")
    print(f"  never cited                : {orphans or 'none'}")
    print()
    for r in reverse:
        flag = "" if r["cited"] else "  <-- ORPHAN"
        print(f"  {r['group']:>5s} files={r['files']:>3d} bytes={r['bytes']:>8d}{flag}")
    print()
    print(f"verdict: {v['reading']}")
    print(f"wrote: {OUT / 'evidence_refs.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
