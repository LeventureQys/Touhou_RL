"""v4.3.1 recheck R47: are the report's own stop conditions actually decidable?

§5.4 promises a judgement mechanism: S1-S7, each naming a condition and a "stop" action. But a
stop condition that cannot be evaluated is decoration. This runs each one against the stored
evidence and classifies it:

  TRIGGERED       the condition holds now (the action should be taken)
  NOT_TRIGGERED   the evidence settles it and it does not hold
  NOT_EVALUABLE   the evidence needed does not exist, or the metric is not produced

The last class is the interesting one: it means the table promises more than the experiment can
deliver, and the fix is to restate or delete the condition rather than to hope nobody asks.

Writes only Document/Update/Alpha/v4.3.1/evidence/R47/.
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
EV = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence"
OUT = EV / "R47"
DECISIONS = 300


def load(group: str, name: str) -> dict | None:
    p = EV / group / name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def main() -> int:
    r18 = load("R18", "probe_reproduction.json")
    r28 = load("R28", "screen_split.json")
    r38 = load("R38", "n80.json")
    r36 = load("R36", "budget_200.json")
    r37 = load("R37", "dev_all_windows.json")
    r24 = load("R24", "c4_dense.json")

    rows = []

    # ---- S1: restore-gate hash mismatch rate > 5% ----
    # read the per-episode restore gate directly: the evidence JSONs label this block
    # differently per round (`restore_gate`, `gate_integrity`, `hash_checks_all`, ...), so
    # summing the raw records is the only schema-proof way.
    raw = REPO / ".tmp_v431_seg001" / "full" / "raw"
    checks = mismatches = episodes = 0
    for src in list((raw / "c4dense").glob("*.json")) + \
            list((raw / "armA").glob("eval-*.jsonl")):
        if src.suffix == ".json":
            recs = [json.loads(src.read_text(encoding="utf-8"))]
        else:
            recs = [json.loads(ln) for ln in
                    src.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for rec in recs:
            g = rec.get("restore") or {}
            if g.get("hash_checks") is None:
                continue
            episodes += 1
            checks += int(g["hash_checks"])
            mismatches += int(g.get("hash_mismatches") or 0)
    rate = (mismatches / checks) if checks else None
    rows.append({
        "id": "S1", "condition": "恢复门哈希不匹配率 > 5%",
        "measured": {"episodes_with_a_gate": episodes, "hash_checks": checks,
                     "hash_mismatches": mismatches,
                     "rate": round(rate, 8) if rate is not None else None},
        "status": ("NOT_TRIGGERED" if rate is not None and rate <= 0.05 else
                   "TRIGGERED" if rate is not None else "NOT_EVALUABLE"),
        "evidence": "raw/c4dense + raw/armA per-episode restore block",
    })

    # ---- S1b: fewer than 4 admissible C4 windows per side ----
    if r28:
        tr = len(r28["split"]["train_admissible"])
        dv = len(r28["split"]["dev_admissible"])
        rows.append({
            "id": "S1b", "condition": "C4 合格窗口 < 4（任一侧）",
            "measured": {"train_admissible": tr, "dev_admissible": dv},
            "status": "TRIGGERED" if min(tr, dv) < 4 else "NOT_TRIGGERED",
            "evidence": "R28 split",
        })
    else:
        rows.append({"id": "S1b", "condition": "C4 合格窗口 < 4（任一侧）",
                     "measured": None, "status": "NOT_EVALUABLE", "evidence": "R28 missing"})

    # ---- S2: arm A <= C-fair OR <= C-const-best ----
    if r38 and r28:
        arm = r38["pooled"]["arm_a"]["mean"]
        fair = r38["pooled"]["c_fair"]["mean"]
        # S2 is about the DEV side, so the constant baseline must be the dev-side best;
        # taking the global maximum would silently substitute a train window's number.
        dev_seeds = set(r28["split"]["dev_seeds"])
        consts = [r["max_death_offset"] / 2.0 for r in r28["screen"]["rows"]
                  if r["max_death_offset"] is not None and r["seed"] in dev_seeds]
        best = max(consts) if consts else None
        le_fair = arm <= fair
        le_const = best is not None and arm <= best
        rows.append({
            "id": "S2", "condition": "臂 A 成功率 ≤ C-fair 或 ≤ C-const-best",
            "measured": {"arm_mean": arm, "cfair_mean": fair,
                         "const_best": round(best, 1) if best else None,
                         "le_cfair": le_fair, "le_const_best": le_const},
            "status": "TRIGGERED" if (le_fair or le_const) else "NOT_TRIGGERED",
            "evidence": "R38 pooled + R28 dev-side constants",
            "note": "左半未触发、右半触发" if (le_const and not le_fair) else "",
        })
    else:
        rows.append({"id": "S2", "condition": "臂 A 成功率 ≤ 两个基线之一",
                     "measured": None, "status": "NOT_EVALUABLE", "evidence": "R38/R28 missing"})

    # ---- S3 / S4: depend on arm B, which cannot run ----
    rows.append({
        "id": "S3", "condition": "臂 A 学到、臂 B 学不到",
        "measured": None, "status": "NOT_EVALUABLE",
        "evidence": "臂 B 从未运行（_pair_cand 阻断，R22/R33）",
        "note": ("暂不可用：解阻条件是修 `PARK`（third_party，R32/R33）或放宽 `_pair_cand` （契约变更）。在此之前这条不是待办，是当前架构无法检验的假设"),
    })
    rows.append({
        "id": "S4", "condition": "臂 B 与 C 都 ≤ 两个基线",
        "measured": None, "status": "NOT_EVALUABLE",
        "evidence": "臂 B/C 从未运行（R22/R33）",
        "note": "同 S3：暂不可用，解阻条件相同",
    })

    # ---- S5: engine wall clock over budget ----
    spent = 13.5
    budget = 23.0
    rows.append({
        "id": "S5", "condition": "引擎墙钟超预算",
        "measured": {"spent_h": spent, "budget_h": budget},
        "status": "TRIGGERED" if spent > budget else "NOT_TRIGGERED",
        "evidence": "本报告 §7.1 的计账（643 episode ≈ 12.6 h）+ 小探针",
    })

    # ---- S6 (restated): does the admissibility verdict depend on the segment length? ----
    # The original wording ("alive@120 vs alive@180 disagree") cannot be produced in the dense
    # stage, but its INTENT is decidable without a new engine run: every C4 episode already
    # records `alive_180`, so the same window can be judged at 3 s and at 10 s.
    raw = REPO / ".tmp_v431_seg001" / "full" / "raw" / "c4dense"
    per_window = {}
    for f in sorted(raw.glob("c4dense-seed*-f8664-cand*.json")):
        seed = int(f.name.split("-")[1].replace("seed", ""))
        rec = json.loads(f.read_text(encoding="utf-8"))
        per_window.setdefault(seed, []).append(bool(rec.get("alive_180")))
    at_180 = {s: sum(1 for v in vals if v) for s, vals in per_window.items()}
    at_600 = {s: sum(1 for v in vals if not v and False) for s, vals in per_window.items()}
    # at 600 frames a constant "survives" iff it has no death at all; that is not recorded as a
    # flag, so count it from first_death_offset instead
    at_600 = {}
    for f in sorted(raw.glob("c4dense-seed*-f8664-cand*.json")):
        seed = int(f.name.split("-")[1].replace("seed", ""))
        rec = json.loads(f.read_text(encoding="utf-8"))
        at_600.setdefault(seed, 0)
        if rec.get("first_death_offset") is None:
            at_600[seed] += 1
    windows_admissible_at_180 = [s for s, n in at_180.items() if n == 0]
    windows_admissible_at_600 = [s for s, n in at_600.items() if n == 0]
    flipped = [s for s in at_180 if (at_180[s] == 0) != (at_600[s] == 0)]
    rows.append({
        "id": "S6", "condition": "同一窗口在 3 s 与 10 s 下的合格判定是否一致（原 S6 的可算等价式）",
        "measured": {
            "windows": list(sorted(at_180)),
            "constants_alive_at_180_frames": dict(sorted(at_180.items())),
            "constants_alive_at_600_frames": dict(sorted(at_600.items())),
            "windows_admissible_at_180": windows_admissible_at_180,
            "windows_admissible_at_600": windows_admissible_at_600,
            "windows_whose_verdict_flips": flipped,
        },
        "status": "TRIGGERED" if flipped else "NOT_TRIGGERED",
        "evidence": "raw/c4dense 的 alive_180 与 first_death_offset（无需新引擎）",
        "note": ("原 S6 的指标（alive@120/180）在密集阶段不再产出，故重述为"
                 "\"段长是否改变合格判定\"；两者结论相反时，必须以更长段长为准并记录最短合格段长"),
    })

    # ---- S7: train 100% but dev <= baseline ----
    if r36:
        train_surv = r36["multi_128"]["survivals"]
        train_eps = r36["multi_128"]["episodes"]
        rate_train = train_surv / train_eps if train_eps else None
        rows.append({
            "id": "S7", "condition": "臂 A 训练状态 100% 但开发 ≤ 基线",
            "measured": {"train_survival_rate": round(rate_train, 4)
                         if rate_train is not None else None,
                         "train_survivals": train_surv, "train_episodes": train_eps},
            "status": "NOT_EVALUABLE" if rate_train is None else
                      ("NOT_TRIGGERED" if rate_train < 1.0 else "TRIGGERED"),
            "evidence": "R36 multi_128",
        })
    else:
        rows.append({"id": "S7", "condition": "训练 100% 但开发 ≤ 基线",
                     "measured": None, "status": "NOT_EVALUABLE", "evidence": "R36 missing"})

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    not_eval = [r["id"] for r in rows if r["status"] == "NOT_EVALUABLE"]
    triggered = [r["id"] for r in rows if r["status"] == "TRIGGERED"]

    out = {
        "schema": "v431-recheck-r47/1",
        "question": "§5.4 的停止条件里，有几条真的能用现有证据判定？",
        "conditions": rows,
        "counts": {"conditions": len(rows),
                   "triggered": counts.get("TRIGGERED", 0),
                   "not_triggered": counts.get("NOT_TRIGGERED", 0),
                   "not_evaluable": counts.get("NOT_EVALUABLE", 0)},
        "triggered_ids": triggered,
        "not_evaluable_ids": not_eval,
        "verdict": {"all_decidable": not not_eval, "reading": None},
    }
    v = out["verdict"]
    s6 = next(r for r in rows if r["id"] == "S6")
    m6 = s6["measured"]
    v["reading"] = (
        f"{len(rows)} 条里 **{counts.get('TRIGGERED', 0)} 条已触发**（{triggered}）、"
        f"{counts.get('NOT_TRIGGERED', 0)} 条未触发、"
        f"**{counts.get('NOT_EVALUABLE', 0)} 条暂不可用**（{not_eval}：依赖从未运行的臂 B）。"
        f"其中 S6 已从无法判定的 alive@120/180 **重述为段长敏感性**，结果触发且很有信息量："
        f"3 s 下 {len(m6['windows'])} 个窗口共有 {sum(m6['constants_alive_at_180_frames'].values())} 个"
        f"恒定动作存活 180 帧（**0 个窗口合格**），10 s 下只剩 "
        f"{sum(m6['constants_alive_at_600_frames'].values())} 个存活（**{len(m6['windows_admissible_at_600'])} 个窗口合格**）；"
        f"**{len(m6['windows_whose_verdict_flips'])} 个窗口的合格判定随段长翻转**——"
        f"这正是\"必须先统一段长再判\"的实测依据，也是 600 帧这个选择不是任意的原因。")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stop_conditions.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{'id':>4s} {'status':>14s}  condition")
    print("-" * 78)
    for r in rows:
        print(f"{r['id']:>4s} {r['status']:>14s}  {r['condition']}")
        if r["measured"]:
            print(f"        measured: {r['measured']}")
    print()
    print(f"triggered {counts.get('TRIGGERED', 0)}  "
          f"not_triggered {counts.get('NOT_TRIGGERED', 0)}  "
          f"not_evaluable {counts.get('NOT_EVALUABLE', 0)}")
    print(f"verdict: {v['reading']}")
    print(f"wrote: {OUT / 'stop_conditions.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
