"""v4.3.1 recheck R15: is the 18-way candidate space actually 18 distinct actions?

SEG-001 rev B rests on the A1 table: 18 first actions per state, some dying and some
surviving. That table only means what it says if the 18 candidates map to 18 *distinct*
engine button masks. If two candidates collapse to the same mask, the "18 arms" contain
duplicate actions, the safe-set size is overstated, and the death share is distorted.

This checks, offline and deterministically:

  A. `cand_to_mask` injectivity over 0..17
  B. the round trip `mask_to_cand(cand_to_mask(c)) == (c, bomb)`
  C. whether `focus` is encoded in the mask at all (it changes the player hitbox in
     Touhou, so a focus-insensitive mask would make two candidates physically identical)
  D. which bytes of the mask vary across the 18 candidates

Read-only. Writes only Document/Update/Alpha/v4.3.1/evidence/R15/.
"""

from __future__ import annotations

import json
import pathlib
import sys

from touhou_rl.v420 import contracts as C

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R15"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    n = int(C.N_MOVE_FOCUS)

    masks = {}
    problems = []
    for cand in range(n):
        try:
            m = int(C.cand_to_mask(cand))
        except Exception as exc:                     # noqa: BLE001 - report, do not hide
            problems.append({"cand": cand, "error": repr(exc)})
            continue
        masks[cand] = m

    inverse = {}
    for cand in range(n):
        if cand not in masks:
            continue
        try:
            cand_back, bomb_back = C.mask_to_cand(masks[cand])
        except Exception as exc:                     # noqa: BLE001
            problems.append({"cand": cand, "mask": masks[cand], "inverse_error": repr(exc)})
            continue
        inverse[cand] = {"cand": int(cand_back), "bomb": int(bomb_back)}

    by_mask: dict[int, list[int]] = {}
    for cand, m in masks.items():
        by_mask.setdefault(m, []).append(cand)
    collisions = {hex(m): cs for m, cs in by_mask.items() if len(cs) > 1}

    roundtrip_ok = all(inverse.get(c, {}).get("cand") == c for c in masks)
    bomb_zero = all(inverse.get(c, {}).get("bomb") == 0 for c in masks)

    # which bits vary across the candidate set
    or_all = 0
    and_all = 0xFF
    for m in masks.values():
        or_all |= m
        and_all &= m

    # focus sensitivity: pair candidates that differ only by `focus`
    focus_pairs = []
    seen = {}
    for cand in sorted(masks):
        # assume the cand table is ordered so that cand // 9 == focus and cand % 9 == move
        key = cand % 9
        seen.setdefault(key, []).append((cand, masks[cand]))
    for move, items in sorted(seen.items()):
        if len(items) > 1:
            focus_pairs.append({
                "move": move,
                "cands": [c for c, _ in items],
                "masks": [hex(m) for _, m in items],
                "masks_differ": len({m for _, m in items}) == len(items),
            })

    out = {
        "schema": "v431-recheck-r15/1",
        "n_move_focus": n,
        "bomb_binary": int(C.BOMB_BINARY),
        "masks": {str(c): {"mask": masks[c], "mask_hex": hex(masks[c])} for c in masks},
        "inverse": {str(c): v for c, v in inverse.items()},
        "counts": {
            "candidates": n,
            "distinct_masks": len(by_mask),
            "collisions": len(collisions),
            "problems": len(problems),
            "roundtrip_ok": bool(roundtrip_ok),
            "bomb_always_0": bool(bomb_zero),
        },
        "collision_detail": collisions,
        "problems": problems,
        "bits": {
            "or_of_all_masks": hex(or_all),
            "and_of_all_masks": hex(and_all),
            "varying_bits": hex(or_all & ~and_all & 0xFF),
        },
        "focus_axis": {
            "assumed_layout": "cand // 9 == focus, cand % 9 == move (checked below)",
            "pairs": focus_pairs,
            "all_pairs_differ_in_mask": all(p["masks_differ"] for p in focus_pairs)
            if focus_pairs else None,
        },
    }

    (OUT / "candidate_space.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"candidates={n}  distinct masks={len(by_mask)}  collisions={len(collisions)}")
    print(f"roundtrip_ok={roundtrip_ok}  bomb_always_0={bomb_zero}  problems={len(problems)}")
    print(f"varying bits across the candidate set: {out['bits']['varying_bits']}")
    print()
    print(f"{'cand':>4s} {'mask':>6s} {'hex':>6s} {'back(cand,bomb)':>18s}")
    for cand in sorted(masks):
        inv = inverse.get(cand, {})
        print(f"{cand:4d} {masks[cand]:6d} {hex(masks[cand]):>6s} "
              f"{str((inv.get('cand'), inv.get('bomb'))):>18s}")
    print()
    if collisions:
        print("!! COLLISIONS:", {k: v for k, v in collisions.items()})
    print(f"focus pairs: {len(focus_pairs)}; all differ in mask: "
          f"{out['focus_axis']['all_pairs_differ_in_mask']}")
    print()
    print(f"wrote: {OUT / 'candidate_space.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
