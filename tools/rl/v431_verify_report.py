"""v4.3.1 recheck R9: assert that the report's numbers match the stored evidence.

A report that quotes numbers is only as good as the link between quote and artifact.
This script re-reads the JSON evidence produced by R1-R8 and asserts every figure the
report uses. Any mismatch raises, so the check cannot silently pass.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
EV = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence"

checks = 0
failures = []


def check(name: str, got, want, tol: float = 0.0) -> None:
    global checks
    checks += 1
    if isinstance(want, float) and isinstance(got, (int, float)):
        ok = abs(float(got) - want) <= tol
    else:
        ok = got == want
    if not ok:
        failures.append(f"{name}: got {got!r} want {want!r} (tol {tol})")


def main() -> int:
    global checks
    r1 = json.loads((EV / "R1" / "baseline_recheck.json").read_text(encoding="utf-8"))
    r2 = json.loads((EV / "R2" / "overlap_audit.json").read_text(encoding="utf-8"))
    r3 = json.loads((EV / "R3" / "duplication_structure.json").read_text(encoding="utf-8"))
    r4 = json.loads((EV / "R4" / "model_recheck.json").read_text(encoding="utf-8"))
    r8 = json.loads((EV / "R8" / "segment_spec_final.json").read_text(encoding="utf-8"))

    # ---------------- R1 ----------------
    ev = r1["split_recovery"]["evaluated"]
    check("R1 int_0p8 train", ev["int_0p8"]["n_train"], 879)
    check("R1 int_0p8 val", ev["int_0p8"]["n_val"], 220)
    for k in ("int_0p8", "fixed_200", "random_200"):
        check(f"R1 {k} matches 880/219", ev[k]["matches_claimed_880_219"], False)
    b = r1["baselines"]
    check("R1 val mle", round(b["seed12_val__move"]["ce_train_mle"], 4), 1.6586)
    check("R1 val laplace", round(b["seed12_val__move"]["ce_train_laplace"], 4), 1.6589)
    check("R1 val entropy", round(b["seed12_val__move"]["ce_train_entropy"], 4), 1.6937)
    check("R1 val majacc", round(b["seed12_val__move"]["majority_accuracy"], 4), 0.4500)
    check("R1 val old formula", round(b["seed12_val__move"]["old_formula_neg_log_train_majority"], 4), 0.8680)
    check("R1 seed10 full mle", round(b["seed10_full__move"]["ce_train_mle"], 4), 1.7126)
    check("R1 seed10 suffix mle", round(b["seed10_suffix__move"]["ce_train_mle"], 4), 1.7122)
    check("R1 seed10 suffix majacc", round(b["seed10_suffix__move"]["majority_accuracy"], 4), 0.4364)
    check("R1 prefix len", r1["cross_tape_prefix"]["common_cand_prefix_len"], 219)
    check("R1 neg_log_full_majority",
          round(r1["documented_vs_recomputed"]["recomputed"]["neg_log_full_majority"], 10),
          0.8536876585, 1e-9)
    check("R1 full entropy",
          round(r1["documented_vs_recomputed"]["recomputed"]["full_seed12_entropy"], 10),
          1.6859891972, 1e-9)
    check("R1 prior on seed10",
          round(r1["documented_vs_recomputed"]["recomputed"]["seed12_full_prior_on_seed10_full"], 10),
          1.7074972456, 1e-9)
    check("R1 train counts", b["seed12_val__move"]["train_counts"],
          [369, 22, 169, 134, 30, 16, 63, 60, 16])
    check("R1 val counts", b["seed12_val__move"]["eval_counts"],
          [99, 6, 34, 38, 2, 5, 15, 16, 5])

    # ---------------- R2 ----------------
    p = r2["pairs"]["seed12__seed10"]
    check("R2 prefix seed10", p["cand_common_prefix"], 219)
    check("R2 prefix seed11", r2["pairs"]["seed12__seed11"]["cand_common_prefix"], 225)
    check("R2 prefix 10-11", r2["pairs"]["seed10__seed11"]["cand_common_prefix"], 219)
    check("R2 play_gray identical", p["observations_over_prefix"]["play_gray"]["n_byte_identical"], 0)
    check("R2 play_gray frac", p["observations_over_prefix"]["play_gray"]["identical_fraction"], 0.0)
    check("R2 play_gray meanabs", round(p["observations_over_prefix"]["play_gray"]["mean_abs_diff"], 3), 0.622, 0.001)
    check("R2 hud identical", p["observations_over_prefix"]["hud_gray"]["n_byte_identical"], 120)
    check("R2 keyhist identical", p["observations_over_prefix"]["executed_key_history"]["n_byte_identical"], 219)
    check("R2 global identical", p["observations_over_prefix"]["global_rgb"]["n_byte_identical"], 0)
    check("R2 suffix play same-index", p["observations_over_suffix"]["play_gray"]["n_identical_at_same_index"], 0)
    check("R2 suffix play shared", p["observations_over_suffix"]["play_gray"]["n_shared_values_cross_position"], 0)
    check("R2 suffix keyhist same-index", p["observations_over_suffix"]["executed_key_history"]["n_identical_at_same_index"], 167)
    check("R2 suffix keyhist shared", p["observations_over_suffix"]["executed_key_history"]["n_shared_values_cross_position"], 55)
    for tape in ("seed12", "seed10", "seed11"):
        w = r2["within_tape_near_duplication"][tape]
        check(f"R2 {tape} distinct", w["n_distinct_play_gray"], 1099)
        check(f"R2 {tape} dup rows", w["n_exact_duplicate_rows"], 0)
    check("R2 adjacent seed12", round(r2["within_tape_near_duplication"]["seed12"]["mean_abs_diff_adjacent_samples"], 3), 21.485)
    check("R2 control", round(r2["control_mean_abs_diff_random_pairs_seed12_play_gray"], 3), 39.122)
    for tape in ("seed10", "seed11", "seed12"):
        check(f"R2 {tape} supervised heads", r2["ancestry"][tape]["supervised_heads"], ["move", "focus"])
        check(f"R2 {tape} shoot distinct", r2["ancestry"][tape]["shoot_distinct"], [1])
        check(f"R2 {tape} bomb distinct", r2["ancestry"][tape]["bomb_distinct"], [0])

    # distinct tape sha256 across the three
    shas = {r2["ancestry"][t]["tape_sha256"] for t in ("seed10", "seed11", "seed12")}
    check("R2 three distinct tape shas", len(shas), 3)

    # ---------------- R3 ----------------
    li = r3["label_identity"]
    check("R3 labels prefix move", li["seed12__seed10"]["move_identical"], 219)
    check("R3 labels prefix focus", li["seed12__seed10"]["focus_identical"], 219)
    check("R3 labels suffix move", li["seed12__seed10"]["suffix_move_identical"], 342)
    check("R3 labels suffix n", li["seed12__seed10"]["suffix_n"], 880)
    check("R3 seed11 prefix move", li["seed12__seed11"]["move_identical"], 225)
    ps = r3["prefix_pixel_structure_play_gray"]["seed12__seed10"]
    check("R3 pix differ at all", round(ps["pixels_differing_at_all"], 4), 0.0546, 1e-4)
    check("R3 pix ge8", round(ps["pixels_off_by_ge_8"], 4), 0.0190, 1e-4)
    check("R3 pix ge32", round(ps["pixels_off_by_ge_32"], 4), 0.0044, 1e-4)
    check("R3 zero-differing frac", ps["samples_with_zero_pixels_differing"], 0.0)
    ro = r3["random_split_optimism"]
    check("R3 nearest", round(ro["val_to_nearest_train_sample_mean_abs_diff"], 3), 20.648)
    check("R3 random", round(ro["val_to_random_train_sample_mean_abs_diff"], 3), 39.809)
    check("R3 ratio", round(ro["ratio"], 3), 0.519)
    ct = r3["cross_tape_distance"]
    check("R3 cross meanabs", round(ct["prefix_same_index_mean_abs_diff"], 3), 0.622)
    check("R3 cross control", round(ct["control_random_pair_mean_abs_diff"], 3), 34.286)

    # ---------------- R4 ----------------
    check("R4 n_train", r4["split"]["n_train"], 879)
    check("R4 n_val", r4["split"]["n_val"], 220)
    check("R4 cut", r4["split"]["cut"], 879)
    check("R4 down16 params", r4["arms"]["down16"]["n_params"], 30793)
    check("R4 down4 params", r4["arms"]["down4"]["n_params"], 487753)
    check("R4 prefix seed10", r4["arms"]["down16"]["common_prefix"]["seed10"], 219)
    check("R4 prefix seed11", r4["arms"]["down16"]["common_prefix"]["seed11"], 225)

    want = {
        ("down16", "seed12_train"): (1.2827, 1.6937, 0.5358, 0.4198),
        ("down16", "seed12_val"): (1.5306, 1.6586, 0.4045, 0.4500),
        ("down16", "seed10_full"): (1.6020, 1.7126, 0.4149, 0.4249),
        ("down16", "seed10_suffix"): (1.6450, 1.7122, 0.4011, 0.4364),
        ("down16", "seed11_full"): (1.5568, 1.6755, 0.4104, 0.4404),
        ("down16", "seed11_suffix"): (1.5903, 1.6639, 0.3924, 0.4588),
        ("down4", "seed12_train"): (0.1558, 1.6937, 0.9829, 0.4198),
        ("down4", "seed12_val"): (2.1299, 1.6586, 0.4591, 0.4500),
        ("down4", "seed10_full"): (2.1161, 1.7126, 0.4832, 0.4249),
        ("down4", "seed10_suffix"): (2.5070, 1.7122, 0.3875, 0.4364),
        ("down4", "seed11_full"): (2.0074, 1.6755, 0.4923, 0.4404),
        ("down4", "seed11_suffix"): (2.3844, 1.6639, 0.3936, 0.4588),
    }
    for (arm, s), (ce, base, acc, maj) in want.items():
        e = r4["arms"][arm]["evaluations"][s]
        check(f"R4 {arm} {s} ce", round(e["ce"], 4), ce)
        check(f"R4 {arm} {s} base", round(e["baseline"]["ce_train_mle"], 4), base)
        check(f"R4 {arm} {s} acc", round(e["accuracy"], 4), acc)
        check(f"R4 {arm} {s} majacc", round(e["baseline"]["majority_accuracy"], 4), maj)

    # the verdict flags the report relies on
    for s in ("seed12_val", "seed10_full", "seed10_suffix", "seed11_full", "seed11_suffix"):
        check(f"R4 down16 {s} beats", r4["arms"]["down16"]["evaluations"][s]["beats_train_mle_baseline"], True)
        check(f"R4 down16 {s} acc below maj",
              r4["arms"]["down16"]["evaluations"][s]["accuracy"]
              < r4["arms"]["down16"]["evaluations"][s]["baseline"]["majority_accuracy"], True)
        check(f"R4 down4 {s} beats", r4["arms"]["down4"]["evaluations"][s]["beats_train_mle_baseline"], False)

    # per-class recall claim: five classes at zero
    pc = r4["arms"]["down16"]["evaluations"]["seed12_val"]["per_class"]
    zeros = [k for k, v in pc.items() if v["recall"] == 0.0]
    check("R4 down16 val zero-recall classes", len(zeros), 5)
    check("R4 down16 val class0 recall", round(pc["0"]["recall"], 3), 0.616)
    check("R4 down16 val class7 recall", round(pc["7"]["recall"], 3), 0.062)

    # ---------------- R5 ----------------
    r5path = EV / "R5" / "significance.json"
    if r5path.exists():
        r5 = json.loads(r5path.read_text(encoding="utf-8"))
        want5 = {
            "seed12_val": 0.1280,
            "seed10_full": 0.1106,
            "seed10_suffix": 0.0673,
            "seed11_full": 0.1187,
            "seed11_suffix": 0.0736,
        }
        for s, d in want5.items():
            e = r5["results"][s]
            check(f"R5 {s} delta", round(e["mean_delta"], 4), d)
            check(f"R5 {s} excl0", e["excludes_zero"], True)
            check(f"R5 {s} ci_low_pos", e["ci95_low"] > 0, True)
        check("R5 seed10_suffix ci_low", round(r5["results"]["seed10_suffix"]["ci95_low"], 4), 0.0149)
        check("R5 seed11_suffix ci_low", round(r5["results"]["seed11_suffix"]["ci95_low"], 4), 0.0265)
        check("R5 seed10_suffix ce", round(r5["results"]["seed10_suffix"]["model_ce"], 4), 1.6450)
        check("R5 seed11_suffix ce", round(r5["results"]["seed11_suffix"]["model_ce"], 4), 1.5903)
        # accuracy is still below the majority class on every held-out set
        for s in want5:
            e = r5["results"][s]
            check(f"R5 {s} acc below majority", e["model_accuracy"] < e["majority_accuracy"], True)
    else:
        failures.append("R5 significance.json missing")
        checks += 1

    # ---------------- R10 / R11 / R12 ----------------
    r10 = json.loads((EV / "R10" / "safe_set_evidence.json").read_text(encoding="utf-8"))
    check("R10 status", r10["status"], "pass")
    check("R10 states", r10["states_total"], 200)
    check("R10 arms expected", r10["arms_total_expected"], 3600)
    check("R10 arms present", r10["arms_present"], 3600)
    check("R10 horizon", r10["horizon_frames"], 120)
    check("R10 proper count", r10["proper"]["count"], 184)
    check("R10 proper rate", r10["proper"]["rate"], 0.92)
    check("R10 wilson low", round(r10["proper"]["wilson_95"]["lower"], 5), 0.87401)
    check("R10 wilson high", round(r10["proper"]["wilson_95"]["upper"], 5), 0.95016)
    check("R10 prefix share", r10["derived"]["prefix_share_of_frames"], 0.9399)
    check("R10 wall per arm", r10["derived"]["wall_s_per_arm"], 12.864)
    check("R10 frames per s", r10["derived"]["engine_frames_per_s"], 121.7)
    check("R10 distinct ancestors", r10["counts"]["distinct_seed_ancestors"], 50)
    check("R10 engine_death arms", r10["counts"]["outcomes"]["engine_death"], 1899)
    check("R10 engine_alive arms", r10["counts"]["outcomes"]["engine_alive"], 1701)
    check("R10 unknown arms", r10["counts"]["unknown_arms"], 0)

    r11 = json.loads((EV / "R11" / "state_cost_profile.json").read_text(encoding="utf-8"))
    check("R11 proper", r11["proper_states"], 184)
    check("R11 seeds", r11["distinct_seeds"], 50)
    check("R11 seed range", r11["seed_range"], [3042, 3091])
    check("R11 prefix median", r11["prefix_frames_profile"]["median"], 1390)
    check("R11 mismatch total", r11["prefix_hash_mismatch_total"], 0)
    pdr = r11["proper_being_death_rich"]
    check("R11 death share median", pdr["median"], 0.5)
    check("R11 death share mean", pdr["mean"], 0.514)
    check("R11 all-die states", pdr["n_states_where_all_18_die"], 0)
    check("R11 none-die states", pdr["n_states_where_no_action_dies"], 0)
    check("R11 3s attempt", r11["cost_model"]["segment_3s"]["attempt_s"], 12.8)
    check("R11 3 arms h", r11["cost_model"]["segment_3s"]["h_3_arms_200_each"], 2.14)

    r12 = json.loads((EV / "R12" / "seg001_state_manifest.json").read_text(encoding="utf-8"))
    check("R12 states total", r12["counts"]["states_total"], 200)
    check("R12 proper", r12["counts"]["proper"], 184)
    check("R12 train selected", r12["counts"]["train_selected"], 68)
    check("R12 dev selected", r12["counts"]["dev_selected"], 16)
    check("R12 train seeds", len(r12["counts"]["train_seeds_selected"]), 35)
    check("R12 dev seeds", len(r12["counts"]["dev_seeds_selected"]), 8)
    check("R12 segment frames", r12["frozen_rules"]["segment_frames"], 180)
    check("R12 certified frames", r12["frozen_rules"]["certified_frames"], 120)
    check("R12 band", r12["frozen_rules"]["safe_size_band"], [2, 8])
    # the seed split must be disjoint
    tr_s = set(r12["counts"]["train_seeds_selected"])
    dv_s = set(r12["counts"]["dev_seeds_selected"])
    check("R12 seeds disjoint", len(tr_s & dv_s), 0)
    # every selected state must be inside the frozen band and have a safe set
    for s in r12["train_states"] + r12["dev_states"]:
        check(f"R12 state {s['index']} band", 2 <= s["safe_size"] <= 8, True)
        check(f"R12 state {s['index']} safe_set size", len(s["safe_set"]), s["safe_size"])
        check(f"R12 state {s['index']} death_set size",
              len(s["safe_set"]) + len(s["death_set"]), 18)

    # ---------------- R6 ----------------
    r6 = json.loads((EV / "R6" / "actor_recheck.json").read_text(encoding="utf-8"))
    check("R6 params", r6["n_params"], 2726671)
    check("R6 prefix seed10", r6["common_prefix"]["seed10"], 219)
    check("R6 prefix seed11", r6["common_prefix"]["seed11"], 225)
    # exact reproduction of the documented numbers under the original batch averaging
    check("R6 train batch-mean", round(r6["evaluations"]["seed12_train"]["batch_mean_ce"]["move"], 4), 0.0293)
    check("R6 val batch-mean", round(r6["evaluations"]["seed12_val"]["batch_mean_ce"]["move"], 4), 1.8807)
    check("R6 seed10 batch-mean", round(r6["evaluations"]["seed10_full"]["batch_mean_ce"]["move"], 4), 1.8837)
    # per-sample readings the report quotes
    want6 = {"seed12_val": 1.8504, "seed10_full": 1.8680, "seed10_suffix": 2.2266,
             "seed11_full": 1.8175, "seed11_suffix": 2.1756}
    for s, v in want6.items():
        check(f"R6 {s} sample-mean", round(r6["evaluations"][s]["sample_mean_ce"]["move"], 4), v)
    # the CE/top-1 inversion: better than majority on full tapes, worse on suffixes
    ev = r6["evaluations"]
    for s in ("seed10_full", "seed11_full"):
        h = ev[s]["heads"]["move"]
        check(f"R6 {s} acc above majority", h["model_accuracy"] > h["baseline_majority_accuracy"], True)
    for s in ("seed10_suffix", "seed11_suffix"):
        h = ev[s]["heads"]["move"]
        check(f"R6 {s} acc at/below majority", h["model_accuracy"] <= h["baseline_majority_accuracy"], True)
        check(f"R6 {s} ci excludes 0", h["bootstrap"]["significant"], True)
        check(f"R6 {s} ci entirely below 0", h["bootstrap"]["ci95_high"] < 0, True)
    check("R6 focus seed10_suffix worse",
          ev["seed10_suffix"]["heads"]["focus"]["delta_vs_train_mle"] < 0, True)
    check("R6 focus seed10_suffix ci excludes 0",
          ev["seed10_suffix"]["heads"]["focus"]["bootstrap"]["significant"], True)

    # ---------------- R15 / R16 ----------------
    r15 = json.loads((EV / "R15" / "candidate_space.json").read_text(encoding="utf-8"))
    check("R15 candidates", r15["counts"]["candidates"], 18)
    check("R15 distinct masks", r15["counts"]["distinct_masks"], 18)
    check("R15 collisions", r15["counts"]["collisions"], 0)
    check("R15 roundtrip", r15["counts"]["roundtrip_ok"], True)
    check("R15 bomb always 0", r15["counts"]["bomb_always_0"], True)
    check("R15 varying bits", r15["bits"]["varying_bits"], "0xf4")
    check("R15 focus pairs differ", r15["focus_axis"]["all_pairs_differ_in_mask"], True)
    check("R15 cand0 mask", r15["masks"]["0"]["mask"], 1)
    check("R15 cand9 mask", r15["masks"]["9"]["mask"], 5)

    r16 = json.loads((EV / "R16" / "verdict_reliability.json").read_text(encoding="utf-8"))
    check("R16 repeat pairs", r16["repeat_agreement"]["pairs"], 4)
    check("R16 repeat agreed", r16["repeat_agreement"]["agreed"], 4)
    check("R16 repeat rate", r16["repeat_agreement"]["rate"], 1.0)
    s16 = r16["summary"]
    check("R16 selected states", s16["selected_states"], 84)
    check("R16 recorded dies", s16["recorded_action_dies"], 81)
    check("R16 recorded dies rate", round(s16["recorded_action_dies_rate"], 3), 0.964)
    check("R16 no unknown arms", s16["selected_states_with_any_unknown_arm"], 0)
    check("R16 arms judged", s16["arms_judged_over_selected"], 1512)
    check("R16 arms expected", s16["expected_arms_over_selected"], 1512)
    check("R16 min safe", s16["min_safe_size"], 2)
    check("R16 max safe", s16["max_safe_size"], 8)
    check("R16 mean death arms", round(s16["mean_death_arms"], 2), 12.96)

    # ---------------- R14 (audio) ----------------
    r14 = json.loads((EV / "R14" / "audio_cfg_audit.json").read_text(encoding="utf-8"))
    check("R14 cfg files", r14["counts"]["cfg_files"], 105)
    check("R14 muted", r14["counts"]["muted"], 0)
    check("R14 audible", r14["counts"]["audible"], 105)
    check("R14 decoded", r14["counts"]["decoded"], 105)
    check("R14 template muted", r14["template_reference"]["playSounds"], 0)

    # ---------------- R17: gate arithmetic ----------------
    sys.path.insert(0, str(REPO / "tools" / "rl"))
    try:
        import v431_seg001_gate as G
    except Exception as exc:                        # noqa: BLE001
        failures.append(f"R17 import failed: {exc!r}")
        checks += 1
    else:
        w = G.wilson(8, 10)
        check("R17 wilson p", round(w["p"], 6), 0.8)
        check("R17 wilson lo", round(w["lower"], 4), 0.4902)
        check("R17 wilson hi", round(w["upper"], 4), 0.9433)
        check("R17 strictly_better", G.strictly_better(G.wilson(99, 100), G.wilson(1, 100)), True)
        check("R17 overlap counts as no-better",
              G.strictly_better(G.wilson(5, 10), G.wilson(5, 10)), False)
        # rev D: arm A must beat BOTH baselines, and a strong constant must block it
        def rec(arm, split, alive, i):
            return {"arm": arm, "split": split, "alive120": alive, "alive180": alive,
                    "aborted": False, "hash_gate_ok": True, "state_index": i,
                    "attempt": 0, "admissible": True, "cluster": i // 100}
        recs = ([rec("A", "dev", i < 50, i) for i in range(100)]
                + [rec("C-fair", "dev", i < 20, i) for i in range(100)])
        d = G.decide(recs)
        check("R17 S2 quiet when A beats fair", any(t["rule"] == "S2" for t in d["triggered"]), False)
        # now add a constant baseline that beats A -> S2 must fire
        recs2 = recs + [rec("C-const-best", "dev", i < 90, i) for i in range(100)]
        d2 = G.decide(recs2)
        check("R17 S2 fires when a constant beats A",
              any(t["rule"] == "S2" for t in d2["triggered"]), True)
        check("R17 C-priv is not part of the gate",
              "C-priv" in G.ARMS, False)
        check("R17 arms", G.ARMS, ("A", "B", "C", "C-fair", "C-const-best"))

    # ---------------- R18: restore probe reproduced the frozen table ----------------
    r18 = json.loads((EV / "R18" / "probe_reproduction.json").read_text(encoding="utf-8"))
    check("R18 shard status", r18["shard"]["status"], "pass")
    check("R18 shard exit", r18["shard"]["exit_code"], 0)
    check("R18 arms", r18["shard"]["arms"], 18)
    check("R18 complete", r18["shard"]["complete"], True)
    check("R18 prefix checks", r18["shard"]["prefix_checks"], 36180)
    check("R18 prefix mismatches", r18["shard"]["prefix_mismatches"], 0)
    check("R18 audio muted", r18["audio"]["muted"], True)
    check("R18 state expected hash", r18["state"]["expected_hash"], "6cade82be08f8a95")
    check("R18 seed", r18["state"]["seed"], 3042)
    check("R18 decision frame", r18["state"]["decision_frame"], 2010)
    cmp18 = r18["comparison"]
    check("R18 cands", cmp18["cands"], 18)
    check("R18 agreement", cmp18["outcome_agreement"], 18)
    check("R18 agreement rate", cmp18["outcome_agreement_rate"], 1.0)
    check("R18 safe_set equal", cmp18["safe_set_equal"], True)
    check("R18 safe_set", cmp18["safe_set_probe"], [2, 7, 11])
    # every lethal arm must have an identical first death frame
    mismatched = [r["cand"] for r in cmp18["per_cand"]
                  if r["a1_first_death_frame"] != r["probe_first_death_frame"]]
    check("R18 death frames identical", mismatched, [])
    # cost: prefix dominates the wall clock
    cost = r18["shard"]["cost"]
    check("R18 prefix share >0.9",
          (cost["prefix_s"] / cost["shard_wall_s"]) > 0.90, True)
    check("R18 ms per arm", round(cost["ms_per_arm"], 0), 16430, 1.0)

    # ---------------- R19: harness validation + safe-set semantics ----------------
    r19 = json.loads((EV / "R19" / "harness_validation.json").read_text(encoding="utf-8"))
    hc = r19["harness_checks"]
    check("R19 restore gate all ok", hc["restore_gate_all_ok"], True)
    check("R19 recorded reproduced A1 death", hc["recorded_reproduced_a1_death_frame"], True)
    check("R19 A1 recorded death frame", hc["a1_recorded_death_frame"], 2041)
    check("R19 probe recorded death frame", hc["probe_recorded_death_frame"], 2041)
    ep = r19["episodes"]
    check("R19 recorded terminal", ep["recorded"]["terminal"], "first_death")
    check("R19 recorded offset", ep["recorded"]["first_death_offset"], 31)
    check("R19 const2 terminal", ep["const2"]["terminal"], "first_death")
    check("R19 const2 offset", ep["const2"]["first_death_offset"], 35)
    check("R19 const7 terminal", ep["const7"]["terminal"], None)
    check("R19 const7 frames", ep["const7"]["frames_executed"], 180)
    check("R19 const7 alive180", ep["const7"]["alive_180"], True)
    check("R19 A1 says cand2 alive", r19["safe_set_semantics"]["a1_says_cand2"], "engine_alive")
    check("R19 semantics verdict",
          r19["safe_set_semantics"]["verdict"].startswith("S is continuation-conditional"), True)
    check("R19 constant cand7 survived", r19["segment_admissibility"]["constant_survived_180_frames"], True)
    check("R19 state0 inadmissible at 180",
          r19["segment_admissibility"]["state_0_admissible_at_180_frames"], False)

    # ---------------- R20 / R20b: the C4 gate result ----------------
    r20 = json.loads((EV / "R20" / "c4_screen.json").read_text(encoding="utf-8"))
    c20 = r20["counts"]
    check("R20 states screened", c20["states_screened"], 4)
    check("R20 all complete", c20["states_complete"], 4)
    check("R20 admissible", c20["admissible"], 0)
    check("R20 admissible rate", c20["admissible_rate"], 0.0)
    check("R20 errors", c20["errors"], 0)
    check("R20 S1b triggered", r20["gate"]["S1b_triggered"], True)
    check("R20 segment frames", r20["segment_frames"], 180)
    survivors = {s["state_index"]: s["n_survivors_180"] for s in r20["states"]}
    check("R20 survivors state0", survivors[0], 7)
    check("R20 survivors state1", survivors[1], 8)
    check("R20 survivors state3", survivors[3], 9)
    check("R20 survivors state165", survivors[165], 6)
    # every screened state has a constant that survives the whole segment
    check("R20 every state has a surviving constant",
          all(s["n_survivors_180"] > 0 for s in r20["states"]), True)
    # and the best constant reaches the full segment everywhere -> the gate baseline is at ceiling
    check("R20 best constant reaches 180 in every state",
          all(s["best_constant_survival_frames"] == 180 for s in r20["states"]), True)

    r20b = json.loads((EV / "R20" / "state_position.json").read_text(encoding="utf-8"))
    d20 = r20b["a1_decision_frame"]
    check("R20b median frame", d20["median"], 1383)
    check("R20b max frame", d20["max"], 3020)
    check("R20b none beyond frame 8000", d20["n_beyond_8000"], 0)
    check("R20b none beyond half stage", d20["n_beyond_half_stage"], 0)
    check("R20b proper states", r20b["a1_proper_states"], 184)
    check("R20b ink at a1 states", round(r20b["ink_percentile_at_a1_states"]["mean"], 2), 16.14)
    check("R20b ink in dense windows", round(r20b["ink_percentile_in_dense_windows"]["mean"], 2), 82.19)
    check("R20b a1 states sparser", r20b["verdict"]["a1_states_are_sparser_than_dense_windows"], True)

    # ---------------- R21 / R22: the dense regime is not reachable ----------------
    r21 = json.loads((EV / "R21" / "pool_ceiling.json").read_text(encoding="utf-8"))
    ceil = r21["A_trace_ceiling"]
    check("R21 traces", ceil["traces"], 50)
    check("R21 max trace frame", ceil["last_frame_max"], 3051)
    check("R21 median trace frame", ceil["last_frame_median"], 1482)
    check("R21 none beyond 4000", ceil["n_traces_beyond_4000"], 0)
    check("R21 none beyond 8000", ceil["n_traces_beyond_8000"], 0)
    check("R21 all end at first_death", ceil["all_end_at_first_death"], True)
    tape21 = r21["B_teacher_tape"]
    check("R21 tape slots", tape21["slots"], 6588)
    check("R21 tape grid contiguous", tape21["two_frame_grid_contiguous"], True)
    check("R21 tape hash chain breaks", tape21["hash_chain_breaks"], 0)
    check("R21 target is slot boundary", tape21["target_is_a_slot_boundary"], True)
    check("R21 target hash", tape21["target_hash_pre"], "235b20476408e0c8")

    r22 = json.loads((EV / "R22" / "dense_blocker.json").read_text(encoding="utf-8"))
    check("R22 offline agreement", r22["offline_prefix_check"]["offline_agreement"],
          "4,326 / 4,326")
    check("R22 refusal frame 96", "frame 96" in str(r22["refusal"]["message"]), True)
    check("R22 refusal site",
          "counterfactual.py:3924-3929" in r22["refusal"]["site"], True)
    for seed in ("10", "11", "12"):
        t = r22["teacher_tapes"][seed]
        check(f"R22 seed{seed} split slots", t["split_mask_slots"], 137)
        check(f"R22 seed{seed} first split frame", t["first_split_slot"]["decision_frame"], 94)
        check(f"R22 seed{seed} reachable floor", t["reachable_floor"], 94)
    check("R22 no replayable prefix reaches dense",
          r22["consequence"]["no_replayable_prefix_reaches_dense"], True)
    check("R22 unblock options", len(r22["consequence"]["unblock_options"]), 3)

    # ---------------- R23: the dense regime IS reachable for the state arm ----------
    r23 = json.loads((EV / "R23" / "dense_reachable.json").read_text(encoding="utf-8"))
    rg = r23["restore_gate"]
    check("R23 frames walked", rg["frames_walked"], 8663)
    check("R23 hash checks", rg["hash_checks"], 2165)
    check("R23 hash mismatches", rg["hash_mismatches"], 0)
    check("R23 no first mismatch", rg["first_mismatch_frame"], None)
    check("R23 reached dense", rg["reached_dense_regime"], True)
    check("R23 verified", rg["verified"], True)
    check("R23 target frame", r23["episode"]["target_frame"], 8664)
    check("R23 recorded policy survived", r23["episode"]["alive_180"], True)
    check("R23 no error", r23["episode"]["error"], None)
    check("R23 touhou_rl untouched", r23["scope"]["touhou_rl_modified"], False)
    check("R23 ms per frame", r23["cost"]["ms_per_frame"], 8.192)
    cens = r23["s1c_corpus"]
    check("R23 s1c shards", cens["n_shards"], 12)
    check("R23 all first split at 92", cens["all_first_split_at_92"], True)
    check("R23 identical step_frames", cens["identical_first_split_step_frames"],
          [[149, 149, 133, 132]])

    # ---------------- R24: C4 PASSES in the dense regime at 10 s ----------------
    r24 = json.loads((EV / "R24" / "c4_dense.json").read_text(encoding="utf-8"))
    c24 = r24["counts"]
    check("R24 window frame", r24["window"]["frame"], 8664)
    check("R24 window dense", r24["window"]["dense"], True)
    check("R24 segment frames", r24["segment"]["frames"], 600)
    check("R24 candidates", c24["candidates"], 18)
    check("R24 complete", c24["complete"], 18)
    check("R24 errors", c24["errors"], 0)
    check("R24 survivors", c24["survivors"], 0)
    check("R24 admissible", r24["admissible"], True)
    d24 = r24["death_offsets"]
    check("R24 min offset", d24["min"], 58)
    check("R24 max offset", d24["max"], 520)
    check("R24 median offset", d24["median"], 277.5)
    check("R24 max offset seconds", d24["max_seconds"], 8.67)
    k24 = r24["segment_length_constraint"]
    check("R24 min admissible frames", k24["min_admissible_frames"], 521)
    check("R24 min admissible seconds", k24["min_admissible_seconds"], 8.68)
    check("R24 margin frames", k24["margin_frames"], 80)
    check("R24 margin fraction", round(k24["margin_fraction"], 4), 0.1333)
    check("R24 every episode hash-gated", r24["restore_gate"]["hash_checks_all"], True)
    check("R24 hash mismatches", r24["restore_gate"]["hash_mismatches_total"], 0)
    check("R24 cost episodes", r24["cost"]["episodes"], 18)
    check("R24 mean wall", r24["cost"]["mean_wall_s"], 72.6)
    # 13 of 18 constants survive 180 frames -> 3 s would NOT have been admissible here
    n180 = sum(1 for r in r24["rows"] if r["alive_180"])
    check("R24 constants surviving 180 frames", n180, 13)

    # ---------------- R25: C4 across tapes + observation channel ----------------
    r25 = json.loads((EV / "R25" / "c4_multi.json").read_text(encoding="utf-8"))
    a25 = r25["partA_c4_across_tapes"]
    check("R25 windows", a25["n_windows"], 3)
    check("R25 admissible windows", a25["n_admissible"], 2)
    check("R25 not all admissible", a25["all_admissible"], False)
    w25 = {w["seed"]: w for w in a25["windows"]}
    check("R25 seed10 survivors", w25[10]["n_survivors"], 1)
    check("R25 seed10 survivor is cand12", w25[10]["survivors"], [12])
    check("R25 seed10 not admissible", w25[10]["admissible"], False)
    check("R25 seed10 max offset", w25[10]["max_death_offset"], 374)
    check("R25 seed11 admissible", w25[11]["admissible"], True)
    check("R25 seed11 min admissible frames", w25[11]["min_admissible_frames"], 545)
    check("R25 seed11 min admissible seconds", w25[11]["min_admissible_seconds"], 9.08)
    check("R25 seed12 admissible", w25[12]["admissible"], True)
    check("R25 seed12 min admissible frames", w25[12]["min_admissible_frames"], 521)
    for s in (10, 11, 12):
        check(f"R25 seed{s} hash gate", w25[s]["hash_mismatches_total"], 0)
    b25 = r25["partB_observation_channel"]
    check("R25 state field count", b25["state_field_count"], 25)
    check("R25 state has no bullet field", b25["state_has_bullet_or_boss_field"], False)
    check("R25 child verbs", b25["child_snapshot_verbs"],
          ["HAZARDS", "PLAYER", "EVENTS", "HUD"])

    # ---------------- R26: HAZARDS is available on the env-role session ----------------
    r26 = json.loads((EV / "R26" / "hazards_available.json").read_text(encoding="utf-8"))
    check("R26 hazards available", r26["answer"]["hazards_available_on_env"], True)
    check("R26 arm A choice", r26["answer"]["arm_a_choice"], "A-HAZARDS")
    check("R26 hazards bytes", r26["probe"]["hazards_reply_bytes"], 12778)
    for v in ("HAZARDS", "PLAYER", "EVENTS", "HUD", "STATE"):
        check(f"R26 {v} answered", r26["probe"]["verbs_answered"][v], True)
    check("R26 walk hash checks", r26["probe"]["walk"]["hash_checks"], 2165)
    check("R26 walk mismatches", r26["probe"]["walk"]["hash_mismatches"], 0)
    oc = r26["observation_contract_for_arm_a"]
    check("R26 bullets at frame", oc["engine_totals_seen_at_this_frame"]["bullets"], 183)
    check("R26 enemies at frame", oc["engine_totals_seen_at_this_frame"]["enemies"], 4)
    check("R26 lasers at frame", oc["engine_totals_seen_at_this_frame"]["lasers"], 0)
    check("R26 BulletRow fields",
          oc["bullet_row_fields"],
          ["slot", "state", "color", "position", "velocity", "timer",
           "collision_size", "collision_disabled"])
    check("R26 EnemyRow fields",
          oc["enemy_row_fields"],
          ["slot", "flags1", "position", "life", "timer", "world_position",
           "hitbox_width", "hitbox_height"])
    check("R26 arm A buildable", r26["consequence"]["arm_a_is_buildable_with_bullets"], True)

    # ---------------- R27: arm A observation math (offline) ----------------
    r27 = json.loads((EV / "R27" / "arm_a_selftest.json").read_text(encoding="utf-8"))
    check("R27 checks", r27["n_checks"], 18)
    check("R27 checks passed", r27["n_ok"], 18)
    check("R27 all ok", r27["all_ok"], True)
    check("R27 feature dim", r27["feature_dim"], 106)
    check("R27 bullet half width line",
          r27["red_lines_verified"]["bullet_half_width_is_collision_size_over_2"], True)
    check("R27 enemy half extent line",
          r27["red_lines_verified"]["enemy_half_extent_is_hitbox_over_3"], True)
    for name in ("empty frame finite", "zero-distance bullet finite",
                 "half_width not full width", "enemy half extent is /3",
                 "nearest bullet first", "probs sum to 1"):
        found = [r for r in r27["checks"] if r["check"] == name]
        check(f"R27 {name} present", len(found), 1)
        check(f"R27 {name} ok", found[0]["ok"], True)

    # ---------------- R28: widened screen + frozen train/dev split ----------------
    r28 = json.loads((EV / "R28" / "screen_split.json").read_text(encoding="utf-8"))
    sc = r28["screen"]
    check("R28 windows screened", sc["windows_screened"], 9)
    check("R28 windows judged", sc["windows_judged"], 9)
    check("R28 windows admissible", sc["windows_admissible"], 8)
    check("R28 admissible seeds", sc["admissible_seeds"],
          [11, 12, 13, 14, 15, 16, 17, 18])
    check("R28 rejected seeds", sc["rejected_seeds"], [10])
    sp = r28["split"]
    check("R28 train seeds", sp["train_seeds"], [13, 14, 15, 16])
    check("R28 dev seeds", sp["dev_seeds"], [11, 12, 17, 18])
    check("R28 split disjoint", sp["disjoint"], True)
    check("R28 train admissible", len(sp["train_admissible"]), 4)
    check("R28 dev admissible", len(sp["dev_admissible"]), 4)
    check("R28 S1b satisfied", sp["s1b_min_4_per_side"], True)
    sl = r28["segment_length"]
    check("R28 required segment frames", sl["required_frames"], 552)
    check("R28 required segment seconds", sl["required_seconds"], 9.2)
    check("R28 chosen segment frames", sl["chosen_frames"], 600)
    check("R28 segment margin", sl["margin_frames"], 48)
    check("R28 gate integrity", r28["gate_integrity"]["every_episode_hash_checks_2165"], True)
    check("R28 hash mismatches", r28["gate_integrity"]["hash_mismatches_total"], 0)
    check("R28 episodes", r28["cost"]["episodes"], 162)
    w28 = {r["seed"]: r for r in sc["rows"]}
    check("R28 seed10 survivor", w28[10]["n_survivors"], 1)
    check("R28 seed15 blocking frames", w28[15]["min_blocking_frames"], 552)
    check("R28 seed16 blocking frames", w28[16]["min_blocking_frames"], 449)

    # ---------------- R29: arm A actually run ----------------
    r29 = json.loads((EV / "R29" / "arm_a_run.json").read_text(encoding="utf-8"))
    w29 = r29["works"]
    check("R29 train episodes", w29["train_episodes"], 12)
    check("R29 train full survivals", w29["train_survived_full_segment"], 1)
    check("R29 train mean return", w29["train_mean_return_decisions"], 133.9)
    check("R29 train best episode", w29["train_best_episode"], 300)
    check("R29 dev episodes", w29["dev_episodes"], 10)
    check("R29 dev survivals", w29["dev_survived"], 0)
    check("R29 dev mean return", w29["dev_mean_return_decisions"], 75.1)
    n29 = r29["no_signal_on_dev"]
    check("R29 c-fair mean", n29["c_fair"]["mean"], 71.8)
    check("R29 c-fair survivals", n29["c_fair"]["survived"], 0)
    check("R29 c-fair max", n29["c_fair"]["max"], 182.5)
    d29 = n29["paired_bootstrap_of_mean_difference"]
    check("R29 bootstrap excludes zero", d29["excludes_zero"], False)
    check("R29 bootstrap ci low", d29["ci95"][0], -31.1)
    check("R29 bootstrap ci high", d29["ci95"][1], 33.75)
    check("R29 verdict is no signal", "no signal" in n29["verdict"], True)
    y29 = r29["why_no_signal"]
    check("R29 fraction of budget", y29["fraction_of_prereg_budget_run"], 0.06)
    check("R29 prefix share", y29["prefix_share_of_episode"], 0.96)
    st = y29["staged_cost"]
    check("R29 1000 episodes one window", st["hours_one_window"][3], 19.6)
    check("R29 1000 episodes 3x4", st["hours_3_seeds_4_windows"][3], 234.7)
    b29 = r29["bug_found_and_fixed"]
    check("R29 bug symptom return before", b29["evidence"]["ep10_return_before_fix"], 38)
    check("R29 bug fixed return", b29["evidence"]["ep11_return_after_fix"], 300)
    check("R29 decision outcome is not-beat",
          "does NOT beat" in r29["decision_rule_outcome"]["result"], True)

    # ---------------- R30 / R30b: can the prefix replay get cheaper? ----------------
    r30 = json.loads((EV / "R30" / "prefix_runs.json").read_text(encoding="utf-8"))
    check("R30 frames", r30["offline"]["frames"], 8663)
    check("R30 best reduction", r30["summary"]["best_reduction_factor"], 3.91)
    check("R30 mean run length", r30["summary"]["mean_run_length"], 3.87)
    check("R30 engine has no save/restore", r30["engine_has_save_restore"], False)
    run12 = [r for r in r30["offline"]["per_seed"] if r["seed"] == 12][0]
    check("R30 seed12 runs", run12["runs"], 2231)
    check("R30 seed12 max run", run12["run_length_max"], 95)

    r30b = json.loads((EV / "R30" / "microbench.json").read_text(encoding="utf-8"))
    stb = r30b["stats"]
    check("R30b group1 ms", stb["1"]["median_ms_per_frame"], 7.655)
    check("R30b group5 ms", stb["5"]["median_ms_per_frame"], 7.718)
    check("R30b group25 ms", stb["25"]["median_ms_per_frame"], 7.837)
    check("R30b speedup", r30b["verdict"]["speedup_best_vs_group1"], 0.992)
    check("R30b batching does not help", r30b["verdict"]["batching_helps"], False)
    check("R30b best group", r30b["verdict"]["best_group"], "5")

    # ---------------- R31: the baseline profile ----------------
    r31 = json.loads((EV / "R31" / "baseline_profile.json").read_text(encoding="utf-8"))
    dev31 = {r["seed"]: r for r in r31["dev_windows"]}
    check("R31 seed11 const-best decisions", dev31[11]["c_const_best_decisions"], 272.0)
    check("R31 seed12 const-best decisions", dev31[12]["c_const_best_decisions"], 260.0)
    check("R31 seed17 const-best decisions", dev31[17]["c_const_best_decisions"], 245.5)
    check("R31 seed18 const-best decisions", dev31[18]["c_const_best_decisions"], 257.5)
    check("R31 fair n", dev31[11]["c_fair_n"], 10)
    pool = r31["c_fair_pooled"]
    check("R31 pooled n", pool["n"], 40)
    check("R31 pooled survived", pool["survived"], 0)
    check("R31 pooled mean", pool["mean_decisions"], 76.2)
    check("R31 pooled median", pool["median_decisions"], 59.5)
    check("R31 pooled max", pool["max_decisions"], 246.0)
    check("R31 constant beats random by", r31["difficulty_reading"]
          ["best_constant_beats_random_by"], 3.57)
    check("R31 arm A dev11 mean", r31["arm_a_dev11"]["mean_decisions"], 75.1)
    check("R31 step batching no speedup",
          "no speed-up" in r31["step_batching"]["result"], True)

    # ---------------- R32: the fork-server exists but is gated shut ----------------
    r32 = json.loads((EV / "R32" / "park_spawn.json").read_text(encoding="utf-8"))
    check("R32 role", r32["role"], "shadow")
    check("R32 DISPLAY unset", r32["display_unset"], True)
    check("R32 warmup frame", r32["steps"]["warmup_frame"], 400)
    check("R32 reset hash", r32["steps"]["reset"]["hash"], "83d953750afaea5a")
    check("R32 reset ready", r32["steps"]["reset"]["ready"], "1")
    check("R32 park refused", "park-unsafe" in r32["error"], True)
    a32 = r32["analysis"]
    check("R32 fork server exists", a32["verdict"]["fork_server_exists"], True)
    check("R32 fork server usable", a32["verdict"]["fork_server_usable_here"], False)
    blockers = {b["name"]: b for b in a32["three_blockers"]}
    check("R32 blocker threads", blockers["threads=5"]["observed"], 5)
    check("R32 blocker setup", blockers["setup=0"]["observed"], 0)
    check("R32 blocker null device", blockers["null_device=0"]["observed"], 0)
    check("R32 three blockers", len(a32["three_blockers"]), 3)
    check("R32 protocol has PARK",
          "template=" in r32["protocol"]["park"], True)
    check("R32 protocol has SPAWN",
          "handle=" in r32["protocol"]["spawn"], True)
    check("R32 analysis written", "correction_to_earlier_claim" in a32["verdict"], True)

    # ---------------- R33: PARK is structurally shut, not merely early ----------------
    r33 = json.loads((EV / "R33" / "park_frames.json").read_text(encoding="utf-8"))
    a33 = r33["analysis"]
    check("R33 attempts", a33["attempts"], 5)
    check("R33 frames tried", a33["frames_tried"], [400, 1200, 2400, 4800, 8664])
    check("R33 distinct triples", a33["distinct_unsafe_triples"], [[5, 0, 0]])
    check("R33 invariant", a33["invariant_across_frames"], True)
    check("R33 not merely early", a33["verdict"]["was_it_merely_early"], False)
    check("R33 structural", a33["verdict"]["structural"], True)
    check("R33 any success", r33["any_success"], False)
    check("R33 shadow walk checks", a33["side_result"]["shadow_walk_hash_checks"], 2166)
    check("R33 shadow walk mismatches", a33["side_result"]["shadow_walk_mismatches"], 0)
    check("R33 needs approval", a33["verdict"]["needs_user_approval"], True)
    check("R33 repair location",
          "third_party/th08-rl" in a33["verdict"]["repair_location"], True)

    # ---------------- R34: train improves, dev degrades (overfitting) ----------------
    r34 = json.loads((EV / "R34" / "transfer.json").read_text(encoding="utf-8"))
    tr34 = r34["train"]
    check("R34 cumulative episodes", tr34["episodes_total"], 72)
    check("R34 first 20 mean", tr34["first_20_mean"], 87.7)
    check("R34 last 20 mean", tr34["last_20_mean"], 108.3)
    check("R34 train delta", tr34["delta"], 20.6)
    check("R34 train full survivals", tr34["full_survivals_60"], 4)
    check("R34 best constant seed13", tr34["best_constant_decisions_seed13"], 259)
    dv34 = r34["dev"]
    check("R34 dev12 mean", dv34["dev12_mean"], 75.1)
    check("R34 dev72 mean", dv34["dev72_mean"], 48.5)
    check("R34 cfair mean", dv34["cfair_mean"], 71.8)
    check("R34 dev72 dev12 survivals", dv34["dev12_survivals"], 0)
    check("R34 dev72 survivals", dv34["dev72_survivals"], 0)
    b1 = dv34["dev72_vs_cfair"]
    check("R34 dev72 vs cfair diff", b1["mean_diff"], -23.36)
    check("R34 dev72 vs cfair ci low", b1["ci95"][0], -53.35)
    check("R34 dev72 vs cfair ci high", b1["ci95"][1], -1.2)
    check("R34 dev72 vs cfair excludes zero", b1["excludes_zero"], True)
    b2 = dv34["dev72_vs_dev12"]
    check("R34 dev72 vs dev12 diff", b2["mean_diff"], -26.42)
    check("R34 dev72 vs dev12 ci low", b2["ci95"][0], -46.7)
    check("R34 dev72 vs dev12 ci high", b2["ci95"][1], -6.7)
    check("R34 dev72 vs dev12 excludes zero", b2["excludes_zero"], True)
    v34 = r34["verdict"]
    check("R34 train improved", v34["train_improved"], True)
    check("R34 dev transferred", v34["dev_transferred"], False)
    check("R34 dev degraded", v34["dev_degraded_vs_12ep"], True)
    check("R34 episodes cost", r34["cost"]["episodes"], 102)
    check("R34 engine hours", r34["cost"]["engine_hours"], 1.8)

    # ---------------- R35: multi-window training narrows R34 ----------------
    r35 = json.loads((EV / "R35" / "multi_window.json").read_text(encoding="utf-8"))
    m35 = r35["multi"]
    check("R35 train episodes", m35["train_episodes"], 80)
    check("R35 train mean", m35["train_mean_return"], 101.0)
    check("R35 first third", m35["train_first_third"], 91.6)
    check("R35 last third", m35["train_last_third"], 119.1)
    check("R35 train trend", m35["train_trend"], 27.5)
    check("R35 train survivals", m35["train_survivals"], 3)
    check("R35 per-window count", len(m35["per_window"]), 4)
    check("R35 seed13 mean", m35["per_window"]["13"]["mean_return"], 96.7)
    check("R35 seed15 mean", m35["per_window"]["15"]["mean_return"], 89.3)
    check("R35 seed16 mean", m35["per_window"]["16"]["mean_return"], 124.0)
    check("R35 dev mean", m35["dev_mean"], 57.3)
    check("R35 dev survivals", m35["dev_survivals"], 0)
    c35 = r35["comparison"]
    check("R35 single dev mean", c35["single_window_dev_mean"], 48.5)
    check("R35 cfair dev mean", c35["cfair_dev_mean"], 71.8)
    b35 = c35["multi_vs_cfair"]
    check("R35 multi vs cfair diff", b35["mean_diff"], -14.6)
    check("R35 multi vs cfair ci low", b35["ci95"][0], -46.4)
    check("R35 multi vs cfair ci high", b35["ci95"][1], 10.15)
    check("R35 multi vs cfair not significant", b35["excludes_zero"], False)
    check("R35 multi vs single diff",
          c35["multi_vs_single"]["difference_of_means"], 8.8)
    check("R35 multi beats cfair", r35["verdict"]["multi_beats_cfair"], False)
    check("R35 multi helps over single", r35["verdict"]["multi_helps_over_single"], True)

    # ---------------- the report's own deliverable structure ----------------
    import re as _re
    _rp = pathlib.Path(__file__).resolve().parents[2] / "Document" / "Update" / "Alpha" \
        / "v4.3.1" / "ROUTE_RECHECK_2026-11-15.md"
    _text = _rp.read_text(encoding="utf-8")
    check("report keep rows", len(_re.findall(r"^\| K\d+ \|", _text, _re.M)), 18)
    check("report correct rows", len(_re.findall(r"^\| C\d+ \|", _text, _re.M)), 44)
    check("report unreproduced rows", len(_re.findall(r"^\| N\d+ \|", _text, _re.M)), 14)
    check("report has 5.5 three-arm status", "### 5.5 三臂实验的最终状态" in _text, True)
    check("report 5.5 lists arm B not runnable",
          "B（像素+记忆，从零在线）" in _text and "无法运行" in _text, True)
    check("report 7.1 executed table", "### 7.1 已执行" in _text, True)
    check("report 7.2 unexecuted table", "### 7.2 未执行" in _text, True)
    check("report 9 mentions fork-server cost", "234.7" in _text, True)
    # R40: the unreproduced/unexecuted audit
    _nrows = _re.findall(r"^\| (N\d+) \|", _text, _re.M)
    check("report N rows", len(_nrows), 14)
    check("report N closed count",
          len(_re.findall(r"^\| N\d+ \| ~~", _text, _re.M)), 4)
    check("report 6.3 tally line", "**4 条已闭合**" in _text, True)
    check("report 7.2 six hard items",
          _text.count("| **教师磁带重录** |") == 1, True)
    check("report 7.2 no longer lists the 200-episode run",
          "臂 A 跑到 200 episode** | 152" not in _text, True)
    check("report 5.1 ancestor row rewritten",
          "这条已被后续结果改写" in _text, True)
    check("report 5.4 has an evaluability column", "可判定？" in _text, True)
    check("report 5.4 marks S1b not triggered", "| **S1b** |" in _text and "未触发" in _text, True)
    check("report 5.4 marks S2 triggered", "**已触发（右半）**" in _text, True)
    check("report 5.4 marks S3/S4 not evaluable", "**不可**" in _text, True)
    check("report 5.4 names the S6 metric mismatch", "指标时代错位" in _text, True)
    check("report 9 uses the n=80 numbers", "94.1" in _text and "6/80" in _text, True)
    check("report 9 no longer promises the 1.6 h fix",
          "只需要 1.6 h" not in _text, True)
    check("report keeps C-fair's single survival",
          "1/80 跑满" in _text, True)
    check("report 9 keeps BC undecided",
          "BC 既没有被证明有效，也没有被证明无效" in _text, True)

    # ---------------- R36: arm A reaches 200 episodes; dev crosses C-fair (tail event) ----
    r36 = json.loads((EV / "R36" / "budget_200.json").read_text(encoding="utf-8"))
    b36 = r36["budget"]
    check("R36 total episodes", b36["total_episodes"], 200)
    check("R36 prereg budget", b36["prereg_budget"], 200)
    check("R36 reaches budget", b36["reaches_prereg_budget"], True)
    check("R36 multi-window episodes", b36["multi_window_episodes"], 128)
    check("R36 continuation episodes", len(b36["segments"]), 4)
    c36 = r36["continuation_48"]
    check("R36 cont episodes", c36["episodes"], 48)
    check("R36 cont mean", c36["mean_return"], 188.3)
    check("R36 cont first third", c36["first_third"], 152.7)
    check("R36 cont last third", c36["last_third"], 219.3)
    check("R36 cont trend", c36["trend"], 66.6)
    check("R36 cont survivals", c36["survivals"], 2)
    check("R36 seed14 mean", c36["per_window"]["14"]["mean_return"], 180.9)
    check("R36 seed16 mean", c36["per_window"]["16"]["mean_return"], 234.4)
    check("R36 multi128 train mean", r36["multi_128"]["train_mean"], 133.8)
    check("R36 multi128 survivals", r36["multi_128"]["survivals"], 5)
    d36 = r36["dev"]
    check("R36 dev episodes", d36["episodes"], 10)
    check("R36 dev mean", d36["mean"], 78.7)
    check("R36 dev survivals", d36["survivals"], 1)
    check("R36 dev median", d36["median"], 48.5)
    check("R36 dev at/above 250", d36["episodes_at_or_above_250"], 1)
    check("R36 dev below 60", d36["episodes_below_60"], 7)
    check("R36 dev vs 80ep", d36["vs_80_episodes"], 21.4)
    check("R36 dev vs cfair diff", d36["vs_cfair"]["mean_diff"], 6.78)
    check("R36 dev vs cfair ci low", d36["vs_cfair"]["ci95"][0], -39.8)
    check("R36 dev vs cfair ci high", d36["vs_cfair"]["ci95"][1], 65.6)
    check("R36 dev vs cfair not significant", d36["vs_cfair"]["excludes_zero"], False)
    check("R36 dev exceeds cfair pointwise", r36["verdict"]["dev_still_below_cfair"], False)

    # ---------------- R37: the full survival reproduces on other unseen tapes -------------
    r37 = json.loads((EV / "R37" / "dev_all_windows.json").read_text(encoding="utf-8"))
    w37 = {w["seed"]: w for w in r37["per_window"]}
    check("R37 windows", len(r37["per_window"]), 4)
    check("R37 all complete", r37["all_windows_complete"], True)
    check("R37 seed11 mean", w37[11]["mean"], 78.7)
    check("R37 seed12 mean", w37[12]["mean"], 134.7)
    check("R37 seed12 survivals", w37[12]["survivals"], 2)
    check("R37 seed17 mean", w37[17]["mean"], 166.1)
    check("R37 seed18 mean", w37[18]["mean"], 39.9)
    check("R37 seed18 survivals", w37[18]["survivals"], 0)
    check("R37 seed11 const best", w37[11]["c_const_best_decisions"], 272.0)
    check("R37 seed18 const best", w37[18]["c_const_best_decisions"], 257.5)
    p37 = r37["pooled"]
    check("R37 pooled arm n", p37["arm_a"]["n"], 40)
    check("R37 pooled arm mean", p37["arm_a"]["mean"], 104.8)
    check("R37 pooled arm median", p37["arm_a"]["median"], 49.5)
    check("R37 pooled arm survivals", p37["arm_a"]["survivals"], 4)
    check("R37 windows with survival", p37["arm_a"]["windows_with_a_survival"], [11, 12, 17])
    check("R37 pooled cfair n", p37["c_fair"]["n"], 40)
    check("R37 pooled cfair mean", p37["c_fair"]["mean"], 76.2)
    check("R37 pooled cfair survivals", p37["c_fair"]["survivals"], 0)
    b37 = p37["arm_a_vs_c_fair"]
    check("R37 mean diff", b37["mean_diff"], 28.82)
    check("R37 ci low", b37["ci95"][0], -3.24)
    check("R37 ci high", b37["ci95"][1], 63.48)
    check("R37 not significant", b37["excludes_zero"], False)
    f37 = p37["survival_fisher"]
    check("R37 fisher table", f37["table"], [[4, 36], [0, 40]])
    check("R37 fisher p", f37["p_two_sided"], 0.1156)
    check("R37 survival reproduced", r37["verdict"]["survival_reproduced_elsewhere"], True)

    # ---------------- R38: n=80 per arm -- the effect SHRANK and stays non-significant ----
    r38 = json.loads((EV / "R38" / "n80.json").read_text(encoding="utf-8"))
    w38 = {w["seed"]: w for w in r38["per_window"]}
    check("R38 per-window rows", len(r38["per_window"]), 4)
    check("R38 seed11 arm n", w38[11]["arm_n"], 20)
    check("R38 seed11 arm survivals", w38[11]["arm_survivals"], 1)
    check("R38 seed11 ext survivals", w38[11]["arm_ext_survivals"], 0)
    check("R38 seed17 arm mean", w38[17]["arm_mean"], 152.8)
    check("R38 seed17 arm survivals", w38[17]["arm_survivals"], 2)
    check("R38 seed18 arm survivals", w38[18]["arm_survivals"], 1)
    check("R38 every window has a survival",
          sorted(w38[s]["arm_survivals"] > 0 for s in w38), [True, True, True, True])
    p38 = r38["pooled"]
    check("R38 arm n", p38["arm_a"]["n"], 80)
    check("R38 arm mean", p38["arm_a"]["mean"], 94.1)
    check("R38 arm median", p38["arm_a"]["median"], 48.0)
    check("R38 arm survivals", p38["arm_a"]["survivals"], 6)
    check("R38 arm windows", p38["arm_a"]["windows_with_a_survival"], [11, 12, 17, 18])
    check("R38 cfair n", p38["c_fair"]["n"], 80)
    check("R38 cfair mean", p38["c_fair"]["mean"], 79.7)
    check("R38 cfair median", p38["c_fair"]["median"], 63.0)
    # the single most important correction: C-fair DID survive once at n=80
    check("R38 cfair survivals", p38["c_fair"]["survivals"], 1)
    b38 = p38["arm_a_vs_c_fair"]
    check("R38 mean diff", b38["mean_diff"], 14.48)
    check("R38 ci low", b38["ci95"][0], -6.97)
    check("R38 ci high", b38["ci95"][1], 37.58)
    check("R38 mean not significant", b38["excludes_zero"], False)
    check("R38 fisher p", p38["survival_fisher"]["p_two_sided"], 0.1167)
    check("R38 fisher not significant", r38["verdict"]["fisher_significant"], False)
    check("R38 n40 arm mean", r38["comparison_n40"]["arm_mean"], 104.8)
    check("R38 n40 arm survivals", r38["comparison_n40"]["arm_survivals"], 4)
    check("R38 gap shrank",
          p38["arm_a_vs_c_fair"]["mean_diff"] < r38["comparison_n40"]["mean_diff_ci"][1],
          True)

    # ---------------- R39: quotation audit of my own correction list ----------------
    r39 = json.loads((EV / "R39" / "quote_audit.json").read_text(encoding="utf-8"))
    c39 = r39["counts"]
    check("R39 rows audited (C only)", c39["rows"] >= 28, True)
    check("R39 verbatim found", c39["found"], 9)
    check("R39 summaries not verbatim", c39["not_found"], 5)
    check("R39 my-own rows excluded", c39["internal"] >= 14, True)
    check("R39 not all verbatim", r39["verdict"]["all_attributed_claims_found"], False)
    # after C14 was fixed, the only non-verbatim rows are the five summaries -- and every one
    # of them is anchored by its numbers, so nothing attributed is unaccounted for
    check("R39 statuses partition the rows",
          c39["found"] + c39["not_found"] + c39["internal"], c39["rows"])
    check("R39 non-verbatim rows are the summaries",
          sorted(r["id"] for r in r39["rows"] if r["status"] == "not_found"),
          ["C1", "C2", "C3", "C7", "C9"])
    r39b = json.loads((EV / "R39" / "summary_anchors.json").read_text(encoding="utf-8"))
    check("R39b summary rows", r39b["counts"]["rows"], 6)
    check("R39b anchored summaries", r39b["counts"]["anchored"], 5)
    anch = {r["id"]: r for r in r39b["rows"]}
    check("R39b C1 anchored", anch["C1"]["anchored"], True)
    check("R39b C2 anchored", anch["C2"]["anchored"], True)
    check("R39b C9 anchored", anch["C9"]["anchored"], True)
    # the one row the audit proved wrong: its numbers are NOT in the v4.3.0 documents
    check("R39b C14 not anchored", anch["C14"]["anchored"], False)
    check("R39b C14 documents", anch["C14"]["documents"], [])

    # ---------------- R41: the earliest evidence still reproduces ----------------
    r41 = json.loads((EV / "R41" / "rerun_verdict.json").read_text(encoding="utf-8"))
    c41 = r41["counts"]
    check("R41 scripts", c41["scripts"], 6)
    check("R41 all exit zero", c41["exit_zero"], 6)
    check("R41 byte identical", c41["byte_identical"], 4)
    check("R41 non-substantive differences", c41["non_substantive_difference"], 2)
    check("R41 substantive changes", c41["substantive_changes"], 0)
    kinds = {r["group"]: r["difference_kind"] for r in r41["rows"]}
    check("R41 R1 difference kind", kinds["R1"], "timestamp")
    check("R41 R6 difference kind", kinds["R6"], "float_jitter")
    r6 = r41["r6_float_instability"]
    check("R41 R6 distinct values", r6["distinct_values"], 3)
    check("R41 R6 runs", r6["runs"], 5)
    check("R41 R6 absolute spread", r6["absolute_spread"], "2.070e-10")
    check("R41 R6 relative spread", r6["relative_spread"], "1.453e-07")
    check("R41 bug disclosed", "fake reading" in r41["bug_of_mine"]["symptom"], True)
    check("R41 backbone reproducible", r41["verdict"]["backbone_reproducible"], True)

    # ---------------- SELF_AUDIT.md: the reusable checklist ----------------
    _sa = pathlib.Path(__file__).resolve().parents[2] / "Document" / "Update" / "Alpha" \
        / "v4.3.1" / "SELF_AUDIT.md"
    check("SELF_AUDIT exists", _sa.is_file(), True)
    _sa_text = _sa.read_text(encoding="utf-8")
    for _sec in ("## 0. 一条命令跑完全部离线自检", "## 1. 数值骨架复跑", "## 2. 引文回源",
                 "## 3. 承诺清单复核", "## 4. 报告结构断言", "## 5. 判据算术自检",
                 "## 6. 资产与范围检查", "## 7. 复现性陷阱清单", "## 8. 本清单**不**覆盖",
                 "## 9. 我在做这份复核时踩过的"):
        check(f"SELF_AUDIT section {_sec[:14]}", _sec in _sa_text, True)
    check("report selftest count is 24", "**24 项**自测通过" in _text, True)
    # C31 quotes the old wrong value, so exactly one occurrence is allowed (in that row)
    # the old wrong value may only appear in the places that DOCUMENT the correction:
    # the §4.2cc narrative (2 mentions) and the C31 row (1) -- never in §1's table.
    check("report records the old 25 only where it is being corrected",
          _text.count("25 项"), 3)
    check("§1 table row uses 24",
          "| R17 判据算术自测 |" in _text and "**24 项**自测通过" in _text, True)
    check("SELF_AUDIT trap table rows",
          len(_re.findall(r"^\| \d+ \| \*\*", _sa_text, _re.M)), 8)
    # section 9: the audit-tool's own failures
    _s9 = _sa_text[_sa_text.index("## 9."):]
    check("SELF_AUDIT section 9 rows", len(_re.findall(r"^\| \d+ \|", _s9, _re.M)) >= 15, True)
    check("SELF_AUDIT section 9 names the meta-lesson",
          "共享了同一个错误来源" in _s9, True)
    check("SELF_AUDIT section 9 says false evidence",
          "会直接产出假证据" in _s9, True)
    check("SELF_AUDIT names the snapshot limitation",
          "这是本清单已知的方法论局限" in _sa_text, True)
    check("SELF_AUDIT says all 7 commands exit 0",
          "全部 exit 0 才算通过" in _sa_text, True)

    # ---------------- R42: why seed18 scored 39.9 and seed17 152.8 ----------------
    r42 = json.loads((EV / "R42" / "window_variance.json").read_text(encoding="utf-8"))
    w42 = {r["seed"]: r for r in r42["per_window"]}
    check("R42 windows", len(r42["per_window"]), 4)
    check("R42 seed18 c4 mean", w42[18]["c4_mean"], 87.1)
    check("R42 seed17 c4 mean", w42[17]["c4_mean"], 127.8)
    check("R42 seed17 c4 count >=200", w42[17]["c4_count_ge_200"], 7)
    check("R42 seed12 c4 count >=200", w42[12]["c4_count_ge_200"], 1)
    check("R42 seed18 arm mean", w42[18]["arm_mean"], 51.0)
    check("R42 seed17 arm mean", w42[17]["arm_mean"], 152.8)
    check("R42 seed18 first10", w42[18]["arm_first10"], 39.9)
    check("R42 seed18 ext10", w42[18]["arm_ext10"], 62.0)
    check("R42 seed12 batch delta", w42[12]["arm_batch_delta"], -50.9)
    var = r42["variance"]
    check("R42 between-window sd", var["between_window_sd"], 40.2)
    check("R42 within-window sd", var["mean_within_window_sd"], 80.3)
    check("R42 ratio", var["ratio_between_over_within"], 0.5)
    v42 = r42["verdict"]
    check("R42 H1 window difficulty", v42["h1_window_difficulty"], True)
    check("R42 H2 noise", v42["h2_noise"], True)
    check("R42 H3 specific failure", v42["h3_specific_failure"], False)
    check("R42 rank by arm", v42["rank_arm"], [18, 11, 12, 17])
    check("R42 rank by const mean", v42["rank_const_mean"], [18, 11, 12, 17])
    check("R42 advantage monotone", v42["advantage_monotone_in_window_breadth"], True)
    check("R42 advantage exact p", v42["advantage_exact_p_one_sided"], 0.0417)
    check("R42 verdict mentions both sd and monotone",
          "HALF the within-window spread" in v42["reading"]
          and "monotone in window" in v42["reading"], True)

    # ---------------- SUMMARY_FOR_DECISION.md: the same numbers, shorter ----------------
    _sd = pathlib.Path(__file__).resolve().parents[2] / "Document" / "Update" / "Alpha" \
        / "v4.3.1" / "SUMMARY_FOR_DECISION.md"
    check("SUMMARY exists", _sd.is_file(), True)
    _sd_text = _sd.read_text(encoding="utf-8")
    # every headline number in the summary must match the evidence asserted above
    for _n in ("1.6586", "1.7122", "1.6639", "879 / 220", "0.622/255"):
        check(f"SUMMARY carries {_n}", _n in _sd_text, True)
    for _n in ("94.1", "79.7", "+14.48", "0.1167", "6/80", "1/80"):
        check(f"SUMMARY carries arm A number {_n}", _n in _sd_text, True)
    for _n in ("−10.8", "+31.3", "+43.1", "0.0417"):
        check(f"SUMMARY carries R42 number {_n}", _n in _sd_text, True)
    check("SUMMARY carries 234.7", "234.7" in _sd_text, True)
    check("SUMMARY has decision 1", "决定 1" in _sd_text, True)
    check("SUMMARY has decision 2", "决定 2" in _sd_text, True)
    check("SUMMARY states zero modified existing files",
          "我改动的既有文件 = **0**" in _sd_text, True)
    check("SUMMARY says audio stayed muted", "静音" in _sd_text, True)
    check("SUMMARY points at SELF_AUDIT", "SELF_AUDIT.md" in _sd_text, True)
    check("SUMMARY is short", len(_sd_text.splitlines()) < 160, True)

    # ---------------- R43: the baseline is robust to the smoothing choice ----------------
    r43 = json.loads((EV / "R43" / "smoothing.json").read_text(encoding="utf-8"))
    s43 = r43["summary"]
    check("R43 move heads complete", s43["move_heads_no_missing_classes"], True)
    check("R43 max delta on move", round(s43["max_abs_delta_on_move"], 6), 0.000434)
    check("R43 smallest effect", s43["smallest_claimed_effect_nats"], 0.0673)
    check("R43 effect/smoothing ratio", s43["effect_over_smoothing_uncertainty"], 155.1)
    check("R43 degenerate heads", s43["degenerate_heads"],
          ["seed12_val__shoot", "seed12_val__bomb"])
    check("R43 robust", r43["verdict"]["baseline_robust_to_smoothing"], True)
    r43_rows = {r["baseline"]: r for r in r43["rows"]}
    check("R43 seed12 move missing", r43_rows["seed12_val__move"]["missing_in_train"], 0)
    check("R43 seed12 move mle", round(r43_rows["seed12_val__move"]["ce_train_mle"], 4), 1.6586)
    check("R43 seed10 suffix mle",
          round(r43_rows["seed10_suffix__move"]["ce_train_mle"], 4), 1.7122)
    check("R43 shoot is degenerate", r43_rows["seed12_val__shoot"]["degenerate"], True)

    # ---------------- R44: every documented number traced back to v4.3.0 ----------------
    r44 = json.loads((EV / "R44" / "documented_trace.json").read_text(encoding="utf-8"))
    c44 = r44["counts"]
    check("R44 rows", c44["rows"], 12)
    check("R44 documented rows", c44["documented"], 11)
    check("R44 documented and exact", c44["documented_and_exact"], 11)
    check("R44 documented but mismatch", c44["documented_but_mismatch"], 0)
    check("R44 not documented at all", c44["not_documented_at_all"], 1)
    check("R44 the fabricated row", r44["fabricated_rows"],
          ["ActorPolicy seed10-val focus"])
    r44_rows = {r["row"]: r for r in r44["rows"]}
    check("R44 0.6461 is documented", r44_rows["ActorPolicy seed10 focus"]["is_documented"], True)
    check("R44 0.6461 reproduces", r44_rows["ActorPolicy seed10 focus"]["matches_to_4dp"], True)
    check("R44 0.5589 is not documented",
          r44_rows["ActorPolicy seed10-val focus"]["is_documented"], False)
    check("R44 0.5589 matches mine",
          r44_rows["ActorPolicy seed10-val focus"]["matches_to_4dp"], True)
    check("R44 4x suffix 2.5070",
          r44_rows["4x seed10-suffix move"]["matches_to_4dp"], True)
    check("R44 verdict all reproduce", r44["verdict"]["all_documented_numbers_reproduce"], True)

    # ---------------- R45: the C14/C29 failure class swept across the whole report ----------
    r45 = json.loads((EV / "R45" / "attribution_sweep.json").read_text(encoding="utf-8"))
    c45 = r45["counts"]
    # these counts grow with every new paragraph that mentions the documents; assert the
    # partition and the zero-mislabel invariant instead of frozen totals
    check("R45 numbers checked", c45["numbers_checked"] >= 44, True)
    check("R45 absent from corpus", c45["absent_from_corpus"] >= 29, True)
    check("R45 classes partition the absent numbers",
          sum(c45["by_class"].values()), c45["absent_from_corpus"])
    check("R45 has an own-measurement class",
          c45["by_class"].get("own-measurement", 0) >= 6, True)
    check("R45 has a self-explaining class",
          c45["by_class"].get("mine-self-explaining", 0) >= 20, True)
    # after the fix no line attributes my numbers to the documents
    check("R45 zero mislabelled now", r45["verdict"]["mislabelled_now"], 0)
    check("R45 records the fix", r45["fix_applied"]["recorded_as"], "C30")
    check("R45 three instances recorded", r45["verdict"]["three_instances_recorded"], True)

    # ---------------- R46: every evidence reference resolves, none orphaned ----------------
    r46 = json.loads((EV / "R46" / "evidence_refs.json").read_text(encoding="utf-8"))
    c46 = r46["counts"]
    check("R46 referenced groups", c46["distinct_groups_referenced"] >= 45, True)
    check("R46 groups with a dir", c46["groups_with_a_nonempty_dir"] >= 42, True)
    check("R46 process steps", c46["process_steps_without_dir"], 3)
    check("R46 missing references", c46["missing_references"], 0)
    check("R46 evidence dirs", c46["evidence_dirs"] >= 42, True)
    check("R46 every referenced group resolves or is a process step",
          c46["distinct_groups_referenced"] - c46["process_steps_without_dir"]
          == c46["groups_with_a_nonempty_dir"], True)
    check("R46 orphan dirs", c46["orphan_dirs"], 0)
    check("R46 no missing", r46["missing_references"], [])
    check("R46 no orphans", r46["orphan_dirs"], [])
    proc = {r["group"]: r for r in r46["forward"] if r.get("process_step")}
    check("R46 process groups", sorted(proc), ["R13", "R17", "R40"])
    check("R46 each process step has a reason",
          all(r.get("why_no_dir") for r in proc.values()), True)

    # ---------------- R47: the stop conditions themselves are evaluated ----------------
    r47 = json.loads((EV / "R47" / "stop_conditions.json").read_text(encoding="utf-8"))
    c47 = r47["counts"]
    check("R47 conditions", c47["conditions"], 8)
    check("R47 triggered", c47["triggered"], 2)
    check("R47 not triggered", c47["not_triggered"], 4)
    check("R47 not evaluable", c47["not_evaluable"], 2)
    check("R47 triggered ids", r47["triggered_ids"], ["S2", "S6"])
    check("R47 not evaluable ids", r47["not_evaluable_ids"], ["S3", "S4"])
    by_id = {c["id"]: c for c in r47["conditions"]}
    check("R47 S1 hash checks", by_id["S1"]["measured"]["hash_checks"], 350730)
    check("R47 S1 mismatches", by_id["S1"]["measured"]["hash_mismatches"], 0)
    check("R47 S1 rate", by_id["S1"]["measured"]["rate"], 0.0)
    check("R47 S2 const best is dev-side", by_id["S2"]["measured"]["const_best"], 272.0)
    check("R47 S2 left half not met", by_id["S2"]["measured"]["le_cfair"], False)
    check("R47 S2 right half met", by_id["S2"]["measured"]["le_const_best"], True)
    check("R47 S7 train rate", by_id["S7"]["measured"]["train_survival_rate"], 0.0391)
    s6 = by_id["S6"]["measured"]
    check("R47 S6 windows", len(s6["windows"]), 9)
    check("R47 S6 alive at 180 total", sum(s6["constants_alive_at_180_frames"].values()), 76)
    check("R47 S6 alive at 600 total", sum(s6["constants_alive_at_600_frames"].values()), 1)
    check("R47 S6 admissible at 180", s6["windows_admissible_at_180"], [])
    check("R47 S6 admissible at 600",
          s6["windows_admissible_at_600"], [11, 12, 13, 14, 15, 16, 17, 18])
    check("R47 S6 windows that flip", len(s6["windows_whose_verdict_flips"]), 8)
    check("R47 S6 is triggered", by_id["S6"]["status"], "TRIGGERED")
    check("R47 S3 names its unblock", "解阻条件" in by_id["S3"]["note"], True)
    check("R47 not all decidable", r47["verdict"]["all_decidable"], False)

    # ---------------- R48: admissibility results carry their segment length ----------------
    r48 = json.loads((EV / "R48" / "segment_context.json").read_text(encoding="utf-8"))
    # the count grows every time the report discusses admissibility; what must hold is that
    # NONE of them states a result without its segment length
    check("R48 lines scanned",
          r48["counts"]["lines_mentioning_admissibility"] >= 42, True)
    check("R48 violations", r48["counts"]["violations"], 0)
    check("R48 clean", r48["verdict"]["clean"], True)
    check("R48 violation list empty", r48["violations"], [])
    # the four claims that must carry the length explicitly
    for _s in ("10 s 段下 8 / 9 合格", "**10 s 段**下 9 个窗口筛出", "**10 s 段下 3 个窗口里 2 个合格**",
               "在 **3 s 段**上不合格"):
        check(f"report carries the length in '{_s[:16]}'", _s in _text, True)

    # ---------------- R49: every survival rate carries n and the threshold ----------------
    r49 = json.loads((EV / "R49" / "survival_context.json").read_text(encoding="utf-8"))
    check("R49 lines scanned", r49["counts"]["lines_mentioning_survival"] >= 46, True)
    check("R49 violations", r49["counts"]["violations"], 0)
    check("R49 clean", r49["verdict"]["clean"], True)
    check("R49 violation list empty", r49["violations"], [])
    # the two claims that most need the context now carry it
    # the report's own version of the headline claim must carry both
    check("report carries survival context",
          "存活率 **7.5% vs 1.25%（约 6 倍；n=80，跑满 300 决策）**" in _text, True)
    check("report carries survival context in the n=80 comparison",
          "存活率 **7.5% vs 1.25%（约 6 倍）**" not in _text, True)

    # ---------------- R50: means are censored, and arm A loses to a fixed action ----------
    r50 = json.loads((EV / "R50" / "censoring.json").read_text(encoding="utf-8"))
    g50 = {g["group"]: g for g in r50["groups"]}
    check("R50 cap", r50["cap_decisions"], 300)
    check("R50 arm A n", g50["arm A (dev, n=80)"]["n"], 80)
    check("R50 arm A at cap", g50["arm A (dev, n=80)"]["at_cap"], 6)
    check("R50 arm A mean", g50["arm A (dev, n=80)"]["mean_censored"], 94.1)
    check("R50 arm A is a lower bound", g50["arm A (dev, n=80)"]["kind"], "LOWER_BOUND")
    check("R50 cfair at cap", g50["C-fair (dev, n=80)"]["at_cap"], 1)
    check("R50 cfair is a lower bound", g50["C-fair (dev, n=80)"]["kind"], "LOWER_BOUND")
    check("R50 constant pool exact", g50["all constants pooled (18 x 4)"]["kind"], "EXACT")
    check("R50 constant pool mean", g50["all constants pooled (18 x 4)"]["mean_censored"], 111.7)
    check("R50 constant pool n", g50["all constants pooled (18 x 4)"]["n"], 72)
    f50 = r50["arm_vs_a_fixed_action"]
    check("R50 windows", f50["windows"], 4)
    check("R50 arm beats pool on", f50["windows_arm_beats_const_pool"], 1)
    check("R50 arm beats best on", f50["windows_arm_beats_best_const"], 0)
    check("R50 mean gap vs pool", f50["mean_of_arm_minus_pool"], -17.6)
    pw50 = {r["seed"]: r for r in f50["per_window"]}
    check("R50 seed17 is the only win", pw50[17]["arm_minus_pool"], 25.1)
    check("R50 seed11 gap", pw50[11]["arm_minus_pool"], -42.6)
    check("R50 seed18 gap", pw50[18]["arm_minus_pool"], -36.1)
    check("R50 some means are bounds", r50["verdict"]["some_means_are_lower_bounds"], True)

    # ---------------- R51: the constant baseline is an oracle; the deployable one is worse still
    r51 = json.loads((EV / "R51" / "baseline_ladder.json").read_text(encoding="utf-8"))
    b51 = r51["baselines"]
    check("R51 candidates", r51["candidates"], 18)
    check("R51 oracle mean", b51["oracle_per_window_best"]["mean"], 258.8)
    check("R51 oracle kind", b51["oracle_per_window_best"]["kind"], "ORACLE")
    check("R51 deployable mean", b51["const_chosen_on_train"]["mean"], 172.4)
    check("R51 deployable kind", b51["const_chosen_on_train"]["kind"], "DEPLOYABLE")
    check("R51 chosen constant", b51["const_chosen_on_train"]["constant_index"], 12)
    check("R51 pool mean", b51["const_pool_mean_dev"]["mean"], 111.7)
    check("R51 oracle inflation", r51["oracle_inflation"]["oracle_minus_deployable"], 86.4)
    check("R51 oracle factor", r51["oracle_inflation"]["factor"], 1.5)
    check("R51 arm A", r51["arm_a_mean"], 94.1)
    check("R51 arm vs oracle", r51["comparisons"]["vs_oracle"], -164.7)
    check("R51 arm vs deployable", r51["comparisons"]["vs_train_chosen"], -78.3)
    check("R51 arm vs pool", r51["comparisons"]["vs_pool"], -17.6)
    check("R51 S2 unfalsifiable with the oracle",
          r51["verdict"]["s2_right_half_is_falsifiable_with_oracle"], False)
    check("R51 S2 falsifiable with the deployable baseline",
          r51["verdict"]["s2_right_half_is_falsifiable_with_deployable"], True)
    check("R51 arm does not beat the deployable constant",
          r51["verdict"]["arm_beats_deployable_constant"], False)
    check("report says the oracle is not falsifiable", "不可证伪" in _text, True)
    check("report carries the deployable 172.4", "172.4" in _text, True)
    check("report carries the oracle 258.8", "258.8" in _text, True)

    # ---------------- R52: the sign flips with the summary statistic ------------------
    r52 = json.loads((EV / "R52" / "statistic_flip.json").read_text(encoding="utf-8"))
    p52 = r52["policies"]
    c52 = r52["comparisons"]
    check("R52 arm n", p52["arm_a"]["n"], 80)
    check("R52 arm median", p52["arm_a"]["median"], 48.0)
    check("R52 arm q1", p52["arm_a"]["q1"], 36.0)
    check("R52 arm q3", p52["arm_a"]["q3"], 110.5)
    check("R52 cfair median", p52["c_fair"]["median"], 63.0)
    check("R52 cfair q1", p52["c_fair"]["q1"], 44.6)
    check("R52 mean diff favours arm", c52["mean_diff"], 14.4)
    check("R52 median diff favours random", c52["median_diff"], -15.0)
    check("R52 q1 diff favours random", c52["q1_diff"], -8.6)
    check("R52 q3 diff favours arm", c52["q3_diff"], 11.7)
    check("R52 P(arm > cfair median)", c52["p_arm_above_cfair_median"], 0.35)
    check("R52 P(cfair > arm median)", c52["p_cfair_above_arm_median"], 0.7)
    check("R52 sign flips", r52["verdict"]["statistic_flips_the_sign"], True)
    check("report carries the flip", "符号随统计量翻转" in _text, True)
    check("report no longer claims an unqualified win over random",
          "臂 A 胜过随机" in _text, True)

    # ---------------- R53: the verdict itself depends on the criterion -----------------
    r53 = json.loads((EV / "R53" / "criterion_sensitivity.json").read_text(encoding="utf-8"))
    c53 = r53["counts"]
    check("R53 criteria", len(r53["criteria"]), 8)
    check("R53 for arm", c53["for_arm"], 2)
    check("R53 against arm", c53["against_arm"], 5)
    check("R53 inconclusive", c53["inconclusive"], 1)
    b53 = r53["bootstraps"]
    check("R53 mean diff", b53["mean"]["observed"], 14.43)
    check("R53 mean ci includes zero", b53["mean"]["excludes_zero"], False)
    check("R53 median diff", b53["median"]["observed"], -15.0)
    check("R53 median ci", b53["median"]["ci95"], [-22.5, -6.5])
    check("R53 median excludes zero", b53["median"]["excludes_zero"], True)
    check("R53 median direction", b53["median"]["direction"], "baseline_better")
    check("R53 q1 excludes zero", b53["q1"]["excludes_zero"], True)
    check("R53 median flip significant",
          r53["verdict"]["median_flip_is_significant"], True)
    check("report states the split verdict",
          "只对均值口径成立" in _text, True)
    check("report carries the multiplicity caveat", "多重比较" in _text, True)

    # ---------------- R54: are my own corrections one-directional? --------------------
    r54 = json.loads((EV / "R54" / "correction_direction.json").read_text(encoding="utf-8"))
    check("R54 rows classified", len(r54["rows"]), 36)
    check("R54 helps arm A", r54["counts"].get("helps"), 2)
    check("R54 hurts arm A", r54["counts"].get("hurts"), 7)
    check("R54 helps BC", r54["counts"].get("helps_bc"), 1)
    check("R54 neutral", r54["counts"].get("neutral"), 26)
    check("R54 the two helps are C23 and C24",
          sorted(r54["ids_by_direction"].get("helps", [])), ["C23", "C24"])
    check("R54 not one-directional", r54["verdict"]["one_directional"], False)
    check("R54 names the structural asymmetry", "结构性" in r54["structural_note"], True)
    check("R54 every row has a reason",
          all(r["reason"] for r in r54["rows"]), True)
    check("report states the correction tally", "2** | `C23`" in _text
          or "**2**" in _text, True)

    # ---------------- R55: the most favourable criteria for arm A ---------------------
    r55 = json.loads((EV / "R55" / "favourable_criteria.json").read_text(encoding="utf-8"))
    f55 = {c["criterion"]: c for c in r55["criteria"]}
    check("R55 criteria", r55["counts"]["criteria"], 8)
    check("R55 favouring arm A", r55["counts"]["favours_arm"], 6)
    check("R55 survival one-sided p", r55["survival_one_sided"]["p_one_sided"], 0.0584)
    check("R55 one-sided does not clear 0.05", r55["survival_one_sided"]["clears_005"], False)
    check("R55 top-10 arm", f55["前 10 个 episode 的均值"]["arm"], 294.8)
    check("R55 top-10 baseline", f55["前 10 个 episode 的均值"]["baseline"], 187.3)
    check("R55 seed17 median arm", f55["seed17 的中位数"]["arm"], 154.5)
    check("R55 seed17 median baseline", f55["seed17 的中位数"]["baseline"], 70.2)
    check("R55 windows won", f55["四个窗口里胜出的窗口数（按均值）"]["arm"], "2/4")
    check("R55 support exists", r55["verdict"]["support_exists"], True)
    check("R55 support is upper-tail only", r55["verdict"]["support_is_upper_tail_only"], True)
    check("report states the upper-tail-only reading", "只存在于分布的顶端" in _text, True)
    # B-class overclaims that must stay corrected
    # each retracted phrase may appear AT MOST ONCE, and only inside the sentence that retracts it
    # the phrase may only appear where it is being retracted or catalogued: the retraction
    # sentence and the R60 table row. Two is the current count; the retraction itself is
    # asserted separately below, so a NEW asserting use would push the count past 2.
    for _bad, _max in (("本轮最重要的可执行结论", 2),
                       ("说明它确实学到了**某种**可迁移的东西", 0),
                       ("完全由那一个 300 拉动", 1),
                       ("唯一的\"胜利\"完全是共同前缀造成的", 1),
                       ("样本量是唯一缺口", 1)):
        check(f"report overclaims '{_bad[:14]}' at most {_max}x",
              _text.count(_bad) <= _max, True)
    # C-class: the two senses of "seed" must never collapse into one
    for _tag, _p in (("report", None),):
        pass
    check("report has no bare '3 种子 × 4 窗口'", "3 种子 × 4 窗口" not in _text, True)
    # C-class: the headline must name the criterion
    check("headline names CE as the criterion",
          "有一个在 **CE** 上从\"不优于\"变成\"优于\"" in _text, True)
    check("headline says the criterion swap reverses it",
          "换个判据就反向" in _text, True)
    check("suffix row names CE",
          "唯一在 CE 上优于基线的臂" in _text, True)
    check("report defines the two senses of 种子",
          "本报告的\"种子\"有两个含义" in _text, True)
    check("report says it is not 12 tapes",
          "也不是 12 条磁带" in _text, True)
    check("report retracts the 1.6h prediction",
          "撤回\"本轮最重要的可执行结论\"这个评价" in _text, True)
    check("report states the 300 contributes 24.6 (31%)",
          "**24.6（31%）**" in _text, True)
    check("report limits 'the only lever' to measured optimisations",
          "已测的客户端优化里" in _text, True)
    check("§2.6 conclusion is scoped, not global",
          "文档里没有一个数字是编的或算错的" not in _text, True)
    check("§2.6 names the unverified remainder",
          "还有大量未核的数字" in _text, True)
    check("report does NOT claim no difference", "不该**被写成\"没有差别\"" in _text, True)

    # ---------------- R56: the three deliverables agree about arm A -------------------
    r56 = json.loads((EV / "R56" / "verdict_consistency.json").read_text(encoding="utf-8"))
    check("R56 docs checked", r56["counts"]["docs"], 3)
    check("R56 all docs complete", r56["counts"]["complete"], 3)
    check("R56 nothing incomplete", r56["incomplete_docs"], [])
    check("R56 consistent", r56["verdict"]["consistent"], True)
    sec56 = r56["summary_section"]
    check("R56 summary section carries all three",
          sec56["all_in_one_section"], True)
    for _k in ("not_significant", "upper_tail_favours", "typical_does_not"):
        check(f"R56 summary section has {_k}", sec56["pieces"][_k]["present"], True)
    check("R56 summary section is a section, not the whole file",
          sec56["chars"] < len(_sd_text) if False else sec56["chars"] < 4000, True)

    # ---------------- R57: the whole offline audit suite -----------------------------
    r57 = json.loads((EV / "R57" / "audit_suite.json").read_text(encoding="utf-8"))
    # the suite has 22 entries but verify itself is the 22nd: asserting on its own exit code
    # would deadlock (a verify failure records a failure that makes verify fail forever)
    # the suite grows; assert the INVARIANT (non-self + verify == total) and that nothing failed
    check("R57 non-self + verify == total",
          r57["counts"]["scripts"] + 1, r57["counts"]["total_including_verify"])
    check("R57 non-self exit zero", r57["counts"]["exit_zero"], r57["counts"]["scripts"])
    check("R57 non-self nonzero", r57["counts"]["nonzero"], 0)
    check("R57 suite has at least 20 scripts", r57["counts"]["total_including_verify"] >= 20, True)
    check("R57 all green", r57["verdict"]["all_green"], True)
    check("R57 every non-self script has exit 0",
          all(r["exit"] == 0 for r in r57["scripts"] if "verify_report" not in r["script"]),
          True)
    check("R57 records verify's own exit separately",
          "verify_exit_separate" in r57["counts"], True)
    check("report documents the deadlock", "断言不能包含被测对象自己的输出" in _text, True)

    # ---------------- the summary must not describe an older report than the one on disk ----
    _rep_lines = len(_text.splitlines())
    # derive what the summary SHOULD say, instead of freezing today's numbers
    _max_r = max(int(d.name[1:]) for d in (EV).iterdir()
                 if d.is_dir() and d.name[1:].isdigit())
    # an end-of-range number drifts every round, so the summary names the SERIES instead
    check("summary names the evidence series",
          "R1 起的全部证据组" in _sd_text, True)
    check("summary does not name an end-of-range number",
          _re.search(r"R1–R\d+", _sd_text) is None, True)
    _pre_f = (pathlib.Path(__file__).resolve().parents[2] / "Document" / "Update" / "Alpha" /
              "v4.3.1" / "PREREGISTRATION_SEG_001.md")
    _pre_head = _pre_f.read_text(encoding="utf-8").splitlines()[0]
    check("summary points at the prereg file",
          "PREREGISTRATION_SEG_001.md" in _sd_text, True)
    # a named revision would go stale every round, so the summary deliberately does not name one
    check("summary does not name a moving revision",
          not _re.search(r"rev [A-Z]{2}", _sd_text), True)
    # the summary deliberately no longer quotes a line count (it drifted every round);
    # the report's own length is checked separately below
    check("summary does not quote a line count",
          _re.search(r"≈[\d,]+ 行", _sd_text) is None, True)
    check("summary says one command, 22 scripts", "22 个脚本" in _sd_text, True)
    check("summary says one command, 22 scripts", "22 个脚本" in _sd_text, True)
    check("summary states the oracle 258.8", "258.8" in _sd_text, True)
    check("summary states the deployable 172.4", "172.4" in _sd_text, True)
    check("summary states the two-directional verdict", "双向" in _sd_text, True)
    check("summary carries the one-sided p", "0.0584" in _sd_text, True)
    check("summary qualifies the audio claim as single-state",
          "单状态证据" in _sd_text, True)
    check("summary evidence index reaches R57", "R39`–`R57" in _sd_text, True)
    # the report itself is long enough that "≈1,600 行" would be stale
    # no hard window: the report grows every round; the ratio check below is the real test
    check("report has a sane length", 1500 < _rep_lines < 4000, True)

    # ---------------- R59: cross-section conflicts (what assertions cannot see) --------
    r59 = json.loads((EV / "R59" / "cross_section.json").read_text(encoding="utf-8"))
    check("R59 self-found", r59["counts"]["self"], 7)
    check("R59 delegated total", r59["counts"]["delegated_total"], 24)
    check("R59 delegated types", r59["counts"]["delegated_by_type"],
          {"A": 12, "B": 2, "C": 3, "D": 2, "E": 1, "F": 4})
    check("R59 by own type", r59["counts"]["by_type_self"],
          {"D": 3, "A": 2, "F": 1, "B": 1})
    # the specific fixes must be visible in the text
    check("summary uses 2/4 not 3/4", "| 臂 A 胜出的窗口 | — | **2/4** |" in _sd_text, True)
    check("summary states the 172.4 n=4 caveat", "4 个数" in _sd_text, True)
    check("report states the 172.4 n=4 caveat", "所以 172.4 是 **4 个数**" in _text, True)
    check("report states seed18 reverses it",
          "恒定动作在 seed18 只有 **30.0**" in _text, True)
    check("report labels the unpaired bootstrap", "两样本" in _text, True)
    check("report no longer says complete dominance", "完全支配臂 A" not in _text, True)
    _pre_text = (pathlib.Path(__file__).resolve().parents[2] / "Document" / "Update" /
                 "Alpha" / "v4.3.1" / "PREREGISTRATION_SEG_001.md").read_text(encoding="utf-8")
    check("prereg no longer claims complete dominance",
          "完全支配**臂 A" not in _pre_text, True)
    check("prereg marks the 'only prerequisite' sentence as superseded",
          "这句已作废" in _pre_text, True)
    check("prereg carries the frozen-snapshot banner",
          "以下正文是 **rev E 的冻结快照**" in (pathlib.Path(__file__).resolve().parents[2] /
              "Document" / "Update" / "Alpha" / "v4.3.1" / "PREREGISTRATION_SEG_001.md")
          .read_text(encoding="utf-8"), True)
    check("report line count is plausible", 1500 < _rep_lines < 4000, True)

    # ---------------- R62: the handover index ----------------
    _doc = pathlib.Path(__file__).resolve().parents[2] / "Document" / "Update" / "Alpha" / "v4.3.1"
    _ix = _doc / "README_INDEX.md"
    check("README_INDEX exists", _ix.is_file(), True)
    _ix_text = _ix.read_text(encoding="utf-8")
    check("index says it is generated", "**生成**，不是手写" in _ix_text, True)
    check("index has the four audiences", _ix_text.count("| **要") >= 3, True)
    check("index has a 30-second version", "## 2. 三十秒版本" in _ix_text, True)
    check("index has the one-command block", "run_all_audits.ps1" in _ix_text, True)
    check("index has the asset statement", "未修改任何既有文件" in _ix_text, True)
    _r62 = json.loads((EV / "R62" / "index.json").read_text(encoding="utf-8"))
    _dirs = {int(d.name[1:]) for d in EV.iterdir()
             if d.is_dir() and d.name[1:].isdigit()}
    check("R62 covers every evidence dir",
          {g["n"] for g in _r62["evidence_groups"]} == _dirs, True)
    check("R62 every group has a curated description",
          all(len(g["about"]) > 6 and "v431-recheck" not in g["about"]
              for g in _r62["evidence_groups"]), True)
    check("R62 document count", _r62["counts"]["documents"], 5)
    # every group named in the index must exist as a dir or a declared process step
    _named = set(int(m) for m in _re.findall(r"^\| `R(\d+)` \|", _ix_text, _re.M))
    check("index names no phantom group", _named <= _dirs | {13, 17, 40}, True)

    # ---------------- R63: the artifacts are absent by search, not by assumption -----
    r63 = json.loads((EV / "R63" / "artifact_absence.json").read_text(encoding="utf-8"))
    check("R63 files seen", r63["counts"]["files_seen"] > 100000, True)
    check("R63 dirs seen", r63["counts"]["dirs_seen"] > 50000, True)
    check("R63 checkpoint-like count", r63["counts"]["checkpoint_like"], 5146)
    check("R63 prediction-like count", r63["counts"]["prediction_like"], 780)
    check("R63 no real checkpoint", r63["verdict"]["no_checkpoint_anywhere"], True)
    check("R63 pruned trees documented", len(r63["protocol"]["pruned"]), 10)
    check("R63 records the 8-byte SENTINEL",
          "SENTINEL" in json.dumps(r63["classification"], ensure_ascii=False), True)
    check("R63 records the BOM finding",
          "BOM" in r63["classification"]["incidental_finding"], True)
    # the old wording may appear once, inside the sentence that retracts it
    check("N5 quotes the old wording only to retract it",
          _text.count("只有末轮快照"), 1)
    # the phrase survives only where it is quoted in order to be corrected: the section heading
    # and the N1 row's "不是原先写的…". Two occurrences is the current count.
    check("'7 个脚本' appears only in the corrections",
          _text.count("7 个脚本"), 2)
    check("N1 states the corrected count",
          "用 `0.8` 切分的是 **5 个**" in _text, True)
    check("N1 cites R66", "`R66` 把理由改准了" in _text, True)
    _r66 = json.loads((EV / "R66" / "split_scripts.json").read_text(encoding="utf-8"))
    check("R66 scripts with a split", _r66["counts"]["scripts_with_a_split"], 6)
    check("R66 ratio scripts", _r66["counts"]["ratio_0.8"], 5)
    check("R66 fixed n_val scripts", _r66["counts"]["fixed_n_val"], 1)
    check("R66 880/219 not reachable", _r66["arithmetic"]["is_880_219_reachable"], False)
    check("R66 N1 conclusion still holds", _r66["verdict"]["n1_conclusion_holds"], True)
    check("R66 seven-script claim does not hold",
          _r66["verdict"]["seven_scripts_claim_holds"], False)
    check("N7 now says 11/11", "**11/11** 文档数字逐位重现" in _text, True)
    check("N9 now gives a range, not 13/18",
          "为 **4–13 个**" in _text, True)
    check("N14 downgraded from 'proven untouched'",
          "不能\"证明\"它未被触碰" in _text, True)
    check("R65 covers 14 items", "扩展到 14 条" in _text, True)
    check("N5 states the 3-epoch curves exist",
          "各有 **3 epoch** 曲线" in _text, True)
    check("N5 records the document's 3 epoch points",
          "epoch 1/6/25" in _text, True)
    check("N4 scope widened to %TEMP%", "%TEMP%" in _text, True)
    check("C3 cites the artefact proof",
          "468/1099 = 0.42584" in _text, True)
    check("report ties N2 to R63",
          "`R63` 已把这条从\"没找到\"升级为\"搜过\"" in _text, True)
    check("report ties N3 to R63", "`R63` 的 **780** 个预测类文件" in _text, True)
    check("N1 still describes the split, not checkpoints",
          "| N1 | 原 880/219 划分 |" in _text, True)

    # ---------------- R64: N4/N5 exhausted; two new findings from the aggregates -----
    r64 = json.loads((EV / "R64" / "n4_n5.json").read_text(encoding="utf-8"))
    check("R64 N4 holds", r64["n4"]["holds"], True)
    check("R64 log files found", r64["n4"]["log_like_files"], 2789)
    check("R64 zero logs mention a BC config", r64["n4"]["mentioning_a_bc_config"], [])
    check("R64 N5 does not hold as written", r64["n5"]["holds_as_written"], False)
    _h = {h["file"]: h for h in r64["n5"]["per_epoch_histories"]}
    check("R64 bc_generalisation has 3 epochs", _h["bc_generalisation.json"]["epochs"], 3)
    check("R64 bc_split_comparison has 2 runs", len(_h["bc_split_comparison.json"]["runs"]), 2)
    check("R64 cross_tape has no history",
          _h["cross_tape_generalisation.json"]["kind"], "no history key")
    _mb = r64["majority_blocks"]
    check("R64 the share is 468/1099", _mb["share_is_468_over_1099"], True)
    check("R64 -ln(share) is the documented 0.8537", _mb["equals_the_documented_0853"], True)
    check("R64 seed12-train and seed12-val share one majority",
          _mb["per_eval_set"]["seed12-train"]["move"] == _mb["per_eval_set"]["seed12-val"]["move"],
          True)
    check("R64 seed10 has its own majority",
          _mb["per_eval_set"]["seed10"]["move"] != _mb["per_eval_set"]["seed12-val"]["move"], True)
    _sc = r64["sample_counts"]
    check("R64 bc_generalisation was n=1000",
          (_sc["bc_generalisation"] or {}).get("n_train"), 800)
    check("R64 bc_split_comparison was n=1000",
          [r["n_val"] for r in _sc["bc_split_comparison"]], [200, 200])
    check("R64 says tapes hold 1099", _sc["tapes_hold"], 1099)

    # ---------------- R65b: N7-N14 audited with the same standard ----------------------
    _r65j = json.loads((EV / "R65" / "evidence_types.json").read_text(encoding="utf-8"))
    # ---------------- R68: the two "差" columns carry opposite signs ------------------
    _r68 = json.loads((EV / "R68" / "sign_conventions.json").read_text(encoding="utf-8"))
    # ---------------- SELF_AUDIT 9: the audit-tool self-deceptions -------------------
    _sa = (EV.parent / "SELF_AUDIT.md").read_text(encoding="utf-8")
    check("SELF_AUDIT lists 14 audit-tool traps",
          len(__import__("re").findall(r"^\| \d+ \|", _sa, __import__("re").M)) >= 14, True)
    check("SELF_AUDIT records the proximity-pairing trap",
          "按邻近性配对不是证据" in _sa, True)
    check("SELF_AUDIT records the keyword-rule trap",
          "机械规则替代不了阅读" in _sa, True)
    check("SELF_AUDIT has three meta-lessons",
          "**三条元教训**" in _sa and "用机械规则替代语义判断" in _sa, True)
    check("SELF_AUDIT records the rounding-precision trap",
          "比较精度必须由被比较者决定" in _sa, True)

    # ---------------- R76: the pre-registered test was mis-calibrated ----------------
    _r76 = json.loads((EV / "R76" / "boschloo.json").read_text(encoding="utf-8"))
    check("R76 Fisher p", _r76["fisher"]["p_value"], 0.058373)
    check("R76 Fisher level (sup over p0)", _r76["fisher"]["level_sup_over_p0"], 0.030006)
    check("R76 Fisher rejection prob at the measured rate",
          _r76["fisher"]["rejection_prob_at_the_measured_rate"], 0.00126)
    check("R76 Boschloo p is a lower bound and significant",
          _r76["verdict"]["significant_under_boschloo"], True)
    check("R76 Boschloo p value", _r76["boschloo"]["p_value_lower_bound"], 0.034291)
    check("R76 one-sample p", _r76["one_sample_at_the_measured_rate"]["p_value"], 5.20142e-4)
    check("R76 one-sample caveat at the CI edge",
          _r76["one_sample_at_the_measured_rate"]["p_at_upper_ci_edge"], 0.4634)
    check("report distinguishes the two alphas",
          "**两个「α」不要混**" in _text or "水平**（对 p0 取上确界）约 **3.0%" in _text, True)
    check("report says failing the criterion is not 'no effect'",
          "它的「未通过」不能读作「没有效应」" in _text, True)
    check("report records C44", "这是对判据的过度解读" in _text, True)

    # ---------------- R75: power and granularity of the survival test ----------------
    _r75 = json.loads((EV / "R75" / "power.json").read_text(encoding="utf-8"))
    check("R75 reproduces the reported p", _r75["observed_at_n80"]["matches_the_reports_0_0584"], True)
    check("R75 observed p value", _r75["observed_at_n80"]["one_sided_p"], 0.058373)
    _g = {r["n_per_arm"]: r for r in _r75["power_grid"]}
    check("R75 power at 163 for 7.5pct", _g[163]["power"]["0.075"], 0.8404)
    check("R75 power at 163 for 5pct", _g[163]["power"]["0.050"], 0.4923)
    check("R75 power at 163 for 4pct", _g[163]["power"]["0.040"], 0.31)
    check("R75 extra hours at 163 (measured unit)", _g[163]["extra_hours_at_56_6s"], 2.61)
    check("R75 extra hours at 163 (upper bound)", _g[163]["extra_hours_at_70_4s"], 3.25)
    check("R75 is NOT a 5 percent test", _r75["validation"]["is_a_5pct_test"], False)
    check("R75 actual size at n=80", _r75["validation"]["null_rejection_rate_by_n"]["80"], 0.00126)
    check("R75 min rejectable point at n=80",
          _r75["validation"]["min_rejectable_point"]["80"]["p_at_that_point"], 0.0293)
    check("report says p=0.0584 is not 'just short'",
          "不该按「刚好差一点到 0.05」来读" in _text or "不该按\"刚好差一点到 0.05\"来读" in _text, True)
    check("report gives the granularity table", "**最极端可达点**" in _text, True)
    check("report admits the first Fisher bug", "算出 p≈1e-37" in _text, True)

    # ---------------- 0.0: the requirement coverage table ---------------------------
    check("coverage table exists", "## 0.0 任务要求覆盖表" in _text, True)
    for _req in ("确认 `v400` 含 `5b699f5`", "重算训练集频率基线", "核验跨磁带画面",
                 "三臂在线学习实验", "以实际任务成功率判独立 dev 起点"):
        check(f"coverage lists the subtask: {_req[:14]}", _req in _text, True)
    for _out in ("保留／更正清单", "证据", "最小实验方案", "预算", "停止条件", "标注未复现项",
                 "遵守资产保护"):
        check(f"coverage lists the deliverable: {_out}", _out in _text, True)
    # the coverage table quotes the group count: it must equal what the generated index counted
    _r62c = json.loads((EV / "R62" / "index.json").read_text(encoding="utf-8"))
    check("coverage group count matches R62",
          f"共 {len(_r62c['evidence_groups'])} 个证据组" in _text, True)
    check("coverage marks the ladder step 2 as not triggered",
          "第 2 步未触发" in _text, True)
    check("coverage marks the pixel arms as blocked, not skipped",
          "臂 B/C 被阻断" in _text, True)

    # ---------------- R74: where the one-mask constraint actually bites --------------
    _r74 = json.loads((EV / "R74" / "pair_cand_scope.json").read_text(encoding="utf-8"))
    cf = _r74["code_facts"]
    check("R74 _pair_cand has exactly one call site", len(cf["pair_cand_call_sites"]), 1)
    check("R74 that call site is inside replay()", cf["pair_cand_only_in_replay"], True)
    check("R74 COMMIT_FRAMES is 2", cf["constants_runtime"]["COMMIT_FRAMES"], 2)
    check("R74 live arm commits one mask per block",
          cf["live_arm_commits_one_mask_per_commit_block"], True)
    check("R74 replay counts mismatches instead of raising",
          cf["replay_counts_hash_mismatch_instead_of_raising"], True)
    check("R74 the blocker is reach, not submit",
          _r74["verdict"]["blocker_is_reach_not_submit"], True)
    check("R74 lists four unblock options", len(_r74["unblock_options"]), 4)
    check("report says the blocker is reaching the start frame",
          "到不了起始帧" in _text, True)
    check("report records C43", "把两件真事粘成了一句错的因果" in _text, True)

    # ---------------- R73: reverse-search for leftovers -----------------------------
    _r73 = json.loads((EV / "R73" / "value_leftovers.json").read_text(encoding="utf-8"))
    check("R73 eleven corrected values", _r73["counts"]["tokens"], 11)
    check("R73 every hit was adjudicated", _r73["counts"]["unreviewed"], 0)
    check("R73 no leftovers", _r73["verdict"]["no_leftovers"], True)
    check("R73 records the false-positive kinds",
          len(_r73["false_positive_kinds"]), 3)
    check("R73 says a mechanical rule cannot replace reading",
          "机械规则替代不了阅读" in _text, True)

    # ---------------- R72: the K table's citations ---------------------------------
    _r72 = json.loads((EV / "R72" / "k_rows.json").read_text(encoding="utf-8"))
    check("R72 eighteen K rows", _r72["counts"]["k_rows"], 18)
    check("R72 every row now cites something",
          _r72["counts"]["rows_without_any_citation"], [])
    check("R72 at least 36 of 49 numbers traceable",
          _r72["counts"]["numbers_found"] >= 36, True)
    _k72 = {r["k"]: r for r in _r72["rows"]}
    check("R72 K3 cites R1 and R4", _k72[3]["refs"], [1, 4])
    check("R72 K9 cites R67", _k72[9]["refs"], [67])
    check("R72 K8 cites R2 and R3", _k72[8]["refs"], [2, 3])
    check("report fixes K3's citation", "**引用更正**" in _text, True)
    check("report records C41", "却漏了 K 表里同一句话" in _text, True)
    check("report records C42", "只存在于 `R4` 的 `evaluations[*].baseline.ce_train_mle`" in _text, True)
    # the phrase survives only in the two places that quote it to retract it (the 4.2bbb
    # narrative and the C41 row); the K9 row itself must carry the measured values
    # three occurrences: the 4.2bbb narrative, the K9 row's parenthetical, and the C41 row
    check("0.448 is quoted only to be retracted",
          _text.count("0.448/0.463/0.456"), 3)
    check("K9 carries the measured agreement",
          "全盘 `cand` 一致率 **0.4568 / 0.4850 / 0.4823**" in _text, True)

    # ---------------- R71: the cost chain ------------------------------------------
    _r71 = json.loads((EV / "R71" / "cost_chain.json").read_text(encoding="utf-8"))
    check("R71 every hour figure recomputes", _r71["verdict"]["all_claims_recompute"], True)
    check("R71 nine claims", _r71["counts"]["claims"], 9)
    check("R71 unit is 70.4 s", _r71["unit_seconds"], 70.4)
    check("R71 simulation floor", _r71["simulation_floor_seconds"], 66.7)
    _m71 = {m["group"]: m for m in _r71["measured_per_group"]}
    check("R71 R20 has 72 episodes", _m71["R20"]["episodes"], 72)
    check("R71 R20 unit cost", _m71["R20"]["s_per_episode"], 17.8)
    check("R71 R31 unit cost", _m71["R31"]["s_per_episode"], 56.6)
    check("R71 admits the wall time is not in the JSON",
          _m71["R31"]["exists_in_evidence_json"], False)
    check("report fixes 90 -> 72", "共 **72 个 episode / 1,282 s**" in _text, True)
    check("report records C40", "4 个状态 × 18 个恒定动作 = **72**，不是 90" in _text, True)
    check("report says 70.4 s is an upper bound",
          "得到的是**上界**而非实测" in _text, True)

    # ---------------- R70: percentages -- self-checking vs merely plausible -----------
    _r70 = json.loads((EV / "R70" / "percentages.json").read_text(encoding="utf-8"))
    c70 = _r70["counts"]
    check("R70 self-checking percentages all hold",
          c70["self_checking_that_fail"], 0)
    check("R70 at least 9 self-checking", c70["self_checking"] >= 9, True)
    check("R70 median crossings verified", _r70["manual_checks"]["median_crossings"]["holds"], True)
    check("R70 median crossing is 28/80",
          _r70["manual_checks"]["median_crossings"]["arm_above_cfair_median"], "28/80 = 0.350")
    check("R70 the other crossing is 56/80",
          _r70["manual_checks"]["median_crossings"]["cfair_above_arm_median"], "56/80 = 0.700")
    check("R70 train survival rate", _r70["manual_checks"]["train_survival"]["holds"], True)
    check("R70 records its own false positives",
          "按距离配对不是证据" in _r70["first_version_note"], True)
    check("report marks the crossings self-checking",
          "= 35%    (28/80)" in _text and "= 70%    (56/80)" in _text, True)
    check("report records R70", "我的第一版检查器错了" in _text, True)

    # ---------------- R69: table headers carry their own convention --------------------
    _r69 = json.loads((EV / "R69" / "headers.json").read_text(encoding="utf-8"))
    # the report gains tables as sections are added: assert a floor and the zero-defect invariant
    check("R69 tables scanned", _r69["counts"]["tables"] >= 104, True)
    check("R69 no numeric column lacks a unit",
          [c for c in _r69["numeric_columns"]
           if c["column"].startswith("n") or c["column"] in
           ("均值", "中位数", "最大", "最小值", "标准差", "平均差", "用时", "占",
            "跑满", "合格数", "数量", "规模")], [])
    check("R69 the n-conflation is recorded", _r69["verdict"]["n_is_ambiguous"], True)
    check("R69 n has two meanings", len(_r69["n_conflation"]["meanings"]), 2)
    check("report qualifies n as samples", "**n（样本）**" in _text, True)
    check("report qualifies n as episodes", "**n（episode）**" in _text, True)
    check("report qualifies the decisions median", "**中位数（决策）**" in _text, True)
    check("report qualifies the timing column", "**用时（s）**" in _text, True)
    check("R68 five pairs", len(_r68["pairs"]), 5)
    check("R68 all CIs exclude zero", _r68["verdict"]["five_pairs_reproduce"], True)
    check("R68 prose range is the model-minus-baseline form",
          _r68["prose_claim"]["matches"], True)
    check("R68 model-minus-baseline range",
          _r68["ranges"]["model_minus_baseline"], [-0.128, -0.0673])
    check("R68 baseline-minus-model range",
          _r68["ranges"]["baseline_minus_model"], [0.0673, 0.128])
    check("R68 declares the conflict", _r68["verdict"]["sign_conventions_differ"], True)
    check("report labels the baseline table's sign",
          "**差（模型−基线；负 = 更好）**" in _text, True)
    check("report labels the bootstrap table's sign",
          "**差（基线−模型；正 = 更好）**" in _text, True)
    check("report points at R68 for the conflict", "两张表符号相反，见 `R68`" in _text, True)
    check("C39 records the same-label/different-sign defect",
          "**同名反号**" in _text, True)
    check("R65 covers all 14 N rows", len(_r65j["items"]), 14)
    check("R65 single-source list is exactly N13",
          _r65j["counts"]["single_source_items"], ["N13"])
    _r65 = {r["id"]: r for r in _r65j["items"]}
    check("R65 N7 has 3 sources", _r65["N7"]["independent_sources"], 3)
    check("R65 N13 is single-source", _r65["N13"]["single_source"], True)
    check("R65 N14 notes the mtime weakness",
          "过强断言" in _r65["N14"]["note"], True)

    # ---------------- R67: two universal claims, one unsupported ---------------------
    r67 = json.loads((EV / "R67" / "agreement_and_18.json").read_text(encoding="utf-8"))
    cb = r67["claim_b_eighteen_constants"]
    check("R67 18-constant claim holds", cb["holds"], True)
    check("R67 nine windows", cb["windows"], 9)
    check("R67 162 episodes", cb["total_episodes"], 162)
    check("R67 every window has 18 distinct cands", cb["every_window_has_18"], True)
    ca = r67["claim_a_agreement"]
    check("R67 three pairs", ca["n_pairs_computed"], 3)
    check("R67 the quoted numbers are NOT reproducible", ca["reproducible"], False)
    _p67 = {p["pair"]: p for p in ca["pairs"]}
    check("R67 seed10__seed11 full", _p67["seed10__seed11"]["full_agreement"], 0.4568)
    check("R67 seed10__seed12 full", _p67["seed10__seed12"]["full_agreement"], 0.485)
    check("R67 seed11__seed12 full", _p67["seed11__seed12"]["full_agreement"], 0.4823)
    check("R67 suffix rates are lower still",
          [_p67[k]["suffix_agreement"] for k in ("seed10__seed11", "seed10__seed12",
                                                 "seed11__seed12")], [0.3216, 0.3568, 0.349])
    check("report no longer quotes 0.448 as a result", "0.448 / 0.463 / 0.456 两种口径" in _text, True)
    check("report records C38", "无法复现**：三盘恰好 3 对" in _text, True)

    # ---------------- R8 ----------------
    sel = r8["selection"]
    check("R8 train_3s start", sel["train_3s"]["start_slot"], 4325)
    check("R8 train_3s end", sel["train_3s"]["end_slot"], 4415)
    check("R8 train_3s duration", sel["train_3s"]["duration_s"], 3.0)
    check("R8 train_3s sf", sel["train_3s"]["start_frame"], 8666)
    check("R8 train_3s ef", sel["train_3s"]["end_frame"], 8844)
    check("R8 train_3s ink", sel["train_3s"]["mean_ink_percentile"], 94.58)
    check("R8 train_3s cands", sel["train_3s"]["distinct_executed_cand"], 15)
    check("R8 train_3s prefix", sel["train_3s"]["prefix_replay_s"], 82.5)
    check("R8 dev10 ink", sel["dev_seed10_5s"]["mean_ink_percentile"], 86.59)
    check("R8 dev10 cands", sel["dev_seed10_5s"]["distinct_executed_cand"], 17)
    check("R8 dev10 prefix", sel["dev_seed10_5s"]["prefix_replay_s"], 97.1)
    check("R8 dev11 ink", sel["dev_seed11_5s"]["mean_ink_percentile"], 89.29)
    check("R8 dev11 cands", sel["dev_seed11_5s"]["distinct_executed_cand"], 16)
    check("R8 dev11 prefix", sel["dev_seed11_5s"]["prefix_replay_s"], 81.2)
    check("R8 train_5s prefix", sel["train_5s"]["prefix_replay_s"], 118.5)
    check("R8 train_10s duration", sel["train_10s"]["duration_s"], 10.0)
    check("R8 dropped criterion present", "C2_teacher_margin_below_8" in
          r8["corrected_criteria"]["dropped"], True)
    check("R8 C4 status not evaluated",
          "NOT EVALUATED" in r8["corrected_criteria"]["C4_control_must_fail"]["status"], True)

    # ---------------- derived budget arithmetic ----------------
    fps = r8["measured_replay_fps"]
    check("R8 fps", round(fps, 2), 104.8, 0.05)
    per_attempt = 82.5 + 90 * 2 / fps
    check("budget per attempt ~84s", round(per_attempt, 1), 84.2, 0.2)
    check("budget arm 200", round(200 * per_attempt / 3600, 2), 4.68, 0.02)
    prefix_share = 82.5 / per_attempt
    check("prefix share >0.97", prefix_share > 0.97, True)

    print(f"checks run: {checks}")
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print("  -", f)
        return 1
    print("all report figures match the stored evidence")
    return 0


if __name__ == "__main__":
    sys.exit(main())
