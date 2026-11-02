"""``python -m touhou_rl.escape`` entry point (D0 scope only).

Implemented today: ``schema``, ``validate``, ``baseline-fixture``, ``snapshot``,
``evidence``.  The four commands the plan lists for later stages (``synth``,
``evaluate``, ``render``, ``benchmark``) exist as explicit stubs: they print
which task will implement them and exit with code 3 instead of silently doing
nothing or pretending to run.

No subcommand touches the game, memory or a network.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

from . import PACKAGE_STAGE, baseline, contracts, snapshot

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_NOT_IMPLEMENTED = 3

STUBS = {
    "benchmark": "D3/D4 (forecast and search budgets)",
}


def _load_documents(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "documents" in payload:
        payload = payload["documents"]
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"{path}: expected a JSON object or array of objects")


def cmd_schema(args: argparse.Namespace) -> int:
    schema = contracts.contract_schema()
    text = json.dumps(schema, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")
    else:
        sys.stdout.write(text)
    return EXIT_OK


def cmd_validate(args: argparse.Namespace) -> int:
    path = Path(args.path)
    documents = _load_documents(path)
    failures = 0
    for index, document in enumerate(documents):
        issues = contracts.validate_document(document)
        label = document.get("schema", "?") if isinstance(document, dict) else "?"
        if issues:
            failures += 1
            print(f"document {index} ({label}): {len(issues)} issue(s)")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"document {index} ({label}): ok")
    return EXIT_INVALID if failures else EXIT_OK


def cmd_baseline_fixture(args: argparse.Namespace) -> int:
    path = Path(args.out)
    if args.check:
        issues = baseline.verify_golden(path)
        if issues:
            print(f"{path}: {len(issues)} difference(s)")
            for issue in issues:
                print(f"  - {issue}")
            return EXIT_INVALID
        print(f"{path}: matches the generated fixture")
        return EXIT_OK
    document = baseline.write_golden(path)
    print(f"wrote {path}")
    for key, value in document["digests"].items():
        print(f"  {key}: {value}")
    return EXIT_OK


def cmd_snapshot(args: argparse.Namespace) -> int:
    out = Path(args.out)
    document = snapshot.write_snapshot(out, Path(args.root))
    print(f"wrote {out}")
    print(f"  baseline_surface_digest: {document['baseline_surface_digest']}")
    git = document["git"]
    print(f"  git: {'available' if git.get('available') else 'unavailable'} dirty={git.get('dirty')}")
    return EXIT_OK


def cmd_evidence(args: argparse.Namespace) -> int:
    base = Path(args.base)
    include = tuple(item.strip() for item in args.include.split(",") if item.strip())
    manifest = snapshot.evidence_manifest(base, include)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    missing = [entry["path"] for entry in manifest["entries"] if entry.get("missing")]
    print(f"wrote {out} ({len(manifest['entries'])} entries, digest {manifest['digest']})")
    for path in missing:
        print(f"  missing: {path}")
    return EXIT_INVALID if missing else EXIT_OK


def cmd_audit_traces(args: argparse.Namespace) -> int:
    from . import audit

    paths = [Path(item) for item in args.paths]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        print(f"missing trace file(s): {', '.join(missing)}", file=sys.stderr)
        return EXIT_INVALID
    report = audit.audit_paths(paths, max_decisions=args.max_decisions)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        totals = report["totals"]
        print(f"wrote {out}")
        print(
            f"  episodes={totals['episodes']} decisions={totals['decisions']} "
            f"complete={totals['complete_ratio']} off_model_speed={totals['off_model_ratio']}"
        )
    else:
        sys.stdout.write(text)
    return EXIT_OK


def cmd_seeds(args: argparse.Namespace) -> int:
    from . import seeds

    document = seeds.write(Path(args.out), master=args.master)
    print(f"wrote {args.out}")
    print(f"  families={len(document['families'])} seeds/family={document['seeds_per_family']}")
    print(f"  edge_instances={len(document['edge_instances'])} digest={document['digest']}")
    return EXIT_OK


def cmd_shadow(args: argparse.Namespace) -> int:
    from . import shadow

    summaries = []
    for trace, sidecar in args.pair:
        if not Path(trace).is_file() or not Path(sidecar).is_file():
            print(f"missing trace or sidecar: {trace} / {sidecar}", file=sys.stderr)
            return EXIT_INVALID
        summaries.append(
            shadow.shadow_episode(
                trace,
                sidecar,
                max_decisions=args.max_decisions,
                margin=args.margin,
                examples=args.examples,
                skip_overrides=args.skip_overrides,
            )
        )
    totals = {
        "episodes": len(summaries),
        "decisions_checked": sum(item["decisions_checked"] for item in summaries),
        "frame_collisions": sum(item["frame_collisions"] for item in summaries),
        "near_misses_0_5": sum(item["near_misses_0_5"] for item in summaries),
        "swept_collisions": sum(item["swept_collisions"] for item in summaries),
        "swept_only_collisions": sum(item["swept_only_collisions"] for item in summaries),
        "reported_safe_but_frame_collision": sum(
            item["reported_safe_but_frame_collision"] for item in summaries
        ),
        "collisions_within_3_decisions_of_death": sum(
            item["collisions_within_3_decisions_of_death"] for item in summaries
        ),
        "skipped": {},
    }
    for item in summaries:
        for key, value in item["skipped"].items():
            totals["skipped"][key] = totals["skipped"].get(key, 0) + value
    document = {"schema": shadow.SCHEMA, "episodes": summaries, "totals": totals}
    text = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")
        print(
            f"  checked={totals['decisions_checked']} frame_collisions={totals['frame_collisions']} "
            f"reported_safe={totals['reported_safe_but_frame_collision']} "
            f"swept_only={totals['swept_only_collisions']} "
            f"near_death={totals['collisions_within_3_decisions_of_death']}"
        )
    else:
        sys.stdout.write(text)
    return EXIT_OK


def _select_scenario(args: argparse.Namespace):
    from . import families

    if getattr(args, "edge", None):
        instances = families.edge_instances()
        if args.edge not in instances:
            raise SystemExit(f"unknown edge instance {args.edge}")
        return instances[args.edge]
    if getattr(args, "family", None):
        return families.build(args.family, args.seed)
    raise SystemExit("one of --family/--seed or --edge is required")


def cmd_synth(args: argparse.Namespace) -> int:
    from . import families, synth

    if args.list:
        for family in families.FAMILY_IDS:
            print(family)
        for edge_id in families.edge_instances():
            print(edge_id)
        return EXIT_OK
    scenario = _select_scenario(args)
    document = dict(scenario.to_dict())
    document["signature"] = synth.scenario_signature(scenario)
    document["initial_clearance"] = round(synth.initial_clearance(scenario), 4)
    text = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")
    else:
        sys.stdout.write(text)
    return EXIT_OK


def cmd_reference(args: argparse.Namespace) -> int:
    from . import families, reference, synth

    if args.sweep:
        rows = []
        for family in families.FAMILY_IDS:
            scenario = families.build(family, args.seed)
            result = reference.solve(scenario, node_budget=args.budget)
            rows.append({"scenario_id": scenario.scenario_id, "kind": "family", **result.to_dict()})
        for edge_id, scenario in families.edge_instances().items():
            result = reference.solve(scenario, node_budget=args.budget)
            rows.append({"scenario_id": edge_id, "kind": "edge", **result.to_dict()})
        document = {
            "schema": "v370-reference-sweep/1",
            "seed": args.seed,
            "budget": args.budget,
            "rows": rows,
            "status_counts": {},
        }
        for row in rows:
            document["status_counts"][row["status"]] = document["status_counts"].get(row["status"], 0) + 1
        text = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            print(f"wrote {out}")
            print(f"  status_counts={document['status_counts']}")
        else:
            sys.stdout.write(text)
        return EXIT_OK

    scenario = _select_scenario(args)
    result = reference.solve(scenario, node_budget=args.budget)
    document = {
        "schema": "v370-reference/1",
        "scenario": synth.scenario_signature(scenario),
        "scenario_id": scenario.scenario_id,
        "expected": scenario.expected,
        "result": result.to_dict(),
    }
    text = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")
    else:
        sys.stdout.write(text)
    print(f"{scenario.scenario_id}: {result.status.value} nodes={result.nodes} "
          f"min_clearance={result.min_clearance}")
    return EXIT_OK


def cmd_forecast(args: argparse.Namespace) -> int:
    import time

    from touhou_rl.escape import forecast as forecast_module
    from touhou_rl.escape import synth

    scenario = _select_scenario(args)
    config = forecast_module.ForecastConfig(
        horizon_frames=args.horizon,
        inflation_px=args.inflation,
        cell_px=args.cell,
        max_entities=args.max_entities,
    )
    frame = synth.SyntheticEnv(scenario).frame
    started = time.perf_counter()
    field = forecast_module.DangerField.from_scenario(scenario, frame, config)
    built = forecast_module.build(field, elapsed_ms=(time.perf_counter() - started) * 1000.0)
    if args.benchmark:
        import numpy as np

        rng = np.random.default_rng(0)
        builds = []
        for _ in range(args.benchmark):
            begin = time.perf_counter()
            field = forecast_module.DangerField.from_scenario(scenario, frame, config)
            forecast_module.build(field)
            builds.append((time.perf_counter() - begin) * 1000.0)
        queries = []
        for _ in range(1000):
            x = float(rng.uniform(8.0, 376.0))
            y = float(rng.uniform(16.0, 432.0))
            t = int(rng.integers(1, config.horizon_frames + 1))
            begin = time.perf_counter()
            field.clearance(x, y, t)
            queries.append((time.perf_counter() - begin) * 1000.0)
        document = {
            "schema": "v370-forecast-benchmark/1",
            "scenario_id": scenario.scenario_id,
            "entities": int(field.positions.shape[0]),
            "nodes": len(config.time_nodes()),
            "build_ms": {
                "runs": len(builds),
                "p50": round(float(np.percentile(builds, 50)), 3),
                "p95": round(float(np.percentile(builds, 95)), 3),
                "max": round(float(np.max(builds)), 3),
                "target_ms": 4.0,
            },
            "query_ms": {
                "runs": len(queries),
                "p50": round(float(np.percentile(queries, 50)), 4),
                "p95": round(float(np.percentile(queries, 95)), 4),
            },
        }
        text = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            print(f"wrote {out}")
        else:
            sys.stdout.write(text)
        return EXIT_OK

    document = built.to_json_dict()
    text = json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out} ({len(text) // 1024} KiB)")
    else:
        sys.stdout.write(text)
    print(
        f"{scenario.scenario_id}: nodes={len(document['time_nodes_frames'])} "
        f"kept={int(field.positions.shape[0])} "
        f"dropped={document['dropped_entities']} "
        f"shape={document['coverage_shape']} "
        f"calibrated={document['calibrated']} "
        f"build_ms={built.build_ms:.2f}"
    )
    return EXIT_OK


def cmd_record(args: argparse.Namespace) -> int:
    from . import controller as controller_module
    from . import forecast as forecast_module
    from . import trace as trace_module

    scenario = _select_scenario(args)
    config = controller_module.ControllerConfig(
        planner=planner_module_config(args),
    )
    summary = trace_module.record_episode(
        scenario,
        out_dir=Path(args.out),
        config=config,
        forecast_config=forecast_module.ForecastConfig(horizon_frames=args.horizon),
        max_decisions=args.max_decisions,
    )
    print(f"wrote {args.out}/{summary['trace']} ({summary['decisions']} decisions)")
    print(f"  hit={summary['hit']} frames={summary['frames']} survived={summary['survived']}")
    return EXIT_OK


def planner_module_config(args: argparse.Namespace):
    from . import planner as planner_module

    return planner_module.PlannerConfig(
        horizon_decisions=args.plan_horizon,
        continuation_decisions=args.plan_continuation,
        beam_width=args.beam,
    )


def cmd_render(args: argparse.Namespace) -> int:
    from . import render as render_module
    from . import trace as trace_module

    trace_path = Path(args.trace)
    sidecar = Path(args.sidecar) if args.sidecar else trace_path.with_suffix(".npz")
    if not trace_path.is_file() or not sidecar.is_file():
        print(f"missing trace or sidecar: {trace_path} / {sidecar}", file=sys.stderr)
        return EXIT_INVALID
    trace = trace_module.load_trace(trace_path)
    if args.text_only:
        text = render_module.render_text(trace)
        if args.out:
            out = Path(args.out)
            if out.is_dir() or out.suffix == "":
                out = out / f"{trace[0]['episode']}.md"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            print(f"wrote {out}")
        else:
            sys.stdout.write(text)
        return EXIT_OK
    result = render_module.render_episode(trace_path, sidecar, args.out or "replays", decision=args.decision)
    print(f"wrote {result['text']}")
    print(f"wrote {result['png']}")
    print(f"  classification={result['attribution']['classification']} hit_frame={result['attribution']['hit_frame']}")
    return EXIT_OK


def cmd_evaluate(args: argparse.Namespace) -> int:
    from . import controller as controller_module
    from . import evaluate as evaluate_module
    from . import families, planner as planner_module, reference

    if args.split == "formal":
        config_path = Path(args.config)
        if not config_path.is_file():
            print(f"formal split needs a frozen config: {config_path} is missing", file=sys.stderr)
            return EXIT_INVALID
        frozen = json.loads(config_path.read_text(encoding="utf-8"))
        out_path = Path(args.out or "Document/Update/Alpha/v3.7.0/data/d7-formal-eval.json")
        if out_path.is_file() and not args.force:
            print(
                f"{out_path} already exists; the formal split is written once (use --force only after "
                "documenting why the earlier run is void)",
                file=sys.stderr,
            )
            return EXIT_INVALID
        frozen_seeds = frozen["evaluation"]
        scenarios = []
        for family in frozen_seeds["family_subset"]:
            for seed in frozen_seeds["formal_seed_indices"]:
                scenarios.append(families.build(family, int(seed)))
        edges = families.edge_instances()
        for edge_id in frozen_seeds["edges"]:
            scenarios.append(edges[edge_id])
        subject = controller_module.ControllerConfig(
            planner=planner_module.PlannerConfig(**frozen["planning"]),
            **{key: value for key, value in frozen["controller"].items()},
        )
        payloads = [(scenario, frozen_seeds["reference_budget"]) for scenario in scenarios]
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            labels = dict(pool.map(evaluate_module.label_one, payloads))
        document = evaluate_module.evaluate(
            scenarios, subject_config=subject, reference_status=labels, workers=args.workers
        )
        sparse_rows = [row for row in document["rows"] if row["family"] == "SC-17"]
        sparse_success = (
            sum(1 for row in sparse_rows if row["subject"]["survived"]) / len(sparse_rows)
            if sparse_rows
            else None
        )
        # A-03 labels for every closing instance in the split (not a fixed dev id).
        closing = [
            scenario
            for scenario in scenarios
            if scenario.family in ("SC-15", "SC-16") or scenario.scenario_id in ("E-05", "E-06")
        ]
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                latest_safe = dict(pool.map(evaluate_module.label_safe, [(item,) for item in closing]))
        else:
            latest_safe = dict(
                evaluate_module.label_safe((item,)) for item in closing
            )
        document["gates"] = evaluate_module.gates(
            document, sparse_success=sparse_success, labels=latest_safe
        )
        document["split"] = "formal"
        document["config"] = str(config_path)
        document["scenarios"] = [row["scenario_id"] for row in document["rows"]]
        evaluate_module.write(document, out_path)
        print(f"wrote {out_path} (formal, {len(document['rows'])} scenarios)")
        for gate, payload in document["gates"].items():
            print(f"  {gate}: {payload['status']}")
        return EXIT_OK
    if args.split != "dev":
        print(
            "only the dev split is implemented so far; the formal split must be frozen first (D7)",
            file=sys.stderr,
        )
        return EXIT_NOT_IMPLEMENTED
    scenarios = []
    seeds = range(max(1, args.dev_seeds))
    for family in ("SC-02", "SC-03", "SC-08", "SC-09", "SC-15", "SC-16", "SC-17"):
        for seed in seeds:
            scenarios.append(families.build(family, int(seed)))
    edges = families.edge_instances()
    for edge_id in ("E-01", "E-02", "E-05", "E-06", "E-07", "E-08", "E-09", "E-10", "E-11", "E-12", "E-16", "E-21", "E-22"):
        scenarios.append(edges[edge_id])
    sparse = [families.build("SC-17", seed) for seed in range(1, 4) if seed not in seeds]
    payloads = [(scenario, args.reference_budget) for scenario in scenarios + sparse]
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            labels = dict(pool.map(evaluate_module.label_one, payloads))
    else:
        labels = dict(evaluate_module.label_one(payload) for payload in payloads)
    # A-03 labels for every closing instance in the dev split, not a fixed id.
    closing = [
        scenario
        for scenario in scenarios + sparse
        if scenario.family in ("SC-15", "SC-16") or scenario.scenario_id in ("E-05", "E-06")
    ]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            latest_safe = dict(pool.map(evaluate_module.label_safe, [(item,) for item in closing]))
    else:
        latest_safe = dict(evaluate_module.label_safe((item,)) for item in closing)
    config = controller_module.ControllerConfig(
        planner=planner_module.PlannerConfig(
            horizon_decisions=args.plan_horizon,
            continuation_decisions=args.plan_continuation,
            beam_width=1,
        )
    )
    document = evaluate_module.evaluate(
        scenarios + sparse,
        subject_config=config,
        reference_status=labels,
        workers=args.workers,
    )
    sparse_rows = [row for row in document["rows"] if row["family"] == "SC-17"]
    sparse_success = (
        sum(1 for row in sparse_rows if row["subject"]["survived"]) / len(sparse_rows)
        if sparse_rows
        else None
    )
    document["gates"] = evaluate_module.gates(document, sparse_success=sparse_success, labels=latest_safe)
    document["split"] = "dev"
    document["reference_budget"] = args.reference_budget
    document["scenarios"] = [row["scenario_id"] for row in document["rows"]]
    default_out = (
        f"Document/Update/Alpha/v3.7.0/data/d7-dev-eval-12x2-seeds{max(1, args.dev_seeds)}.json"
    )
    out = Path(args.out or default_out)
    evaluate_module.write(document, out)
    print(f"wrote {out}")
    for gate, payload in document["gates"].items():
        print(f"  {gate}: {payload['status']} {({k: v for k, v in payload.items() if k not in ('status', 'note')})}")
    return EXIT_OK


def cmd_live(args: argparse.Namespace) -> int:
    """Live shadow harness (D8/D9).

    With ``--trace`` this is a *dry run*: recorded observations are fed through the
    exact live loop, so the code path (freshness check, frame-counter check,
    completeness ratio) is exercised without the game.  Without ``--trace`` the same
    command is what runs against the live adapter once it is wired.
    """

    from . import controller as controller_module
    from . import live as live_module
    from . import probe as probe_module
    from . import replay_source

    planner_config = controller_module.ControllerConfig(
        planner=controller_module.planner.PlannerConfig(
            horizon_decisions=args.plan_horizon,
            continuation_decisions=args.plan_continuation,
            beam_width=1,
        )
    )
    if not args.trace:
        print("live: no --trace and no game adapter wired; this is the D8/D9 entry point")
        return EXIT_INVALID

    source = replay_source.RecordedSource(args.trace, start=args.start, limit=args.decisions)
    observations = []
    while len(observations) < args.decisions:
        observation = source.read()
        # The source is a cursor: without advance() every read returns the same
        # record (which made the frame-counter check see a constant frame).
        source.advance()
        if observation is None:
            break
        observations.append(observation)
    if not observations:
        print(f"live: no readable observations in {args.trace}")
        return EXIT_FAILURE

    index = {"next": 0}

    def read():
        position = index["next"]
        index["next"] = position + 1
        return observations[position] if position < len(observations) else None

    document = live_module.run_live(
        read,
        decisions=len(observations),
        config=probe_module.ProbeConfig(decisions=len(observations), planner=planner_config),
    )
    document["source"] = args.trace
    document["dry_run"] = True
    out = args.out or "Document/Update/Alpha/v3.7.0/data/live-dryrun.json"
    live_module.write(document, out)
    print(f"wrote {out}")
    print(
        f"  decisions {document['decisions']} stale {document['stale_reads']} "
        f"complete {document['complete_ratio']} same_frame {document['same_frame_verified']}"
    )
    print(
        f"  intents {document['intents']} p50 {document['plan_ms_p50']} p95 {document['plan_ms_p95']} "
        f"misses {document['deadline_misses']}"
    )
    print(f"  B-05 {live_module.b05_check(document)['status']} B-07 {live_module.b07_check(document)['status']}")
    return EXIT_OK


def cmd_probe(args: argparse.Namespace) -> int:
    from . import controller as controller_module
    from . import paired as paired_module
    from . import probe as probe_module

    scenario = _select_scenario(args)
    config = probe_module.ProbeConfig(
        decisions=args.decisions,
        survey=args.survey,
        planner=controller_module.ControllerConfig(
            planner=planner_module_config(args),
        ),
    )
    document = probe_module.run_probe(probe_module.ReplaySource(scenario), scenario, config=config)
    out = Path(args.out or f"Document/Update/Alpha/v3.7.0/data/d8-probe-{scenario.scenario_id.replace('/', '-')}.json")
    probe_module.write_probe(document, out)
    print(f"wrote {out} ({document['decisions']} decisions, readable {document['readable']})")
    if document.get("paired"):
        print(
            f"  paired: safe_choice={document['paired']['safe_choice_rate']} "
            f"missed={document['paired']['missed_rate']} doomed={document['paired']['doomed_rate']}"
        )
    if args.trace_out:
        from . import trace as trace_module

        trace_module.record_episode(
            scenario,
            out_dir=Path(args.trace_out),
            config=config.planner,
        )
        print(f"  trace: {args.trace_out}")
    return EXIT_OK


def cmd_experiment(args: argparse.Namespace) -> int:
    """v3.7.1 experiment driver: gates, single episodes and paired sessions.

    ``gates`` needs no game (deterministic fake clock) and is the command a
    reviewer can run to check C-01..C-04 for themselves.  ``run``/``pair`` do need
    the game and refuse to touch the window without ``--confirm-foreground``,
    because SendInput steals keyboard focus from whoever is using the machine.
    """

    from . import experiment as experiment_module

    if args.mode == "gates":
        document = experiment_module.run_fake_gates()
        out = experiment_module.write(
            document,
            args.out or "Document/Update/Alpha/v3.7.1/data/c01-c04-gates.json",
        )
        print(f"wrote {out}")
        for name, status in document["status"].items():
            print(f"  {name}: {status}")
        print(f"  overall: {'pass' if document['passed'] else 'fail'}")
        return EXIT_OK if document["passed"] else EXIT_INVALID

    if not args.confirm_foreground:
        print(
            "refusing to drive the game without --confirm-foreground: SendInput takes the "
            "foreground window and will type into whatever the owner of this machine is doing",
            file=sys.stderr,
        )
        return EXIT_INVALID

    from ..env import EnvConfig, Th08Env
    from . import arms as arms_module
    from . import controller as controller_module
    from . import harness as harness_module

    controller_config = controller_module.ControllerConfig(
        planner=controller_module.planner.PlannerConfig(
            horizon_decisions=args.horizon,
            continuation_decisions=args.continuation,
            beam_width=args.beam,
        ),
        deadline_ms=args.deadline_ms,
    )
    config = harness_module.HarnessConfig(
        frames=args.frames,
        frameskip=args.frameskip,
        actuation_lag_frames=args.actuation_lag,
        inject_delay_ms=args.inject_delay_ms,
        wall_budget_s=args.wall_budget,
        record_frames=not args.no_frames,
    )
    env_config = EnvConfig(
        run_dir=Path(args.run_dir),
        practice_stage=args.stage,
        difficulty=args.difficulty,
        frameskip=args.frameskip,
        windowed=True,
        use_postmessage_input=False,
    )
    env = Th08Env(env_config)
    try:
        if args.mode == "run":
            arm = arms_module.make_arm(args.arm, controller_config=controller_config)
            document = experiment_module.run_live_episode(
                env,
                arm,
                config=config,
                episode_id=args.episode_id or f"{args.arm}-1",
                index=1,
            )
            out = experiment_module.write(document, args.out or "Document/Update/Alpha/v3.7.1/data/live-run.json")
            print(f"wrote {out}")
            print(
                f"  arm={document['arm']} exposure={document['exposure_frames']} "
                f"decisions={document['decisions']} deaths={document['deaths_in_window']} "
                f"termination={document['termination']}"
            )
            return EXIT_OK if document["technical_failure"] is None else EXIT_INVALID

        document = experiment_module.run_pair_session(
            env,
            pairs=args.pairs,
            order_scheme=args.order,
            seed=args.seed,
            config=config,
            arm_names=(args.first_arm, args.second_arm),
            arm_factory=lambda name: arms_module.make_arm(
                name,
                controller_config=controller_config,
                delay_ms=args.delay_ms,
            ),
            progress=lambda index, entry: print(index, entry, flush=True),
            wall_budget_s=args.episode_budget,
        )
        out = experiment_module.write(document, args.out or "Document/Update/Alpha/v3.7.1/data/live-pairs.json")
        print(f"wrote {out}")
        return EXIT_OK
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass


def cmd_student(args: argparse.Namespace) -> int:
    from . import controller as controller_module
    from . import student as student_module

    config = student_module.StudentConfig(
        episodes=args.episodes,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        planner=controller_module.ControllerConfig(
            planner=planner_module_config(args),
        ),
    )
    document = student_module.train_and_evaluate(config)
    out = Path(args.out or "Document/Update/Alpha/v3.7.0/data/d10-student.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({document['samples']} samples, {document['episodes']} episodes)")
    print(f"  train accuracy={document['train']['accuracy']} test accuracy={document['test']['accuracy']}")
    for loop in document["closed_loop"]:
        print(f"  loop {loop['scenario']}: survived={loop['survived']} frames={loop['frames']}")
    return EXIT_OK


def _stub(command: str) -> Callable[[argparse.Namespace], int]:
    def run(_args: argparse.Namespace) -> int:
        print(
            f"'{command}' is not implemented: planned for {STUBS[command]}. "
            "This package is at R1/D0 (contract and baseline freeze) only.",
            file=sys.stderr,
        )
        return EXIT_NOT_IMPLEMENTED

    return run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m touhou_rl.escape",
        description=f"v3.7.0 escape tooling -- {PACKAGE_STAGE}",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    schema = sub.add_parser("schema", help="print or write the generated JSON schema")
    schema.add_argument("--out", help="write here instead of stdout")
    schema.set_defaults(func=cmd_schema)

    validate = sub.add_parser("validate", help="validate one document or an array of documents")
    validate.add_argument("path")
    validate.set_defaults(func=cmd_validate)

    fixture = sub.add_parser(
        "baseline-fixture",
        help="write or verify the digest-pinned old-pilot fixture",
    )
    fixture.add_argument(
        "--out",
        default="Document/Update/Alpha/v3.7.0/fixtures/baseline-golden.json",
        help="fixture path",
    )
    fixture.add_argument("--check", action="store_true", help="verify instead of write")
    fixture.set_defaults(func=cmd_baseline_fixture)

    snap = sub.add_parser("snapshot", help="write a code/worktree hash snapshot")
    snap.add_argument("--out", default="Document/Update/Alpha/v3.7.0/data/manifest.json")
    snap.add_argument("--root", default=".")
    snap.set_defaults(func=cmd_snapshot)

    evidence = sub.add_parser("evidence", help="hash a comma-separated artifact list")
    evidence.add_argument("--base", required=True)
    evidence.add_argument("--out", required=True)
    evidence.add_argument("--include", required=True)
    evidence.set_defaults(func=cmd_evidence)

    audit_traces = sub.add_parser(
        "audit-traces",
        help="read-only audit of recorded JSONL traces (D0 evidence)",
    )
    audit_traces.add_argument("paths", nargs="+")
    audit_traces.add_argument("--out")
    audit_traces.add_argument(
        "--max-decisions", type=int, default=None, help="stop after N decisions per episode"
    )
    audit_traces.set_defaults(func=cmd_audit_traces)

    seed_parser = sub.add_parser("seeds", help="write the pre-registered scenario seed lists")
    seed_parser.add_argument("--out", default="Document/Update/Alpha/v3.7.0/preregistration/seeds-v1.json")
    seed_parser.add_argument("--master", type=int, default=37000)
    seed_parser.set_defaults(func=cmd_seeds)

    shadow_parser = sub.add_parser(
        "shadow",
        help="re-check recorded decisions with the exact AABB sweep (D1)",
    )
    shadow_parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("TRACE", "SIDECAR"),
        required=True,
        help="one JSONL trace and its NPZ bullet sidecar (repeatable)",
    )
    shadow_parser.add_argument("--out")
    shadow_parser.add_argument("--max-decisions", type=int, default=None)
    shadow_parser.add_argument("--margin", type=float, default=0.0)
    shadow_parser.add_argument("--examples", type=int, default=12)
    shadow_parser.add_argument(
        "--skip-overrides",
        action="store_true",
        help="skip decisions where a tactic replaced the pilot's action",
    )
    shadow_parser.set_defaults(func=cmd_shadow)

    for name, stage in STUBS.items():
        stub = sub.add_parser(name, help=f"not implemented (planned: {stage})")
        stub.set_defaults(func=_stub(name))

    synth_parser = sub.add_parser("synth", help="build and describe a synthetic scenario (D2)")
    synth_parser.add_argument("--family", choices=None)
    synth_parser.add_argument("--seed", type=int, default=0)
    synth_parser.add_argument("--edge")
    synth_parser.add_argument("--list", action="store_true")
    synth_parser.add_argument("--out")
    synth_parser.set_defaults(func=cmd_synth)

    reference_parser = sub.add_parser(
        "reference", help="run the independent reference solver (D2)"
    )
    reference_parser.add_argument("--family")
    reference_parser.add_argument("--seed", type=int, default=0)
    reference_parser.add_argument("--edge")
    reference_parser.add_argument("--budget", type=int, default=4000)
    reference_parser.add_argument("--sweep", action="store_true")
    reference_parser.add_argument("--out")
    reference_parser.set_defaults(func=cmd_reference)

    forecast_parser = sub.add_parser(
        "forecast", help="build the time-indexed danger forecast for a scenario (D3)"
    )
    forecast_parser.add_argument("--family")
    forecast_parser.add_argument("--seed", type=int, default=0)
    forecast_parser.add_argument("--edge")
    forecast_parser.add_argument("--horizon", type=int, default=180)
    forecast_parser.add_argument("--inflation", type=float, default=0.0)
    forecast_parser.add_argument("--cell", type=float, default=16.0)
    forecast_parser.add_argument("--max-entities", type=int, default=1024)
    forecast_parser.add_argument("--benchmark", type=int, default=0, metavar="RUNS")
    forecast_parser.add_argument("--out")
    forecast_parser.set_defaults(func=cmd_forecast)

    record_parser = sub.add_parser("record", help="run the controller and write a replayable trace (D6)")
    record_parser.add_argument("--family")
    record_parser.add_argument("--seed", type=int, default=0)
    record_parser.add_argument("--edge")
    record_parser.add_argument("--out", required=True)
    record_parser.add_argument("--plan-horizon", type=int, default=12)
    record_parser.add_argument("--plan-continuation", type=int, default=2)
    record_parser.add_argument("--beam", type=int, default=1)
    record_parser.add_argument("--horizon", type=int, default=180)
    record_parser.add_argument("--max-decisions", type=int, default=None)
    record_parser.set_defaults(func=cmd_record)

    render_parser = sub.add_parser("render", help="render a recorded trace (text or layered PNG)")
    render_parser.add_argument("--trace", required=True)
    render_parser.add_argument("--sidecar")
    render_parser.add_argument("--out")
    render_parser.add_argument("--decision", type=int, default=None)
    render_parser.add_argument("--text-only", action="store_true")
    render_parser.set_defaults(func=cmd_render)

    evaluate_parser = sub.add_parser("evaluate", help="dev-set evaluation and gate arithmetic (D7)")
    evaluate_parser.add_argument("--split", default="dev", choices=("dev", "formal"))
    evaluate_parser.add_argument("--out", default=None)
    evaluate_parser.add_argument("--reference-budget", type=int, default=4000)
    evaluate_parser.add_argument("--plan-horizon", type=int, default=12)
    evaluate_parser.add_argument("--plan-continuation", type=int, default=2)
    evaluate_parser.add_argument("--workers", type=int, default=8)
    evaluate_parser.add_argument(
        "--config", default="Document/Update/Alpha/v3.7.0/configs/formal-v1.json"
    )
    evaluate_parser.add_argument("--force", action="store_true")

    probe_parser = sub.add_parser("probe", help="read-only shadow probe (D8/D9 skeleton, offline source)")
    probe_parser.add_argument("--family")
    probe_parser.add_argument("--seed", type=int, default=0)
    probe_parser.add_argument("--edge")
    probe_parser.add_argument("--decisions", type=int, default=40)
    probe_parser.add_argument("--survey", action="store_true", help="privileged 18-action survey per decision")
    probe_parser.add_argument("--out", default=None)
    probe_parser.add_argument("--trace-out", default=None)
    probe_parser.add_argument("--plan-horizon", type=int, default=12)
    probe_parser.add_argument("--plan-continuation", type=int, default=2)
    probe_parser.add_argument("--beam", type=int, default=1)
    probe_parser.set_defaults(func=cmd_probe)

    live_parser = sub.add_parser(
        "live",
        help="live shadow harness (D8/D9): same loop as the recorded probe, driven by an adapter",
    )
    live_parser.add_argument(
        "--trace",
        default=None,
        help="dry run: feed a recorded trace through the live loop (exercises the live code path)",
    )
    live_parser.add_argument("--start", type=int, default=280)
    live_parser.add_argument("--decisions", type=int, default=60)
    live_parser.add_argument("--out", default=None)
    live_parser.add_argument("--plan-horizon", type=int, default=12)
    live_parser.add_argument("--plan-continuation", type=int, default=2)
    live_parser.set_defaults(func=cmd_live)

    experiment_parser = sub.add_parser(
        "experiment",
        help="v3.7.1 clock/exposure gates and the single-pipeline paired run",
    )
    experiment_parser.add_argument("mode", choices=("gates", "run", "pair"))
    experiment_parser.add_argument("--out", default=None)
    experiment_parser.add_argument(
        "--frames", type=int, default=1800, help="logical-frame window length (the stop rule)"
    )
    experiment_parser.add_argument("--frameskip", type=int, default=2)
    experiment_parser.add_argument(
        "--actuation-lag",
        type=int,
        default=1,
        help="frames between observation and key write; 0 reproduces the v3.7.0 confound",
    )
    experiment_parser.add_argument("--inject-delay-ms", type=float, default=0.0)
    experiment_parser.add_argument("--wall-budget", type=float, default=180.0)
    experiment_parser.add_argument("--episode-budget", type=float, default=120.0)
    experiment_parser.add_argument("--arm", default="subject", choices=("baseline", "subject", "noop"))
    experiment_parser.add_argument(
        "--first-arm",
        default="baseline",
        choices=("baseline", "subject", "hybrid_veto", "delayed_baseline", "noop"),
        help="condition A of the paired session",
    )
    experiment_parser.add_argument(
        "--second-arm",
        default="hybrid_veto",
        choices=("baseline", "subject", "hybrid_veto", "delayed_baseline", "noop"),
        help="condition B of the paired session",
    )
    experiment_parser.add_argument(
        "--delay-ms",
        type=float,
        default=0.0,
        help="planning cost injected into delayed_baseline (the C-04 latency control)",
    )
    experiment_parser.add_argument("--episode-id", default=None)
    experiment_parser.add_argument("--pairs", type=int, default=30)
    experiment_parser.add_argument("--order", default="balanced_random", choices=("ab", "ba", "balanced_random"))
    experiment_parser.add_argument("--seed", type=int, default=0)
    experiment_parser.add_argument("--stage", type=int, default=0)
    experiment_parser.add_argument("--difficulty", type=int, default=3)
    experiment_parser.add_argument("--horizon", type=int, default=6)
    experiment_parser.add_argument("--continuation", type=int, default=1)
    experiment_parser.add_argument("--beam", type=int, default=1)
    experiment_parser.add_argument("--deadline-ms", type=float, default=33.3)
    experiment_parser.add_argument("--run-dir", default=".tmp/v371-session")
    experiment_parser.add_argument("--no-frames", action="store_true", help="do not store the per-frame series")
    experiment_parser.add_argument(
        "--confirm-foreground",
        action="store_true",
        help="acknowledge that SendInput will take the foreground window",
    )
    experiment_parser.set_defaults(func=cmd_experiment)

    student_parser = sub.add_parser("student", help="train/evaluate the vision student on synthetic frames (D10–D12)")
    student_parser.add_argument("--episodes", type=int, default=12)
    student_parser.add_argument("--epochs", type=int, default=12)
    student_parser.add_argument("--seed", type=int, default=7)
    student_parser.add_argument("--device", default="auto")
    student_parser.add_argument("--out", default=None)
    student_parser.add_argument("--plan-horizon", type=int, default=4)
    student_parser.add_argument("--plan-continuation", type=int, default=1)
    student_parser.add_argument("--beam", type=int, default=1)
    student_parser.set_defaults(func=cmd_student)
    evaluate_parser.add_argument("--dev-seeds", type=int, default=1)
    evaluate_parser.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover - exercised through ``main``
    raise SystemExit(main())
