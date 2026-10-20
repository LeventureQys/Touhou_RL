"""Check the numbers §11.11 quotes about the CLI switchboard and the new tests."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[7]
TRAINER = ROOT / "touhou_rl" / "vision_train.py"
TESTS = ROOT / "touhou_rl" / "tests" / "test_v104_refine.py"
NEW_TESTS = (
    "test_cli_seed_reaches_the_config",
    "test_cli_flags_are_not_silently_ignored",
    "test_every_cli_flag_is_read_when_building_the_config",
    "test_seed_is_recorded_in_the_summary_and_recipe",
)


def main() -> int:
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
    main_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    flags = []
    for node in ast.walk(main_fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            for argument in node.args:
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    if argument.value.startswith("--"):
                        flags.append(argument.value)
    source = TESTS.read_text(encoding="utf-8")
    start = source.index("def test_cli_flags_are_not_silently_ignored")
    end = source.index("def test_every_cli_flag_is_read_when_building_the_config")
    table_cases = len(re.findall(r"^\s*\(\[", source[start:end], re.MULTILINE))
    present = {name: f"def {name}" in source for name in NEW_TESTS}
    line = next(
        index
        for index, text in enumerate(TRAINER.read_text(encoding="utf-8").splitlines(), start=1)
        if "seed=args.seed" in text
    )
    print(f"CLI flags: {len(flags)}")
    print(f"wiring-table cases: {table_cases}")
    print(f"new tests present: {present}")
    print(f"seed=args.seed at line: {line}")
    ok = len(flags) == 49 and table_cases == 24 and all(present.values())
    print("OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
