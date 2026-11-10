# EXP-P0-014 WP1 retail capture diagnostic

## Scope and guardrails

- Provider/model: DeepSeek / DeepSeek v4.1 flash.
- Only authorized diagnostic artifacts were used/created under `.tmp_v400_retail_*`; the original executable, game data, official collector, and fixture inputs were read-only.
- No replay trajectory was fabricated. The second fixture was not attempted because the first case did not produce a single capture row.

## Attempt

Command (WSL, bounded):

```text
wsl.exe -e bash -lc 'set +e; out=/mnt/d/WorkShop/Touhou_RL/.tmp_v400_retail_diag_easy0_display0; rm -rf "$out"; prefix=/mnt/d/WorkShop/Touhou_RL/.tmp_v400_retail_diag_easy0_prefix; rm -rf "$prefix"; export WINEPREFIX="$prefix" WINEARCH=win32 DISPLAY=:0 WINEDEBUG=-all; timeout 150s python3 /mnt/d/WorkShop/Touhou_RL/tools/rl/v400_retail_py310_launcher.py --target /mnt/d/WorkShop/Touhou_RL/third_party/th08/resources/th08.exe --game-data /root/v400-data/game/th08.dat --bgm-data /root/v400-data/game/thbgm.dat --replay /mnt/d/WorkShop/Touhou_RL/.tmp_v400_retail_fixtures/th8_ud2cdc.rpy --output-dir "$out" --wine /usr/bin/wine --observer-backend ptrace --display :0 --timeout 120 --keep-runtime'
```

Observed shell result: `CAPTURE_EXIT=124` (outer `timeout` killed after 150 seconds). This was an independent `WINEARCH=win32` prefix and explicit WSLg `DISPLAY=:0`. The official collector reached metadata/runtime initialization, so target and replay identity checks passed:

- target SHA-256: `330fbdbf58a710829d65277b4f312cfbb38d5448b3df523e79350b879213d924`
- replay `easy-0-b` SHA-256: `79739491e1cb1d735ef9a14814a79dfe8e2c4872d5f2bf0174c79f40ada36989`
- Wine: `wine-6.0.3 (Ubuntu 6.0.3~repack-1)`
- Python: `3.10.12`
- backend: Linux ptrace hardware execute breakpoints
- observer attestation and replay metadata were emitted.

## Failure evidence

Artifact directory: `.tmp_v400_retail_diag_easy0_display0/`

- `rows.jsonl`: **0 bytes, 0 rows**, SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `complete.json`: **absent**
- all event streams (`menu`, `rng`, `score`, `item`, `effect`): 0 bytes
- `metadata.json` and `observer.json` exist and attest the expected retail executable/replay.

The run therefore did not reach the observer capture boundary or completion boundary. This is a bounded diagnostic failure, not a successful capture and not a valid trajectory. No second fixture was run.

## Diagnosis / precise block

The requested independent-prefix + official `--display :0` path remains blocked: Wine32 starts enough to emit runtime metadata, but under this WSLg display the ptrace observer never reaches the first capture breakpoint within 120 seconds; the bounded wrapper exits 124, with zero rows and no `complete.json`. Existing private-Xvfb and earlier `:0` attempts likewise had zero rows. Further unbounded retries are prohibited. To unblock, a human must provide a verified compatible X/Win32 GUI/debug environment (or diagnose why Wine 6.0.3 + WSLg ptrace cannot execute/observe this binary); then rerun the same official command and require nonzero rows, `complete.json`, row SHA, and official suite verification before attempting fixture two.
