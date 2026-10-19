# S3-6 CPU CLI evidence: run every real command, keep stdout/stderr/exit separately.
#
# Usage (from the workspace root):
#   $py = "C:\Users\a5935\AppData\Local\Programs\Python\Python312\python.exe"
#   $env:PYTHONPATH = "D:\WorkShop\Touhou_RL"
#   pwsh -File Document\Update\Alpha\v1.0.4\Stage3\S3-6\evidence\run_cli_evidence.ps1 `
#        -Python $py -Fixtures D:\WorkShop\Touhou_RL\.pytest-tmp\S3-6\cli-fixtures
#
# Prerequisites: `make_synthetic_fixtures.py --out <Fixtures> --clean` has run.
# Every fixture is synthetic; a 0/1/2 here proves the tool's verdict logic and the
# exit-code contract, not that v1.0.4 passed any acceptance item.

param(
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$Fixtures,
    [string]$Evidence = ""
)

$ErrorActionPreference = "Continue"
$rows = @()
if (-not $Evidence) { $Evidence = $PSScriptRoot }
$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..\..\..\..")).Path
$env:PYTHONPATH = $workspace

function Write-Utf8NoBom {
    param([string]$Path, [string[]]$Lines)
    $text = ($Lines -join "`r`n") + "`r`n"
    [System.IO.File]::WriteAllText($Path, $text, (New-Object System.Text.UTF8Encoding($false)))
}

function Invoke-Evidence {
    param([string]$Name, [string[]]$Command)
    $outFile = Join-Path $Evidence "$Name.out"
    $errFile = Join-Path $Evidence "$Name.err"
    $logFile = Join-Path $Evidence "$Name.log"
    # Start-Process redirects the *raw* streams; PowerShell's own "2>" would wrap
    # native stderr in a NativeCommandError record and pollute the evidence.
    $quoted = $Command | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
    $process = Start-Process -FilePath $Python -ArgumentList $quoted -Wait -PassThru -NoNewWindow `
        -RedirectStandardOutput $outFile -RedirectStandardError $errFile
    $code = $process.ExitCode
    $lines = @()
    $lines += "COMMAND: $Python " + ($Command -join " ")
    $lines += "EXIT: $code"
    $lines += "--- STDOUT ---"
    if (Test-Path $outFile) { $lines += Get-Content $outFile }
    $lines += "--- STDERR ---"
    if (Test-Path $errFile) { $lines += Get-Content $errFile }
    Write-Utf8NoBom -Path $logFile -Lines $lines
    Write-Host ("[{0}] exit={1}" -f $Name, $code)
    $script:rows += [pscustomobject]@{ name = $Name; exit = $code }
    return $code
}

$cmp = Join-Path $Fixtures "compare"
$check = Join-Path $Fixtures "check"
$a = Join-Path $cmp "dataset-a"
$b = Join-Path $cmp "dataset-b"
$mb = Join-Path $b "manifest-b.json"
$ck = Join-Path $cmp "state-depth3.pt"
$v3 = Join-Path $cmp "v3-stub.pt"
$tool = (Resolve-Path (Join-Path $workspace "tools\rl\compare_state_models.py")).Path
$checker = (Resolve-Path (Join-Path $workspace "tools\rl\check_v104.py")).Path

# 1-3: the two sampling modes, and the A/B A-03 ratio with the S3-1-style manifest.
Invoke-Evidence "10-compare-full-val" @(
    $tool, "--dataset", $a, "--mode", "full-val",
    "--checkpoint", "depth3=$ck", "--out", (Join-Path $Evidence "compare-full-val.json")
) | Out-Null
Invoke-Evidence "11-compare-a-stage-sample" @(
    $tool, "--dataset", $a, "--mode", "stage-sample", "--per-stage", "2",
    "--checkpoint", "depth3=$ck", "--out", (Join-Path $Evidence "compare-a-stage-sample.json")
) | Out-Null
Invoke-Evidence "12-compare-b-stage-sample" @(
    $tool, "--dataset", $b, "--manifest", $mb, "--mode", "stage-sample", "--per-stage", "2",
    "--checkpoint", "depth3=$ck",
    "--baseline", (Join-Path $Evidence "compare-a-stage-sample.json"),
    "--out", (Join-Path $Evidence "compare-b-stage-sample.json")
) | Out-Null

# 4: B without its read-only manifest must be refused, never re-split.
Invoke-Evidence "13-compare-missing-manifest" @(
    $tool, "--dataset", $b, "--manifest", (Join-Path $b "does-not-exist.json"),
    "--mode", "stage-sample", "--checkpoint", "depth3=$ck"
) | Out-Null

# 5: a shard whose bytes changed after the manifest was written.
$tampered = Join-Path $Fixtures "compare\dataset-b-tampered"
if (Test-Path $tampered) { Remove-Item $tampered -Recurse -Force }
Copy-Item $b $tampered -Recurse
$victim = Get-ChildItem $tampered -Filter "collect-*.npz" | Sort-Object Name | Select-Object -First 1
Add-Content -Path $victim.FullName -Value "0" -NoNewline
Invoke-Evidence "14-compare-tampered-shard" @(
    $tool, "--dataset", $tampered, "--manifest", (Join-Path $tampered "manifest-b.json"),
    "--mode", "stage-sample", "--checkpoint", "depth3=$ck"
) | Out-Null

# 6: a rewritten validation index hash (shards intact) must be caught by the split check.
$bad = Join-Path $cmp "dataset-a-badmanifest"
Invoke-Evidence "15-compare-tampered-manifest" @(
    $tool, "--dataset", $bad, "--mode", "stage-sample", "--checkpoint", "depth3=$ck"
) | Out-Null

# 7: an unknown checkpoint arch must be rejected, not coerced to v1.
$unknown = Join-Path $cmp "unknown-arch.pt"
Invoke-Evidence "16-compare-unknown-arch" @(
    $tool, "--dataset", $a, "--mode", "stage-sample", "--checkpoint", "unknown=$unknown"
) | Out-Null

# 8: a real v3-st checkpoint (G-v3 approved) must dispatch and evaluate, not block.
$v3 = Join-Path $cmp "v3-policy.pt"
Invoke-Evidence "17-compare-v3-dispatch" @(
    $tool, "--dataset", $a, "--mode", "stage-sample", "--per-stage", "2",
    "--checkpoint", "v3=$v3", "--out", (Join-Path $Evidence "compare-v3-stage-sample.json")
) | Out-Null

# 8b: a stale v3 payload whose config layout is not the S3-5 one must be rejected.
$v3bad = Join-Path $cmp "v3-badconfig.pt"
Invoke-Evidence "17b-compare-v3-badconfig" @(
    $tool, "--dataset", $a, "--mode", "stage-sample", "--checkpoint", "v3bad=$v3bad"
) | Out-Null

# 9: evaluate --vision-depth conflict -> stderr prefix + exit 2 (old behaviour was exit 1).
$evalOut = Join-Path $Fixtures "eval-depth-mismatch"
Invoke-Evidence "18-evaluate-depth-mismatch" @(
    "-m", "touhou_rl.evaluate", "--vision-checkpoint", $ck, "--vision-depth", "1",
    "--device", "cpu", "--episodes", "1", "--out", $evalOut
) | Out-Null

# 10-13: the check_v104 exit paths on the same structure (names are sequential; the
# exit code of each run is recorded in cli-exit-summary.json).
$variants = @("pass", "fail", "zerobucket", "blocked")
for ($index = 0; $index -lt $variants.Count; $index++) {
    $variant = $variants[$index]
    Invoke-Evidence ("2{0}-check-v104-{1}" -f $index, $variant) @(
        $checker, "--artifacts", (Join-Path $check (Join-Path $variant "final-artifacts.json")),
        "--root", $check, "--out", (Join-Path $Evidence "check-v104-$variant.json")
    ) | Out-Null
}

$summary = $rows | ConvertTo-Json
Write-Utf8NoBom -Path (Join-Path $Evidence "cli-exit-summary.json") -Lines ($summary -split "`r?`n")
Write-Host $summary
exit 0
