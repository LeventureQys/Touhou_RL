<#
.SYNOPSIS
    One-command verification for the v1.0.2 TH08 RL framework.

.DESCRIPTION
    Steps (each recorded in third_party/th08/build/rl/verify-report.json):
      1. python import check for every runtime module
      2. unit tests (pytest)
      3. repository hygiene: third_party/th08 must stay untouched
      4. optional live checks (-SkipGame omits them):
           L-1 replay capture smoke test (3 minutes into a temp dataset)
           L-2 env smoke test (reset + 120 random steps)

.PARAMETER SkipGame
    Skip the live game tests. Useful on CI or when the machine must stay free.

.PARAMETER Python
    Python executable; defaults to the interpreter used to launch this script.

.NOTES
    Run from the repository root:
        powershell -NoProfile -ExecutionPolicy Bypass -File tools\rl\Verify-RL.ps1
#>
param(
    [switch]$SkipGame,
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
function Test-UsablePython([string]$candidate) {
    if (-not $candidate) { return $false }
    if (-not (Test-Path $candidate)) { return $false }
    if ($candidate -like "*WindowsApps*") { return $false }   # Microsoft Store alias stub
    return $true
}
if (-not (Test-UsablePython $Python)) {
    $Python = ""
    foreach ($candidate in @(
        "C:\Users\a5935\AppData\Local\Programs\Python\Python312\python.exe",
        (Get-Command python -ErrorAction SilentlyContinue).Source,
        (Get-Command python3 -ErrorAction SilentlyContinue).Source
    )) {
        if (Test-UsablePython $candidate) { $Python = $candidate; break }
    }
}
if (-not $Python) {
    Write-Host "No usable Python interpreter found; pass -Python <path>." -ForegroundColor Red
    exit 2
}
$reportDir = Join-Path $root "third_party\th08\build\rl"
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
$reportPath = Join-Path $reportDir "verify-report.json"
$steps = @()
$startedAt = Get-Date

function Invoke-Step {
    param([string]$Name, [scriptblock]$Action)
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $ok = $false
    $output = ""
    try {
        $output = (& $Action 2>&1 | Out-String)
        $ok = $LASTEXITCODE -eq 0 -or $null -eq $LASTEXITCODE
        if ($LASTEXITCODE -ne $null -and $LASTEXITCODE -ne 0) { $ok = $false }
    }
    catch {
        $ok = $false
        $output = "$output`n$($_.Exception.Message)"
    }
    $sw.Stop()
    $trimmed = ($output -split "`n" | Select-Object -First 12) -join " | "
    $script:steps += [pscustomobject]@{
        name = $Name
        ok = $ok
        durationSeconds = [math]::Round($sw.Elapsed.TotalSeconds, 1)
        output = $trimmed.Trim()
    }
    Write-Host ("[{0}] {1} ({2:N1}s)" -f $(if ($ok) { "ok" } else { "FAIL" }), $Name, $sw.Elapsed.TotalSeconds)
    return $ok
}

$allOk = $true

$allOk = (Invoke-Step "imports" {
    & $Python -c "import touhou_rl, touhou_rl.env, touhou_rl.models, touhou_rl.capture_dataset, touhou_rl.bc, touhou_rl.ppo, touhou_rl.evaluate, touhou_rl.play; print('imports ok', touhou_rl.__version__)"
}) -and $allOk

$allOk = (Invoke-Step "unit-tests" {
    & $Python -m pytest touhou_rl/tests -q
}) -and $allOk

$allOk = (Invoke-Step "repo-hygiene" {
    $git = "C:\Program Files\Git\cmd\git.exe"
    if (-not (Test-Path $git)) { Write-Output "git not found"; exit 1 }
    $dirty = & $git -C third_party/th08 status --short
    if ($dirty) { Write-Output ($dirty -join "; "); exit 1 }
    Write-Output "third_party/th08 clean"
}) -and $allOk

if (-not $SkipGame) {
    $allOk = (Invoke-Step "live-capture" {
        & $Python -m touhou_rl.capture_dataset `
            --replay third_party/th08/tests/fixtures/replays/th08-1.00d/accepted/th8_ud2ce5.rpy `
            --out third_party/th08/build/rl/datasets/verify-smoke `
            --tag verify-smoke --max-minutes 3 --shard-size 2048 --overwrite --quiet
    }) -and $allOk

    $allOk = (Invoke-Step "live-env" {
        & $Python -c @"
import json
from touhou_rl.env import EnvConfig, Th08Env
import touhou_rl.spec as spec
env = Th08Env(EnvConfig(team=0, difficulty=spec.LUNATIC, frameskip=2, max_steps=120))
obs, info = env.reset()
frames = [info['state']['frame']]
for _ in range(120):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    frames.append(info['frame'])
    if terminated or truncated:
        break
env.close()
print(json.dumps({'steps': len(frames) - 1, 'frame_start': frames[0], 'frame_end': frames[-1],
                  'monotonic': all(b >= a for a, b in zip(frames, frames[1:]))}))
"@
    }) -and $allOk
}

$report = [pscustomobject]@{
    schema = "th08-rl-verify/1"
    startedAt = $startedAt.ToString("o")
    finishedAt = (Get-Date).ToString("o")
    durationSeconds = [math]::Round(((Get-Date) - $startedAt).TotalSeconds, 1)
    python = $Python
    skipGame = [bool]$SkipGame
    pass = $allOk
    steps = $steps
}
$json = $report | ConvertTo-Json -Depth 6
# PowerShell 5.1's Set-Content -Encoding UTF8 writes a BOM, which breaks strict
# JSON readers (json.load in Python). Write UTF-8 without BOM instead.
[System.IO.File]::WriteAllText($reportPath, $json, (New-Object System.Text.UTF8Encoding($false)))
Write-Host ""
Write-Host ("verify report: {0}" -f $reportPath)
Write-Host ("result: {0}" -f $(if ($allOk) { "PASS" } else { "FAIL" }))
if ($allOk) { exit 0 } else { exit 1 }
