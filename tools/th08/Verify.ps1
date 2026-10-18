#requires -version 5.1
# TH08 v1.0.0 S3-5 entry: end-to-end verification (env -> build -> runtime -> tests -> runs -> hygiene).
param(
    [switch]$SkipBuild,
    [switch]$SkipRun
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'Th08Pipeline.psm1') -Force

$gitExe = 'C:\Program Files\Git\cmd\git.exe'
$ctx = Get-Th08Context
$startedAt = Get-Date
$steps = New-Object System.Collections.ArrayList
$overallPass = $true

function Add-Step {
    param([string]$Name, [bool]$Ok, [int]$ExitCode, [string]$Detail)
    [void]$steps.Add([pscustomobject]@{ name = $Name; ok = $Ok; exitCode = $ExitCode; detail = $Detail })
    Write-Output (('[{0}] {1} :: {2}' -f $(if ($Ok) { 'OK  ' } else { 'FAIL' }), $Name, $Detail))
}

# 1 env
try {
    $target = Assert-Th08Target -Path $ctx.TargetExePath
    $prefix = Assert-Th08Prefix -Path $ctx.PrefixDir
    Add-Step -Name 'env' -Ok $true -ExitCode 0 -Detail ('target=' + $target.Sha256.Substring(0, 16) + '... prefixFiles=' + $prefix.FileCount)
} catch {
    $overallPass = $false
    Add-Step -Name 'env' -Ok $false -ExitCode (Resolve-Th08ExitCode -Message $_.Exception.Message) -Detail $_.Exception.Message
}

# 2 build
if ($SkipBuild) {
    $normalOk = $false
    $bugfixOk = $false
    $detail = ''
    try {
        $normal = Get-Content -LiteralPath (Join-Path $ctx.ReportsDir 'build-normal.json') -Raw | ConvertFrom-Json
        $bugfix = Get-Content -LiteralPath (Join-Path $ctx.ReportsDir 'build-bugfix.json') -Raw | ConvertFrom-Json
        $normalOk = ($normal.exitCode -eq 0 -and $normal.artifact.pe.machine -eq '0x014C')
        $bugfixOk = ($bugfix.exitCode -eq 0 -and $bugfix.artifact.pe.machine -eq '0x014C')
        $detail = 'reused existing reports'
    } catch {
        $detail = 'missing build reports: ' + $_.Exception.Message
    }
    if ($normalOk -and $bugfixOk) { Add-Step -Name 'build' -Ok $true -ExitCode 0 -Detail $detail }
    else { $overallPass = $false; Add-Step -Name 'build' -Ok $false -ExitCode 4 -Detail $detail }
} else {
    try {
        $normal = Invoke-Th08Build -Context $ctx -BuildType normal -Fresh
        $bugfix = Invoke-Th08Build -Context $ctx -BuildType bugfix -Fresh
        $detail = ('normal=' + $normal.artifact.size + 'B/' + $normal.durationSeconds + 's bugfix=' + $bugfix.artifact.size + 'B/' + $bugfix.durationSeconds + 's')
        if ($normal.artifact.sha256 -eq $bugfix.artifact.sha256) {
            $overallPass = $false
            Add-Step -Name 'build' -Ok $false -ExitCode 4 -Detail ($detail + ' but both artifacts share a hash')
        } else {
            Add-Step -Name 'build' -Ok $true -ExitCode 0 -Detail $detail
        }
    } catch {
        $overallPass = $false
        Add-Step -Name 'build' -Ok $false -ExitCode (Resolve-Th08ExitCode -Message $_.Exception.Message) -Detail $_.Exception.Message
    }
}

# 3 runtime
try {
    # score.dat is deliberately not seeded: the game rewrites it at runtime, so a
    # seeded copy would make the assembly-time manifest snapshot go stale.
    $manifest = Initialize-Th08Runtime -Context $ctx -DataDir 'D:\Games\[th08] 东方永夜抄 (日文版)\[th08] 东方永夜抄 (日文版)'
    $check = Assert-Th08Runtime -Context $ctx
    Add-Step -Name 'runtime' -Ok $true -ExitCode 0 -Detail ('files=' + $manifest.files.Count + ' exe=' + $manifest.exe.Sha256.Substring(0, 16) + '...')
} catch {
    $overallPass = $false
    Add-Step -Name 'runtime' -Ok $false -ExitCode (Resolve-Th08ExitCode -Message $_.Exception.Message) -Detail $_.Exception.Message
}

# 4 unit tests
$testScript = Join-Path $ctx.ToolsDir 'tests\Invoke-UnitTests.ps1'
if (Test-Path -LiteralPath $testScript) {
    $testOut = Join-Path $ctx.ReportsDir 'unit-tests-output.txt'
    $testResult = Invoke-Th08CapturedProcess -FilePath 'powershell.exe' `
        -Arguments @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $testScript) `
        -WorkingDirectory $ctx.RepoRoot -StdOutPath $testOut -StdErrPath (Join-Path $ctx.ReportsDir 'unit-tests-error.txt') `
        -TimeoutMinutes 10
    $summary = @($testResult.StdOut -split "`r?`n" | Where-Object { $_ -match '^TOTAL ' } | Select-Object -Last 1)
    $summaryText = ''
    if ($summary.Count -gt 0) { $summaryText = $summary[0] }
    if ($testResult.ExitCode -eq 0) {
        Add-Step -Name 'unit-tests' -Ok $true -ExitCode 0 -Detail $summaryText
    } else {
        $overallPass = $false
        Add-Step -Name 'unit-tests' -Ok $false -ExitCode 6 -Detail ('exit=' + $testResult.ExitCode + ' ' + $summaryText)
    }
} else {
    $overallPass = $false
    Add-Step -Name 'unit-tests' -Ok $false -ExitCode 6 -Detail 'test script missing'
}

# 5/6 runs
if ($SkipRun) {
    Add-Step -Name 'run-gameplay' -Ok $true -ExitCode 0 -Detail 'skipped by request'
    Add-Step -Name 'run-menuexit' -Ok $true -ExitCode 0 -Detail 'skipped by request'
} else {
    foreach ($pair in @(@{ Name = 'run-gameplay'; Profile = 'l2-gameplay' }, @{ Name = 'run-menuexit'; Profile = 'l2-menuexit' })) {
        try {
            $profilePath = Join-Path $ctx.ToolsDir ('profiles\' + $pair.Profile + '.json')
            $report = Invoke-Th08Run -Context $ctx -ProfilePath $profilePath -Tag 'verify'
            Add-Step -Name $pair.Name -Ok $true -ExitCode 0 -Detail ('runId=' + $report.runId + ' exit=' + $report.exit.code + ' method=' + $report.exit.method + ' shots=' + @($report.shots).Count)
        } catch {
            $overallPass = $false
            Add-Step -Name $pair.Name -Ok $false -ExitCode (Resolve-Th08ExitCode -Message $_.Exception.Message) -Detail $_.Exception.Message
        }
    }
}

# 7 repo hygiene
try {
    $gitAvailable = Test-Path -LiteralPath $gitExe
    $submoduleStatus = ''
    $rootStatus = ''
    if ($gitAvailable) {
        $submoduleStatus = (& $gitExe -C $ctx.Th08Root status --short 2>&1 | Out-String).Trim()
        $rootStatus = (& $gitExe -C $ctx.RepoRoot status --short 2>&1 | Out-String).Trim()
    }
    $privateInRoot = @($rootStatus -split "`r?`n" | Where-Object { $_ -match '\.dat$|\.exe$|scripts/prefix|third_party/th08/build' })
    $ok = ($submoduleStatus -eq '') -and ($privateInRoot.Count -eq 0)
    $detail = 'submoduleClean=' + ($submoduleStatus -eq '') + ' privateTracked=' + $privateInRoot.Count
    if (-not $gitAvailable) { $detail += ' (git not found at ' + $gitExe + ')' }
    if ($ok) { Add-Step -Name 'repo-hygiene' -Ok $true -ExitCode 0 -Detail $detail }
    else { $overallPass = $false; Add-Step -Name 'repo-hygiene' -Ok $false -ExitCode 3 -Detail ($detail + ' submoduleStatus=' + $submoduleStatus) }
} catch {
    $overallPass = $false
    Add-Step -Name 'repo-hygiene' -Ok $false -ExitCode 3 -Detail $_.Exception.Message
}

# 8 claims.csv untouched
try {
    $claimsPath = Join-Path $ctx.Th08Root 'config\claims.csv'
    $claimsLines = @(Get-Content -LiteralPath $claimsPath -ErrorAction Stop)
    if ($claimsLines.Count -eq 1) { Add-Step -Name 'claims-untouched' -Ok $true -ExitCode 0 -Detail 'header only' }
    else { $overallPass = $false; Add-Step -Name 'claims-untouched' -Ok $false -ExitCode 3 -Detail ('lines=' + $claimsLines.Count) }
} catch {
    $overallPass = $false
    Add-Step -Name 'claims-untouched' -Ok $false -ExitCode 3 -Detail $_.Exception.Message
}

$finishedAt = Get-Date
$report = [pscustomobject]@{
    schema     = 'th08-verify-report/1'
    startedAt  = $startedAt.ToString('o')
    finishedAt = $finishedAt.ToString('o')
    durationSeconds = [math]::Round(($finishedAt - $startedAt).TotalSeconds, 1)
    steps      = $steps
    pass       = $overallPass
}
Write-Th08JsonFile -Path (Join-Path $ctx.ReportsDir 'verify-report.json') -Object $report | Out-Null

Write-Output ''
Write-Output ("verify pass=" + $overallPass + " steps=" + $steps.Count + " seconds=" + $report.durationSeconds)
Write-Output ("report: " + (Join-Path $ctx.ReportsDir 'verify-report.json'))
if (-not $overallPass) {
    $failedStep = @($steps | Where-Object { -not $_.ok } | Select-Object -First 1)
    if ($failedStep.Count -gt 0) {
        $map = $failedStep[0].name
        if ($map -eq 'build') { exit 4 }
        if ($map -like 'run-*') { exit 5 }
        if ($map -eq 'unit-tests') { exit 6 }
        exit 3
    }
    exit 3
}
exit 0
