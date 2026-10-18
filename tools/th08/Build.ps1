#requires -version 5.1
# TH08 v1.0.0 S3-2 entry: cold build the normal (evidence) and bugfix (runtime) images.
param(
    [ValidateSet('normal', 'bugfix', 'both')][string]$BuildType = 'both',
    [switch]$Fresh,
    [int]$Jobs = 1,
    [string]$PythonExe
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'Th08Pipeline.psm1') -Force

try {
    $context = Get-Th08Context
    $null = Assert-Th08Target -Path $context.TargetExePath

    if ($BuildType -ne 'normal') {
        if (-not (Test-Path -LiteralPath (Join-Path $context.ReportsDir 'build-normal.json'))) {
            Write-Host 'note: normal build report not present; exactness cannot be claimed from this run'
        }
    }

    $reports = @()
    if ($BuildType -eq 'normal' -or $BuildType -eq 'both') {
        Write-Host 'building normal (exact-facing) image ...'
        $reports += Invoke-Th08Build -Context $context -BuildType normal -Fresh:$Fresh -Jobs $Jobs -PythonExe $PythonExe
    }
    if ($BuildType -eq 'bugfix' -or $BuildType -eq 'both') {
        Write-Host 'building bugfix (runtime) image ...'
        $reports += Invoke-Th08Build -Context $context -BuildType bugfix -Fresh:$Fresh -Jobs $Jobs -PythonExe $PythonExe
    }

    foreach ($report in $reports) {
        Write-Host ("buildType=" + $report.buildType + " seconds=" + $report.durationSeconds + " exitCode=" + $report.exitCode)
        Write-Host ("  artifact=" + $report.artifact.path + " size=" + $report.artifact.size)
        Write-Host ("  sha256=" + $report.artifact.sha256)
        Write-Host ("  pe machine=" + $report.artifact.pe.machine + " subsystem=" + $report.artifact.pe.subsystem + " imageBase=" + $report.artifact.pe.imageBase)
        Write-Host ("  report=" + (Join-Path $context.ReportsDir ('build-' + $report.buildType + '.json')))
    }
    exit 0
} catch {
    $message = $_.Exception.Message
    Write-Host ("FAILED: " + $message)
    exit (Resolve-Th08ExitCode -Message $message)
}
