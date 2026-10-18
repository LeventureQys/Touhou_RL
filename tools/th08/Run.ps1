#requires -version 5.1
# TH08 v1.0.0 S3-4 entry: launch the staged runtime, drive it, capture evidence.
param(
    [ValidateSet('l2-gameplay', 'l2-menuexit')][string]$Profile = 'l2-gameplay',
    [string]$Tag = '',
    [int]$WindowTimeoutSeconds = 30,
    [int]$TotalTimeoutSeconds = 60
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'Th08Pipeline.psm1') -Force

try {
    $context = Get-Th08Context
    $profilePath = Join-Path $context.ToolsDir ('profiles\' + $Profile + '.json')
    $report = Invoke-Th08Run -Context $context -ProfilePath $profilePath -Tag $Tag `
        -WindowTimeoutSeconds $WindowTimeoutSeconds -TotalTimeoutSeconds $TotalTimeoutSeconds

    Write-Host ("runId: " + $report.runId + "  profile: " + $report.profile + "  pass: " + $report.pass)
    Write-Host ("window: '" + $report.process.windowTitle + "' client=" + $report.process.clientWidth + "x" + $report.process.clientHeight + " firstWindowMs=" + $report.process.firstWindowMs)
    Write-Host ("steps: " + @($report.steps).Count + "  foregroundChecks: " + $report.foregroundChecks + "  totalSeconds: " + $report.totalSeconds)
    Write-Host ("exit: code=" + $report.exit.code + " method=" + $report.exit.method + " forced=" + $report.exit.forced)
    Write-Host ("log audit: pass=" + $report.log.audit.pass + " length=" + $report.log.audit.length)
    foreach ($shot in $report.shots) {
        Write-Host ("  shot " + $shot.name + " " + $shot.width + "x" + $shot.height + " colors=" + $shot.distinctColors + " luma=" + $shot.meanLuma)
    }
    Write-Host ("report: " + (Join-Path $context.EvidenceDir ($report.runId + '_report.json')))
    exit 0
} catch {
    $message = $_.Exception.Message
    Write-Host ("FAILED: " + $message)
    exit (Resolve-Th08ExitCode -Message $message)
}
