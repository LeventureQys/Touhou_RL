#requires -version 5.1
# TH08 v1.0.0 S3-3 entry: stage a self-contained runtime directory from retail data.
param(
    [string]$DataDir = 'D:\Games\[th08] 东方永夜抄 (日文版)\[th08] 东方永夜抄 (日文版)',
    [string]$CfgSource,
    [switch]$Refresh,
    [switch]$UseHardLink,
    [switch]$IncludeScoreDat
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'Th08Pipeline.psm1') -Force

try {
    $context = Get-Th08Context
    $artifact = Join-Path $context.BuildDir 'th08.exe'
    if (-not (Test-Path -LiteralPath $artifact)) {
        Write-Host 'build artifact missing; run Build.ps1 -BuildType bugfix first'
        exit 2
    }
    if (-not (Test-Path -LiteralPath (Join-Path $context.ReportsDir 'build-bugfix.json'))) {
        Write-Host 'build-bugfix.json missing; run Build.ps1 -BuildType bugfix first'
        exit 2
    }

    $manifest = Initialize-Th08Runtime -Context $context -DataDir $DataDir -CfgSource $CfgSource `
        -Refresh:$Refresh -UseHardLink:$UseHardLink -IncludeScoreDat:$IncludeScoreDat
    $check = Assert-Th08Runtime -Context $context

    $totalBytes = 0
    foreach ($file in $manifest.files) { $totalBytes += $file.Size }
    Write-Host ("play dir: " + $context.PlayDir)
    Write-Host ("files=" + $manifest.files.Count + " bytes=" + $totalBytes + " exeSha256=" + $manifest.exe.Sha256)
    foreach ($file in $manifest.files) {
        Write-Host ("  " + $file.Name + " size=" + $file.Size + " action=" + $file.Action)
    }
    Write-Host ("cfg length=" + $manifest.cfg.Length + " windowed=" + $manifest.cfg.Windowed + " version=" + $manifest.cfg.Version)
    Write-Host ("manifest: " + (Join-Path $context.PlayDir 'runtime-manifest.json'))
    exit 0
} catch {
    $message = $_.Exception.Message
    Write-Host ("FAILED: " + $message)
    exit (Resolve-Th08ExitCode -Message $message)
}
