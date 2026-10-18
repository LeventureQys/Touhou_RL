#requires -version 5.1
# TH08 v1.0.0 S3-1 entry: bootstrap the private VC7/DX8 prefix and the canonical target.
param(
    [string]$PrefixSource = 'D:\WorkShop\th08-main\th08-main\scripts\prefix',
    [string]$TargetSource = 'D:\Games\[th08] 东方永夜抄 (日文版)\[th08] 东方永夜抄 (日文版)\th08.exe',
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'Th08Pipeline.psm1') -Force

try {
    $context = Get-Th08Context
    $report = Initialize-Th08Environment -Context $context -PrefixSource $PrefixSource -TargetSource $TargetSource -Force:$Force
    Write-Host ("prefix: " + $report.prefix.destination)
    Write-Host ("prefix files=" + $report.prefix.files + " bytes=" + $report.prefix.bytes + " copied=" + $report.prefix.copied)
    Write-Host ("target: " + $report.target.path + " size=" + $report.target.size + " sha256=" + $report.target.sha256)
    Write-Host ("report: " + (Join-Path $context.ReportsDir 'env-report.json'))
    exit 0
} catch {
    $message = $_.Exception.Message
    Write-Host ("FAILED: " + $message)
    exit (Resolve-Th08ExitCode -Message $message)
}
