#requires -version 5.1
# TH08 v1.0.0 unit tests. Self-contained (no Pester). Exit 0 only when every case passes.
param(
    [string]$Filter = '*'
)

$ErrorActionPreference = 'Stop'
$modulePath = Join-Path (Split-Path -Parent $PSScriptRoot) 'Th08Pipeline.psm1'
Import-Module $modulePath -Force

Add-Type -AssemblyName System.Drawing -ErrorAction SilentlyContinue

$ctx = Get-Th08Context
$results = New-Object System.Collections.ArrayList
$tempRoot = Join-Path $env:TEMP ('th08-test-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null

function New-TestFile {
    param([string]$Name, [byte[]]$Bytes)
    $path = Join-Path $tempRoot $Name
    [System.IO.File]::WriteAllBytes($path, $Bytes)
    return $path
}

function Invoke-Case {
    param(
        [string]$Id,
        [string]$Description,
        [scriptblock]$Body
    )
    $script:caseCount++
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        & $Body
        $watch.Stop()
        [void]$results.Add([pscustomobject]@{ Id = $Id; Description = $Description; Pass = $true; Detail = ''; Ms = $watch.ElapsedMilliseconds })
        Write-Output ("PASS " + $Id + " " + $Description)
    } catch {
        $watch.Stop()
        $detail = $_.Exception.Message
        [void]$results.Add([pscustomobject]@{ Id = $Id; Description = $Description; Pass = $false; Detail = $detail; Ms = $watch.ElapsedMilliseconds })
        Write-Output ("FAIL " + $Id + " " + $Description + " :: " + $detail)
    }
}

function Assert-Condition {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw ("assertion failed: " + $Message) }
}

function Assert-Throws {
    param([scriptblock]$Body, [string]$ExpectedCode)
    $threw = $false
    $message = ''
    try { & $Body | Out-Null } catch { $threw = $true; $message = $_.Exception.Message }
    if (-not $threw) { throw ("expected " + $ExpectedCode + " but no exception was raised") }
    if ($message -notlike ('*' + $ExpectedCode + '*')) {
        throw ("expected " + $ExpectedCode + " but got: " + $message)
    }
}

Write-Output ("TH08 unit tests  temp=" + $tempRoot)
Write-Output ("context th08Root=" + $ctx.Th08Root)
$script:caseCount = 0

try {
    Invoke-Case -Id 'T-01' -Description 'canonical target hash accepted' -Body {
        $target = Assert-Th08Target -Path $ctx.TargetExePath
        Assert-Condition ($target.Verified -eq $true) 'Verified must be true'
        Assert-Condition ($target.Size -eq 840704) ('size must be 840704, got ' + $target.Size)
        Assert-Condition ($target.Sha256 -eq '330FBDBF58A710829D65277B4F312CFBB38D5448B3DF523E79350B879213D924') 'sha256 mismatch'
    }

    Invoke-Case -Id 'T-02' -Description 'tampered binary rejected (E02)' -Body {
        $bytes = [System.IO.File]::ReadAllBytes($ctx.TargetExePath)
        $bytes[$bytes.Length - 1] = [byte]($bytes[$bytes.Length - 1] -bxor 0xFF)
        $tampered = New-TestFile -Name 'tampered.exe' -Bytes $bytes
        Assert-Throws -Body { Assert-Th08Target -Path $tampered } -ExpectedCode 'TH08-E02'
    }

    Invoke-Case -Id 'T-03' -Description 'PE32 identity parsed from target' -Body {
        $pe = Get-Th08PeInfo -Path $ctx.TargetExePath
        Assert-Condition ($pe.IsPe32 -eq $true) 'IsPe32'
        Assert-Condition ($pe.Machine -eq '0x014C') ('machine ' + $pe.Machine)
        Assert-Condition ($pe.Subsystem -eq 2) ('subsystem ' + $pe.Subsystem)
        Assert-Condition ($pe.ImageBase -eq '0x00400000') ('imageBase ' + $pe.ImageBase)
        Assert-Condition ($pe.Size -eq 840704) ('size ' + $pe.Size)
    }

    Invoke-Case -Id 'T-04' -Description 'non PE file rejected (E05)' -Body {
        $text = New-TestFile -Name 'not-a-pe.bin' -Bytes ([System.Text.Encoding]::ASCII.GetBytes('this is not a portable executable'))
        Assert-Throws -Body { Get-Th08PeInfo -Path $text } -ExpectedCode 'TH08-E05'
    }

    Invoke-Case -Id 'T-05' -Description 'windowed config fields rewritten' -Body {
        $source = [byte[]]::new(60)
        $versionBytes = [System.BitConverter]::GetBytes([uint32]0x00080001)
        [System.Array]::Copy($versionBytes, 0, $source, 0x14, 4)
        $source[0x22] = 0
        $sourcePath = New-TestFile -Name 'cfg-src.bin' -Bytes $source
        $destination = Join-Path $tempRoot 'cfg-out.bin'
        $info = Write-Th08WindowedConfig -SourceCfg $sourcePath -Destination $destination
        Assert-Condition ($info.Length -eq 60) ('length ' + $info.Length)
        Assert-Condition ($info.Windowed -eq 1) ('windowed ' + $info.Windowed)
        Assert-Condition ($info.Version -eq '0x00080001') ('version ' + $info.Version)
        $written = [System.IO.File]::ReadAllBytes($destination)
        Assert-Condition ($written[0x22] -eq 1) 'byte 0x22 must be 1 on disk'
        Assert-Condition ([System.BitConverter]::ToUInt32($written, 0x14) -eq 0x00080001) 'version must be preserved'
    }

    Invoke-Case -Id 'T-06' -Description 'windowed config write is idempotent' -Body {
        $sourcePath = Join-Path $tempRoot 'cfg-src.bin'
        $first = Join-Path $tempRoot 'cfg-a.bin'
        $second = Join-Path $tempRoot 'cfg-b.bin'
        $a = Write-Th08WindowedConfig -SourceCfg $sourcePath -Destination $first
        $b = Write-Th08WindowedConfig -SourceCfg $sourcePath -Destination $second
        Assert-Condition ($a.Sha256 -eq $b.Sha256) ('hashes differ: ' + $a.Sha256 + ' vs ' + $b.Sha256)
    }

    Invoke-Case -Id 'T-07' -Description 'config with wrong length rejected (E06)' -Body {
        $shortCfg = New-TestFile -Name 'cfg-short.bin' -Bytes ([byte[]]::new(59))
        $destination = Join-Path $tempRoot 'cfg-short-out.bin'
        Assert-Throws -Body { Write-Th08WindowedConfig -SourceCfg $shortCfg -Destination $destination } -ExpectedCode 'TH08-E06'
    }

    Invoke-Case -Id 'T-08' -Description 'staged runtime directory validates' -Body {
        $check = Assert-Th08Runtime -Context $ctx
        Assert-Condition ($check.Ok -eq $true) 'runtime check must pass'
        Assert-Condition ($check.ExeHashChecked -eq $true) 'exe hash must be cross checked with build report'

        # Manifest contract (design doc 6.4): assembly-time snapshot, every entry
        # carries an explicit Mutable flag, and score.dat is not seeded by default.
        $manifestPath = Join-Path $ctx.PlayDir 'runtime-manifest.json'
        Assert-Condition (Test-Path -LiteralPath $manifestPath) 'runtime-manifest.json missing'
        $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
        Assert-Condition ([bool]$manifest.note) 'manifest note missing'
        $names = @($manifest.files | ForEach-Object { $_.Name })
        foreach ($name in @('th08.dat', 'thbgm.dat')) {
            Assert-Condition ($names -contains $name) ('manifest must list ' + $name)
        }
        Assert-Condition ($names -notcontains 'score.dat') 'score.dat must not be seeded by default'
        foreach ($file in $manifest.files) {
            Assert-Condition ($null -ne $file.Mutable) ('manifest entry lacks Mutable flag: ' + $file.Name)
            Assert-Condition ($file.Mutable -eq $false) ('immutable entry flagged mutable: ' + $file.Name)
        }
    }

    Invoke-Case -Id 'T-09' -Description 'runtime directory missing data rejected (E06)' -Body {
        $emptyDir = Join-Path $tempRoot 'empty-run'
        New-Item -ItemType Directory -Force -Path $emptyDir | Out-Null
        $fakeContext = [pscustomobject]@{
            PlayDir    = $emptyDir
            ReportsDir = $ctx.ReportsDir
            BuildDir   = $ctx.BuildDir
        }
        Assert-Throws -Body { Assert-Th08Runtime -Context $fakeContext } -ExpectedCode 'TH08-E06'
    }

    Invoke-Case -Id 'T-10' -Description 'l2-gameplay key script parses' -Body {
        $profilePath = Join-Path (Split-Path -Parent $PSScriptRoot) 'profiles\l2-gameplay.json'
        $script = Read-Th08KeyScript -Path $profilePath
        Assert-Condition ($script.Name -eq 'l2-gameplay') 'name'
        Assert-Condition (@($script.Steps).Count -gt 5) ('steps ' + @($script.Steps).Count)
        $shotNames = @($script.Steps | Where-Object { $_.ShotName } | ForEach-Object { $_.ShotName })
        Assert-Condition ($shotNames -contains '06-stage1-play') 'must include 06-stage1-play'
    }

    Invoke-Case -Id 'T-11' -Description 'unknown key script action rejected (E12)' -Body {
        $bad = New-TestFile -Name 'bad-action.json' -Bytes ([System.Text.Encoding]::UTF8.GetBytes('{"schema":"th08-key-script/1","name":"bad","steps":[{"action":"jump"}]}'))
        Assert-Throws -Body { Read-Th08KeyScript -Path $bad } -ExpectedCode 'TH08-E12'
    }

    Invoke-Case -Id 'T-12' -Description 'out of range virtual key rejected (E12)' -Body {
        $bad = New-TestFile -Name 'bad-vk.json' -Bytes ([System.Text.Encoding]::UTF8.GetBytes('{"schema":"th08-key-script/1","name":"bad","steps":[{"action":"key","vk":300}]}'))
        Assert-Throws -Body { Read-Th08KeyScript -Path $bad } -ExpectedCode 'TH08-E12'
    }

    Invoke-Case -Id 'T-13' -Description 'log audit accepts normal boot log' -Body {
        $text = "T&L HAL で動作しま～す`r`nDirectSound は正常に初期化されました`r`nKeyBoard OK`r`n"
        $path = New-TestFile -Name 'log-ok.txt' -Bytes ([System.Text.Encoding]::GetEncoding(932).GetBytes($text))
        $audit = Test-Th08LogAudit -LogPath $path
        Assert-Condition ($audit.Pass -eq $true) ('audit must pass; reason=' + $audit.Reason)
        Assert-Condition (@($audit.RequiredMissing).Count -eq 0) 'no missing markers'
    }

    Invoke-Case -Id 'T-14' -Description 'log audit rejects fatal marker' -Body {
        $text = "DirectSound は正常に初期化されました`r`nKeyBoard OK`r`nFatal error: something broke`r`n"
        $path = New-TestFile -Name 'log-fatal.txt' -Bytes ([System.Text.Encoding]::GetEncoding(932).GetBytes($text))
        $audit = Test-Th08LogAudit -LogPath $path
        Assert-Condition ($audit.Pass -eq $false) 'audit must fail'
        Assert-Condition (@($audit.ForbiddenFound) -contains 'Fatal') 'Fatal must be reported'
    }

    Invoke-Case -Id 'T-15' -Description 'blank screenshot detection' -Body {
        $black = New-Object System.Drawing.Bitmap 640, 480
        $graphics = [System.Drawing.Graphics]::FromImage($black)
        $graphics.Clear([System.Drawing.Color]::Black)
        $graphics.Dispose()
        $blackPath = Join-Path $tempRoot 'blank.png'
        $black.Save($blackPath, [System.Drawing.Imaging.ImageFormat]::Png)
        $black.Dispose()
        $stats = Get-Th08ImageStats -Path $blackPath
        $blankCheck = Test-Th08ImageBlank -Stats $stats
        Assert-Condition ($blankCheck.Blank -eq $true) ('blank must be detected; stats=' + $stats.DistinctColors + '/' + $stats.MeanLuma)

        $colorful = New-Object System.Drawing.Bitmap 640, 480
        for ($y = 0; $y -lt 480; $y++) {
            for ($x = 0; $x -lt 640; $x += 4) {
                $colorful.SetPixel($x, $y, [System.Drawing.Color]::FromArgb(255, ($x % 256), ($y % 256), (($x + $y) % 256)))
            }
        }
        $colorPath = Join-Path $tempRoot 'colorful.png'
        $colorful.Save($colorPath, [System.Drawing.Imaging.ImageFormat]::Png)
        $colorful.Dispose()
        $colorStats = Get-Th08ImageStats -Path $colorPath
        $colorCheck = Test-Th08ImageBlank -Stats $colorStats
        Assert-Condition ($colorCheck.Blank -eq $false) ('colorful image must not be blank; stats=' + $colorStats.DistinctColors + '/' + $colorStats.MeanLuma)
    }

    Invoke-Case -Id 'T-16' -Description 'python >= 3.11 resolution' -Body {
        $python = Resolve-Th08Python
        Assert-Condition ($python.Path -like '*python.exe') ('path ' + $python.Path)
        Assert-Condition ($python.Version -match 'Python 3\.(1[1-9]|[2-9][0-9])') ('version ' + $python.Version)
    }

    Invoke-Case -Id 'T-17' -Description 'unusable python candidate rejected (E07)' -Body {
        $fake = New-TestFile -Name 'python.exe' -Bytes ([System.Text.Encoding]::ASCII.GetBytes('not a real interpreter'))
        Assert-Throws -Body { Resolve-Th08Python -PythonExe $fake } -ExpectedCode 'TH08-E07'
    }

    Invoke-Case -Id 'T-18' -Description 'build report code hash matches artifact .text' -Body {
        $artifact = Join-Path $ctx.BuildDir 'th08.exe'
        Assert-Condition (Test-Path -LiteralPath $artifact) 'build/th08.exe missing; run Build.ps1 first'
        $artifactHash = Get-Th08Sha256 -Path $artifact
        $matched = $false
        foreach ($name in @('build-normal.json', 'build-bugfix.json')) {
            $reportPath = Join-Path $ctx.ReportsDir $name
            if (-not (Test-Path -LiteralPath $reportPath)) { continue }
            $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
            if ($report.artifact.sha256 -ne $artifactHash) { continue }
            $codeHash = Get-Th08CodeHash -Path $artifact
            Assert-Condition ($codeHash.Length -eq 64) ('code hash length ' + $codeHash.Length)
            Assert-Condition ($codeHash -eq $report.artifact.codeSha256) ('code hash mismatch: ' + $codeHash + ' vs ' + $report.artifact.codeSha256)
            $matched = $true
            break
        }
        Assert-Condition $matched 'no build report matches the current build/th08.exe'
    }
} finally {
    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

$passed = @($results | Where-Object { $_.Pass }).Count
$failed = @($results | Where-Object { -not $_.Pass }).Count
Write-Output ''
Write-Output ("TOTAL " + $results.Count + "  PASSED " + $passed + "  FAILED " + $failed)
foreach ($result in ($results | Where-Object { -not $_.Pass })) {
    Write-Output ("  FAILED " + $result.Id + " :: " + $result.Detail)
}

if ($failed -gt 0) { exit 6 }
exit 0
