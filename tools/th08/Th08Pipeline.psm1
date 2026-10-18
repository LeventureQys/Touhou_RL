#requires -version 5.1
<#
TH08 v1.0.0 pipeline module.

All logic that must be unit-testable lives here. Entry scripts only wire
parameters, print summaries and map exceptions to process exit codes.

Contracts: Document/Update/Alpha/v1.0.0/设计文档.md section 6.
#>

# ---------------------------------------------------------------------------
# Constants (single source of truth, mirrors 设计文档.md section 6.1)
# ---------------------------------------------------------------------------
$script:Th08TargetSize = 840704
$script:Th08TargetSha256 = '330FBDBF58A710829D65277B4F312CFBB38D5448B3DF523E79350B879213D924'
$script:Th08DataSize = 46838025
$script:Th08BgmSize = 449961024
$script:Th08ScoreSize = 17774
$script:Th08CfgLength = 60
$script:Th08CfgWindowedOffset = 0x22
$script:Th08CfgVersion = 0x00080001
$script:Th08BlankMinDistinctColors = 64
$script:Th08BlankMinMeanLuma = 2
$script:Th08PeMachine = 0x014C
$script:Th08PeSubsystem = 2
$script:Th08PeImageBase = 0x00400000
$script:Th08VkZ = 0x5A
$script:Th08VkX = 0x58
$script:Th08VkEsc = 0x1B
$script:Th08VkEnter = 0x0D
$script:Th08VkShift = 0x10
$script:Th08VkCtrl = 0x11
$script:Th08VkUp = 0x26
$script:Th08VkDown = 0x28
$script:Th08VkLeft = 0x25
$script:Th08VkRight = 0x27
$script:Th08BuildLock = $false
$script:Th08ModuleRoot = $PSScriptRoot

# ---------------------------------------------------------------------------
# Native interop (all struct handling stays inside C#; PowerShell never
# assigns fields of nested value types)
# ---------------------------------------------------------------------------
if (-not ('Th08Native' -as [type])) {
    $nativeSource = @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

public static class Th08Native
{
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT { public int Left; public int Top; public int Right; public int Bottom; }

    [StructLayout(LayoutKind.Sequential)]
    public struct POINT { public int X; public int Y; }

    [StructLayout(LayoutKind.Sequential)]
    public struct KEYBDINPUT { public ushort wVk; public ushort wScan; public uint dwFlags; public uint time; public IntPtr dwExtraInfo; }

    [StructLayout(LayoutKind.Sequential)]
    public struct MOUSEINPUT { public int dx; public int dy; public uint mouseData; public uint dwFlags; public uint time; public IntPtr dwExtraInfo; }

    [StructLayout(LayoutKind.Sequential)]
    public struct HARDWAREINPUT { public uint uMsg; public ushort wParamL; public ushort wParamH; }

    [StructLayout(LayoutKind.Explicit)]
    public struct INPUTUNION
    {
        [FieldOffset(0)] public MOUSEINPUT mi;
        [FieldOffset(0)] public KEYBDINPUT ki;
        [FieldOffset(0)] public HARDWAREINPUT hi;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct INPUT { public uint type; public INPUTUNION u; }

    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool BringWindowToTop(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
    [DllImport("user32.dll")] public static extern bool IsWindow(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr hWnd, out RECT r);
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hWnd, out RECT r);
    [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr hWnd, ref POINT p);
    [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr hWnd, uint msg, IntPtr wParam, IntPtr lParam);
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint pid);
    [DllImport("user32.dll")] public static extern bool AttachThreadInput(uint idAttach, uint idAttachTo, bool fAttach);
    [DllImport("user32.dll")] public static extern IntPtr SetFocus(IntPtr hWnd);
    [DllImport("kernel32.dll")] public static extern uint GetCurrentThreadId();
    [DllImport("user32.dll")] public static extern int GetWindowTextLength(IntPtr hWnd);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int maxCount);
    [DllImport("user32.dll")] public static extern uint MapVirtualKey(uint code, uint mapType);
    [DllImport("user32.dll", SetLastError = true)] public static extern uint SendInput(uint nInputs, INPUT[] pInputs, int cbSize);
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumWindowsCallback callback, IntPtr lParam);
    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
    [DllImport("shcore.dll")] public static extern int SetProcessDpiAwareness(int value);

    public delegate bool EnumWindowsCallback(IntPtr hWnd, IntPtr lParam);

    public const int SW_RESTORE = 9;
    public const uint INPUT_KEYBOARD = 1;
    public const uint KEYEVENTF_KEYUP = 2;
    public const uint WM_CLOSE = 0x0010;

    [DllImport("user32.dll", CharSet = CharSet.Ansi)] public static extern int GetWindowTextA(IntPtr hWnd, byte[] text, int maxCount);

    // Without DPI awareness Windows virtualizes GetWindowRect/ClientToScreen while
    // CopyFromScreen works in physical pixels, so the captured client area is
    // shifted by the display scale factor.
    public static bool EnableDpiAwareness()
    {
        try { if (SetProcessDpiAwareness(2) == 0) { return true; } } catch { }
        try { return SetProcessDPIAware(); } catch { }
        return false;
    }

    public static IntPtr[] WindowsOfProcess(uint pid)
    {
        List<IntPtr> found = new List<IntPtr>();
        EnumWindows(delegate(IntPtr hWnd, IntPtr lParam)
        {
            uint windowPid;
            GetWindowThreadProcessId(hWnd, out windowPid);
            if (windowPid == pid && IsWindowVisible(hWnd)) { found.Add(hWnd); }
            return true;
        }, IntPtr.Zero);
        return found.ToArray();
    }

    // The game recreates its window during start-up, so the handle must be
    // re-resolved from the process instead of cached once.
    public static IntPtr MainWindow(uint pid)
    {
        IntPtr best = IntPtr.Zero;
        int bestArea = 0;
        foreach (IntPtr hWnd in WindowsOfProcess(pid))
        {
            RECT r;
            GetClientRect(hWnd, out r);
            int area = (r.Right - r.Left) * (r.Bottom - r.Top);
            if (area > bestArea) { bestArea = area; best = hWnd; }
        }
        return best;
    }

    public static int ClientAreaWidth(IntPtr hWnd)
    {
        RECT r;
        GetClientRect(hWnd, out r);
        return r.Right - r.Left;
    }

    public static int ClientAreaHeight(IntPtr hWnd)
    {
        RECT r;
        GetClientRect(hWnd, out r);
        return r.Bottom - r.Top;
    }

    public static string WindowTitle(IntPtr hWnd)
    {
        int length = GetWindowTextLength(hWnd);
        StringBuilder sb = new StringBuilder(length + 2);
        GetWindowText(hWnd, sb, sb.Capacity);
        return sb.ToString();
    }

    // TH08 sets its caption through the ANSI entry point with Shift-JIS bytes, so
    // the Unicode accessor returns system-codepage mojibake. The raw ANSI bytes can
    // be decoded as CP932 to recover the original Japanese title.
    public static byte[] WindowTitleBytes(IntPtr hWnd)
    {
        int length = GetWindowTextLength(hWnd);
        byte[] buffer = new byte[length + 2];
        GetWindowTextA(hWnd, buffer, buffer.Length);
        return buffer;
    }

    public static int[] ClientGeometry(IntPtr hWnd)
    {
        RECT r;
        GetClientRect(hWnd, out r);
        POINT p = new POINT();
        p.X = 0;
        p.Y = 0;
        ClientToScreen(hWnd, ref p);
        return new int[] { p.X, p.Y, r.Right - r.Left, r.Bottom - r.Top };
    }

    public static int[] WindowBounds(IntPtr hWnd)
    {
        RECT r;
        GetWindowRect(hWnd, out r);
        return new int[] { r.Left, r.Top, r.Right - r.Left, r.Bottom - r.Top };
    }

    public static uint WindowProcessId(IntPtr hWnd)
    {
        uint pid;
        GetWindowThreadProcessId(hWnd, out pid);
        return pid;
    }

    public static uint ForegroundPid()
    {
        return WindowProcessId(GetForegroundWindow());
    }

    public static bool Focus(IntPtr hWnd)
    {
        IntPtr foreground = GetForegroundWindow();
        if (foreground == hWnd) { return true; }

        ShowWindow(hWnd, SW_RESTORE);
        BringWindowToTop(hWnd);

        uint currentThread = GetCurrentThreadId();
        uint foregroundThread = 0;
        uint foregroundPid = 0;
        if (foreground != IntPtr.Zero) { foregroundThread = GetWindowThreadProcessId(foreground, out foregroundPid); }
        bool attached = false;
        if (foregroundThread != 0 && foregroundThread != currentThread) { attached = AttachThreadInput(currentThread, foregroundThread, true); }
        SetForegroundWindow(hWnd);
        SetFocus(hWnd);
        if (attached) { AttachThreadInput(currentThread, foregroundThread, false); }

        return GetForegroundWindow() == hWnd;
    }

    public static uint SendKey(ushort vk, bool keyUp)
    {
        INPUT input = new INPUT();
        input.type = INPUT_KEYBOARD;
        input.u.ki.wVk = vk;
        input.u.ki.wScan = (ushort)MapVirtualKey(vk, 0);
        input.u.ki.dwFlags = keyUp ? KEYEVENTF_KEYUP : 0u;
        input.u.ki.time = 0;
        input.u.ki.dwExtraInfo = IntPtr.Zero;
        INPUT[] buffer = new INPUT[1];
        buffer[0] = input;
        return SendInput(1, buffer, Marshal.SizeOf(typeof(INPUT)));
    }

    public static bool CloseWindow(IntPtr hWnd)
    {
        return PostMessage(hWnd, WM_CLOSE, IntPtr.Zero, IntPtr.Zero);
    }
}
'@
    Add-Type -TypeDefinition $nativeSource -Language CSharp
}

# Must run before any window/coordinate work in this process.
try { [void][Th08Native]::EnableDpiAwareness() } catch { }

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

function Get-Th08Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw ("TH08-E02: file not found (path=$Path)")
    }
    $hash = Get-FileHash -LiteralPath $Path -Algorithm SHA256
    return $hash.Hash.ToUpperInvariant()
}

function Write-Th08JsonFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][object]$Object
    )
    $directory = Split-Path -Parent $Path
    if ($directory -and -not (Test-Path -LiteralPath $directory)) {
        New-Item -ItemType Directory -Force -Path $directory | Out-Null
    }
    $json = $Object | ConvertTo-Json -Depth 12
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $json, $encoding)
    return $Path
}

function Write-Th08Report {
    param(
        [Parameter(Mandatory = $true)][string]$ReportsDir,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][object]$Object
    )
    if (-not (Test-Path -LiteralPath $ReportsDir)) {
        New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null
    }
    return (Write-Th08JsonFile -Path (Join-Path $ReportsDir $Name) -Object $Object)
}

# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

function Get-Th08Context {
    param([string]$RepoRoot)

    if (-not $RepoRoot) {
        $moduleRoot = $script:Th08ModuleRoot
        if (-not $moduleRoot) { $moduleRoot = $PSScriptRoot }
        # module lives in <RepoRoot>\tools\th08
        $RepoRoot = Split-Path -Parent (Split-Path -Parent $moduleRoot)
    }

    $th08Root = Join-Path $RepoRoot 'third_party\th08'
    $scriptsDir = Join-Path $th08Root 'scripts'
    if (-not (Test-Path -LiteralPath $th08Root)) {
        throw ("TH08-E01: th08 submodule missing (path=$th08Root)")
    }
    if (-not (Test-Path -LiteralPath $scriptsDir)) {
        throw ("TH08-E01: th08 scripts directory missing (path=$scriptsDir)")
    }

    $buildDir = Join-Path $th08Root 'build'
    $context = [pscustomobject]@{
        RepoRoot     = $RepoRoot
        Th08Root     = $th08Root
        ToolsDir     = (Join-Path $RepoRoot 'tools\th08')
        ScriptsDir   = $scriptsDir
        TargetExePath = (Join-Path $th08Root 'resources\th08.exe')
        PrefixDir    = (Join-Path $scriptsDir 'prefix')
        BuildDir     = $buildDir
        PlayDir      = (Join-Path $buildDir 'play')
        EvidenceDir  = (Join-Path $buildDir 'play\evidence')
        ReportsDir   = (Join-Path $buildDir 'reports')
        PythonExe    = $null
        PythonVersion = $null
    }

    try {
        $python = Resolve-Th08Python
        $context.PythonExe = $python.Path
        $context.PythonVersion = $python.Version
    } catch {
        # Resolution failure is reported by the callers that actually need Python (E07).
        $context.PythonExe = $null
        $context.PythonVersion = $null
    }

    return $context
}

# ---------------------------------------------------------------------------
# Python resolution (E07)
# ---------------------------------------------------------------------------

function Test-Th08PythonCandidate {
    param([Parameter(Mandatory = $true)][string]$ExePath)

    if (-not $ExePath) { return $null }
    if (-not (Test-Path -LiteralPath $ExePath -PathType Leaf)) { return $null }
    if ($ExePath -like '*\Microsoft\WindowsApps\python.exe') { return $null }

    try {
        $output = & $ExePath --version 2>&1
    } catch {
        return $null
    }
    if (-not $output) { return $null }
    $text = ($output | Out-String).Trim()
    if ($text -notmatch 'Python 3\.(1[1-9]|[2-9][0-9])') { return $null }

    $versionText = $text
    $parsed = $null
    if ($text -match '(3\.\d+\.\d+)') { $parsed = [version]$matches[1] }
    return [pscustomobject]@{ Path = $ExePath; Version = $versionText; ParsedVersion = $parsed }
}

function Resolve-Th08Python {
    param([string]$PythonExe)

    if ($PythonExe) {
        $candidate = Test-Th08PythonCandidate -ExePath $PythonExe
        if (-not $candidate) {
            throw ("TH08-E07: python >=3.11 not usable (path=$PythonExe)")
        }
        return $candidate
    }

    $onPath = Get-Command python -ErrorAction SilentlyContinue
    if ($onPath) {
        $candidate = Test-Th08PythonCandidate -ExePath $onPath.Source
        if ($candidate) { return $candidate }
    }

    $roots = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python'),
        'C:\Python'
    )
    $found = @()
    foreach ($root in $roots) {
        if (-not (Test-Path -LiteralPath $root)) { continue }
        $found += @(Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -like 'Python3*' } |
            ForEach-Object { Join-Path $_.FullName 'python.exe' })
    }
    $candidates = @()
    foreach ($path in $found) {
        $candidate = Test-Th08PythonCandidate -ExePath $path
        if ($candidate) { $candidates += $candidate }
    }
    if ($candidates.Count -eq 0) {
        throw ("TH08-E07: python >=3.11 not found (searched=PATH,LOCALAPPDATA\Programs\Python,C:\Python)")
    }
    $best = $candidates | Sort-Object -Property ParsedVersion -Descending | Select-Object -First 1
    return $best
}

# ---------------------------------------------------------------------------
# S3-1: prefix + canonical target
# ---------------------------------------------------------------------------

function Get-Th08CodeHash {
    param([Parameter(Mandatory = $true)][string]$Path)

    # The linker stamps PE TimeDateStamp, so whole-file hashes differ between two
    # otherwise identical builds. The .text section hash is the deterministic
    # evidence that code generation reproduced.
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    if ($bytes.Length -lt 0x40 -or $bytes[0] -ne 0x4D -or $bytes[1] -ne 0x5A) {
        throw ("TH08-E05: not a PE file (path=$Path)")
    }
    $e_lfanew = [System.BitConverter]::ToInt32($bytes, 0x3C)
    $sectionCount = [System.BitConverter]::ToUInt16($bytes, $e_lfanew + 6)
    $optionalSize = [System.BitConverter]::ToUInt16($bytes, $e_lfanew + 20)
    $sectionOffset = $e_lfanew + 24 + $optionalSize
    for ($index = 0; $index -lt $sectionCount; $index++) {
        $offset = $sectionOffset + $index * 40
        $name = [System.Text.Encoding]::ASCII.GetString($bytes, $offset, 8).TrimEnd([char]0)
        if ($name -eq '.text') {
            $rawSize = [int][System.BitConverter]::ToUInt32($bytes, $offset + 16)
            $rawPointer = [int][System.BitConverter]::ToUInt32($bytes, $offset + 20)
            if ($rawSize -le 0 -or ($rawPointer + $rawSize) -gt $bytes.Length) {
                throw ("TH08-E05: invalid .text extent (path=$Path; size=$rawSize; pointer=$rawPointer)")
            }
            $sha = [System.Security.Cryptography.SHA256]::Create()
            try {
                $hash = $sha.ComputeHash($bytes, $rawPointer, $rawSize)
            } finally {
                $sha.Dispose()
            }
            return ([System.BitConverter]::ToString($hash)).Replace('-', '')
        }
    }
    throw ("TH08-E05: .text section missing (path=$Path)")
}

function Assert-Th08Prefix {
    param([Parameter(Mandatory = $true)][string]$Path)

    $vsBase = Join-Path $Path 'PROGRAM FILES\MICROSOFT VISUAL STUDIO .NET\VC7'
    $clPath = Join-Path $vsBase 'BIN\CL.EXE'
    $dxHeader = Join-Path $Path 'mssdk\include\d3dx8.h'
    $dxLib = Join-Path $Path 'mssdk\lib\d3dx8.lib'
    $ninja = Join-Path $Path 'ninja\ninja.exe'

    $missing = @()
    foreach ($pair in @(
            @{ Name = 'cl'; Path = $clPath },
            @{ Name = 'd3dx8.h'; Path = $dxHeader },
            @{ Name = 'd3dx8.lib'; Path = $dxLib },
            @{ Name = 'ninja.exe'; Path = $ninja })) {
        if (-not (Test-Path -LiteralPath $pair.Path -PathType Leaf)) { $missing += $pair.Name }
    }
    if ($missing.Count -gt 0) {
        throw ("TH08-E01: prefix incomplete (path=$Path; missing=" + ($missing -join ',') + ')')
    }

    $files = @(Get-ChildItem -LiteralPath $Path -Recurse -File -ErrorAction SilentlyContinue)
    $bytes = ($files | Measure-Object -Property Length -Sum).Sum

    return [pscustomobject]@{
        Path       = $Path
        ClPath     = $clPath
        D3dx8Header = $dxHeader
        D3dx8Lib   = $dxLib
        NinjaPath  = $ninja
        FileCount  = $files.Count
        Bytes      = $bytes
    }
}

function Assert-Th08Target {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [int]$ExpectedSize = $script:Th08TargetSize,
        [string]$ExpectedSha256 = $script:Th08TargetSha256
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw ("TH08-E02: target executable missing (path=$Path)")
    }
    $item = Get-Item -LiteralPath $Path
    if ($item.Length -ne $ExpectedSize) {
        throw ("TH08-E02: target size mismatch (path=$Path; expected=$ExpectedSize; actual=" + $item.Length + ')')
    }
    $hash = Get-Th08Sha256 -Path $Path
    $expected = $ExpectedSha256.ToUpperInvariant()
    if ($hash -ne $expected) {
        throw ("TH08-E02: target sha256 mismatch (path=$Path; expected=$expected; actual=$hash)")
    }

    return [pscustomobject]@{ Path = $Path; Size = $item.Length; Sha256 = $hash; Verified = $true }
}

function Initialize-Th08Environment {
    param(
        [Parameter(Mandatory = $true)][object]$Context,
        [Parameter(Mandatory = $true)][string]$PrefixSource,
        [Parameter(Mandatory = $true)][string]$TargetSource,
        [switch]$Force
    )

    $prefixSourceInfo = Assert-Th08Prefix -Path $PrefixSource

    $prefixCopied = $false
    if ($Force -or -not (Test-Path -LiteralPath $Context.PrefixDir)) {
        if (Test-Path -LiteralPath $Context.PrefixDir) {
            Remove-Item -LiteralPath $Context.PrefixDir -Recurse -Force
        }
        $parent = Split-Path -Parent $Context.PrefixDir
        if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
        Copy-Item -LiteralPath $PrefixSource -Destination $Context.PrefixDir -Recurse -Force
        $prefixCopied = $true
    }

    $prefixInfo = Assert-Th08Prefix -Path $Context.PrefixDir

    $targetCopied = $false
    if ($Force -or -not (Test-Path -LiteralPath $Context.TargetExePath)) {
        $parent = Split-Path -Parent $Context.TargetExePath
        if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
        Copy-Item -LiteralPath $TargetSource -Destination $Context.TargetExePath -Force
        $targetCopied = $true
    }

    $targetInfo = Assert-Th08Target -Path $Context.TargetExePath

    $report = [pscustomobject]@{
        schema    = 'th08-env-report/1'
        createdAt = (Get-Date).ToString('o')
        prefix    = [pscustomobject]@{
            source      = $PrefixSource
            destination = $prefixInfo.Path
            copied      = $prefixCopied
            files       = $prefixInfo.FileCount
            bytes       = $prefixInfo.Bytes
            checks      = [pscustomobject]@{
                cl          = (Test-Path -LiteralPath $prefixInfo.ClPath)
                d3dx8Header = (Test-Path -LiteralPath $prefixInfo.D3dx8Header)
                d3dx8Lib    = (Test-Path -LiteralPath $prefixInfo.D3dx8Lib)
                ninja       = (Test-Path -LiteralPath $prefixInfo.NinjaPath)
            }
        }
        target    = [pscustomobject]@{
            path   = $targetInfo.Path
            size   = $targetInfo.Size
            sha256 = $targetInfo.Sha256
            copied = $targetCopied
        }
        ok        = $true
    }

    Write-Th08Report -ReportsDir $Context.ReportsDir -Name 'env-report.json' -Object $report | Out-Null
    return $report
}

# ---------------------------------------------------------------------------
# PE inspection (E05)
# ---------------------------------------------------------------------------

function Get-Th08PeInfo {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw ("TH08-E05: executable missing (path=$Path)")
    }
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    if ($bytes.Length -lt 0x40) {
        throw ("TH08-E05: file too small to be PE (path=$Path; size=" + $bytes.Length + ')')
    }
    if ($bytes[0] -ne 0x4D -or $bytes[1] -ne 0x5A) {
        throw ("TH08-E05: missing MZ signature (path=$Path)")
    }
    $e_lfanew = [System.BitConverter]::ToInt32($bytes, 0x3C)
    if ($e_lfanew -le 0 -or ($e_lfanew + 0x40) -ge $bytes.Length) {
        throw ("TH08-E05: invalid e_lfanew (path=$Path; value=$e_lfanew)")
    }
    if ($bytes[$e_lfanew] -ne 0x50 -or $bytes[$e_lfanew + 1] -ne 0x45 -or
        $bytes[$e_lfanew + 2] -ne 0x00 -or $bytes[$e_lfanew + 3] -ne 0x00) {
        throw ("TH08-E05: missing PE signature (path=$Path)")
    }

    $machine = [System.BitConverter]::ToUInt16($bytes, $e_lfanew + 4)
    $optionalHeader = $e_lfanew + 24
    $magic = [System.BitConverter]::ToUInt16($bytes, $optionalHeader)
    if ($magic -ne 0x10B) {
        throw ("TH08-E05: not PE32 (path=$Path; optionalHeaderMagic=0x" + $magic.ToString('X4') + ')')
    }
    $imageBase = [System.BitConverter]::ToUInt32($bytes, $optionalHeader + 28)
    $subsystem = [System.BitConverter]::ToUInt16($bytes, $optionalHeader + 68)

    return [pscustomobject]@{
        Path      = $Path
        IsPe32    = $true
        Machine   = ('0x' + $machine.ToString('X4'))
        Subsystem = [int]$subsystem
        ImageBase = ('0x' + $imageBase.ToString('X8'))
        Size      = $bytes.Length
    }
}

function Assert-Th08PeInfo {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [int]$ExpectedMachine = $script:Th08PeMachine,
        [int]$ExpectedSubsystem = $script:Th08PeSubsystem,
        [long]$ExpectedImageBase = $script:Th08PeImageBase
    )

    $info = Get-Th08PeInfo -Path $Path
    $machineValue = [System.Convert]::ToInt32($info.Machine.Substring(2), 16)
    $imageBaseValue = [System.Convert]::ToInt64($info.ImageBase.Substring(2), 16)
    if ($machineValue -ne $ExpectedMachine) {
        throw ("TH08-E05: machine mismatch (path=$Path; expected=0x" + $ExpectedMachine.ToString('X4') + '; actual=' + $info.Machine + ')')
    }
    if ($info.Subsystem -ne $ExpectedSubsystem) {
        throw ("TH08-E05: subsystem mismatch (path=$Path; expected=$ExpectedSubsystem; actual=" + $info.Subsystem + ')')
    }
    if ($imageBaseValue -ne $ExpectedImageBase) {
        throw ("TH08-E05: image base mismatch (path=$Path; expected=0x" + $ExpectedImageBase.ToString('X8') + '; actual=' + $info.ImageBase + ')')
    }
    return $info
}

# ---------------------------------------------------------------------------
# S3-2: build (E04)
# ---------------------------------------------------------------------------

function Invoke-Th08CapturedProcess {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [Parameter(Mandatory = $true)][string]$StdOutPath,
        [Parameter(Mandatory = $true)][string]$StdErrPath,
        [int]$TimeoutMinutes = 30
    )

    # Start-Process -PassThru does not expose ExitCode on Windows PowerShell 5.1,
    # so drive System.Diagnostics.Process directly.
    $quoted = @()
    foreach ($argument in $Arguments) {
        if ($argument -match '[\s"]') { $quoted += ('"' + ($argument -replace '"', '\"') + '"') }
        else { $quoted += $argument }
    }

    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $FilePath
    $startInfo.Arguments = ($quoted -join ' ')
    $startInfo.WorkingDirectory = $WorkingDirectory
    $startInfo.UseShellExecute = $false
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.CreateNoWindow = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    if (-not $process.Start()) {
        throw ("TH08-E04: could not start process (path=$FilePath)")
    }

    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $completed = $process.WaitForExit([int]($TimeoutMinutes * 60 * 1000))
    $timedOut = $false
    if (-not $completed) {
        $timedOut = $true
        try { & taskkill.exe /PID $process.Id /T /F | Out-Null } catch { }
        try { [void]$process.WaitForExit(10000) } catch { }
    }

    $stdout = ''
    $stderr = ''
    try { $stdout = $stdoutTask.Result } catch { }
    try { $stderr = $stderrTask.Result } catch { }
    if ($null -eq $stdout) { $stdout = '' }
    if ($null -eq $stderr) { $stderr = '' }

    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($StdOutPath, $stdout, $encoding)
    [System.IO.File]::WriteAllText($StdErrPath, $stderr, $encoding)

    $exitCode = $null
    if (-not $timedOut) {
        try { $exitCode = $process.ExitCode } catch { $exitCode = $null }
    }
    $process.Dispose()

    return [pscustomobject]@{
        ExitCode = $exitCode
        TimedOut = $timedOut
        StdOut   = $stdout
        StdErr   = $stderr
    }
}

function Invoke-Th08Build {
    param(
        [Parameter(Mandatory = $true)][object]$Context,
        [Parameter(Mandatory = $true)][ValidateSet('normal', 'bugfix')][string]$BuildType,
        [switch]$Fresh,
        [int]$Jobs = 1,
        [int]$TimeoutMinutes = 30,
        [string]$PythonExe
    )

    if ($script:Th08BuildLock) {
        throw ("TH08-E04: another build is already running in this session (buildType=$BuildType)")
    }

    $python = Resolve-Th08Python -PythonExe $PythonExe
    if (-not (Test-Path -LiteralPath $Context.ReportsDir)) {
        New-Item -ItemType Directory -Force -Path $Context.ReportsDir | Out-Null
    }

    # Single-writer guard: the th08 repository requires serial VC7 builds. Another
    # session (or a stray editor build) holding cl/link/ninja would corrupt the
    # shared build graph, so wait briefly and then refuse.
    $guardDeadline = (Get-Date).AddSeconds(120)
    while ($true) {
        $busy = @(Get-Process -Name 'cl', 'link', 'ninja' -ErrorAction SilentlyContinue)
        if ($busy.Count -eq 0) { break }
        if ((Get-Date) -gt $guardDeadline) {
            throw ("TH08-E04: another VC7 build is running (pids=" + (($busy | ForEach-Object { $_.Id }) -join ',') + '); refusing to build concurrently')
        }
        Start-Sleep -Seconds 3
    }

    $artifactPath = Join-Path $Context.BuildDir 'th08.exe'
    $outFile = Join-Path $Context.ReportsDir ("build-" + $BuildType + "-stdout.txt")
    $errFile = Join-Path $Context.ReportsDir ("build-" + $BuildType + "-stderr.txt")

    $argumentList = @('scripts\build.py', '--build-type', $BuildType, '-j', [string]$Jobs)
    if ($Fresh) { $argumentList += '--fresh' }
    $commandText = ($python.Path + ' ' + ($argumentList -join ' '))

    $script:Th08BuildLock = $true
    $startedAt = Get-Date
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $result = Invoke-Th08CapturedProcess -FilePath $python.Path -Arguments $argumentList `
            -WorkingDirectory $Context.Th08Root -StdOutPath $outFile -StdErrPath $errFile `
            -TimeoutMinutes $TimeoutMinutes
    } finally {
        $stopwatch.Stop()
        $script:Th08BuildLock = $false
    }

    $exitCode = $result.ExitCode
    $timedOut = $result.TimedOut

    $tail = @($result.StdOut -split "`r?`n" | Where-Object { $_ -ne '' } | Select-Object -Last 20) +
            @($result.StdErr -split "`r?`n" | Where-Object { $_ -ne '' } | Select-Object -Last 20)
    $tail = @($tail)

    if ($timedOut) {
        throw ("TH08-E04: build timed out (buildType=$BuildType; minutes=$TimeoutMinutes; command=$commandText)")
    }
    if ($null -eq $exitCode -or $exitCode -ne 0) {
        throw ("TH08-E04: build failed (buildType=$BuildType; exitCode=$exitCode; command=$commandText; tail=" + (($tail | Select-Object -Last 8) -join ' | ') + ')')
    }
    if (-not (Test-Path -LiteralPath $artifactPath -PathType Leaf)) {
        throw ("TH08-E04: build produced no artifact (buildType=$BuildType; path=$artifactPath)")
    }

    $pe = Assert-Th08PeInfo -Path $artifactPath
    $artifactHash = Get-Th08Sha256 -Path $artifactPath

    $report = [pscustomobject]@{
        schema          = 'th08-build-report/1'
        buildType       = $BuildType
        startedAt       = $startedAt.ToString('o')
        durationSeconds = [math]::Round($stopwatch.Elapsed.TotalSeconds, 1)
        pythonExe       = $python.Path
        pythonVersion   = $python.Version
        command         = $commandText
        exitCode        = $exitCode
        artifact        = [pscustomobject]@{
            path     = $artifactPath
            size     = $pe.Size
            sha256   = $artifactHash
            codeSha256 = (Get-Th08CodeHash -Path $artifactPath)
            pe       = [pscustomobject]@{
                machine   = $pe.Machine
                subsystem = $pe.Subsystem
                imageBase = $pe.ImageBase
                isPe32    = $pe.IsPe32
            }
        }
        logTail         = $tail
    }

    Write-Th08Report -ReportsDir $Context.ReportsDir -Name ("build-" + $BuildType + ".json") -Object $report | Out-Null
    return $report
}

# ---------------------------------------------------------------------------
# S3-3: runtime staging (E03/E06)
# ---------------------------------------------------------------------------

function Write-Th08WindowedConfig {
    param(
        [Parameter(Mandatory = $true)][string]$SourceCfg,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    if (-not (Test-Path -LiteralPath $SourceCfg -PathType Leaf)) {
        throw ("TH08-E06: config source missing (path=$SourceCfg)")
    }
    $bytes = [System.IO.File]::ReadAllBytes($SourceCfg)
    if ($bytes.Length -ne $script:Th08CfgLength) {
        throw ("TH08-E06: config source length invalid (path=$SourceCfg; expected=$($script:Th08CfgLength); actual=" + $bytes.Length + ')')
    }
    $version = [System.BitConverter]::ToUInt32($bytes, 0x14)
    if ($version -ne $script:Th08CfgVersion) {
        throw ("TH08-E06: config version unsupported (path=$SourceCfg; expected=0x" + $script:Th08CfgVersion.ToString('X8') + '; actual=0x' + $version.ToString('X8') + ')')
    }
    $bytes[$script:Th08CfgWindowedOffset] = 1
    $directory = Split-Path -Parent $Destination
    if ($directory -and -not (Test-Path -LiteralPath $directory)) {
        New-Item -ItemType Directory -Force -Path $directory | Out-Null
    }
    [System.IO.File]::WriteAllBytes($Destination, $bytes)

    return [pscustomobject]@{
        Path     = $Destination
        Length   = $bytes.Length
        Windowed = [int]$bytes[$script:Th08CfgWindowedOffset]
        Version  = ('0x' + $version.ToString('X8'))
        Sha256   = (Get-Th08Sha256 -Path $Destination)
    }
}

function Get-Th08RuntimeManifest {
    param(
        [Parameter(Mandatory = $true)][string]$DataDir,
        [Parameter(Mandatory = $true)][string]$ExePath,
        [string]$CfgSource,
        [switch]$IncludeScoreDat
    )

    if (-not (Test-Path -LiteralPath $DataDir)) {
        throw ("TH08-E03: retail data directory missing (path=$DataDir)")
    }

    $files = @()
    $required = @(
        @{ Name = 'th08.dat'; Size = $script:Th08DataSize },
        @{ Name = 'thbgm.dat'; Size = $script:Th08BgmSize }
    )
    foreach ($entry in $required) {
        $source = Join-Path $DataDir $entry.Name
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            throw ("TH08-E03: retail data file missing (path=$source)")
        }
        $size = (Get-Item -LiteralPath $source).Length
        if ($size -ne $entry.Size) {
            throw ("TH08-E03: retail data size mismatch (path=$source; expected=" + $entry.Size + '; actual=' + $size + ')')
        }
        $files += [pscustomobject]@{
            Name   = $entry.Name
            Size   = $size
            Sha256 = (Get-Th08Sha256 -Path $source)
            Source = $source
            Action = 'copy'
        }
    }

    if ($IncludeScoreDat) {
        $scoreSource = Join-Path $DataDir 'score.dat'
        if (Test-Path -LiteralPath $scoreSource -PathType Leaf) {
            $files += [pscustomobject]@{
                Name    = 'score.dat'
                Size    = (Get-Item -LiteralPath $scoreSource).Length
                Sha256  = (Get-Th08Sha256 -Path $scoreSource)
                Source  = $scoreSource
                Action  = 'copy'
                Mutable = $true
            }
        }
    }

    if (-not $CfgSource) { $CfgSource = Join-Path $DataDir 'th08.cfg' }
    if (-not (Test-Path -LiteralPath $CfgSource -PathType Leaf)) {
        throw ("TH08-E06: config source missing (path=$CfgSource)")
    }
    $cfgBytes = [System.IO.File]::ReadAllBytes($CfgSource)
    if ($cfgBytes.Length -ne $script:Th08CfgLength) {
        throw ("TH08-E06: config source length invalid (path=$CfgSource; expected=$($script:Th08CfgLength); actual=" + $cfgBytes.Length + ')')
    }

    if (-not (Test-Path -LiteralPath $ExePath -PathType Leaf)) {
        throw ("TH08-E05: runtime executable missing (path=$ExePath)")
    }
    $pe = Assert-Th08PeInfo -Path $ExePath

    return [pscustomobject]@{
        DataDir = $DataDir
        Files   = $files
        Cfg     = [pscustomobject]@{ Source = $CfgSource; Length = $cfgBytes.Length }
        Exe     = [pscustomobject]@{ Path = $ExePath; Size = $pe.Size; Sha256 = (Get-Th08Sha256 -Path $ExePath) }
    }
}

function Initialize-Th08Runtime {
    param(
        [Parameter(Mandatory = $true)][object]$Context,
        [Parameter(Mandatory = $true)][string]$DataDir,
        [string]$CfgSource,
        [switch]$Refresh,
        [switch]$UseHardLink,
        [switch]$IncludeScoreDat
    )

    $exePath = Join-Path $Context.BuildDir 'th08.exe'
    $manifestInput = Get-Th08RuntimeManifest -DataDir $DataDir -ExePath $exePath -CfgSource $CfgSource -IncludeScoreDat:$IncludeScoreDat

    foreach ($name in @('play', 'play\replay', 'play\backup', 'play\evidence')) {
        $directory = Join-Path $Context.BuildDir $name
        if (-not (Test-Path -LiteralPath $directory)) {
            New-Item -ItemType Directory -Force -Path $directory | Out-Null
        }
    }

    $copied = 0
    $skipped = 0
    $results = @()
    foreach ($file in $manifestInput.Files) {
        $destination = Join-Path $Context.PlayDir $file.Name
        $needsCopy = $true
        if (-not $Refresh -and (Test-Path -LiteralPath $destination -PathType Leaf)) {
            $existingSize = (Get-Item -LiteralPath $destination).Length
            if ($existingSize -eq $file.Size) {
                $existingHash = Get-Th08Sha256 -Path $destination
                if ($existingHash -eq $file.Sha256) { $needsCopy = $false }
            }
        }
        $action = 'skipped'
        if ($needsCopy) {
            $action = 'copy'
            $useLink = $UseHardLink -and ($file.Name -eq 'th08.dat' -or $file.Name -eq 'thbgm.dat')
            if ($useLink) {
                try {
                    if (Test-Path -LiteralPath $destination) { Remove-Item -LiteralPath $destination -Force }
                    New-Item -ItemType HardLink -Path $destination -Target $file.Source -ErrorAction Stop | Out-Null
                    $action = 'hardlink'
                } catch {
                    Copy-Item -LiteralPath $file.Source -Destination $destination -Force
                    $action = 'copy'
                }
            } else {
                Copy-Item -LiteralPath $file.Source -Destination $destination -Force
            }
            $copied++
        } else {
            $skipped++
        }
        $results += [pscustomobject]@{
            Name    = $file.Name
            Size    = (Get-Item -LiteralPath $destination).Length
            Sha256  = (Get-Th08Sha256 -Path $destination)
            Source  = $file.Source
            Action  = $action
            Mutable = [bool]$file.Mutable
        }
    }

    $cfgInfo = Write-Th08WindowedConfig -SourceCfg $manifestInput.Cfg.Source -Destination (Join-Path $Context.PlayDir 'th08.cfg')

    $playExe = Join-Path $Context.PlayDir 'th08.exe'
    Copy-Item -LiteralPath $manifestInput.Exe.Path -Destination $playExe -Force

    $manifest = [pscustomobject]@{
        schema    = 'th08-runtime-manifest/1'
        createdAt = (Get-Date).ToString('o')
        note      = 'hashes are an assembly-time snapshot; entries with mutable=true are rewritten by the game at runtime'
        dataDir   = $DataDir
        files     = $results
        cfg       = [pscustomobject]@{ Length = $cfgInfo.Length; Windowed = $cfgInfo.Windowed; Version = $cfgInfo.Version }
        exe       = [pscustomobject]@{ buildType = 'bugfix'; Size = (Get-Item -LiteralPath $playExe).Length; Sha256 = (Get-Th08Sha256 -Path $playExe) }
    }

    Write-Th08JsonFile -Path (Join-Path $Context.PlayDir 'runtime-manifest.json') -Object $manifest | Out-Null
    return $manifest
}

function Assert-Th08Runtime {
    param([Parameter(Mandatory = $true)][object]$Context)

    $checks = @()
    $failures = @()

    $expected = @{
        'th08.exe'   = $null
        'th08.dat'   = $script:Th08DataSize
        'thbgm.dat'  = $script:Th08BgmSize
        'th08.cfg'   = $script:Th08CfgLength
    }
    foreach ($name in $expected.Keys) {
        $path = Join-Path $Context.PlayDir $name
        $ok = (Test-Path -LiteralPath $path -PathType Leaf)
        $size = $null
        if ($ok) {
            $size = (Get-Item -LiteralPath $path).Length
            if ($null -ne $expected[$name] -and $size -ne $expected[$name]) {
                $ok = $false
                $failures += ("size:" + $name)
            }
        } else {
            $failures += ("missing:" + $name)
        }
        $checks += [pscustomobject]@{ Item = $name; Ok = $ok; Size = $size }
    }

    foreach ($directory in @('replay', 'backup', 'evidence')) {
        $path = Join-Path $Context.PlayDir $directory
        $ok = (Test-Path -LiteralPath $path -PathType Container)
        if (-not $ok) { $failures += ("missing-dir:" + $directory) }
        $checks += [pscustomobject]@{ Item = $directory; Ok = $ok; Size = $null }
    }

    $cfgPath = Join-Path $Context.PlayDir 'th08.cfg'
    if (Test-Path -LiteralPath $cfgPath -PathType Leaf) {
        $cfgBytes = [System.IO.File]::ReadAllBytes($cfgPath)
        $windowed = $null
        $version = $null
        if ($cfgBytes.Length -eq $script:Th08CfgLength) {
            $windowed = [int]$cfgBytes[$script:Th08CfgWindowedOffset]
            $version = [System.BitConverter]::ToUInt32($cfgBytes, 0x14)
        }
        $cfgOk = ($cfgBytes.Length -eq $script:Th08CfgLength) -and ($windowed -eq 1) -and ($version -eq $script:Th08CfgVersion)
        if (-not $cfgOk) { $failures += 'cfg-fields' }
        $checks += [pscustomobject]@{ Item = 'cfg-fields'; Ok = $cfgOk; Size = $cfgBytes.Length }
    }

    $exeHashChecked = $false
    $playExePath = Join-Path $Context.PlayDir 'th08.exe'
    $buildReportPath = Join-Path $Context.ReportsDir 'build-bugfix.json'
    if ((Test-Path -LiteralPath $buildReportPath -PathType Leaf) -and (Test-Path -LiteralPath $playExePath -PathType Leaf)) {
        $buildReport = Get-Content -LiteralPath $buildReportPath -Raw | ConvertFrom-Json
        $expectedHash = $buildReport.artifact.sha256
        if ($expectedHash) {
            $actualHash = Get-Th08Sha256 -Path $playExePath
            $hashOk = ($actualHash -eq $expectedHash.ToUpperInvariant())
            if (-not $hashOk) { $failures += 'exe-hash' }
            $checks += [pscustomobject]@{ Item = 'exe-hash'; Ok = $hashOk; Size = $null }
            $exeHashChecked = $true
        }
    }

    if ($failures.Count -gt 0) {
        throw ("TH08-E06: runtime directory invalid (path=$($Context.PlayDir); failures=" + ($failures -join ',') + ')')
    }

    return [pscustomobject]@{ Ok = $true; Checks = $checks; ExeHashChecked = $exeHashChecked }
}

# ---------------------------------------------------------------------------
# S3-4: key scripts, screenshots, log audit, run (E08..E12)
# ---------------------------------------------------------------------------

function Read-Th08KeyScript {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw ("TH08-E12: key script missing (path=$Path)")
    }
    $raw = Get-Content -LiteralPath $Path -Raw
    try {
        $parsed = $raw | ConvertFrom-Json
    } catch {
        throw ("TH08-E12: key script is not valid json (path=$Path)")
    }
    if ($parsed.schema -ne 'th08-key-script/1') {
        throw ("TH08-E12: key script schema unsupported (path=$Path; schema=$($parsed.schema))")
    }
    if (-not $parsed.name) {
        throw ("TH08-E12: key script has no name (path=$Path)")
    }
    if (-not $parsed.steps -or @($parsed.steps).Count -eq 0) {
        throw ("TH08-E12: key script has no steps (path=$Path)")
    }

    $steps = @()
    $index = 0
    foreach ($step in @($parsed.steps)) {
        $index++
        $action = [string]$step.action
        if ($action -notin @('wait', 'key', 'hold', 'shot', 'close')) {
            throw ("TH08-E12: unknown action (path=$Path; step=$index; action=$action)")
        }
        $normalized = [pscustomobject]@{
            Index    = $index
            Action   = $action
            Ms       = $null
            Vk       = $null
            HoldMs   = $null
            ShotName = $null
        }
        switch ($action) {
            'wait' {
                if ($null -eq $step.ms) { throw ("TH08-E12: wait step requires ms (path=$Path; step=$index)") }
                $ms = [int]$step.ms
                if ($ms -lt 50 -or $ms -gt 30000) { throw ("TH08-E12: wait ms out of range (path=$Path; step=$index; ms=$ms)") }
                $normalized.Ms = $ms
            }
            'key' {
                if ($null -eq $step.vk) { throw ("TH08-E12: key step requires vk (path=$Path; step=$index)") }
                $vk = [int]$step.vk
                if ($vk -lt 1 -or $vk -gt 255) { throw ("TH08-E12: key vk out of range (path=$Path; step=$index; vk=$vk)") }
                $holdMs = 120
                if ($null -ne $step.holdMs) { $holdMs = [int]$step.holdMs }
                if ($holdMs -lt 30 -or $holdMs -gt 2000) { throw ("TH08-E12: key holdMs out of range (path=$Path; step=$index; holdMs=$holdMs)") }
                $normalized.Vk = $vk
                $normalized.HoldMs = $holdMs
            }
            'hold' {
                if ($null -eq $step.vk) { throw ("TH08-E12: hold step requires vk (path=$Path; step=$index)") }
                $vk = [int]$step.vk
                if ($vk -lt 1 -or $vk -gt 255) { throw ("TH08-E12: hold vk out of range (path=$Path; step=$index; vk=$vk)") }
                if ($null -eq $step.ms) { throw ("TH08-E12: hold step requires ms (path=$Path; step=$index)") }
                $ms = [int]$step.ms
                if ($ms -lt 50 -or $ms -gt 30000) { throw ("TH08-E12: hold ms out of range (path=$Path; step=$index; ms=$ms)") }
                $normalized.Vk = $vk
                $normalized.Ms = $ms
            }
            'shot' {
                if (-not $step.name) { throw ("TH08-E12: shot step requires name (path=$Path; step=$index)") }
            }
        }
        if ($step.shot) {
            if ([string]$step.shot -notmatch '^[0-9A-Za-z_-]{1,40}$') {
                throw ("TH08-E12: invalid shot name (path=$Path; step=$index; shot=$($step.shot))")
            }
            $normalized.ShotName = [string]$step.shot
        }
        if ($action -eq 'shot') { $normalized.ShotName = [string]$step.name }
        if ($action -eq 'close' -and $index -ne @($parsed.steps).Count) {
            throw ("TH08-E12: close must be the last step (path=$Path; step=$index)")
        }
        $steps += $normalized
    }

    return [pscustomobject]@{ Path = $Path; Name = [string]$parsed.name; Steps = $steps }
}

function Get-Th08ImageStats {
    param([Parameter(Mandatory = $true)][string]$Path)

    Add-Type -AssemblyName System.Drawing -ErrorAction SilentlyContinue
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw ("TH08-E11: screenshot missing (path=$Path)")
    }
    $source = [System.Drawing.Image]::FromFile((Resolve-Path -LiteralPath $Path).Path)
    try {
        $width = $source.Width
        $height = $source.Height
        $sampleWidth = 64
        $sampleHeight = 48
        $sample = New-Object System.Drawing.Bitmap $sampleWidth, $sampleHeight
        $graphics = [System.Drawing.Graphics]::FromImage($sample)
        $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
        $graphics.DrawImage($source, 0, 0, $sampleWidth, $sampleHeight)
        $graphics.Dispose()

        $colors = New-Object 'System.Collections.Generic.HashSet[int]'
        $lumaSum = 0.0
        $count = 0
        for ($y = 0; $y -lt $sampleHeight; $y++) {
            for ($x = 0; $x -lt $sampleWidth; $x++) {
                $pixel = $sample.GetPixel($x, $y)
                [void]$colors.Add($pixel.ToArgb())
                $lumaSum += (0.299 * $pixel.R + 0.587 * $pixel.G + 0.114 * $pixel.B)
                $count++
            }
        }
        $sample.Dispose()

        return [pscustomobject]@{
            Path           = $Path
            Width          = $width
            Height         = $height
            DistinctColors = $colors.Count
            MeanLuma       = [math]::Round(($lumaSum / $count), 2)
        }
    } finally {
        $source.Dispose()
    }
}

function Test-Th08ImageBlank {
    param([Parameter(Mandatory = $true)][object]$Stats)

    $blank = $false
    $reason = @()
    if ($Stats.DistinctColors -lt $script:Th08BlankMinDistinctColors) {
        $blank = $true
        $reason += ('distinctColors=' + $Stats.DistinctColors + ' < ' + $script:Th08BlankMinDistinctColors)
    }
    if ($Stats.MeanLuma -lt $script:Th08BlankMinMeanLuma) {
        $blank = $true
        $reason += ('meanLuma=' + $Stats.MeanLuma + ' < ' + $script:Th08BlankMinMeanLuma)
    }
    return [pscustomobject]@{ Blank = $blank; Reason = ($reason -join ';') }
}

function Get-Th08LogText {
    param([Parameter(Mandatory = $true)][string]$LogPath)

    if (-not (Test-Path -LiteralPath $LogPath -PathType Leaf)) { return $null }
    $bytes = [System.IO.File]::ReadAllBytes($LogPath)
    if ($bytes.Length -eq 0) { return '' }
    $encoding = [System.Text.Encoding]::GetEncoding(932)
    return $encoding.GetString($bytes)
}

function Test-Th08LogAudit {
    param([Parameter(Mandatory = $true)][string]$LogPath)

    $text = Get-Th08LogText -LogPath $LogPath
    if ($null -eq $text) {
        return [pscustomobject]@{ Pass = $false; RequiredMissing = @('log.txt'); ForbiddenFound = @(); Length = 0; Reason = 'missing' }
    }
    if ($text.Length -eq 0) {
        return [pscustomobject]@{ Pass = $false; RequiredMissing = @('non-empty'); ForbiddenFound = @(); Length = 0; Reason = 'empty' }
    }

    $required = @('DirectSound', 'KeyBoard')
    $requiredMissing = @()
    foreach ($token in $required) {
        if ($text -notmatch [regex]::Escape($token)) { $requiredMissing += $token }
    }
    $forbidden = @('Fatal', '致命的', 'ERROR:', 'Error :')
    $forbiddenFound = @()
    foreach ($token in $forbidden) {
        if ($text -match [regex]::Escape($token)) { $forbiddenFound += $token }
    }

    $pass = ($requiredMissing.Count -eq 0) -and ($forbiddenFound.Count -eq 0)
    $reason = ''
    if (-not $pass) {
        $reason = 'requiredMissing=' + ($requiredMissing -join ',') + ';forbiddenFound=' + ($forbiddenFound -join ',')
    }
    return [pscustomobject]@{
        Pass           = $pass
        RequiredMissing = $requiredMissing
        ForbiddenFound = $forbiddenFound
        Length         = $text.Length
        Reason         = $reason
    }
}

function Get-Th08LiveWindowHandle {
    param(
        [Parameter(Mandatory = $true)][object]$Process,
        [IntPtr]$Fallback = [IntPtr]::Zero
    )

    $candidate = [Th08Native]::MainWindow([uint32]$Process.Id)
    if ($candidate -ne [IntPtr]::Zero) { return $candidate }
    return $Fallback
}

function Save-Th08Screenshot {
    param(
        [Parameter(Mandatory = $true)][IntPtr]$WindowHandle,
        [Parameter(Mandatory = $true)][string]$Path
    )

    Add-Type -AssemblyName System.Drawing -ErrorAction SilentlyContinue
    $geometry = [Th08Native]::ClientGeometry($WindowHandle)
    $originX = $geometry[0]
    $originY = $geometry[1]
    $width = $geometry[2]
    $height = $geometry[3]
    if ($width -lt 16 -or $height -lt 16) {
        throw ("TH08-E11: client area too small (width=$width; height=$height)")
    }
    $directory = Split-Path -Parent $Path
    if ($directory -and -not (Test-Path -LiteralPath $directory)) {
        New-Item -ItemType Directory -Force -Path $directory | Out-Null
    }

    $bitmap = New-Object System.Drawing.Bitmap $width, $height
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    try {
        $graphics.CopyFromScreen($originX, $originY, 0, 0, (New-Object System.Drawing.Size $width, $height))
        $bitmap.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
    } finally {
        $graphics.Dispose()
        $bitmap.Dispose()
    }
    return $Path
}

function Invoke-Th08Run {
    param(
        [Parameter(Mandatory = $true)][object]$Context,
        [Parameter(Mandatory = $true)][string]$ProfilePath,
        [string]$Tag = '',
        [int]$WindowTimeoutSeconds = 30,
        [int]$TotalTimeoutSeconds = 60,
        [int]$ExitWaitSeconds = 15
    )

    $scriptInfo = Read-Th08KeyScript -Path $ProfilePath
    $runtimeCheck = Assert-Th08Runtime -Context $Context

    $existing = @(Get-Process -Name 'th08' -ErrorAction SilentlyContinue)
    if ($existing.Count -gt 0) {
        throw ("TH08-E10: another th08 process is already running (pids=" + (($existing | ForEach-Object { $_.Id }) -join ',') + ')')
    }

    if (-not (Test-Path -LiteralPath $Context.EvidenceDir)) {
        New-Item -ItemType Directory -Force -Path $Context.EvidenceDir | Out-Null
    }

    $runId = (Get-Date).ToString('yyyyMMdd-HHmmss')
    $reportPath = Join-Path $Context.EvidenceDir ($runId + '_report.json')
    $suffix = 1
    while (Test-Path -LiteralPath $reportPath) {
        $suffix++
        $runId = (Get-Date).ToString('yyyyMMdd-HHmmss') + '-' + $suffix
        $reportPath = Join-Path $Context.EvidenceDir ($runId + '_report.json')
    }

    $playExe = Join-Path $Context.PlayDir 'th08.exe'
    $exeHash = Get-Th08Sha256 -Path $playExe
    $exeSize = (Get-Item -LiteralPath $playExe).Length

    $startedAt = Get-Date
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $process = $null
    $windowHandle = [IntPtr]::Zero
    $firstWindowMs = $null
    $windowTitle = ''
    $clientWidth = 0
    $clientHeight = 0
    $stepsReport = @()
    $shots = @()
    $exitCode = $null
    $exitMethod = 'natural'
    $exitForced = $false
    $failure = $null
    $foregroundChecks = 0

    try {
        # A stale log from a previous launch would poison the audit of this run.
        $staleLog = Join-Path $Context.PlayDir 'log.txt'
        if (Test-Path -LiteralPath $staleLog) { Remove-Item -LiteralPath $staleLog -Force }

        # The application is DPI unaware: without the compatibility layer Windows
        # upscales its 640x480 window to 125%, which breaks 1:1 pixel evidence.
        $previousCompatLayer = $env:__COMPAT_LAYER
        $env:__COMPAT_LAYER = 'HIGHDPIAWARE'
        try {
            $process = Start-Process -FilePath $playExe -WorkingDirectory $Context.PlayDir -PassThru
        } finally {
            if ($null -eq $previousCompatLayer) { Remove-Item Env:__COMPAT_LAYER -ErrorAction SilentlyContinue }
            else { $env:__COMPAT_LAYER = $previousCompatLayer }
        }

        $windowDeadline = (Get-Date).AddSeconds($WindowTimeoutSeconds)
        while ((Get-Date) -lt $windowDeadline) {
            Start-Sleep -Milliseconds 250
            $process.Refresh()
            if ($process.HasExited) {
                throw ("TH08-E10: game exited before creating a window (runId=$runId; exitCode=" + $process.ExitCode + ')')
            }
            $candidate = [Th08Native]::MainWindow([uint32]$process.Id)
            if ($candidate -ne [IntPtr]::Zero -and
                [Th08Native]::ClientAreaWidth($candidate) -ge 16 -and
                [Th08Native]::ClientAreaHeight($candidate) -ge 16) {
                $windowHandle = $candidate
                if ($null -eq $firstWindowMs) { $firstWindowMs = [math]::Round($stopwatch.Elapsed.TotalMilliseconds, 0) }
                break
            }
        }
        if ($windowHandle -eq [IntPtr]::Zero) {
            throw ("TH08-E08: game window did not appear within $WindowTimeoutSeconds seconds (runId=$runId)")
        }

        $titleBytes = [Th08Native]::WindowTitleBytes($windowHandle)
        $windowTitle = ([System.Text.Encoding]::GetEncoding(932).GetString($titleBytes)).TrimEnd([char]0, [char]13, [char]10)
        $windowTitleUnicode = [Th08Native]::WindowTitle($windowHandle)
        $clientWidth = [Th08Native]::ClientAreaWidth($windowHandle)
        $clientHeight = [Th08Native]::ClientAreaHeight($windowHandle)

        $focused = $false
        for ($attempt = 0; $attempt -lt 3; $attempt++) {
            $foregroundChecks++
            if ([Th08Native]::Focus($windowHandle)) { $focused = $true; break }
            Start-Sleep -Milliseconds 300
        }
        if (-not $focused) {
            throw ("TH08-E08: could not bring the game window to the foreground (runId=$runId; pid=" + $process.Id + ')')
        }

        $totalDeadline = (Get-Date).AddSeconds($TotalTimeoutSeconds)
        # A profile without a close step expects the game to exit by itself
        # (title-menu Quit). That exit is success, not a lost process.
        $expectsNaturalExit = @($scriptInfo.Steps | Where-Object { $_.Action -eq 'close' }).Count -eq 0
        $naturalExit = $false

        foreach ($step in $scriptInfo.Steps) {
            if ((Get-Date) -gt $totalDeadline) {
                throw ("TH08-E10: run exceeded the total time budget (runId=$runId; seconds=$TotalTimeoutSeconds)")
            }
            if ($process.HasExited) {
                if ($expectsNaturalExit) {
                    $naturalExit = $true
                    $exitMethod = 'natural'
                    try { $exitCode = $process.ExitCode } catch { $exitCode = 0 }
                    break
                }
                throw ("TH08-E10: game exited during step execution (runId=$runId; step=$($step.Index); exitCode=" + $process.ExitCode + ')')
            }

            # The game recreates its window during start-up/display changes, so the
            # handle is re-resolved from the process before every step.
            $liveWindow = [Th08Native]::MainWindow([uint32]$process.Id)
            if ($liveWindow -ne [IntPtr]::Zero) { $windowHandle = $liveWindow }

            $stepInfo = [pscustomobject]@{
                index      = $step.Index
                action     = $step.Action
                ok         = $true
                details    = ''
                screenshot = $null
            }

            switch ($step.Action) {
                'wait' {
                    $waitUntil = (Get-Date).AddMilliseconds($step.Ms)
                    while ((Get-Date) -lt $waitUntil) {
                        Start-Sleep -Milliseconds 150
                        $process.Refresh()
                        if ($process.HasExited) {
                            if ($expectsNaturalExit) {
                                $naturalExit = $true
                                $exitMethod = 'natural'
                                try { $exitCode = $process.ExitCode } catch { $exitCode = 0 }
                                break
                            }
                            throw ("TH08-E10: game exited during wait (runId=$runId; step=$($step.Index); exitCode=" + $process.ExitCode + ')')
                        }
                    }
                    if ($naturalExit) { $stepInfo.details = 'process exited naturally during wait' }
                    else { $stepInfo.details = ('waited ' + $step.Ms + 'ms') }
                }
                'key' {
                    $windowHandle = Get-Th08LiveWindowHandle -Process $process -Fallback $windowHandle
                    $foregroundChecks++
                    if (-not [Th08Native]::Focus($windowHandle)) {
                        throw ("TH08-E08: foreground lost before key injection (runId=$runId; step=$($step.Index); vk=" + $step.Vk + ')')
                    }
                    [void][Th08Native]::SendKey([uint16]$step.Vk, $false)
                    Start-Sleep -Milliseconds $step.HoldMs
                    [void][Th08Native]::SendKey([uint16]$step.Vk, $true)
                    $stepInfo.details = ('key vk=' + $step.Vk + ' holdMs=' + $step.HoldMs)
                }
                'hold' {
                    $windowHandle = Get-Th08LiveWindowHandle -Process $process -Fallback $windowHandle
                    $foregroundChecks++
                    if (-not [Th08Native]::Focus($windowHandle)) {
                        throw ("TH08-E08: foreground lost before key hold (runId=$runId; step=$($step.Index); vk=" + $step.Vk + ')')
                    }
                    [void][Th08Native]::SendKey([uint16]$step.Vk, $false)
                    $holdUntil = (Get-Date).AddMilliseconds($step.Ms)
                    while ((Get-Date) -lt $holdUntil) {
                        Start-Sleep -Milliseconds 150
                        $process.Refresh()
                        if ($process.HasExited) {
                            [void][Th08Native]::SendKey([uint16]$step.Vk, $true)
                            if ($expectsNaturalExit) {
                                $naturalExit = $true
                                $exitMethod = 'natural'
                                try { $exitCode = $process.ExitCode } catch { $exitCode = 0 }
                                break
                            }
                            throw ("TH08-E10: game exited during key hold (runId=$runId; step=$($step.Index); exitCode=" + $process.ExitCode + ')')
                        }
                    }
                    [void][Th08Native]::SendKey([uint16]$step.Vk, $true)
                    $stepInfo.details = ('hold vk=' + $step.Vk + ' ms=' + $step.Ms)
                }
                'shot' {
                    $windowHandle = Get-Th08LiveWindowHandle -Process $process -Fallback $windowHandle
                    $foregroundChecks++
                    if (-not [Th08Native]::Focus($windowHandle)) {
                        throw ("TH08-E08: foreground lost before screenshot (runId=$runId; step=$($step.Index))")
                    }
                    Start-Sleep -Milliseconds 150
                    $shotPath = Join-Path $Context.EvidenceDir ($runId + '_' + $step.ShotName + '.png')
                    Save-Th08Screenshot -WindowHandle $windowHandle -Path $shotPath | Out-Null
                    $stats = Get-Th08ImageStats -Path $shotPath
                    $blankCheck = Test-Th08ImageBlank -Stats $stats
                    if ($blankCheck.Blank) {
                        throw ("TH08-E11: screenshot appears blank (runId=$runId; shot=$($step.ShotName); " + $blankCheck.Reason + ')')
                    }
                    $shots += [pscustomobject]@{
                        name           = $step.ShotName
                        path           = $shotPath
                        width          = $stats.Width
                        height         = $stats.Height
                        distinctColors = $stats.DistinctColors
                        meanLuma       = $stats.MeanLuma
                    }
                    $stepInfo.screenshot = $shotPath
                    $stepInfo.details = ('shot ' + $step.ShotName)
                }
                'close' {
                    $windowHandle = Get-Th08LiveWindowHandle -Process $process -Fallback $windowHandle
                    [void][Th08Native]::CloseWindow($windowHandle)
                    $exitMethod = 'wm-close'
                    $stepInfo.details = 'WM_CLOSE sent'
                }
            }

            if ($step.ShotName -and $step.Action -ne 'shot') {
                $windowHandle = Get-Th08LiveWindowHandle -Process $process -Fallback $windowHandle
                $shotPath = Join-Path $Context.EvidenceDir ($runId + '_' + $step.ShotName + '.png')
                Save-Th08Screenshot -WindowHandle $windowHandle -Path $shotPath | Out-Null
                $stats = Get-Th08ImageStats -Path $shotPath
                $blankCheck = Test-Th08ImageBlank -Stats $stats
                if ($blankCheck.Blank) {
                    throw ("TH08-E11: screenshot appears blank (runId=$runId; shot=$($step.ShotName); " + $blankCheck.Reason + ')')
                }
                $shots += [pscustomobject]@{
                    name           = $step.ShotName
                    path           = $shotPath
                    width          = $stats.Width
                    height         = $stats.Height
                    distinctColors = $stats.DistinctColors
                    meanLuma       = $stats.MeanLuma
                }
                $stepInfo.screenshot = $shotPath
            }

            $stepsReport += $stepInfo

            if ($naturalExit) { break }
        }

        if ($naturalExit) {
            $process.Refresh()
            if (-not $process.HasExited) {
                [void]$process.WaitForExit($ExitWaitSeconds * 1000)
                $process.Refresh()
            }
            try { $exitCode = $process.ExitCode } catch { }
        } else {
            if ($exitMethod -ne 'wm-close') { $exitMethod = 'natural' }

            $process.Refresh()
            if (-not $process.WaitForExit($ExitWaitSeconds * 1000)) {
                $exitForced = $true
                $exitMethod = 'forced'
                Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
                $process.WaitForExit(5000) | Out-Null
            }
            $process.Refresh()
            $exitCode = $process.ExitCode
        }
    } catch {
        $failure = $_
        if ($process -and -not $process.HasExited) {
            $exitForced = $true
            $exitMethod = 'forced'
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            try { $process.WaitForExit(5000) | Out-Null } catch { }
        }
        if ($process) {
            $process.Refresh()
            try { $exitCode = $process.ExitCode } catch { $exitCode = $null }
        }
    } finally {
        $stopwatch.Stop()
        $logPath = Join-Path $Context.PlayDir 'log.txt'
        $logAudit = $null
        $logCopyPath = $null
        if (Test-Path -LiteralPath $logPath -PathType Leaf) {
            $logCopyPath = Join-Path $Context.EvidenceDir ($runId + '_log.txt')
            Copy-Item -LiteralPath $logPath -Destination $logCopyPath -Force
            $logAudit = Test-Th08LogAudit -LogPath $logCopyPath
        } else {
            $logAudit = [pscustomobject]@{ Pass = $false; RequiredMissing = @('log.txt'); ForbiddenFound = @(); Length = 0; Reason = 'missing' }
        }

        $requiredShots = @()
        if ($scriptInfo.Name -eq 'l2-gameplay') {
            $requiredShots = @('01-title', '02-difficulty', '03-character', '04-stage1-start', '05-stage1-move', '06-stage1-play', '07-stage1-later', '08-pause')
        } else {
            $requiredShots = @($scriptInfo.Steps | Where-Object { $_.ShotName } | ForEach-Object { $_.ShotName })
        }
        $capturedNames = @($shots | ForEach-Object { $_.name })
        $missingShots = @($requiredShots | Where-Object { $_ -notin $capturedNames })

        $failureText = $null
        if ($failure) { $failureText = $failure.Exception.Message }

        if (-not $failure) {
            if ($missingShots.Count -gt 0) {
                $failure = 'missing screenshots: ' + ($missingShots -join ',')
            } elseif (-not $logAudit.Pass) {
                $failure = 'log audit failed: ' + $logAudit.Reason
            } elseif ($exitForced) {
                $failure = 'game process was terminated forcefully'
            } elseif ($null -eq $exitCode -or $exitCode -ne 0) {
                $failure = 'game exited with code ' + $exitCode
            }
            if ($failure -and -not $failureText) { $failureText = [string]$failure }
        }

        $report = [pscustomobject]@{
            schema       = 'th08-run-report/1'
            runId        = $runId
            tag          = $Tag
            profile      = $scriptInfo.Name
            startedAt    = $startedAt.ToString('o')
            exe          = [pscustomobject]@{ path = $playExe; size = $exeSize; sha256 = $exeHash }
            process      = [pscustomobject]@{
                pid           = $(if ($process) { $process.Id } else { $null })
                firstWindowMs = $firstWindowMs
                windowTitle   = $windowTitle
                windowTitleUnicode = $windowTitleUnicode
                clientWidth   = $clientWidth
                clientHeight  = $clientHeight
            }
            steps        = $stepsReport
            shots        = $shots
            exit         = [pscustomobject]@{ code = $exitCode; forced = $exitForced; method = $exitMethod }
            log          = [pscustomobject]@{
                path   = $logCopyPath
                sha256 = $(if ($logCopyPath) { Get-Th08Sha256 -Path $logCopyPath } else { $null })
                audit  = [pscustomobject]@{
                    pass            = [bool]$logAudit.Pass
                    requiredMissing = @($logAudit.RequiredMissing)
                    forbiddenFound  = @($logAudit.ForbiddenFound)
                    length          = $logAudit.Length
                    reason          = $logAudit.Reason
                }
            }
            foregroundChecks = $foregroundChecks
            totalSeconds  = [math]::Round($stopwatch.Elapsed.TotalSeconds, 1)
            pass          = (-not $failure)
            failure       = $failureText
        }
        Write-Th08JsonFile -Path $reportPath -Object $report | Out-Null

        if ($failure) {
            $code = 'E10'
            if ($failureText -and ($failureText -match 'TH08-(E[0-9]{2})')) { $code = $matches[1] }
            throw ("TH08-" + $code + ": run failed (runId=$runId; report=$reportPath; reason=$failureText)")
        }
    }

    return (Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json)
}

function Resolve-Th08ExitCode {
    param([string]$Message)

    if ($Message -and ($Message -match 'TH08-(E[0-9]{2})')) {
        switch ($matches[1]) {
            'E01' { return 2 }
            'E02' { return 3 }
            'E03' { return 2 }
            'E04' { return 4 }
            'E05' { return 3 }
            'E06' { return 3 }
            'E07' { return 2 }
            default { return 5 }
        }
    }
    return 1
}

Export-ModuleMember -Function @(
    'Get-Th08Context',
    'Resolve-Th08Python',
    'Resolve-Th08ExitCode',
    'Invoke-Th08CapturedProcess',
    'Get-Th08CodeHash',
    'Test-Th08PythonCandidate',
    'Assert-Th08Prefix',
    'Assert-Th08Target',
    'Get-Th08PeInfo',
    'Assert-Th08PeInfo',
    'Initialize-Th08Environment',
    'Invoke-Th08Build',
    'Write-Th08WindowedConfig',
    'Get-Th08RuntimeManifest',
    'Initialize-Th08Runtime',
    'Assert-Th08Runtime',
    'Read-Th08KeyScript',
    'Get-Th08ImageStats',
    'Test-Th08ImageBlank',
    'Get-Th08LogText',
    'Test-Th08LogAudit',
    'Save-Th08Screenshot',
    'Invoke-Th08Run',
    'Get-Th08Sha256',
    'Write-Th08JsonFile',
    'Write-Th08Report'
)
