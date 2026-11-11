<# Stop only process identities recorded by Start-ServyApp.ps1; never stop MySQL. #>
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$statePath = Join-Path (Join-Path ([System.IO.Path]::GetTempPath()) 'ServyApp') 'current.json'
if (-not (Test-Path -LiteralPath $statePath -PathType Leaf)) {
    Write-Host 'No Servy app process manifest exists; nothing was stopped.'
    return
}

function Get-ProcessSnapshot {
    param([int]$ProcessId)
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId" -ErrorAction SilentlyContinue
    if ($null -eq $process -or [string]::IsNullOrWhiteSpace($process.CommandLine)) { return $null }
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try { $commandHash = [BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($process.CommandLine))).Replace('-', '') }
    finally { $sha.Dispose() }
    [pscustomobject]@{
        PID = [int]$process.ProcessId
        Image = [string]$process.ExecutablePath
        CreatedUtcTicks = ([datetime]$process.CreationDate).ToUniversalTime().Ticks
        CommandHash = $commandHash
    }
}

$state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
$remaining = @()
foreach ($expected in @($state.Processes | Sort-Object Depth -Descending)) {
    $actual = Get-ProcessSnapshot -ProcessId ([int]$expected.PID)
    if ($null -eq $actual) { continue }
    $matches = $actual.Image -ieq [string]$expected.Image -and $actual.CreatedUtcTicks -eq [long]$expected.CreatedUtcTicks -and $actual.CommandHash -eq [string]$expected.CommandHash
    if (-not $matches) {
        Write-Warning "PID $($expected.PID) no longer matches the recorded Servy identity; it was left untouched."
        $remaining += $expected
        continue
    }
    Stop-Process -Id ([int]$expected.PID) -ErrorAction Stop
    try { Wait-Process -Id ([int]$expected.PID) -Timeout 10 -ErrorAction Stop } catch { }
    if ($null -ne (Get-Process -Id ([int]$expected.PID) -ErrorAction SilentlyContinue)) {
        Write-Warning "PID $($expected.PID) did not exit; it was not force-stopped."
        $remaining += $expected
    } else {
        Write-Host "Stopped recorded $($expected.Role) process PID $($expected.PID)."
    }
}

if ($remaining.Count -eq 0) {
    Remove-Item -LiteralPath $statePath -Force
    Write-Host 'Servy app process manifest removed. MySQL was not touched.'
} else {
    Write-Warning 'Some recorded process identities remained; the manifest was preserved for inspection.'
}
