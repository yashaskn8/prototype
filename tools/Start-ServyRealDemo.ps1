# Reopen the existing localhost-only, read-only real-data demo.
# No migrations, seeding, account creation, or company-data changes.
param([switch]$Watch, [switch]$Restart, [switch]$ReviewRequest, [switch]$RebuildFromVerifiedDocs)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$privateLocal = 'C:\Users\prana\AppData\Local\Packages\OpenAI.Codex_2p2nqsd0c76g0\LocalCache\Local'
$demoFolder = Join-Path $privateLocal 'Servy\private_evaluations\servy_real_demo_20261031_01'
$accessFile = Join-Path $demoFolder 'real_demo_access.txt'
$credentialFile = Join-Path $privateLocal 'Servy\private_evaluations\servy_eval_credentials\servy_eval_ro.dpapi'

if (!(Test-Path -LiteralPath $accessFile) -or !(Test-Path -LiteralPath $credentialFile)) {
    throw 'The existing private demo identity or SELECT-only database credential is unavailable.'
}

function Test-LocalPort([int]$port) {
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $attempt = $client.BeginConnect('127.0.0.1', $port, $null, $null)
        if (!$attempt.AsyncWaitHandle.WaitOne(1000)) { return $false }
        $client.EndConnect($attempt)
        return $true
    } catch { return $false } finally { $client.Dispose() }
}

# Only stop known real-data demo processes; never kill an unrelated listener.
if ($Restart) {
    foreach ($demoPort in @(8002, 5175)) {
        $listeners = @(Get-NetTCPConnection -LocalPort $demoPort -State Listen -ErrorAction SilentlyContinue)
        foreach ($ownedPid in @($listeners | Select-Object -ExpandProperty OwningProcess -Unique)) {
            $running = Get-CimInstance Win32_Process -Filter "ProcessId = $ownedPid" -ErrorAction SilentlyContinue
            $command = [string]$running.CommandLine
            $isDemo = (($demoPort -eq 8002 -and $command -match 'servy_real_demo\.py') -or
                       ($demoPort -eq 5175 -and $command -match '(?:vite\.js|node_modules[\\/]+vite)'))
            if (!$isDemo) {
                throw "Cannot restart: port $demoPort belongs to a process that is not the Servy real-data demo."
            }
            Stop-Process -Id $ownedPid -Force -ErrorAction Stop
        }
    }
    Start-Sleep -Seconds 1
}

if (!(Test-LocalPort 3306)) {
    $mysql = 'C:\mysql\mysql-8.0.41-winx64\bin\mysqld.exe'
    $config = 'C:\mysql\my.ini'
    if (!(Test-Path -LiteralPath $mysql) -or !(Test-Path -LiteralPath $config)) {
        throw 'The existing local MySQL installation is unavailable.'
    }
    Start-Process -FilePath $mysql -ArgumentList '--defaults-file=C:\mysql\my.ini' -WindowStyle Hidden | Out-Null
    for ($i = 0; $i -lt 15 -and !(Test-LocalPort 3306); $i++) { Start-Sleep -Seconds 1 }
    if (!(Test-LocalPort 3306)) { throw 'Existing MySQL did not become ready; demo was not started.' }
}

$env:LOCALAPPDATA = $privateLocal
$env:SERVY_REAL_DEMO_MODE = '1'
$env:DJANGO_DEBUG = '1'
$env:SERVY_LLM_PROVIDER = 'extractive'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:SERVY_ALLOW_REMOTE_LLM = '0'
$env:SERVY_REAL_DEMO_REBUILD_VERIFIED_SOURCES = if ($RebuildFromVerifiedDocs) { '1' } else { '0' }
if ($ReviewRequest -and $RebuildFromVerifiedDocs) {
    throw 'Run either the read-only review diagnostic or the verified-source rebuild, not both.'
}
if ($RebuildFromVerifiedDocs -and (Test-LocalPort 8002)) {
    throw 'The previous demo backend is running. Use -Restart -RebuildFromVerifiedDocs.'
}

if ($ReviewRequest) {
    $python = Join-Path $projectRoot '.venv\Scripts\python.exe'
    Push-Location -LiteralPath $projectRoot
    try {
        & $python '-B' 'tools\servy_real_demo.py' '--index-review-request'
        if ($LASTEXITCODE -ne 0) { throw 'Read-only Chroma review request failed.' }
    } finally {
        Pop-Location
    }
    return
}

if (!(Test-LocalPort 8002)) {
    Start-Process -FilePath (Join-Path $projectRoot '.venv\Scripts\python.exe') `
        -ArgumentList '-B','-u','tools\servy_real_demo.py' -WorkingDirectory $projectRoot `
        -WindowStyle Hidden -RedirectStandardOutput (Join-Path $demoFolder 'startup.out') `
        -RedirectStandardError (Join-Path $demoFolder 'startup.err') | Out-Null
    $backendChecks = if ($RebuildFromVerifiedDocs) { 240 } else { 30 }
    for ($i = 0; $i -lt $backendChecks -and !(Test-LocalPort 8002); $i++) { Start-Sleep -Seconds 1 }
    if (!(Test-LocalPort 8002)) { throw "Read-only demo backend failed; inspect private startup.err locally." }
}

$env:VITE_SERVY_REAL_DEMO = '1'
if (!(Test-LocalPort 5175)) {
    Start-Process -FilePath 'npm.cmd' -ArgumentList 'run','dev','--','--host','127.0.0.1','--port','5175','--strictPort' `
        -WorkingDirectory (Join-Path $projectRoot 'frontend') -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $demoFolder 'vite.out') `
        -RedirectStandardError (Join-Path $demoFolder 'vite.err') | Out-Null
    for ($i = 0; $i -lt 15 -and !(Test-LocalPort 5175); $i++) { Start-Sleep -Seconds 1 }
    if (!(Test-LocalPort 5175)) { throw "Demo frontend failed; inspect private vite.err locally." }
}

Write-Output 'Real-data demo ready: http://127.0.0.1:5175/'
Write-Output ('Enter the separate code from: ' + $accessFile)
if ($Watch) {
    Write-Output 'Watching safe backend request status. Press Ctrl+C to stop watching; the servers remain running.'
    Get-Content -LiteralPath (Join-Path $demoFolder 'startup.out') -Tail 1 -Wait
}
