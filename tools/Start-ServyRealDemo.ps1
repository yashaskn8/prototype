# Reopen the existing localhost-only, read-only real-data demo.
# No migrations, seeding, account creation, or company-data changes.
param([switch]$Watch, [switch]$Restart, [switch]$ReviewRequest, [switch]$RebuildFromVerifiedDocs, [switch]$NoLocalChat)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$privateLocal = 'C:\Users\prana\AppData\Local\Packages\OpenAI.Codex_2p2nqsd0c76g0\LocalCache\Local'
$demoFolder = Join-Path $privateLocal 'Servy\private_evaluations\servy_real_demo_20261031_01'
$accessFile = Join-Path $demoFolder 'real_demo_access.txt'
$credentialFile = Join-Path $privateLocal 'Servy\private_evaluations\servy_eval_credentials\servy_eval_ro.dpapi'
# Resolve the Windows known folder before redirecting LOCALAPPDATA to the
# demo's private identity directory. This keeps the ticket SQLite path stable.
$sharedLocalAppData = [System.Environment]::GetFolderPath([System.Environment+SpecialFolder]::LocalApplicationData)
if ([string]::IsNullOrWhiteSpace($sharedLocalAppData)) {
    $sharedLocalAppData = Join-Path $env:USERPROFILE 'AppData\Local'
}
$env:SERVY_LOCAL_TICKET_DB_PATH = Join-Path $sharedLocalAppData 'Servy\local-ticket-workspace.sqlite3'

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
    $backendPortActive = Test-LocalPort 8002
    $frontendPortActive = Test-LocalPort 5175
    $processes = @()
    if ($backendPortActive -or $frontendPortActive) {
        # Process identity is needed only when a demo listener exists. If CIM
        # is unavailable in that case, fail closed before stopping anything.
        $processes = @(Get-CimInstance Win32_Process -ErrorAction Stop)
    }
    $expectedPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
    $frontendRoot = Join-Path $projectRoot 'frontend'
    $backendRoots = @($processes | Where-Object {
        $_.Name -match '^python(?:\.exe)?$' -and
        [string]::Equals($_.ExecutablePath, $expectedPython, [System.StringComparison]::OrdinalIgnoreCase) -and
        $_.CommandLine -match 'servy_real_demo\.py' -and
        $_.CommandLine -match [regex]::Escape($projectRoot)
    })
    $backendChildren = @($processes | Where-Object {
        $_.Name -match '^python(?:\.exe)?$' -and
        $_.CommandLine -match 'servy_real_demo\.py' -and
        @($backendRoots | Select-Object -ExpandProperty ProcessId) -contains $_.ParentProcessId
    })
    $viteServers = @($processes | Where-Object {
        $_.Name -match '^node(?:\.exe)?$' -and
        $_.CommandLine -match [regex]::Escape($frontendRoot) -and
        $_.CommandLine -match 'vite\.js' -and
        $_.CommandLine -match '--port\s+5175\b'
    })
    if ($backendPortActive -and !$backendRoots.Count) {
        throw 'Cannot restart: port 8002 is reachable but no verified project backend process was found.'
    }
    if ($frontendPortActive -and !$viteServers.Count) {
        throw 'Cannot restart: port 5175 is reachable but no verified project Vite process for port 5175 was found.'
    }

    foreach ($child in $backendChildren) {
        Stop-Process -Id $child.ProcessId -Force -ErrorAction Stop
    }
    foreach ($backend in $backendRoots) {
        Stop-Process -Id $backend.ProcessId -Force -ErrorAction Stop
    }
    $viteLaunchers = @()
    foreach ($vite in $viteServers) {
        $ancestorPid = $vite.ParentProcessId
        while ($ancestorPid) {
            $ancestor = $processes | Where-Object { $_.ProcessId -eq $ancestorPid } | Select-Object -First 1
            if (!$ancestor -or $ancestor.Name -notmatch '^(?:cmd|node)(?:\.exe)?$' -or
                $ancestor.CommandLine -notmatch '--port\s+5175\b' -or
                ($ancestor.CommandLine -notmatch '\bvite\b' -and
                 $ancestor.CommandLine -notmatch 'npm(?:-cli\.js|\.cmd).{0,100}\brun dev\b')) {
                break
            }
            $viteLaunchers += $ancestor
            $ancestorPid = $ancestor.ParentProcessId
        }
    }
    # Stop only the verified port-5175 launch chain, from its outer launcher
    # down to Vite. In particular, a separate default-port 5173 Vite stays up.
    for ($i = $viteLaunchers.Count - 1; $i -ge 0; $i--) {
        Stop-Process -Id $viteLaunchers[$i].ProcessId -Force -ErrorAction SilentlyContinue
    }
    foreach ($vite in $viteServers) {
        Stop-Process -Id $vite.ProcessId -Force -ErrorAction Stop
    }
    for ($i = 0; $i -lt 15; $i++) {
        if (!(Test-LocalPort 8002) -and !(Test-LocalPort 5175)) { break }
        Start-Sleep -Seconds 1
    }
    if ((Test-LocalPort 8002) -or (Test-LocalPort 5175)) {
        throw 'A Servy demo port remained active after stopping its verified local process tree.'
    }
}

if (!(Test-LocalPort 3306)) {
    $existingMysql = @(Get-CimInstance Win32_Process -Filter "Name='mysqld.exe'" -ErrorAction SilentlyContinue)
    if ($existingMysql.Count -gt 0) {
        throw 'MySQL port 3306 is not reachable while an existing mysqld.exe process is running. Refusing to start another MySQL instance.'
    }
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
# Technical claims stay extractive; a local open-weight model may only
# choose from already approved evidence IDs and vetted follow-up prompts.
$env:SERVY_LLM_PROVIDER = 'extractive'
$env:SERVY_REAL_DEMO_DIALOGUE_MODE = if ($NoLocalChat) { 'rules' } else { 'local' }
if (-not $env:SERVY_REAL_DEMO_CHAT_MODEL) {
    $env:SERVY_REAL_DEMO_CHAT_MODEL = 'qwen2.5:3b'
}
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
