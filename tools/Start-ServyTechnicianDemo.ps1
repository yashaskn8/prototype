<# Start the separate code-gated technician view for authorized RAG tickets.
   Uses only the dedicated local SQLite ticket workspace; never connects to MySQL. #>
[CmdletBinding()]
param([ValidateRange(10, 120)][int]$ReadinessTimeoutSeconds = 45, [switch]$NoWatch)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$frontendRoot = Join-Path $projectRoot 'frontend'
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$npm = (Get-Command npm.cmd -ErrorAction Stop).Source
$viteEntry = Join-Path $frontendRoot 'node_modules\vite\bin\vite.js'
$privateLocal = 'C:\Users\prana\AppData\Local\Packages\OpenAI.Codex_2p2nqsd0c76g0\LocalCache\Local'
$instanceRoot = Join-Path $privateLocal 'Servy\private_evaluations\servy_technician_demo_20261111_01'
$accessFile = Join-Path $instanceRoot 'technician_access.txt'
$logRoot = Join-Path $instanceRoot 'logs'

# Match the exact local ticket database path used by Start-ServyRealDemo.ps1.
# Resolve the actual Windows known folder, not a LOCALAPPDATA value that
# Start-ServyRealDemo.ps1 may have temporarily overridden in this shell.
$sharedLocalAppData = [System.Environment]::GetFolderPath([System.Environment+SpecialFolder]::LocalApplicationData)
if ([string]::IsNullOrWhiteSpace($sharedLocalAppData)) {
    $sharedLocalAppData = Join-Path $env:USERPROFILE 'AppData\Local'
}
$ticketDb = Join-Path $sharedLocalAppData 'Servy\local-ticket-workspace.sqlite3'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Project virtualenv Python is missing: $python" }
if (-not (Test-Path -LiteralPath $viteEntry -PathType Leaf)) { throw 'Installed Vite dependencies are missing; no package reinstall was attempted.' }
$workspaceExists = Test-Path -LiteralPath $ticketDb -PathType Leaf
if (-not $workspaceExists) {
    Write-Warning "No customer RAG ticket workspace exists yet. The technician portal will start with an empty queue without creating a database. After the first customer ticket is confirmed, use Refresh queue."
}

function Test-LocalPort([int]$Port) {
    $client = [System.Net.Sockets.TcpClient]::new()
    $attempt = $null
    try {
        $attempt = $client.BeginConnect('127.0.0.1', $Port, $null, $null)
        if (-not $attempt.AsyncWaitHandle.WaitOne(800)) { return $false }
        $client.EndConnect($attempt)
        return $true
    } catch {
        return $false
    } finally {
        if ($null -ne $attempt) { $attempt.AsyncWaitHandle.Close() }
        $client.Dispose()
    }

}

foreach ($port in 8003, 5176) {
    if (Test-LocalPort $port) {
        throw "Technician demo port $port is already in use. No process was stopped or duplicated. Inspect the existing listener before retrying."
    }
}

New-Item -ItemType Directory -Path $instanceRoot -Force | Out-Null
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$runId = Get-Date -Format 'yyyyMMdd-HHmmss'
$backendOut = Join-Path $logRoot "$runId.backend.stdout.log"
$backendErr = Join-Path $logRoot "$runId.backend.stderr.log"
$frontendOut = Join-Path $logRoot "$runId.frontend.stdout.log"
$frontendErr = Join-Path $logRoot "$runId.frontend.stderr.log"

$names = @('LOCALAPPDATA', 'SERVY_LOCAL_TICKET_DB_PATH', 'VITE_SERVY_REAL_DEMO', 'VITE_SERVY_TECH_DEMO')
$old = @{}
foreach ($name in $names) { $old[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
$backend = $null
$frontend = $null

try {
    # The technician service reads only this existing file. It does not bootstrap
    # or initialize SQLite, Django, MySQL, migrations, or company records.
    [Environment]::SetEnvironmentVariable('LOCALAPPDATA', $privateLocal, 'Process')
    [Environment]::SetEnvironmentVariable('SERVY_LOCAL_TICKET_DB_PATH', $ticketDb, 'Process')

    Write-Host 'Starting the private technician ticket service on 127.0.0.1:8003...'
    $backend = Start-Process -FilePath $python -ArgumentList @('-B', '-u', 'tools\servy_technician_demo.py') -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $backendOut -RedirectStandardError $backendErr

    $deadline = [DateTime]::UtcNow.AddSeconds($ReadinessTimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline -and -not (Test-LocalPort 8003)) {
        if ($null -eq (Get-Process -Id $backend.Id -ErrorAction SilentlyContinue)) {
            $detail = if (Test-Path $backendErr) { Get-Content -LiteralPath $backendErr -Raw } else { 'No backend error log was produced.' }
            throw "Technician ticket service exited during startup. $detail"
        }
        Start-Sleep -Milliseconds 500
    }
    if (-not (Test-LocalPort 8003)) { throw "Technician ticket service did not open port 8003. Inspect $backendErr." }

    $accessDeadline = [DateTime]::UtcNow.AddSeconds(5)
    while ([DateTime]::UtcNow -lt $accessDeadline -and -not (Test-Path -LiteralPath $accessFile -PathType Leaf)) {
        Start-Sleep -Milliseconds 200
    }
    if (-not (Test-Path -LiteralPath $accessFile -PathType Leaf)) {
        throw "Technician service started but its private access-code file was not created. Inspect $backendErr."
    }

    [Environment]::SetEnvironmentVariable('VITE_SERVY_REAL_DEMO', '0', 'Process')
    [Environment]::SetEnvironmentVariable('VITE_SERVY_TECH_DEMO', '1', 'Process')
    Write-Host 'Starting separate technician UI on 127.0.0.1:5176...'
    $frontend = Start-Process -FilePath $npm -ArgumentList @('run', 'dev', '--', '--host', '127.0.0.1', '--port', '5176', '--strictPort') -WorkingDirectory $frontendRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $frontendOut -RedirectStandardError $frontendErr

    $deadline = [DateTime]::UtcNow.AddSeconds($ReadinessTimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline -and -not (Test-LocalPort 5176)) {
        if ($null -eq (Get-Process -Id $frontend.Id -ErrorAction SilentlyContinue)) {
            $detail = if (Test-Path $frontendErr) { Get-Content -LiteralPath $frontendErr -Raw } else { 'No frontend error log was produced.' }
            throw "Technician Vite process exited during startup. $detail"
        }
        Start-Sleep -Milliseconds 500
    }
    if (-not (Test-LocalPort 5176)) { throw "Technician UI did not open port 5176. Inspect $frontendErr." }

    Write-Host ''
    Write-Host 'Authorized technician instance is ready.' -ForegroundColor Green
    Write-Host 'Technician UI:  http://127.0.0.1:5176/'
    Write-Host 'Access code:    open the separate local file below'
    Write-Host "Access file:    $accessFile"
    Write-Host "Ticket storage: $ticketDb (separate local SQLite workspace)"
    if (-not $workspaceExists) {
        Write-Host 'Queue state:     Waiting for the first customer RAG ticket; refresh the portal after submission.'
    } else {
        Write-Host 'Queue state:     Existing local RAG ticket workspace detected.'
    }
    Write-Host "Logs:           $logRoot"
    Write-Host 'No MySQL connection, migration, company record write, or service-call creation was performed.'
} catch {
    # Do not force-stop processes. If startup partially succeeded, report their
    # PIDs and logs so the owner can inspect and stop the verified instance.
    Write-Error "Technician demo launcher failed: $($_.Exception.Message)"
    if ($null -ne $backend) { Write-Warning "Backend launcher PID: $($backend.Id); log: $backendErr" }
    if ($null -ne $frontend) { Write-Warning "Frontend launcher PID: $($frontend.Id); log: $frontendErr" }
    throw
} finally {
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $old[$name], 'Process') }
}

if (-not $NoWatch) {
    $activityLogs = @($backendOut, $backendErr, $frontendOut, $frontendErr) |
        Where-Object { Test-Path -LiteralPath $_ -PathType Leaf }
    Write-Host ''
    Write-Host 'Live technician logs: HTTP request statuses and Vite development-server activity.'
    Write-Host 'Press Ctrl+C to close this log viewer only; the technician backend and UI remain running.'
    Get-Content -Path $activityLogs -Tail 15 -Wait
}
