<# Start the separate code-gated technician view. The official technician inbox
   APIs are served by the access-code gateway on 127.0.0.1:8003, backed only by the
   isolated servy_dev MySQL database (servy_dev_app). Never production servy. #>
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

$devEnvFile = Join-Path $projectRoot '.env.mysql.servy_dev'
$devMedia = 'C:\Users\prana\AppData\Local\Servy\servy_dev_media'
# The isolated development index; the gateway must never default to production Chroma.
$devChroma = 'C:\Users\prana\AppData\Local\Servy\servy_dev_chroma'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Project virtualenv Python is missing: $python" }
if (-not (Test-Path -LiteralPath $viteEntry -PathType Leaf)) { throw 'Installed Vite dependencies are missing; no package reinstall was attempted.' }
if (-not (Test-Path -LiteralPath $devEnvFile -PathType Leaf)) { throw 'The servy_dev MySQL settings file is missing.' }
if (-not (Test-Path -LiteralPath $accessFile -PathType Leaf)) { throw 'The existing technician access-code file is missing.' }

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

if (-not (Test-LocalPort 3306)) { throw 'Local MySQL is not reachable on port 3306. Start it (or run Start-ServyRealDemo.ps1) first.' }
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

$names = @('SERVY_DB_ENGINE', 'SERVY_MYSQL_ENV_FILE', 'SERVY_MEDIA_ROOT', 'SERVY_CHROMA_PERSIST_DIR', 'SERVY_DEMO_GATEWAY',
    'SERVY_TECHNICIAN_ACCESS_FILE', 'DJANGO_DEBUG', 'VITE_SERVY_REAL_DEMO', 'VITE_SERVY_TECH_DEMO')
$old = @{}
foreach ($name in $names) { $old[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
$backend = $null
$frontend = $null

try {
    # No migrations, seeding or company-record writes; servy_dev only.
    [Environment]::SetEnvironmentVariable('SERVY_DB_ENGINE', 'mysql', 'Process')
    [Environment]::SetEnvironmentVariable('SERVY_MYSQL_ENV_FILE', $devEnvFile, 'Process')
    [Environment]::SetEnvironmentVariable('SERVY_MEDIA_ROOT', $devMedia, 'Process')
    [Environment]::SetEnvironmentVariable('SERVY_CHROMA_PERSIST_DIR', $devChroma, 'Process')
    [Environment]::SetEnvironmentVariable('SERVY_DEMO_GATEWAY', 'technician', 'Process')
    [Environment]::SetEnvironmentVariable('SERVY_TECHNICIAN_ACCESS_FILE', $accessFile, 'Process')
    [Environment]::SetEnvironmentVariable('DJANGO_DEBUG', '1', 'Process')

    Push-Location -LiteralPath $projectRoot
    try {
        & $python '-B' 'manage.py' 'check_dev_database' *> $null
        if ($LASTEXITCODE -ne 0) { throw 'The servy_dev isolation check failed; the Technician gateway was not started. Run manage.py check_dev_database for details.' }
    } finally {
        Pop-Location
    }
    Write-Host 'servy_dev isolation check: PASS'

    Write-Host 'Starting the technician gateway on 127.0.0.1:8003...'
    $backend = Start-Process -FilePath $python -ArgumentList @('-B', '-u', 'manage.py', 'runserver', '127.0.0.1:8003', '--noreload') -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $backendOut -RedirectStandardError $backendErr

    $deadline = [DateTime]::UtcNow.AddSeconds($ReadinessTimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline -and -not (Test-LocalPort 8003)) {
        if ($null -eq (Get-Process -Id $backend.Id -ErrorAction SilentlyContinue)) {
            $detail = if (Test-Path $backendErr) { Get-Content -LiteralPath $backendErr -Raw } else { 'No backend error log was produced.' }
            throw "Technician ticket service exited during startup. $detail"
        }
        Start-Sleep -Milliseconds 500
    }
    if (-not (Test-LocalPort 8003)) { throw "Technician ticket service did not open port 8003. Inspect $backendErr." }


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

    # Exercise the browser-facing Vite proxy with the private technician code.
    # A listening port alone does not prove that the ticket API is reachable.
    $technicianCode = (Get-Content -LiteralPath $accessFile -Raw).Trim()
    try {
        $queueProbe = Invoke-WebRequest -Uri 'http://127.0.0.1:5176/tech-api/technician/inbox/' -Headers @{ Authorization = "Technician $technicianCode" } -Method Get -UseBasicParsing -TimeoutSec 15
        $queuePayload = $queueProbe.Content | ConvertFrom-Json
        if ($queueProbe.StatusCode -ne 200 -or $null -eq $queuePayload -or $null -eq $queuePayload.PSObject.Properties['tickets']) {
            throw 'The technician ticket queue response was empty or invalid.'
        }
        Write-Host ('Technician API check: PASS (' + @($queuePayload.tickets).Count + ' authorized tickets visible)') -ForegroundColor Green
    } catch {
        throw 'Technician API readiness failed through the browser-facing Vite proxy. Inspect the current backend and frontend error logs before the demo.'
    } finally {
        $technicianCode = $null
        $queuePayload = $null
        $queueProbe = $null
    }

    Write-Host ''
    Write-Host 'Authorized technician instance is ready.' -ForegroundColor Green
    Write-Host 'Technician UI:  http://127.0.0.1:5176/'
    Write-Host 'Access code:    open the separate local file below'
    Write-Host "Access file:    $accessFile"
    Write-Host 'Ticket storage: servy_dev (isolated local development MySQL database)'
    Write-Host "Logs:           $logRoot"
    Write-Host 'No migration, seeding or company record write was performed.'
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
    Write-Host ''
    Write-Host 'Live technician events: queue polls, ticket reads/updates, downloads and authorization failures.'
    Write-Host 'Press Ctrl+C to close this log viewer only; the technician backend and UI remain running.'
    Get-Content -LiteralPath $backendErr -Tail 20 -Wait
}
