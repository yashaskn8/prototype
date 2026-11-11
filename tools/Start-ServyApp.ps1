<# Start the normal authenticated Servy app without touching MySQL state. #>
[CmdletBinding()]
param([ValidateRange(10, 180)][int]$ReadinessTimeoutSeconds = 60)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$frontendRoot = Join-Path $repoRoot 'frontend'
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
$npm = (Get-Command npm.cmd -ErrorAction Stop).Source
$node = (Get-Command node.exe -ErrorAction Stop).Source
$powershell = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
$stateRoot = Join-Path ([System.IO.Path]::GetTempPath()) 'ServyApp'
$statePath = Join-Path $stateRoot 'current.json'
$viteEntry = Join-Path $frontendRoot 'node_modules\vite\bin\vite.js'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Project virtualenv Python is missing: $python" }
if (-not (Test-Path -LiteralPath $npm -PathType Leaf) -or -not (Test-Path -LiteralPath $node -PathType Leaf)) { throw 'Node.js and npm are required.' }
if (-not (Test-Path -LiteralPath $viteEntry -PathType Leaf)) { throw 'Installed Vite dependencies are missing; no package reinstall was attempted.' }
$package = Get-Content -LiteralPath (Join-Path $frontendRoot 'package.json') -Raw | ConvertFrom-Json
if ($package.scripts.dev -ne 'vite') { throw "Expected frontend dev script 'vite'; found '$($package.scripts.dev)'." }

function Get-CommandHash {
    param([string]$CommandLine)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($CommandLine))).Replace('-', '') }
    finally { $sha.Dispose() }
}

function ConvertTo-ProcessSnapshot {
    param($Process, [string]$Role, [int]$Depth)
    if ($null -eq $Process -or [string]::IsNullOrWhiteSpace($Process.CommandLine)) { return $null }
    [pscustomobject]@{
        Role = $Role
        PID = [int]$Process.ProcessId
        ParentPID = [int]$Process.ParentProcessId
        Depth = $Depth
        Image = [string]$Process.ExecutablePath
        CommandLine = [string]$Process.CommandLine
        CreatedUtcTicks = ([datetime]$Process.CreationDate).ToUniversalTime().Ticks
        CommandHash = Get-CommandHash -CommandLine ([string]$Process.CommandLine)
    }
}

function Get-ProcessSnapshot {
    param([int]$ProcessId, [string]$Role)
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId" -ErrorAction SilentlyContinue
    ConvertTo-ProcessSnapshot -Process $process -Role $Role -Depth 0
}

function Get-ProcessTreeSnapshots {
    param([int]$RootPID, [string]$Role)
    $all = @(Get-CimInstance Win32_Process -ErrorAction Stop)
    $root = $all | Where-Object { [int]$_.ProcessId -eq $RootPID } | Select-Object -First 1
    if ($null -eq $root) { throw "Started process $RootPID disappeared before it could be recorded." }
    $known = @{}
    $rootSnapshot = ConvertTo-ProcessSnapshot -Process $root -Role $Role -Depth 0
    if ($null -eq $rootSnapshot) { throw "Could not record process $RootPID." }
    $known[$RootPID] = $rootSnapshot
    $changed = $true
    while ($changed) {
        $changed = $false
        foreach ($item in $all) {
            $pidValue = [int]$item.ProcessId
            $parentValue = [int]$item.ParentProcessId
            if (-not $known.ContainsKey($pidValue) -and $known.ContainsKey($parentValue)) {
                $snapshot = ConvertTo-ProcessSnapshot -Process $item -Role $Role -Depth ($known[$parentValue].Depth + 1)
                if ($null -ne $snapshot) { $known[$pidValue] = $snapshot; $changed = $true }
            }
        }
    }
    return @($known.Values | Sort-Object Depth, PID)
}

function Test-ProcessSnapshot {
    param($Snapshot)
    if ($null -eq $Snapshot) { return $false }
    $actual = Get-ProcessSnapshot -ProcessId ([int]$Snapshot.PID) -Role ([string]$Snapshot.Role)
    if ($null -eq $actual) { return $false }
    return ($actual.Image -ieq [string]$Snapshot.Image -and
        $actual.CreatedUtcTicks -eq [long]$Snapshot.CreatedUtcTicks -and
        $actual.CommandHash -eq [string]$Snapshot.CommandHash)
}

function Assert-PortFree {
    param([int]$Port)
    $probe = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Parse('127.0.0.1'), $Port)
    try { $probe.Start() }
    catch {
        $ownerText = 'owner could not be identified'
        try {
            $owners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop | Select-Object -ExpandProperty OwningProcess -Unique)
            if ($owners.Count -gt 0) { $ownerText = "PID(s) $($owners -join ', ')" }
        } catch { }
        throw "Port $Port is unavailable ($ownerText). No process was stopped."
    } finally { if ($probe.Server.IsBound) { $probe.Stop() } }
}

function Wait-ForHttp200 {
    param([string]$Url, [int]$ProcessId, [datetime]$Deadline)
    while ([datetime]::UtcNow -lt $Deadline) {
        if ($null -eq (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) { throw "Process $ProcessId exited before $Url became ready." }
        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
            if ([int]$response.StatusCode -eq 200) { return $response }
        } catch { }
        Start-Sleep -Milliseconds 500
    }
    throw "Timed out waiting for HTTP 200 from $Url. See logs under $stateRoot."
}

function Stop-ExactProcess {
    param($Snapshot)
    if (Test-ProcessSnapshot $Snapshot) {
        Stop-Process -Id ([int]$Snapshot.PID) -ErrorAction Stop
        try { Wait-Process -Id ([int]$Snapshot.PID) -Timeout 10 -ErrorAction Stop } catch { }
    }
}

# Reuse a complete healthy launch. Never duplicate a partial or untracked one.
if (Test-Path -LiteralPath $statePath -PathType Leaf) {
    $oldState = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    $oldProcesses = @($oldState.Processes)
    $stillOwned = @($oldProcesses | Where-Object { Test-ProcessSnapshot $_ })
    if ($stillOwned.Count -gt 0) {
        if ($stillOwned.Count -ne $oldProcesses.Count -or
            -not ($stillOwned | Where-Object { $_.Role -eq 'backend' -and $_.Depth -eq 0 }) -or
            -not ($stillOwned | Where-Object { $_.Role -eq 'frontend' -and $_.Depth -eq 0 })) {
            throw "Tracked Servy processes are incomplete. Inspect $statePath; no process was stopped or duplicated."
        }
        $deadline = [datetime]::UtcNow.AddSeconds(5)
        Wait-ForHttp200 -Url 'http://127.0.0.1:8000/api/auth/csrf/' -ProcessId ([int]($stillOwned | Where-Object { $_.Role -eq 'backend' -and $_.Depth -eq 0 }).PID) -Deadline $deadline | Out-Null
        Wait-ForHttp200 -Url 'http://127.0.0.1:5173/' -ProcessId ([int]($stillOwned | Where-Object { $_.Role -eq 'frontend' -and $_.Depth -eq 0 }).PID) -Deadline $deadline | Out-Null
        Write-Host 'Servy normal app is already running and healthy.'
        Write-Host 'Customer:   http://127.0.0.1:5173/'
        Write-Host 'Technician: http://127.0.0.1:5173/technician/inbox'
        return
    }
    Remove-Item -LiteralPath $statePath -Force
}

# Bind probes prevent stopping or replacing unidentified listeners.
foreach ($port in 8000, 5173) { Assert-PortFree -Port $port }
New-Item -ItemType Directory -Path $stateRoot -Force | Out-Null
$runId = Get-Date -Format 'yyyyMMdd-HHmmss'
$runLog = Join-Path $stateRoot $runId
New-Item -ItemType Directory -Path $runLog -Force | Out-Null

$envNames = @('SERVY_DB_ENGINE', 'DJANGO_SETTINGS_MODULE', 'VITE_SERVY_REAL_DEMO', 'SERVY_UI_TEST_API_TARGET')
$oldEnvironment = @{}
foreach ($name in $envNames) { $oldEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
$backendProcess = $null
$frontendProcess = $null

try {
    [Environment]::SetEnvironmentVariable('SERVY_DB_ENGINE', 'mysql', 'Process')
    [Environment]::SetEnvironmentVariable('DJANGO_SETTINGS_MODULE', 'servy_rag.settings', 'Process')

    Write-Host 'Checking configured MySQL readiness (read-only)...'
    $mysqlCheck = & $python (Join-Path $repoRoot 'manage.py') check_mysql_ready 2>&1
    $mysqlExit = $LASTEXITCODE
    $mysqlCheck | ForEach-Object { Write-Host $_ }
    if ($mysqlExit -ne 0) { throw 'Configured MySQL readiness check failed. No app server was started.' }

    Write-Host 'Checking Django configuration...'
    & $python (Join-Path $repoRoot 'manage.py') check
    if ($LASTEXITCODE -ne 0) { throw 'Django system check failed. No app server was started.' }

    Write-Host 'Checking migration state without applying migrations...'
    $migrationPlan = & $python (Join-Path $repoRoot 'manage.py') showmigrations --plan 2>&1
    if ($LASTEXITCODE -ne 0) { $migrationPlan | ForEach-Object { Write-Host $_ }; throw 'Could not read migration state.' }
    if (($migrationPlan | Out-String) -match '(?m)^\s*\[\s*\]') { throw 'The schema has unapplied migrations. None were run and no server was started.' }

    $routeProbe = @'
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
import django
django.setup()
from django.urls import resolve
paths = [
    "/api/auth/csrf/", "/api/auth/login/", "/api/auth/me/",
    "/api/customer-support/conversations/",
    "/api/customer-support/conversations/00000000-0000-0000-0000-000000000001/messages/",
    "/api/customer-support/conversations/00000000-0000-0000-0000-000000000001/ticket/",
    "/api/technician/inbox/", "/api/technician/inbox/1/",
    "/api/technician/inbox/1/acknowledge/",
]
for path in paths:
    match = resolve(path)
    print(f"{path} -> {match.url_name}")
'@
    $routeOutput = & $python -c $routeProbe 2>&1
    if ($LASTEXITCODE -ne 0) { $routeOutput | ForEach-Object { Write-Host $_ }; throw 'One or more required API routes failed to resolve.' }
    $routeOutput | ForEach-Object { Write-Host $_ }

    Write-Host 'Starting Django on 127.0.0.1:8000...'
    $backendProcess = Start-Process -FilePath $python -ArgumentList @((Join-Path $repoRoot 'manage.py'), 'runserver', '127.0.0.1:8000', '--noreload') -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runLog 'backend.stdout.log') -RedirectStandardError (Join-Path $runLog 'backend.stderr.log')

    # Set both before npm/Vite starts so server configuration and import.meta.env agree.
    [Environment]::SetEnvironmentVariable('VITE_SERVY_REAL_DEMO', '0', 'Process')
    [Environment]::SetEnvironmentVariable('SERVY_UI_TEST_API_TARGET', 'http://127.0.0.1:8000', 'Process')
    $escapedNpm = $npm.Replace("'", "''")
    $npmCommand = "& '$escapedNpm' run dev -- --host 127.0.0.1 --port 5173 --strictPort"
    $encodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($npmCommand))
    Write-Host 'Starting npm run dev in normal-app mode on 127.0.0.1:5173...'
    $frontendProcess = Start-Process -FilePath $powershell -ArgumentList @('-NoLogo', '-NoProfile', '-NonInteractive', '-EncodedCommand', $encodedCommand) -WorkingDirectory $frontendRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runLog 'frontend.stdout.log') -RedirectStandardError (Join-Path $runLog 'frontend.stderr.log')

    $deadline = [datetime]::UtcNow.AddSeconds($ReadinessTimeoutSeconds)
    Wait-ForHttp200 -Url 'http://127.0.0.1:8000/api/auth/csrf/' -ProcessId $backendProcess.Id -Deadline $deadline | Out-Null
    Wait-ForHttp200 -Url 'http://127.0.0.1:5173/' -ProcessId $frontendProcess.Id -Deadline $deadline | Out-Null
    $proxyResponse = Invoke-WebRequest -Uri 'http://127.0.0.1:5173/api/auth/csrf/' -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
    if ([int]$proxyResponse.StatusCode -ne 200) { throw 'Vite /api proxy did not return the Django CSRF endpoint.' }
    Wait-ForHttp200 -Url 'http://127.0.0.1:5173/technician/inbox' -ProcessId $frontendProcess.Id -Deadline $deadline | Out-Null

    $backendSnapshots = @(Get-ProcessTreeSnapshots -RootPID $backendProcess.Id -Role 'backend')
    $frontendSnapshots = @(Get-ProcessTreeSnapshots -RootPID $frontendProcess.Id -Role 'frontend')
    $viteServerProcesses = @($frontendSnapshots | Where-Object {
        $_.Image -match '(?i)\\node\.exe$' -and
        $_.CommandLine -match '(?i)vite[\\/]bin[\\/]vite\.js' -and
        $_.CommandLine -match '--port\s+5173'
    })
    if ($viteServerProcesses.Count -ne 1) {
        $frontendSnapshots | Select-Object PID,ParentPID,Depth,Image,CommandLine | Format-List | Out-String | Write-Host
        throw "Expected one Vite process on port 5173; found $($viteServerProcesses.Count)."
    }
    $state = [pscustomobject]@{ RepositoryRoot = $repoRoot; RunLogDirectory = $runLog; Processes = @($backendSnapshots + $frontendSnapshots) }
    $temporaryStatePath = Join-Path $stateRoot 'current.json.tmp'
    $state | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $temporaryStatePath -Encoding UTF8
    Move-Item -LiteralPath $temporaryStatePath -Destination $statePath -Force

    Write-Host ''
    Write-Host 'Servy normal app is ready.' -ForegroundColor Green
    Write-Host 'Backend:    http://127.0.0.1:8000'
    Write-Host "Django PID: $($backendProcess.Id)"
    Write-Host 'Customer:   http://127.0.0.1:5173/'
    Write-Host 'Technician: http://127.0.0.1:5173/technician/inbox'
    Write-Host "Vite PID:   $($viteServerProcesses[0].PID) (launcher PID $($frontendProcess.Id))"
    Write-Host "Logs:       $runLog"
} catch {
    Write-Error "Servy launcher failed: $($_.Exception.Message)"
    # Clean up only this invocation's exact process tree.
    foreach ($target in @(@{ Process = $frontendProcess; Role = 'frontend' }, @{ Process = $backendProcess; Role = 'backend' })) {
        if ($null -ne $target.Process) {
            try {
                $tree = @(Get-ProcessTreeSnapshots -RootPID $target.Process.Id -Role $target.Role -ErrorAction Stop | Sort-Object Depth -Descending)
                foreach ($snapshot in $tree) { Stop-ExactProcess -Snapshot $snapshot }
            } catch {
                $snapshot = Get-ProcessSnapshot -ProcessId $target.Process.Id -Role $target.Role
                if ($null -ne $snapshot) { Stop-ExactProcess -Snapshot $snapshot }
            }
        }
    }
    throw
} finally {
    foreach ($name in $envNames) { [Environment]::SetEnvironmentVariable($name, $oldEnvironment[$name], 'Process') }
}
