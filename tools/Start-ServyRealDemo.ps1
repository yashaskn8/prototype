# Start the localhost-only Customer RAG demo: the official Django APIs served by
# the access-code gateway on 127.0.0.1:8002, backed only by the isolated servy_dev
# MySQL database (servy_dev_app). No migrations, seeding or company-data changes.
param([switch]$Watch, [switch]$NoWatch, [switch]$Restart, [switch]$ReviewRequest, [switch]$RebuildFromVerifiedDocs, [switch]$NoLocalChat)
$environmentNames = @(
    'SERVY_DB_ENGINE', 'SERVY_MYSQL_ENV_FILE', 'SERVY_MEDIA_ROOT', 'SERVY_CHROMA_PERSIST_DIR',
    'SERVY_SERVING_EMBEDDING_PROFILE', 'SERVY_DEMO_GATEWAY', 'SERVY_DEMO_ACCESS_FILE',
    'SERVY_CUSTOMER_DATA_MODE', 'DJANGO_DEBUG',
    'SERVY_LLM_PROVIDER', 'SERVY_LLM_REQUIRE_OLLAMA', 'SERVY_OLLAMA_MODEL',
    'SERVY_CUSTOMER_ANSWER_GENERATION', 'SERVY_CUSTOMER_GENERATION_URL',
    'SERVY_CUSTOMER_GENERATION_MODEL', 'SERVY_CUSTOMER_GENERATION_TIMEOUT',
    'SERVY_REAL_DEMO_DIALOGUE_MODE',
    'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'SERVY_ALLOW_REMOTE_LLM',
    'SERVY_REAL_DEMO_REBUILD_VERIFIED_SOURCES', 'VITE_SERVY_REAL_DEMO'
)
$originalEnvironment = @{}
foreach ($name in $environmentNames) {
    $originalEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}

try {
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$privateLocal = 'C:\Users\prana\AppData\Local\Packages\OpenAI.Codex_2p2nqsd0c76g0\LocalCache\Local'
$demoFolder = Join-Path $privateLocal 'Servy\private_evaluations\servy_real_demo_20261031_01'
$backendOutLog = Join-Path $demoFolder 'startup.out'
$backendErrLog = Join-Path $demoFolder 'startup.err'
$viteOutLog = Join-Path $demoFolder 'vite.out'
$viteErrLog = Join-Path $demoFolder 'vite.err'
$activityLogPaths = @($backendOutLog, $backendErrLog, $viteOutLog, $viteErrLog)
$accessFile = Join-Path $demoFolder 'real_demo_access.txt'
$devEnvFile = Join-Path $projectRoot '.env.mysql.servy_dev'
$devMedia = 'C:\Users\prana\AppData\Local\Servy\servy_dev_media'
$devChroma = 'C:\Users\prana\AppData\Local\Servy\servy_dev_chroma'

if (!(Test-Path -LiteralPath $accessFile) -or !(Test-Path -LiteralPath $devEnvFile) -or !(Test-Path -LiteralPath $devChroma)) {
    throw 'The existing demo access file, the servy_dev MySQL settings or the isolated development BGE index is unavailable.'
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
        $_.CommandLine -match 'manage\.py\S*\s+runserver\s+127\.0\.0\.1:8002\b'
    })
    $backendChildren = @($processes | Where-Object {
        $_.Name -match '^python(?:\.exe)?$' -and
        $_.CommandLine -match 'manage\.py\S*\s+runserver\s+127\.0\.0\.1:8002\b' -and
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

$env:SERVY_DB_ENGINE = 'mysql'
$env:SERVY_MYSQL_ENV_FILE = $devEnvFile
$env:SERVY_MEDIA_ROOT = $devMedia
$env:SERVY_CHROMA_PERSIST_DIR = $devChroma
$env:SERVY_SERVING_EMBEDDING_PROFILE = 'bge-small'
$env:SERVY_DEMO_GATEWAY = 'customer'
$env:SERVY_DEMO_ACCESS_FILE = $accessFile
$env:SERVY_CUSTOMER_DATA_MODE = 'demo_only'
$env:DJANGO_DEBUG = '1'
# Retrieval, evidence selection and safety remain deterministic. Qwen runs
# only after claims and source citations are selected; it receives reauthorized
# full source chunks and cannot add facts absent from those passages. Invalid
# or unavailable output falls back to the exact cited source wording.
$env:SERVY_ALLOW_REMOTE_LLM = '0'
# Retrieval, ranking, evidence selection and safety remain deterministic.
# Qwen writes customer-facing wording only from reauthorized cited source chunks.
$env:SERVY_LLM_PROVIDER = 'extractive'
if ($NoLocalChat) {
    $env:SERVY_LLM_PROVIDER = 'extractive'
    $env:SERVY_CUSTOMER_ANSWER_GENERATION = 'disabled'
    Write-Output 'Qwen generation disabled; using deterministic source wording.'
} else {
    $env:SERVY_CUSTOMER_ANSWER_GENERATION = 'ollama'
    $env:SERVY_CUSTOMER_GENERATION_URL = 'http://127.0.0.1:11434/api/chat'
    $env:SERVY_CUSTOMER_GENERATION_MODEL = 'qwen3.5:2b-q4_K_M'
    $env:SERVY_CUSTOMER_GENERATION_TIMEOUT = '45'
    Write-Output 'Customer RAG answer generation: local Qwen enabled; cited-source validation and extractive fallback retained.'
    try {
        $qwenTags = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 3
    } catch {
        throw 'Qwen is enabled but local Ollama is not reachable. Start Ollama, then rerun this launcher; use -NoLocalChat only when intentionally selecting extractive-only mode.'
    }
    $qwenReady = @($qwenTags.models | Where-Object {
        $_.name -eq 'qwen3.5:2b-q4_K_M'
    }).Count -gt 0
    if (!$qwenReady) {
        throw 'Qwen is enabled but qwen3.5:2b-q4_K_M is not installed. Run: ollama pull qwen3.5:2b-q4_K_M'
    }
    Write-Output 'Qwen model check: PASS (qwen3.5:2b-q4_K_M present locally).'
}
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:SERVY_ALLOW_REMOTE_LLM = '0'
if ($ReviewRequest -or $RebuildFromVerifiedDocs) {
    throw 'The legacy read-only index review and rebuild options are not used by the servy_dev gateway.'
}

if (!(Test-LocalPort 8002)) {
    $pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
    # Refuse to start unless the live connection is the isolated servy_dev database
    # with the restricted servy_dev_app account.
    Push-Location -LiteralPath $projectRoot
    try {
        & $pythonPath '-B' 'manage.py' 'check_dev_database' *> $null
        if ($LASTEXITCODE -ne 0) { throw 'The servy_dev isolation check failed; the Customer gateway was not started. Run manage.py check_dev_database for details.' }
    } finally {
        Pop-Location
    }
    Write-Output 'servy_dev isolation check: PASS'

    # A previous invocation may have timed out while its model/index load was
    # still running. Reuse only this exact project's virtualenv Python process;
    # never launch a duplicate backend or stop an unidentified process.
    $matchingBackends = @(
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction Stop |
        Where-Object {
            [string]::Equals([string]$_.ExecutablePath, $pythonPath, [System.StringComparison]::OrdinalIgnoreCase) -and
            $_.CommandLine -match 'manage\.py\S*\s+runserver\s+127\.0\.0\.1:8002\b'
        }
    )
    if ($matchingBackends.Count -gt 1) {
        throw "Found $($matchingBackends.Count) matching Servy real-demo Python processes but port 8002 is closed. No process was stopped or duplicated; inspect Task Manager and the private logs."
    }
    if ($matchingBackends.Count -eq 1) {
        $backendProcessId = [int]$matchingBackends[0].ProcessId
        Write-Output "Reusing existing Servy real-demo backend process $backendProcessId while it finishes startup."
    } else {
        $backendProcess = Start-Process -FilePath $pythonPath `
            -ArgumentList '-B','-u','manage.py','runserver','127.0.0.1:8002','--noreload' -WorkingDirectory $projectRoot `
            -WindowStyle Hidden -PassThru -RedirectStandardOutput $backendOutLog `
            -RedirectStandardError $backendErrLog
        $backendProcessId = [int]$backendProcess.Id
    }

    # A private BGE index reconstruction can take several minutes on CPU.
    # Wait safely, and distinguish an exited child from a slow startup.
    $backendChecks = 240
    for ($i = 0; $i -lt $backendChecks -and !(Test-LocalPort 8002); $i++) {
        if ($null -eq (Get-Process -Id $backendProcessId -ErrorAction SilentlyContinue)) { break }
        Start-Sleep -Seconds 1
    }
    if (!(Test-LocalPort 8002)) {
        $processState = if ($null -eq (Get-Process -Id $backendProcessId -ErrorAction SilentlyContinue)) {
            "Python backend process $backendProcessId exited."
        } else {
            "Python backend process $backendProcessId is still running but did not open port 8002 within $backendChecks seconds."
        }
        throw "$processState Inspect the private logs: stdout=$backendOutLog ; stderr=$backendErrLog"
    }
}

$env:VITE_SERVY_REAL_DEMO = '1'
if (!(Test-LocalPort 5175)) {
    Start-Process -FilePath 'npm.cmd' -ArgumentList 'run','dev','--','--host','127.0.0.1','--port','5175','--strictPort' `
        -WorkingDirectory (Join-Path $projectRoot 'frontend') -WindowStyle Hidden `
        -RedirectStandardOutput $viteOutLog `
        -RedirectStandardError $viteErrLog | Out-Null
    for ($i = 0; $i -lt 15 -and !(Test-LocalPort 5175); $i++) { Start-Sleep -Seconds 1 }
    if (!(Test-LocalPort 5175)) { throw "Demo frontend failed; inspect private vite.err locally." }
}

Write-Output 'Real-data demo ready: http://127.0.0.1:5175/'
Write-Output ('Enter the separate code from: ' + $accessFile)

# Verify the browser-facing Vite proxy and the authenticated customer API, not
# just open TCP ports. This also emits a safe HTTP 200 entry in the backend log.
$demoToken = (Get-Content -LiteralPath $accessFile -Raw).Trim()
try {
    $healthProbe = Invoke-WebRequest -Uri 'http://127.0.0.1:5175/demo-api/health/' -Headers @{
        Authorization = "Demo $demoToken"
        'X-Servy-Request-Source' = 'launcher-readiness'
    } -Method Get -UseBasicParsing -TimeoutSec 30
    $healthPayload = $healthProbe.Content | ConvertFrom-Json
    if ($healthProbe.StatusCode -ne 200 -or $healthPayload.status -ne 'ok') {
        throw 'The authenticated demo health check failed.'
    }
    Write-Output 'Customer RAG API check: PASS (authenticated gateway health endpoint on servy_dev)'
} catch {
    throw 'Customer RAG API readiness failed through the browser-facing Vite proxy. Inspect startup.err and vite.err before the demo.'
} finally {
    $demoToken = $null
    $healthPayload = $null
    $healthProbe = $null
}

if (-not $NoWatch) {
    Write-Output ''
    Write-Output 'Live customer RAG events: gateway requests and status (Django request log).'
    Write-Output 'Private stdout and Vite logs remain in the demo log folder for troubleshooting.'
    Write-Output 'Press Ctrl+C to close this log viewer only; the backend and frontend processes remain running.'
    if (Test-Path -LiteralPath $backendErrLog -PathType Leaf) {
        Get-Content -LiteralPath $backendErrLog -Tail 25 -Wait
    } else {
        Write-Warning "Customer RAG backend log is not available at: $backendErrLog"
    }
}
} finally {
    foreach ($name in $environmentNames) {
        [Environment]::SetEnvironmentVariable($name, $originalEnvironment[$name], 'Process')
    }
}
