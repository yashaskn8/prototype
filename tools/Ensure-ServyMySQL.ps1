function Test-ServyMySqlPort {
    [CmdletBinding()]
    param([ValidateRange(1, 65535)][int]$Port = 3306)

    $client = [System.Net.Sockets.TcpClient]::new()
    $attempt = $null
    try {
        $attempt = $client.BeginConnect('127.0.0.1', $Port, $null, $null)
        if (-not $attempt.AsyncWaitHandle.WaitOne(1000)) { return $false }
        $client.EndConnect($attempt)
        return $true
    } catch {
        return $false
    } finally {
        if ($null -ne $attempt) { $attempt.AsyncWaitHandle.Close() }
        $client.Dispose()
    }
}

function Ensure-ServyMySQL {
    [CmdletBinding()]
    param(
        [ValidateRange(15, 240)][int]$TimeoutSeconds = 60,
        [ValidateRange(1, 65535)][int]$Port = 3306
    )

    $ErrorActionPreference = 'Stop'
    $mysqlExe = 'C:\mysql\mysql-8.0.41-winx64\bin\mysqld.exe'
    $mysqlConfig = 'C:\mysql\my.ini'

    if (Test-ServyMySqlPort -Port $Port) {
        Write-Host "A local listener already accepts TCP connections on 127.0.0.1:$Port. The Django read-only identity check will verify that it is the configured Servy database."
        return
    }

    if (-not (Test-Path -LiteralPath $mysqlExe -PathType Leaf)) {
        throw "MySQL is stopped and the expected existing server binary is missing: $mysqlExe. No replacement server was installed."
    }
    if (-not (Test-Path -LiteralPath $mysqlConfig -PathType Leaf)) {
        throw "MySQL is stopped and its existing configuration is missing: $mysqlConfig. No configuration was generated."
    }

    $mysqlExe = (Get-Item -LiteralPath $mysqlExe).FullName
    $mysqlConfig = (Get-Item -LiteralPath $mysqlConfig).FullName

    # Validate only the existing server configuration and data directory.
    # Never initialize, repair, migrate, seed, or recreate a MySQL data directory.
    $settings = @{}
    $insideServerSection = $false
    foreach ($line in [System.IO.File]::ReadAllLines($mysqlConfig)) {
        $trimmed = $line.Trim()
        if ($trimmed -match '^\[([^\]]+)\]\s*$') {
            $insideServerSection = ($Matches[1] -ieq 'mysqld')
            continue
        }
        if ($insideServerSection -and $trimmed -notmatch '^[#;]' -and
            $trimmed -match '^([A-Za-z0-9_-]+)\s*=\s*(.*?)\s*$') {
            $settings[$Matches[1].ToLowerInvariant()] = $Matches[2].Trim().Trim('"').Trim("'")
        }
    }

    $configuredPort = 3306
    if ($settings.ContainsKey('port')) {
        $parsedPort = 0
        if (-not [int]::TryParse([string]$settings['port'], [ref]$parsedPort)) {
            throw "The port setting in $mysqlConfig is not a valid integer. No MySQL process was started."
        }
        $configuredPort = $parsedPort
    }
    if ($configuredPort -ne $Port) {
        throw "The existing MySQL configuration uses port $configuredPort, but Servy expects $Port. No configuration was changed and no server was started."
    }

    if (-not $settings.ContainsKey('datadir') -or
        [string]::IsNullOrWhiteSpace([string]$settings['datadir'])) {
        throw "The existing [mysqld] section in $mysqlConfig does not declare datadir. Refusing to guess which database files to open."
    }
    $dataDirectory = [string]$settings['datadir']
    if (-not [System.IO.Path]::IsPathRooted($dataDirectory)) {
        throw "The configured MySQL data directory is not an absolute path. Refusing to resolve it relative to an unknown working directory."
    }
    $dataDirectory = [System.IO.Path]::GetFullPath($dataDirectory)
    if (-not (Test-Path -LiteralPath $dataDirectory -PathType Container) -or
        -not (Test-Path -LiteralPath (Join-Path $dataDirectory 'mysql') -PathType Container)) {
        throw "The existing MySQL data directory or its mysql system schema is missing: $dataDirectory. Do not run --initialize; inspect the existing data directory first."
    }

    # Discover every mysqld process before starting anything. If one exists
    # but is unhealthy or belongs to another config, fail closed rather than
    # risk opening the same data directory with a second server.
    $mysqlProcesses = @(Get-CimInstance Win32_Process -Filter "Name='mysqld.exe'" -ErrorAction Stop)
    $foreignProcesses = @($mysqlProcesses | Where-Object {
        [string]::IsNullOrWhiteSpace([string]$_.ExecutablePath) -or
        -not [string]::Equals([string]$_.ExecutablePath, $mysqlExe, [System.StringComparison]::OrdinalIgnoreCase) -or
        [string]::IsNullOrWhiteSpace([string]$_.CommandLine) -or
        ([string]$_.CommandLine).IndexOf($mysqlConfig, [System.StringComparison]::OrdinalIgnoreCase) -lt 0
    })
    if ($foreignProcesses.Count -gt 0) {
        $details = ($foreignProcesses | ForEach-Object { "PID $($_.ProcessId): $($_.ExecutablePath)" }) -join '; '
        throw "Port $Port is closed, but another or unidentifiable mysqld.exe process exists ($details). No second MySQL process was started. Inspect that process and its configured error log."
    }

    $mysqlServices = @(Get-CimInstance Win32_Service -ErrorAction Stop | Where-Object {
        ([string]$_.Name -match '(?i)mysql|maria') -or
        ([string]$_.DisplayName -match '(?i)mysql|maria') -or
        ([string]$_.PathName -match '(?i)(mysqld|mariadbd)\.exe')
    })
    $matchingServices = @($mysqlServices | Where-Object {
        -not [string]::IsNullOrWhiteSpace([string]$_.PathName) -and
        ([string]$_.PathName).IndexOf($mysqlExe, [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -and
        ([string]$_.PathName).IndexOf($mysqlConfig, [System.StringComparison]::OrdinalIgnoreCase) -ge 0
    })
    $matchingServiceNames = @($matchingServices | ForEach-Object { [string]$_.Name })
    $otherServices = @($mysqlServices | Where-Object {
        $candidateServiceName = [string]$_.Name
        $matchingServiceNames -notcontains $candidateServiceName
    })
    if ($otherServices.Count -gt 0) {
        $details = ($otherServices | ForEach-Object { "$($_.Name) [$($_.State)]" }) -join ', '
        throw "A MySQL/MariaDB Windows service exists but does not match the configured Servy binary and my.ini ($details). No second server was started."
    }
    if ($matchingServices.Count -gt 1) {
        $names = ($matchingServices | Select-Object -ExpandProperty Name) -join ', '
        throw "More than one Windows service points to the configured Servy MySQL instance ($names). Refusing ambiguous startup."
    }

    $serviceName = $null
    $startedProcess = $null
    $existingProcessIds = @($mysqlProcesses | ForEach-Object { [int]$_.ProcessId })

    if ($existingProcessIds.Count -gt 0) {
        Write-Host "Found the existing configured mysqld process(es): $($existingProcessIds -join ', '). Waiting for the existing instance; no duplicate will be started."
    } elseif ($matchingServices.Count -eq 1) {
        $serviceName = [string]$matchingServices[0].Name
        $service = Get-Service -Name $serviceName -ErrorAction Stop
        if ($service.Status -eq 'Stopped') {
            Write-Host "Starting the already-registered MySQL Windows service '$serviceName' with its existing my.ini..."
            Start-Service -Name $serviceName -ErrorAction Stop
        } elseif ($service.Status -ne 'Running' -and $service.Status -ne 'StartPending') {
            throw "The existing MySQL service '$serviceName' is in state '$($service.Status)'. It was not reconfigured or replaced."
        } else {
            Write-Host "The existing MySQL service '$serviceName' is already $($service.Status). Waiting for its listener."
        }
    } else {
        # No service and no mysqld process exist. Start the already-installed
        # server directly with its existing config and existing data directory.
        $logDirectory = Join-Path ([System.IO.Path]::GetTempPath()) 'ServyApp\mysql'
        New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
        $runStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
        $stdoutLog = Join-Path $logDirectory "$runStamp.stdout.log"
        $stderrLog = Join-Path $logDirectory "$runStamp.stderr.log"

        Write-Host "No MySQL service or server process is running. Starting the existing MySQL 8.0.41 binary with $mysqlConfig and data directory $dataDirectory..."
        $startedProcess = Start-Process -FilePath $mysqlExe -ArgumentList @("--defaults-file=$mysqlConfig") -WorkingDirectory (Split-Path -Parent $mysqlExe) -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog
        Write-Host "Started mysqld PID $($startedProcess.Id). Startup output: $logDirectory"
    }

    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        if (Test-ServyMySqlPort -Port $Port) {
            Write-Host "Existing MySQL instance is accepting TCP connections on 127.0.0.1:$Port. Running Django's read-only database identity/readiness gate next."
            return
        }

        if ($serviceName) {
            $currentService = Get-Service -Name $serviceName -ErrorAction Stop
            if ($currentService.Status -eq 'Stopped') {
                throw "MySQL service '$serviceName' stopped before opening port $Port. Inspect the MySQL error log configured in $mysqlConfig. No data files were changed by this launcher."
            }
        } elseif ($null -ne $startedProcess) {
            $currentProcess = Get-Process -Id $startedProcess.Id -ErrorAction SilentlyContinue
            if ($null -eq $currentProcess) {
                throw "mysqld PID $($startedProcess.Id) exited before opening port $Port. Inspect $stderrLog and the MySQL error log configured in $mysqlConfig. No initialization or repair was attempted."
            }
        } elseif ($existingProcessIds.Count -gt 0) {
            $liveIds = @($existingProcessIds | Where-Object { $null -ne (Get-Process -Id $_ -ErrorAction SilentlyContinue) })
            if ($liveIds.Count -eq 0) {
                throw "The previously observed mysqld process(es) exited before opening port $Port. Re-run only after inspecting the MySQL error log; this invocation did not start a replacement server."
            }
        }

        Start-Sleep -Milliseconds 500
    }

    $logHint = if ($settings.ContainsKey('log-error') -and -not [string]::IsNullOrWhiteSpace([string]$settings['log-error'])) {
        " MySQL error log setting: $($settings['log-error'])."
    } else {
        " Check the MySQL error log under the configured data directory: $dataDirectory."
    }
    throw "The existing MySQL instance did not open 127.0.0.1:$Port within $TimeoutSeconds seconds. Django and Vite were not started.$logHint"
}
