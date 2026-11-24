<#
.SYNOPSIS
    Servy local test runner - runs Django tests on the isolated MySQL test database.

.DESCRIPTION
    Selects the isolated test account and database through SERVY_MYSQL_ENV_FILE
    (.env.mysql.servy_dev_test -> test_servy_dev), verifies the resolved backend and
    database name before anything runs, then runs the tests with --keepdb (the
    restricted test account cannot create databases). MySQL is the only database
    engine; no other database is contacted and no other database is modified.

.PARAMETER TestLabel
    Django test label(s) to run. Defaults to "core".

.PARAMETER Verbosity
    Django test verbosity level (0-3). Defaults to 2.

.EXAMPLE
    .\tools\run_local_tests.ps1 -TestLabel "core.tests_identifiers_canonical"
    .\tools\run_local_tests.ps1 -TestLabel "core.tests_conversation_quality core.tests_manual_grounding" -Verbosity 1
#>
param(
    [string]$TestLabel = "core",
    [int]$Verbosity = 2
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

# ------------------------------------------------------------------
# 1. Select the isolated MySQL test database BEFORE Django starts
# ------------------------------------------------------------------
$envFile = ".env.mysql.servy_dev_test"
if (-not (Test-Path $envFile)) {
    Write-Host "FATAL: $envFile not found. It must name the isolated test account and test_servy_dev." -ForegroundColor Red
    exit 1
}
$env:SERVY_DB_ENGINE = "mysql"
$env:SERVY_MYSQL_ENV_FILE = $envFile
$env:DJANGO_DEBUG = "1"
$env:DJANGO_SETTINGS_MODULE = "servy_rag.settings"

Write-Host ""
Write-Host "  Servy Local Test Runner (isolated MySQL test database)" -ForegroundColor Cyan
Write-Host "  Test label: $TestLabel   Verbosity: $Verbosity"
Write-Host ""

# ------------------------------------------------------------------
# 2. Verify the resolved backend and database name (no connection is opened)
# ------------------------------------------------------------------
Write-Host "[1/2] Verifying the resolved database..." -ForegroundColor Yellow
$backendCheck = & $python -c @"
import sys, django
django.setup()
from django.conf import settings
db = settings.DATABASES['default']
name = db.get('NAME', '')
print('ENGINE=' + db['ENGINE'])
print('DATABASE=' + name)
if 'mysql' not in db['ENGINE'] or name != 'test_servy_dev':
    print('ABORT: tests run only on the isolated MySQL database test_servy_dev.')
    sys.exit(1)
print('OK: isolated MySQL test database confirmed.')
"@
if ($LASTEXITCODE -ne 0) {
    Write-Host $backendCheck
    Write-Host "FATAL: Database verification failed. Aborting." -ForegroundColor Red
    exit 1
}
Write-Host $backendCheck -ForegroundColor Green

# ------------------------------------------------------------------
# 3. Run the tests
# ------------------------------------------------------------------
Write-Host "[2/2] Running tests: $TestLabel" -ForegroundColor Yellow
$ErrorActionPreference = "Continue"
& $python manage.py test @($TestLabel -split '\s+') -v $Verbosity --keepdb --noinput
$exitCode = $LASTEXITCODE
if ($exitCode -eq 0) {
    Write-Host "LOCAL TESTS PASSED (isolated MySQL test database)" -ForegroundColor Green
} else {
    Write-Host "LOCAL TESTS FAILED (exit code: $exitCode)" -ForegroundColor Red
}
exit $exitCode
