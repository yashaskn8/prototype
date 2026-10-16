<#
.SYNOPSIS
    Servy local test runner — runs Django tests against an isolated SQLite in-memory database.

.DESCRIPTION
    Executes the Django test suite using SQLite instead of MySQL.
    This ensures tests can run locally without a MySQL server.

    Process-level environment variables are set BEFORE Django initialization,
    and the script verifies the resolved backend before proceeding.

    No production database is contacted. No persistent state is modified.

.PARAMETER TestLabel
    Django test label(s) to run. Defaults to "core".

.PARAMETER Verbosity
    Django test verbosity level (0-3). Defaults to 2.

.EXAMPLE
    .\tools\run_local_tests.ps1
    .\tools\run_local_tests.ps1 -TestLabel "core.tests_identifiers_canonical"
    .\tools\run_local_tests.ps1 -TestLabel "core" -Verbosity 1

.NOTES
    MySQL CI coverage is provided by .github/workflows/ci.yml and is NOT
    replaced by this local SQLite runner. A green local run validates logic
    compatibility; MySQL integration validation requires CI.
#>
param(
    [string]$TestLabel = "core",
    [int]$Verbosity = 2
)

$ErrorActionPreference = "Stop"

# ------------------------------------------------------------------
# 1. Set isolated environment BEFORE Django touches anything
# ------------------------------------------------------------------
$env:SERVY_DB_ENGINE = "sqlite3"
$env:DJANGO_DEBUG = "1"
$env:DJANGO_SETTINGS_MODULE = "servy_rag.settings"

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Servy Local Test Runner (SQLite Isolation Mode)" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  SERVY_DB_ENGINE   = $env:SERVY_DB_ENGINE"
Write-Host "  DJANGO_DEBUG      = $env:DJANGO_DEBUG"
Write-Host "  Test label        = $TestLabel"
Write-Host "  Verbosity         = $Verbosity"
Write-Host ""

# ------------------------------------------------------------------
# 2. Verify the resolved Django database backend
# ------------------------------------------------------------------
Write-Host "[1/3] Verifying Django database backend..." -ForegroundColor Yellow

$backendCheck = python -c @"
import os, sys
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'servy_rag.settings')
import django; django.setup()
from django.conf import settings
engine = settings.DATABASES['default']['ENGINE']
test_name = settings.DATABASES['default'].get('TEST', {}).get('NAME', 'N/A')
print(f'ENGINE={engine}')
print(f'TEST_DB={test_name}')
if 'mysql' in engine:
    print('ABORT: Resolved backend is MySQL. Refusing to proceed.')
    sys.exit(1)
if 'sqlite' not in engine:
    print(f'ABORT: Unexpected backend {engine}.')
    sys.exit(1)
print('OK: SQLite backend confirmed.')
"@

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "FATAL: Backend verification failed. Aborting." -ForegroundColor Red
    Write-Host $backendCheck
    exit 1
}

Write-Host $backendCheck -ForegroundColor Green
Write-Host ""

# ------------------------------------------------------------------
# 3. Run Django system check
# ------------------------------------------------------------------
Write-Host "[2/3] Running Django system checks..." -ForegroundColor Yellow
$prevErrorAction = $ErrorActionPreference
$ErrorActionPreference = "Continue"
python manage.py check --settings=servy_rag.settings
$checkExit = $LASTEXITCODE
$ErrorActionPreference = $prevErrorAction
if ($checkExit -ne 0) {
    Write-Host "FATAL: Django system check failed." -ForegroundColor Red
    exit 1
}
Write-Host "System checks passed." -ForegroundColor Green
Write-Host ""

# ------------------------------------------------------------------
# 4. Run the test suite
# ------------------------------------------------------------------
Write-Host "[3/3] Running tests: $TestLabel (verbosity=$Verbosity)" -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------"
$ErrorActionPreference = "Continue"
python manage.py test $TestLabel -v $Verbosity --no-input
$exitCode = $LASTEXITCODE
$ErrorActionPreference = $prevErrorAction

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
if ($exitCode -eq 0) {
    Write-Host "  LOCAL TESTS PASSED (SQLite isolation mode)" -ForegroundColor Green
    Write-Host "  NOTE: This does NOT replace MySQL CI validation." -ForegroundColor Yellow
} else {
    Write-Host "  LOCAL TESTS FAILED (exit code: $exitCode)" -ForegroundColor Red
}
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

exit $exitCode
