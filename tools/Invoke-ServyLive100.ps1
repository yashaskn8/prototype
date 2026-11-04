param(
    [Parameter(Mandatory=$true)][ValidateSet('prepare','run')][string]$Phase,
    [Parameter(Mandatory=$true)][string]$DataDirectory
)
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectDirectory
try {
    & '.\.venv\Scripts\python.exe' -u 'tools\servy_live100_benchmark.py' $Phase --data-dir $DataDirectory
    if ($LASTEXITCODE -ne 0) { throw "Benchmark stopped with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
