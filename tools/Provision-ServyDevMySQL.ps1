<# Provision the isolated local development databases for Servy (one-time, local only).

   Creates:
     servy_dev       + account servy_dev_app   (ALL on servy_dev.* only)        -> runtime
     test_servy_dev  + account servy_dev_test  (ALL on test_servy_dev.* only)   -> disposable tests
   No global privileges; no access to `servy` or `test_servy`.

   You type the MySQL root password at the mysql prompt in this terminal; it is
   never stored or shown. The two account passwords are generated here, written
   only to the git-ignored files .env.mysql.servy_dev and .env.mysql.servy_dev_test,
   and never printed. Existing files are never overwritten. #>
[CmdletBinding()]
param(
    [string]$MySqlBin = 'C:\mysql\mysql-8.0.41-winx64\bin',
    [string]$MySqlHost = '127.0.0.1',
    [int]$Port = 3306,
    [string]$AdminUser = 'root'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$mysql = Join-Path $MySqlBin 'mysql.exe'
if (-not (Test-Path -LiteralPath $mysql -PathType Leaf)) { throw "mysql.exe not found at $mysql" }

$targets = @(
    @{ Database = 'servy_dev'; User = 'servy_dev_app'; File = Join-Path $repoRoot '.env.mysql.servy_dev' },
    @{ Database = 'test_servy_dev'; User = 'servy_dev_test'; File = Join-Path $repoRoot '.env.mysql.servy_dev_test' }
)
foreach ($t in $targets) {
    if (Test-Path -LiteralPath $t.File) { throw "$($t.File) already exists; it was not changed. Remove it yourself only if you intend to re-provision." }
    Push-Location $repoRoot
    try { git check-ignore -q -- (Split-Path $t.File -Leaf); if ($LASTEXITCODE -ne 0) { throw "$($t.File) is not git-ignored; refusing to write credentials." } }
    finally { Pop-Location }
}

function New-Secret([int]$Length = 40) {
    $alphabet = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789'.ToCharArray()
    $bytes = New-Object byte[] $Length
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    -join ($bytes | ForEach-Object { $alphabet[$_ % $alphabet.Length] })
}

$sql = New-Object System.Text.StringBuilder
foreach ($t in $targets) {
    $t.Password = New-Secret
    [void]$sql.AppendLine("CREATE DATABASE IF NOT EXISTS ``$($t.Database)`` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
    [void]$sql.AppendLine("CREATE USER '$($t.User)'@'$MySqlHost' IDENTIFIED BY '$($t.Password)';")
    [void]$sql.AppendLine("GRANT ALL PRIVILEGES ON ``$($t.Database)``.* TO '$($t.User)'@'$MySqlHost';")
}
[void]$sql.AppendLine('FLUSH PRIVILEGES;')

# The SQL contains the new passwords, so it lives only in a private temp file
# that is deleted immediately after mysql reads it.
$sqlFile = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), "servy-provision-$([guid]::NewGuid().ToString('N')).sql")
try {
    [System.IO.File]::WriteAllText($sqlFile, $sql.ToString(), (New-Object System.Text.UTF8Encoding($false)))
    $sql = $null
    Write-Host "Enter the MySQL '$AdminUser' password when prompted (input is hidden)."
    & $mysql "--host=$MySqlHost" "--port=$Port" "--user=$AdminUser" '--password' "--execute=source $($sqlFile -replace '\\','/')"
    if ($LASTEXITCODE -ne 0) { throw "mysql exited with code $LASTEXITCODE; no credential files were written." }
}
finally {
    if (Test-Path -LiteralPath $sqlFile) { Remove-Item -LiteralPath $sqlFile -Force }
}

foreach ($t in $targets) {
    $content = @(
        "MYSQL_HOST=$MySqlHost", "MYSQL_PORT=$Port",
        "MYSQL_DATABASE=$($t.Database)", "MYSQL_USER=$($t.User)", "MYSQL_PASSWORD=$($t.Password)"
    ) -join "`n"
    [System.IO.File]::WriteAllText($t.File, $content + "`n", (New-Object System.Text.UTF8Encoding($false)))
    $t.Password = $null
    Write-Host "Wrote $(Split-Path $t.File -Leaf) (git-ignored; password not shown)."
}
Write-Host 'Provisioned servy_dev (servy_dev_app) and test_servy_dev (servy_dev_test). No global privileges were granted.'
