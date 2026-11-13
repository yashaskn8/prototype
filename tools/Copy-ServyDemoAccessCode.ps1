<# Copy a local demo access code to the clipboard for signing in.

   Customer RAG  (http://127.0.0.1:5175/):  .\tools\Copy-ServyDemoAccessCode.ps1 -App Customer
   Technician    (http://127.0.0.1:5176/):  .\tools\Copy-ServyDemoAccessCode.ps1 -App Technician

   The code is read from the same private local file the launcher uses
   (tools\Start-ServyRealDemo.ps1 / tools\Start-ServyTechnicianDemo.ps1) and is
   placed on the clipboard only. It is never printed, logged, or written
   anywhere else, and this script never creates or changes a code. #>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidateSet('Customer', 'Technician')][string]$App
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Same private folder as both launchers.
$privateLocal = 'C:\Users\prana\AppData\Local\Packages\OpenAI.Codex_2p2nqsd0c76g0\LocalCache\Local'
$targets = @{
    Customer   = @{
        File = Join-Path $privateLocal 'Servy\private_evaluations\servy_real_demo_20261031_01\real_demo_access.txt'
        Url = 'http://127.0.0.1:5175/'
        Launcher = '.\tools\Start-ServyRealDemo.ps1'
    }
    Technician = @{
        File = Join-Path $privateLocal 'Servy\private_evaluations\servy_technician_demo_20261111_01\technician_access.txt'
        Url = 'http://127.0.0.1:5176/'
        Launcher = '.\tools\Start-ServyTechnicianDemo.ps1'
    }
}
$target = $targets[$App]

if (-not (Test-Path -LiteralPath $target.File -PathType Leaf)) {
    throw "No $App access code exists yet. Start the app once with $($target.Launcher); it creates the code."
}
$code = (Get-Content -LiteralPath $target.File -Raw).Trim()
if ($code.Length -lt 40) {
    throw "The $App access-code file is not valid. It was not changed."
}
Set-Clipboard -Value $code
$code = $null
Write-Host "$App access code copied to the clipboard (not shown)."
Write-Host "Open $($target.Url) and paste it into the access-code field."
