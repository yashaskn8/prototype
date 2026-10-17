$statusFile = "C:\mysql\service_install_status.txt"
try {
    Write-Host "Installing MySQL80 service..."
    $installOut = & "C:\mysql\mysql-8.0.41-winx64\bin\mysqld.exe" --install MySQL80 --defaults-file="C:\mysql\my.ini" 2>&1
    Set-Service -Name MySQL80 -StartupType Automatic
    Start-Service -Name MySQL80
    $svc = Get-Service -Name MySQL80
    "SUCCESS: Status=$($svc.Status), StartupType=$($svc.StartType), Output=$installOut" | Out-File -FilePath $statusFile -Encoding utf8
} catch {
    "ERROR: $($_.Exception.Message)" | Out-File -FilePath $statusFile -Encoding utf8
}
