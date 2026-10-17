@echo off
echo ====================================================================
echo Registering and Starting MySQL 8.0 Native Windows Service (MySQL80)
echo ====================================================================
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] This script must be run as Administrator!
    echo Please right-click this file and select "Run as administrator".
    pause
    exit /b 1
)

echo [1/3] Stopping any standalone mysqld processes...
taskkill /F /IM mysqld.exe >nul 2>&1

echo [2/3] Installing MySQL80 Windows Service...
"C:\mysql\mysql-8.0.41-winx64\bin\mysqld.exe" --install MySQL80 --defaults-file="C:\mysql\my.ini"

echo [3/3] Setting service startup to Automatic and starting service...
sc config MySQL80 start= auto
net start MySQL80

echo.
sc query MySQL80
echo.
echo ====================================================================
echo MySQL 8.0 Windows Service is registered and started successfully!
echo ====================================================================
pause
