@echo off
echo =========================================================
echo Installing MySQL 8.0 Windows Service...
echo =========================================================
cd /d "C:\mysql"
"C:\mysql\mysql-8.0.41-winx64\bin\mysqld.exe" --install MySQL80 --defaults-file="C:\mysql\my.ini"
sc config MySQL80 start= auto
net start MySQL80
echo.
echo Service status:
sc query MySQL80
echo.
pause
