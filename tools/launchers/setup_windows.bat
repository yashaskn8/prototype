@echo off
setlocal
rem Run from the repository root regardless of the caller's directory.
cd /d "%~dp0..\.."
if not exist .venv python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python manage.py migrate
python manage.py check

echo.
echo Setup complete. Start the app with tools\Start-ServyApp.ps1.
