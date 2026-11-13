@echo off
setlocal
rem Run from the repository root regardless of the caller's directory.
cd /d "%~dp0..\.."
if not exist .venv (
  echo Virtual environment not found. Run tools\launchers\setup_windows.bat first.
  exit /b 1
)
call .venv\Scripts\activate
python manage.py migrate
python manage.py runserver
