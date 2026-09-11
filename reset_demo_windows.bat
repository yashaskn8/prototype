@echo off
setlocal
if not exist .venv (
  echo Run run_windows.bat once first.
  exit /b 1
)
call .venv\Scripts\activate
python manage.py seed_demo --reset
