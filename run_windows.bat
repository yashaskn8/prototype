@echo off
setlocal
if not exist .venv (
  echo Virtual environment not found. Run setup_windows.bat first.
  exit /b 1
)
call .venv\Scripts\activate
python manage.py migrate
python manage.py runserver
