@echo off
setlocal
if not exist .venv python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python manage.py migrate
python manage.py check

echo.
echo Setup complete. Run run_windows.bat to start Servy RAG.
