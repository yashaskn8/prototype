@echo off
echo Starting Servy RAG Backend (Django on port 8000)...
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python manage.py runserver 8000
