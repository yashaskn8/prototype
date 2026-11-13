@echo off
echo Starting Servy RAG Backend and Frontend...
start "Servy Backend (Django)" cmd /k "%~dp0run_backend.bat"
timeout /t 3 /nobreak >nul
start "Servy Frontend (React/Vite)" cmd /k "%~dp0run_frontend.bat"
echo.
echo Both servers started!
echo Open your browser at: http://localhost:5173
echo.
pause
