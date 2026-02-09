@echo off
echo Checking if Ollama is running...
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo Ollama is already running.
) else (
    echo Starting Ollama serve...
    start "" "C:\Users\LENOVO\AppData\Local\Programs\Ollama\ollama.exe" serve
    echo Waiting for Ollama to start...
    timeout /t 5 /nobreak >nul
)
echo Activating virtual environment...
call venv\Scripts\activate
if errorlevel 1 (
    echo Failed to activate virtual environment. Please ensure it exists.
    pause
    exit /b 1
)
echo Starting the application...
python app.py
