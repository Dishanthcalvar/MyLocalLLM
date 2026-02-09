@echo off
setlocal

rem Ensure Ollama binary directory is on PATH
set "OLLAMA_BIN=%LocalAppData%\Programs\Ollama"
if exist "%OLLAMA_BIN%\ollama.exe" (
    set "PATH=%OLLAMA_BIN%;%PATH%"
) else (
    echo Could not find Ollama in "%OLLAMA_BIN%". Please install or update the path in start.bat.
    pause
    exit /b 1
)

echo Checking if Ollama is running...
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo Ollama is already running.
) else (
    echo Starting Ollama serve...
    start "Ollama Serve" /MIN ollama serve
    echo Waiting for Ollama to start...
    timeout /t 5 /nobreak >nul
)

echo Activating virtual environment...
if exist "venv\Scripts\activate" (
    call venv\Scripts\activate
) else (
    echo Virtual environment not found. Please create it or adjust the script.
    pause
    exit /b 1
)

echo Starting the Streamlit application...
streamlit run app.py

endlocal
