@echo off
setlocal EnableDelayedExpansion

echo ========================================
echo    Multi-Max Direct Launcher
echo ========================================
echo.

:: Get script directory for reliable path handling
pushd "%~dp0"
set "SCRIPT_DIR=%CD%"
set "PARENT_DIR=%CD%\.."

:: Add windows directory to Python path for update_checker
set "PYTHONPATH=%SCRIPT_DIR%;%PARENT_DIR%;%PYTHONPATH%"

:: Make sure the parent directory exists and has a main.py
if not exist "%PARENT_DIR%\main.py" (
    :: Try to repair by copying main.py from windows dir
    if exist "%SCRIPT_DIR%\main.py" (
        echo Copying main.py to parent directory...
        copy "%SCRIPT_DIR%\main.py" "%PARENT_DIR%\main.py" /Y
    ) else (
        echo ERROR: main.py not found. Cannot launch application.
        pause
        exit /b 1
    )
)

:: Make sure update_checker.py is accessible
if not exist "%PARENT_DIR%\update_checker.py" (
    if exist "%SCRIPT_DIR%\update_checker.py" (
        echo Copying update_checker.py to parent directory for easier imports...
        copy "%SCRIPT_DIR%\update_checker.py" "%PARENT_DIR%\update_checker.py" /Y
    )
)

:: Make sure VERSION file exists
if not exist "%PARENT_DIR%\VERSION" (
    echo Creating VERSION file...
    echo 1.0.0> "%PARENT_DIR%\VERSION"
)

:: Create logs directory if it doesn't exist
if not exist "%PARENT_DIR%\logs" (
    echo Creating logs directory...
    mkdir "%PARENT_DIR%\logs"
)

:: Change to the parent directory to run the app
cd "%PARENT_DIR%"

:: Look for the virtual environment
set "VENV_ACTIVATED=0"

if exist "%PARENT_DIR%\multi-max\Scripts\activate.bat" (
    echo Activating multi-max virtual environment...
    call "%PARENT_DIR%\multi-max\Scripts\activate.bat"
    set "VENV_ACTIVATED=1"
) else if exist "%PARENT_DIR%\venv\Scripts\activate.bat" (
    echo Activating venv virtual environment...
    call "%PARENT_DIR%\venv\Scripts\activate.bat"
    set "VENV_ACTIVATED=1"
) else if exist "%PARENT_DIR%\.venv\Scripts\activate.bat" (
    echo Activating .venv virtual environment...
    call "%PARENT_DIR%\.venv\Scripts\activate.bat"
    set "VENV_ACTIVATED=1"
) else (
    echo WARNING: No virtual environment found.
    echo Using system Python (this may not work if dependencies aren't installed globally)
)

:: Install required packages if needed
if "%VENV_ACTIVATED%"=="1" (
    echo Checking for required packages...
    pip list | findstr "numpy" >nul
    if %ERRORLEVEL% NEQ 0 (
        echo Installing required packages...
        pip install numpy pygame opencv-python python-dotenv psutil yt-dlp
    )
)

:: Actually run the application
echo.
echo Running Multi-Max (from directory: %CD%)...
echo.

python "%PARENT_DIR%\main.py" %*

:: Deactivate virtual environment if it was activated
if "%VENV_ACTIVATED%"=="1" (
    deactivate
)

echo.
echo Multi-Max has exited.
echo.

:: Return to original directory
cd "%SCRIPT_DIR%"
pause
endlocal 