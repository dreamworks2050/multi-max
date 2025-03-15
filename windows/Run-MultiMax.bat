@echo off
setlocal EnableDelayedExpansion

:: Store original directory and script path for reliable path handling
set "ORIGINAL_DIR=%CD%"
pushd "%~dp0"
set "SCRIPT_DIR=%CD%"
popd
set "PARENT_DIR=%SCRIPT_DIR%\.."

echo ========================================
echo    Multi-Max Windows Launcher
echo ========================================

:: Check if we have a virtual environment
set "VENV_ACTIVATE=%PARENT_DIR%\multi-max\Scripts\activate.bat"
set "ALT_VENV_ACTIVATE=%PARENT_DIR%\venv\Scripts\activate.bat"

if exist "%VENV_ACTIVATE%" (
    echo Activating virtual environment...
    call "%VENV_ACTIVATE%"
) else if exist "%ALT_VENV_ACTIVATE%" (
    echo Activating alternative virtual environment...
    call "%ALT_VENV_ACTIVATE%"
) else (
    echo No virtual environment found. Using system Python.
)

:: Check for Python
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is required but not found.
    echo Please install Python before running Multi-Max.
    echo You can download Python from: https://www.python.org/downloads/windows/
    pause
    exit /b 1
)

:: Navigate to the parent directory
cd "%PARENT_DIR%"

:: Check if main.py exists
if not exist "main.py" (
    echo ERROR: main.py not found in %PARENT_DIR%
    
    :: Try to find and copy the Windows version if it exists
    if exist "%SCRIPT_DIR%\main.py" (
        echo Found Windows version in windows directory. Copying it...
        copy /Y "%SCRIPT_DIR%\main.py" "%PARENT_DIR%\main.py" >nul
        echo Windows version copied successfully.
    ) else (
        echo ERROR: Could not find main.py in any expected location.
        pause
        exit /b 1
    )
)

:: Run the application
echo Starting Multi-Max...
python main.py

:: If we get here, the application has exited
echo Multi-Max has exited.
if "%1"=="nowait" exit /b 0
timeout /t 5
exit /b 0

:: Return to the original directory
cd "%SCRIPT_DIR%"
pause
cd "%ORIGINAL_DIR%"
endlocal 