@echo off
echo ========================================
echo    Multi-Max Windows Launcher
echo ========================================
echo.

:: Check if we're running from the windows directory
if not exist "check_windows_version.ps1" (
    echo ERROR: This launcher must be run from the windows directory.
    echo Please navigate to the windows directory first.
    pause
    exit /b 1
)

:: Run the version check to ensure we have the Windows version
echo Checking if Windows version is installed...
powershell -ExecutionPolicy Bypass -File "check_windows_version.ps1"

:: Change to the parent directory where main.py is located
cd ..

:: Check if the virtual environment exists
if exist "multi-max\Scripts\activate.bat" (
    :: Activate the virtual environment and run main.py
    echo Activating virtual environment...
    call multi-max\Scripts\activate.bat
    
    echo.
    echo Starting Multi-Max (Windows Version)...
    echo.
    
    python main.py
    
    :: Deactivate the virtual environment
    deactivate
) else if exist ".venv\Scripts\activate.bat" (
    :: Try alternate venv name
    echo Activating virtual environment (.venv)...
    call .venv\Scripts\activate.bat
    
    echo.
    echo Starting Multi-Max (Windows Version)...
    echo.
    
    python main.py
    
    :: Deactivate the virtual environment
    deactivate
) else if exist "venv\Scripts\activate.bat" (
    :: Try another alternate venv name
    echo Activating virtual environment (venv)...
    call venv\Scripts\activate.bat
    
    echo.
    echo Starting Multi-Max (Windows Version)...
    echo.
    
    python main.py
    
    :: Deactivate the virtual environment
    deactivate
) else (
    :: No virtual environment found, try direct run
    echo WARNING: No virtual environment found. Attempting to run directly...
    echo This might fail if dependencies aren't installed globally.
    echo.
    
    python main.py
)

echo.
if %ERRORLEVEL% NEQ 0 (
    echo Multi-Max encountered an error while running.
    echo Please check the error messages above.
) else (
    echo Multi-Max has exited normally.
)

pause 