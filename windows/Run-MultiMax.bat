@echo off
setlocal EnableDelayedExpansion

:: Store original directory to return to later
set "ORIGINAL_DIR=%CD%"

echo ========================================
echo    Multi-Max Windows Launcher
echo ========================================
echo.

:: Verify Windows environment
ver | findstr /i "Windows" >nul
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: This launcher is designed for Windows systems.
    echo It may not work correctly on other operating systems.
    echo.
)

:: Get script directory for reliable path handling
pushd "%~dp0"
set "SCRIPT_DIR=%CD%"
set "PARENT_DIR=%CD%\.."

:: Check if we're running from the windows directory
if not exist "check_windows_version.ps1" (
    echo ERROR: This launcher must be run from the windows directory.
    echo Please navigate to the windows directory first.
    popd
    pause
    exit /b 1
)

:: Check if Python is installed
echo Checking Python installation...
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not found in your PATH.
    echo Please install Python and ensure it's added to your PATH.
    popd
    pause
    exit /b 1
)

:: Check Python version
python -c "import sys; print('Python version:', '.'.join(map(str, sys.version_info[:3]))); sys.exit(0 if sys.version_info >= (3, 6) else 1)" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Your Python version may be outdated.
    echo Multi-Max requires Python 3.6 or newer.
    echo.
)

:: Check for FFmpeg (required for video processing)
echo Checking for FFmpeg...
where ffmpeg >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: FFmpeg is not found in your PATH.
    echo Some video processing features may not work correctly.
    echo Would you like to open the FFmpeg download page? (Y/N)
    set /p install_ffmpeg=
    if /i "!install_ffmpeg!"=="Y" (
        start https://ffmpeg.org/download.html#build-windows
        echo Please download and install FFmpeg, then add it to your PATH.
        echo After installing, please restart this launcher.
        pause
        exit /b 1
    )
    echo.
)

:: Run the version check to ensure we have the Windows version
echo Checking if Windows version is installed...
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%\check_windows_version.ps1"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Version check failed. Please ensure PowerShell is available.
    popd
    pause
    exit /b 1
)

:: Change to the parent directory where main.py is located
cd "%PARENT_DIR%"

:: Check if main.py exists
if not exist "main.py" (
    echo ERROR: main.py not found. Multi-Max may not be installed correctly.
    echo Please run the Windows installer first.
    cd "%SCRIPT_DIR%"
    pause
    exit /b 1
)

:: Check for dependencies
echo Checking for required Python packages...
python -c "try: import cv2, numpy, pygame, dotenv, psutil, queue; print('All required packages are installed.'); exit(0); except ImportError as e: print(f'Missing package: {str(e)}'); exit(1)" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Some required Python packages are missing.
    echo.
    echo Would you like to install the required dependencies now? (Y/N)
    set /p install_deps=
    if /i "!install_deps!"=="Y" (
        echo Installing dependencies...
        pip install -r "%SCRIPT_DIR%\requirements.txt"
        
        if %ERRORLEVEL% NEQ 0 (
            echo ERROR: Failed to install dependencies. Please try manually:
            echo pip install -r "%SCRIPT_DIR%\requirements.txt"
            pause
            exit /b 1
        ) else (
            echo Successfully installed dependencies.
        )
    ) else (
        echo WARNING: Continuing without installing dependencies.
        echo The application may not function correctly.
        echo.
    )
)

:: Create temp directory for logging if it doesn't exist
if not exist "%PARENT_DIR%\logs" (
    mkdir "%PARENT_DIR%\logs"
)

:: Set log file with timestamp
set "log_file=%PARENT_DIR%\logs\multimax_log_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.txt"
set "log_file=%log_file: =0%"

:: Check if the virtual environment exists
if exist "multi-max\Scripts\activate.bat" (
    :: Activate the virtual environment and run main.py
    echo Activating virtual environment...
    call multi-max\Scripts\activate.bat
    
    echo.
    echo Starting Multi-Max (Windows Version)...
    echo.
    
    python main.py > "%log_file%" 2>&1
    set ERRORLEVEL_SAVED=%ERRORLEVEL%
    
    :: Deactivate the virtual environment
    deactivate
) else if exist ".venv\Scripts\activate.bat" (
    :: Try alternate venv name
    echo Activating virtual environment (.venv)...
    call .venv\Scripts\activate.bat
    
    echo.
    echo Starting Multi-Max (Windows Version)...
    echo.
    
    python main.py > "%log_file%" 2>&1
    set ERRORLEVEL_SAVED=%ERRORLEVEL%
    
    :: Deactivate the virtual environment
    deactivate
) else if exist "venv\Scripts\activate.bat" (
    :: Try another alternate venv name
    echo Activating virtual environment (venv)...
    call venv\Scripts\activate.bat
    
    echo.
    echo Starting Multi-Max (Windows Version)...
    echo.
    
    python main.py > "%log_file%" 2>&1
    set ERRORLEVEL_SAVED=%ERRORLEVEL%
    
    :: Deactivate the virtual environment
    deactivate
) else (
    :: No virtual environment found, try direct run
    echo WARNING: No virtual environment found. Attempting to run directly...
    echo This might fail if dependencies aren't installed globally.
    echo.
    
    python main.py > "%log_file%" 2>&1
    set ERRORLEVEL_SAVED=%ERRORLEVEL%
    
    if %ERRORLEVEL_SAVED% NEQ 0 (
        echo.
        echo ERROR: Failed to run Multi-Max. You may need to install dependencies.
        echo Try running: pip install -r "%SCRIPT_DIR%\requirements.txt"
        echo.
    )
)

echo.
if %ERRORLEVEL_SAVED% NEQ 0 (
    echo Multi-Max encountered an error while running.
    echo Please check the error messages above or in the log file:
    echo %log_file%
) else (
    echo Multi-Max has exited normally.
    echo Log file saved to: %log_file%
)

:: Return to the original directory
cd "%SCRIPT_DIR%"
pause
cd "%ORIGINAL_DIR%"
endlocal 