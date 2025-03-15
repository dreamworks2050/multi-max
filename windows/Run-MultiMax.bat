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

:: Add script directory to PYTHONPATH to ensure update_checker can be found
set "PYTHONPATH=%SCRIPT_DIR%;%PYTHONPATH%"

:: Process command line arguments
set SKIP_UPDATE_CHECK=0
set ADDITIONAL_ARGS=
set FORCE_UPDATE=0
set RUN_UPDATER_ONLY=0

:process_args
if "%~1"=="" goto :continue_startup
if /i "%~1"=="--skip-update-check" (
    set SKIP_UPDATE_CHECK=1
    shift
    goto :process_args
) else if /i "%~1"=="--force-update" (
    set FORCE_UPDATE=1
    shift
    goto :process_args
) else if /i "%~1"=="--version" (
    echo Checking Multi-Max version...
    python -c "import sys; sys.path.append(r'%SCRIPT_DIR%'); from update_checker import VERSION; print(f'Multi-Max (Windows) version {VERSION}')"
    popd
    exit /b 0
) else if /i "%~1"=="--update" (
    echo Checking for updates...
    set RUN_UPDATER_ONLY=1
    python "%SCRIPT_DIR%\update_checker.py" --auto-update
    
    if %ERRORLEVEL% EQU 100 (
        echo.
        echo Update was successful. Please restart Multi-Max to use the new version.
        popd
        pause
        exit /b 0
    ) else if %ERRORLEVEL% NEQ 0 (
        echo Update check failed. Please check your internet connection.
    )
    
    if "%~2"=="" (
        echo.
        echo Would you like to continue launching Multi-Max? (Y/N)
        set /p continue_after_update=
        if /i "!continue_after_update!"=="N" (
            popd
            exit /b 0
        )
    )
    shift
    goto :process_args
) else if /i "%~1"=="--debug-updater" (
    echo Running update checker diagnostics...
    call "%SCRIPT_DIR%\debug_update.bat"
    popd
    exit /b 0
) else (
    set ADDITIONAL_ARGS=!ADDITIONAL_ARGS! %1
    shift
    goto :process_args
)

:continue_startup

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

:: Check if update_checker.py exists
if not exist "%SCRIPT_DIR%\update_checker.py" (
    echo WARNING: update_checker.py not found. Update checks will be disabled.
    set SKIP_UPDATE_CHECK=1
    echo.
)

:: Create logs directory if it doesn't exist
if not exist "%PARENT_DIR%\logs" (
    echo Creating logs directory...
    mkdir "%PARENT_DIR%\logs"
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

:: Build the command line with the update check flag if needed
set "COMMAND_LINE=python main.py"
if %SKIP_UPDATE_CHECK% EQU 1 (
    set "COMMAND_LINE=%COMMAND_LINE% --skip-update-check"
)
if %FORCE_UPDATE% EQU 1 (
    set "COMMAND_LINE=%COMMAND_LINE% --force-update"
)
if not "%ADDITIONAL_ARGS%"=="" (
    set "COMMAND_LINE=%COMMAND_LINE% %ADDITIONAL_ARGS%"
)

:: Check if the virtual environment exists
if exist "multi-max\Scripts\activate.bat" (
    :: Activate the virtual environment and run main.py
    echo Activating virtual environment...
    call multi-max\Scripts\activate.bat
    
    echo.
    echo Starting Multi-Max (Windows Version)...
    echo.
    
    %COMMAND_LINE% > "%log_file%" 2>&1
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
    
    %COMMAND_LINE% > "%log_file%" 2>&1
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
    
    %COMMAND_LINE% > "%log_file%" 2>&1
    set ERRORLEVEL_SAVED=%ERRORLEVEL%
    
    :: Deactivate the virtual environment
    deactivate
) else (
    :: No virtual environment found, try direct run
    echo WARNING: No virtual environment found. Attempting to run directly...
    echo This might fail if dependencies aren't installed globally.
    echo.
    
    %COMMAND_LINE% > "%log_file%" 2>&1
    set ERRORLEVEL_SAVED=%ERRORLEVEL%
    
    if %ERRORLEVEL_SAVED% NEQ 0 (
        echo.
        echo ERROR: Failed to run Multi-Max. You may need to install dependencies.
        echo Try running: pip install -r "%SCRIPT_DIR%\requirements.txt"
        echo.
    )
)

echo.
if %ERRORLEVEL_SAVED% EQU 100 (
    echo Multi-Max has been updated.
    echo Please restart the application to use the new version.
) else if %ERRORLEVEL_SAVED% NEQ 0 (
    echo Multi-Max encountered an error while running.
    echo Please check the error messages above or in the log file:
    echo %log_file%
    echo.
    echo If you're having issues with the update checker, run:
    echo   Run-MultiMax.bat --debug-updater
) else (
    echo Multi-Max has exited normally.
    echo Log file saved to: %log_file%
)

:: Return to the original directory
cd "%SCRIPT_DIR%"
pause
cd "%ORIGINAL_DIR%"
endlocal 