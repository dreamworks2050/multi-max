@echo off
setlocal enabledelayedexpansion

echo ======================================================
echo     Multi-Max Windows Installation Fix Tool
echo ======================================================
echo.

rem Get script directory for reliable path handling
pushd "%~dp0"
set "SCRIPT_DIR=%CD%"
set "PARENT_DIR=%CD%\.."

echo Script directory: %SCRIPT_DIR%
echo Repository directory: %PARENT_DIR%

echo.
echo === Checking prerequisites ===
echo.

rem Check for Python
echo Checking for Python...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.7 or higher from https://www.python.org/downloads/
    echo Be sure to check 'Add Python to PATH' during installation.
    goto :error
)

for /f "tokens=*" %%a in ('python -c "import platform; print(platform.python_version())"') do set PYTHON_VERSION=%%a
echo Python version: %PYTHON_VERSION%

rem Check for Git (optional)
echo Checking for Git...
git --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Git is not installed. Update features will not work.
) else (
    echo Git is installed.
)

echo.
echo === Gathering system information ===
echo.

echo Windows version: %OS%
echo Current directory: %CD%

echo.
echo === Checking installation structure ===
echo.

rem Create logs directory if it doesn't exist
if not exist "%PARENT_DIR%\logs" (
    echo Logs directory not found. Creating...
    mkdir "%PARENT_DIR%\logs"
) else (
    echo Logs directory exists.
)

rem Create VERSION file if it doesn't exist
if not exist "%PARENT_DIR%\VERSION" (
    echo VERSION file not found. Creating...
    echo 1.0.0 > "%PARENT_DIR%\VERSION"
) else (
    echo VERSION file exists.
)

rem Check and fix main.py
if not exist "%PARENT_DIR%\main.py" (
    echo ERROR: main.py not found in repository root.
    
    rem Try to repair by copying from windows directory
    if exist "%SCRIPT_DIR%\main.py" (
        echo Found Windows main.py. Copying to repository root...
        copy "%SCRIPT_DIR%\main.py" "%PARENT_DIR%\" > nul
        echo main.py copied successfully.
    ) else (
        echo ERROR: Windows main.py not found either. Cannot repair.
    )
) else (
    echo main.py exists. Checking if it's the Windows version...
    
    rem Check for Windows marker in the file
    findstr "__windows_specific_version__" "%PARENT_DIR%\main.py" > nul
    if %ERRORLEVEL% NEQ 0 (
        echo main.py doesn't contain Windows marker.
        
        rem Backup and replace
        if exist "%SCRIPT_DIR%\main.py" (
            echo Backing up current main.py and replacing with Windows version...
            copy "%PARENT_DIR%\main.py" "%PARENT_DIR%\main.py.backup.%date:~-4,4%%date:~-7,2%%date:~-10,2%" > nul
            copy "%SCRIPT_DIR%\main.py" "%PARENT_DIR%\" > nul
            echo main.py replaced with Windows version.
        ) else (
            echo ERROR: Windows main.py not found. Cannot replace.
        )
    ) else (
        echo main.py contains Windows marker. Looks good.
    )
)

rem Check and fix update_checker.py
if not exist "%PARENT_DIR%\update_checker.py" (
    echo update_checker.py not found in repository root.
    
    rem Try to copy the simplified version
    if exist "%SCRIPT_DIR%\simple_update_checker.py" (
        echo Found simplified update checker. Copying to repository root...
        copy "%SCRIPT_DIR%\simple_update_checker.py" "%PARENT_DIR%\update_checker.py" > nul
        echo update_checker.py copied successfully.
    ) else (
        echo WARNING: simplified update checker not found. Updates may not work.
    )
) else (
    echo update_checker.py exists. Backing up and replacing with simplified version...
    copy "%PARENT_DIR%\update_checker.py" "%PARENT_DIR%\update_checker.py.backup.%date:~-4,4%%date:~-7,2%%date:~-10,2%" > nul
    
    if exist "%SCRIPT_DIR%\simple_update_checker.py" (
        copy "%SCRIPT_DIR%\simple_update_checker.py" "%PARENT_DIR%\update_checker.py" > nul
        echo update_checker.py replaced with simplified version.
    ) else (
        echo WARNING: simplified update checker not found. Keeping existing version.
    )
)

echo.
echo === Checking Python environment ===
echo.

rem Check for virtual environment
set "VENV_FOUND=0"
set "VENV_PATH="

if exist "%PARENT_DIR%\venv\Scripts\activate.bat" (
    set "VENV_FOUND=1"
    set "VENV_PATH=%PARENT_DIR%\venv"
    echo Found virtual environment: venv
) else if exist "%PARENT_DIR%\.venv\Scripts\activate.bat" (
    set "VENV_FOUND=1"
    set "VENV_PATH=%PARENT_DIR%\.venv"
    echo Found virtual environment: .venv
) else if exist "%PARENT_DIR%\multi-max\Scripts\activate.bat" (
    set "VENV_FOUND=1"
    set "VENV_PATH=%PARENT_DIR%\multi-max"
    echo Found virtual environment: multi-max
)

if "%VENV_FOUND%"=="1" (
    echo Testing virtual environment activation...
    call "%VENV_PATH%\Scripts\activate.bat"
    
    if %ERRORLEVEL% NEQ 0 (
        echo Error activating virtual environment.
        echo Creating a new virtual environment...
        
        rem Rename broken venv directory
        ren "%VENV_PATH%" "%VENV_PATH%_broken"
        
        rem Create a new one
        cd "%PARENT_DIR%"
        python -m venv venv
        
        rem Activate and install packages
        call "%PARENT_DIR%\venv\Scripts\activate.bat"
        python -m pip install --upgrade pip
        python -m pip install numpy opencv-python python-dotenv pygame psutil
        call deactivate
        
        echo New virtual environment created and packages installed.
    ) else (
        echo Virtual environment activation successful.
        
        rem Check for missing packages
        echo Checking for required packages...
        set "MISSING_PACKAGES="
        
        python -c "import numpy" 2>nul
        if %ERRORLEVEL% NEQ 0 set "MISSING_PACKAGES=!MISSING_PACKAGES! numpy"
        
        python -c "import cv2" 2>nul
        if %ERRORLEVEL% NEQ 0 set "MISSING_PACKAGES=!MISSING_PACKAGES! opencv-python"
        
        python -c "import dotenv" 2>nul
        if %ERRORLEVEL% NEQ 0 set "MISSING_PACKAGES=!MISSING_PACKAGES! python-dotenv"
        
        python -c "import pygame" 2>nul
        if %ERRORLEVEL% NEQ 0 set "MISSING_PACKAGES=!MISSING_PACKAGES! pygame"
        
        python -c "import psutil" 2>nul
        if %ERRORLEVEL% NEQ 0 set "MISSING_PACKAGES=!MISSING_PACKAGES! psutil"
        
        if not "!MISSING_PACKAGES!"=="" (
            echo Missing packages detected: !MISSING_PACKAGES!
            echo Installing missing packages...
            
            for %%p in (!MISSING_PACKAGES!) do (
                echo Installing %%p...
                python -m pip install %%p
            )
            
            echo Packages installed successfully.
        ) else (
            echo All required packages are installed.
        )
        
        call deactivate
    )
) else (
    echo No virtual environment found. Creating one...
    
    rem Create a new virtual environment
    cd "%PARENT_DIR%"
    python -m venv venv
    
    rem Activate and install packages
    call "%PARENT_DIR%\venv\Scripts\activate.bat"
    python -m pip install --upgrade pip
    python -m pip install numpy opencv-python python-dotenv pygame psutil
    call deactivate
    
    echo New virtual environment created and packages installed.
)

echo.
echo === Testing application launch capability ===
echo.

rem Check if Simple-Run.bat exists
if not exist "%SCRIPT_DIR%\Simple-Run.bat" (
    echo Simple-Run.bat not found.
    echo Please run the full installer to recreate Simple-Run.bat
) else (
    echo Simple-Run.bat exists. Launch this file to start the application.
)

echo.
echo ======================================================
echo         Installation Fix Completed!
echo ======================================================
echo.
echo The Multi-Max installation has been repaired.
echo To run the application, use 'Simple-Run.bat' in the windows directory.
echo.
echo If you still encounter issues, please:
echo 1. Check the logs in the 'logs' directory
echo 2. Run the full installer again
echo 3. Report the issue with logs attached
echo.
echo ======================================================
echo.

rem Ask if user wants to run the application
set /p RUN_NOW="Would you like to try running Multi-Max now? (y/n): "
if /i "%RUN_NOW%"=="y" (
    echo Launching Multi-Max...
    if exist "%SCRIPT_DIR%\Simple-Run.bat" (
        call "%SCRIPT_DIR%\Simple-Run.bat"
    ) else (
        echo ERROR: Simple-Run.bat not found. Cannot launch application.
    )
)

goto :end

:error
echo.
echo Installation fix failed.
echo Please fix the issues mentioned above and try again.
echo.
pause
exit /b 1

:end
echo.
echo Fix process completed.
pause
exit /b 0 