@echo off
setlocal enabledelayedexpansion

echo ======================================================
echo           Multi-Max Simple Installation
echo ======================================================
echo.

:: Get the script directory
set "SCRIPT_DIR=%~dp0"
set "PARENT_DIR=%SCRIPT_DIR%.."

:: Check for Python
echo Checking for Python...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found in PATH.
    echo Please install Python 3.7 or higher and add it to your PATH.
    echo You can download Python from https://www.python.org/downloads/
    goto :error
)

:: Check for Git
echo Checking for Git...
git --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Git not found in PATH.
    echo Update features will not work without Git.
    echo You can download Git from https://git-scm.com/download/win
)

:: Create a virtual environment if it doesn't exist
if not exist "%PARENT_DIR%\venv" (
    echo Creating virtual environment...
    cd /d "%PARENT_DIR%"
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to create virtual environment.
        echo Please check your Python installation.
        goto :error
    )
    echo Virtual environment created successfully.
) else (
    echo Virtual environment already exists.
)

:: Activate the virtual environment
echo Activating virtual environment...
call "%PARENT_DIR%\venv\Scripts\activate.bat"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate virtual environment.
    goto :error
)

:: Install required packages
echo Installing required packages...
pip install --upgrade pip
pip install numpy opencv-python python-dotenv

:: Create logs directory if it doesn't exist
if not exist "%PARENT_DIR%\logs" (
    echo Creating logs directory...
    mkdir "%PARENT_DIR%\logs"
)

:: Create VERSION file if it doesn't exist
if not exist "%PARENT_DIR%\VERSION" (
    echo Creating VERSION file...
    echo 1.0.0 > "%PARENT_DIR%\VERSION"
)

:: Ensure main.py is in parent directory
if not exist "%PARENT_DIR%\main.py" (
    if exist "%SCRIPT_DIR%main.py" (
        echo Copying main.py to parent directory...
        copy "%SCRIPT_DIR%main.py" "%PARENT_DIR%\" > nul
    ) else (
        echo WARNING: main.py not found. Please ensure it's correctly installed.
    )
)

:: Deactivate virtual environment
call deactivate

echo.
echo ======================================================
echo         Multi-Max Installation Complete
echo ======================================================
echo.
echo Installation has been completed successfully.
echo To run Multi-Max, use the Simple-Run.bat launcher.
echo.
echo You can find the launcher at:
echo %SCRIPT_DIR%Simple-Run.bat
echo.
goto :end

:error
echo.
echo ======================================================
echo         Multi-Max Installation Failed
echo ======================================================
echo.
echo Please fix the errors and try again.
echo.

:end
pause
exit /b %ERRORLEVEL% 