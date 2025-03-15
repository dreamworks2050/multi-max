@echo off
setlocal enabledelayedexpansion

echo ======================================================
echo                Multi-Max Simple Launcher
echo ======================================================
echo.

:: Get the script directory and parent directory
set "SCRIPT_DIR=%~dp0"
set "PARENT_DIR=%SCRIPT_DIR%.."

:: Add directories to Python path
set "PYTHONPATH=%PARENT_DIR%;%SCRIPT_DIR%;%PYTHONPATH%"
echo Setting PYTHONPATH to include: %SCRIPT_DIR% and %PARENT_DIR%

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

:: Check if main.py exists in parent directory
if not exist "%PARENT_DIR%\main.py" (
    echo Main script not found in parent directory.
    
    :: Try to copy from windows directory if it exists there
    if exist "%SCRIPT_DIR%main.py" (
        echo Copying main.py from windows directory...
        copy "%SCRIPT_DIR%main.py" "%PARENT_DIR%\" > nul
        echo Main script copied successfully.
    ) else (
        echo ERROR: main.py not found in windows directory either.
        echo Please ensure main.py is correctly installed.
        goto :error
    )
)

:: Change to parent directory
echo Changing to parent directory: %PARENT_DIR%
cd /d "%PARENT_DIR%"

:: Check for virtual environment and activate if found
set "VENV_ACTIVATED=0"
set "VENV_PATHS=multi-max venv .venv"

for %%p in (%VENV_PATHS%) do (
    if exist "%%p\Scripts\activate.bat" (
        echo Found virtual environment: %%p
        call "%%p\Scripts\activate.bat"
        set "VENV_ACTIVATED=1"
        goto :after_venv_check
    )
)

:after_venv_check
if "%VENV_ACTIVATED%"=="0" (
    echo WARNING: No virtual environment found. Using system Python.
    echo This might cause dependency issues.
)

:: Check for the simplified update checker and run it
if exist "%SCRIPT_DIR%simple_update_checker.py" (
    echo Running update check...
    python "%SCRIPT_DIR%simple_update_checker.py" %*
    
    set "UPDATE_RESULT=%ERRORLEVEL%"
    
    if "!UPDATE_RESULT!"=="100" (
        echo Application has been updated. Please restart the launcher.
        goto :end
    )
    
    if "!UPDATE_RESULT!"=="50" (
        echo.
        set /p "DO_UPDATE=Would you like to update now? (y/n): "
        if /i "!DO_UPDATE!"=="y" (
            echo Running update...
            python "%SCRIPT_DIR%simple_update_checker.py" --auto-update
            if "!ERRORLEVEL!"=="100" (
                echo Application has been updated. Please restart the launcher.
                goto :end
            )
        )
    )
) else (
    echo WARNING: Update checker not found at %SCRIPT_DIR%simple_update_checker.py
)

:: Run the main application
echo.
echo ======================================================
echo                Starting Multi-Max
echo ======================================================
echo.

python "%PARENT_DIR%\main.py" %*
set "APP_RESULT=%ERRORLEVEL%"

:: Deactivate virtual environment if it was activated
if "%VENV_ACTIVATED%"=="1" (
    call deactivate
)

echo.
if "!APP_RESULT!"=="0" (
    echo Multi-Max exited successfully.
) else (
    echo Multi-Max exited with code: !APP_RESULT!
)
goto :end

:error
echo.
echo ERROR: Failed to run Multi-Max.
echo.
echo Please report this issue with the following information:
echo - Windows version: %OS%
echo - Python version: 
python --version 2>nul || echo Python not found in PATH
echo - Current directory: %CD%
echo - Script directory: %SCRIPT_DIR%
echo.
pause
exit /b 1

:end
echo.
echo ======================================================
echo                Multi-Max Session Ended
echo ======================================================
pause
exit /b 0 