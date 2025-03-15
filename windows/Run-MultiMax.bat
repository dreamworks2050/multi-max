@echo off
setlocal EnableDelayedExpansion

echo ========================================
echo    Multi-Max Windows Launcher
echo ========================================

:: Create a logs directory if it doesn't exist
if not exist "logs" mkdir logs
set "LOG_FILE=logs\launcher-%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log"
set "LOG_FILE=%LOG_FILE: =0%"

:: Start logging
echo Multi-Max Launcher started at %date% %time% > "%LOG_FILE%"

:: Store the original directory
set "ORIGINAL_DIR=%CD%"
echo Original directory: %ORIGINAL_DIR% >> "%LOG_FILE%"

:: Change to the script directory
cd /d "%~dp0"
set "WINDOWS_DIR=%CD%"
set "PARENT_DIR=%CD%\.."

echo Running from: %WINDOWS_DIR%
echo Script directory: %WINDOWS_DIR% >> "%LOG_FILE%"
echo Parent directory: %PARENT_DIR% >> "%LOG_FILE%"

:: Check if Python is installed
echo Checking for Python installation...
echo Checking for Python installation... >> "%LOG_FILE%"
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo ERROR: Python is not installed or not in PATH. >> "%LOG_FILE%"
    echo Please install Python from https://www.python.org/downloads/windows/
    echo.
    echo This error has been logged to: %LOG_FILE%
    pause
    exit /b 1
)
echo Python found in PATH >> "%LOG_FILE%"

:: Check for version file
if not exist "version.txt" (
    echo WARNING: version.txt not found, will create it from remote source >> "%LOG_FILE%"
    
    :: Try to get the version from the remote repository
    echo Attempting to get version from remote repository... >> "%LOG_FILE%"
    powershell -Command "try { $version = (Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/dreamworks2050/multi-max/main/windows/version.txt' -UseBasicParsing).Content.Trim(); if ($version) { $version | Out-File 'version.txt' -NoNewline; Write-Output \"Downloaded remote version: $version\" } else { Write-Output 'Empty response from remote' } } catch { Write-Output \"Error fetching remote version: $_\"; 'unknown' | Out-File 'version.txt' -NoNewline }" >> "%LOG_FILE%" 2>&1
    
    if %ERRORLEVEL% NEQ 0 (
        echo WARNING: Failed to get remote version, setting placeholder version >> "%LOG_FILE%"
        echo unknown > "version.txt"
    )
    
    echo Created version.txt file >> "%LOG_FILE%"
)

:: Check if main.py exists
if not exist "main.py" (
    echo WARNING: main.py not found in the windows directory.
    echo WARNING: main.py not found in the windows directory. >> "%LOG_FILE%"
    
    :: Check if it exists in the parent directory
    if exist "%PARENT_DIR%\main.py" (
        echo Found main.py in the parent directory.
        echo Found main.py in the parent directory. >> "%LOG_FILE%"
        echo Will copy it to the windows directory for proper execution...
        echo Copying main.py from parent directory... >> "%LOG_FILE%"
        copy "%PARENT_DIR%\main.py" "main.py" >nul
        
        if not exist "main.py" (
            echo ERROR: Failed to copy main.py to the windows directory.
            echo ERROR: Failed to copy main.py to the windows directory. >> "%LOG_FILE%"
            echo Please try running the application from the parent directory instead.
            echo.
            echo This error has been logged to: %LOG_FILE%
            cd /d "%ORIGINAL_DIR%"
            pause
            exit /b 1
        )
    ) else (
        echo ERROR: main.py not found in either windows or parent directory.
        echo ERROR: main.py not found in either windows or parent directory. >> "%LOG_FILE%"
        echo The application may not be properly installed.
        echo Please try reinstalling Multi-Max.
        echo.
        echo This error has been logged to: %LOG_FILE%
        cd /d "%ORIGINAL_DIR%"
        pause
        exit /b 1
    )
)

:: Check if update.py exists
if not exist "update.py" (
    echo WARNING: update.py not found, updates will not work properly >> "%LOG_FILE%"
    echo WARNING: update.py not found. Updates will not work properly.
    echo The application will still run, but you won't be able to update automatically.
)

:: Check if a virtual environment exists and activate it
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    echo Activating virtual environment from windows directory >> "%LOG_FILE%"
    call "venv\Scripts\activate.bat"
) else if exist "%PARENT_DIR%\venv\Scripts\activate.bat" (
    echo Activating virtual environment from parent directory...
    echo Activating virtual environment from parent directory >> "%LOG_FILE%"
    call "%PARENT_DIR%\venv\Scripts\activate.bat"
) else (
    echo No virtual environment found, using system Python >> "%LOG_FILE%"
)

:: Run the program
echo Starting Multi-Max...
echo Starting Multi-Max at %date% %time% >> "%LOG_FILE%"
python main.py %*
set EXIT_CODE=%ERRORLEVEL%

:: Report status
echo Application exited with code: %EXIT_CODE% >> "%LOG_FILE%"
if %EXIT_CODE% NEQ 0 (
    echo.
    echo Application exited with code: %EXIT_CODE%
    echo If you're experiencing issues, please check the logs or reinstall.
    echo.
    echo Log file: %LOG_FILE%
    echo The window will close in 5 seconds...
    timeout /t 5 >nul
)

:: Return to the original directory
cd /d "%ORIGINAL_DIR%"
echo Returned to original directory >> "%LOG_FILE%"
echo Launcher completed at %date% %time% >> "%LOG_FILE%"

exit /b %EXIT_CODE% 