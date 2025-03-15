@echo off
setlocal EnableDelayedExpansion

:: Store original directory and script path for reliable path handling
set "ORIGINAL_DIR=%CD%"
pushd "%~dp0"
set "SCRIPT_DIR=%CD%"
popd
set "PARENT_DIR=%SCRIPT_DIR%\.."

echo ========================================
echo    Multi-Max Windows Installer
echo ========================================
echo This installer will set up Multi-Max optimized for Windows systems.
echo.

:: Check for administrative privileges
NET SESSION >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo This installer requires administrative privileges.
    echo Please right-click this file and select "Run as administrator".
    pause
    exit /b 1
)

:: Check if running on Windows
ver | findstr /i "Windows" >nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: This installer is specifically for Windows systems.
    echo You appear to be running on a different operating system.
    echo Please use the appropriate installer for your system.
    pause
    exit /b 1
)

:: Check for Python installation
echo Checking for Python...
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is required but not found.
    echo Please install Python and make sure it's in your PATH.
    echo Download Python from: https://www.python.org/downloads/windows/
    echo.
    echo Would you like to open the Python download page? (Y/N)
    set /p open_python=
    if /i "!open_python!"=="Y" (
        start https://www.python.org/downloads/windows/
    )
    pause
    exit /b 1
)

:: Check Python version
echo Checking Python version...
python -c "import sys; print('Python version:', '.'.join(map(str, sys.version_info[:3]))); sys.exit(0 if sys.version_info >= (3, 6) else 1)" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Multi-Max requires Python 3.6 or newer.
    echo Please upgrade your Python installation.
    pause
    exit /b 1
)

echo Checking for PowerShell...
where powershell >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: PowerShell is required but not found.
    echo Please make sure PowerShell is installed on your system.
    pause
    exit /b 1
)

:: Check for pip
echo Checking for pip...
python -m pip --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: pip is required but not found.
    echo Installing pip...
    python -m ensurepip --upgrade
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to install pip.
        echo Please install pip manually or reinstall Python with pip included.
        pause
        exit /b 1
    )
)

:: Check for FFmpeg (required for video processing)
echo Checking for FFmpeg...
where ffmpeg >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: FFmpeg is not found in your PATH.
    echo Multi-Max requires FFmpeg for video processing.
    echo Would you like to open the FFmpeg download page? (Y/N)
    set /p install_ffmpeg=
    if /i "!install_ffmpeg!"=="Y" (
        start https://ffmpeg.org/download.html#build-windows
        echo Please download and install FFmpeg, then add it to your PATH.
        echo You can continue with installation, but some features may not work correctly.
    )
    echo.
)

:: Check for yt-dlp (required for YouTube streams)
echo Checking for yt-dlp...
where yt-dlp >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo INFO: yt-dlp command line tool not found. Will install Python package instead.
    python -m pip install -U yt-dlp
    if %ERRORLEVEL% NEQ 0 (
        echo WARNING: Failed to install yt-dlp package.
        echo YouTube streaming may not work correctly.
    ) else {
        echo Successfully installed yt-dlp package.
    }
)

:: Create necessary directories
echo Creating required directories...
if not exist "%PARENT_DIR%\logs" mkdir "%PARENT_DIR%\logs"

echo Starting Windows-specific installation...
echo This will install the Windows-compatible version of Multi-Max.
echo.

:: Create backup of existing files with timestamp
set "BACKUP_TIME=%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "BACKUP_TIME=%BACKUP_TIME: =0%"

if exist "%PARENT_DIR%\main.py" (
    echo Creating backup of existing main.py...
    copy /Y "%PARENT_DIR%\main.py" "%PARENT_DIR%\main.py.backup-%BACKUP_TIME%" >nul
    if %ERRORLEVEL% EQU 0 (
        echo Backup created as main.py.backup-%BACKUP_TIME%
    )
)

if exist "%PARENT_DIR%\.env" (
    echo Creating backup of existing .env...
    copy /Y "%PARENT_DIR%\.env" "%PARENT_DIR%\.env.backup-%BACKUP_TIME%" >nul
    if %ERRORLEVEL% EQU 0 (
        echo Backup created as .env.backup-%BACKUP_TIME%
    )
)

:: Run the PowerShell installer with bypass execution policy
echo Running PowerShell installer...
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%\..\install-multi-max.ps1"
set POWERSHELL_EXIT=%ERRORLEVEL%

:: Check if the installation was successful
if %POWERSHELL_EXIT% EQU 0 (
    echo.
    echo ========================================
    echo PowerShell installation completed successfully!
    echo ========================================
    echo.
) else (
    echo.
    echo ========================================
    echo PowerShell installation encountered an error!
    echo ========================================
    echo.
    echo PowerShell installation failed. Attempting direct file installation...
    echo.
    
    :: Try direct copy if PowerShell installer fails
    if exist "%SCRIPT_DIR%\main.py" (
        echo Copying Windows main.py to parent directory...
        copy /Y "%SCRIPT_DIR%\main.py" "%PARENT_DIR%\main.py" >nul
        if %ERRORLEVEL% EQU 0 (
            echo Successfully copied main.py
        ) else (
            echo Failed to copy main.py
        )
    )
    
    if exist "%SCRIPT_DIR%\.env" (
        echo Copying Windows .env to parent directory...
        copy /Y "%SCRIPT_DIR%\.env" "%PARENT_DIR%\.env" >nul
        if %ERRORLEVEL% EQU 0 (
            echo Successfully copied .env
        ) else (
            echo Failed to copy .env
        )
    )
)

echo.
echo Installing required dependencies...
python -m pip install -r "%SCRIPT_DIR%\requirements.txt"

if %ERRORLEVEL% EQU 0 (
    echo Dependencies installed successfully.
) else (
    echo Warning: Some dependencies could not be installed automatically.
    echo You may need to manually install dependencies using:
    echo python -m pip install -r "%SCRIPT_DIR%\requirements.txt"
)

echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
echo The Windows-compatible version of Multi-Max has been installed.
echo.
echo To run Multi-Max, navigate to the installation directory and run:
echo python main.py
echo.
echo Or use the desktop shortcut if created.
echo.

echo Would you like to create a desktop shortcut? (Y/N)
set /p createShortcut=

if /i "!createShortcut!"=="Y" (
    echo Creating desktop shortcut...
    powershell -ExecutionPolicy Bypass -Command "$WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut([System.Environment]::GetFolderPath('Desktop') + '\Multi-Max.lnk'); $Shortcut.TargetPath = '%SCRIPT_DIR%\Run-MultiMax.bat'; $Shortcut.WorkingDirectory = '%SCRIPT_DIR%'; $Shortcut.Description = 'Multi-Max Windows Version'; $Shortcut.Save()"
    if %ERRORLEVEL% EQU 0 (
        echo Shortcut created successfully.
    ) else (
        echo Warning: Failed to create desktop shortcut.
    )
)

echo.
echo Would you like to run Multi-Max now? (Y/N)
set /p runNow=

if /i "!runNow!"=="Y" (
    echo Starting Multi-Max...
    start "" "%SCRIPT_DIR%\Run-MultiMax.bat"
)

cd "%ORIGINAL_DIR%"
echo.
echo Press any key to exit installer...
pause
endlocal 