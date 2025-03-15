@echo off
setlocal EnableDelayedExpansion

echo ========================================
echo    Multi-Max Windows Updater
echo ========================================
echo This will completely reinstall Multi-Max from GitHub.
echo All existing files will be removed and replaced.
echo.

:: Store original directory and script path for reliable path handling
set "ORIGINAL_DIR=%CD%"
pushd "%~dp0"
set "SCRIPT_DIR=%CD%"
popd
set "PARENT_DIR=%SCRIPT_DIR%\.."
set "ROOT_DIR=%PARENT_DIR%"

:: Save the path to this script for restarting later if needed
set "THIS_SCRIPT=%~f0"

:: Check for administrative privileges
NET SESSION >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo This updater requires administrative privileges.
    echo Please right-click this file and select "Run as administrator".
    
    :: Try to restart with admin rights
    echo Attempting to restart with administrator privileges...
    powershell -Command "Start-Process -FilePath '%THIS_SCRIPT%' -Verb RunAs"
    exit /b 1
)

:: Check for internet connection
ping github.com -n 1 -w 1000 >nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Cannot connect to GitHub. Please check your internet connection.
    pause
    exit /b 1
)

:: Check for Git installation
where git >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Git is required but not found.
    echo Please install Git before updating Multi-Max.
    echo You can download Git from: https://git-scm.com/download/win
    pause
    exit /b 1
)

:: Check for Python installation
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is required but not found.
    echo Please install Python before updating Multi-Max.
    echo You can download Python from: https://www.python.org/downloads/windows/
    pause
    exit /b 1
)

:: Create a backup of user configuration
echo Creating backup of user configuration...
set "BACKUP_TIME=%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "BACKUP_TIME=%BACKUP_TIME: =0%"
set "BACKUP_DIR=%TEMP%\multi-max-backup-%BACKUP_TIME%"

mkdir "%BACKUP_DIR%" 2>nul

:: Backup important user files
if exist "%ROOT_DIR%\.env" (
    copy /Y "%ROOT_DIR%\.env" "%BACKUP_DIR%\.env" >nul
    echo Backed up .env configuration
)

:: Stop any running instances of Multi-Max
echo Stopping any running instances of Multi-Max...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Multi-Max*" >nul 2>&1
taskkill /F /IM pythonw.exe /FI "WINDOWTITLE eq Multi-Max*" >nul 2>&1

:: Wait a moment to ensure processes are terminated
timeout /t 2 >nul

:: Create a temporary directory for the clone
set "TEMP_DIR=%TEMP%\multi-max-update-%BACKUP_TIME%"
mkdir "%TEMP_DIR%" 2>nul

:: Download only the windows folder from the repository
echo Downloading latest Windows version from GitHub...
mkdir "%TEMP_DIR%" 2>nul
cd "%TEMP_DIR%"

:: Initialize a git repository
git init >nul 2>&1
git remote add origin https://github.com/dreamworks2050/multi-max.git >nul 2>&1

:: Set up sparse checkout to only get the windows folder
git config core.sparseCheckout true >nul 2>&1
echo windows/ > .git/info/sparse-checkout

:: Pull only the main branch with depth 1 to minimize download size
git pull --depth=1 origin main >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to download the latest version from GitHub.
    echo Please check your internet connection and try again.
    pause
    exit /b 1
)

echo Successfully downloaded the Windows-specific files.

:: Preserve all user data from windows directory before replacing
if exist "%ROOT_DIR%\.env" (
    copy /Y "%ROOT_DIR%\.env" "%BACKUP_DIR%\.env" >nul
    echo Backed up .env configuration
)

:: Remove all files from the current installation except .git and backups
echo Removing old files...
for /d %%D in ("%ROOT_DIR%\*") do (
    if /i not "%%~nxD"==".git" (
        if /i not "%%~nxD"=="logs" (
            rmdir /S /Q "%%D" >nul 2>&1
        )
    )
)

for %%F in ("%ROOT_DIR%\*") do (
    if not "%%~xF"==".git" (
        if not "%%~nxF"=="Update-MultiMax.bat" (
            del /F /Q "%%F" >nul 2>&1
        )
    )
)

:: Copy all files from the temporary clone to the installation directory
echo Installing new version...
if exist "%TEMP_DIR%\windows" (
    :: Create windows directory if it doesn't exist
    if not exist "%ROOT_DIR%\windows" mkdir "%ROOT_DIR%\windows"
    
    :: Copy all windows folder contents
    xcopy /E /I /H /Y "%TEMP_DIR%\windows\*" "%ROOT_DIR%\windows\" >nul
    
    :: Copy the Windows main.py to the root directory
    if exist "%TEMP_DIR%\windows\main.py" (
        copy /Y "%TEMP_DIR%\windows\main.py" "%ROOT_DIR%\main.py" >nul
        echo Installed Windows-specific main.py
    )
    
    echo Installed Windows-specific files
) else (
    echo ERROR: Windows folder not found in downloaded files.
    echo Please check the repository structure.
    pause
    exit /b 1
)

:: Restore user configuration
if exist "%BACKUP_DIR%\.env" (
    copy /Y "%BACKUP_DIR%\.env" "%ROOT_DIR%\.env" >nul
    echo Restored user configuration
)

:: Ensure the version file exists and is consistent
echo Setting up version information...
if exist "%TEMP_DIR%\windows\version.txt" (
    :: Copy version to windows directory
    copy /Y "%TEMP_DIR%\windows\version.txt" "%ROOT_DIR%\windows\version.txt" >nul
    :: Also copy to root for backwards compatibility
    copy /Y "%TEMP_DIR%\windows\version.txt" "%ROOT_DIR%\version.txt" >nul
    
    :: Read and display version
    for /f "tokens=*" %%a in ('type "%TEMP_DIR%\windows\version.txt"') do (
        echo Installed version: %%a
    )
)

:: Run the Windows installer to update dependencies
echo Installing dependencies...
if exist "%ROOT_DIR%\windows\Install-Windows.bat" (
    call "%ROOT_DIR%\windows\Install-Windows.bat" /silent
) else (
    echo WARNING: Windows installer not found. Dependency installation may be incomplete.
    timeout /t 3 >nul
)

:: Clean up temporary directory
rmdir /S /Q "%TEMP_DIR%" >nul 2>&1

echo.
echo ========================================
echo    Update completed successfully!
echo ========================================
echo.
echo Multi-Max has been updated to the latest version.
echo.

:: Launch the updated Multi-Max
echo Starting Multi-Max...
cd "%ROOT_DIR%"
start "" "%ROOT_DIR%\windows\Run-MultiMax.bat"

exit 