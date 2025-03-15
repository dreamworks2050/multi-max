@echo off
setlocal EnableDelayedExpansion

:: Create a log file to track the update process
set "LOG_FILE=%TEMP%\multimax-update-log-%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.txt"
set "LOG_FILE=%LOG_FILE: =0%"

echo ======================================================== > "%LOG_FILE%"
echo    Multi-Max Windows Updater Log                         >> "%LOG_FILE%"
echo    Started at: %date% %time%                             >> "%LOG_FILE%"
echo ======================================================== >> "%LOG_FILE%"

echo ========================================
echo    Multi-Max Windows Updater
echo ========================================
echo This will completely reinstall Multi-Max from GitHub.
echo All existing files will be removed and replaced.
echo.
echo Creating log file at: %LOG_FILE%
echo.

:: Log the start of the process
echo Starting update process >> "%LOG_FILE%"

:: Store original directory and script path for reliable path handling
set "ORIGINAL_DIR=%CD%"
pushd "%~dp0"
set "SCRIPT_DIR=%CD%"
popd
set "PARENT_DIR=%SCRIPT_DIR%\.."
set "ROOT_DIR=%PARENT_DIR%"

echo Script directory: %SCRIPT_DIR% >> "%LOG_FILE%"
echo Parent directory: %PARENT_DIR% >> "%LOG_FILE%"
echo Root directory: %ROOT_DIR% >> "%LOG_FILE%"

:: Save the path to this script for restarting later if needed
set "THIS_SCRIPT=%~f0"

:: Check for administrative privileges
echo Checking for administrative privileges... >> "%LOG_FILE%"
NET SESSION >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo This updater requires administrative privileges.
    echo Please right-click this file and select "Run as administrator".
    
    echo Failed administrative privileges check >> "%LOG_FILE%"
    
    :: Try to restart with admin rights
    echo Attempting to restart with administrator privileges...
    echo Attempting to restart with administrator privileges... >> "%LOG_FILE%"
    
    :: Create a visible indicator that we're trying to elevate
    echo @echo off > "%TEMP%\multimax-restart-admin.bat"
    echo echo Multi-Max is requesting administrator privileges... >> "%TEMP%\multimax-restart-admin.bat"
    echo echo Please click "Yes" on the User Account Control prompt. >> "%TEMP%\multimax-restart-admin.bat"
    echo echo This window will close automatically. >> "%TEMP%\multimax-restart-admin.bat"
    echo timeout /t 10 >> "%TEMP%\multimax-restart-admin.bat"
    echo start /wait powershell -Command "Start-Process -FilePath '%THIS_SCRIPT%' -Verb RunAs" >> "%TEMP%\multimax-restart-admin.bat"
    echo exit >> "%TEMP%\multimax-restart-admin.bat"
    
    start "" "%TEMP%\multimax-restart-admin.bat"
    
    echo Exiting non-admin instance >> "%LOG_FILE%"
    exit /b 1
)

echo Administrative privileges confirmed >> "%LOG_FILE%"
echo Running with administrative privileges.

:: Check for internet connection
echo Checking internet connection... >> "%LOG_FILE%"
echo Checking internet connection...
ping github.com -n 1 -w 1000 >nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Cannot connect to GitHub. Please check your internet connection.
    echo Failed internet connection test >> "%LOG_FILE%"
    echo Press any key to exit...
    pause > nul
    exit /b 1
)

echo Internet connection confirmed >> "%LOG_FILE%"
echo Internet connection confirmed.

:: Check for Git installation
echo Checking for Git... >> "%LOG_FILE%"
echo Checking for Git...
where git >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Git is required but not found.
    echo Please install Git before updating Multi-Max.
    echo You can download Git from: https://git-scm.com/download/win
    echo Git not found >> "%LOG_FILE%"
    echo Press any key to exit...
    pause > nul
    exit /b 1
)

echo Git found >> "%LOG_FILE%"
echo Git installation confirmed.

:: Check for Python installation
echo Checking for Python... >> "%LOG_FILE%"
echo Checking for Python...
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is required but not found.
    echo Please install Python before updating Multi-Max.
    echo You can download Python from: https://www.python.org/downloads/windows/
    echo Python not found >> "%LOG_FILE%"
    echo Press any key to exit...
    pause > nul
    exit /b 1
)

echo Python found >> "%LOG_FILE%"
echo Python installation confirmed.

:: Create a backup of user configuration
echo Creating backup of user configuration... >> "%LOG_FILE%"
echo Creating backup of user configuration...
set "BACKUP_TIME=%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "BACKUP_TIME=%BACKUP_TIME: =0%"
set "BACKUP_DIR=%TEMP%\multi-max-backup-%BACKUP_TIME%"

mkdir "%BACKUP_DIR%" 2>nul
echo Backup directory: %BACKUP_DIR% >> "%LOG_FILE%"

:: Backup important user files
if exist "%ROOT_DIR%\.env" (
    copy /Y "%ROOT_DIR%\.env" "%BACKUP_DIR%\.env" >nul
    echo Backed up .env configuration >> "%LOG_FILE%"
    echo Backed up .env configuration
)

:: Stop any running instances of Multi-Max
echo Stopping any running instances of Multi-Max... >> "%LOG_FILE%"
echo Stopping any running instances of Multi-Max...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Multi-Max*" >nul 2>&1
taskkill /F /IM pythonw.exe /FI "WINDOWTITLE eq Multi-Max*" >nul 2>&1

:: Wait a moment to ensure processes are terminated
timeout /t 2 >nul

:: Create a temporary directory for the clone
set "TEMP_DIR=%TEMP%\multi-max-update-%BACKUP_TIME%"
mkdir "%TEMP_DIR%" 2>nul
echo Temporary directory: %TEMP_DIR% >> "%LOG_FILE%"

:: Download only the windows folder from the repository
echo Downloading latest Windows version from GitHub... >> "%LOG_FILE%"
echo.
echo Downloading latest Windows version from GitHub...
echo This may take a moment...
mkdir "%TEMP_DIR%" 2>nul
cd "%TEMP_DIR%"

:: Initialize a git repository
echo Initializing git repository... >> "%LOG_FILE%"
git init >nul 2>&1
git remote add origin https://github.com/dreamworks2050/multi-max.git >nul 2>&1

:: Set up sparse checkout to only get the windows folder
echo Setting up sparse checkout... >> "%LOG_FILE%"
git config core.sparseCheckout true >nul 2>&1
echo windows/ > .git/info/sparse-checkout

:: Pull only the main branch with depth 1 to minimize download size
echo Pulling from GitHub... >> "%LOG_FILE%"
git pull --depth=1 origin main >> "%LOG_FILE%" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to download the latest version from GitHub.
    echo Please check your internet connection and try again.
    echo Git pull failed with code %ERRORLEVEL% >> "%LOG_FILE%"
    type "%LOG_FILE%"
    echo.
    echo Press any key to exit...
    pause > nul
    exit /b 1
)

echo Successfully downloaded the Windows-specific files. >> "%LOG_FILE%"
echo Successfully downloaded the Windows-specific files.

:: Preserve all user data from windows directory before replacing
if exist "%ROOT_DIR%\windows\version.txt" (
    copy /Y "%ROOT_DIR%\windows\version.txt" "%BACKUP_DIR%\version.txt" >nul
    echo Backed up version information >> "%LOG_FILE%"
    echo Backed up version information
)

:: Remove all files from the current installation except .git and backups
echo Removing old files... >> "%LOG_FILE%"
echo.
echo Removing old files...

:: Log the directories that will be removed
echo Directories to remove: >> "%LOG_FILE%"
for /d %%D in ("%ROOT_DIR%\*") do (
    if /i not "%%~nxD"==".git" (
        if /i not "%%~nxD"=="logs" (
            echo - %%D >> "%LOG_FILE%"
        )
    )
)

:: Log the files that will be removed
echo Files to remove: >> "%LOG_FILE%"
for %%F in ("%ROOT_DIR%\*") do (
    if not "%%~xF"==".git" (
        if not "%%~nxF"=="Update-MultiMax.bat" (
            echo - %%F >> "%LOG_FILE%"
        )
    )
)

:: Actually remove the directories
for /d %%D in ("%ROOT_DIR%\*") do (
    if /i not "%%~nxD"==".git" (
        if /i not "%%~nxD"=="logs" (
            rmdir /S /Q "%%D" >nul 2>&1
            echo Removed directory: %%D >> "%LOG_FILE%"
        )
    )
)

:: Actually remove the files
for %%F in ("%ROOT_DIR%\*") do (
    if not "%%~xF"==".git" (
        if not "%%~nxF"=="Update-MultiMax.bat" (
            del /F /Q "%%F" >nul 2>&1
            echo Removed file: %%F >> "%LOG_FILE%"
        )
    )
)

:: Copy all files from the temporary clone to the installation directory
echo Installing new version... >> "%LOG_FILE%"
echo.
echo Installing new version...
if exist "%TEMP_DIR%\windows" (
    :: Create windows directory if it doesn't exist
    if not exist "%ROOT_DIR%\windows" mkdir "%ROOT_DIR%\windows"
    echo Created windows directory >> "%LOG_FILE%"
    
    :: Copy all windows folder contents
    xcopy /E /I /H /Y "%TEMP_DIR%\windows\*" "%ROOT_DIR%\windows\" >> "%LOG_FILE%" 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to copy windows files. >> "%LOG_FILE%"
        echo ERROR: Failed to copy windows files.
    ) else (
        echo Copied windows files successfully >> "%LOG_FILE%"
    )
    
    :: Copy the Windows main.py to the root directory
    if exist "%TEMP_DIR%\windows\main.py" (
        copy /Y "%TEMP_DIR%\windows\main.py" "%ROOT_DIR%\main.py" >> "%LOG_FILE%" 2>&1
        echo Installed Windows-specific main.py >> "%LOG_FILE%"
        echo Installed Windows-specific main.py
    ) else (
        echo WARNING: main.py not found in downloaded files >> "%LOG_FILE%"
        echo WARNING: main.py not found in downloaded files
    )
    
    echo Installed Windows-specific files >> "%LOG_FILE%"
    echo Installed Windows-specific files
) else (
    echo ERROR: Windows folder not found in downloaded files. >> "%LOG_FILE%"
    echo ERROR: Windows folder not found in downloaded files.
    echo Please check the repository structure.
    echo Directory listing of %TEMP_DIR%: >> "%LOG_FILE%"
    dir "%TEMP_DIR%" /s >> "%LOG_FILE%"
    type "%LOG_FILE%"
    echo.
    echo Press any key to exit...
    pause > nul
    exit /b 1
)

:: Restore user configuration
if exist "%BACKUP_DIR%\.env" (
    copy /Y "%BACKUP_DIR%\.env" "%ROOT_DIR%\.env" >> "%LOG_FILE%" 2>&1
    echo Restored user configuration >> "%LOG_FILE%"
    echo Restored user configuration
)

:: Ensure the version file exists and is consistent
echo Setting up version information... >> "%LOG_FILE%"
echo Setting up version information...
if exist "%TEMP_DIR%\windows\version.txt" (
    :: Copy version to windows directory
    copy /Y "%TEMP_DIR%\windows\version.txt" "%ROOT_DIR%\windows\version.txt" >> "%LOG_FILE%" 2>&1
    :: Also copy to root for backwards compatibility
    copy /Y "%TEMP_DIR%\windows\version.txt" "%ROOT_DIR%\version.txt" >> "%LOG_FILE%" 2>&1
    
    :: Read and display version
    for /f "tokens=*" %%a in ('type "%TEMP_DIR%\windows\version.txt"') do (
        echo Installed version: %%a >> "%LOG_FILE%"
        echo Installed version: %%a
    )
) else (
    echo WARNING: version.txt not found in downloaded files >> "%LOG_FILE%"
    echo WARNING: version.txt not found in downloaded files
    echo Creating default version file with content "1.0.2" >> "%LOG_FILE%"
    echo 1.0.2 > "%ROOT_DIR%\windows\version.txt"
    echo 1.0.2 > "%ROOT_DIR%\version.txt"
    echo Created default version files with version 1.0.2
)

:: Run the Windows installer to update dependencies
echo Installing dependencies... >> "%LOG_FILE%"
echo.
echo Installing dependencies...
if exist "%ROOT_DIR%\windows\Install-Windows.bat" (
    echo Calling Install-Windows.bat... >> "%LOG_FILE%"
    call "%ROOT_DIR%\windows\Install-Windows.bat" /silent >> "%LOG_FILE%" 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo WARNING: Installer returned error code %ERRORLEVEL% >> "%LOG_FILE%"
        echo WARNING: Installer returned error code %ERRORLEVEL%
    ) else (
        echo Dependencies installed successfully >> "%LOG_FILE%"
        echo Dependencies installed successfully
    )
) else (
    echo WARNING: Windows installer not found. Dependency installation may be incomplete. >> "%LOG_FILE%"
    echo WARNING: Windows installer not found. Dependency installation may be incomplete.
    echo Looking in: "%ROOT_DIR%\windows\Install-Windows.bat" >> "%LOG_FILE%"
    dir "%ROOT_DIR%\windows\" >> "%LOG_FILE%"
    timeout /t 3 >nul
)

:: Clean up temporary directory
echo Cleaning up temporary files... >> "%LOG_FILE%"
echo Cleaning up temporary files...
rmdir /S /Q "%TEMP_DIR%" >nul 2>&1

echo. >> "%LOG_FILE%"
echo ======================================================== >> "%LOG_FILE%"
echo    Update completed successfully!                         >> "%LOG_FILE%"
echo    Completed at: %date% %time%                            >> "%LOG_FILE%"
echo ======================================================== >> "%LOG_FILE%"
echo Log file saved to: %LOG_FILE% >> "%LOG_FILE%"

echo.
echo ========================================
echo    Update completed successfully!
echo ========================================
echo.
echo Multi-Max has been updated to the latest version.
echo Log file saved to: %LOG_FILE%
echo.

:: Launch the updated Multi-Max
echo Starting Multi-Max... >> "%LOG_FILE%"
echo Starting Multi-Max...
cd "%ROOT_DIR%"

:: Give the user a confirmation before exiting
echo.
echo Update complete! The application will start in 5 seconds...
echo If it doesn't start automatically, you can find it at:
echo %ROOT_DIR%\windows\Run-MultiMax.bat
echo.

:: Copy log file to the installation directory for reference
copy "%LOG_FILE%" "%ROOT_DIR%\update-log.txt" >nul 2>&1

timeout /t 5 >nul

start "" "%ROOT_DIR%\windows\Run-MultiMax.bat"

exit 