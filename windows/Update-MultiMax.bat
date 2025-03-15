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

:: Set the installation mode - controls whether files go to the base directory or windows subdirectory
:: INSTALL_TO_BASE=1 means install all files to the base multi-max directory
:: INSTALL_TO_BASE=0 means install all files to the windows subdirectory only
set "INSTALL_TO_BASE=0"

echo Installation mode: >> "%LOG_FILE%"
if %INSTALL_TO_BASE% EQU 1 (
    echo All files will be installed to the base Multi-Max directory >> "%LOG_FILE%"
    echo All files will be installed to the base Multi-Max directory
) else (
    echo All files will be installed to the windows subdirectory only >> "%LOG_FILE%"
    echo All files will be installed to the windows subdirectory only
)

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
    echo Backed up .env configuration from root directory >> "%LOG_FILE%"
    echo Backed up .env configuration from root directory
)

if exist "%ROOT_DIR%\windows\.env" (
    copy /Y "%ROOT_DIR%\windows\.env" "%BACKUP_DIR%\windows-.env" >nul
    echo Backed up .env configuration from windows directory >> "%LOG_FILE%"
    echo Backed up .env configuration from windows directory
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
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to initialize git repository. >> "%LOG_FILE%"
    echo ERROR: Failed to initialize git repository.
    echo Press any key to exit...
    pause > nul
    exit /b 1
)

git remote add origin https://github.com/dreamworks2050/multi-max.git >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to add git remote. >> "%LOG_FILE%"
    echo ERROR: Failed to add git remote.
    echo Press any key to exit...
    pause > nul
    exit /b 1
)

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

:: Check if the windows directory was successfully downloaded
if not exist "%TEMP_DIR%\windows" (
    echo ERROR: Windows directory not found in downloaded repository. >> "%LOG_FILE%"
    echo ERROR: Windows directory not found in downloaded repository.
    echo This could indicate a repository structure change.
    echo Press any key to exit...
    pause > nul
    exit /b 1
)

:: Preserve version information before removing files
if exist "%ROOT_DIR%\windows\version.txt" (
    copy /Y "%ROOT_DIR%\windows\version.txt" "%BACKUP_DIR%\version.txt" >nul
    echo Backed up version information from windows directory >> "%LOG_FILE%"
)
if exist "%ROOT_DIR%\version.txt" (
    copy /Y "%ROOT_DIR%\version.txt" "%BACKUP_DIR%\root-version.txt" >nul
    echo Backed up version information from root directory >> "%LOG_FILE%"
)

:: Set target installation directory based on installation mode
if %INSTALL_TO_BASE% EQU 1 (
    set "INSTALL_DIR=%ROOT_DIR%"
) else (
    set "INSTALL_DIR=%ROOT_DIR%\windows"
    :: Create windows directory if it doesn't exist
    if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%" 2>nul
)

echo Target installation directory: %INSTALL_DIR% >> "%LOG_FILE%"

:: Remove existing files based on install mode
echo Removing old files... >> "%LOG_FILE%"
echo.
echo Removing old files...

if %INSTALL_TO_BASE% EQU 1 (
    :: If installing to base, preserve the windows folder but clean it completely
    echo Cleaning the base directory (preserving the windows directory structure)... >> "%LOG_FILE%"
    
    :: Remove all files from root except specific ones to preserve
    for %%F in ("%ROOT_DIR%\*") do (
        if /i not "%%~nxF"=="windows" (
            if /i not "%%~nxF"=="logs" (
                if /i not "%%~nxF"==".git" (
                    if /i not "%%~nxF"=="Update-MultiMax.bat" (
                        del /F /Q "%%F" >nul 2>&1
                        echo Removed file: %%F >> "%LOG_FILE%"
                    )
                )
            )
        )
    )
    
    :: Remove all directories except specific ones to preserve
    for /d %%D in ("%ROOT_DIR%\*") do (
        if /i not "%%~nxD"=="windows" (
            if /i not "%%~nxD"=="logs" (
                if /i not "%%~nxD"==".git" (
                    rmdir /S /Q "%%D" >nul 2>&1
                    echo Removed directory: %%D >> "%LOG_FILE%"
                )
            )
        )
    )
    
    :: Now clean the windows directory completely
    echo Cleaning the windows directory... >> "%LOG_FILE%"
    if exist "%ROOT_DIR%\windows" (
        for %%F in ("%ROOT_DIR%\windows\*") do (
            del /F /Q "%%F" >nul 2>&1
            echo Removed file: %%F >> "%LOG_FILE%"
        )
        
        for /d %%D in ("%ROOT_DIR%\windows\*") do (
            rmdir /S /Q "%%D" >nul 2>&1
            echo Removed directory: %%D >> "%LOG_FILE%"
        )
    )
) else (
    :: If installing to windows subdirectory only, just clean that directory
    echo Cleaning the windows directory only... >> "%LOG_FILE%"
    if exist "%ROOT_DIR%\windows" (
        for %%F in ("%ROOT_DIR%\windows\*") do (
            if /i not "%%~nxF"==".env" (
                del /F /Q "%%F" >nul 2>&1
                echo Removed file: %%F >> "%LOG_FILE%"
            ) else (
                echo Preserved file: %%F >> "%LOG_FILE%"
            )
        )
        
        for /d %%D in ("%ROOT_DIR%\windows\*") do (
            if /i not "%%~nxD"=="__pycache__" (
                rmdir /S /Q "%%D" >nul 2>&1
                echo Removed directory: %%D >> "%LOG_FILE%"
            ) else (
                echo Preserved directory: %%D >> "%LOG_FILE%"
            )
        )
    ) else (
        echo Creating windows directory... >> "%LOG_FILE%"
        mkdir "%ROOT_DIR%\windows" 2>nul
    )
    
    :: Remove any launcher scripts from the root directory
    if exist "%ROOT_DIR%\Run-MultiMax.bat" (
        del /F /Q "%ROOT_DIR%\Run-MultiMax.bat" >nul 2>&1
        echo Removed file: %ROOT_DIR%\Run-MultiMax.bat >> "%LOG_FILE%"
    )
)

:: Copy all files from the temporary clone to the installation directory
echo Installing new version... >> "%LOG_FILE%"
echo.
echo Installing new version...

if %INSTALL_TO_BASE% EQU 1 (
    :: Install all windows files directly to the base directory
    echo Installing all files to the base directory... >> "%LOG_FILE%"
    if exist "%TEMP_DIR%\windows" (
        xcopy /E /I /H /Y "%TEMP_DIR%\windows\*" "%INSTALL_DIR%\" >> "%LOG_FILE%" 2>&1
        if %ERRORLEVEL% NEQ 0 (
            echo ERROR: Failed to copy files to base directory. >> "%LOG_FILE%"
            echo ERROR: Failed to copy files to base directory.
        ) else (
            echo Copied all Windows files to base directory successfully >> "%LOG_FILE%"
        )
    ) else (
        echo ERROR: Windows folder not found in downloaded files. >> "%LOG_FILE%"
    )
) else (
    :: Install all windows files to the windows subdirectory
    echo Installing all files to the windows subdirectory... >> "%LOG_FILE%"
    if exist "%TEMP_DIR%\windows" (
        :: Verify the downloaded files
        dir "%TEMP_DIR%\windows" >> "%LOG_FILE%" 2>&1
        
        :: Copy the files
        xcopy /E /I /H /Y "%TEMP_DIR%\windows\*" "%INSTALL_DIR%\" >> "%LOG_FILE%" 2>&1
        if %ERRORLEVEL% NEQ 0 (
            echo ERROR: Failed to copy files to windows subdirectory. >> "%LOG_FILE%"
            echo ERROR: Failed to copy files to windows subdirectory.
            echo Attempting to diagnose the issue... >> "%LOG_FILE%"
            echo.
            
            :: Detailed error diagnostics
            echo ===== Directory Information ===== >> "%LOG_FILE%"
            echo Checking source directory... >> "%LOG_FILE%"
            dir "%TEMP_DIR%\windows" >> "%LOG_FILE%" 2>&1
            
            echo Checking target directory... >> "%LOG_FILE%"
            dir "%INSTALL_DIR%" >> "%LOG_FILE%" 2>&1
            
            echo ===== Access Rights Check ===== >> "%LOG_FILE%"
            echo Checking if target directory is writable... >> "%LOG_FILE%"
            echo Test > "%INSTALL_DIR%\write_test.tmp" 2>> "%LOG_FILE%"
            if exist "%INSTALL_DIR%\write_test.tmp" (
                echo Target directory is writable >> "%LOG_FILE%"
                del "%INSTALL_DIR%\write_test.tmp" >nul 2>&1
            ) else (
                echo Target directory is NOT writable >> "%LOG_FILE%"
            )
            
            echo Press any key to attempt alternative copy method...
            pause > nul
            
            :: Try a different copy method
            echo Attempting alternative copy method... >> "%LOG_FILE%"
            robocopy "%TEMP_DIR%\windows" "%INSTALL_DIR%" /E /NFL /NDL /NJH /NJS /nc /ns /np >> "%LOG_FILE%" 2>&1
            echo Alternative copy completed with exit code %ERRORLEVEL% >> "%LOG_FILE%"
        ) else (
            echo Copied all Windows files to windows subdirectory successfully >> "%LOG_FILE%"
        )
        
        :: Create launcher in root directory for convenience
        echo Creating launcher in root directory... >> "%LOG_FILE%"
        echo @echo off > "%ROOT_DIR%\Run-MultiMax.bat"
        echo cd /d "%%~dp0\windows" >> "%ROOT_DIR%\Run-MultiMax.bat"
        echo python main.py %%* >> "%ROOT_DIR%\Run-MultiMax.bat"
        echo exit /b %%ERRORLEVEL%% >> "%ROOT_DIR%\Run-MultiMax.bat"
        echo Created launcher in root directory >> "%LOG_FILE%"
    ) else (
        echo ERROR: Windows folder not found in downloaded files. >> "%LOG_FILE%"
        echo ERROR: Windows folder not found in downloaded files.
    )
)

:: Restore user configuration
echo Restoring user configuration... >> "%LOG_FILE%"
if %INSTALL_TO_BASE% EQU 1 (
    if exist "%BACKUP_DIR%\.env" (
        copy /Y "%BACKUP_DIR%\.env" "%INSTALL_DIR%\.env" >> "%LOG_FILE%" 2>&1
        echo Restored user configuration to base directory >> "%LOG_FILE%"
        echo Restored user configuration to base directory
    )
) else (
    if exist "%BACKUP_DIR%\windows-.env" (
        copy /Y "%BACKUP_DIR%\windows-.env" "%INSTALL_DIR%\.env" >> "%LOG_FILE%" 2>&1
        echo Restored user configuration to windows directory >> "%LOG_FILE%"
        echo Restored user configuration to windows directory
    ) else if exist "%BACKUP_DIR%\.env" (
        copy /Y "%BACKUP_DIR%\.env" "%INSTALL_DIR%\.env" >> "%LOG_FILE%" 2>&1
        echo Restored user configuration from root to windows directory >> "%LOG_FILE%"
        echo Restored user configuration from root to windows directory
    )
)

:: Ensure the version file exists in the correct location
echo Setting up version information... >> "%LOG_FILE%"
echo Setting up version information...

if exist "%TEMP_DIR%\windows\version.txt" (
    :: Copy version to the appropriate directory
    copy /Y "%TEMP_DIR%\windows\version.txt" "%INSTALL_DIR%\version.txt" >> "%LOG_FILE%" 2>&1
    echo Installed version file to: %INSTALL_DIR%\version.txt >> "%LOG_FILE%"
    
    :: Read and display version
    for /f "tokens=*" %%a in ('type "%TEMP_DIR%\windows\version.txt"') do (
        echo Installed version: %%a >> "%LOG_FILE%"
        echo Installed version: %%a
    )
) else (
    echo WARNING: version.txt not found in downloaded files >> "%LOG_FILE%"
    echo WARNING: version.txt not found in downloaded files
    echo Creating default version file with content "1.0.1" >> "%LOG_FILE%"
    echo 1.0.1 > "%INSTALL_DIR%\version.txt"
    echo Created default version file with version 1.0.1 in %INSTALL_DIR% >> "%LOG_FILE%"
    echo Created default version file with version 1.0.1
)

:: Verify critical files exist after installation
echo Verifying critical files... >> "%LOG_FILE%"
echo Verifying critical files...

set "MISSING_FILES="
if not exist "%INSTALL_DIR%\main.py" (
    set "MISSING_FILES=!MISSING_FILES! main.py"
    echo CRITICAL: main.py is missing >> "%LOG_FILE%"
    echo CRITICAL: main.py is missing
)
if not exist "%INSTALL_DIR%\update.py" (
    set "MISSING_FILES=!MISSING_FILES! update.py"
    echo CRITICAL: update.py is missing >> "%LOG_FILE%"
    echo CRITICAL: update.py is missing
)
if not exist "%INSTALL_DIR%\version.txt" (
    set "MISSING_FILES=!MISSING_FILES! version.txt"
    echo CRITICAL: version.txt is missing >> "%LOG_FILE%"
    echo CRITICAL: version.txt is missing
)
if not exist "%INSTALL_DIR%\requirements.txt" (
    set "MISSING_FILES=!MISSING_FILES! requirements.txt"
    echo CRITICAL: requirements.txt is missing >> "%LOG_FILE%"
    echo CRITICAL: requirements.txt is missing
)

if not "!MISSING_FILES!"=="" (
    echo WARNING: Some critical files are missing: !MISSING_FILES! >> "%LOG_FILE%"
    echo WARNING: Some critical files are missing: !MISSING_FILES!
    echo Attempting to recover from backup... >> "%LOG_FILE%"
    echo Attempting to recover from backup...
    
    :: Try to recover from our backup
    if exist "%BACKUP_DIR%\main.py" (
        copy /Y "%BACKUP_DIR%\main.py" "%INSTALL_DIR%\main.py" >> "%LOG_FILE%" 2>&1
        echo Recovered main.py from backup >> "%LOG_FILE%"
        echo Recovered main.py from backup
    )
    if exist "%BACKUP_DIR%\update.py" (
        copy /Y "%BACKUP_DIR%\update.py" "%INSTALL_DIR%\update.py" >> "%LOG_FILE%" 2>&1
        echo Recovered update.py from backup >> "%LOG_FILE%"
        echo Recovered update.py from backup
    )
    if exist "%BACKUP_DIR%\version.txt" (
        copy /Y "%BACKUP_DIR%\version.txt" "%INSTALL_DIR%\version.txt" >> "%LOG_FILE%" 2>&1
        echo Recovered version.txt from backup >> "%LOG_FILE%"
        echo Recovered version.txt from backup
    )
    if exist "%BACKUP_DIR%\requirements.txt" (
        copy /Y "%BACKUP_DIR%\requirements.txt" "%INSTALL_DIR%\requirements.txt" >> "%LOG_FILE%" 2>&1
        echo Recovered requirements.txt from backup >> "%LOG_FILE%"
        echo Recovered requirements.txt from backup
    )
) else (
    echo All critical files verified >> "%LOG_FILE%"
    echo All critical files verified
)

:: Run the Windows installer to update dependencies
echo Installing dependencies... >> "%LOG_FILE%"
echo.
echo Installing dependencies...

if exist "%INSTALL_DIR%\Install-Windows.bat" (
    echo Calling Install-Windows.bat from installation directory... >> "%LOG_FILE%"
    echo Calling Install-Windows.bat... This may take a few minutes...
    cd "%INSTALL_DIR%"
    call "%INSTALL_DIR%\Install-Windows.bat" /silent >> "%LOG_FILE%" 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo WARNING: Install-Windows.bat returned error code %ERRORLEVEL% >> "%LOG_FILE%"
        echo WARNING: There may have been issues installing dependencies.
        echo Please check the log file for details: %LOG_FILE%
    ) else (
        echo Dependencies installed successfully >> "%LOG_FILE%"
        echo Dependencies installed successfully
    )
) else (
    echo WARNING: Install-Windows.bat not found in installation directory. >> "%LOG_FILE%"
    echo WARNING: Install-Windows.bat not found. Dependencies may not be updated.
    echo Attempting to install requirements directly... >> "%LOG_FILE%"
    
    if exist "%INSTALL_DIR%\requirements.txt" (
        echo Installing Python packages from requirements.txt... >> "%LOG_FILE%"
        python -m pip install -r "%INSTALL_DIR%\requirements.txt" >> "%LOG_FILE%" 2>&1
        echo Python packages updated >> "%LOG_FILE%"
        echo Python packages updated
    ) else (
        echo WARNING: requirements.txt not found. Cannot update Python packages. >> "%LOG_FILE%"
        echo WARNING: requirements.txt not found. Cannot update Python packages.
    )
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

:: Copy log file to the installation directory for reference
copy "%LOG_FILE%" "%INSTALL_DIR%\update-log.txt" >nul 2>&1

:: Launch the updated Multi-Max
echo Starting Multi-Max... >> "%LOG_FILE%"
echo Starting Multi-Max...

:: Set the run path based on installation mode
if %INSTALL_TO_BASE% EQU 1 (
    set "RUN_PATH=%ROOT_DIR%\main.py"
) else (
    set "RUN_PATH=%ROOT_DIR%\windows\main.py"
)

:: Give the user a confirmation before exiting
echo.
echo Update complete! The application will start in 5 seconds...
echo If it doesn't start automatically, you can run it from:
if %INSTALL_TO_BASE% EQU 1 (
    echo %ROOT_DIR%\main.py
) else (
    echo %ROOT_DIR%\windows\main.py
    echo Or use the launcher: %ROOT_DIR%\Run-MultiMax.bat
)
echo.

timeout /t 5 >nul

:: Start the program based on installation mode
if %INSTALL_TO_BASE% EQU 1 (
    cd "%ROOT_DIR%"
    start "" python main.py
) else (
    cd "%ROOT_DIR%\windows"
    start "" python main.py
)

exit 