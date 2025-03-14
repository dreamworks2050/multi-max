@echo off
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

echo Checking for PowerShell...
where powershell >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: PowerShell is required but not found.
    echo Please make sure PowerShell is installed on your system.
    pause
    exit /b 1
)

echo Starting Windows-specific installation...
echo This will install the Windows-compatible version of Multi-Max.
echo.

:: Run the PowerShell installer with bypass execution policy
powershell -ExecutionPolicy Bypass -File "%~dp0..\install-multi-max.ps1"

:: Check if the installation was successful
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Installation completed successfully!
    echo ========================================
    echo.
    echo The Windows-compatible version of Multi-Max has been installed.
    echo To run Multi-Max, navigate to the installation directory and run:
    echo python main.py
    echo.
    echo Or use the desktop shortcut if it was created.
    echo.
) else (
    echo.
    echo ========================================
    echo Installation encountered an error!
    echo ========================================
    echo.
    echo Please check the logs above for details on what went wrong.
    echo If you see errors about missing "Quartz" modules, it means
    echo the Windows-specific version was not properly applied.
    echo.
)

pause 