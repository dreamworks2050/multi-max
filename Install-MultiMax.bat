@echo off
echo ======================================
echo   Multi-Max Windows Installer Launcher
echo ======================================
echo This script will download and run the Multi-Max installer with administrator privileges.
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause > nul

:: Run PowerShell with admin rights to download and execute the installer
powershell -ExecutionPolicy Bypass -Command "Start-Process PowerShell -ArgumentList '-ExecutionPolicy Bypass -Command \"Invoke-WebRequest -Uri ''https://raw.githubusercontent.com/multi-max/multi-max/main/install-multi-max.ps1'' -OutFile $env:TEMP\install-multi-max.ps1; & $env:TEMP\install-multi-max.ps1\"' -Verb RunAs"

:: End of script 