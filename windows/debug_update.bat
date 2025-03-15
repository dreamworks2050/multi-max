@echo off
setlocal EnableDelayedExpansion

echo ========================================
echo    Multi-Max Update Checker Diagnostics
echo ========================================
echo.

:: Get script directory for reliable path handling
pushd "%~dp0"
set "SCRIPT_DIR=%CD%"
set "PARENT_DIR=%CD%\.."

echo Running diagnostic tests for the update checker...
echo.

echo === Python Version ===
python --version
echo.

echo === Git Version ===
git --version 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Git is not installed or not in PATH
    echo The update checker requires Git to function.
) else (
    echo Git is available
)
echo.

echo === Testing Python Import Path ===
python -c "import sys; print('sys.path first entries:'); [print(f'  - {p}') for p in sys.path[:5]]"
echo.

echo === Checking if update_checker.py exists ===
if exist "%SCRIPT_DIR%\update_checker.py" (
    echo update_checker.py found in %SCRIPT_DIR%
) else (
    echo ERROR: update_checker.py not found in %SCRIPT_DIR%
)
echo.

echo === Checking VERSION file ===
if exist "%PARENT_DIR%\VERSION" (
    echo VERSION file found: 
    type "%PARENT_DIR%\VERSION"
) else (
    echo WARNING: VERSION file not found at %PARENT_DIR%\VERSION
)
echo.

echo === Testing update_checker import ===
python -c "try: from update_checker import VERSION; print(f'Successfully imported update_checker. Version: {VERSION}'); exit(0); except ImportError as e: print(f'Error importing update_checker: {e}'); exit(1)"
if %ERRORLEVEL% NEQ 0 (
    echo Trying alternate import method...
    python -c "import sys; sys.path.insert(0, r'%SCRIPT_DIR%'); try: from update_checker import VERSION; print(f'Successfully imported update_checker with path adjustment. Version: {VERSION}'); except ImportError as e: print(f'Still cannot import update_checker: {e}')"
)
echo.

echo === Testing Git repository detection ===
cd "%PARENT_DIR%"
python -c "import subprocess, os; print(f'Current directory: {os.getcwd()}'); try: result = subprocess.run(['git', 'rev-parse', '--is-inside-work-tree'], capture_output=True, text=True); print(f'Git repo check result: {result.stdout.strip()}'); print(f'Return code: {result.returncode}'); exit(0 if result.returncode == 0 and result.stdout.strip() == 'true' else 1); except Exception as e: print(f'Error checking git repo: {e}'); exit(1)"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: This directory does not appear to be a Git repository.
    echo The update checker will not work without a valid Git repository.
) else (
    echo This directory is a valid Git repository.
)
echo.

echo === Running update_checker self-check ===
cd "%PARENT_DIR%"
set PYTHONPATH=%SCRIPT_DIR%;%PYTHONPATH%
python "%SCRIPT_DIR%\update_checker.py" --self-check
echo.

echo === Running update check in debug mode ===
python "%SCRIPT_DIR%\update_checker.py" --debug
echo.

echo Diagnostic information complete. Please check logs directory for detailed logs.
echo If you're still having issues, please send the output of this script to support.
echo.

cd "%SCRIPT_DIR%"
pause
endlocal 