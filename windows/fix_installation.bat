@echo off
setlocal EnableDelayedExpansion

echo ========================================
echo    Multi-Max Installation Repair Tool
echo ========================================
echo.

:: Get script directory for reliable path handling
pushd "%~dp0"
set "SCRIPT_DIR=%CD%"
set "PARENT_DIR=%CD%\.."

echo CURRENT PATHS:
echo Script directory: %SCRIPT_DIR%
echo Parent directory: %PARENT_DIR%
echo.

echo === Step 1: Checking installation structure ===
if not exist "%PARENT_DIR%\main.py" (
    echo WARNING: main.py not found in parent directory
    echo Checking if Windows version is properly installed...
    
    :: Copy windows/main.py to the parent directory if needed
    if exist "%SCRIPT_DIR%\main.py" (
        echo Copying Windows-specific main.py to parent directory...
        copy "%SCRIPT_DIR%\main.py" "%PARENT_DIR%\main.py" /Y
        echo Windows main.py installed.
    ) else (
        echo ERROR: Windows main.py not found!
        echo Cannot proceed with repair.
        goto :error
    )
) else (
    echo main.py found in parent directory.
)

:: Create logs directory if it doesn't exist
if not exist "%PARENT_DIR%\logs" (
    echo Creating logs directory...
    mkdir "%PARENT_DIR%\logs"
    echo Logs directory created.
) else (
    echo Logs directory exists.
)

:: Create VERSION file if it doesn't exist
if not exist "%PARENT_DIR%\VERSION" (
    echo Creating VERSION file...
    echo 1.0.0> "%PARENT_DIR%\VERSION"
    echo VERSION file created.
) else (
    echo VERSION file exists.
)

echo.
echo === Step 2: Checking Python environment ===
echo.

:: Check Python version
python --version
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found in PATH!
    echo Make sure Python is installed and in your PATH.
    goto :error
)

:: Check for virtual environment
set "VENV_FOUND=0"
set "VENV_PATH="

if exist "%PARENT_DIR%\multi-max\Scripts\activate.bat" (
    set "VENV_FOUND=1"
    set "VENV_PATH=%PARENT_DIR%\multi-max"
    echo Found virtual environment: multi-max
) else if exist "%PARENT_DIR%\.venv\Scripts\activate.bat" (
    set "VENV_FOUND=1"
    set "VENV_PATH=%PARENT_DIR%\.venv"
    echo Found virtual environment: .venv
) else if exist "%PARENT_DIR%\venv\Scripts\activate.bat" (
    set "VENV_FOUND=1"
    set "VENV_PATH=%PARENT_DIR%\venv"
    echo Found virtual environment: venv
) else (
    echo WARNING: No virtual environment found.
    echo We'll create a new one.
    
    cd "%PARENT_DIR%"
    echo Creating new virtual environment...
    python -m venv venv
    
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to create virtual environment.
        goto :error
    ) else (
        set "VENV_FOUND=1"
        set "VENV_PATH=%PARENT_DIR%\venv"
        echo Created new virtual environment: venv
    )
)

echo.
echo === Step 3: Installing required packages ===
echo.

:: Install packages in the virtual environment
if "%VENV_FOUND%"=="1" (
    echo Activating virtual environment...
    call "%VENV_PATH%\Scripts\activate.bat"
    
    echo Installing required packages...
    pip install numpy pygame opencv-python python-dotenv psutil yt-dlp
    
    if %ERRORLEVEL% NEQ 0 (
        echo WARNING: Some packages may not have installed correctly.
    ) else (
        echo Packages installed successfully.
    )
    
    deactivate
) else (
    echo Installing packages globally (not recommended)...
    pip install numpy pygame opencv-python python-dotenv psutil yt-dlp
)

echo.
echo === Step 4: Fixing update checker ===
echo.

:: Check if update_checker.py exists
if not exist "%SCRIPT_DIR%\update_checker.py" (
    echo ERROR: update_checker.py not found!
    echo Cannot proceed with update checker repair.
) else (
    echo update_checker.py found.
    
    :: Create a standalone tester for the update checker
    echo Creating update checker test script...
    
    (
        echo import sys
        echo import os
        echo print("Python Executable:", sys.executable^)
        echo print("Current Directory:", os.getcwd(^)^)
        echo script_dir = '%SCRIPT_DIR:\=\\%'
        echo print("Script Directory:", script_dir^)
        echo 
        echo # Add windows directory to path
        echo if script_dir not in sys.path:
        echo     sys.path.insert(0, script_dir^)
        echo     print(f"Added {script_dir} to sys.path"^)
        echo 
        echo print("\nAttempting to import update_checker...")
        echo try:
        echo     import update_checker
        echo     print("Success! Found update_checker module at:", update_checker.__file__^)
        echo     print("Version:", update_checker.VERSION^)
        echo except ImportError as e:
        echo     print("Failed to import update_checker:", e^)
        echo 
        echo print("\nPython Path:")
        echo for p in sys.path:
        echo     print("-", p^)
    ) > "%PARENT_DIR%\test_update_checker.py"
    
    echo Test script created.
    echo Running test script...
    
    cd "%PARENT_DIR%"
    python test_update_checker.py
)

echo.
echo === Step 5: Testing main script path resolution ===
echo.

:: Create a test script for main.py path resolution
echo Creating main.py path test script...

(
    echo import os
    echo import sys
    echo print("Python Executable:", sys.executable^)
    echo print("Current Directory:", os.getcwd(^)^)
    echo print("__file__:", __file__^)
    echo script_path = os.path.abspath(__file__^)
    echo print("Script Absolute Path:", script_path^)
    echo script_dir = os.path.dirname(script_path^)
    echo print("Script Directory:", script_dir^)
    echo parent_dir = os.path.dirname(script_dir^)
    echo print("Parent Directory:", parent_dir^)
    echo 
    echo # Check if main.py exists
    echo main_script = os.path.join(parent_dir, 'main.py'^)
    echo print("Looking for main.py at:", main_script^)
    echo print("main.py exists:", os.path.exists(main_script^)^)
    echo 
    echo # Try to resolve paths from different starting points
    echo print("\nTrying different path resolution methods:")
    echo 
    echo # Method 1: From script dir
    echo windows_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__^)^)^), 'windows'^)
    echo parent_dir1 = os.path.dirname(windows_dir^)
    echo print("Method 1 - Windows Dir:", windows_dir^)
    echo print("Method 1 - Parent Dir:", parent_dir1^)
    echo print("Method 1 - main.py path:", os.path.join(parent_dir1, 'main.py'^)^)
    echo print("Method 1 - path exists:", os.path.exists(os.path.join(parent_dir1, 'main.py'^)^)^)
) > "%SCRIPT_DIR%\test_paths.py"

echo Path test script created.
echo Running path test script...

cd "%SCRIPT_DIR%"
python test_paths.py

echo.
echo === Step 6: Applying fixes ===
echo.

:: Copy update_checker.py to the parent directory in case of import issues
if exist "%SCRIPT_DIR%\update_checker.py" (
    echo Copying update_checker.py to the parent directory for easier imports...
    copy "%SCRIPT_DIR%\update_checker.py" "%PARENT_DIR%\update_checker.py" /Y
    echo update_checker.py copied to parent directory.
)

:: Ensure the VERSION file has the correct content
echo Writing VERSION file...
echo 1.0.0> "%PARENT_DIR%\VERSION"

:: Create a Run-MultiMax-Fixed.bat file with corrected paths
echo Creating fixed Run-MultiMax.bat with correct paths...

(
    echo @echo off
    echo setlocal EnableDelayedExpansion
    echo.
    echo :: Store original directory to return to later
    echo set "ORIGINAL_DIR=%%CD%%"
    echo.
    echo echo ========================================
    echo echo    Multi-Max Windows Launcher (Fixed)
    echo echo ========================================
    echo echo.
    echo.
    echo :: Get script directory for reliable path handling
    echo pushd "%%~dp0"
    echo set "SCRIPT_DIR=%%CD%%"
    echo set "PARENT_DIR=%%CD%%\.."
    echo.
    echo :: Add both script directory and parent directory to PYTHONPATH 
    echo :: This ensures update_checker and main.py can be found
    echo set "PYTHONPATH=%%SCRIPT_DIR%%;%%PARENT_DIR%%;%%PYTHONPATH%%"
    echo.
    echo :: Change to the parent directory for reliable execution
    echo cd "%%PARENT_DIR%%"
    echo.
    echo :: Check if main.py exists
    echo if not exist "%%PARENT_DIR%%\main.py" (
    echo     echo ERROR: main.py not found at %%PARENT_DIR%%\main.py
    echo     echo Cannot start Multi-Max.
    echo     pause
    echo     exit /b 1
    echo ^)
    echo.
    echo :: Activate virtual environment if found
    echo if exist "%%PARENT_DIR%%\venv\Scripts\activate.bat" (
    echo     echo Activating virtual environment...
    echo     call "%%PARENT_DIR%%\venv\Scripts\activate.bat"
    echo ^) else if exist "%%PARENT_DIR%%\.venv\Scripts\activate.bat" (
    echo     echo Activating virtual environment...
    echo     call "%%PARENT_DIR%%\.venv\Scripts\activate.bat"
    echo ^) else if exist "%%PARENT_DIR%%\multi-max\Scripts\activate.bat" (
    echo     echo Activating virtual environment...
    echo     call "%%PARENT_DIR%%\multi-max\Scripts\activate.bat"
    echo ^) else (
    echo     echo WARNING: No virtual environment found, using system Python.
    echo ^)
    echo.
    echo :: Run Multi-Max using reliable paths
    echo echo Starting Multi-Max...
    echo python "%%PARENT_DIR%%\main.py" %%*
    echo.
    echo :: Deactivate virtual environment if it was activated
    echo if defined VIRTUAL_ENV (
    echo     deactivate
    echo ^)
    echo.
    echo :: Return to original directory
    echo cd "%%ORIGINAL_DIR%%"
    echo echo.
    echo echo Multi-Max has exited.
    echo pause
    echo.
    echo endlocal
) > "%SCRIPT_DIR%\Run-MultiMax-Fixed.bat"

echo Fixed launcher created: Run-MultiMax-Fixed.bat

echo.
echo ========================================
echo    Repair process completed
echo ========================================
echo.
echo To run Multi-Max with these fixes:
echo.
echo 1. Use the Run-MultiMax-Fixed.bat script
echo    This script uses corrected paths and ensures 
echo    the update checker can be found.
echo.
echo 2. If you still encounter issues:
echo    - Make sure Git is installed if you want the update checker to work
echo    - Check the logs directory for error details
echo.
echo ========================================
echo.

cd "%SCRIPT_DIR%"
pause
exit /b 0

:error
echo.
echo Repair process encountered errors.
echo Please fix the issues mentioned above and try again.
echo.
pause
exit /b 1 