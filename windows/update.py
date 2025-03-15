"""
Multi-Max Update Helper for Windows
===================================

This module helps with the complete reinstallation of Multi-Max when an update is required.
"""

import os
import sys
import subprocess
import logging
import time
import platform
import tempfile
import datetime
import ctypes
import shutil

# Import the common update script template from main.py if possible
try:
    from main import UPDATE_SCRIPT_TEMPLATE
except ImportError:
    # If we can't import it, use a basic template that will at least check for the real script
    UPDATE_SCRIPT_TEMPLATE = """@echo off
echo Multi-Max Update - Emergency Update Script
echo This is a simplified update script that will attempt to download the full updater.
echo.

set "TEMP_DIR=%TEMP%\\multi-max-emergency-update"
mkdir "%TEMP_DIR%" 2>nul
cd "%TEMP_DIR%"

echo Checking if Git is available...
where git >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Git is required but not found.
    echo Please install Git and try again.
    echo You can download Git from: https://git-scm.com/download/win
    echo.
    pause
    exit /b 1
)

echo Downloading latest version of the updater from GitHub...
git init
git remote add origin https://github.com/dreamworks2050/multi-max.git
git config core.sparseCheckout true
echo windows/Update-MultiMax.bat > .git/info/sparse-checkout

git pull --depth=1 origin main

if exist "windows\\Update-MultiMax.bat" (
    echo Found the full updater script!
    echo Running the proper updater now...
    call "windows\\Update-MultiMax.bat"
    exit /b
) else (
    echo Failed to download the proper updater script.
    echo Please download Multi-Max manually from GitHub.
    echo.
    pause
    exit /b 1
)
"""

def is_admin():
    """Check if the current process has administrator privileges on Windows."""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception as e:
        logging.warning(f"Failed to check admin privileges: {e}")
        return False

def elevate_privileges(script_path):
    """Attempt to restart the current script with administrator privileges."""
    try:
        # Create a visible indicator
        marker_path = os.path.join(tempfile.gettempdir(), "multimax_elevation_request.txt")
        with open(marker_path, 'w') as f:
            f.write(f"Elevation requested at: {datetime.datetime.now()}\n")
            f.write(f"For script: {script_path}\n")
        
        # Use ctypes to trigger UAC
        ctypes.windll.shell32.ShellExecuteW(
            None, "runas", sys.executable, script_path, None, 1
        )
        return True
    except Exception as e:
        logging.error(f"Failed to elevate privileges: {e}")
        return False

def get_script_paths():
    """
    Get all possible paths where the updater script might be located.
    Returns a list of possible paths in order of preference.
    """
    try:
        # Get the directory where this script is located (windows directory)
        windows_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(windows_dir)
        
        # All possible paths where Update-MultiMax.bat might be
        script_paths = [
            os.path.join(windows_dir, "Update-MultiMax.bat"),  # windows/Update-MultiMax.bat
            os.path.join(parent_dir, "Update-MultiMax.bat"),   # multi-max/Update-MultiMax.bat
            os.path.join(parent_dir, "windows", "Update-MultiMax.bat"),  # multi-max/windows/Update-MultiMax.bat
            os.path.join(os.getcwd(), "Update-MultiMax.bat"),  # Current directory/Update-MultiMax.bat
            os.path.join(os.getcwd(), "windows", "Update-MultiMax.bat"),  # Current directory/windows/Update-MultiMax.bat
        ]
        
        logging.debug(f"Possible update script paths: {script_paths}")
        return script_paths
    except Exception as e:
        logging.error(f"Error determining script paths: {e}")
        # Provide some fallback paths
        return [
            "Update-MultiMax.bat",
            "windows/Update-MultiMax.bat",
            os.path.join(tempfile.gettempdir(), "Update-MultiMax.bat")
        ]

def ensure_update_script_exists():
    """
    Ensures that the Update-MultiMax.bat file exists in the windows directory.
    Creates it if it doesn't exist.
    
    Returns:
        str: Path to the existing/created script
    """
    windows_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(windows_dir)
    
    # Define paths
    windows_script_path = os.path.join(windows_dir, "Update-MultiMax.bat")
    parent_script_path = os.path.join(parent_dir, "Update-MultiMax.bat")
    
    # Check if the script exists in any location
    windows_script_exists = os.path.exists(windows_script_path)
    parent_script_exists = os.path.exists(parent_script_path)
    
    # If it exists in parent directory but not in windows directory, copy it
    if parent_script_exists and not windows_script_exists:
        try:
            logging.info(f"Copying Update-MultiMax.bat from parent directory to windows directory")
            shutil.copy2(parent_script_path, windows_script_path)
            logging.info(f"Copied update script to: {windows_script_path}")
            return windows_script_path
        except Exception as e:
            logging.error(f"Failed to copy parent update script to windows directory: {e}")
            # Return the parent script path as fallback
            return parent_script_path
    
    # Create the script if it doesn't exist in either location
    if not windows_script_exists and not parent_script_exists:
        try:
            logging.info(f"Creating Update-MultiMax.bat in windows directory: {windows_script_path}")
            with open(windows_script_path, 'w') as f:
                f.write(UPDATE_SCRIPT_TEMPLATE)
            logging.info(f"Created update script at: {windows_script_path}")
            
            # Also copy it to parent directory for compatibility
            try:
                shutil.copy2(windows_script_path, parent_script_path)
                logging.info(f"Also copied update script to parent directory: {parent_script_path}")
            except Exception as copy_err:
                logging.warning(f"Could not copy update script to parent directory: {copy_err}")
        except Exception as e:
            logging.error(f"Failed to create windows update script: {e}")
            
            # Try creating in parent directory as fallback
            try:
                logging.info(f"Trying to create update script in parent directory instead: {parent_script_path}")
                with open(parent_script_path, 'w') as f:
                    f.write(UPDATE_SCRIPT_TEMPLATE)
                logging.info(f"Created update script in parent directory: {parent_script_path}")
                return parent_script_path
            except Exception as parent_err:
                logging.error(f"Failed to create update script in parent directory: {parent_err}")
                return None
    
    # Return the windows script path if it exists, otherwise the parent script path
    if windows_script_exists:
        logging.info(f"Using existing update script at: {windows_script_path}")
        return windows_script_path
    elif parent_script_exists:
        logging.info(f"Using existing update script at: {parent_script_path}")
        return parent_script_path
    else:
        logging.error("Failed to find or create update script")
        return None

def find_existing_update_script():
    """
    Find an existing Update-MultiMax.bat script in the possible locations.
    
    Returns:
        str or None: Path to the existing script, or None if not found
    """
    script_paths = get_script_paths()
    
    # Check each possible location
    for path in script_paths:
        if os.path.exists(path):
            logging.info(f"Found existing update script at: {path}")
            return path
    
    logging.warning("Could not find existing update script in any of the expected locations")
    return None

def create_update_launcher(update_script_path, pid=None):
    """
    Creates a batch file that waits for a process to exit (if PID is provided)
    and then launches the update script.
    
    Args:
        update_script_path (str): Path to the update script
        pid (int, optional): Process ID to wait for
        
    Returns:
        str: Path to the created launcher batch file
    """
    try:
        # Create a temporary batch file
        temp_dir = os.environ.get('TEMP', os.path.expanduser('~'))
        batch_file = os.path.join(temp_dir, "multimax_update_launcher.bat")
        
        # Make sure the update_script_path exists before proceeding
        if not os.path.exists(update_script_path):
            logging.error(f"Update script does not exist at: {update_script_path}")
            
            # Try to recreate it in the same location
            try:
                windows_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(windows_dir)
                
                # If the path is in the windows directory
                if os.path.dirname(update_script_path) == windows_dir:
                    logging.info(f"Attempting to recreate update script in windows directory")
                    with open(update_script_path, 'w') as f:
                        f.write(UPDATE_SCRIPT_TEMPLATE)
                    logging.info(f"Recreated update script at: {update_script_path}")
                # If the path is in the parent directory
                elif os.path.dirname(update_script_path) == parent_dir:
                    logging.info(f"Attempting to recreate update script in parent directory")
                    with open(update_script_path, 'w') as f:
                        f.write(UPDATE_SCRIPT_TEMPLATE)
                    logging.info(f"Recreated update script at: {update_script_path}")
                else:
                    logging.warning(f"Cannot recreate update script at unknown location: {update_script_path}")
                    # Create a copy in temp directory as fallback
                    fallback_path = os.path.join(temp_dir, "Update-MultiMax.bat")
                    with open(fallback_path, 'w') as f:
                        f.write(UPDATE_SCRIPT_TEMPLATE)
                    logging.info(f"Created fallback update script at: {fallback_path}")
                    update_script_path = fallback_path
            except Exception as e:
                logging.error(f"Failed to recreate update script: {e}")
                # Create a copy in temp directory as fallback
                try:
                    fallback_path = os.path.join(temp_dir, "Update-MultiMax.bat")
                    with open(fallback_path, 'w') as f:
                        f.write(UPDATE_SCRIPT_TEMPLATE)
                    logging.info(f"Created fallback update script at: {fallback_path}")
                    update_script_path = fallback_path
                except Exception as fallback_err:
                    logging.error(f"Failed to create fallback update script: {fallback_err}")
                    return None
        
        # Make sure the update script has its executable bit set (not really needed on Windows, but good practice)
        try:
            os.chmod(update_script_path, 0o755)
            logging.debug(f"Set executable permission on update script")
        except Exception as chmod_err:
            logging.warning(f"Could not set executable permission on update script: {chmod_err}")
        
        # Create launcher script content
        with open(batch_file, 'w') as f:
            f.write('@echo off\n')
            f.write('title Multi-Max Update Launcher\n')
            f.write('echo ==============================================\n')
            f.write('echo    Multi-Max Update Launcher\n')
            f.write('echo ==============================================\n')
            f.write('echo This window will handle the update process.\n')
            f.write('echo Please do not close this window.\n')
            
            # If PID is provided, wait for the process to exit
            if pid:
                f.write('echo.\n')
                f.write(f'echo Waiting for process with PID {pid} to exit...\n')
                f.write('timeout /t 3 > nul\n')
            
            # Store the update script in a variable to make sure it's properly escaped
            f.write(f'set "UPDATE_SCRIPT={update_script_path}"\n')
            f.write('echo.\n')
            f.write('echo Starting update from: %UPDATE_SCRIPT%\n')
            f.write('echo.\n')
            
            # Check if the script exists before trying to run it
            f.write('if not exist "%UPDATE_SCRIPT%" (\n')
            f.write('    echo ERROR: Update script not found at: %UPDATE_SCRIPT%\n')
            f.write('    echo Creating emergency update script...\n')
            
            # Create an emergency script right in the current directory
            emergency_script = "Emergency-Update-MultiMax.bat"
            f.write(f'    echo @echo off > "{emergency_script}"\n')
            f.write(f'    echo echo Multi-Max Emergency Update >> "{emergency_script}"\n')
            f.write(f'    echo echo This is an emergency script to download the latest version. >> "{emergency_script}"\n')
            f.write(f'    echo echo. >> "{emergency_script}"\n')
            f.write(f'    echo echo Creating temporary directory... >> "{emergency_script}"\n')
            f.write(f'    echo set "TEMP_DIR=%%TEMP%%\\multi-max-update" >> "{emergency_script}"\n')
            f.write(f'    echo mkdir "%%TEMP_DIR%%" 2^>nul >> "{emergency_script}"\n')
            f.write(f'    echo cd /d "%%TEMP_DIR%%" >> "{emergency_script}"\n')
            f.write(f'    echo echo. >> "{emergency_script}"\n')
            f.write(f'    echo echo Downloading latest version... >> "{emergency_script}"\n')
            f.write(f'    echo powershell -Command "Invoke-WebRequest -Uri \'https://github.com/dreamworks2050/multi-max/raw/main/windows/Update-MultiMax.bat\' -OutFile \'Update-MultiMax.bat\'" >> "{emergency_script}"\n')
            f.write(f'    echo if exist "Update-MultiMax.bat" ( >> "{emergency_script}"\n')
            f.write(f'    echo     echo Found the updater! Running it now... >> "{emergency_script}"\n')
            f.write(f'    echo     call "Update-MultiMax.bat" >> "{emergency_script}"\n')
            f.write(f'    echo     exit /b >> "{emergency_script}"\n')
            f.write(f'    echo ) else ( >> "{emergency_script}"\n')
            f.write(f'    echo     echo Could not download updater. >> "{emergency_script}"\n')
            f.write(f'    echo     echo Please download Multi-Max manually from GitHub. >> "{emergency_script}"\n')
            f.write(f'    echo     echo. >> "{emergency_script}"\n')
            f.write(f'    echo     pause >> "{emergency_script}"\n')
            f.write(f'    echo     exit /b 1 >> "{emergency_script}"\n')
            f.write(f'    echo ) >> "{emergency_script}"\n')
            
            f.write(f'    set "UPDATE_SCRIPT={emergency_script}"\n')
            f.write(')\n\n')
            
            # Try multiple methods to launch the update script with admin privileges
            f.write('echo Attempting to run update with administrative privileges...\n')
            f.write('powershell -Command "Start-Process -FilePath \'%UPDATE_SCRIPT%\' -Verb RunAs"\n')
            f.write('if %ERRORLEVEL% NEQ 0 (\n')
            f.write('    echo Failed to launch with PowerShell. Trying alternative method...\n')
            
            # Try direct call to the batch file
            f.write('    call "%UPDATE_SCRIPT%"\n')
            f.write('    if %ERRORLEVEL% NEQ 0 (\n')
            f.write('        echo The system cannot find the file %UPDATE_SCRIPT%.\n')
            
            # Last resort - try to open it with explorer
            f.write('        echo Trying to open it with explorer...\n')
            f.write('        explorer "%UPDATE_SCRIPT%"\n')
            f.write('    )\n')
            f.write(')\n')
            
            f.write('echo.\n')
            f.write('echo If a User Account Control prompt appears, please click "Yes".\n')
            f.write('echo.\n')
            f.write('echo This window will close in 10 seconds.\n')
            f.write('timeout /t 10\n')
            
            # As a last resort, provide instructions for manual update
            f.write('echo.\n')
            f.write('echo Waiting for  3 seconds, press a key to continue ...\n')
            f.write('timeout /t 3\n')
        
        logging.info(f"Created update launcher batch file at {batch_file}")
        return batch_file
    
    except Exception as e:
        logging.error(f"Failed to create update launcher: {e}")
        return None

def validate_update_script(script_path):
    """
    Validates that the update script exists and appears to be valid.
    
    Args:
        script_path (str): Path to the update script
        
    Returns:
        bool: True if script exists and appears valid, False otherwise
    """
    if not os.path.exists(script_path):
        logging.error(f"Update script does not exist at: {script_path}")
        return False
    
    try:
        # Check for minimum file size (should be at least a few KB)
        file_size = os.path.getsize(script_path)
        if file_size < 1000:  # Less than 1KB is suspicious
            logging.warning(f"Update script is suspiciously small ({file_size} bytes): {script_path}")
            
            # Check content for basic expected strings
            with open(script_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if '@echo off' not in content or 'setlocal' not in content:
                    logging.error(f"Update script appears to be invalid (missing basic batch commands)")
                    return False
                
                # Check for key functionality markers
                markers = [
                    "Multi-Max Windows Updater",
                    "Check for Git",
                    "Download",
                    "GitHub",
                    "Install"
                ]
                
                missing_markers = [m for m in markers if m not in content]
                if missing_markers:
                    logging.error(f"Update script is missing expected content: {', '.join(missing_markers)}")
                    return False
        
        logging.info(f"Update script validated successfully: {script_path}")
        return True
    except Exception as e:
        logging.error(f"Error validating update script: {e}")
        return False

def start_update_process():
    """
    Start the complete update process by launching the Update-MultiMax.bat script.
    The script will:
    1. Wipe the entire folder (or just windows folder based on configuration)
    2. Redownload the GitHub repo (windows folder only)
    3. Reinstall everything from scratch
    4. Launch the updated version
    
    Returns:
        bool: True if update process was started, False otherwise
    """
    logging.info("Initiating complete update process...")
    
    # Create a detailed log file
    log_file = os.path.join(
        tempfile.gettempdir(), 
        f"multimax-update-launcher-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    )
    
    try:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
        
        logging.info(f"Created update log file: {log_file}")
        logging.info(f"Python version: {sys.version}")
        logging.info(f"Platform: {platform.platform()}")
        logging.info(f"Running as administrator: {is_admin()}")
    except Exception as e:
        logging.warning(f"Failed to set up logging to file: {e}")
    
    # Ensure we're on Windows
    if platform.system() != 'Windows':
        logging.error("Update process is only supported on Windows")
        return False
    
    # Find or create the update script
    try:
        # First look for existing update script
        update_script = find_existing_update_script()
        
        # If not found, create one in the windows directory
        if not update_script:
            logging.info("No existing update script found, creating one")
            update_script = ensure_update_script_exists()
            
            # Copy to current directory as a fallback
            try:
                current_dir_script = os.path.join(os.getcwd(), "Update-MultiMax.bat")
                shutil.copy2(update_script, current_dir_script)
                logging.info(f"Also copied update script to current directory: {current_dir_script}")
            except Exception as copy_err:
                logging.warning(f"Could not copy update script to current directory: {copy_err}")
        
        if not update_script or not os.path.exists(update_script):
            logging.error("Failed to find or create update script")
            return False
            
        # Validate the update script
        if not validate_update_script(update_script):
            logging.error("Update script validation failed, attempting to recreate")
            # Try to recreate the script
            update_script = ensure_update_script_exists()
            if not validate_update_script(update_script):
                logging.error("Failed to create a valid update script")
                return False
            
        logging.info(f"Using update script at: {update_script}")
        
        # Create a marker file to indicate an update is in progress
        windows_dir = os.path.dirname(os.path.abspath(__file__))
        marker_file = os.path.join(windows_dir, "update_in_progress.txt")
        try:
            with open(marker_file, 'w') as f:
                f.write(f"Update started at: {time.ctime()}\n")
                f.write(f"Update script: {update_script}\n")
                f.write(f"Launcher log: {log_file}\n")
                f.write(f"Process ID: {os.getpid()}\n")
        except Exception as e:
            logging.warning(f"Could not create update marker file: {e}")
        
        # Create update launcher batch file
        launcher_batch = create_update_launcher(update_script, os.getpid())
        if not launcher_batch:
            logging.error("Failed to create update launcher batch file")
            return False
        
        # Try multiple launch methods, starting with the best one
        launch_success = False
        
        # Method 1: Using the launcher batch file with a visible window
        try:
            logging.info("Launching update script (Method 1: Using launcher batch file with visible window)...")
            subprocess.Popen(
                f'cmd.exe /c start "Multi-Max Update Launcher" "{launcher_batch}"', 
                shell=True
            )
            launch_success = True
            logging.info("Started update launcher with visible window")
        except Exception as e1:
            logging.warning(f"Method 1 failed to launch update launcher: {e1}")
            
            # Method 2: Direct cmd.exe call to the update script
            if not launch_success:
                try:
                    logging.info("Launching update script (Method 2: Direct cmd.exe call)...")
                    cmd = f'cmd.exe /c start "Multi-Max Update" "{update_script}"'
                    subprocess.Popen(cmd, shell=True)
                    launch_success = True
                    logging.info("Started update process with direct cmd.exe call")
                except Exception as e2:
                    logging.warning(f"Method 2 failed to launch update script: {e2}")
                    
                    # Method 3: Using PowerShell to elevate privileges
                    if not launch_success:
                        try:
                            logging.info("Launching update script (Method 3: Using PowerShell elevation)...")
                            # Use PowerShell to run with elevated privileges
                            ps_cmd = f'Start-Process -FilePath "{update_script}" -Verb RunAs'
                            subprocess.Popen(
                                ['powershell.exe', '-Command', ps_cmd],
                                shell=True
                            )
                            launch_success = True
                            logging.info("Started update process with PowerShell elevation")
                        except Exception as e3:
                            logging.warning(f"Method 3 failed to launch update script: {e3}")
                            
                            # Method 4: Last resort - try to call directly
                            if not launch_success:
                                try:
                                    logging.info("Launching update script (Method 4: Direct call as fallback)...")
                                    subprocess.Popen(update_script, shell=True)
                                    launch_success = True
                                    logging.info("Started update process with direct call")
                                except Exception as e4:
                                    logging.error(f"Method 4 failed to launch update script: {e4}")
        
        if launch_success:
            logging.info("Update process initiated successfully")
            
            # Create a success marker
            success_marker = os.path.join(windows_dir, "update_launch_success.txt")
            try:
                with open(success_marker, 'w') as f:
                    f.write(f"Update launch succeeded at: {time.ctime()}\n")
                    f.write(f"Update script: {update_script}\n")
                    f.write(f"Launcher log: {log_file}\n")
                    f.write(f"Process ID: {os.getpid()}\n")
            except Exception as e:
                logging.warning(f"Could not create success marker file: {e}")
                
            # Sleep to give the update script time to start
            time.sleep(2)
            return True
        else:
            logging.error("All methods to launch update script failed")
            
            # Create a failure marker with detailed diagnostics
            failure_marker = os.path.join(windows_dir, "update_launch_failure.txt")
            try:
                with open(failure_marker, 'w') as f:
                    f.write(f"Update launch FAILED at: {time.ctime()}\n")
                    f.write(f"Update script path: {update_script}\n")
                    f.write(f"Update script exists: {os.path.exists(update_script)}\n")
                    f.write(f"Launcher batch: {launcher_batch}\n")
                    f.write(f"Launcher batch exists: {os.path.exists(launcher_batch)}\n")
                    f.write(f"Launcher log: {log_file}\n")
                    f.write(f"Process ID: {os.getpid()}\n")
                    
                    # Check if the parent directory is writeable
                    parent_dir = os.path.dirname(windows_dir)
                    try:
                        test_file = os.path.join(parent_dir, "write_test.tmp")
                        with open(test_file, 'w') as tf:
                            tf.write("test")
                        os.remove(test_file)
                        f.write(f"Parent directory is writeable: Yes\n")
                    except Exception as write_err:
                        f.write(f"Parent directory is writeable: No - {write_err}\n")
                    
                    # List all files in the directory
                    f.write(f"\nFiles in {windows_dir}:\n")
                    try:
                        for filename in os.listdir(windows_dir):
                            file_path = os.path.join(windows_dir, filename)
                            if os.path.isfile(file_path):
                                f.write(f"  {filename} ({os.path.getsize(file_path)} bytes)\n")
                            else:
                                f.write(f"  {filename} (directory)\n")
                    except Exception as dir_err:
                        f.write(f"Error listing directory: {dir_err}\n")
                    
                    # Try to diagnose Python installation
                    f.write("\nPython installation:\n")
                    try:
                        f.write(f"  Python executable: {sys.executable}\n")
                        f.write(f"  Python version: {sys.version}\n")
                        f.write(f"  Python path: {sys.path}\n")
                    except Exception as py_err:
                        f.write(f"Error getting Python info: {py_err}\n")
            except Exception as marker_err:
                logging.warning(f"Could not create failure marker file: {marker_err}")
                
            return False
            
    except Exception as e:
        logging.error(f"Failed to start update process: {e}")
        logging.exception("Stack trace:")
        return False
        
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Run the update if executed directly
    logging.info("Starting update launcher directly")
    
    # First, make sure the update script exists
    update_script = find_existing_update_script()
    if not update_script:
        update_script = ensure_update_script_exists()
    
    # Check if we're running as admin, and if not, try to elevate
    if not is_admin():
        logging.info("Not running as admin, attempting to elevate privileges...")
        if elevate_privileges(__file__):
            logging.info("Elevation request sent, exiting current instance")
            sys.exit(0)
        else:
            logging.warning("Could not elevate privileges, will try to continue anyway")
    
    # Start the update process
    if start_update_process():
        logging.info("Update process started successfully")
        # Exit immediately to allow the update script to take over
        sys.exit(0)
    else:
        logging.error("Failed to start the update process")
        sys.exit(1) 