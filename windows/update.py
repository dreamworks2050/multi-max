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

def is_admin():
    """Check if the current process has administrator privileges on Windows."""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except:
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

def start_update_process():
    """
    Start the complete update process by launching the Update-MultiMax.bat script.
    The script will:
    1. Wipe the entire folder
    2. Redownload the GitHub repo
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
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    logging.info(f"Created update log file: {log_file}")
    logging.info(f"Python version: {sys.version}")
    logging.info(f"Platform: {platform.platform()}")
    logging.info(f"Running as administrator: {is_admin()}")
    
    # Ensure we're on Windows
    if platform.system() != 'Windows':
        logging.error("Update process is only supported on Windows")
        return False
    
    # Find the update script
    try:
        windows_dir = os.path.dirname(os.path.abspath(__file__))
        logging.info(f"Windows directory: {windows_dir}")
        update_script = os.path.join(windows_dir, "Update-MultiMax.bat")
        
        if not os.path.exists(update_script):
            logging.error(f"Update script not found at: {update_script}")
            # Try to find it in other locations
            parent_dir = os.path.dirname(windows_dir)
            alt_locations = [
                os.path.join(parent_dir, "Update-MultiMax.bat"),
                os.path.join(parent_dir, "windows", "Update-MultiMax.bat")
            ]
            
            for loc in alt_locations:
                logging.info(f"Checking alternative location: {loc}")
                if os.path.exists(loc):
                    update_script = loc
                    logging.info(f"Found update script at alternative location: {update_script}")
                    break
            else:
                logging.error("Update script not found in any expected location")
                return False
        
        logging.info(f"Found update script at: {update_script}")
        
        # Create a marker file to indicate an update is in progress
        marker_file = os.path.join(windows_dir, "update_in_progress.txt")
        try:
            with open(marker_file, 'w') as f:
                f.write(f"Update started at: {time.ctime()}\n")
                f.write(f"Update script: {update_script}\n")
                f.write(f"Launcher log: {log_file}\n")
        except Exception as e:
            logging.warning(f"Could not create update marker file: {e}")
        
        # Write a batch script that can help launch the updater with elevated privileges if needed
        launcher_batch = os.path.join(tempfile.gettempdir(), "launch_multimax_update.bat")
        try:
            with open(launcher_batch, 'w') as f:
                f.write('@echo off\n')
                f.write('echo Multi-Max Update Launcher\n')
                f.write('echo ===================================\n')
                f.write('echo This will launch the Multi-Max updater with administrator privileges if needed.\n')
                f.write('echo.\n')
                f.write('echo If you see a User Account Control prompt, please click "Yes" to allow the update to proceed.\n')
                f.write('echo.\n')
                f.write('echo Starting update process...\n')
                f.write('echo.\n')
                f.write(f'powershell -Command "Start-Process -FilePath \'{update_script}\' -Verb RunAs"\n')
                f.write('echo Update process has started. You can close this window.\n')
                f.write('timeout /t 10\n')
            logging.info(f"Created launcher batch file: {launcher_batch}")
        except Exception as e:
            logging.warning(f"Could not create launcher batch file: {e}")
        
        # Try multiple launch methods, starting with the best one
        launch_success = False
        
        # Method 1: Direct subprocess call with a visible window
        try:
            logging.info("Launching update script (Method 1: Direct subprocess with visible window)...")
            # Command to create a new command prompt and run the script
            # This approach should show a visible window to the user
            si = subprocess.STARTUPINFO()
            si.dwFlags = subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = 1  # SW_SHOWNORMAL
            
            proc = subprocess.Popen(
                [update_script], 
                shell=True, 
                startupinfo=si,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            logging.info(f"Subprocess started with PID: {proc.pid if hasattr(proc, 'pid') else 'unknown'}")
            launch_success = True
        except Exception as e1:
            logging.warning(f"Method 1 failed to launch update script: {e1}")
            
            # Method 2: Using the launcher batch file
            if not launch_success:
                try:
                    logging.info("Launching update script (Method 2: Using launcher batch file)...")
                    subprocess.Popen(
                        [launcher_batch], 
                        shell=True,
                        creationflags=subprocess.CREATE_NEW_CONSOLE
                    )
                    launch_success = True
                except Exception as e2:
                    logging.warning(f"Method 2 failed to launch update script: {e2}")
                    
                    # Method 3: Using cmd.exe to start
                    if not launch_success:
                        try:
                            logging.info("Launching update script (Method 3: Using cmd.exe)...")
                            cmd = f'cmd.exe /c start "Multi-Max Update" "{update_script}"'
                            subprocess.Popen(cmd, shell=True)
                            launch_success = True
                        except Exception as e3:
                            logging.warning(f"Method 3 failed to launch update script: {e3}")
                            
                            # Method 4: Last resort - try to elevate with shell execute
                            if not launch_success:
                                try:
                                    logging.info("Launching update script (Method 4: Shell Execute with elevation)...")
                                    ctypes.windll.shell32.ShellExecuteW(
                                        None, "runas", "cmd.exe", f"/c start \"Multi-Max Update\" \"{update_script}\"", None, 1
                                    )
                                    launch_success = True
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
                    f.write(f"Launcher log: {log_file}\n")
                    
                    # Check if the parent directory is writeable
                    parent_dir = os.path.dirname(windows_dir)
                    try:
                        test_file = os.path.join(parent_dir, "write_test.tmp")
                        with open(test_file, 'w') as tf:
                            tf.write("test")
                        os.remove(test_file)
                        f.write(f"Parent directory is writeable: Yes\n")
                    except:
                        f.write(f"Parent directory is writeable: No\n")
                    
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