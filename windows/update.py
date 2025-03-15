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
    
    # Ensure we're on Windows
    if platform.system() != 'Windows':
        logging.error("Update process is only supported on Windows")
        return False
    
    # Find the update script
    try:
        windows_dir = os.path.dirname(os.path.abspath(__file__))
        update_script = os.path.join(windows_dir, "Update-MultiMax.bat")
        
        if not os.path.exists(update_script):
            logging.error(f"Update script not found at: {update_script}")
            return False
        
        logging.info(f"Found update script at: {update_script}")
        
        # Create a marker file to indicate an update is in progress
        marker_file = os.path.join(windows_dir, "update_in_progress.txt")
        try:
            with open(marker_file, 'w') as f:
                f.write(f"Update started at: {time.ctime()}\n")
                f.write(f"Update script: {update_script}\n")
        except Exception as e:
            logging.warning(f"Could not create update marker file: {e}")
        
        # Launch the update script with various methods
        launch_success = False
        
        # Method 1: Direct subprocess call
        try:
            logging.info("Launching update script (method 1)...")
            subprocess.Popen([update_script], shell=True)
            launch_success = True
        except Exception as e1:
            logging.warning(f"Method 1 failed to launch update script: {e1}")
            
            # Method 2: Using cmd.exe to start
            if not launch_success:
                try:
                    logging.info("Launching update script (method 2)...")
                    subprocess.Popen(['cmd.exe', '/c', 'start', update_script], shell=True)
                    launch_success = True
                except Exception as e2:
                    logging.warning(f"Method 2 failed to launch update script: {e2}")
                    
                    # Method 3: Direct cmd string
                    if not launch_success:
                        try:
                            logging.info("Launching update script (method 3)...")
                            cmd = f'cmd.exe /c start "" "{update_script}"'
                            subprocess.Popen(cmd, shell=True)
                            launch_success = True
                        except Exception as e3:
                            logging.error(f"Method 3 failed to launch update script: {e3}")
        
        if launch_success:
            logging.info("Update process initiated successfully")
            # Sleep to give the update script time to start
            time.sleep(2)
            return True
        else:
            logging.error("All methods to launch update script failed")
            return False
            
    except Exception as e:
        logging.error(f"Failed to start update process: {e}")
        return False
        
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Run the update if executed directly
    if start_update_process():
        logging.info("Update process started successfully")
        # Exit immediately to allow the update script to take over
        sys.exit(0)
    else:
        logging.error("Failed to start the update process")
        sys.exit(1) 