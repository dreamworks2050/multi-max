#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Max - Windows Version Launcher

This script serves as the entry point for the Windows version of Multi-Max.
It performs platform-specific checks and launches the main application.
"""

import os
import sys
import time
import argparse
import subprocess
import logging
from logging.handlers import RotatingFileHandler
import platform

# Set up logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'multimax_windows_{time.strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(log_file, maxBytes=10485760, backupCount=5)
    ]
)

logger = logging.getLogger('multimax_windows')

# Try to import the update checker
UPDATE_CHECKER_AVAILABLE = False
try:
    from update_checker import check_for_updates, display_update_message, VERSION
    UPDATE_CHECKER_AVAILABLE = True
    logger.info(f"Update checker available. Current version: {VERSION}")
except ImportError as e:
    logger.warning(f"Update checker not available: {e}")
    # Define VERSION for use if update_checker is not available
    VERSION = "1.0.0"

def check_windows_environment():
    """Check if running on Windows and verify system compatibility."""
    if platform.system() != 'Windows':
        logger.warning(f"Running on non-Windows system: {platform.system()}")
        print("WARNING: This is the Windows version of Multi-Max, but you're running on",
              platform.system())
        print("The application may not function correctly.")
        print()
        return False
    
    # Check Windows version
    if platform.release() not in ['10', '11'] and not platform.release().startswith('Server'):
        logger.warning(f"Unsupported Windows version: {platform.release()}")
        print(f"WARNING: Windows {platform.release()} is not officially supported.")
        print("Multi-Max is designed for Windows 10/11.")
        print("The application may not function correctly.")
        print()
        return False
    
    return True

def check_requirements():
    """Check for required dependencies."""
    # Check Python version
    python_version = platform.python_version_tuple()
    if int(python_version[0]) < 3 or (int(python_version[0]) == 3 and int(python_version[1]) < 6):
        logger.warning(f"Unsupported Python version: {platform.python_version()}")
        print(f"WARNING: Python {platform.python_version()} is not supported.")
        print("Multi-Max requires Python 3.6 or newer.")
        return False
    
    # Check for required packages
    required_packages = ['numpy', 'pygame', 'opencv-python', 'python-dotenv', 'psutil']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.split('-')[0])  # Handle packages like opencv-python
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing required packages: {', '.join(missing_packages)}")
        print("WARNING: The following required packages are missing:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        print()
        return False
    
    return True

def check_and_update(auto_update=False):
    """
    Check for updates and optionally perform an auto-update.
    
    Args:
        auto_update (bool): Whether to automatically update if available
        
    Returns:
        bool: True if the application should continue, False if it should exit
    """
    if not UPDATE_CHECKER_AVAILABLE:
        logger.warning("Update checker not available. Skipping update check.")
        return True
    
    try:
        # Change to the root directory for the update check
        original_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        logger.info("Checking for updates...")
        update_available, update_performed, current_version, remote_version = check_for_updates(auto_update)
        
        # Display update message
        display_update_message(update_available, update_performed, current_version, remote_version)
        
        # Change back to the original directory
        os.chdir(original_dir)
        
        if update_performed:
            logger.info("Update was performed. Exiting to allow restart with new version.")
            print("\nPlease restart Multi-Max to use the new version.")
            return False
        
        return True
    except Exception as e:
        logger.exception(f"Error during update check: {e}")
        print(f"Warning: Failed to check for updates: {e}")
        return True

def main():
    """Main entry point for the Windows version of Multi-Max."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Multi-Max Windows Launcher')
    parser.add_argument('--skip-update-check', action='store_true', 
                        help='Skip checking for updates at startup')
    parser.add_argument('--force-update', action='store_true',
                        help='Force update check and auto-update if available')
    parser.add_argument('--version', action='store_true',
                        help='Display version information and exit')
    
    # Pass along any other arguments to the main script
    parser.add_argument('args', nargs='*', 
                        help='Additional arguments to pass to the main script')
    
    args, unknown_args = parser.parse_known_args()
    
    # Handle version request
    if args.version:
        print(f"Multi-Max (Windows) version {VERSION}")
        return 0
    
    # Print welcome message
    print("\n" + "="*60)
    print("   🚀 Multi-Max Windows Launcher 🚀")
    print("="*60)
    
    # Check Windows environment
    check_windows_environment()
    
    # Check for dependencies
    if not check_requirements():
        print("WARNING: Some requirements are not met.")
        print("The application may not function correctly.")
        print()
    
    # Check for updates if not skipped
    if not args.skip_update_check or args.force_update:
        if not check_and_update(args.force_update):
            return 100  # Special exit code indicating update was performed
    
    # Get parent directory to find main.py
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_script = os.path.join(parent_dir, 'main.py')
    
    if not os.path.exists(main_script):
        logger.error(f"Main script not found: {main_script}")
        print(f"ERROR: Main script not found: {main_script}")
        print("Multi-Max may not be installed correctly.")
        return 1
    
    # Build command line for main.py
    cmd = [sys.executable, main_script]
    
    # Add passed through args
    if args.args:
        cmd.extend(args.args)
    
    # Add any unknown args
    if unknown_args:
        cmd.extend(unknown_args)
    
    logger.info(f"Launching main application: {' '.join(cmd)}")
    print("\nLaunching Multi-Max...\n")
    
    # Launch the main application
    try:
        return subprocess.call(cmd)
    except Exception as e:
        logger.exception(f"Error launching main application: {e}")
        print(f"ERROR: Failed to launch Multi-Max: {e}")
        return 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nMulti-Max terminated by user.")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        print(f"\nAn error occurred: {e}")
        sys.exit(1)