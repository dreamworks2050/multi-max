#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Max - Windows Version Launcher

This script serves as the entry point for the Windows version of Multi-Max.
It performs platform-specific checks and launches the main application.
"""

# Windows-specific version marker - DO NOT REMOVE - used by installer to verify correct version
__windows_specific_version__ = True

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

# Version for when update_checker is not available
VERSION = "1.0.0"

# Try to import the update checker
UPDATE_CHECKER_AVAILABLE = False
try:
    # Add the current directory to the path to ensure update_checker can be found
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Now try to import
    from update_checker import check_for_updates, display_update_message, VERSION
    UPDATE_CHECKER_AVAILABLE = True
    logger.info(f"Update checker available. Current version: {VERSION}")
except ImportError as e:
    logger.warning(f"Update checker not available: {e}")
    # Try an alternate method to import update_checker
    try:
        windows_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'windows')
        if os.path.exists(os.path.join(windows_dir, 'update_checker.py')):
            sys.path.append(windows_dir)
            from update_checker import check_for_updates, display_update_message, VERSION
            UPDATE_CHECKER_AVAILABLE = True
            logger.info(f"Update checker available (alternate path). Current version: {VERSION}")
    except ImportError as e2:
        logger.warning(f"Update checker still not available after trying alternate path: {e2}")

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
    script_path = os.path.abspath(__file__)
    windows_dir = os.path.dirname(script_path)
    parent_dir = os.path.dirname(windows_dir)
    main_script = os.path.join(parent_dir, 'main.py')
    
    logger.info(f"Script path: {script_path}")
    logger.info(f"Windows directory: {windows_dir}")
    logger.info(f"Parent directory: {parent_dir}")
    logger.info(f"Looking for main script at: {main_script}")
    
    if not os.path.exists(main_script):
        logger.error(f"Main script not found: {main_script}")
        print(f"ERROR: Main script not found: {main_script}")
        print("Multi-Max may not be installed correctly.")
        
        # Additional error info to help diagnose
        print("\nDebugging information:")
        print(f"Current directory: {os.getcwd()}")
        print(f"__file__: {__file__}")
        print(f"Script path: {script_path}")
        print(f"Windows directory: {windows_dir}")
        print(f"Parent directory: {parent_dir}")
        print(f"Checking if windows dir exists: {os.path.exists(windows_dir)}")
        print(f"Checking if parent dir exists: {os.path.exists(parent_dir)}")
        
        # List files in parent dir to help diagnose
        try:
            parent_files = os.listdir(parent_dir)
            print(f"\nFiles in parent directory ({parent_dir}):")
            for f in parent_files:
                print(f"  - {f}")
        except Exception as e:
            print(f"Could not list files in parent directory: {e}")
            
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