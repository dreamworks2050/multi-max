#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Multi-Max Update Checker for Windows

This is a simplified version of the update checker that avoids complex
Git operations that might fail on Windows systems.
"""

import os
import sys
import subprocess
import platform
import time
from datetime import datetime

# Version information
VERSION = "1.0.0"
BUILD_DATE = "2023-03-15"

# Simple logging setup
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'simple_update_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

def log(message, level="INFO"):
    """Simple logging function"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"{timestamp} - {level} - {message}"
    print(log_message)
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_message + "\n")

def get_version_from_file():
    """Get the version from the VERSION file"""
    try:
        version_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'VERSION')
        if os.path.exists(version_path):
            with open(version_path, 'r') as f:
                return f.read().strip()
        return VERSION
    except Exception as e:
        log(f"Error reading VERSION file: {e}", "ERROR")
        return VERSION

def is_git_available():
    """Check if Git is available"""
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def is_git_repo():
    """Check if the current directory is a Git repository"""
    if not is_git_available():
        log("Git is not available", "WARNING")
        return False

    try:
        # Try the current directory
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0 and result.stdout.strip() == 'true':
            return True
        
        # Try the parent directory
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=parent_dir,
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0 and result.stdout.strip() == 'true':
            os.chdir(parent_dir)
            return True
            
        return False
    except Exception as e:
        log(f"Error checking Git repository: {e}", "ERROR")
        return False

def update_application():
    """Perform a simple update using git pull"""
    if not is_git_repo():
        log("Not a Git repository. Cannot update.", "ERROR")
        return False
    
    try:
        # Stash changes
        subprocess.run(["git", "stash"], capture_output=True, check=False)
        
        # Pull changes
        result = subprocess.run(
            ["git", "pull"],
            capture_output=True,
            text=True,
            check=True
        )
        
        log(f"Update result: {result.stdout}")
        
        if "Already up to date" in result.stdout:
            log("No updates available. Already up to date.")
            return False
        
        log("Update successful!")
        return True
    except Exception as e:
        log(f"Error updating application: {e}", "ERROR")
        return False

def check_for_updates(auto_update=False):
    """Simplified update check"""
    log(f"Checking for updates (auto_update={auto_update})...")
    
    # Print system information for debugging
    log(f"Python version: {platform.python_version()}")
    log(f"System: {platform.system()} {platform.release()}")
    log(f"Current directory: {os.getcwd()}")
    log(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    
    # Check if we can perform Git operations
    if not is_git_available():
        log("Git is not available. Cannot check for updates.", "WARNING")
        return False, False
    
    if not is_git_repo():
        log("Not in a Git repository. Cannot check for updates.", "WARNING")
        return False, False
    
    # Try to fetch the latest changes
    try:
        log("Fetching updates...")
        subprocess.run(["git", "fetch"], capture_output=True, check=True)
    except Exception as e:
        log(f"Error fetching updates: {e}", "ERROR")
        return False, False
    
    # Check if we're behind the remote
    try:
        result = subprocess.run(
            ["git", "status", "-uno"],
            capture_output=True,
            text=True,
            check=True
        )
        
        if "Your branch is behind" in result.stdout:
            log("Updates available!")
            
            if auto_update:
                updated = update_application()
                return True, updated
            else:
                return True, False
        else:
            log("No updates available.")
            return False, False
    except Exception as e:
        log(f"Error checking update status: {e}", "ERROR")
        return False, False

def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(description='Simple Multi-Max Update Checker')
    parser.add_argument('--auto-update', action='store_true', help='Automatically update if available')
    parser.add_argument('--version', action='store_true', help='Show version information')
    args = parser.parse_args()
    
    if args.version:
        print(f"Multi-Max version {get_version_from_file()} (built {BUILD_DATE})")
        return 0
    
    print("=" * 60)
    print("  Multi-Max Simple Update Checker")
    print("=" * 60)
    print()
    
    updates_available, updated = check_for_updates(args.auto_update)
    
    if updated:
        print("\n" + "=" * 60)
        print("  🎉 Multi-Max has been updated successfully! 🎉")
        print("  Please restart the application to use the new version.")
        print("=" * 60 + "\n")
        return 100  # Special exit code for "updated"
    elif updates_available:
        print("\n" + "=" * 60)
        print("  ⚠️ A new version of Multi-Max is available!")
        print("  Run with --auto-update to update automatically.")
        print("=" * 60 + "\n")
        return 50  # Special exit code for "updates available"
    else:
        print(f"\nYou're running Multi-Max version {get_version_from_file()}")
        if not is_git_available() or not is_git_repo():
            print("Note: Git is not available or this is not a Git repository.")
            print("Some features like automatic updates may not work.")
        return 0  # No updates or cannot check

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nUpdate check cancelled.")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        log(f"Unhandled exception: {e}", "ERROR")
        sys.exit(1) 