#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Max Update Checker

This module handles version checking and automatic updates for Multi-Max.
It compares the local version with the remote repository and can automatically
pull updates if requested.
"""

import os
import sys
import logging
import subprocess
import re
import platform
import time
from datetime import datetime

# Version information
VERSION = "1.0.0"
BUILD_DATE = "2023-03-15"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                         'logs', f'update_checker_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))
    ]
)
logger = logging.getLogger('update_checker')

def is_git_repo():
    """Check if the current directory is a git repository."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--is-inside-work-tree'],
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode == 0 and result.stdout.strip() == 'true'
    except FileNotFoundError:
        logger.warning("Git command not found. Cannot check for updates.")
        return False

def get_current_branch():
    """Get the name of the current git branch."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.error(f"Failed to get current branch: {e}")
        return None

def fetch_updates():
    """Fetch updates from the remote repository."""
    try:
        subprocess.run(
            ['git', 'fetch', '--all'],
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.error(f"Failed to fetch updates: {e}")
        return False

def get_commit_count_difference(branch_name):
    """
    Get the number of commits the local branch is behind the remote branch.
    Returns: (commits_behind, commits_ahead)
    """
    try:
        # Make sure we have the latest updates
        fetch_updates()
        
        # Check how many commits behind we are
        behind_result = subprocess.run(
            ['git', 'rev-list', '--left-right', '--count', f'{branch_name}...origin/{branch_name}'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the output (format is "X Y" where X is behind and Y is ahead)
        match = re.match(r'(\d+)\s+(\d+)', behind_result.stdout.strip())
        if match:
            behind, ahead = map(int, match.groups())
            return behind, ahead
        
        return 0, 0
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.error(f"Failed to check commit difference: {e}")
        return 0, 0

def get_latest_remote_version():
    """
    Try to get the latest version number from the remote repository.
    This assumes there's a VERSION or similar file in the repo.
    """
    try:
        # Try to get the version from the remote main.py file
        result = subprocess.run(
            ['git', 'show', 'origin/main:windows/update_checker.py'],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            # Look for VERSION = "x.y.z" pattern
            match = re.search(r'VERSION\s*=\s*[\'"]([^\'"]+)[\'"]', result.stdout)
            if match:
                return match.group(1)
        
        # If that fails, try to get the version from a VERSION file if it exists
        result = subprocess.run(
            ['git', 'show', 'origin/main:VERSION'],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            return result.stdout.strip()
            
        return None
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.error(f"Failed to get remote version: {e}")
        return None

def perform_update():
    """
    Perform an automatic update by pulling the latest changes from the remote repository.
    Returns True if successful, False otherwise.
    """
    try:
        # First stash any local changes to avoid conflicts
        subprocess.run(
            ['git', 'stash'],
            capture_output=True,
            text=True,
            check=False  # Don't raise an exception if there's nothing to stash
        )
        
        # Pull the latest changes
        result = subprocess.run(
            ['git', 'pull', '--ff-only'],
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info(f"Update result: {result.stdout}")
        return "Already up to date" not in result.stdout
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.error(f"Failed to perform update: {e}")
        return False

def check_for_updates(auto_update=False):
    """
    Check if there are updates available for the application.
    
    Args:
        auto_update (bool): Whether to automatically update if updates are available.
        
    Returns:
        tuple: (update_available, update_performed, current_version, remote_version)
    """
    # Initialize return values
    update_available = False
    update_performed = False
    current_version = VERSION
    remote_version = None
    
    # Check if we're in a git repository
    if not is_git_repo():
        logger.warning("Not a git repository. Cannot check for updates.")
        return update_available, update_performed, current_version, remote_version
    
    # Get the current branch
    branch = get_current_branch()
    if not branch:
        logger.warning("Could not determine current branch. Cannot check for updates.")
        return update_available, update_performed, current_version, remote_version
    
    # If we're not on main or master branch, don't update
    if branch not in ['main', 'master']:
        logger.info(f"Current branch is {branch}, not main/master. Skipping update check.")
        return update_available, update_performed, current_version, remote_version
    
    # Fetch updates
    if not fetch_updates():
        logger.warning("Failed to fetch updates. Cannot check for updates.")
        return update_available, update_performed, current_version, remote_version
    
    # Check how many commits behind we are
    commits_behind, commits_ahead = get_commit_count_difference(branch)
    
    # Try to get the latest version from the remote
    remote_version = get_latest_remote_version()
    
    if commits_behind > 0:
        update_available = True
        logger.info(f"Update available. Local branch is {commits_behind} commits behind remote.")
        
        if remote_version:
            logger.info(f"Current version: {current_version}, Remote version: {remote_version}")
        
        if auto_update:
            logger.info("Performing automatic update...")
            update_performed = perform_update()
            
            if update_performed:
                logger.info("Update completed successfully.")
            else:
                logger.warning("Update was not performed or completed with issues.")
    else:
        logger.info("No updates available. You're running the latest version.")
    
    return update_available, update_performed, current_version, remote_version

def display_update_message(update_available, update_performed, current_version, remote_version):
    """Display a user-friendly message about the update status."""
    if update_performed:
        print("\n" + "="*60)
        print("  🎉 Multi-Max has been updated successfully! 🎉")
        if remote_version:
            print(f"  Version {current_version} → {remote_version}")
        print("  Please restart the application to use the new version.")
        print("="*60 + "\n")
    elif update_available:
        print("\n" + "="*60)
        print("  ⚠️ A new version of Multi-Max is available!")
        if remote_version:
            print(f"  Current version: {current_version}")
            print(f"  Available version: {remote_version}")
        print("  Run with --update to update automatically.")
        print("="*60 + "\n")
    else:
        if remote_version:
            print(f"\nYou're running Multi-Max version {current_version} (latest)")
        else:
            print(f"\nYou're running Multi-Max version {current_version}")

def main():
    """Main function to run the update checker."""
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Multi-Max Update Checker')
    parser.add_argument('--auto-update', action='store_true', help='Automatically update if a new version is available')
    parser.add_argument('--version', action='store_true', help='Display the current version and exit')
    args = parser.parse_args()
    
    if args.version:
        print(f"Multi-Max Update Checker v{VERSION} (built {BUILD_DATE})")
        sys.exit(0)
    
    print("Checking for Multi-Max updates...")
    
    # Change to the root directory of the project
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Check for updates
    update_available, update_performed, current_version, remote_version = check_for_updates(args.auto_update)
    
    # Display update message
    display_update_message(update_available, update_performed, current_version, remote_version)
    
    # Return appropriate exit code
    if update_performed:
        # Return a specific code to indicate an update was performed
        return 100
    elif update_available:
        # Return a specific code to indicate an update is available
        return 50
    else:
        # Return 0 to indicate no update needed
        return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nUpdate check cancelled.")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        print(f"\nAn error occurred while checking for updates: {e}")
        sys.exit(1) 