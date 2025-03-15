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
import traceback

# Version information - DO NOT REMOVE OR MODIFY THIS LINE FORMAT
VERSION = "1.0.0"
BUILD_DATE = "2023-03-15"

# Configure logging
try:
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'update_checker_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
except Exception as e:
    # Fallback if there's an issue with log file setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    print(f"Warning: Failed to set up log file: {e}")

logger = logging.getLogger('update_checker')

def get_script_info():
    """Get information about the script and environment for debugging."""
    info = {
        "__file__": __file__,
        "abspath": os.path.abspath(__file__),
        "dirname": os.path.dirname(os.path.abspath(__file__)),
        "parent_dir": os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "cwd": os.getcwd(),
        "sys.path": sys.path,
        "platform": platform.system(),
        "python_version": platform.python_version()
    }
    return info

def log_script_info():
    """Log script information for debugging purposes."""
    info = get_script_info()
    logger.info("Update checker script information:")
    for key, value in info.items():
        if key == "sys.path":
            logger.info(f"  {key}: (first 3 entries): {value[:3]}")
        else:
            logger.info(f"  {key}: {value}")

def self_check():
    """Run a self-check to verify the update checker is correctly installed."""
    results = {"status": "pass", "issues": []}
    
    # Check if we can import ourselves
    try:
        import update_checker
        logger.info(f"Self-import successful: {update_checker.__file__}")
    except ImportError as e:
        results["status"] = "fail"
        results["issues"].append(f"Cannot import update_checker module: {e}")
    
    # Check if the VERSION variable is defined correctly
    if "VERSION" not in globals() or not VERSION:
        results["status"] = "fail"
        results["issues"].append("VERSION variable is not defined or empty")
    
    # Check if git is available
    try:
        subprocess.run(['git', '--version'], capture_output=True, text=True, check=True)
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        results["status"] = "warning"
        results["issues"].append(f"Git command not available: {e}")
    
    # Log the results
    if results["status"] == "pass":
        logger.info("Self-check passed. Update checker is correctly installed.")
    else:
        level = logging.WARNING if results["status"] == "warning" else logging.ERROR
        logger.log(level, f"Self-check {results['status']}:")
        for issue in results["issues"]:
            logger.log(level, f"  - {issue}")
    
    return results

def is_git_repo():
    """Check if the current directory is a git repository."""
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # Log directories we're checking
    logger.info(f"Checking for Git repository in: {os.getcwd()}")
    logger.info(f"Script directory: {script_dir}")
    logger.info(f"Parent directory: {parent_dir}")
    
    # Try both the current directory and parent directory
    for check_dir in [os.getcwd(), parent_dir]:
        logger.info(f"Checking if {check_dir} is a Git repository...")
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--is-inside-work-tree'],
                cwd=check_dir,
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0 and result.stdout.strip() == 'true':
                logger.info(f"Found Git repository in {check_dir}")
                # Change to this directory for subsequent Git operations
                os.chdir(check_dir)
                return True
        except FileNotFoundError:
            logger.warning("Git command not found. Cannot check for updates.")
            return False
        except Exception as e:
            logger.warning(f"Error checking Git repository in {check_dir}: {e}")
    
    logger.warning("No Git repository found in current or parent directory.")
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
        branch = result.stdout.strip()
        logger.info(f"Current branch: {branch}")
        return branch
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.error(f"Failed to get current branch: {e}")
        return None

def fetch_updates():
    """Fetch updates from the remote repository."""
    try:
        logger.info("Fetching updates from remote repository...")
        result = subprocess.run(
            ['git', 'fetch', '--all'],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info("Updates fetched successfully")
        logger.debug(f"Fetch output: {result.stdout}")
        return True
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.error(f"Failed to fetch updates: {e}")
        if hasattr(e, 'stderr'):
            logger.error(f"Error details: {e.stderr}")
        return False

def get_commit_count_difference(branch_name):
    """
    Get the number of commits the local branch is behind the remote branch.
    Returns: (commits_behind, commits_ahead)
    """
    try:
        # Make sure we have the latest updates
        fetch_updates()
        
        logger.info(f"Checking commit difference for branch: {branch_name}")
        
        # First check if remote branch exists
        remote_branch_check = subprocess.run(
            ['git', 'ls-remote', '--heads', 'origin', branch_name],
            capture_output=True,
            text=True,
            check=False
        )
        
        if not remote_branch_check.stdout.strip():
            logger.warning(f"Remote branch 'origin/{branch_name}' does not exist")
            return 0, 0
        
        # Check how many commits behind we are
        behind_result = subprocess.run(
            ['git', 'rev-list', '--left-right', '--count', f'{branch_name}...origin/{branch_name}'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the output (format is "X Y" where X is behind and Y is ahead)
        output = behind_result.stdout.strip()
        logger.info(f"Commit difference output: {output}")
        
        match = re.match(r'(\d+)\s+(\d+)', output)
        if match:
            behind, ahead = map(int, match.groups())
            logger.info(f"Local branch is {behind} commits behind and {ahead} commits ahead of remote")
            return behind, ahead
        
        logger.warning(f"Could not parse commit difference output: {output}")
        return 0, 0
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.error(f"Failed to check commit difference: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            logger.error(f"Error details: {e.stderr}")
        return 0, 0

def get_latest_remote_version():
    """
    Try to get the latest version number from the remote repository.
    This assumes there's a VERSION or similar file in the repo.
    """
    try:
        logger.info("Attempting to get latest version from remote repository...")
        
        # Try to get the version from the remote update_checker.py file
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
                version = match.group(1)
                logger.info(f"Found remote version in update_checker.py: {version}")
                return version
            else:
                logger.warning("VERSION pattern not found in remote update_checker.py")
        else:
            logger.warning(f"Failed to get update_checker.py from remote: {result.stderr}")
            # Try an alternate branch
            alt_result = subprocess.run(
                ['git', 'show', 'origin/master:windows/update_checker.py'],
                capture_output=True,
                text=True,
                check=False
            )
            if alt_result.returncode == 0:
                match = re.search(r'VERSION\s*=\s*[\'"]([^\'"]+)[\'"]', alt_result.stdout)
                if match:
                    version = match.group(1)
                    logger.info(f"Found remote version in master branch: {version}")
                    return version
        
        # If that fails, try to get the version from a VERSION file if it exists
        logger.info("Trying to get version from VERSION file...")
        result = subprocess.run(
            ['git', 'show', 'origin/main:VERSION'],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            version = result.stdout.strip()
            logger.info(f"Found remote version in VERSION file: {version}")
            return version
        else:
            # Try master branch VERSION file
            alt_result = subprocess.run(
                ['git', 'show', 'origin/master:VERSION'],
                capture_output=True,
                text=True,
                check=False
            )
            if alt_result.returncode == 0:
                version = alt_result.stdout.strip()
                logger.info(f"Found remote version in master branch VERSION file: {version}")
                return version
            
        logger.warning("Could not find version information in remote repository")
        return None
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.error(f"Failed to get remote version: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            logger.error(f"Error details: {e.stderr}")
        return None

def perform_update():
    """
    Perform an automatic update by pulling the latest changes from the remote repository.
    Returns True if successful, False otherwise.
    """
    try:
        logger.info("Performing automatic update...")
        
        # First stash any local changes to avoid conflicts
        stash_result = subprocess.run(
            ['git', 'stash'],
            capture_output=True,
            text=True,
            check=False  # Don't raise an exception if there's nothing to stash
        )
        
        if "No local changes to save" in stash_result.stdout:
            logger.info("No local changes to stash")
        else:
            logger.info("Local changes stashed")
        
        # Pull the latest changes
        logger.info("Pulling latest changes...")
        result = subprocess.run(
            ['git', 'pull', '--ff-only'],
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info(f"Update result: {result.stdout}")
        updated = "Already up to date" not in result.stdout
        
        if updated:
            logger.info("Repository updated successfully")
        else:
            logger.info("Repository already up to date")
            
        return updated
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.error(f"Failed to perform update: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            logger.error(f"Error details: {e.stderr}")
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
    
    # Log script info for debugging
    log_script_info()
    
    # Run self-check
    self_check_results = self_check()
    if self_check_results["status"] == "fail":
        logger.error("Self-check failed. Update checking may not work correctly.")
    
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
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with verbose logging')
    parser.add_argument('--self-check', action='store_true', help='Run self-check and exit')
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Self-check only
    if args.self_check:
        log_script_info()
        results = self_check()
        print(f"Self-check status: {results['status'].upper()}")
        if results["issues"]:
            print("Issues found:")
            for issue in results["issues"]:
                print(f"  - {issue}")
        else:
            print("No issues found. Update checker is correctly installed.")
        return 0
    
    # Version info only
    if args.version:
        print(f"Multi-Max Update Checker v{VERSION} (built {BUILD_DATE})")
        return 0
    
    print("Checking for Multi-Max updates...")
    
    # Change to the root directory of the project
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        logger.info(f"Changing to parent directory: {parent_dir}")
        os.chdir(parent_dir)
    except Exception as e:
        logger.error(f"Failed to change to parent directory: {e}")
        traceback.print_exc()
    
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
        exit_code = main()
        logger.info(f"Update checker exiting with code {exit_code}")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nUpdate check cancelled.")
        logger.info("Update check cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        print(f"\nAn error occurred while checking for updates: {e}")
        traceback.print_exc()
        sys.exit(1) 