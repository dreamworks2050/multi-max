#!/usr/bin/env python3
"""
Dependency Copier for Multi-Max

This script automatically copies all required dependencies from the virtual environment
to local directories for PyInstaller packaging. It should be run before packaging the
application with PyInstaller.

Copied dependencies:
- OpenCV (cv2)
- NumPy
- Python-dotenv
- Pygame
- PyObjC frameworks (Quartz, Cocoa, AppKit, Foundation, CoreFoundation, objc, PyObjCTools)
- psutil
- memory_profiler
"""

import os
import sys
import shutil
import site
import importlib
import argparse
import glob
from pathlib import Path

# List of packages to copy as directories
DIR_PACKAGES = [
    "cv2", 
    "numpy",
    "pygame", 
    "psutil",
    "Quartz",
    "Cocoa", 
    "objc", 
    "AppKit", 
    "Foundation", 
    "CoreFoundation", 
    "PyObjCTools",
    "dotenv"
]

# List of packages to copy as individual files
FILE_PACKAGES = [
    "memory_profiler"
]

def find_venv_site_packages():
    """Find the site-packages directory of the current virtual environment."""
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        # We're in a virtual environment
        if sys.platform == 'win32':
            site_packages = os.path.join(sys.prefix, 'Lib', 'site-packages')
        else:
            # Try to find the site-packages directory
            for path in site.getsitepackages():
                if path.endswith('site-packages'):
                    site_packages = path
                    break
            else:
                # If we can't find it, use a common pattern
                python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
                site_packages = os.path.join(sys.prefix, 'lib', f'python{python_version}', 'site-packages')
        
        return site_packages
    else:
        print("Warning: Not running in a virtual environment.")
        if site.ENABLE_USER_SITE:
            return site.USER_SITE
        else:
            return site.getsitepackages()[0]

def get_package_location(package_name):
    """Get the location of a package."""
    try:
        # Try to import the package to get its location
        package = importlib.import_module(package_name)
        if hasattr(package, '__file__'):
            if package.__file__ and '__init__.py' in package.__file__:
                # It's a package, return the directory
                return os.path.dirname(package.__file__)
            else:
                # It's a module, return the file
                return package.__file__
        return None
    except ImportError:
        print(f"Warning: Could not import package {package_name}")
        return None

def is_valid_path(path):
    """Check if a path exists and looks valid."""
    return path and os.path.exists(path)

def find_package_in_site_packages(site_packages, package_name):
    """Try to find a package in the site-packages directory."""
    # Try exact path
    direct_path = os.path.join(site_packages, package_name)
    if os.path.exists(direct_path):
        if os.path.isdir(direct_path):
            return direct_path
        else:
            # If it's a file, return it
            return direct_path
            
    # Check for directories and files with the package name as prefix
    possible_paths = glob.glob(os.path.join(site_packages, f"{package_name}*"))
    for path in possible_paths:
        basename = os.path.basename(path)
        if basename == package_name or basename.startswith(f"{package_name}-") or basename.startswith(f"{package_name}."):
            if os.path.isdir(path):
                # Return the directory if it contains __init__.py
                if os.path.exists(os.path.join(path, "__init__.py")):
                    return path
            else:
                # Return file if it matches the package name exactly
                if basename == f"{package_name}.py":
                    return path
    
    return None

def copy_package(package_name, source_path, dest_dir):
    """Copy a package from source_path to dest_dir."""
    dest_path = os.path.join(dest_dir, f"{package_name}_copy")
    
    # Create the destination directory
    os.makedirs(dest_path, exist_ok=True)
    
    # Determine if we're copying a directory or a file
    if os.path.isdir(source_path):
        # Copy the directory recursively
        print(f"Copying directory {source_path} to {dest_path}")
        
        # Remove previous copy if it exists
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)
            
        # Copy the directory
        shutil.copytree(source_path, dest_path, symlinks=True)
    else:
        # Copy the file
        print(f"Copying file {source_path} to {dest_path}")
        
        # Extract the filename
        filename = os.path.basename(source_path)
        
        # Copy the file
        shutil.copy2(source_path, os.path.join(dest_path, filename))
    
    return dest_path

def main():
    """Main function to copy dependencies."""
    parser = argparse.ArgumentParser(description="Copy dependencies for PyInstaller packaging")
    parser.add_argument("--venv", help="Path to the virtual environment")
    args = parser.parse_args()
    
    if args.venv:
        # Use the provided virtual environment path
        if sys.platform == 'win32':
            site_packages = os.path.join(args.venv, 'Lib', 'site-packages')
        else:
            # Try to find python version in the venv's bin directory
            python_path = list(Path(args.venv).glob('bin/python*'))
            if python_path:
                python_ver = os.path.basename(str(python_path[0]))
                if python_ver.startswith('python'):
                    python_ver = python_ver[6:]  # Remove "python" prefix
                    site_packages = os.path.join(args.venv, 'lib', f'python{python_ver}', 'site-packages')
                else:
                    # Default to the current Python version
                    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
                    site_packages = os.path.join(args.venv, 'lib', f'python{python_version}', 'site-packages')
            else:
                # Default to the current Python version
                python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
                site_packages = os.path.join(args.venv, 'lib', f'python{python_version}', 'site-packages')
    else:
        # Find the site-packages directory automatically
        site_packages = find_venv_site_packages()
    
    if not os.path.exists(site_packages):
        print(f"Error: Site packages directory not found: {site_packages}")
        sys.exit(1)
    
    print(f"Using site-packages directory: {site_packages}")
    
    # Get the current directory as the destination for copies
    current_dir = os.getcwd()
    
    # Copy directory packages
    for package_name in DIR_PACKAGES:
        print(f"\nProcessing package: {package_name}")
        
        # Try to get the package location by importing it
        package_path = get_package_location(package_name)
        
        # If we couldn't find it by importing, try finding it in site-packages
        if not is_valid_path(package_path):
            package_path = find_package_in_site_packages(site_packages, package_name)
        
        if is_valid_path(package_path):
            copy_package(package_name, package_path, current_dir)
        else:
            print(f"Warning: Could not find package {package_name}")
    
    # Copy file packages
    for package_name in FILE_PACKAGES:
        print(f"\nProcessing file package: {package_name}")
        
        # Try to get the package location by importing it
        package_path = get_package_location(package_name)
        
        # If we couldn't find it by importing, try finding it in site-packages
        if not is_valid_path(package_path):
            package_path = find_package_in_site_packages(site_packages, package_name)
        
        if is_valid_path(package_path):
            copy_package(package_name, package_path, current_dir)
        else:
            print(f"Warning: Could not find package {package_name}")
    
    print("\nDependency copying completed!")
    print("To package the application with PyInstaller, use:")
    print("pyinstaller --clean --noconfirm main.spec")

if __name__ == "__main__":
    main() 