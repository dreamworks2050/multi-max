# Multi-Max Simplified Windows Installation Guide

## Overview

This guide provides instructions for using the simplified installation process for Multi-Max on Windows, which addresses the most common issues encountered with the standard installation method.

## Quick Start

```powershell
# Run this in PowerShell to install Multi-Max with the simplified installer:
irm -useb https://raw.githubusercontent.com/dreamworks2050/multi-max/main/windows/install-simple-wrapper.ps1 | iex
```

## Common Issues Fixed by This Installation Method

The simplified installation addresses several common issues:

1. **Path Resolution Problems**: Ensures `main.py` is correctly located and accessible from the right directory.
2. **Update Checker Issues**: Uses a more reliable update checker that works better on Windows.
3. **Missing Dependencies**: Properly installs and configures dependencies within the correct Python environment.
4. **Virtual Environment Problems**: Creates and configures the virtual environment correctly.
5. **Windows-Specific Code**: Ensures the Windows-specific version of the code is properly installed.

## Manual Installation Method

If the quick start method doesn't work for you, follow these steps for a manual installation:

1. Download the repository as a ZIP file or clone it using Git:
   ```
   git clone https://github.com/dreamworks2050/multi-max.git
   ```

2. Navigate to the `windows` directory:
   ```
   cd multi-max\windows
   ```

3. Run the simple installation batch file:
   ```
   Simple-Install.bat
   ```

4. Launch the application using the simple run batch file:
   ```
   Simple-Run.bat
   ```

## Fixing an Existing Installation

If you've already installed Multi-Max but are experiencing issues, you can run the fix script:

1. Navigate to the `windows` directory in your Multi-Max installation:
   ```
   cd path\to\multi-max\windows
   ```

2. Run the fix script:
   ```powershell
   # In PowerShell:
   .\fix-installation.ps1

   # OR in Command Prompt:
   fix-installation.bat
   ```

## Directory Structure

After installation, your Multi-Max directory should look like this:

```
multi-max/
├── main.py             # Windows-specific main script
├── update_checker.py   # Simplified update checker
├── VERSION             # Version information file
├── logs/               # Log directory
├── venv/               # Python virtual environment
└── windows/            # Windows-specific utilities
    ├── Simple-Install.bat      # Simplified installation script
    ├── Simple-Run.bat          # Simplified launcher
    ├── fix-installation.ps1    # Installation repair tool
    └── simple_update_checker.py # Source for simplified update checker
```

## Troubleshooting

### "Main script not found" Error

This usually happens when the main.py file is in the wrong location or the path resolution isn't working correctly. The simplified installation ensures the script is in the right place.

### Missing Dependencies

If you see errors about missing dependencies (like `opencv-python` or `python-dotenv`), the fix script will correctly install these in the virtual environment and make sure the environment is activated when running the application.

### Update Checker Issues

The simplified update checker is more reliable on Windows systems and provides better error messages when something goes wrong.

### Python Path Problems

The simplified scripts ensure that the Python path includes the necessary directories to find all modules.

## For Developers

### How the Simplified Installation Works

1. **Environment Setup**: Creates a proper virtual environment and installs all dependencies
2. **Path Configuration**: Ensures correct path resolution by configuring PYTHONPATH
3. **File Management**: Copies key files to the correct locations
4. **Reliability**: Uses more reliable methods for Windows path handling and dependencies

### Modifying the Installation Scripts

If you need to modify the installation scripts, the main files to edit are:

- `Simple-Install.bat`: The main installation script
- `Simple-Run.bat`: The launcher script
- `simple_update_checker.py`: The simplified update checker
- `fix-installation.ps1`: The repair tool

## Support

If you encounter issues with the simplified installation:

1. Check the logs in the `logs` directory
2. Run the fix script to repair common issues
3. Report any persistent problems with detailed logs and error messages

---

## Version History

- **1.0.0**: Initial simplified Windows installation method 