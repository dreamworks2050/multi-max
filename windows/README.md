# Multi-Max for Windows

## Simplified Installation and Update Guide

This guide provides instructions for installing and running Multi-Max on Windows with our new simplified scripts. These scripts address common issues users have experienced with path resolution, Python environments, and Git-based updates.

## Prerequisites

- **Python 3.7 or higher** - [Download Python](https://www.python.org/downloads/)
- **Git** (optional, required for updates) - [Download Git](https://git-scm.com/download/win)

## Quick Start

1. **Download** or clone the Multi-Max repository
2. **Run `Simple-Install.bat`** to set up the environment
3. **Run `Simple-Run.bat`** to start the application

## Installation

### Option 1: Simplified Installation (Recommended)

1. Run the `Simple-Install.bat` script from the `windows` directory.
2. The script will:
   - Check for Python and Git
   - Create a virtual environment
   - Install required packages
   - Set up the necessary directories and files

### Option 2: Manual Installation

If the simplified installation doesn't work for you:

1. Install Python 3.7 or higher
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment: `venv\Scripts\activate`
4. Install dependencies: `pip install numpy opencv-python python-dotenv`
5. Ensure `main.py` is in the parent directory of the `windows` folder

## Running Multi-Max

### Option 1: Using the Simple Launcher (Recommended)

Run the `Simple-Run.bat` script from the `windows` directory. This script will:
- Ensure all paths are correctly set
- Activate the virtual environment if it exists
- Check for updates using the simplified update checker
- Run the main application

### Option 2: Manual Execution

If you prefer to run the application manually:
1. Activate your virtual environment: `venv\Scripts\activate`
2. Change to the parent directory of the `windows` folder
3. Run: `python main.py`

## Update System

Multi-Max includes a simplified update system that works reliably on Windows. 

### How Updates Work

1. The launcher checks for updates when you start the application
2. If updates are available, you'll be prompted to install them
3. After updating, the application will restart automatically

### Manual Update

You can manually check for updates:

1. Open a command prompt in the `windows` directory
2. Run: `python simple_update_checker.py --auto-update`

### Troubleshooting Updates

If updates aren't working:

1. Make sure Git is installed and in your PATH
2. Ensure your Multi-Max installation is a Git repository
3. Check permissions for the repository directory

## Common Issues and Solutions

### Application doesn't start

- Check that Python is installed and in your PATH
- Ensure all required packages are installed
- Check the logs directory for error messages

### "Main script not found" error

- Make sure `main.py` exists in the parent directory of the `windows` folder
- If not, copy it from the `windows` directory or reinstall Multi-Max

### Update checker not working

- Verify Git is installed and in your PATH
- Ensure you have an internet connection
- Check if your repository is properly configured

### Missing dependencies

- Run the `Simple-Install.bat` script again
- Manually install the required packages in your virtual environment

## Support

If you encounter any issues:

1. Check the logs in the `logs` directory
2. Run the `Simple-Run.bat` script with the `--debug` flag
3. Contact support with the log files and error messages

## Advanced Configuration

### VERSION File

The `VERSION` file in the parent directory stores the current version number. This file is used by the update checker to determine if updates are available.

### Update Checker Options

The simplified update checker supports the following options:

- `--auto-update`: Automatically install updates if available
- `--version`: Display version information

You can add these options to the `Simple-Run.bat` command line.

## For Developers

### File Structure

- `windows/`: Contains Windows-specific scripts and utilities
  - `Simple-Install.bat`: Sets up the environment
  - `Simple-Run.bat`: Launches the application
  - `simple_update_checker.py`: Handles update checking and installation
- Parent directory: Contains the main application code

### How the Update System Works

The update system uses Git to check for and apply updates:

1. The system checks if Git is available
2. It verifies that we're in a Git repository
3. It uses `git fetch` to check for remote changes
4. It compares the local and remote branches
5. If updates are available, it can automatically pull them

---

## Version History

- **1.0.0**: Initial simplified Windows release

## Key Differences

1. **Software Rendering**: The Windows version uses software rendering instead of Mac-specific hardware acceleration (Quartz/CoreImage).
2. **No Mac Dependencies**: All Mac-specific dependencies (Quartz, objc, etc.) have been removed.
3. **Simplified Image Processing**: Image processing functions have been reimplemented using OpenCV.
4. **Windows-Specific Path Handling**: Improved file and directory handling for Windows environments.
5. **Logging System**: Automatic logging to files in the logs directory.

## System Requirements

- Windows 8.1 or newer (Windows 10/11 recommended)
- Python 3.6 or newer (Python 3.9+ recommended)
- FFmpeg (for video processing)
- yt-dlp (for YouTube stream connectivity)
- 4GB RAM minimum (8GB+ recommended for higher resolution videos)
- 100MB free disk space (plus space for logs)

## Installation

### Automatic Installation (Recommended)

The Windows version is automatically installed when you run:

1. **Using the Batch File**: Run `windows\Install-Windows.bat` as Administrator
   - Right-click on the file and select "Run as administrator"
   - This is the recommended method and handles all dependencies

2. **Using PowerShell**: Run `install-multi-max.ps1` in PowerShell as Administrator
   - Open PowerShell as Administrator
   - Navigate to the repository directory
   - Run `.\install-multi-max.ps1`

The installer will:
1. Detect that you're running on Windows
2. Check for required dependencies and offer to install them
3. Back up the original Mac version (if it exists)
4. Install the Windows-specific version
5. Create necessary directories for logs
6. Offer to create a desktop shortcut

### Manual Installation

If the automatic installer doesn't work, you can manually install:

1. Copy `windows\main.py` to the parent directory, replacing the existing `main.py`
2. Copy `windows\.env` to the parent directory
3. Install dependencies with: `pip install -r windows\requirements.txt`
4. Ensure FFmpeg is installed and in your PATH

### Verifying Your Installation

To verify you have the correct Windows version installed:

1. Run `windows\check_windows_version.ps1` in PowerShell
2. The script will check if:
   - The main.py file contains the Windows-specific code
   - The environment settings are correctly configured for Windows
   - It can fix any issues automatically if desired

## Running Multi-Max

### Quick Start

1. Run `windows\Run-MultiMax.bat` to start the application
2. The launcher will perform necessary checks and start Multi-Max
3. Logs will be saved to the `logs` directory

### Command Line Options

You can also run directly from the command line:
```
python main.py --grid-size=3 --depth=1 --mode=fractal_depth
```

See all available options with:
```
python main.py --help
```

## Features and Limitations

Most core features of Multi-Max work the same on Windows, including:
- Video playback and capture
- Grid layout and recursion
- Frame buffer management
- YouTube stream connectivity
- Automatic logging

However, some limitations exist:
- Performance may be slower than the Mac version with hardware acceleration
- Some advanced image filters may be simplified
- Memory usage may be higher due to software processing

## Logging and Troubleshooting

Multi-Max on Windows saves log files to the `logs` directory with timestamps.
These logs can help diagnose issues if something goes wrong.

If you encounter issues:

1. First run the version check script: `windows\check_windows_version.ps1`

2. Make sure your `.env` file has the following settings:
   ```
   FORCE_HARDWARE_ACCELERATION=false
   ALLOW_SOFTWARE_FALLBACK=true
   WINDOWS_MODE=true
   USE_SOFTWARE_RENDERING=true
   ```

3. Check if FFmpeg is installed and in your PATH:
   - Run `where ffmpeg` in Command Prompt
   - If not found, download from https://ffmpeg.org/download.html

4. If you get "module not found" errors:
   - Run `pip install -r windows\requirements.txt` to install dependencies
   - Make sure you're using the Windows-specific version

5. Check the log files in the `logs` directory for specific error messages
   
6. If you still have issues after using the Windows installer:
   - Check that the `windows` directory exists in your repository
   - Make sure it contains `main.py`, `.env`, and `requirements.txt`
   - If these files are missing, the repository may not include Windows support yet

## For Developers

When committing changes to this project:

1. Keep the Windows-specific code in the `windows` directory
2. Do not remove the `__windows_specific_version__` marker from `windows/main.py`
3. Make sure the Windows installer properly detects and uses the Windows-specific code
4. Test path handling with spaces and special characters on Windows
5. Test on both platforms before committing changes

## Contributing

When contributing code that should work on both platforms:
1. Place Mac-specific code under platform checks: `if platform.system() == 'Darwin'`
2. Place Windows-specific code under: `if platform.system() == 'Windows'`
3. Add any new dependencies to both requirements.txt files
4. Use cross-platform path handling with `os.path.join()` instead of string concatenation
5. Use forward slashes in paths to ensure cross-platform compatibility

## Update Mechanism

Multi-Max includes an automatic update feature that checks for updates when the application starts. This ensures you're always running the latest version with all bug fixes and new features.

### How It Works

1. **Automatic Update Check**: Each time you start Multi-Max, it checks if a newer version is available in the GitHub repository.
2. **Version Comparison**: The local version is compared with the remote version to determine if an update is needed.
3. **Automatic Updates**: If a newer version is found, Multi-Max can automatically update itself.
4. **Restart Required**: After an update, you'll need to restart the application to use the new version.

### Command Line Options

The following command line options are available for controlling the update process:

- `--skip-update-check`: Skip checking for updates at startup
- `--force-update`: Force an update check and automatically update if available
- `--version`: Display the current version information and exit
- `--update`: Run the update checker directly (via Run-MultiMax.bat)

### Manual Update

You can manually trigger an update check by running:

```
Run-MultiMax.bat --update
```

### Update Configuration

Multi-Max's update behavior is designed to be non-intrusive:

1. It only checks for updates on the `main` or `master` branch
2. Updates are only performed when explicitly requested or when auto-update is enabled
3. Local changes are preserved using git stash before updating

## Version Information

The version number follows [Semantic Versioning](https://semver.org/) format: `MAJOR.MINOR.PATCH`

- **MAJOR**: Incremented for incompatible API changes
- **MINOR**: Incremented for new features in a backward-compatible manner
- **PATCH**: Incremented for backward-compatible bug fixes

The current version is defined in:
1. `VERSION` file in the root directory
2. `windows/update_checker.py` - contains both VERSION and BUILD_DATE

## Troubleshooting

If you encounter issues with the update process:

1. **Failed Updates**: If an update fails, check your internet connection and ensure you have write permissions to the application directory.
2. **Offline Use**: Use the `--skip-update-check` flag to bypass update checking when you're offline.
3. **Manual Update**: If automatic updates fail, you can manually pull the latest changes using Git:
   ```
   git pull origin main
   ```
4. **Logs**: Check the log files in the `logs` directory for detailed information about update attempts.

## For Developers

When releasing a new version:

1. Update the `VERSION` file in the root directory
2. Update the `VERSION` and `BUILD_DATE` in `windows/update_checker.py`
3. Commit these changes with a message like "Bump version to X.Y.Z"
4. Push the changes to the main branch

Users will receive the update when they next start Multi-Max. 