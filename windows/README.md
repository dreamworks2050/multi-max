# Multi-Max for Windows

This directory contains the Windows-specific version of Multi-Max, adapted from the Mac-specific original.

## File Organization

The Windows version of Multi-Max follows this organization structure:

- **All Windows-specific code** is now contained exclusively within the `windows/` directory
- **Configuration files** like `.env` are stored in the `windows/` directory
- **Version information** is stored in `windows/version.txt`
- **Launcher scripts** (`Run-MultiMax.bat`) are provided both in the base directory and in the `windows/` directory

## Key Differences from Mac Version

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

## Key Files

- `main.py` - The main application entry point
- `update.py` - Handles the update process
- `Run-MultiMax.bat` - Script to launch the application
- `Update-MultiMax.bat` - Script to update the application
- `Install-Windows.bat` - Script to install dependencies
- `version.txt` - Contains the current version number
- `requirements.txt` - List of Python dependencies
- `.env` - Environment configuration file (create this file if it doesn't exist)

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
3. Install Windows-specific version in the `windows/` directory
4. Create necessary directories for logs
5. Offer to create a desktop shortcut

### Manual Installation

If the automatic installer doesn't work, you can manually install:

1. Ensure all files are in the `windows/` directory
2. Install dependencies with: `pip install -r windows\requirements.txt`
3. Ensure FFmpeg is installed and in your PATH

## Update Process

The update process works as follows:

1. When a new version is detected, the application will display an update notification
2. When you confirm the update:
   - The application will exit
   - The update script (`Update-MultiMax.bat`) will be launched
   - The script will download the latest Windows-specific files from GitHub
   - It will then update all files in the `windows/` directory
   - Finally, it will restart the application

## Manual Update

If you need to manually update:

1. Close Multi-Max if it's running
2. Run `Update-MultiMax.bat` either from the root directory or the `windows/` directory
3. The update process will run automatically
4. After updating, the application will restart

## Running Multi-Max

### Quick Start

1. Run `windows\Run-MultiMax.bat` or the root `Run-MultiMax.bat` to start the application
2. The launcher will perform necessary checks and start Multi-Max
3. Logs will be saved to the `logs` directory

### Command Line Options

You can also run directly from the command line:
```
cd windows
python main.py --grid-size=3 --depth=1 --mode=fractal_depth
```

See all available options with:
```
python main.py --help
```

## Troubleshooting

If you encounter issues:

1. **Update fails**: Try running `Update-MultiMax.bat` directly as administrator
2. **Application crashes**: Check the logs in the `windows/logs/` directory
3. **Dependencies missing**: Run `Install-Windows.bat` to reinstall dependencies
4. **Version issues**: Delete `version.txt` and restart the application
5. **FFmpeg issues**: Check if FFmpeg is installed and in your PATH:
   - Run `where ffmpeg` in Command Prompt
   - If not found, download from https://ffmpeg.org/download.html
6. **Module errors**: Run `pip install -r windows\requirements.txt` to install dependencies
7. **Environment settings**: Make sure your `.env` file has the following settings:
   ```
   FORCE_HARDWARE_ACCELERATION=false
   ALLOW_SOFTWARE_FALLBACK=true
   WINDOWS_MODE=true
   USE_SOFTWARE_RENDERING=true
   ```

## For Developers

When working on Multi-Max:

1. Keep the Windows-specific code in the `windows/` directory
2. Do not remove the `__windows_specific_version__` marker from `windows/main.py`
3. Update scripts should respect the new file organization
4. Use `os.path.dirname(os.path.abspath(__file__))` to get the Windows directory path
5. Test path handling with spaces and special characters on Windows
6. Test on both platforms before committing changes
7. Place Mac-specific code under platform checks: `if platform.system() == 'Darwin'`
8. Place Windows-specific code under: `if platform.system() == 'Windows'`
9. Add any new dependencies to both requirements.txt files
10. Use cross-platform path handling with `os.path.join()` instead of string concatenation

## File Transfer Between Installs

If you're transferring from a previous installation:

1. Copy your `.env` file to the `windows/` directory
2. Any custom configurations should go in the `windows/` directory
3. The update process will automatically back up your configuration files

## Reporting Issues

If you encounter issues with the Windows version, please report them on GitHub with:

1. The contents of your update log (found in `windows/update-log.txt`)
2. The contents of your application log (found in `windows/logs/`)
3. A description of what you were doing when the issue occurred 