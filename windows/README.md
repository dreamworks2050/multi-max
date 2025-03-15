# Multi-Max for Windows

This directory contains the Windows-specific version of Multi-Max, adapted from the Mac-specific original.

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