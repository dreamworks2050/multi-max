# Multi-Max for Windows

This directory contains the Windows-specific version of Multi-Max, adapted from the Mac-specific original.

## Key Differences

1. **Software Rendering**: The Windows version uses software rendering instead of Mac-specific hardware acceleration (Quartz/CoreImage).
2. **No Mac Dependencies**: All Mac-specific dependencies (Quartz, objc, etc.) have been removed.
3. **Simplified Image Processing**: Image processing functions have been reimplemented using OpenCV.

## Installation

### Automatic Installation (Recommended)

The Windows version is automatically installed when you run:

1. **Using the Batch File**: Run `windows\Install-Windows.bat` as Administrator
2. **Using PowerShell**: Run `install-multi-max.ps1` in PowerShell as Administrator

The installer will:
1. Detect that you're running on Windows
2. Copy the Windows-specific environment configuration
3. Back up the original Mac version (if exists)
4. Install the Windows-specific version

### Verifying Your Installation

To verify you have the correct Windows version installed:

1. Run `windows\check_windows_version.ps1` in PowerShell
2. The script will check if:
   - The main.py file contains the Windows-specific code
   - The environment settings are configured for Windows
   - It can fix any issues automatically if desired

## Features and Limitations

Most core features of Multi-Max work the same on Windows, including:
- Video playback and capture
- Grid layout and recursion
- Frame buffer management
- YouTube stream connectivity

However, some limitations exist:
- Performance may be slower than the Mac version with hardware acceleration
- Some advanced image filters may be simplified
- Memory usage may be higher due to software processing

## Troubleshooting

If you encounter issues:

1. First run the version check script: `windows\check_windows_version.ps1`

2. Make sure your `.env` file has the following settings:
   ```
   FORCE_HARDWARE_ACCELERATION=false
   ALLOW_SOFTWARE_FALLBACK=true
   WINDOWS_MODE=true
   USE_SOFTWARE_RENDERING=true
   ```

3. If you get "module not found" errors related to Mac-specific libraries:
   - This means you're running the Mac version on Windows
   - Run `windows\Install-Windows.bat` to install the Windows version
   
4. If you still have issues after using the Windows installer:
   - Check that the `windows` directory exists in your repository
   - Make sure it contains `main.py`, `.env`, and `requirements.txt`
   - If these files are missing, the repository may not include Windows support yet

## For Developers

When committing changes to this project:

1. Keep the Windows-specific code in the `windows` directory
2. Do not remove the `__windows_specific_version__` marker from `windows/main.py`
3. Make sure the Windows installer properly detects and uses the Windows-specific code
4. Test on both platforms before committing changes

## Contributing

When contributing code that should work on both platforms:
1. Place Mac-specific code under platform checks: `if platform.system() == 'Darwin'`
2. Place Windows-specific code under: `if platform.system() == 'Windows'`
3. Add any new dependencies to both requirements.txt files 