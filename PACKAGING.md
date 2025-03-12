# Packaging Multi-Max with PyInstaller

This guide explains how to package the Multi-Max application into a standalone executable using PyInstaller.

## Prerequisites

- Python 3.9+ with a virtual environment
- PyInstaller installed: `pip install pyinstaller`
- All required dependencies installed in your virtual environment:
  - `opencv-python`
  - `numpy`
  - `pygame`
  - `python-dotenv`
  - `psutil`
  - `memory-profiler`
  - `pyobjc` (on macOS)

## Step 1: Copy Dependencies

First, run the `copy_dependencies.py` script to copy all required dependencies from your virtual environment to local directories:

```bash
# Make sure your virtual environment is activated
python copy_dependencies.py
```

If your virtual environment is in a non-standard location, you can specify it:

```bash
python copy_dependencies.py --venv /path/to/your/venv
```

This script will create the following directories in your project:
- `cv2_copy`
- `numpy_copy`
- `pygame_copy`
- `dotenv_copy`
- `psutil_copy`
- `memory_profiler_copy`
- `Quartz_copy` (on macOS)
- `Cocoa_copy` (on macOS)
- `objc_copy` (on macOS)
- `AppKit_copy` (on macOS)
- `Foundation_copy` (on macOS)
- `CoreFoundation_copy` (on macOS)
- `PyObjCTools_copy` (on macOS)

## Step 2: Use the Spec File

Copy the template spec file to the main spec file:

```bash
cp main.spec.template main.spec
```

## Step 3: Include Configuration Files

Make sure your `.env` file is set up with the required configuration:

```
YOUTUBE_URL=https://www.youtube.com/watch?v=your_video_id
```

The `.env` file is now included in the `datas` section of the `main.spec` file:

```python
datas=[
    # ... other dependencies ...
    ('.env', '.'),  # Include the .env file in the root directory
],
```

If you need to add other configuration files, add them to the `datas` list in the same way.

## Step 4: Build the Application

Use PyInstaller to build the application:

```bash
pyinstaller --clean --noconfirm main.spec
```

The packaged application will be created in the `dist` directory.

> **Note About Warnings**: During the build process, PyInstaller may show warnings about "missing modules" in the console output. This is normal and expected. The dependencies are actually being included through the `datas` section in the spec file, not through the standard Python import system. As long as the build completes successfully, these warnings can be safely ignored.

## Step 5: Test the Application

Test the packaged application to make sure it works:

```bash
# On macOS/Linux
./dist/multi-max

# On Windows
dist\multi-max.exe
```

## Creating a macOS App Bundle (Optional)

To create a macOS app bundle, uncomment the `app = BUNDLE(...)` section in the `main.spec` file, and run PyInstaller again:

```bash
# First, build the executable
pyinstaller --clean --noconfirm main.spec

# Check that the bundled app exists
ls -la dist/multi-max
```

After confirming the executable works, edit the spec file to uncomment the BUNDLE section:

```python
app = BUNDLE(
    exe,
    name='Multi-Max.app',
    icon=None,  # Add path to icon file here
    bundle_identifier='com.yourdomain.multi-max',
    info_plist={
        'CFBundleName': 'Multi-Max',
        'CFBundleDisplayName': 'Multi-Max',
        'CFBundleVersion': '1.0',
        'CFBundleShortVersionString': '1.0',
        'NSHighResolutionCapable': 'True',
    },
)
```

Then run PyInstaller again to create the app bundle:

```bash
pyinstaller --clean --noconfirm main.spec

# The app will be available at dist/Multi-Max.app
```

## Understanding the Spec File

The `main.spec` file is configured to package Multi-Max properly:

1. **Data Files**: The `datas` section includes all copied dependencies.
2. **Hidden Imports**: The `hiddenimports` list includes modules that PyInstaller might not detect.
3. **Runtime Hooks**: Ensures proper initialization of packages.

If you need to add additional dependencies, add them to:
- `copy_dependencies.py` to copy the files
- The `datas` section in `main.spec` to include them in the package
- The `hiddenimports` list in `main.spec` if needed

## Troubleshooting

If you encounter any issues during packaging:

1. **Missing Modules**: If PyInstaller reports "missing module" warnings:
   - Confirm all dependencies are properly copied with `copy_dependencies.py`
   - Check that the module names match exactly in the `datas` section
   - These warnings often don't affect the final executable if the files are properly included

2. **Runtime Errors**: If the packaged application fails to run:
   - Check the PyInstaller warning file: `build/main/warn-main.txt`
   - Run PyInstaller with the `--debug` flag for more verbose output:
     ```bash
     pyinstaller --clean --debug main.spec
     ```
   - Try running with the `--log-level=DEBUG` flag to see detailed import errors:
     ```bash
     ./dist/multi-max --log-level=DEBUG
     ```

3. **Missing Configuration Files**: If the application reports missing configuration files:
   - Ensure the files are properly included in the `datas` section of the spec file
   - For the `.env` file, check that it contains the required variables
   - Consider adding a fallback mechanism in your code for when configuration files are missing

4. **Testing Individual Components**: Create simple test scripts to isolate issues:
   ```python
   # test_imports.py
   import sys
   print(f"Python version: {sys.version}")
   print(f"Executable: {sys.executable}")
   
   # Test specific imports
   import cv2
   print(f"OpenCV version: {cv2.__version__}")
   
   import numpy as np
   print(f"NumPy version: {np.__version__}")
   
   import pygame
   print(f"Pygame version: {pygame.__version__}")
   ```

## Dynamic Libraries and Binary Files

PyInstaller might not correctly detect some native libraries or binaries. If you encounter errors about missing libraries, you may need to:

1. Add the libraries to the `binaries` list in the spec file
2. Use tools like `otool -L` (macOS) or `ldd` (Linux) to check library dependencies
3. Create a custom hook file for specific packages

## Platform-Specific Notes

### macOS

- The packaging script automatically handles PyObjC frameworks, which are required on macOS
- Enable argv_emulation in the spec file for proper command-line argument handling
- For code signing, see PyInstaller's documentation on `codesign_identity`

### Windows

- You might need to include additional DLLs for some libraries
- Use the `--noconsole` flag to create a GUI-only application

### Linux

- Additional system libraries might need to be included
- Consider using AppImage or Snap for broader compatibility

## Current Status

The Multi-Max application has been successfully packaged into both:

1. A standalone executable (`dist/multi-max`)
2. A macOS app bundle (`dist/Multi-Max.app`)

### Next Steps

1. **Test the Application**: Verify both the standalone executable and app bundle work correctly.
   ```bash
   # Test standalone executable
   ./dist/multi-max
   
   # Test app bundle (from Finder or command line)
   open ./dist/Multi-Max.app
   ```

2. **Distribution**: To distribute the application:
   - For direct sharing: Compress the app bundle into a zip file
   - For wider distribution: Create a DMG file using Disk Utility
   - For web distribution: Consider notarizing the app for macOS Gatekeeper

3. **Improvements**:
   - Add a custom icon to the app bundle
   - Customize the Info.plist for better macOS integration
   - Add a proper version numbering system

If you encounter any issues with the packaged application, refer to the troubleshooting section above. 