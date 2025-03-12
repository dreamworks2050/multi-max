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

## Step 3: Build the Application

Use PyInstaller to build the application:

```bash
pyinstaller --clean --noconfirm main.spec
```

The packaged application will be created in the `dist` directory.

## Step 4: Test the Application

Test the packaged application to make sure it works:

```bash
# On macOS/Linux
./dist/multi-max

# On Windows
dist\multi-max.exe
```

## Creating a macOS App Bundle (Optional)

To create a macOS app bundle, uncomment the `app = BUNDLE(...)` section in the `main.spec` file, and run PyInstaller again.

## Troubleshooting

If you encounter any issues during packaging:

1. Check the PyInstaller warning file: `build/main/warn-main.txt`
2. Run PyInstaller with the `--debug` flag for more verbose output:
   ```bash
   pyinstaller --clean --debug main.spec
   ```
3. If specific modules are missing, add them to the `hiddenimports` list in `main.spec`
4. For more complex issues, try building a minimal test application first:
   ```python
   # test_imports.py
   import sys
   print(f"Python version: {sys.version}")
   print(f"Executable: {sys.executable}")
   
   # Test the problematic import
   import cv2
   print(f"OpenCV version: {cv2.__version__}")
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