# Multi-Max: Recursive Video Grid

A Python application for creating recursive video grid effects with FFmpeg and OpenCV.

## Features

- **Infinite Fractal Recursion**: Create mesmerizing visual effects with real-time video recursion
- **Multiple Display Modes**:
  - **Grid Mode**: Simple grid layout with adjustable size and recursion depth
  - **Fractal Mode**: Infinite recursive grid patterns with configurable source position
  - **Fractal Depth Mode**: Advanced infinite recursion with precise depth control
- **Hardware Acceleration**: Optimized for Apple Silicon chips with Metal performance
- **Memory Optimization**: Advanced memory management for deep recursion levels
- **Customizable Grid Layouts**: Adjust grid size on-the-fly
- **Real-time Processing**: Maintains smooth performance even at high recursion levels

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical details about the project structure and implementation
- [PACKAGING.md](PACKAGING.md) - Comprehensive guide for packaging the application

## Requirements

- macOS 11.0 or later
- Apple Silicon (M1/M2/M3) for optimal performance
- Python 3.9+
- Dependencies listed in `requirements.txt`

## Installation Instructions

### Windows Installation

You can install Multi-Max on Windows with a single command. The installer will automatically download and install all required dependencies.

#### One-Line Web Installer (Recommended)

1. Open PowerShell or Command Prompt **as Administrator** (right-click and select "Run as administrator")
2. Copy and paste the following command:

```powershell
powershell -ExecutionPolicy Bypass -Command "iwr -useb https://raw.githubusercontent.com/dreamworks2050/multi-max/main/install-multi-max.ps1 | iex"
```

3. Press Enter and follow the on-screen instructions

#### What the Installer Does

The Windows installer will:

1. Install Chocolatey (Windows package manager) if not already installed
2. Install Git, Python, and FFmpeg
3. Clone this repository to your user folder
4. Create a virtual environment named "multi-max"
5. Install the UV package manager for faster dependency installation
6. Install all required Python packages
7. Configure the application
8. Create desktop shortcuts for easy access

#### Manual Installation (Alternative)

If the one-line installer doesn't work, you can install manually:

1. Install [Python](https://www.python.org/downloads/) (3.8 or later)
2. Install [Git](https://git-scm.com/download/win)
3. Install [FFmpeg](https://ffmpeg.org/download.html) and add it to your PATH
4. Open Command Prompt as Administrator and run:

```cmd
git clone https://github.com/dreamworks2050/multi-max.git
cd multi-max
python -m venv multi-max
multi-max\Scripts\activate
pip install uv
uv pip install -r requirements.txt
```

5. Create a .env file or copy from .env.template if available

### macOS Installation

For macOS users, run the installer script:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/dreamworks2050/multi-max/main/installer.sh)"
```

## Configuration

The application uses a `.env` file for configuration. During installation, a default configuration is created based on your system.

Key settings:
- `FORCE_HARDWARE_ACCELERATION`: Enable/disable hardware acceleration (default: true on Apple Silicon)
- `ALLOW_SOFTWARE_FALLBACK`: Allow fallback to software rendering (default: false on Apple Silicon)
- `DEFAULT_VIDEO_URL`: Default YouTube URL to use
- `LOG_LEVEL`: Logging verbosity level
- `FRAME_BUFFER_SIZE`: Number of frames to buffer

To reset your configuration to defaults, delete the `.env` file and run the installer again:

```bash
rm .env
./installer.sh
```

## Hardware Acceleration

On Apple Silicon (M1/M2/M3) Macs, hardware acceleration is enabled by default for improved performance. If you experience issues, edit the `.env` file and set:

```
FORCE_HARDWARE_ACCELERATION=false
ALLOW_SOFTWARE_FALLBACK=true
```

## Running the Application

After installation:

```bash
# Activate the virtual environment
source venv/bin/activate

# Run the application
python main.py
```

For more options, run:

```bash
python main.py --help
```

## Usage

### GUI Controls

- **F Key**: Toggle between display modes (Grid → Fractal → Fractal Depth → Grid)
- **Up/Down Arrows**: Adjust grid size
- **Number Keys (1-9, 0=10)**:
  - In Grid Mode: Set recursion depth (1-10)
  - In Fractal Mode: Set source position (1=top-left, 2=center, 3=top-right)
  - In Fractal Depth Mode: Access by pressing 4 in Fractal mode, then use Up/Down arrows to adjust depth
- **D Key**: Toggle debug display
- **I Key**: Toggle information overlay
- **ESC Key**: Exit application
- **R Key**: Reload/refresh video stream
- **C Key**: Clear GPU memory caches (for hardware acceleration)
- **P Key**: Toggle performance monitoring

### Display Modes

1. **Grid Mode**: Simple grid layout with separate recursion instances
2. **Fractal Mode**: Creates infinite recursive patterns based on grid layout
3. **Fractal Depth Mode**: Precise control over recursion depth for advanced effects

### Performance Tips

- Lower grid sizes provide better performance
- Use hardware acceleration when available
- For deep recursion effects, use Fractal Depth mode with incremental depth increases
- Monitor the information overlay for performance metrics

## Packaging for Distribution

You can create a standalone macOS application using PyInstaller:

1. Run the dependency copy script:
   ```
   python copy_dependencies.py
   ```

2. Build the application:
   ```
   pyinstaller --clean --noconfirm main.spec
   ```

3. Find your packaged application in the `dist` directory

For more detailed packaging instructions, see [PACKAGING.md](PACKAGING.md).

## Future Plans

- Windows and Linux support
- Additional video effects and filters
- Custom effect plugins
- UI improvements and graphical controls
- Performance optimizations for non-Apple Silicon systems

## License

[MIT License](LICENSE)

## Acknowledgments

- OpenCV for image processing capabilities
- PyGame for display functionality
- Apple Metal framework for hardware acceleration

---

For architecture details and technical documentation, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Troubleshooting

### Windows Installation Issues

- **Execution Policy Error**: If you see errors about execution policy, try running:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```
  
- **UV Installation Fails**: If UV fails to install, the script will automatically fall back to using pip

- **FFmpeg Not Found**: Ensure FFmpeg is properly installed and added to your PATH

- **Virtual Environment Activation Fails**: You can manually activate the virtual environment:
  ```cmd
  cd %USERPROFILE%\multi-max
  multi-max\Scripts\activate.bat
  ```

### For More Help

If you encounter any issues during installation, please [open an issue](https://github.com/dreamworks2050/multi-max/issues) with details about the error. 