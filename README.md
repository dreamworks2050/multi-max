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

## Installation

For easy installation on any system, use the included installer script:

```bash
# Make the installer executable (if needed)
chmod +x installer.sh

# Run the installer
./installer.sh
```

The installer will:
1. Detect your operating system and hardware
2. Install Homebrew (on macOS)
3. Install Python
4. Install FFmpeg
5. Set up a Python virtual environment
6. Install all required dependencies
7. Configure your environment (.env file)

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