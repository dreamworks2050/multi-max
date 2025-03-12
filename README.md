# Multi-Max

A powerful video processing application that provides real-time infinite fractal recursion, optimized specifically for Apple Silicon (M1/M2/M3) hardware.

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

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/multi-max.git
   cd multi-max
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your video source URL:
   ```
   YOUTUBE_URL=https://www.youtube.com/watch?v=your_video_id
   ```

5. Run the application:
   ```
   python main.py
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