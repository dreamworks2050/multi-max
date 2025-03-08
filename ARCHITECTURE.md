# Multi-Max: Recursive Video Grid System Architecture

## Overview

Multi-Max is a Python application that creates a recursive grid effect on live video streams. The application takes a YouTube live stream as input, processes the video frames to create a recursive grid pattern, and displays the result in real-time. The effect creates a visual where each cell in the grid contains a miniature version of the entire previous frame, creating a fractal-like recursive pattern with a configurable depth.

## System Architecture Diagram

```
┌───────────────────────────────────────────────────────────────────────────┐
│                            Configuration Layer                             │
│                                                                           │
│  ┌─────────────┐                                    ┌─────────────────┐   │
│  │ .env File   │◄──────────────────────────────────┤ Command Line    │   │
│  │ (YOUTUBE_URL)│                                   │ Arguments       │   │
│  └─────────────┘                                    └─────────────────┘   │
└───────────┬───────────────────────────────────────────────┬───────────────┘
            │                                               │
            ▼                                               ▼
┌───────────────────────┐                       ┌───────────────────────────┐
│                       │                       │                           │
│  Video Acquisition    │                       │     User Interface        │
│  ┌─────────────────┐  │                       │  ┌─────────────────────┐  │
│  │ yt-dlp          │  │                       │  │ PyGame Window       │  │
│  │ (Stream URL)    │  │                       │  │                     │  │
│  └────────┬────────┘  │                       │  └──────────┬──────────┘  │
│           │           │                       │             │              │
│           ▼           │                       │             ▼              │
│  ┌─────────────────┐  │                       │  ┌─────────────────────┐  │
│  │ OpenCV          │  │                       │  │ Keyboard Controls   │  │
│  │ VideoCapture    │  │                       │  │ - Grid Size (↑/↓)   │  │
│  └────────┬────────┘  │                       │  │ - Depth (0-9)       │  │
│           │           │                       │  │ - Debug Mode (D)    │  │
└───────────┼───────────┘                       │  └──────────┬──────────┘  │
            │                                   └──────────────┼──────────────┘
            │                                                  │
            ▼                                                  │
┌───────────────────────────────────────────────────┐         │
│                                                   │         │
│              Processing Pipeline                  │         │
│  ┌─────────────────────────────────────────────┐ │         │
│  │                                             │ │         │
│  │  ┌─────────────────┐    ┌───────────────┐   │ │         │
│  │  │ Frame Capture   ├───►│ apply_grid_   │   │ │         │
│  │  │                 │    │ effect()      │   │ │◄────────┘
│  │  └─────────────────┘    │ (Depth = n)   │   │ │
│  │                         └───────┬───────┘   │ │
│  │                                 │           │ │
│  │                                 ▼           │ │
│  │                      ┌───────────────────┐  │ │
│  │                      │ generate_grid_    │  │ │
│  │                      │ frame()           │  │ │
│  │                      │ (Grid Size = m×m) │  │ │
│  │                      └─────────┬─────────┘  │ │
│  │                                │            │ │
│  │                                ▼            │ │
│  │                     ┌────────────────────┐  │ │
│  │                     │  Hardware vs.      │  │ │
│  │                     │  Software Path     │  │ │
│  │                     └──┬─────────────┬───┘  │ │
│  │                        │             │      │ │
│  │           ┌────────────┘             └──────┐ │
│  │           │                                 │ │
│  │  ┌────────▼─────────┐         ┌────────────▼─┐│
│  │  │ Core Image       │         │ OpenCV       ││
│  │  │ (Apple Silicon)  │         │ (Software)   ││
│  │  └────────┬─────────┘         └────────┬─────┘│
│  │           │                            │      │
│  │           └────────────┬───────────────┘      │
│  │                        │                      │
│  │                        ▼                      │
│  │              ┌───────────────────┐            │
│  │              │ Memory Management │            │
│  │              │ - GC Collection   │            │
│  │              │ - Buffer Cleanup  │            │
│  │              └─────────┬─────────┘            │
│  │                        │                      │
│  └────────────────────────┼──────────────────────┘
│                           │
│                           ▼
│  ┌─────────────────────────────────────────────┐
│  │                 Display System              │
│  │  ┌─────────────────┐    ┌───────────────┐   │
│  │  │ PyGame Surface  │    │ Information   │   │
│  │  │ Creation        ├───►│ Overlay       │   │
│  │  │                 │    │               │   │
│  │  └─────────────────┘    └───────────────┘   │
│  │                                             │
└──┼─────────────────────────────────────────────┘
   │
   ▼
┌──────────────────┐
│                  │
│  Screen Output   │
│                  │
└──────────────────┘
```

## System Components

### 1. Core Technologies

- **Python**: The application is written in Python, utilizing various libraries for video processing, display, and hardware acceleration.
- **OpenCV (cv2)**: Used for video capture and basic image processing operations.
- **PyGame**: Handles the display window and user input.
- **NumPy**: Provides efficient array operations for image data manipulation.
- **Apple Core Image Framework**: Used for hardware-accelerated image processing on Apple Silicon devices.

### 2. Hardware Acceleration

The application is optimized for Apple Silicon Macs, with specific code paths that leverage the Core Image framework for hardware-accelerated image processing:

- **Detection**: The system automatically detects if it's running on Apple Silicon.
- **Core Image Integration**: Uses Quartz, Foundation, and CoreFoundation through PyObjC to interface with Apple's Core Image framework.
- **Fallback**: If hardware acceleration is unavailable, the system falls back to software-based processing using OpenCV.

### 3. Video Acquisition Module

- **YouTube Stream Fetching**: Uses `yt-dlp` (via subprocess) to retrieve the stream URL of a YouTube video.
- **Video Capture**: OpenCV's `VideoCapture` is used with the FFmpeg backend to capture frames from the stream.
- **Configuration**: Stream URL can be specified via command-line arguments or an environment variable loaded from a `.env` file.

### 4. Image Processing Pipeline

#### 4.1 Frame Capture
- Captures frames from the video stream at a target resolution of 1280×720.
- Handles potential errors during frame capture gracefully.

#### 4.2 Grid Effect Processing
The core image processing is performed by two main functions:

- **`apply_grid_effect()`**: Orchestrates the recursive application of the grid effect to the specified depth.
- **`generate_grid_frame()`**: Creates a single level of the grid effect by:
  - Dividing the frame into a grid of cells (configurable size)
  - Scaling the previous frame to fit into each cell while maintaining aspect ratio
  - Using hardware acceleration when available for scaling and cropping operations
  - Managing memory efficiently to avoid leaks

#### 4.3 Memory Management
- Uses explicit garbage collection to manage memory usage during processing.
- Deletes intermediate frames to reduce memory footprint.
- Implements periodic garbage collection to prevent memory leaks during long-running sessions.

### 5. Display System

- **PyGame Window**: Creates a resizable window for displaying the processed video frames.
- **Overlay Information**: Displays performance metrics and configuration settings as an overlay.
- **Interactive Controls**: Allows users to modify parameters in real-time using keyboard shortcuts.

### 6. User Interface and Controls

- **Keyboard Controls**:
  - Up/Down arrows: Increase/decrease grid size
  - Number keys (1-9, 0): Set recursion depth (0 = depth 10)
  - 'D' key: Toggle debug mode
  - ESC: Exit the application
- **On-Screen Display**: Shows current settings and performance metrics:
  - Grid size and depth configuration
  - Frame statistics (captured, displayed)
  - Processing time and FPS
  - Hardware acceleration status
  - Current mode (DEBUG/GRID)

### 7. Monitoring and Diagnostics

- **Logging System**: Uses Python's logging module to output status information and errors.
- **Performance Metrics**: Tracks and displays:
  - Frame capture, processing, and display counts
  - Processing time per frame
  - Frames per second
  - Dropped frames
- **Periodic Status Updates**: Logs status information at regular intervals.

## Data Flow

1. **Input**: YouTube live stream URL (from command line or `.env` file)
2. **Stream Acquisition**: yt-dlp extracts the direct stream URL
3. **Frame Capture**: OpenCV captures frames from the stream
4. **Processing**: Frames are processed through the recursive grid effect pipeline
5. **Display**: Processed frames are displayed in the PyGame window with overlay information
6. **User Input**: Keyboard commands adjust processing parameters in real-time

## Error Handling and Resilience

- **Graceful Degradation**: Falls back to simpler processing when errors occur
- **Exception Handling**: Comprehensive try/except blocks prevent crashes
- **Signal Handling**: Properly handles program termination signals (SIGINT)
- **Resource Cleanup**: Ensures proper cleanup of video capture and display resources

## Configuration Options

- **Command-Line Arguments**:
  - `--grid-size`: Initial grid dimensions (default: 3×3)
  - `--depth`: Recursive depth of the grid effect (default: 1)
  - `--youtube-url`: URL of the YouTube stream to process
  - `--log-level`: Controls verbosity of logging
  - `--debug`: Starts in debug mode (shows original frame)
- **Environment Variables**:
  - `YOUTUBE_URL`: Default YouTube URL if not specified via command line

## Performance Considerations

1. **Memory Management**: 
   - Explicit garbage collection to prevent memory leaks
   - Careful management of image buffers to avoid excessive memory usage

2. **Processing Optimization**:
   - Hardware acceleration on Apple Silicon for improved performance
   - Efficient image scaling and cropping using Core Image when available

3. **Frame Rate Control**:
   - Targets 30 FPS using PyGame's clock mechanism
   - Monitors and displays actual achieved frame rate

## Future Enhancement Potential

1. **Additional Output Options**: Recording processed video to file
2. **Multiple Input Sources**: Support for camera input or local video files
3. **More Effect Parameters**: Additional visual effects or transformations
4. **GPU Acceleration**: Extend hardware acceleration to non-Apple platforms using CUDA or OpenCL
5. **Web Interface**: Add a web-based control interface for remote operation 

## Code Examples

### 1. Hardware Acceleration Detection

The application automatically detects Apple Silicon hardware and enables hardware acceleration when available:

```python
# Check for Apple Silicon and initialize hardware acceleration
is_apple_silicon = platform.system() == 'Darwin' and platform.machine().startswith('arm')
hardware_acceleration_available = False

if is_apple_silicon:
    try:
        import objc
        from Foundation import NSData, NSMutableData
        from Quartz import CIContext, CIImage, CIFilter, kCIFormatRGBA8, kCIContextUseSoftwareRenderer, CIVector
        from CoreFoundation import CFDataCreate
        ci_context = CIContext.contextWithOptions_({kCIContextUseSoftwareRenderer: False})
        hardware_acceleration_available = True
        logging.info("Hardware acceleration enabled on Apple Silicon")
    except ImportError as e:
        logging.warning(f"Hardware acceleration unavailable: {e}")
else:
    logging.info("Not running on Apple Silicon, using software processing")
```

### 2. Core Image Acceleration Implementation

When hardware acceleration is available, the application uses Core Image for efficient scaling and cropping:

```python
if hardware_acceleration_available:
    ci_img = cv_to_ci_image(previous_frame)
    scale_filter = CIFilter.filterWithName_("CILanczosScaleTransform")
    scale_filter.setValue_forKey_(ci_img, "inputImage")
    scale_filter.setValue_forKey_(scaled_h / h, "inputScale")
    scale_filter.setValue_forKey_(1.0, "inputAspectRatio")
    scaled_ci = scale_filter.valueForKey_("outputImage")
    crop_filter = CIFilter.filterWithName_("CICrop")
    crop_filter.setValue_forKey_(scaled_ci, "inputImage")
    rect_vector = CIVector.vectorWithX_Y_Z_W_(crop_x, 0, cell_w, cell_h)
    crop_filter.setValue_forKey_(rect_vector, "inputRectangle")
    cell_ci = crop_filter.valueForKey_("outputImage")
    cell = ci_to_cv_image(cell_ci, cell_w, cell_h)
    # Clean up Core Image objects
    del ci_img, scaled_ci, cell_ci
else:
    scaled = cv2.resize(previous_frame, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
    cell = scaled[:, crop_x:crop_x + cell_w]
    del scaled  # Clean up scaled frame
```

### 3. Recursive Grid Effect Implementation

The core algorithm applies the grid effect recursively to create the fractal-like pattern:

```python
def apply_grid_effect(frame, grid_size, depth):
    if frame is None:
        return None
    if debug_mode or depth == 0:
        logging.info("Depth 0: Returning original frame (debug mode or depth=0)")
        return frame.copy()
    
    try:
        previous_frame = frame.copy()  # Start with the original frame
        for d in range(1, depth + 1):
            new_frame = generate_grid_frame(previous_frame, grid_size, d)
            if new_frame is None:
                logging.error(f"Aborting at depth {d} due to frame generation failure")
                break
            # Replace previous frame with the new one and delete the old one
            del previous_frame
            previous_frame = new_frame
            gc.collect()  # Force garbage collection after each depth
        return previous_frame
    except Exception as e:
        logging.error(f"Grid effect pipeline error: {e}")
        return frame.copy()
```

### 4. Memory Management Techniques

The application implements explicit memory management to prevent memory leaks during processing:

```python
# Clean up cell frame
del cell
gc.collect()  # Force garbage collection after each cell

# Periodic garbage collection in the main loop
if current_time - last_gc_time > 10.0:
    gc.collect()
    last_gc_time = current_time
```

### 5. Real-time User Controls

The application allows for dynamic parameter adjustments via keyboard controls:

```python
def handle_keyboard_event(key_name):
    global grid_size, depth, debug_mode
    if key_name == 'up':
        grid_size += 1
        logging.info(f"Grid size increased to {grid_size}")
    elif key_name == 'down' and grid_size > 1:
        grid_size -= 1
        logging.info(f"Grid size decreased to {grid_size}")
    elif key_name in '1234567890':
        try:
            depth = int(key_name) if key_name != '0' else 10
            logging.info(f"Recursion depth set to {depth}")
        except ValueError:
            logging.warning(f"Invalid depth value: {key_name}")
    elif key_name == 'd':
        debug_mode = not debug_mode
        logging.info(f"Debug mode {'enabled' if debug_mode else 'disabled'}")
```

### 6. YouTube Stream Acquisition

The application uses `yt-dlp` to fetch the direct stream URL:

```python
def get_stream_url(url):
    try:
        result = subprocess.run(["yt-dlp", "-f", "best", "--get-url", url],
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception as e:
        logging.error(f"Error getting YouTube stream URL: {e}")
        return None
```

### 7. Status Display and Monitoring

The application provides detailed runtime information:

```python
# Overlay stats
info_surface = pygame.Surface((screen.get_width(), 150), pygame.SRCALPHA)
info_surface.fill((0, 0, 0, 128))
texts = [
    f"Grid: {grid_size}x{grid_size}, Depth: {depth}",
    f"Captured: {frame_count}, Displayed: {displayed_count}",
    f"Processing: {last_process_time * 1000:.1f}ms",
    f"FPS: {int(clock.get_fps())}",
    f"Hardware: {'Enabled' if hardware_acceleration_available else 'Disabled'}",
    f"Mode: {'DEBUG' if debug_mode else 'GRID'} (press 'd' to toggle)"
]
for i, text in enumerate(texts):
    text_surface = font.render(text, True, (255, 255, 255))
    info_surface.blit(text_surface, (10, 10 + i * 20))
screen.blit(info_surface, (0, 0))
```

## Deployment Requirements

### Dependencies

The following dependencies are required to run the application:

- Python 3.8+
- OpenCV (cv2)
- NumPy
- PyGame
- Pygame
- Python-dotenv
- yt-dlp (command-line tool)
- FFmpeg
- PyObjC (for macOS/Apple Silicon hardware acceleration)

### Hardware Recommendations

For optimal performance:
- Apple Silicon Mac (for hardware acceleration)
- 8GB+ RAM (16GB+ recommended for higher grid sizes and depths)
- Stable internet connection for YouTube streaming

### Environment Setup

1. Clone the repository
2. Create a Python virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Install yt-dlp and FFmpeg using package manager (brew, apt, etc.)
5. Create `.env` file with your YouTube URL: `YOUTUBE_URL=https://www.youtube.com/watch?v=yourvideoid`
6. Run the application: `python main.py` 