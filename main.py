import cv2
import numpy as np
import subprocess
from pynput import keyboard
import threading
import queue
import time
import logging
import signal
import sys
import platform
import argparse

# Set up logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check if running on macOS with Apple Silicon
is_apple_silicon = platform.system() == 'Darwin' and platform.machine().startswith('arm')

# Apple Silicon GPU acceleration setup
if is_apple_silicon:
    try:
        import objc
        from Foundation import NSData, NSMutableData
        from Quartz import (
            CIContext, CIImage, CIFilter, 
            kCIFormatRGBA8, kCIContextUseSoftwareRenderer,
            CGRect, CGPoint, CGSize
        )
        from CoreFoundation import CFDataCreate
        
        # CoreVideo is accessed via Quartz
        try:
            from Quartz.CoreVideo import CVPixelBufferCreate, CVPixelBufferGetBaseAddress, kCVPixelFormatType_32BGRA
            has_corevideo = True
        except (ImportError, AttributeError):
            logging.warning("CoreVideo functions not available")
            has_corevideo = False
        
        # Try to set up hardware-accelerated Core Image context - simplified approach
        try:
            # Create default context first
            ci_context = CIContext.context()
            hardware_acceleration_available = True
            logging.info("Using Core Image context (possibly hardware accelerated)")
        except Exception as e:
            logging.warning(f"Core Image context creation failed: {e}")
            ci_context = None
            hardware_acceleration_available = False
    except (ImportError, AttributeError) as e:
        logging.warning(f"Could not initialize Apple Silicon acceleration: {e}")
        hardware_acceleration_available = False
else:
    hardware_acceleration_available = False
    logging.info("Apple Silicon hardware acceleration not available on this system")

# Global variables
n = 3  # Initial grid size (n x n)
d = 1  # Initial recursion depth

# Locks for thread safety
param_lock = threading.Lock()  # For n and d
frame_lock = threading.Lock()  # For latest processed frame

# Queues for buffering
capture_queue = queue.Queue(maxsize=30)  # Increased buffer for captured frames
process_queue = queue.Queue(maxsize=30)  # Increased buffer for processed frames

# Latest processed frame for display
latest_processed_frame = None

# Control flags
running = True

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global running
    logging.info("Shutting down...")
    running = False
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Utility functions for Apple Silicon hardware acceleration
if is_apple_silicon and hardware_acceleration_available:
    def cv_to_ci_image(cv_img):
        """Convert OpenCV image to Core Image format for GPU processing"""
        try:
            # Ensure we're working with the right format
            if cv_img.shape[2] == 3:
                # Convert BGR to BGRA (needed for Core Image)
                cv_img_rgba = cv2.cvtColor(cv_img, cv2.COLOR_BGR2BGRA)
            else:
                cv_img_rgba = cv_img
            
            # Get dimensions
            height, width = cv_img_rgba.shape[:2]
            bytes_per_row = width * 4
            
            # Create NSData from numpy array
            data_buffer = cv_img_rgba.tobytes()
            data_provider = CFDataCreate(None, data_buffer, len(data_buffer))
            
            # Create CIImage from data
            ci_img = CIImage.imageWithBitmapData_bytesPerRow_size_format_colorSpace_(
                data_provider,
                bytes_per_row,
                CGSize(width, height),
                kCIFormatRGBA8,
                None
            )
            return ci_img
        except Exception as e:
            logging.warning(f"Error in cv_to_ci_image: {e}")
            raise
    
    def ci_to_cv_image(ci_img, output_width, output_height):
        """Convert Core Image back to OpenCV format"""
        try:
            # Create buffer for the output
            output_bytes_per_row = output_width * 4
            output_data = NSMutableData.dataWithLength_(output_height * output_bytes_per_row)
            
            # Render CIImage to buffer
            ci_context.render_toBitmap_rowBytes_bounds_format_colorSpace_(
                ci_img,
                output_data.mutableBytes(),
                output_bytes_per_row,
                CGRect(CGPoint(0, 0), CGSize(output_width, output_height)),
                kCIFormatRGBA8,
                None
            )
            
            # Convert back to numpy array
            buffer = np.frombuffer(output_data, dtype=np.uint8).reshape(output_height, output_width, 4)
            return cv2.cvtColor(buffer, cv2.COLOR_BGRA2BGR)
        except Exception as e:
            logging.warning(f"Error in ci_to_cv_image: {e}")
            raise
    
    def ci_resize_image(ci_img, target_width, target_height):
        """Resize a CIImage using GPU acceleration"""
        try:
            # Use Core Image lanczos scale filter (high quality)
            scale_filter = CIFilter.filterWithName_("CILanczosScaleTransform")
            if scale_filter is None:
                raise ValueError("CILanczosScaleTransform filter not available")
                
            scale_filter.setValue_forKey_(ci_img, "inputImage")
            
            # Calculate scale ratio
            original_size = ci_img.extent().size
            scale_x = target_width / original_size.width
            scale_y = target_height / original_size.height
            
            scale_filter.setValue_forKey_(scale_x, "inputScale")
            scale_filter.setValue_forKey_(scale_y, "inputAspectRatio")
            
            result = scale_filter.valueForKey_("outputImage")
            if result is None:
                raise ValueError("Filter output is None")
            return result
        except Exception as e:
            logging.warning(f"Error in ci_resize_image: {e}")
            raise

def grid_arrange(frame, n, current_d):
    """Recursively arrange a frame into an n x n grid for current_d levels."""
    # Base case: if depth is 0, return the original frame
    if current_d == 0:
        return frame
    
    h, w = frame.shape[:2]
    
    # Create a blank canvas the same size as the input frame
    result = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Calculate cell dimensions
    cell_h = h // n
    cell_w = w // n
    
    # Handle extra pixels
    extra_h = h % n
    extra_w = w % n
    
    # If cells would be too small, return a colored frame
    if cell_w < 2 or cell_h < 2:
        average_color = tuple(int(x) for x in cv2.mean(frame)[:3])
        return np.full((h, w, 3), average_color, dtype=np.uint8)
    
    # For each cell in the grid
    for i in range(n):
        # Calculate cell height (distribute extra pixels)
        current_cell_h = cell_h + (1 if i < extra_h else 0)
        y = sum(cell_h + (1 if idx < extra_h else 0) for idx in range(i))
        
        for j in range(n):
            # Calculate cell width (distribute extra pixels)
            current_cell_w = cell_w + (1 if j < extra_w else 0)
            x = sum(cell_w + (1 if idx < extra_w else 0) for idx in range(j))
            
            # Resize the original frame to fit this exact cell size
            cell_content = cv2.resize(frame, (current_cell_w, current_cell_h))
            
            # If we need to go deeper, apply recursion
            if current_d > 1:
                cell_content = grid_arrange(cell_content, n, current_d - 1)
            
            # Place the cell content in the grid
            result[y:y+current_cell_h, x:x+current_cell_w] = cell_content
    
    return result

def on_press(key):
    """Handle keyboard inputs to adjust grid size (n) and recursion depth (d)."""
    global n, d
    try:
        with param_lock:
            if key == keyboard.Key.up:
                n += 1
                logging.info(f"Grid size increased to {n}x{n}")
            elif key == keyboard.Key.down:
                if n > 1:
                    n -= 1
                    logging.info(f"Grid size decreased to {n}x{n}")
            elif hasattr(key, 'char') and key.char in '1234567890':
                d = int(key.char) if key.char != '0' else 10
                logging.info(f"Recursion depth set to {d}")
            elif key == keyboard.Key.esc:
                global running
                running = False
    except AttributeError:
        pass

def capture_frames(cap, capture_queue):
    """Capture frames and add them to the capture queue."""
    consecutive_failures = 0
    reconnect_delay = 1
    max_reconnect_delay = 30
    
    # Frame rate control
    target_capture_fps = 30
    min_frame_interval = 1.0 / target_capture_fps
    last_capture_time = time.time()
    
    # Set up hardware-accelerated video capture if available
    if is_apple_silicon and hardware_acceleration_available:
        # Enable hardware decoding in OpenCV if available
        cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        try:
            # Check if hardware acceleration is active
            hw_accel = cap.get(cv2.CAP_PROP_HW_ACCELERATION)
            if hw_accel > 0:
                logging.info(f"Video hardware acceleration enabled (mode: {hw_accel})")
            else:
                logging.info("Hardware video decoding not available for this stream")
        except:
            logging.info("Could not query hardware acceleration status")
    
    # Performance monitoring
    capture_count = 0
    last_log_time = time.time()
    log_interval = 10.0  # Log capture stats every 10 seconds
    
    while running:
        # Control capture rate to avoid overwhelming the queue
        current_time = time.time()
        time_since_last = current_time - last_capture_time
        
        # Skip capturing if we're capturing too fast or the queue is full
        if time_since_last < min_frame_interval:
            # Sleep to maintain frame rate and reduce CPU usage
            sleep_time = min_frame_interval - time_since_last
            time.sleep(min(sleep_time, 0.01))  # Don't sleep too long
            continue
        
        # Skip capturing if the queue is too full
        if capture_queue.qsize() > 20:
            time.sleep(0.01)
            continue
        
        # Capture a frame
        ret, frame = cap.read()
        last_capture_time = time.time()  # Update even if capture fails
        
        if not ret:
            consecutive_failures += 1
            if consecutive_failures > 5:
                logging.warning(f"Multiple capture failures, attempting to reconnect...")
                time.sleep(reconnect_delay)
                
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                
                # Try to reconnect
                try:
                    cap.release()
                    result = subprocess.run(
                        ["yt-dlp", "-f", "best", "--get-url", youtube_url],
                        capture_output=True, text=True, check=True
                    )
                    stream_url = result.stdout.strip()
                    
                    # Create a new capture object
                    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                    
                    # Reconfigure capture properties
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 30)
                    
                    # Re-enable hardware acceleration
                    if is_apple_silicon and hardware_acceleration_available:
                        cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                    
                    if cap.isOpened():
                        logging.info("Successfully reconnected to stream")
                        consecutive_failures = 0
                        reconnect_delay = 1
                    else:
                        logging.error("Failed to reconnect to stream")
                except Exception as e:
                    logging.error(f"Error during reconnection: {e}")
                
                continue
            
            # Short delay on failure
            time.sleep(0.1)
            continue
        
        # Reset failure counter on success
        consecutive_failures = 0
        
        # Add the frame to the queue if not full
        try:
            capture_queue.put(frame, timeout=0.1)
            capture_count += 1
        except queue.Full:
            # Queue is full, skip this frame
            pass
        
        # Log capture stats periodically
        current_time = time.time()
        if current_time - last_log_time > log_interval:
            capture_fps = capture_count / (current_time - last_log_time)
            logging.info(f"Capture: {capture_fps:.1f} FPS, Queue size: {capture_queue.qsize()}")
            capture_count = 0
            last_log_time = current_time

def process_frames(capture_queue, process_queue):
    """Process frames from the capture queue and add them to the process queue."""
    last_process_time = time.time()
    frame_interval = 1.0 / 60  # Aim for higher processing rate
    frame_count = 0
    last_log_time = time.time()
    
    # Processing stats
    process_times = []
    avg_process_time = 0
    
    # Keep track of the current parameters
    current_n = n
    current_d = d
    
    # Limit logging frequency
    log_interval = 5.0  # Only log every 5 seconds
    
    while running:
        try:
            # Monitor queue sizes and adjust behavior
            capture_size = capture_queue.qsize()
            process_size = process_queue.qsize()
            
            # Adaptive frame skipping based on queue sizes
            skip_frames = 0
            
            # If process queue is getting full, we need to slow down
            if process_size > 15:
                # Wait a bit before processing more
                time.sleep(0.01)
                continue
                
            # Check if we should skip frames based on CPU load
            if process_size < 3 and capture_size > 5:
                # Process queue is low but capture queue has frames - process faster
                skip_frames = 0
            elif capture_size > 10:
                # Capture queue building up - skip some frames
                skip_frames = min(3, capture_size // 5)
            
            # Get the current parameters to avoid locking too often
            if frame_count % 10 == 0:
                with param_lock:
                    current_n = n
                    current_d = d
            
            # Calculate current complexity
            complexity = current_n ** current_d
            
            # Skip frames if needed
            if skip_frames > 0:
                for _ in range(skip_frames):
                    if not capture_queue.empty():
                        capture_queue.get_nowait()
            
            # Get a frame to process
            try:
                frame = capture_queue.get(timeout=0.1)
            except queue.Empty:
                # No frames to process, wait a bit
                time.sleep(0.01)
                continue
            
            # Skip processing for very high complexity when queue is filling
            if complexity > 100 and process_size > 5:
                # Skip intensive processing if we're falling behind
                if avg_process_time * 2 > frame_interval:
                    # Just pass through the original frame
                    process_queue.put(frame, timeout=0.1)
                    continue
            
            # Check if the frame is valid
            if frame is None or frame.size == 0:
                # Skip invalid frames
                continue
            
            # Start timing the processing
            process_start = time.time()
            
            # Process the frame
            try:
                # Use our grid_arrange function
                processed_frame = grid_arrange(frame.copy(), current_n, current_d)
                
                # Verify processed frame
                if processed_frame is None or processed_frame.size == 0:
                    # Use original frame as fallback
                    processed_frame = frame.copy()
                elif processed_frame.shape != frame.shape:
                    # Ensure consistent dimensions
                    processed_frame = cv2.resize(processed_frame, (frame.shape[1], frame.shape[0]))
                
                # Add to process queue with a timeout
                process_queue.put(processed_frame, timeout=0.1)
                
                # Update processing time stats
                process_time = time.time() - process_start
                process_times.append(process_time)
                
                # Keep only recent process times
                if len(process_times) > 30:
                    process_times.pop(0)
                
                # Calculate average processing time
                avg_process_time = sum(process_times) / len(process_times)
                
                # Update frame count
                frame_count += 1
                
                # Log processing stats periodically
                current_time = time.time()
                if current_time - last_log_time > log_interval:
                    # Calculate processing FPS
                    process_fps = frame_count / (current_time - last_log_time)
                    # Log performance metrics
                    logging.info(f"Processing: {process_fps:.1f} FPS, Avg time: {avg_process_time*1000:.1f}ms, Complexity: {complexity}")
                    # Reset counters
                    frame_count = 0
                    last_log_time = current_time
                
            except Exception as e:
                # Log errors but continue processing
                logging.error(f"Frame processing error: {e}")
                
                # Add the original frame to keep the pipeline going
                try:
                    process_queue.put(frame, timeout=0.1)
                except queue.Full:
                    pass
        
        except Exception as e:
            # Handle any other errors
            import traceback
            logging.error(f"Error in process_frames: {e}")
            logging.error(traceback.format_exc())
            time.sleep(0.1)

def format_large_number(num):
    """Format a large number into a human-readable string with suffixes."""
    if num < 1000:
        return str(num)
    suffixes = [
        (1e18, 'quintillion'),
        (1e15, 'quadrillion'),
        (1e12, 'trillion'),
        (1e9, 'billion'),
        (1e6, 'million'),
        (1e3, 'thousand'),
    ]
    for threshold, suffix in suffixes:
        if num >= threshold:
            return f"{num / threshold:.2f} {suffix}"
    return str(num)

def display_frames(process_queue):
    """Display the latest processed frame at a consistent frame rate using Pygame."""
    global latest_processed_frame, running
    
    # Import pygame here to avoid issues if not used
    import pygame
    import pygame.freetype
    
    # Initialize pygame - this must happen in this thread
    pygame.init()
    pygame.freetype.init()
    
    # Setup timing variables
    target_fps = 30  # Increased target fps for smoother playback
    target_delay = 1.0 / target_fps
    frame_times = []
    last_fps_print = time.time()
    frame_count = 0
    current_fps = 0
    
    # Set up fonts
    pygame_font = pygame.freetype.SysFont("Arial", 24)
    
    # Setup the display window
    try:
        logging.info("Setting up Pygame window")
        
        # Set display flags for better performance
        display_flags = pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE
        
        # Get screen info
        info = pygame.display.Info()
        screen_width, screen_height = 1280, 720  # Default size
        
        # Try to get actual screen size if available
        if hasattr(info, 'current_w') and hasattr(info, 'current_h'):
            if info.current_w > 0 and info.current_h > 0:
                screen_width, screen_height = info.current_w, info.current_h
        
        # Create a display surface
        screen = pygame.display.set_mode((screen_width, screen_height), display_flags)
        pygame.display.set_caption("Recursive Grid Livestream")
        
        # Pre-allocate buffers for triple buffering
        buffer_surfaces = [None, None, None]
        current_buffer = 0
        
        # Function to convert OpenCV frame to Pygame surface with caching
        surface_cache = {}
        max_cache_size = 10  # Limit cache size
        
        def cv_to_pygame(cv_img, size=None):
            # Create a cache key based on frame content hash and size
            if size is None:
                size = (cv_img.shape[1], cv_img.shape[0])
                
            # Use the first and last bytes of the image as a cache key
            # This is faster than a full hash but still effective for similar frames
            hash_key = (cv_img.shape, cv_img.tobytes()[:100] + cv_img.tobytes()[-100:])
            size_key = size
            cache_key = (hash_key, size_key)
            
            # Check if we have this frame cached
            if cache_key in surface_cache:
                return surface_cache[cache_key]
            
            # If cache is full, remove oldest entry
            if len(surface_cache) >= max_cache_size:
                surface_cache.clear()
            
            # Convert BGR to RGB
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            # Ensure the array has the right format for Pygame
            rgb_img = np.ascontiguousarray(rgb_img)
            
            try:
                # Create a Pygame surface from the numpy array
                surface = pygame.surfarray.make_surface(rgb_img.swapaxes(0, 1))
                
                # Resize if needed
                if size != (rgb_img.shape[1], rgb_img.shape[0]):
                    try:
                        surface = pygame.transform.smoothscale(surface, size)
                    except ValueError:
                        surface = pygame.transform.scale(surface, size)
                
                # Cache the surface
                surface_cache[cache_key] = surface
                return surface
                
            except Exception as e:
                # Fallback method if make_surface fails
                logging.warning(f"Surface creation failed, using fallback: {e}")
                h, w = rgb_img.shape[:2]
                surface = pygame.Surface((w, h))
                pygame.pixelcopy.array_to_surface(surface, rgb_img.swapaxes(0, 1))
                
                # Resize if needed
                if size != (w, h):
                    try:
                        surface = pygame.transform.scale(surface, size)
                    except Exception:
                        pass
                
                # Cache the surface
                surface_cache[cache_key] = surface
                return surface
        
        # Flag to indicate successful window setup
        pygame_display = True
        logging.info("Pygame window setup successful")
    except Exception as e:
        import traceback
        logging.error(f"Failed to set up Pygame window: {e}")
        logging.error(traceback.format_exc())
        running = False
        pygame.quit()
        return
    
    # Set up a frame buffer for smoother playback
    frame_buffer = []
    buffer_size = 3  # Number of frames to buffer
    
    logging.info("Buffering frames before playback...")
    buffer_frames = 5
    
    try:
        while process_queue.qsize() < buffer_frames and running:
            # Process Pygame events while buffering
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
            time.sleep(0.1)
        
        logging.info("Starting playback...")
        
        # Main display loop
        clock = pygame.time.Clock()
        last_frame_time = time.time()
        
        while running:
            start_time = time.time()
            
            # Process Pygame events - do this first for responsiveness
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break
                elif event.type == pygame.VIDEORESIZE:
                    # Handle window resize
                    screen = pygame.display.set_mode((event.w, event.h), display_flags)
            
            # Fill buffer if needed
            while len(frame_buffer) < buffer_size and not process_queue.empty():
                try:
                    new_frame = process_queue.get_nowait()
                    frame_buffer.append(new_frame)
                except queue.Empty:
                    break
            
            # Calculate the time since last frame
            now = time.time()
            elapsed = now - last_frame_time
            
            # Only render a new frame if it's time (maintain target fps)
            if elapsed >= target_delay and frame_buffer:
                last_frame_time = now
                
                # Get next frame from buffer
                display_frame = frame_buffer.pop(0)
                
                # Update parameters
                with param_lock:
                    current_n, current_d = n, d
                
                # Calculate complexity for overlay
                complexity = current_n ** current_d
                
                # Store latest frame for potential saving
                with frame_lock:
                    latest_processed_frame = display_frame.copy()
                
                # Calculate the screen dimensions once
                screen_w, screen_h = screen.get_width(), screen.get_height()
                
                # Calculate the aspect ratio for proper scaling
                frame_aspect = display_frame.shape[1] / display_frame.shape[0]
                screen_aspect = screen_w / screen_h
                
                # Determine the right scale to fill the screen completely
                if abs(frame_aspect - screen_aspect) < 0.01:
                    # Almost the same aspect ratio, use the full screen
                    scaled_width, scaled_height = screen_w, screen_h
                elif frame_aspect > screen_aspect:
                    # Frame is wider than screen
                    scaled_width = screen_w
                    scaled_height = int(scaled_width / frame_aspect)
                else:
                    # Frame is taller than screen
                    scaled_height = screen_h
                    scaled_width = int(scaled_height * frame_aspect)
                
                # Get a scaled surface from our function (with caching)
                scaled_surface = cv_to_pygame(display_frame, (scaled_width, scaled_height))
                
                # Center the image on screen
                x_offset = (screen_w - scaled_width) // 2
                y_offset = (screen_h - scaled_height) // 2
                
                # Fill the background with black - only areas that need it
                if x_offset > 0:
                    screen.fill((0, 0, 0), (0, 0, x_offset, screen_h))
                    screen.fill((0, 0, 0), (screen_w - x_offset, 0, x_offset, screen_h))
                if y_offset > 0:
                    screen.fill((0, 0, 0), (0, 0, screen_w, y_offset))
                    screen.fill((0, 0, 0), (0, screen_h - y_offset, screen_w, y_offset))
                
                # Draw the scaled frame centered on screen
                screen.blit(scaled_surface, (x_offset, y_offset))
                
                # Calculate and update FPS
                frame_count += 1
                frame_times.append(time.time())
                # Keep only recent frame times for FPS calculation
                current_time = time.time()
                while frame_times and current_time - frame_times[0] > 1.0:
                    frame_times.pop(0)
                
                if frame_times:
                    current_fps = len(frame_times)
                
                # Add text overlays
                complexity_text = f"Grid: {current_n}x{current_n}, Depth: {current_d}, Complexity: {format_large_number(complexity)}"
                fps_text = f"FPS: {current_fps}"
                
                # Render text with shadow for better visibility
                shadow_color = (0, 0, 0)
                text_color = (255, 255, 255)
                
                # Render shadow first
                pygame_font.render_to(screen, (12, 32), complexity_text, shadow_color)
                pygame_font.render_to(screen, (12, 62), fps_text, shadow_color)
                
                # Render text
                pygame_font.render_to(screen, (10, 30), complexity_text, text_color)
                pygame_font.render_to(screen, (10, 60), fps_text, text_color)
                
                # Update the display - only once per frame
                pygame.display.flip()
                
                # Get more frames if buffer is low
                while len(frame_buffer) < buffer_size and not process_queue.empty():
                    try:
                        new_frame = process_queue.get_nowait()
                        frame_buffer.append(new_frame)
                    except queue.Empty:
                        break
            else:
                # If we're not rendering a new frame, we should at least process events
                # and prevent CPU spinning by sleeping a tiny amount
                pygame.time.wait(1)
            
            # Let pygame manage the framerate
            clock.tick(60)  # Cap at 60 FPS for UI smoothness
                
    except Exception as e:
        import traceback
        logging.error(f"Critical error in display_frames: {e}")
        logging.error(traceback.format_exc())
    
    finally:
        # Clean up Pygame
        running = False
        pygame.quit()
        logging.info("Pygame display closed")

youtube_url = "https://www.youtube.com/watch?v=ZzWBpGwKoaI"

def main():
    global running, youtube_url, n, d
    
    # Command line argument parsing
    parser = argparse.ArgumentParser(description='Recursive Video Grid')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no display window)')
    parser.add_argument('--grid-size', type=int, default=3, help='Initial grid size (NxN)')
    parser.add_argument('--depth', type=int, default=1, help='Initial recursion depth')
    parser.add_argument('--save-frames', action='store_true', help='Save frames as images every few seconds')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to save frames in')
    args = parser.parse_args()
    
    # Set initial grid size and depth from command line
    n = args.grid_size
    d = args.depth
    
    # Create output directory if saving frames
    if args.save_frames:
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        logging.info(f"Frames will be saved to {args.output_dir}/")
    
    # Enable OpenCV hardware acceleration features if available
    if is_apple_silicon:
        try:
            # Set OpenCV to use acceleration if available
            cv2.setNumThreads(0)  # Disable OpenCV's threading to prevent conflicts with our own
            cv2.ocl.setUseOpenCL(True)  # Enable OpenCL acceleration
            logging.info("OpenCV acceleration options configured")
        except Exception as e:
            logging.warning(f"Could not configure OpenCV acceleration: {e}")
    
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    logging.info("Initializing video stream...")
    
    try:
        result = subprocess.run(
            ["yt-dlp", "-f", "best", "--get-url", youtube_url],
            capture_output=True, text=True, check=True
        )
        stream_url = result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Error fetching stream URL: {e}")
        return

    # Set up video capture with hardware acceleration if available
    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logging.error("Error: Could not open video stream.")
        return
    
    # Configure video capture properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 30)  # Increased buffer size
    
    # Enable hardware decoding if available
    if is_apple_silicon and hardware_acceleration_available:
        cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('h', '2', '6', '4'))  # Use h264 for hardware acceleration
    
    threads = []
    
    # Create the threads
    capture_thread = threading.Thread(target=capture_frames, args=(cap, capture_queue))
    process_thread = threading.Thread(target=process_frames, args=(capture_queue, process_queue))
    
    # Define frame saving function
    def save_frames_periodically(process_queue, output_dir):
        frame_count = 0
        while running:
            try:
                # Save a frame every 5 seconds
                frame = process_queue.get(timeout=5.0)
                
                # Save the frame
                timestamp = int(time.time())
                with param_lock:
                    current_n, current_d = n, d
                
                filename = f"{output_dir}/grid_n{current_n}_d{current_d}_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                
                logging.info(f"Saved frame to {filename}")
                frame_count += 1
                
            except queue.Empty:
                time.sleep(0.5)
            except Exception as e:
                logging.error(f"Error saving frame: {e}")
                time.sleep(1.0)
    
    # Configure the display thread and start processing threads
    if args.headless or args.save_frames:
        if args.save_frames:
            logging.info(f"Running in frame saving mode, saving to {args.output_dir}/")
            display_thread = threading.Thread(target=save_frames_periodically, args=(process_queue, args.output_dir))
        else:
            logging.info("Running in headless mode (no display)")
            
            # Define a headless display function
            def headless_display(process_queue):
                logging.info("Headless display starting - monitoring performance only")
                frame_count = 0
                last_log_time = time.time()
                
                while running:
                    try:
                        # Just consume frames and log statistics
                        frame = process_queue.get(timeout=0.5)
                        frame_count += 1
                        
                        # Log performance every 5 seconds
                        current_time = time.time()
                        if current_time - last_log_time >= 5.0:
                            fps = frame_count / (current_time - last_log_time)
                            with param_lock:
                                current_n, current_d = n, d
                            complexity = current_n ** current_d
                            
                            logging.info(f"Processing at {fps:.1f} FPS, Grid: {current_n}x{current_n}, " +
                                        f"Depth: {current_d}, Complexity: {format_large_number(complexity)}")
                            
                            frame_count = 0
                            last_log_time = current_time
                            
                            # Capture a sample frame every 5 seconds for testing
                            with frame_lock:
                                global latest_processed_frame
                                latest_processed_frame = frame.copy()
                                
                    except queue.Empty:
                        time.sleep(0.1)
                    except Exception as e:
                        logging.error(f"Error in headless display: {e}")
                        time.sleep(0.5)
            
            display_thread = threading.Thread(target=headless_display, args=(process_queue,))
        
        # Configure threads for better performance
        capture_thread.daemon = True
        process_thread.daemon = True
        display_thread.daemon = True
        
        # Start all threads
        threads.append(capture_thread)
        threads.append(process_thread)
        threads.append(display_thread)
        
        for thread in threads:
            thread.start()
        
        # Add performance monitoring thread if on Apple Silicon
        if is_apple_silicon and hardware_acceleration_available:
            def monitor_performance():
                """Monitor system performance and adjust processing parameters if needed"""
                import psutil
                while running:
                    try:
                        # Get CPU usage
                        cpu_percent = psutil.cpu_percent(interval=5.0)
                        memory_percent = psutil.virtual_memory().percent
                        
                        # Log performance metrics
                        logging.info(f"Performance: CPU {cpu_percent}%, Memory {memory_percent}%, " + 
                                    f"Queues: Capture {capture_queue.qsize()}, Process {process_queue.qsize()}")
                        
                        # Auto-adjust parameters if system is overloaded
                        with param_lock:
                            global d
                            if cpu_percent > 90 and d > 1:
                                d -= 1
                                logging.info(f"High CPU load, automatically reducing recursion depth to {d}")
                    except Exception as e:
                        logging.error(f"Error in performance monitor: {e}")
                        time.sleep(5)
            
            try:
                import psutil
                perf_thread = threading.Thread(target=monitor_performance)
                perf_thread.daemon = True
                threads.append(perf_thread)
                perf_thread.start()
                logging.info("Performance monitoring active")
            except ImportError:
                logging.info("psutil not installed, performance monitoring disabled")
        
        # For headless mode, wait for threads to complete or until interrupted
        try:
            # Wait for all threads to finish
            for thread in threads:
                while thread.is_alive() and running:
                    thread.join(timeout=0.5)
        except KeyboardInterrupt:
            running = False
            logging.info("Interrupt received, shutting down...")
        finally:
            running = False
            logging.info("Shutting down...")
            
            for thread in threads:
                if thread.is_alive():
                    thread.join(timeout=1.0)
                    
            cap.release()
            pygame.quit()
            listener.stop()
            logging.info("Application closed")
    
    else:
        # For display mode, start only the capture and process threads
        # The display function will run on the main thread
        capture_thread.daemon = True
        process_thread.daemon = True
        
        threads.append(capture_thread)
        threads.append(process_thread)
        
        for thread in threads:
            thread.start()
        
        # Add performance monitoring thread if on Apple Silicon
        if is_apple_silicon and hardware_acceleration_available:
            def monitor_performance():
                """Monitor system performance and adjust processing parameters if needed"""
                import psutil
                while running:
                    try:
                        # Get CPU usage
                        cpu_percent = psutil.cpu_percent(interval=5.0)
                        memory_percent = psutil.virtual_memory().percent
                        
                        # Log performance metrics
                        logging.info(f"Performance: CPU {cpu_percent}%, Memory {memory_percent}%, " + 
                                    f"Queues: Capture {capture_queue.qsize()}, Process {process_queue.qsize()}")
                        
                        # Auto-adjust parameters if system is overloaded
                        with param_lock:
                            global d
                            if cpu_percent > 90 and d > 1:
                                d -= 1
                                logging.info(f"High CPU load, automatically reducing recursion depth to {d}")
                    except Exception as e:
                        logging.error(f"Error in performance monitor: {e}")
                        time.sleep(5)
            
            try:
                import psutil
                perf_thread = threading.Thread(target=monitor_performance)
                perf_thread.daemon = True
                threads.append(perf_thread)
                perf_thread.start()
                logging.info("Performance monitoring active")
            except ImportError:
                logging.info("psutil not installed, performance monitoring disabled")
        
        try:
            # Run the display function on the main thread
            display_frames(process_queue)
        except KeyboardInterrupt:
            running = False
            logging.info("Interrupt received, shutting down...")
        finally:
            running = False
            logging.info("Shutting down...")
            
            for thread in threads:
                if thread.is_alive():
                    thread.join(timeout=1.0)
                    
            cap.release()
            pygame.quit()
            listener.stop()
            logging.info("Application closed")

if __name__ == "__main__":
    main()