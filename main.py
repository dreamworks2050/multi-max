import cv2
import numpy as np
import subprocess
import threading
import time
import logging
import signal
import platform
import argparse
import os
from dotenv import load_dotenv
import traceback
import gc
import pygame
import Quartz
import tracemalloc
import psutil
import queue
from memory_profiler import profile

# Global variables section
os.environ['OPENCV_FFMPEG_DEBUG'] = '0'  # Disable verbose FFmpeg output

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def configure_logging(level_name):
    """Configure logging level based on input string."""
    try:
        level = getattr(logging, level_name.upper(), logging.INFO)
        
        class FFmpegFilter(logging.Filter):
            def filter(self, record):
                # Skip low-priority FFmpeg and OpenCV logs
                message = record.getMessage().lower()
                if record.levelno >= logging.WARNING:
                    return True
                if any(term in message for term in [
                    '[opencv', '[ffmpeg', 'opening', 'skip', 
                    'hls request', 'videoplayback', 'manifest.googlevideo',
                    'cannot reuse http connection'  # Added term
                ]):
                    return False
                return True
        
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        for handler in root_logger.handlers:
            handler.addFilter(FFmpegFilter())
        logging.debug("Logging configured with level %s and FFmpeg/OpenCV filter", level_name.upper())
    except AttributeError as e:
        logging.error(f"Invalid log level '{level_name}': {e}")
        logging.getLogger().setLevel(logging.INFO)

# Load hardware acceleration settings
force_hardware_acceleration = os.getenv('FORCE_HARDWARE_ACCELERATION', 'true').lower() == 'true'
allow_software_fallback = os.getenv('ALLOW_SOFTWARE_FALLBACK', 'false').lower() == 'true'

# Load memory tracing settings
enable_memory_tracing = os.getenv('ENABLE_MEMORY_TRACING', 'false').lower() == 'true'

# Global variables for hardware acceleration
is_apple_silicon = platform.system() == 'Darwin' and platform.machine().startswith('arm')
hardware_acceleration_available = False
ci_context = None
context_options = {}

# Global variables for memory management
cleanup_thread_running = False
memory_cleanup_thread = None
cleanup_stats = {
    'total_cleanups': 0,
    'extra_cleanups': 0,
    'last_memory': 0,
    'peak_memory': 0,
    'last_growth_rate': 0,
    'current_interval': 2.0
}

# Global variables to track key state and press duration
key_pressed = {}
key_press_start = {}
key_repeat_delay = 0.2  # Initial delay before repeat starts
key_repeat_interval = 0.03  # Interval between repeats
key_last_repeat = {}

# Setup for FFmpeg error detection without verbose logging
# Instead of setting OPENCV_FFMPEG_DEBUG=1 which produces too much output,
# we'll create a custom FFmpeg error detection system
os.environ['OPENCV_FFMPEG_DEBUG'] = '0'  # Disable verbose FFmpeg output
ffmpeg_http_errors = []  # Store detected HTTP errors to check periodically
youtube_connection_status = {
    'last_success': 0,
    'last_error': 0,
    'error_count': 0,
    'retry_count': 0,
    'current_host': ''
}

def log_youtube_connection_status(status, host='', error=''):
    """
    Log YouTube connection status without verbose details.
    Only logs important connection events rather than every FFmpeg operation.
    
    Args:
        status: Connection status string ('connected', 'error', 'retry', etc.)
        host: Optional host information 
        error: Optional error details
    """
    global youtube_connection_status
    current_time = time.time()
    
    if status == 'connected':
        # Only log new connections or reconnections after errors
        if youtube_connection_status['last_success'] == 0 or youtube_connection_status['error_count'] > 0:
            if host and host != youtube_connection_status['current_host']:
                logging.info(f"Connected to YouTube server: {host}")
                youtube_connection_status['current_host'] = host
            else:
                logging.info("Connected to YouTube stream successfully")
            youtube_connection_status['error_count'] = 0
            youtube_connection_status['retry_count'] = 0
        youtube_connection_status['last_success'] = current_time
    
    elif status == 'error':
        youtube_connection_status['last_error'] = current_time
        youtube_connection_status['error_count'] += 1
        # Only log the first few errors to avoid spamming the log
        if youtube_connection_status['error_count'] <= 3 and error:
            logging.warning(f"YouTube connection error: {error}")
        elif youtube_connection_status['error_count'] == 4:
            logging.warning("Multiple YouTube connection errors - suppressing further error messages")
    
    elif status == 'retry':
        youtube_connection_status['retry_count'] += 1
        # Only log occasional retries to avoid log spam
        if youtube_connection_status['retry_count'] % 5 == 1:
            logging.info(f"Retrying YouTube connection (attempt {youtube_connection_status['retry_count']})")

# Frame buffer for continuous streaming
frame_buffer = None
frame_buffer_lock = None
frame_reader_thread = None
frame_buffer_size = 60  # Increase buffer size further to 60
should_stop_frame_reader = False
current_stream_url = None  # Store the stream URL globally
last_buffer_warning_time = 0  # Track when we last issued a warning
frame_drop_threshold = 0.8  # Drop frames if buffer is more than 80% full
stream_url_refresh_interval = 5 * 60  # Refresh YouTube URL every 5 minutes (reduced from 15)
last_url_refresh_time = 0  # Last time the URL was refreshed

def refresh_youtube_url(original_url):
    """Get a fresh streaming URL from YouTube to handle shifting servers."""
    try:
        logging.info(f"Refreshing YouTube stream URL for: {original_url}")
        # Force a clearing of any cached data by using a timestamp parameter
        ts = int(time.time())
        query_url = f"{original_url}{'&' if '?' in original_url else '?'}_ts={ts}"
        
        # Use a more aggressive approach to get a fresh URL from YouTube
        new_url = get_stream_url(query_url)
        
        if new_url:
            logging.info("Successfully obtained fresh YouTube stream URL")
            return new_url
        else:
            logging.warning("Failed to get a fresh URL, will retry later")
            return None
    except Exception as e:
        logging.error(f"Failed to refresh YouTube URL: {e}")
        return None

def start_frame_reader_thread(video_source, stream_url, original_youtube_url, buffer_size=60):
    """
    Start a background thread to continuously read frames from the video source.
    
    Args:
        video_source: OpenCV VideoCapture object
        stream_url: URL of the video stream for reconnection
        original_youtube_url: Original YouTube URL for periodic refresh
        buffer_size: Maximum number of frames to keep in the buffer
    """
    global frame_buffer, frame_buffer_lock, frame_reader_thread, should_stop_frame_reader
    global current_stream_url, frame_drop_threshold, last_url_refresh_time
    
    # Store the stream URL globally
    current_stream_url = stream_url
    last_url_refresh_time = time.time()
    
    frame_buffer = queue.Queue(maxsize=buffer_size)
    frame_buffer_lock = threading.Lock()
    should_stop_frame_reader = False
    
    def frame_reader_worker():
        """Worker thread that continuously reads frames from the video source."""
        global current_stream_url, last_url_refresh_time
        
        logging.info("Frame reader thread started")
        frames_read = 0
        frames_dropped = 0
        consecutive_failures = 0
        max_consecutive_failures = 5
        target_fps = 30
        frame_interval = 1.0 / target_fps
        last_frame_time = time.time()
        reconnection_backoff = 1.0  # Start with 1 second, will increase with failures
        max_reconnection_backoff = 15.0  # Maximum backoff time in seconds
        
        # Create a dedicated VideoCapture instance for this thread
        # to avoid thread safety issues with FFmpeg
        width = int(video_source.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create a thread-local VideoCapture instance
        thread_cap = None
        last_reconnect_attempt = 0
        reconnect_interval = 3
        
        # Track HTTP connection errors to detect YouTube CDN shifts
        http_errors_count = 0
        http_error_threshold = 3  # Refresh URL after this many HTTP errors
        last_error_time = 0
        error_reset_interval = 60  # Reset error count after 60 seconds without errors
        
        def open_capture(url=None):
            """
            Open the video capture with the given URL or current_stream_url.
            Handles different types of connection failures.
            """
            nonlocal thread_cap, consecutive_failures, reconnection_backoff, last_reconnect_attempt
            
            # Use provided URL or global one
            capture_url = url if url is not None else current_stream_url
            
            if thread_cap is not None:
                thread_cap.release()
                # Longer delay after releasing to ensure complete connection cleanup
                time.sleep(0.5)
                thread_cap = None
                
            if isinstance(capture_url, str) and capture_url:
                logging.info(f"Frame reader opening stream (backoff: {reconnection_backoff:.1f}s)")
                
                # Create dict of advanced FFmpeg options to disable connection reuse
                # This addresses the "Cannot reuse HTTP connection for different host" error
                ffmpeg_options = {
                    "rtsp_transport": "tcp",                   # Use TCP for RTSP
                    "fflags": "nobuffer",                      # Reduce latency
                    "flags": "low_delay",                      # Prioritize low delay
                    "stimeout": "5000000",                     # 5 second timeout in microseconds
                    "reconnect": "1",                          # Enable reconnection
                    "reconnect_streamed": "1",                 # Reconnect if stream fails
                    "reconnect_delay_max": "5",                # Max 5 second reconnect delay
                    "multiple_requests": "0",                  # Disable connection reuse - fixes the YouTube server issue
                    "reuse_socket": "0",                       # Don't reuse sockets
                    "http_persistent": "0",                    # Disable persistent HTTP connections
                }
                
                # Convert dict to FFmpeg options string for OpenCV
                ffmpeg_opt_str = ' '.join([f"-{k} {v}" for k, v in ffmpeg_options.items()])
                
                # Apply options via environment variable which OpenCV/FFmpeg will use
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = ffmpeg_opt_str
                
                # Create VideoCapture with options
                thread_cap = cv2.VideoCapture(capture_url, cv2.CAP_FFMPEG)
                
                # Try setting various options to improve HTTP streaming reliability
                options = [
                    (cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000),  # 5 second timeout
                    (cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000),   # 5 second read timeout
                    (cv2.CAP_PROP_BUFFERSIZE, 3),             # Internal buffer size
                ]
                
                if thread_cap.isOpened():
                    # Set capture properties
                    thread_cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    thread_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    
                    # Apply all options
                    for option, value in options:
                        try:
                            thread_cap.set(option, value)
                        except:
                            pass
                            
                    # Try setting frame rate - not all cameras/drivers support this
                    try:
                        thread_cap.set(cv2.CAP_PROP_FPS, target_fps)
                    except:
                        pass
                        
                    consecutive_failures = 0
                    reconnection_backoff = 1.0  # Reset backoff on success
                    last_reconnect_attempt = time.time()
                    
                    # Extract hostname from URL for better logging
                    host = "unknown"
                    if capture_url and isinstance(capture_url, str):
                        try:
                            from urllib.parse import urlparse
                            parsed_url = urlparse(capture_url)
                            host = parsed_url.netloc
                        except:
                            pass
                    
                    # Use our custom logger with reduced verbosity
                    log_youtube_connection_status('connected', host=host)
                    return True
                else:
                    # Use our custom logger with reduced verbosity
                    log_youtube_connection_status('error', error="Failed to open video capture")
                    logging.error("Failed to open video capture in frame reader thread")
            
            reconnection_backoff = min(reconnection_backoff * 1.5, max_reconnection_backoff)
            return False
            
        # Initialize the capture
        if not open_capture():
            logging.error("Frame reader could not open video source")
        
        # Track if we've filled the buffer at least once
        buffer_filled = False
        
        while not should_stop_frame_reader:
            current_time = time.time()
            
            # Reset HTTP error count if no errors for a while
            if http_errors_count > 0 and current_time - last_error_time > error_reset_interval:
                http_errors_count = 0
                logging.debug("HTTP error count reset after error-free period")
            
            # Check if we need to refresh the YouTube URL periodically or after HTTP errors
            need_refresh = False
            if original_youtube_url:
                # Regular time-based refresh
                if current_time - last_url_refresh_time > stream_url_refresh_interval:
                    need_refresh = True
                    logging.debug(f"Scheduled URL refresh after {stream_url_refresh_interval/60:.1f} minutes")
                # Error-triggered refresh
                elif http_errors_count >= http_error_threshold:
                    need_refresh = True
                    http_errors_count = 0
                    logging.debug(f"Forcing URL refresh after {http_error_threshold} connection errors")
                
                if need_refresh:
                    new_url = refresh_youtube_url(original_youtube_url)
                    if new_url and new_url != current_stream_url:
                        logging.debug("Stream URL refreshed successfully")
                        current_stream_url = new_url
                        # Force reconnection with new URL
                        if open_capture(new_url):
                            logging.debug("Reconnected with fresh YouTube URL")
                    last_url_refresh_time = current_time
            
            # Check if capture is valid
            if thread_cap is None or not thread_cap.isOpened():
                if current_time - last_reconnect_attempt > reconnection_backoff:
                    log_youtube_connection_status('retry')
                    logging.debug(f"Video source disconnected, reconnecting (backoff: {reconnection_backoff:.1f}s)...")
                    open_capture()
                    last_reconnect_attempt = current_time
                time.sleep(0.5)
                continue
            
            # Rate control - don't read frames faster than target FPS if buffer is getting full
            elapsed = current_time - last_frame_time
            buffer_fullness = frame_buffer.qsize() / buffer_size
            
            # If buffer is getting full, slow down frame reading
            if buffer_fullness > frame_drop_threshold and elapsed < frame_interval:
                # Buffer is filling up, sleep to maintain target frame rate
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # Read frame from our thread-local capture
            try:
                ret, frame = thread_cap.read()
                last_frame_time = time.time()  # Update frame time after read
            except Exception as e:
                # Use our custom logger instead of verbose error messages
                error_str = str(e).lower()
                if "http" in error_str or "connect" in error_str or "host" in error_str or "network" in error_str:
                    log_youtube_connection_status('error', error="HTTP connection error")
                    http_errors_count += 1
                    last_error_time = current_time
                else:
                    log_youtube_connection_status('error', error=f"Read error: {type(e).__name__}")
                    
                ret, frame = False, None
                # Force reconnection on exception
                consecutive_failures = max_consecutive_failures
            
            if not ret or frame is None:
                consecutive_failures += 1
                
                # Alternative HTTP error detection without using FFmpeg debug output
                # We'll rely on the pattern of consecutive failures and refresh URL more frequently when failures occur
                if consecutive_failures >= 2:
                    # Multiple consecutive failures often indicate connection issues with YouTube servers
                    current_time = time.time()
                    # Consider this likely to be an HTTP error after multiple consecutive failures
                    if current_time - last_error_time > 5:  # Avoid counting the same error burst multiple times
                        http_errors_count += 1
                        last_error_time = current_time
                        log_youtube_connection_status('error', error="Connection interrupted")
                
                if consecutive_failures >= max_consecutive_failures:
                    log_youtube_connection_status('retry')
                    logging.warning(f"Failed to read frames from video source ({consecutive_failures} consecutive failures)")
                    consecutive_failures = 0
                    # Try to reopen the capture if we're consistently failing
                    if current_time - last_reconnect_attempt > reconnect_interval:
                        logging.warning("Too many consecutive failures, attempting to reconnect...")
                        # Force a URL refresh more aggressively when consecutive failures occur
                        if original_youtube_url and current_time - last_url_refresh_time > 60:  # Only wait 1 minute if failing
                            new_url = refresh_youtube_url(original_youtube_url)
                            if new_url:
                                current_stream_url = new_url
                                logging.debug("Refreshed URL due to consecutive failures")
                                open_capture(new_url)
                                last_url_refresh_time = current_time
                            else:
                                open_capture()
                        else:
                            open_capture()
                time.sleep(min(0.1, reconnection_backoff / 2))  # Adaptive sleep on failure
                continue
            
            consecutive_failures = 0
            frames_read += 1
            
            try:
                # If buffer is extremely full, skip this frame
                buffer_fullness = frame_buffer.qsize() / buffer_size
                if buffer_fullness > 0.95:  # > 95% full
                    frames_dropped += 1
                    continue
                
                # If buffer is full, remove the oldest frame
                if frame_buffer.full():
                    with frame_buffer_lock:
                        try:
                            # Non-blocking, just discard if full
                            frame_buffer.get_nowait()
                            frames_dropped += 1
                        except queue.Empty:
                            pass
                
                # Add the new frame to the buffer
                with frame_buffer_lock:
                    frame_buffer.put(frame.copy(), block=False)
                    
                    # Log when buffer is filled for the first time
                    current_size = frame_buffer.qsize()
                    if not buffer_filled and current_size >= buffer_size * 0.5:  # 50% full
                        buffer_filled = True
                        logging.info(f"Frame buffer filled to {current_size}/{buffer_size} frames")
                    
                # Log status periodically
                if frames_read % 300 == 0:  # Log every ~10 seconds at 30fps
                    logging.info(f"Frame reader stats: Read {frames_read}, Dropped {frames_dropped}, Buffer size {frame_buffer.qsize()}/{buffer_size}")
                    
            except queue.Full:
                frames_dropped += 1
                continue
            except Exception as e:
                logging.error(f"Error in frame reader thread: {e}")
                continue
                
            # Adaptive sleep based on buffer fullness
            if buffer_fullness < 0.2:  # Buffer is nearly empty
                # Don't sleep, read as fast as possible to fill buffer
                pass
            elif buffer_fullness < 0.5:  # Buffer is somewhat empty
                time.sleep(0.001)  # Very short sleep
            else:  # Buffer is at least half full
                time.sleep(0.005)  # Longer sleep to match consumption rate
        
        # Clean up resources
        if thread_cap is not None:
            thread_cap.release()
            
        logging.info(f"Frame reader thread stopped. Total frames read: {frames_read}, dropped: {frames_dropped}")
    
    frame_reader_thread = threading.Thread(target=frame_reader_worker, 
                                          name="FrameReader", 
                                          daemon=True)
    frame_reader_thread.start()
    return frame_reader_thread

def get_latest_frame():
    """
    Get the latest frame from the buffer.
    
    Returns:
        np.ndarray: The next frame from the buffer, or None if no frames are available
    """
    global frame_buffer, frame_buffer_lock, last_buffer_warning_time
    
    if frame_buffer is None or frame_buffer.empty():
        # Limit warning frequency to once per second
        current_time = time.time()
        if current_time - last_buffer_warning_time > 1.0:
            logging.warning("No frames available from buffer")
            last_buffer_warning_time = current_time
        return None
        
    with frame_buffer_lock:
        # IMPORTANT FIX: Don't clear the buffer, just get the next frame
        # This was the main issue causing stuttering - we were emptying the entire buffer every time
        try:
            frame = frame_buffer.get_nowait()
            return frame
        except queue.Empty:
            return None

def stop_frame_reader_thread():
    """Stop the frame reader thread."""
    global should_stop_frame_reader, frame_reader_thread, frame_buffer
    
    if frame_reader_thread is not None and frame_reader_thread.is_alive():
        should_stop_frame_reader = True
        frame_reader_thread.join(timeout=2.0)
        if frame_reader_thread.is_alive():
            logging.warning("Frame reader thread did not terminate gracefully")
            
    # Clear the frame buffer
    if frame_buffer is not None:
        with frame_buffer_lock:
            while not frame_buffer.empty():
                try:
                    frame_buffer.get_nowait()
                except queue.Empty:
                    break

if is_apple_silicon:
    try:
        import objc
        from Foundation import NSData, NSMutableData
        from Quartz import CIContext, CIImage, CIFilter, kCIFormatRGBA8, kCIContextUseSoftwareRenderer, CIVector
        from CoreFoundation import CFDataCreate
        
        logging.info("Detected Apple Silicon, attempting to initialize hardware acceleration")
        
        context_options = {
            kCIContextUseSoftwareRenderer: False,
            "kCIContextOutputColorSpace": None,
            "kCIContextWorkingColorSpace": None,
            "kCIContextHighQualityDownsample": False,
            "kCIContextOutputPremultiplied": True,
            "kCIContextCacheIntermediates": False,
            "kCIContextPriorityRequestLow": True,
            "kCIContextWorkingFormat": kCIFormatRGBA8,
            "kCIContextAllowLowPower": True,
            "kCIContextAllowReductions": True,
            "kCIContextUseSoftwareRenderer": False,
            "kCIContextUsesCoreGraphics": False,
            "kCIContextEnableMetalRenderingAtHighPerformance": True
        }
        
        with objc.autorelease_pool():
            ci_context = CIContext.contextWithOptions_(context_options)
        
        if ci_context is None:
            logging.error("CIContext initialization returned None")
            if force_hardware_acceleration and not allow_software_fallback:
                raise RuntimeError("CIContext initialization returned None and hardware acceleration is required")
        else:
            test_width, test_height = 4, 4
            test_img = np.zeros((test_height, test_width, 4), dtype=np.uint8)
            test_img[:,:] = (255, 0, 0, 255)
            
            if not test_img.flags['C_CONTIGUOUS']:
                test_img = np.ascontiguousarray(test_img)
            
            with objc.autorelease_pool():
                test_data = test_img.tobytes()
                data_provider = CFDataCreate(None, test_data, len(test_data))
                
                if data_provider is not None:
                    test_ci_image = CIImage.imageWithBitmapData_bytesPerRow_size_format_colorSpace_(
                        data_provider, test_width * 4, Quartz.CGSizeMake(test_width, test_height), kCIFormatRGBA8, None)
                    del data_provider
                    
                    if test_ci_image is not None:
                        optimized_image = test_ci_image.imageBySettingProperties_({"CIImageAppleM1Optimized": True})
                        del test_ci_image
                        
                        if optimized_image is not None:
                            logging.info("Hardware acceleration test successful!")
                            hardware_acceleration_available = True
                            del optimized_image
                        else:
                            logging.warning("Failed to optimize test image for Apple M1")
                    else:
                        logging.warning("Failed to create CIImage from test data")
            gc.collect()
    except ImportError as e:
        logging.error(f"Failed to import Apple-specific modules: {e}\n{traceback.format_exc()}")
        if force_hardware_acceleration and not allow_software_fallback:
            raise RuntimeError("Hardware acceleration required but unavailable. Set ALLOW_SOFTWARE_FALLBACK=true to continue.")
        logging.warning("Hardware acceleration unavailable, falling back to software processing.")
    except Exception as e:
        logging.error(f"Unexpected error initializing CIContext: {e}\n{traceback.format_exc()}")
        if force_hardware_acceleration and not allow_software_fallback:
            raise RuntimeError("Hardware acceleration setup failed. Set ALLOW_SOFTWARE_FALLBACK=true to continue.")
        logging.warning("Hardware acceleration setup failed, falling back to software processing.")
else:
    if force_hardware_acceleration and not allow_software_fallback:
        logging.error("Hardware acceleration required but not on Apple Silicon")
        raise RuntimeError("Hardware acceleration required but not on Apple Silicon. Set ALLOW_SOFTWARE_FALLBACK=true to continue.")
    logging.info("Not on Apple Silicon, using software processing")

# Global variables
grid_size = 2
depth = 1
running = True
debug_mode = False
show_info = True
info_hidden_time = 0
mode = "fractal_depth"  # Default mode is fractal_depth
fractal_grid_size = 3  # Starting grid size for fractal mode
fractal_debug = False  # Debug toggle for fractal mode
fractal_source_position = 2  # Position of source in fractal mode
prev_output_frame = None
fractal_depth = 1  # Depth for fractal_depth mode, starting at 1

# Stats
frame_count = 0
processed_count = 0
displayed_count = 0
dropped_count = 0

# Performance tracking
downsample_times = []
downsample_sizes = []
max_tracked_samples = 50

def conditional_profile(func):
    """Only apply @profile decorator if memory tracing is enabled."""
    global enable_memory_tracing
    def dummy_decorator(f):
        return f
    decorator = profile if enable_memory_tracing else dummy_decorator
    return decorator(func)

def create_fractal_grid(live_frame, prev_output, grid_size, source_position=1):
    """
    Create an NxN grid for the infinite fractal effect:
    - Source position 1: Live frame in top-left cell (default)
    - Source position 2: Live frame in center cell (or nearest to center)
    - Source position 3: Live frame in top-right cell
    
    Args:
        live_frame (np.ndarray): The current frame from the video stream.
        prev_output (np.ndarray): The previous output frame of the grid (for recursion).
        grid_size (int): Size of the grid (e.g., 2 for 2x2, 3 for 3x3, etc.).
        source_position (int): Position of the source frame (1=top-left, 2=center, 3=top-right)
    
    Returns:
        np.ndarray: The resulting NxN grid frame.
    """
    if prev_output is None:
        prev_output = np.zeros_like(live_frame)
    
    h, w = live_frame.shape[:2]
    cell_h = h // grid_size
    cell_w = w // grid_size
    
    # Create output array - reuse memory allocation when possible by using zeros_like
    grid_frame = np.zeros_like(live_frame)
    
    # Determine the position of the source frame based on source_position
    source_i, source_j = 0, 0  # Default: top-left (position 1)
    
    if source_position == 2:  # Center
        if grid_size == 2:
            source_i = 0
            source_j = 1  # top-right in a 2x2 grid
        elif grid_size % 2 == 1:
            source_i = source_j = grid_size // 2
        else:
            center = grid_size / 2 - 0.5
            source_i = int(center)
            source_j = int(center)
    elif source_position == 3:  # Top-right
        source_i = 0
        source_j = grid_size - 1
    
    # Buffer for resized cells to avoid redundant memory allocations
    resized_source = None
    resized_prev = None
    
    try:
        for i in range(grid_size):
            for j in range(grid_size):
                y_start = i * cell_h
                y_end = (i + 1) * cell_h if i < grid_size - 1 else h
                x_start = j * cell_w
                x_end = (j + 1) * cell_w if j < grid_size - 1 else w
                
                cell_width = x_end - x_start
                cell_height = y_end - y_start
                
                if i == source_i and j == source_j:
                    # Source cell: Live frame
                    # Reuse buffer if dimensions match
                    if resized_source is None or resized_source.shape[:2] != (cell_height, cell_width):
                        resized_source = np.empty((cell_height, cell_width, 3), dtype=np.uint8)
                    cv2.resize(live_frame, (cell_width, cell_height), dst=resized_source)
                    
                    # Copy to output
                    grid_frame[y_start:y_end, x_start:x_end] = resized_source
                else:
                    # Other cells: Previous output
                    # Reuse buffer if dimensions match
                    if resized_prev is None or resized_prev.shape[:2] != (cell_height, cell_width):
                        resized_prev = np.empty((cell_height, cell_width, 3), dtype=np.uint8)
                    cv2.resize(prev_output, (cell_width, cell_height), dst=resized_prev)
                    
                    # Copy to output
                    grid_frame[y_start:y_end, x_start:x_end] = resized_prev
        
        return grid_frame
    except Exception as e:
        logging.error(f"Error in create_fractal_grid: {e}")
        return live_frame.copy()  # Fallback to original frame on error
    finally:
        # Clean up temporary buffers
        if resized_source is not None:
            del resized_source
        if resized_prev is not None:
            del resized_prev

def compute_fractal_depth(live_frame, depth):
    """
    Compute the fractal output for a given depth in fractal_depth mode.
    - Depth 0: Returns a black image (base case).
    - Depth d: Creates a 2x2 grid with live_frame in top-right and the previous depth's output in other cells.
    
    Args:
        live_frame (np.ndarray): The current frame from the video stream.
        depth (int): The number of fractal depth levels to apply.
    
    Returns:
        np.ndarray: The resulting frame after applying the specified depth.
    """
    if depth == 0:
        return np.zeros_like(live_frame)
    else:
        prev = compute_fractal_depth(live_frame, depth - 1)
        return create_fractal_grid(live_frame, prev, 2, 3)

def handle_keyboard_event(key_name, mod=None):
    """Handle keyboard inputs for adjusting grid settings and mode switching."""
    global grid_size, depth, debug_mode, show_info, info_hidden_time, mode, fractal_grid_size, fractal_debug
    global fractal_source_position, cap, fractal_depth, prev_frames

    old_grid_size = grid_size
    old_depth = depth
    old_debug_mode = debug_mode
    old_fractal_grid_size = fractal_grid_size
    old_fractal_source_position = fractal_source_position
    
    try:
        is_repeat = key_name in ['up', 'down'] and key_pressed.get(key_name, False) and time.time() - key_press_start.get(key_name, 0) > key_repeat_delay
        
        if key_name == '4' and (not mod or not (mod & pygame.KMOD_SHIFT)) and mode == "fractal":
            mode = "fractal_depth"
            fractal_depth = 1
            logging.info("Mode changed to: fractal_depth (Mode [4])")
            depth_info = get_fractal_depth_breakdown(fractal_depth)
            for info in depth_info:
                logging.info(f"  {info}")
            logging.info("Use UP/DOWN arrow keys to increase/decrease depth level (1-100)")
        
        elif mode == "fractal_depth" and key_name == 'up':
            old_fractal_depth = fractal_depth
            fractal_depth = min(100, fractal_depth + 1)
            if old_fractal_depth != fractal_depth:
                # Extend prev_frames if needed
                if len(prev_frames) < fractal_depth:
                    prev_frames.extend([None] * (fractal_depth - len(prev_frames)))
                
                depth_info = get_fractal_depth_breakdown(fractal_depth)
                logging.info(f"Fractal depth set to {fractal_depth}")
                for info in depth_info:
                    logging.info(f"  {info}")
        
        elif mode == "fractal_depth" and key_name == 'down':
            old_fractal_depth = fractal_depth
            fractal_depth = max(1, fractal_depth - 1)
            if old_fractal_depth != fractal_depth:
                # Clean up unused frames when depth decreases
                if old_fractal_depth > fractal_depth:
                    # Set unused frames to None to help with garbage collection
                    for i in range(fractal_depth, len(prev_frames)):
                        prev_frames[i] = None
                    # Truncate the list to match the new depth
                    prev_frames[fractal_depth:] = []
                    gc.collect()
                
                depth_info = get_fractal_depth_breakdown(fractal_depth)
                logging.info(f"Fractal depth set to {fractal_depth}")
                for info in depth_info:
                    logging.info(f"  {info}")
        
        elif key_name == 'f':
            if mode == "grid":
                mode = "fractal"
            elif mode == "fractal":
                mode = "fractal_depth"
            else:
                mode = "grid"
            logging.info(f"Mode changed to: {mode}")
        
        elif key_name == 'up':
            if mode == "fractal":
                if fractal_source_position == 2:
                    if fractal_grid_size == 2:
                        fractal_grid_size = 3
                        logging.info(f"Grid size changed to 3x3 (with center source position)")
                    elif fractal_grid_size % 2 == 0:
                        fractal_grid_size += 1
                    else:
                        fractal_grid_size += 2
                else:
                    fractal_grid_size += 1
                
                frame_width, frame_height = 1280, 720
                visible_screens, is_resolution_limited, max_level, screens_by_category = calculate_visible_screens(fractal_grid_size, frame_width, frame_height)
                visible_recursive_cells = calculate_visible_recursive_cells(fractal_grid_size, frame_width, frame_height)
                
                resolution_text = ""
                if is_resolution_limited:
                    smallest_pixel_size = min(frame_width, frame_height) / (fractal_grid_size ** max_level)
                    resolution_text = f" (at {frame_width}x{frame_height} resolution, limited by {smallest_pixel_size:.2f} pixels per cell)"
                
                if not is_repeat or fractal_grid_size % 5 == 0:
                    logging.info(f"Fractal grid size: {fractal_grid_size}x{fractal_grid_size}")
                    logging.info(f"Infinite Screens: {visible_recursive_cells:,} clusters of screens, each to infinity = {visible_recursive_cells:,} infinities of screens")
            elif mode == "grid":
                grid_size += 1
                if not is_repeat or grid_size % 5 == 0:
                    logging.info(f"Grid size changed: {old_grid_size}x{old_grid_size} → {grid_size}x{grid_size}")
        
        elif key_name == 'down':
            if mode == "fractal":
                if fractal_source_position == 2:
                    if fractal_grid_size == 3:
                        fractal_grid_size = 2
                        logging.info(f"Grid size changed to 2x2 (with top-right source position)")
                    elif fractal_grid_size <= 2:
                        fractal_grid_size = 2
                    else:
                        if fractal_grid_size % 2 == 0:
                            fractal_grid_size -= 1
                        else:
                            fractal_grid_size -= 2
                else:
                    fractal_grid_size = max(1, fractal_grid_size - 1)
                
                frame_width, frame_height = 1280, 720
                visible_screens, is_resolution_limited, max_level, screens_by_category = calculate_visible_screens(fractal_grid_size, frame_width, frame_height)
                visible_recursive_cells = calculate_visible_recursive_cells(fractal_grid_size, frame_width, frame_height)
                
                resolution_text = ""
                if is_resolution_limited:
                    smallest_pixel_size = min(frame_width, frame_height) / (fractal_grid_size ** max_level)
                    resolution_text = f" (at {frame_width}x{frame_height} resolution, limited by {smallest_pixel_size:.2f} pixels per cell)"
                
                if not is_repeat or fractal_grid_size % 5 == 0:
                    logging.info(f"Fractal grid size: {fractal_grid_size}x{fractal_grid_size}")
                    logging.info(f"Infinite Screens: {visible_recursive_cells:,} clusters of screens, each to infinity = {visible_recursive_cells:,} infinities of screens")
            elif mode == "grid":
                grid_size = max(1, grid_size - 1)
                if not is_repeat or grid_size % 5 == 0:
                    logging.info(f"Grid size changed: {old_grid_size}x{old_grid_size} → {grid_size}x{grid_size}")
        
        elif key_name in '1234567890':
            if mode == "grid":
                old_depth = depth
                depth = int(key_name) if key_name != '0' else 10
                logging.info(f"Recursion depth changed: {old_depth} → {depth}")
            elif mode == "fractal" and key_name in '123':
                old_pos = fractal_source_position
                fractal_source_position = int(key_name)
                
                if fractal_source_position == 2 and fractal_grid_size % 2 == 0 and fractal_grid_size != 2:
                    fractal_grid_size += 1
                    logging.info(f"Adjusted grid size to {fractal_grid_size} for center positioning")
                
                position_desc = {1: "top-left", 2: "center", 3: "top-right"}
                if fractal_source_position == 2 and fractal_grid_size == 2:
                    logging.info(f"Fractal source position changed: {position_desc[old_pos]} → top-right (special 2x2 case)")
                else:
                    logging.info(f"Fractal source position changed: {position_desc[old_pos]} → {position_desc[fractal_source_position]}")
        
        elif key_name == 'd':
            if mode == "fractal" or mode == "fractal_depth":
                fractal_debug = not fractal_debug
                logging.info(f"Fractal debug mode: {'enabled' if fractal_debug else 'disabled'}")
            else:
                debug_mode = not debug_mode
                logging.info(f"Debug mode: {'enabled' if debug_mode else 'disabled'}")
        
        elif key_name == 's':
            show_info = not show_info
            logging.info(f"Info overlay: {'shown' if show_info else 'hidden'}")
            if not show_info:
                info_hidden_time = time.time()
        
        if mode == "grid" and (old_grid_size != grid_size or old_depth != depth):
            frame_h, frame_w = 720, 1280
            cell_h = frame_h // grid_size
            cell_w = frame_w // grid_size
            
            frame_size_mb = (frame_h * frame_w * 3) / (1024 * 1024)
            cell_size_mb = frame_size_mb / (grid_size * grid_size)
            
            logging.info(f"Configuration details for Grid {grid_size}x{grid_size}, Depth {depth}:")
            logging.info(f"  - Cell size: ~{cell_w}x{cell_h} pixels")
            logging.info(f"  - Total cells: {grid_size * grid_size} per frame")
            logging.info(f"  - Estimated memory per frame: {frame_size_mb:.2f} MB")
            logging.info(f"  - Estimated memory per cell: {cell_size_mb:.4f} MB")
    except Exception as e:
        logging.error(f"Error in handle_keyboard_event with key '{key_name}': {e}\n{traceback.format_exc()}")

def signal_handler(sig, frame):
    """Handle keyboard interrupts and other signals."""
    global running
    signal_name = signal.Signals(sig).name
    logging.info(f"Received signal {signal_name} ({sig}), shutting down gracefully...")
    running = False

def cv_to_ci_image(cv_img):
    """Convert OpenCV image to Core Image with autorelease pool for memory management."""
    with objc.autorelease_pool():
        data_provider = None
        ci_img = None
        optimized_ci_img = None
        data = None
        cv_img_rgba = None
        result = None
        
        try:
            if cv_img is None or cv_img.size == 0:
                logging.error("Input image to cv_to_ci_image is None or empty")
                return None
            
            if not hardware_acceleration_available:
                if force_hardware_acceleration:
                    logging.warning("Hardware acceleration unavailable but forced - attempting to use CoreImage anyway")
                else:
                    logging.debug("Hardware acceleration unavailable, using software processing")
            
            if len(cv_img.shape) < 3:
                cv_img_rgba = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGRA)
            elif cv_img.shape[2] == 3:
                cv_img_rgba = cv2.cvtColor(cv_img, cv2.COLOR_BGR2BGRA)
            else:
                cv_img_rgba = cv_img.copy()
            
            if cv_img_rgba is None or cv_img_rgba.size == 0:
                logging.error("Failed to convert image to RGBA format")
                return None
                
            height, width = cv_img_rgba.shape[:2]
            
            if not cv_img_rgba.flags['C_CONTIGUOUS']:
                temp = cv_img_rgba
                cv_img_rgba = np.ascontiguousarray(cv_img_rgba)
                del temp
            
            data = cv_img_rgba.tobytes()
            
            data_provider = CFDataCreate(None, data, len(data))
            if data_provider is None:
                error_msg = "Failed to create CFData for CIImage"
                logging.error(error_msg)
                if force_hardware_acceleration and not allow_software_fallback:
                    raise RuntimeError(f"{error_msg} and hardware acceleration is required")
                return None
            
            ci_img = CIImage.imageWithBitmapData_bytesPerRow_size_format_colorSpace_(
                data_provider, width * 4, Quartz.CGSizeMake(width, height), kCIFormatRGBA8, None)
            
            del data_provider
            data_provider = None
            del data
            data = None
            del cv_img_rgba
            cv_img_rgba = None
            
            if ci_img is None:
                error_msg = "CIImage creation failed"
                logging.error(error_msg)
                if force_hardware_acceleration and not allow_software_fallback:
                    raise RuntimeError(f"{error_msg} and hardware acceleration is required")
                return None
            
            optimized_ci_img = ci_img.imageBySettingProperties_({"CIImageAppleM1Optimized": True})
            
            if optimized_ci_img is not None:
                result = optimized_ci_img
                del ci_img
                ci_img = None
            else:
                logging.warning("Failed to optimize CIImage for Apple M1, using unoptimized image")
                result = ci_img
                ci_img = None
            
            if ci_context is not None:
                try:
                    ci_context.clearCaches()
                except Exception as e:
                    logging.warning(f"Failed to clear CI context cache: {e}")
            
            gc.collect()
            
            return result
                
        except Exception as e:
            error_msg = f"Error in cv_to_ci_image: {e}"
            logging.error(f"{error_msg}\n{traceback.format_exc()}")
            if force_hardware_acceleration and not allow_software_fallback:
                raise RuntimeError(f"{error_msg} and hardware acceleration is required")
            return None
        finally:
            if data_provider is not None:
                del data_provider
            if data is not None:
                del data
            if cv_img_rgba is not None:
                del cv_img_rgba
            if ci_img is not None:
                del ci_img
            if optimized_ci_img is not None and optimized_ci_img != result:
                del optimized_ci_img

def ci_to_cv_image(ci_img, width, height):
    """Convert Core Image to OpenCV image with autorelease pool for memory management."""
    with objc.autorelease_pool():
        output_data = None
        buffer = None
        buffer_copy = None
        cv_img_result = None
        
        try:
            if ci_img is None:
                logging.error("Input CIImage to ci_to_cv_image is None")
                return None
            
            if width <= 0 or height <= 0:
                logging.error(f"Invalid dimensions for ci_to_cv_image: {width}x{height}")
                return None
                
            if ci_context is None:
                error_msg = "CIContext is None, cannot render CIImage"
                logging.error(error_msg)
                if force_hardware_acceleration and not allow_software_fallback:
                    raise RuntimeError(f"{error_msg} and hardware acceleration is required")
                return None
            
            try:
                buffer_size = height * width * 4
                aligned_size = ((buffer_size + 15) // 16) * 16
                output_data = NSMutableData.dataWithLength_(aligned_size)
                if output_data is None:
                    error_msg = "Failed to allocate NSMutableData"
                    logging.error(error_msg)
                    if force_hardware_acceleration and not allow_software_fallback:
                        raise RuntimeError(f"{error_msg} and hardware acceleration is required")
                    return None
            except Exception as e:
                logging.error(f"Failed to create output buffer: {e}")
                return None
                
            try:
                ci_context.render_toBitmap_rowBytes_bounds_format_colorSpace_(
                    ci_img, output_data.mutableBytes(), width * 4, Quartz.CGRectMake(0, 0, width, height),
                    kCIFormatRGBA8, None)
                
                ci_context.clearCaches()
                ci_img = None
                gc.collect()
                
            except Exception as e:
                error_msg = f"CIContext.render failed: {e}"
                logging.error(error_msg)
                if force_hardware_acceleration and not allow_software_fallback:
                    raise RuntimeError(f"{error_msg} and hardware acceleration is required")
                return None
            
            try:
                buffer = np.frombuffer(output_data, dtype=np.uint8)
            except Exception as e:
                logging.error(f"Failed to create buffer from NSMutableData: {e}")
                return None
                
            expected_size = height * width * 4
            actual_size = buffer.size
            
            if actual_size != expected_size:
                mismatch_details = (
                    f"Buffer size mismatch: expected {expected_size} ({width}x{height}x4), "
                    f"got {actual_size} (diff: {actual_size - expected_size} bytes)."
                )
                logging.warning(f"{mismatch_details} Adjusting dimensions.")
                
                try:
                    adjusted_height = int(np.sqrt((actual_size / 4) * (height / width)))
                    adjusted_width = int(actual_size / (4 * adjusted_height))
                    height, width = adjusted_height, adjusted_width
                except Exception as e:
                    logging.error(f"Failed to adjust dimensions: {e}")
            
            try:
                if actual_size < height * width * 4:
                    max_pixels = actual_size // 4
                    adjusted_height = int(np.sqrt(max_pixels * (height / width)))
                    adjusted_width = max_pixels // adjusted_height
                    height, width = adjusted_height, adjusted_width
                    
                buffer_copy = buffer[:height * width * 4].reshape(height, width, 4).copy()
            except Exception as e:
                logging.error(f"Failed to reshape buffer: {e}")
                return np.zeros((height, width, 3), dtype=np.uint8)
                
            del buffer
            buffer = None
            del output_data
            output_data = None
            
            gc.collect()
            
            try:
                if buffer_copy.shape[2] == 4:
                    cv_img_result = cv2.cvtColor(buffer_copy, cv2.COLOR_BGRA2BGR)
                else:
                    cv_img_result = buffer_copy[:, :, :3].copy()
            except Exception as e:
                logging.error(f"Failed to convert to BGR format: {e}")
                return np.zeros((height, width, 3), dtype=np.uint8)
                
            del buffer_copy
            buffer_copy = None
            
            gc.collect()
            
            return cv_img_result
        except Exception as e:
            logging.error(f"Error in ci_to_cv_image: {e}\n{traceback.format_exc()}")
            return np.zeros((height, width, 3), dtype=np.uint8)
        finally:
            if output_data is not None:
                del output_data
            if buffer is not None:
                del buffer
            if buffer_copy is not None:
                del buffer_copy
            if ci_context is not None:
                try:
                    ci_context.clearCaches()
                except:
                    pass

def fill_black_pixels(result, grid_size):
    """Fill black pixels in the result image."""
    try:
        if result is None or result.size == 0:
            logging.error("Input to fill_black_pixels is None or empty")
            return
        
        h, w = result.shape[:2]
        black_pixels = np.all(result == 0, axis=2)
        if np.any(black_pixels):
            num_black = np.count_nonzero(black_pixels)
            percent_black = (num_black / (h * w)) * 100
            logging.debug(f"Found {num_black} black pixels ({percent_black:.4f}% of frame)")
            if percent_black > 0.01:
                kernel = np.ones((3, 3), np.uint8)
                if grid_size <= 10:
                    mask = (~black_pixels).astype(np.uint8)
                    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
                    fill_mask = (dilated_mask == 1) & black_pixels
                else:
                    logging.debug("Using morphological operations for large grids")
                    mask = (~black_pixels).astype(np.uint8) * 255
                    dilated = cv2.dilate(mask, kernel, iterations=1)
                    fill_mask = (dilated > 0) & black_pixels
                for c in range(3):
                    channel = result[:, :, c]
                    dilated_channel = cv2.dilate(channel, kernel, iterations=1)
                    channel[fill_mask] = dilated_channel[fill_mask]
    except Exception as e:
        logging.error(f"Error in fill_black_pixels: {e}\n{traceback.format_exc()}")

def hardware_process_grid_cells(small_frame, previous_frame, grid_size):
    """Process grid cells after hardware downsampling."""
    global mode
    
    h, w = previous_frame.shape[:2]
    base_cell_h = h // grid_size
    base_cell_w = w // grid_size
    remainder_h = h % grid_size
    remainder_w = w % grid_size
    
    result = np.empty((h, w, 3), dtype=np.uint8)
    
    small_h, small_w = small_frame.shape[:2]
    if small_h <= 0 or small_w <= 0:
        logging.error(f"Invalid downsampled frame dimensions: {small_w}x{small_h}")
        return None
    
    frame_aspect = 16 / 9
    cell_count = 0
    total_cells = grid_size * grid_size
    
    for i in range(grid_size):
        for j in range(grid_size):
            cell_count += 1
            
            cell_h = base_cell_h + (1 if i < remainder_h else 0)
            cell_w = base_cell_w + (1 if j < remainder_w else 0)
            y_start = sum([base_cell_h + (1 if k < remainder_h else 0) for k in range(i)])
            y_end = y_start + cell_h
            x_start = sum([base_cell_w + (1 if k < remainder_w else 0) for k in range(j)])
            x_end = x_start + cell_w
            y_end = min(y_end, h)
            x_end = min(x_end, w)
            cell_h = y_end - y_start
            cell_w = x_end - x_start
            
            if cell_h < 1 or cell_w < 1:
                logging.warning(f"Cell too small: {cell_w}x{cell_h}, using average color")
                avg_color = np.mean(small_frame, axis=(0, 1)).astype(np.uint8)
                result[y_start:y_end, x_start:x_end] = avg_color
                continue
                
            try:
                if frame_aspect > cell_w / cell_h:
                    required_width = int(small_h * (cell_w / cell_h))
                    crop_x = int((small_w - required_width) / 2)
                    crop_x = max(0, crop_x)
                    crop_w = min(required_width, small_w - crop_x)
                    cropped = small_frame[:, crop_x:crop_x + crop_w] if crop_x + crop_w <= small_w else small_frame[:, :small_w]
                else:
                    required_height = int(small_w * (cell_h / cell_w))
                    crop_y = int((small_h - required_height) / 2)
                    crop_y = max(0, crop_y)
                    crop_h = min(required_height, small_h - crop_y)
                    cropped = small_frame[crop_y:crop_y + crop_h, :] if crop_y + crop_h <= small_h else small_frame[:small_h, :]
                    
                interpolation = cv2.INTER_LINEAR if (cropped.shape[0] < cell_h or cropped.shape[1] < cell_w) else cv2.INTER_AREA
                result[y_start:y_end, x_start:x_end] = cv2.resize(cropped, (cell_w, cell_h), interpolation=interpolation)
                
            except Exception as e:
                if mode == "grid":
                    logging.error(f"Error processing cell {i}x{j}: {e}")
                else:
                    logging.error(f"Error processing cell {i}x{j}: {e}")
                result[y_start:y_end, x_start:x_end] = 0
            
            gc_frequency = max(10, total_cells // 10)
            if cell_count % gc_frequency == 0:
                gc.collect()
                if grid_size > 16 and cell_count % (gc_frequency * 2) == 0 and ci_context is not None:
                    try:
                        ci_context.clearCaches()
                    except Exception:
                        pass
    
    return result

@conditional_profile
def generate_grid_frame(previous_frame, grid_size, current_depth):
    """Generate a grid frame for the specified depth level."""
    global mode
    
    try:
        if previous_frame is None or previous_frame.size == 0:
            if mode == "grid":
                logging.error(f"Depth {current_depth}: Previous frame is None or empty")
            else:
                logging.error("Previous frame is None or empty")
            return None
            
        h, w = previous_frame.shape[:2]
        if h <= 0 or w <= 0:
            if mode == "grid":
                logging.error(f"Depth {current_depth}: Invalid frame dimensions: {w}x{h}")
            else:
                logging.error(f"Invalid frame dimensions: {w}x{h}")
            return None
        
        base_cell_h = h // grid_size
        base_cell_w = w // grid_size
        remainder_h = h % grid_size
        remainder_w = w % grid_size

        result = np.empty((h, w, 3), dtype=np.uint8)

        frame_aspect = 16 / 9
        common_cell_h = base_cell_h
        common_cell_w = base_cell_w
        
        if frame_aspect > common_cell_w / common_cell_h:
            small_h = int(common_cell_h)
            small_w = int(small_h * frame_aspect)
        else:
            small_w = int(common_cell_w)
            small_h = int(small_w / frame_aspect)

        small_w = max(1, small_w)
        small_h = max(1, small_h)

        downsample_start_time = time.time()
        
        if is_apple_silicon and hardware_acceleration_available and current_depth > 1:
            try:
                h, w = previous_frame.shape[:2]
                ci_image = cv_to_ci_image(previous_frame)
                if ci_image is None:
                    if mode == "grid":
                        logging.error(f"Depth {current_depth}: Failed to convert frame to CIImage")
                    else:
                        logging.error("Failed to convert frame to CIImage")
                    raise ValueError("Failed to convert frame to CIImage")
                
                scale_filter = CIFilter.filterWithName_("CILanczosScaleTransform")
                if scale_filter is None:
                    if mode == "grid":
                        logging.error(f"Depth {current_depth}: Failed to create CILanczosScaleTransform filter")
                    else:
                        logging.error("Failed to create CILanczosScaleTransform filter")
                    raise ValueError("Failed to create scale filter")
                
                scale_filter.setValue_forKey_(ci_image, "inputImage")
                scale = 1.0 / grid_size
                scale_filter.setValue_forKey_(scale, "inputScale")
                scale_filter.setValue_forKey_(1.0, "inputAspectRatio")
                
                output_image = scale_filter.valueForKey_("outputImage")
                if output_image is None:
                    if mode == "grid":
                        logging.error(f"Depth {current_depth}: Failed to apply scale filter")
                    else:
                        logging.error("Failed to apply scale filter")
                    raise ValueError("Failed to apply scale filter")
                
                small_frame = ci_to_cv_image(output_image, small_w, small_h)
                if small_frame is not None:
                    small_h, small_w = small_frame.shape[:2]
                    if mode == "grid":
                        logging.debug(f"Depth {current_depth}: Hardware downsampled frame to {small_w}x{small_h}")
                    else:
                        logging.debug(f"Hardware downsampled frame to {small_w}x{small_h}")
                    return hardware_process_grid_cells(small_frame, previous_frame, grid_size)
                else:
                    if mode == "grid":
                        logging.error(f"Depth {current_depth}: Failed to convert CIImage back to CV image")
                    else:
                        logging.error("Failed to convert CIImage back to CV image")
                    raise ValueError("Failed to convert CIImage back to CV image")
            except Exception as e:
                if mode == "grid":
                    logging.error(f"Depth {current_depth}: Error in hardware downsampling: {e}\n{traceback.format_exc()}")
                else:
                    logging.error(f"Error in hardware downsampling: {e}\n{traceback.format_exc()}")
                
                if ci_image is not None:
                    del ci_image
                    ci_image = None
                if scale_filter is not None:
                    del scale_filter
                    scale_filter = None
                
                gc.collect()
                
                try:
                    small_frame = cv2.resize(previous_frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
                    logging.warning(f"Depth {current_depth}: Falling back to software downsampling")
                except Exception as resize_e:
                    logging.error(f"Depth {current_depth}: Software resize also failed: {resize_e}")
                    return None
        else:
            small_w_int = int(small_w)
            small_h_int = int(small_h)
            try:
                small_frame = cv2.resize(previous_frame, (small_w_int, small_h_int), interpolation=cv2.INTER_AREA)
            except cv2.error as e:
                logging.error(f"Depth {current_depth}: OpenCV resize failed: {e}")
                return None
            small_w = small_w_int
            small_h = small_h_int
            logging.debug(f"Depth {current_depth}: Software downsampled frame to {small_w}x{small_h}")
        
        if small_frame is None or small_frame.size == 0:
            if mode == "grid":
                logging.error(f"Depth {current_depth}: Failed to create valid downsampled frame")
            else:
                logging.error("Failed to create valid downsampled frame")
            return None

        small_h, small_w = small_frame.shape[:2]
        if small_h <= 0 or small_w <= 0:
            if mode == "grid":
                logging.error(f"Depth {current_depth}: Invalid downsampled frame dimensions: {small_w}x{small_h}")
            else:
                logging.error(f"Invalid downsampled frame dimensions: {small_w}x{small_h}")
            return None
        
        if mode == "grid":
            logging.debug(f"Depth {current_depth}: Using downsampled frame of size {small_w}x{small_h}")
        else:
            logging.debug(f"Using downsampled frame of size {small_w}x{small_h}")

        downsample_time = time.time() - downsample_start_time
        downsample_times.append(downsample_time)
        downsample_sizes.append((w * h, small_w * small_h))
        
        if len(downsample_times) > max_tracked_samples:
            downsample_times.pop(0)
            downsample_sizes.pop(0)
        
        cell_count = 0
        total_cells = grid_size * grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                cell_count += 1
                
                cell_h = base_cell_h + (1 if i < remainder_h else 0)
                cell_w = base_cell_w + (1 if j < remainder_w else 0)
                y_start = sum([base_cell_h + (1 if k < remainder_h else 0) for k in range(i)])
                y_end = y_start + cell_h
                x_start = sum([base_cell_w + (1 if k < remainder_w else 0) for k in range(j)])
                x_end = x_start + cell_w
                y_end = min(y_end, h)
                x_end = min(x_end, w)
                cell_h = y_end - y_start
                cell_w = x_end - x_start
                
                if cell_h < 1 or cell_w < 1:
                    if mode == "grid":
                        logging.warning(f"Depth {current_depth}: Cell too small: {cell_w}x{cell_h}, using average color")
                    else:
                        logging.warning(f"Cell too small: {cell_w}x{cell_h}, using average color")
                    avg_color = np.mean(small_frame, axis=(0, 1)).astype(np.uint8)
                    result[y_start:y_end, x_start:x_end] = avg_color
                    continue
                    
                try:
                    if frame_aspect > cell_w / cell_h:
                        required_width = int(small_h * (cell_w / cell_h))
                        crop_x = int((small_w - required_width) / 2)
                        crop_x = max(0, crop_x)
                        crop_w = min(required_width, small_w - crop_x)
                        cropped = small_frame[:, crop_x:crop_x + crop_w] if crop_x + crop_w <= small_w else small_frame[:, :small_w]
                    else:
                        required_height = int(small_w * (cell_h / cell_w))
                        crop_y = int((small_h - required_height) / 2)
                        crop_y = max(0, crop_y)
                        crop_h = min(required_height, small_h - crop_y)
                        cropped = small_frame[crop_y:crop_y + crop_h, :] if crop_y + crop_h <= small_h else small_frame[:small_h, :]
                        
                    interpolation = cv2.INTER_LINEAR if (cropped.shape[0] < cell_h or cropped.shape[1] < cell_w) else cv2.INTER_AREA
                    result[y_start:y_end, x_start:x_end] = cv2.resize(cropped, (cell_w, cell_h), interpolation=interpolation)
                    
                except Exception as e:
                    if mode == "grid":
                        logging.error(f"Error processing cell {i}x{j} at depth {current_depth}: {e}")
                    else:
                        logging.error(f"Error processing cell {i}x{j}: {e}")
                    result[y_start:y_end, x_start:x_end] = 0
                
                gc_frequency = max(10, total_cells // 10)
                if cell_count % gc_frequency == 0:
                    gc.collect()
                    if grid_size > 16 and cell_count % (gc_frequency * 2) == 0 and ci_context is not None:
                        try:
                            ci_context.clearCaches()
                        except Exception:
                            pass
        
        del small_frame
        small_frame = None
        
        gc.collect()
        
        fill_black_pixels(result, grid_size)

        if mode == "grid":
            logging.debug(f"Depth {current_depth}: Generated grid frame {w}x{h} with grid {grid_size}x{grid_size}")
        else:
            logging.debug(f"Generated grid frame {w}x{h} with grid {grid_size}x{grid_size}")
        
        return result
    except Exception as e:
        if mode == "grid":
            logging.error(f"Error in generate_grid_frame at depth {current_depth}: {e}\n{traceback.format_exc()}")
        else:
            logging.error(f"Error in generate_grid_frame: {e}\n{traceback.format_exc()}")
        return None

@conditional_profile
def apply_grid_effect(frame, grid_size, depth):
    """Apply recursive grid effect to the frame."""
    global mode
    previous_frame = None
    result = None
    
    try:
        if frame is None or frame.size == 0:
            logging.warning("apply_grid_effect received None or empty frame")
            return None
        if debug_mode or depth == 0:
            if mode == "grid":
                logging.debug("Depth 0: Returning original frame (debug mode or depth=0)")
            else:
                logging.debug("Returning original frame (debug mode)")
            return frame.copy()
            
        previous_frame = frame.copy()
        failed_depths = 0
        max_failed_depths = 2
        
        for d in range(1, depth + 1):
            if is_apple_silicon:
                import objc
                with objc.autorelease_pool():
                    new_frame = generate_grid_frame(previous_frame, grid_size, d)
                    
                    if new_frame is None:
                        if mode == "grid":
                            logging.error(f"Depth {d}: Failed to generate grid frame (returned None)")
                        else:
                            logging.error("Failed to generate grid frame (returned None)")
                        failed_depths += 1
                        if failed_depths > max_failed_depths:
                            if mode == "grid":
                                logging.error(f"Too many failures ({failed_depths}), aborting grid effect")
                            else:
                                logging.error(f"Too many failures ({failed_depths}), aborting effect")
                            result = previous_frame.copy()
                            del previous_frame
                            previous_frame = None
                            gc.collect()
                            return result
                        new_frame = previous_frame.copy()
                    else:
                        failed_depths = 0
                    
                    old_frame = previous_frame
                    previous_frame = new_frame
                    del old_frame
                    new_frame = None
                    
                    gc.collect()
                    
                    if is_apple_silicon and hardware_acceleration_available:
                        try:
                            ci_context.clearCaches()
                            if mode == "grid":
                                logging.debug(f"Depth {d}: Cleared Core Image cache after processing")
                            else:
                                logging.debug("Cleared Core Image cache after processing")
                        except Exception as e:
                            if mode == "grid":
                                logging.error(f"Depth {d}: Failed to clear Core Image cache: {e}")
                            else:
                                logging.error(f"Failed to clear Core Image cache: {e}")
                
                gc.collect()
            else:
                new_frame = generate_grid_frame(previous_frame, grid_size, d)
                
                if new_frame is None:
                    if mode == "grid":
                        logging.error(f"Depth {d}: Failed to generate grid frame (returned None)")
                    else:
                        logging.error("Failed to generate grid frame (returned None)")
                    failed_depths += 1
                    if failed_depths > max_failed_depths:
                        if mode == "grid":
                            logging.error(f"Too many failures ({failed_depths}), aborting grid effect")
                        else:
                            logging.error(f"Too many failures ({failed_depths}), aborting effect")
                        result = previous_frame.copy()
                        del previous_frame
                        previous_frame = None
                        gc.collect()
                        return result
                    new_frame = previous_frame.copy()
                else:
                    failed_depths = 0
                
                old_frame = previous_frame
                previous_frame = new_frame
                del old_frame
                new_frame = None
                
                gc.collect()
        
        result = previous_frame.copy()
        del previous_frame
        previous_frame = None
        gc.collect()
        return result
    except Exception as e:
        logging.error(f"Error in apply_grid_effect: {e}\n{traceback.format_exc()}")
        if previous_frame is not None and id(previous_frame) != id(frame):
            del previous_frame
        return frame.copy() if frame is not None else None
    finally:
        if previous_frame is not None and id(previous_frame) != id(result) and id(previous_frame) != id(frame):
            del previous_frame
        if is_apple_silicon and hardware_acceleration_available and ci_context is not None:
            try:
                ci_context.clearCaches()
            except:
                pass

def get_stream_url(url):
    """Retrieve streaming URL using yt-dlp with enhanced YouTube-specific options."""
    try:
        if not url or url.strip() == "":
            logging.error("Empty URL provided to get_stream_url function")
            return None
            
        logging.info(f"Attempting to get stream URL for: {url}")
        
        # Enhanced yt-dlp options for YouTube streaming
        yt_dlp_args = [
            "yt-dlp", 
            "-f", "best",                     # Get best quality stream
            "--get-url",                      # Only print the URL
            "--no-check-certificate",         # Skip HTTPS certificate validation
            "--no-cache-dir",                 # Don't use cache
            "--force-ipv4",                   # Force IPv4 to avoid some IPv6 issues
            "--no-playlist",                  # Don't process playlists
            "--extractor-retries", "3",       # Retry extraction 3 times
            "--referer", "https://www.youtube.com/",  # Set referer to avoid some blocks
            "--user-agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
            "--socket-timeout", "5",          # 5 second socket timeout
            url
        ]
        
        # Run yt-dlp with enhanced options
        logging.debug(f"Running yt-dlp with args: {' '.join(yt_dlp_args)}")
        result = subprocess.run(yt_dlp_args, capture_output=True, text=True, check=True)
        
        stream_url = result.stdout.strip()
        if not stream_url:
            logging.error("yt-dlp returned an empty stream URL")
            return None
            
        # Verify we have a valid HTTP(S) URL 
        if not stream_url.startswith(('http://', 'https://')):
            logging.error(f"yt-dlp returned an invalid URL format: {stream_url[:30]}...")
            return None
            
        logging.info(f"Successfully retrieved stream URL")
        return stream_url
    except subprocess.CalledProcessError as e:
        logging.error(f"yt-dlp failed with return code {e.returncode}: {e.stderr}")
        # Retry with simplified options if the initial request failed
        try:
            logging.info("Retrying with simplified options...")
            result = subprocess.run([
                "yt-dlp", 
                "-f", "best", 
                "--get-url",
                url
            ], capture_output=True, text=True, check=True)
            
            stream_url = result.stdout.strip()
            if stream_url:
                logging.info("Successfully retrieved stream URL on retry")
                return stream_url
        except:
            pass
        return None
    except FileNotFoundError:
        logging.error("yt-dlp not found. Please install yt-dlp.")
        return None
    except Exception as e:
        logging.error(f"Error getting stream URL: {e}\n{traceback.format_exc()}")
        return None

def get_grid_layer_breakdown(grid_size, max_depth):
    """
    Generate a breakdown of screens at each depth level of the grid effect.
    
    Args:
        grid_size (int): The size of the grid (NxN).
        max_depth (int): The maximum recursion depth.
        
    Returns:
        Tuple of (list of layer strings, total screens, formatted total)
    """
    layer_info = []
    total_screens = 0
    for d in range(1, max_depth + 1):
        screens_at_depth = grid_size ** (2 * d)
        total_screens += screens_at_depth
        layer_desc = (f"Grid frame {d} - depth {d} - {grid_size}x{grid_size} (total {screens_at_depth:,} screens)" 
                      if d == 1 else 
                      f"Grid frame {d} - depth {d} - {grid_size}x{grid_size} of frame {d-1} = (total {screens_at_depth:,} screens)")
        layer_info.append(layer_desc)
    number_units = [
        (1e6, "million"), (1e9, "billion"), (1e12, "trillion"), (1e15, "quadrillion"), (1e18, "quintillion"),
        (1e21, "sextillion"), (1e24, "septillion"), (1e27, "octillion"), (1e30, "nonillion"), (1e33, "decillion"),
        (1e36, "undecillion"), (1e39, "duodecillion"), (1e42, "tredecillion"), (1e45, "quattuordecillion"),
        (1e48, "quindecillion"), (1e51, "sexdecillion"), (1e54, "septendecillion"), (1e57, "octodecillion"),
        (1e60, "novemdecillion"), (1e63, "vigintillion")
    ]
    formatted_total = f"{total_screens:,}"
    for threshold, unit in reversed(number_units):
        if total_screens >= threshold:
            value = total_screens / threshold
            formatted_total = f"{total_screens:,} ({value:.2f} {unit})"
            break
    return layer_info, total_screens, formatted_total

def get_fractal_depth_breakdown(max_depth):
    """
    Generate a breakdown of what happens at each depth level of the fractal depth effect.
    
    Args:
        max_depth (int): The current fractal depth level.
        
    Returns:
        List of strings describing each depth level.
    """
    depth_info = []
    total_loops = 0
    
    for d in range(1, max_depth + 1):
        new_loops = 3**d - 3**(d-1) if d > 1 else 3
        total_loops += new_loops
        
        if d == 1:
            depth_info.append(f"DEPTH {d}: 2×2 grid with original video in top-right, 3 cells with infinite recursion")
        else:
            depth_info.append(f"DEPTH {d}: Each recursive cell from depth {d-1} becomes a 2×2 grid, adding {new_loops:,} new infinite loops")
    
    depth_info.append(f"TOTAL: {total_loops:,} unique infinite loops (with infinite sub-loops each) at depth {max_depth}")
    
    return depth_info

def start_memory_cleanup_thread():
    """Start a background thread to periodically clean memory."""
    global cleanup_thread_running, memory_cleanup_thread
    
    if cleanup_thread_running:
        logging.info("Memory cleanup thread already running")
        return
    
    try:
        import psutil
        import objc
        import gc
        import os
        import time
        import threading
        import logging
    except ImportError as e:
        logging.error(f"Failed to import required modules for memory cleanup thread: {e}")
        return
    
    def memory_cleanup_worker():
        global cleanup_thread_running, cleanup_stats, ci_context, hardware_acceleration_available, is_apple_silicon
        cleanup_thread_running = True
        cleanup_interval = 2.0
        memory_history = []
        history_size = 5
        
        cleanup_stats['current_interval'] = cleanup_interval
        
        try:
            current_process = psutil.Process(os.getpid())
        except Exception as e:
            logging.error(f"Failed to create process object for memory monitoring: {e}")
            current_process = None
        
        try:
            logging.debug("Starting periodic memory cleanup thread")
            while cleanup_thread_running:
                time.sleep(cleanup_interval)
                
                try:
                    if is_apple_silicon and hardware_acceleration_available and ci_context is not None:
                        try:
                            with objc.autorelease_pool():
                                ci_context.clearCaches()
                            logging.debug("Periodic Core Image cache cleanup")
                            cleanup_stats['total_cleanups'] += 1
                        except Exception as e:
                            logging.error(f"Error clearing CI cache in cleanup thread: {e}")
                    
                    try:
                        gc.collect()
                        if hasattr(gc, 'collect') and callable(getattr(gc, 'collect')):
                            gc.collect(2)
                    except Exception as e:
                        logging.error(f"Error during garbage collection in cleanup thread: {e}")
                    
                    try:
                        if current_process is not None:
                            memory_usage = current_process.memory_info().rss / 1024 / 1024
                            memory_history.append(memory_usage)
                            
                            cleanup_stats['last_memory'] = memory_usage
                            cleanup_stats['peak_memory'] = max(cleanup_stats['peak_memory'], memory_usage)
                            
                            if len(memory_history) > history_size:
                                memory_history.pop(0)
                            
                            growth_rate = 0
                            if len(memory_history) >= 2:
                                growth_rate = memory_history[-1] - memory_history[0]
                                cleanup_stats['last_growth_rate'] = growth_rate
                                
                            if memory_usage > 1500 or growth_rate > 100:
                                cleanup_interval = 1.0
                                if growth_rate > 200:
                                    logging.debug(f"Memory growing rapidly ({growth_rate:.2f} MB), performing extra cleanup")
                                    cleanup_stats['extra_cleanups'] += 1
                                    if ci_context is not None:
                                        try:
                                            ci_context.clearCaches()
                                        except Exception:
                                            pass
                                    try:
                                        import sys
                                        if hasattr(sys.intern, 'clear'):
                                            sys.intern.clear()
                                    except Exception as e:
                                        logging.debug(f"Could not clear interned strings: {e}")
                                    gc.collect()
                                    logging.debug("Recreated CIContext to free memory")
                                    memory_to_free = memory_usage - 1500
                                    if memory_to_free > 0:
                                        logging.debug(f"Attempting to free {memory_to_free:.2f} MB of memory")
                                        with objc.autorelease_pool():
                                            ci_context.clearCaches()
                                            minimal_options = {
                                                kCIContextUseSoftwareRenderer: False,
                                                "kCIContextCacheIntermediates": False,
                                                "kCIContextPriorityRequestLow": True
                                            }
                                            old_context = ci_context
                                            ci_context = CIContext.contextWithOptions_(minimal_options)
                                            del old_context
                                            old_context = None
                                        for _ in range(3):
                                            gc.collect()
                            elif memory_usage > 1000:
                                cleanup_interval = 1.5
                            else:
                                cleanup_interval = 2.0
                                
                            cleanup_stats['current_interval'] = cleanup_interval
                    except Exception as e:
                        logging.error(f"Error checking memory usage in cleanup thread: {e}")
                        cleanup_interval = 2.0
                        
                except Exception as e:
                    logging.error(f"Error in memory cleanup thread: {e}")
                    time.sleep(1.0)
        finally:
            cleanup_thread_running = False
            logging.debug("Memory cleanup thread stopped")
    
    memory_cleanup_thread = threading.Thread(target=memory_cleanup_worker, 
                                            name="MemoryCleanup", 
                                            daemon=True)
    memory_cleanup_thread.start()
    
def stop_memory_cleanup_thread():
    """Stop the memory cleanup thread."""
    global cleanup_thread_running, memory_cleanup_thread
    
    try:
        if cleanup_thread_running and memory_cleanup_thread is not None:
            logging.info("Stopping memory cleanup thread...")
            cleanup_thread_running = False
            
            if memory_cleanup_thread.is_alive():
                memory_cleanup_thread.join(timeout=1.0)
                
            memory_cleanup_thread = None
            logging.debug("Memory cleanup thread stopped")
        else:
            logging.debug("Memory cleanup thread not running or already stopped")
    except Exception as e:
        logging.error(f"Error stopping memory cleanup thread: {e}")
        cleanup_thread_running = False
        memory_cleanup_thread = None

def calculate_visible_screens(fractal_grid_size, frame_width, frame_height):
    """Calculate the number of visible screens in fractal mode, categorized by size."""
    if fractal_grid_size < 2:
        return 1, False, 0, {"larger_than_1px": 1, "exactly_1px": 0, "smaller_than_1px": False}
    
    min_dim = min(frame_width, frame_height)
    max_level = int(np.floor(np.log(min_dim) / np.log(fractal_grid_size)))
    r = fractal_grid_size ** 2 - 1
    
    screens_by_category = {
        "larger_than_1px": 0,
        "exactly_1px": 0,
        "smaller_than_1px": False
    }
    
    visible_screens = 0
    
    one_pixel_level = np.log(min_dim) / np.log(fractal_grid_size)
    one_pixel_level_floor = int(np.floor(one_pixel_level))
    one_pixel_level_ceil = int(np.ceil(one_pixel_level))
    
    floor_cell_size = min_dim / (fractal_grid_size ** one_pixel_level_floor)
    ceil_cell_size = min_dim / (fractal_grid_size ** one_pixel_level_ceil)
    
    exact_pixel_level = one_pixel_level_floor if abs(floor_cell_size - 1.0) < abs(ceil_cell_size - 1.0) else one_pixel_level_ceil
    
    for k in range(max_level + 2):
        cell_size = min_dim / (fractal_grid_size ** k)
        screens_at_level = r ** k
        
        if k <= max_level:
            visible_screens += screens_at_level
        
        if k == exact_pixel_level:
            screens_by_category["exactly_1px"] += screens_at_level
        elif cell_size > 1.0:
            screens_by_category["larger_than_1px"] += screens_at_level
        else:
            screens_by_category["smaller_than_1px"] = True
    
    smallest_cell_size = min_dim / (fractal_grid_size ** max_level)
    is_resolution_limited = smallest_cell_size < 1.0
    
    logging.debug(f"Min dimension: {min_dim}, Max level: {max_level}")
    logging.debug(f"One pixel level: {one_pixel_level}, Floor: {one_pixel_level_floor}, Ceil: {one_pixel_level_ceil}")
    logging.debug(f"Floor cell size: {floor_cell_size}, Ceil cell size: {ceil_cell_size}")
    logging.debug(f"Exact pixel level chosen: {exact_pixel_level}")
    
    return visible_screens, is_resolution_limited, max_level, screens_by_category

def calculate_visible_recursive_cells(fractal_grid_size, frame_width, frame_height):
    """Calculate the number of visible recursive cells in fractal mode."""
    if fractal_grid_size < 2:
        return 0
    
    min_dim = min(frame_width, frame_height)
    max_level = int(np.floor(np.log(min_dim) / np.log(fractal_grid_size)))
    r = fractal_grid_size ** 2 - 1
    visible_recursive_cells = 0
    
    for k in range(1, max_level + 1):
        visible_recursive_cells += r ** k
    
    return visible_recursive_cells

def main():
    """Main function with fixed-size frame surface and scaling for display."""
    global running, grid_size, depth, debug_mode, show_info, frame_count, processed_count, displayed_count, dropped_count
    global hardware_acceleration_available, force_hardware_acceleration, allow_software_fallback, is_apple_silicon
    global cap, mode, fractal_grid_size, fractal_debug, fractal_source_position, prev_output_frame, ci_context
    global enable_memory_tracing, fractal_depth, current_stream_url, last_buffer_warning_time, frame_drop_threshold
    global key_pressed, key_press_start, key_last_repeat, last_url_refresh_time, stream_url_refresh_interval
    global prev_frames
    
    key_pressed = {}
    key_press_start = {}
    key_last_repeat = {}
    
    screen = None
    cap = None
    frame = None
    processed_frame = None
    frame_surface = None
    font = None
    
    # Initialize previous frames for fractal depth
    prev_frames = [None] * fractal_depth  # Initialize only what's needed
    # Track last cleanup time for fractal processing
    last_fractal_cleanup_time = time.time()
    fractal_cleanup_interval = 10.0  # Cleanup every 10 seconds
    
    try:
        pygame.init()
        pygame.mixer.quit()
        pygame.display.set_caption("MultiMax Grid")
        display_info = pygame.display.Info()
        screen_width = min(1280, display_info.current_w - 100)
        screen_height = min(720, display_info.current_h - 100)
        screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
        
        screen.fill((255, 255, 255))
        pygame.display.flip()
        
        if is_apple_silicon and hardware_acceleration_available:
            start_memory_cleanup_thread()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Try to load environment variables from .env file
        load_dotenv()
        
        # Set default values for environment variables if not found in .env
        youtube_url = os.getenv('YOUTUBE_URL')
        if not youtube_url:
            # Default YouTube URL if .env is missing or variable not set
            youtube_url = "https://www.youtube.com/watch?v=ZzWBpGwKoaI"
            logging.warning("No YOUTUBE_URL found in .env file. Using default demo URL.")
            
            # Also set default values for other environment variables
            if os.getenv('FORCE_HARDWARE_ACCELERATION') is None:
                os.environ['FORCE_HARDWARE_ACCELERATION'] = 'true'
                force_hardware_acceleration = True
                logging.warning("No FORCE_HARDWARE_ACCELERATION found in .env file. Using default: true")
                
            if os.getenv('ALLOW_SOFTWARE_FALLBACK') is None:
                os.environ['ALLOW_SOFTWARE_FALLBACK'] = 'false'
                allow_software_fallback = False
                logging.warning("No ALLOW_SOFTWARE_FALLBACK found in .env file. Using default: false")
                
            if os.getenv('ENABLE_MEMORY_TRACING') is None:
                os.environ['ENABLE_MEMORY_TRACING'] = 'false'
                enable_memory_tracing = False
                logging.warning("No ENABLE_MEMORY_TRACING found in .env file. Using default: false")
        
        parser = argparse.ArgumentParser(description='Recursive Video Grid')
        parser.add_argument('--grid-size', type=int, default=3,
                            help='Grid size (odd numbers needed for center position in fractal mode)')
        parser.add_argument('--depth', type=int, default=1)
        parser.add_argument('--youtube-url', type=str, default=youtube_url)
        parser.add_argument('--log-level', type=str, default='INFO')
        parser.add_argument('--debug', action='store_true')
        parser.add_argument('--force-hardware', action='store_true')
        parser.add_argument('--allow-software', action='store_true')
        parser.add_argument('--enable-memory-tracing', action='store_true')
        parser.add_argument('--test-hardware-accel', action='store_true')
        parser.add_argument('--mode', type=str, choices=['grid', 'fractal', 'fractal_depth'], default='fractal_depth')
        parser.add_argument('--fractal-source', type=int, choices=[1, 2, 3], default=2,
                            help='Position of source in fractal mode (1=top-left, 2=center with odd grids and 2x2 special case, 3=top-right)')

        args = parser.parse_args()
        grid_size, depth = args.grid_size, args.depth
        mode = args.mode
        fractal_source_position = args.fractal_source
        fractal_grid_size = args.grid_size
        
        if mode == "fractal" and fractal_source_position == 2 and fractal_grid_size % 2 == 0 and fractal_grid_size != 2:
            fractal_grid_size += 1
            logging.info(f"Adjusted grid size to {fractal_grid_size} for center positioning")
        
        configure_logging(args.log_level)
        debug_mode = args.debug
        show_info = True
        
        if args.force_hardware:
            force_hardware_acceleration = True
            logging.info("Hardware acceleration forced via command line")
        if args.allow_software:
            allow_software_fallback = True
            logging.info("Software fallback allowed via command line")
        if args.enable_memory_tracing:
            enable_memory_tracing = True
            logging.info("Memory tracing enabled via command line")
            
        logging.info(f"Starting in mode: {mode.upper()}")
        if mode == "fractal_depth":
            depth_info = get_fractal_depth_breakdown(fractal_depth)
            for info in depth_info:
                logging.info(f"  {info}")
            logging.info("Use UP/DOWN arrow keys to increase/decrease depth level (1-100)")
        elif mode == "fractal":
            logging.info("Press 1-3 to change source position, 4 to switch to fractal depth mode")
        
        if args.test_hardware_accel:
            logging.info("=== Running hardware acceleration diagnostics ===")
            logging.info(f"Platform: {platform.system()}, Machine: {platform.machine()}")
            logging.info(f"Is Apple Silicon: {is_apple_silicon}")
            logging.info(f"Hardware acceleration forced: {force_hardware_acceleration}")
            logging.info(f"Software fallback allowed: {allow_software_fallback}")
            logging.info(f"Hardware acceleration available: {hardware_acceleration_available}")
            
            if is_apple_silicon:
                if ci_context is None:
                    logging.error("CIContext initialization failed")
                else:
                    logging.info("CIContext initialized successfully")
                    test_width, test_height = 16, 16
                    test_img = np.zeros((test_height, test_width, 4), dtype=np.uint8)
                    test_img[:,:] = (255, 0, 0, 255)
                    
                    logging.info("Testing cv_to_ci_image...")
                    ci_test_img = cv_to_ci_image(test_img)
                    if ci_test_img is not None:
                        logging.info("cv_to_ci_image test succeeded")
                        logging.info("Testing ci_to_cv_image...")
                        cv_result = ci_to_cv_image(ci_test_img, test_width, test_height)
                        if cv_result is not None:
                            logging.info("ci_to_cv_image test succeeded")
                            logging.info("All hardware acceleration tests passed!")
                        else:
                            logging.error("ci_to_cv_image test failed")
                    else:
                        logging.error("cv_to_ci_image test failed")
            
            logging.info("Hardware acceleration diagnostics complete, exiting")
            return
            
        logging.info(f"Starting with debug={debug_mode} (toggle with 'd')")
        logging.info(f"Hardware acceleration: forced={force_hardware_acceleration}, software fallback={allow_software_fallback}")
        logging.info(f"Memory tracing: {'enabled' if enable_memory_tracing else 'disabled'}")

        if enable_memory_tracing:
            tracemalloc.start()
            logging.info("Memory tracing started")
        
        process = psutil.Process()

        pygame.init()
        screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
        pygame.display.set_caption("Recursive Grid Livestream")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Arial", 24)
        
        screen.fill((255, 255, 255))
        loading_text = font.render("Loading Video...", True, (0, 0, 0))
        screen.blit(loading_text, (screen.get_width() // 2 - loading_text.get_width() // 2,
                                   screen.get_height() // 2 - loading_text.get_height() // 2))
        pygame.display.flip()

        stream_url = get_stream_url(args.youtube_url)
        if not stream_url:
            logging.error(f"Failed to get stream URL for {args.youtube_url}")
            logging.error("Please check that your YOUTUBE_URL in .env is valid and accessible")
            logging.error("Exiting program due to missing stream URL")
            return

        screen.fill((255, 255, 255))
        connecting_text = font.render("Connecting to Video Stream...", True, (0, 0, 0))
        please_wait_text = font.render("Please wait...", True, (0, 0, 0))
        screen.blit(connecting_text, (screen.get_width() // 2 - connecting_text.get_width() // 2,
                                     screen.get_height() // 2 - connecting_text.get_height() // 2))
        screen.blit(please_wait_text, (screen.get_width() // 2 - please_wait_text.get_width() // 2,
                                      screen.get_height() // 2 + 30))
        pygame.display.flip()

        # Initialize video capture
        logging.info(f"Initializing video capture for stream: {stream_url}")
        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            logging.error(f"Failed to open video stream: {stream_url}")
            logging.error("Please check your internet connection and YouTube URL validity")
            return
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_surface = pygame.Surface((frame_width, frame_height))
        
        # Start the frame reader thread with original YouTube URL for refreshes
        start_frame_reader_thread(cap, stream_url, youtube_url, buffer_size=frame_buffer_size)
        logging.info("Started background frame reader thread")

        last_frame_time = time.time()
        last_status_time = time.time()
        last_process_time = 0
        frames_since_last_log = 0
        successful_grids_since_last_log = 0
        min_process_time = float('inf')
        max_process_time = 0
        total_process_time = 0
        frame_counter = 0

        # Wait for buffer to start filling before entering main loop
        buffer_wait_start = time.time()
        buffer_wait_timeout = 5.0  # Wait up to 5 seconds for buffer to start filling
        while time.time() - buffer_wait_start < buffer_wait_timeout:
            if frame_buffer is not None and not frame_buffer.empty():
                logging.info(f"Buffer started filling after {time.time() - buffer_wait_start:.1f} seconds")
                break
            time.sleep(0.1)
            
            # Update screen with a loading message
            screen.fill((0, 0, 0))
            if font is None:
                font = pygame.font.SysFont('Arial', 24)
            loading_text = font.render(f"Loading stream... ({int(time.time() - buffer_wait_start)}s)", True, (255, 255, 255))
            screen.blit(loading_text, (screen.get_width() // 2 - loading_text.get_width() // 2, 
                                       screen.get_height() // 2 - loading_text.get_height() // 2))
            pygame.display.flip()

        while running:
            current_time = time.time()
            frame_start_time = time.time()

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                elif event.type == pygame.KEYDOWN:
                    key = pygame.key.name(event.key)
                    mod = event.mod
                    
                    if key in ['up', 'down', 'd', 's', 'f']:
                        handle_keyboard_event(key, mod)
                    elif key in '0123456789':
                        if mode != "fractal_depth":
                            handle_keyboard_event(key, mod)
                    
                    if key in ['up', 'down']:
                        key_pressed[key] = True
                        key_press_start[key] = current_time
                        key_last_repeat[key] = current_time
                elif event.type == pygame.KEYUP:
                    key = pygame.key.name(event.key)
                    if key in ['up', 'down']:
                        key_pressed[key] = False
                        key_last_repeat.pop(key, None)

            for key in ['up', 'down']:
                if key_pressed.get(key, False):
                    press_duration = current_time - key_press_start.get(key, current_time)
                    last_repeat_time = key_last_repeat.get(key, 0)
                    
                    if press_duration >= key_repeat_delay and (current_time - last_repeat_time) >= key_repeat_interval:
                        handle_keyboard_event(key)
                        key_last_repeat[key] = current_time

            # Get the latest frame from the buffer
            frame = get_latest_frame()
            if frame is None:
                # If no frame is available, wait a bit and continue
                time.sleep(0.01)  # Very short wait - faster recovery
                continue
            frame_count += 1

            process_start_time = time.time()
            if mode == "grid":
                if debug_mode:
                    processed_frame = frame.copy()
                else:
                    processed_frame = apply_grid_effect(frame, grid_size, depth)
            elif mode == "fractal":
                if fractal_debug:
                    processed_frame = frame.copy()
                else:
                    processed_frame = create_fractal_grid(frame, prev_output_frame, fractal_grid_size, fractal_source_position)
                    prev_output_frame = processed_frame.copy()
            elif mode == "fractal_depth":
                if fractal_debug:
                    processed_frame = frame.copy()
                else:
                    current_time = time.time()
                    try:
                        # Check if we need to clean up previous frames periodically
                        if current_time - last_fractal_cleanup_time > fractal_cleanup_interval:
                            logging.debug("Performing periodic cleanup of fractal depth memory")
                            for i in range(len(prev_frames)):
                                if prev_frames[i] is not None and i >= fractal_depth:
                                    prev_frames[i] = None
                            gc.collect()
                            last_fractal_cleanup_time = current_time
                            
                        temp = frame.copy()  # Start with a copy to avoid modifying the original
                        
                        if is_apple_silicon and hardware_acceleration_available:
                            # Use autorelease pool for Apple Silicon
                            import objc
                            with objc.autorelease_pool():
                                # Ensure prev_frames is properly sized
                                if len(prev_frames) < fractal_depth:
                                    prev_frames.extend([None] * (fractal_depth - len(prev_frames)))
                                
                                for d in range(fractal_depth):
                                    if prev_frames[d] is None:
                                        prev_frames[d] = np.zeros_like(frame)
                                    new_frame = create_fractal_grid(temp, prev_frames[d], 2, 3)
                                    
                                    # Clean up old frame before replacing
                                    old_frame = prev_frames[d]
                                    prev_frames[d] = new_frame.copy()
                                    
                                    # Update temp and clean up
                                    old_temp = temp
                                    temp = new_frame
                                    
                                    # Clean up references we no longer need
                                    del old_frame
                                    if d > 0:  # Don't delete the original frame
                                        del old_temp
                                    del new_frame
                                    
                                    # Force immediate garbage collection during deep recursion
                                    if d > 50 and d % 10 == 0:  # For very deep levels, collect more aggressively
                                        gc.collect()
                        else:
                            # For non-Apple Silicon, still implement good memory management
                            # Ensure prev_frames is properly sized
                            if len(prev_frames) < fractal_depth:
                                prev_frames.extend([None] * (fractal_depth - len(prev_frames)))
                            
                            for d in range(fractal_depth):
                                if prev_frames[d] is None:
                                    prev_frames[d] = np.zeros_like(frame)
                                new_frame = create_fractal_grid(temp, prev_frames[d], 2, 3)
                                
                                # Clean up old frame before replacing
                                old_frame = prev_frames[d]
                                prev_frames[d] = new_frame.copy()
                                
                                # Update temp and clean up
                                old_temp = temp
                                temp = new_frame
                                
                                # Clean up references we no longer need
                                del old_frame
                                if d > 0:  # Don't delete the original frame
                                    del old_temp
                                del new_frame
                                
                                # Force immediate garbage collection during deep recursion
                                if d > 50 and d % 10 == 0:  # For very deep levels, collect more aggressively
                                    gc.collect()
                                
                        processed_frame = temp
                        
                    except Exception as e:
                        logging.error(f"Error in fractal_depth processing: {e}\n{traceback.format_exc()}")
                        processed_frame = frame.copy()
            
            if processed_frame is None:
                logging.warning("Processed frame is None, using original frame")
                processed_frame = frame.copy()
            processed_count += 1
            last_process_time = time.time() - process_start_time
            frames_since_last_log += 1
            successful_grids_since_last_log += 1
            min_process_time = min(min_process_time, last_process_time)
            max_process_time = max(max_process_time, last_process_time)
            total_process_time += last_process_time

            try:
                if processed_frame.size == 0 or not hasattr(processed_frame, 'shape'):
                    logging.warning("Invalid processed frame, using original frame")
                    processed_frame = frame.copy()
                h, w = processed_frame.shape[:2]
                if h != frame_height or w != frame_width:
                    logging.error(f"Processed frame has unexpected dimensions: {w}x{h}, expected {frame_width}x{frame_height}")
                    continue
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                rgb_frame_swapped = rgb_frame.swapaxes(0, 1)
                pygame.surfarray.blit_array(frame_surface, rgb_frame_swapped)
                scaled_surface = pygame.transform.scale(frame_surface, screen.get_size())
                screen.blit(scaled_surface, (0, 0))
                displayed_count += 1
                
                if frame_count % 10 == 0 and ci_context is not None:
                    ci_context.clearCaches()
                
                if show_info:
                    hw_status = "Enabled" if hardware_acceleration_available else "Disabled"
                    if is_apple_silicon and force_hardware_acceleration:
                        hw_status += " (Forced)"
                    if not hardware_acceleration_available and allow_software_fallback:
                        hw_status += " (Software Fallback)"
                    texts = [
                        f"Mode: {mode.upper()}",
                        f"Grid: {grid_size if mode == 'grid' else fractal_grid_size if mode == 'fractal' else 2}x{grid_size if mode == 'grid' else fractal_grid_size if mode == 'fractal' else 2}",
                    ]
                    if mode == "grid":
                        texts.append(f"Depth: {depth}")
                    elif mode == "fractal_depth":
                        texts.append(f"Depth: {fractal_depth}")
                    else:
                        texts.append("Depth: N/A")
                    
                    if mode in ["fractal", "fractal_depth"]:
                        position_desc = {1: "top-left", 2: "center", 3: "top-right"}
                        if mode == "fractal" and fractal_source_position == 2 and fractal_grid_size == 2:
                            texts.append("Source: top-right (special 2x2 case)")
                        elif mode == "fractal":
                            texts.append(f"Source: {position_desc[fractal_source_position]}")
                        else:
                            texts.append("Source: top-right")
                    
                    texts.extend([
                        f"Debug: {'On' if (debug_mode if mode == 'grid' else fractal_debug) else 'Off'}",
                        f"Captured: {frame_count}, Displayed: {displayed_count}",
                        f"Processing: {last_process_time * 1000:.1f}ms",
                        f"FPS: {int(clock.get_fps())}",
                        f"Hardware: {hw_status}",
                    ])
                    if downsample_times:
                        avg_time = sum(downsample_times) / len(downsample_times) * 1000
                        if len(downsample_sizes) > 0:
                            last_original, last_small = downsample_sizes[-1]
                            reduction_ratio = last_original / max(1, last_small)
                            texts.append(f"Downsampling: {avg_time:.1f}ms, Reduction: {reduction_ratio:.1f}x")
                    if not debug_mode and depth > 0 and mode == "grid":
                        texts.append("")
                        texts.append("Grid Layer Breakdown:")
                        layer_info, total_screens, formatted_total = get_grid_layer_breakdown(grid_size, depth)
                        for layer in layer_info:
                            texts.append(f"  {layer}")
                        texts.append(f"  Total screens = {formatted_total}")
                    if mode == "fractal":
                        visible_screens, is_resolution_limited, max_level, screens_by_category = calculate_visible_screens(fractal_grid_size, frame_width, frame_height)
                        visible_recursive_cells = calculate_visible_recursive_cells(fractal_grid_size, frame_width, frame_height)
                        texts.append("")
                        texts.append("Fractal Screens:")
                        texts.append(f"  Infinite Screens: {visible_recursive_cells:,} clusters of screens, each to infinity = {visible_recursive_cells:,} infinities of screens")
                    elif mode == "fractal_depth":
                        texts.append("")
                        texts.append("Fractal Depth Breakdown:")
                        depth_info = get_fractal_depth_breakdown(fractal_depth)
                        for info in depth_info:
                            texts.append(f"  {info}")
                    texts.append("")
                    if mode == "grid":
                        texts.append("Press s to show/hide this info, d to debug, up/down to change grid size")
                        texts.append("Press 1-0 to set recursion depth (1-10)")
                    elif mode == "fractal":
                        texts.append("Press s to show/hide this info, d to debug, up/down grid size, f to switch modes")
                        texts.append("Press 1-3 to change source position: 1=top-left, 2=center (odd grids + 2x2 special case), 3=top-right")
                        texts.append("Press 4 to switch to fractal depth mode")
                    elif mode == "fractal_depth":
                        texts.append("Press s to show/hide this info, d to debug, f to switch modes")
                        texts.append("Press UP/DOWN arrow keys to increase/decrease depth level (1-100)")
                    line_height = 25
                    padding = 20
                    required_height = len(texts) * line_height + padding * 2
                    info_surface = pygame.Surface((screen.get_width(), required_height), pygame.SRCALPHA)
                    info_surface.fill((0, 0, 0, 128))
                    for i, text in enumerate(texts):
                        text_surface = font.render(text, True, (255, 255, 255))
                        info_surface.blit(text_surface, (10, padding + i * line_height))
                    screen.blit(info_surface, (0, 0))
                else:
                    time_since_hidden = current_time - info_hidden_time
                    if time_since_hidden < 5.0:
                        alpha = 128
                        if time_since_hidden > 4.0:
                            fade_progress = time_since_hidden - 4.0
                            alpha = int(128 * (1.0 - fade_progress))
                            alpha = max(0, min(128, alpha))
                        min_height = 40
                        min_info = pygame.Surface((screen.get_width(), min_height), pygame.SRCALPHA)
                        min_info.fill((0, 0, 0, alpha))
                        text_color = (255, 255, 255, min(255, alpha * 2))
                        help_text = font.render("Press 's' to show info", True, text_color)
                        min_info.blit(help_text, (10, 10))
                        screen.blit(min_info, (0, screen.get_height() - min_height))
                pygame.display.flip()
            except pygame.error as e:
                logging.error(f"Pygame display error: {e}\n{traceback.format_exc()}")
                dropped_count += 1
            except Exception as e:
                logging.error(f"Display error: {e}\n{traceback.format_exc()}")
                dropped_count += 1

            if is_apple_silicon and hardware_acceleration_available:
                try:
                    memory_usage = process.memory_info().rss / 1024 / 1024
                    
                    if memory_usage > 1000:
                        try:
                            import objc
                            with objc.autorelease_pool():
                                pass
                            logging.debug("Released Objective-C autorelease pool")
                        except Exception as e:
                            logging.warning(f"Failed to release autorelease pool: {e}")
                    
                    if memory_usage > 1200:
                        logging.debug(f"Medium memory usage detected: {memory_usage:.2f} MB - performing cleanup")
                        gc.collect()
                        ci_context.clearCaches()
                        
                        for var_name in list(locals().keys()):
                            if var_name not in ['ci_context', 'context_options', 'process', 'memory_usage'] and var_name[0] != '_':
                                if var_name in locals():
                                    del locals()[var_name]
                    
                    if memory_usage > 1800:
                        logging.debug(f"High memory usage detected: {memory_usage:.2f} MB - performing context recreation")
                        if is_apple_silicon:
                            try:
                                with objc.autorelease_pool():
                                    old_context = ci_context
                                    stricter_options = dict(context_options)
                                    stricter_options["kCIContextCacheIntermediates"] = False
                                    stricter_options["kCIContextPriorityRequestLow"] = True
                                    new_context = CIContext.contextWithOptions_(stricter_options)
                                    
                                    if new_context is not None:
                                        ci_context = new_context
                                        del old_context
                                        old_context = None
                                        gc.collect()
                                        logging.debug("Recreated CIContext to free memory")
                                        memory_to_free = memory_usage - 1500
                                        if memory_to_free > 0:
                                            logging.debug(f"Attempting to free {memory_to_free:.2f} MB of memory")
                                            with objc.autorelease_pool():
                                                ci_context.clearCaches()
                                                minimal_options = {
                                                    kCIContextUseSoftwareRenderer: False,
                                                    "kCIContextCacheIntermediates": False,
                                                    "kCIContextPriorityRequestLow": True
                                                }
                                                old_context = ci_context
                                                ci_context = CIContext.contextWithOptions_(minimal_options)
                                                del old_context
                                                old_context = None
                                            for _ in range(3):
                                                gc.collect()
                            except Exception as e:
                                logging.error(f"Failed to recreate CIContext: {e}")
                        else:
                            logging.error("Failed to recreate CIContext")
                except Exception as e:
                    logging.error(f"Failed to clear Core Image cache: {e}\n{traceback.format_exc()}")

            gc.collect()

            if 'frame' in locals():
                del frame
            if 'processed_frame' in locals():
                del processed_frame
            if 'rgb_frame' in locals():
                del rgb_frame

            # Enhanced cleanup for fractal processing
            for fractal_var in ['temp', 'new_frame', 'old_frame', 'old_temp']:
                if fractal_var in locals():
                    try:
                        del locals()[fractal_var]
                    except Exception as e:
                        if debug_mode:
                            logging.debug(f"Error cleaning up {fractal_var}: {e}")

            # Perform more aggressive cleanup for fractal depth mode
            if mode == "fractal_depth" and frame_counter % 30 == 0:  # Every 30 frames (about once per second)
                # Clean up unused prev_frames to prevent memory buildup
                if len(prev_frames) > fractal_depth:
                    # Set all excess frames to None and truncate the list
                    for i in range(fractal_depth, len(prev_frames)):
                        prev_frames[i] = None
                    prev_frames[fractal_depth:] = []
                    gc.collect()

            frame_counter += 1

            memory_usage = process.memory_info().rss / 1024 / 1024
            if memory_usage > 2000:
                logging.debug(f"High memory usage detected: {memory_usage:.2f} MB")

            if current_time - last_status_time > 5.0:
                if frames_since_last_log > 0:
                    avg_process_time = total_process_time / frames_since_last_log * 1000
                    summary_lines = [
                        f"===== PERFORMANCE SUMMARY =====",
                    ]
                    
                    if mode == "grid":
                        summary_lines.append(f"Configuration: Grid {grid_size}x{grid_size}, Depth {depth}, Mode: {mode.upper()}")
                    elif mode == "fractal":
                        position_desc = {1: "top-left", 2: "center", 3: "top-right"}
                        if fractal_source_position == 2 and fractal_grid_size == 2:
                            summary_lines.append(f"Configuration: Fractal {fractal_grid_size}x{fractal_grid_size}, Source: top-right (special 2x2 case), Mode: {mode.upper()}")
                        else:
                            summary_lines.append(f"Configuration: Fractal {fractal_grid_size}x{fractal_grid_size}, Source: {position_desc[fractal_source_position]}, Mode: {mode.upper()}")
                    elif mode == "fractal_depth":
                        summary_lines.append(f"Configuration: Fractal Depth 2x2, Depth {fractal_depth}, Source: top-right, Mode: {mode.upper()}")
                        depth_info = get_fractal_depth_breakdown(fractal_depth)
                        for info in depth_info:
                            summary_lines.append(f"  {info}")
                    
                    summary_lines.extend([
                        f"Frames: Captured {frame_count}, Displayed {displayed_count}, Dropped {dropped_count}",
                        f"Processing: Min {min_process_time * 1000:.1f}ms, Max {max_process_time * 1000:.1f}ms, Avg {avg_process_time:.1f}ms",
                        f"Success Rate: {successful_grids_since_last_log}/{frames_since_last_log} frames processed successfully",
                        f"FPS: {int(clock.get_fps())}"
                    ])
                    
                    hw_line = "Hardware Acceleration: "
                    if hardware_acceleration_available:
                        hw_line += "Enabled"
                        if force_hardware_acceleration:
                            hw_line += " (Forced)"
                    else:
                        hw_line += "Disabled"
                        if allow_software_fallback:
                            hw_line += " (Software Fallback)"
                    summary_lines.append(hw_line)
                    if downsample_times and downsample_sizes:
                        avg_downsample_time = sum(downsample_times) / len(downsample_times) * 1000
                        last_original, last_small = downsample_sizes[-1]
                        reduction_ratio = last_original / max(1, last_small)
                        summary_lines.append(f"Downsampling: {avg_downsample_time:.1f}ms, Reduction Ratio: {reduction_ratio:.1f}x")
                    
                    if is_apple_silicon and hardware_acceleration_available:
                        summary_lines.append(f"Memory Management: {cleanup_stats['total_cleanups']} cleanups ({cleanup_stats['extra_cleanups']} emergency), interval: {cleanup_stats['current_interval']:.1f}s")
                        
                    try:
                        memory_usage = process.memory_info().rss / 1024 / 1024
                        summary_lines.append(f"Current Process Memory Usage (Hardware-level): {memory_usage:.2f} MB")
                    except Exception:
                        pass
                    
                    if not debug_mode and depth > 0 and mode == "grid":
                        summary_lines.append("\nGrid Layer Breakdown:")
                        layer_info, total_screens, formatted_total = get_grid_layer_breakdown(grid_size, depth)
                        for layer in layer_info:
                            summary_lines.append(f"  {layer}")
                        summary_lines.append(f"  Total screens = {formatted_total}")
                    
                    if enable_memory_tracing:
                        snapshot = tracemalloc.take_snapshot()
                        top_stats = snapshot.statistics('lineno')
                        summary_lines.append("\nTop 5 Memory Allocations (Python-level):")
                        for stat in top_stats[:5]:
                            summary_lines.append(f"  {stat}")
                    
                    for line in summary_lines:
                        logging.info(line)
                frames_since_last_log = 0
                successful_grids_since_last_log = 0
                min_process_time = float('inf')
                max_process_time = 0
                total_process_time = 0
                last_status_time = current_time
            
            clock.tick(30)

            # Add frame rate control for the main loop
            frame_processing_time = time.time() - frame_start_time
            target_frame_time = 1.0 / 30  # Target 30 FPS
            
            if frame_processing_time < target_frame_time:
                # Only sleep if we're ahead of schedule
                time.sleep(target_frame_time - frame_processing_time)

        # Stop the frame reader thread
        logging.info("Shutting down frame reader thread...")
        stop_frame_reader_thread()
            
        cap.release()
        pygame.quit()
        logging.info("Shutdown complete")
    except Exception as e:
        logging.error(f"Main function crashed: {e}\n{traceback.format_exc()}")
    finally:
        try:
            if is_apple_silicon and hardware_acceleration_available:
                stop_memory_cleanup_thread()
                
            if ci_context is not None:
                try:
                    ci_context.clearCaches()
                except Exception as e:
                    logging.error(f"Error clearing CI context cache during shutdown: {e}")
                    
            if cap is not None:
                try:
                    cap.release()
                    logging.debug("Released video capture resources")
                except Exception as e:
                    logging.error(f"Error releasing video capture: {e}")
            
            for frame_var in ['frame', 'processed_frame']:
                if frame_var in locals() and locals()[frame_var] is not None:
                    try:
                        del locals()[frame_var]
                    except Exception as e:
                        logging.error(f"Error cleaning up {frame_var}: {e}")
            
            for surface_var in ['frame_surface', 'screen']:
                if surface_var in locals() and locals()[surface_var] is not None:
                    try:
                        locals()[surface_var] = None
                    except Exception as e:
                        logging.error(f"Error cleaning up {surface_var}: {e}")
            
            for font_var in ['font']:
                if font_var in locals() and locals()[font_var] is not None:
                    try:
                        locals()[font_var] = None
                    except Exception as e:
                        logging.error(f"Error cleaning up {font_var}: {e}")
            
            try:
                pygame.quit()
                logging.debug("Pygame resources released")
            except Exception as e:
                logging.error(f"Error quitting Pygame: {e}")
            
            gc.collect()
            
            logging.info("Shutdown complete")
        except Exception as e:
            logging.error(f"Error during cleanup: {e}\n{traceback.format_exc()}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Program crashed: {e}\n{traceback.format_exc()}")