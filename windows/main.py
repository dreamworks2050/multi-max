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
import tracemalloc
import psutil
import queue

# Windows-specific version marker - DO NOT REMOVE - used by installer to verify correct version
__windows_specific_version__ = True

# Verify running on Windows
if platform.system() != 'Windows':
    logging.warning("This is the Windows-specific version of Multi-Max running on a non-Windows platform.")
    logging.warning("Some features may not work correctly. Consider using the appropriate version for your platform.")

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
                message = record.getMessage().lower()
                if record.levelno >= logging.WARNING:
                    return True
                if any(term in message for term in [
                    '[opencv', '[ffmpeg', 'opening', 'skip', 
                    'hls request', 'videoplayback', 'manifest.googlevideo',
                    'cannot reuse http connection'
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

# Load hardware acceleration settings (not used on Windows but kept for compatibility)
force_hardware_acceleration = os.getenv('FORCE_HARDWARE_ACCELERATION', 'false').lower() == 'true'
allow_software_fallback = os.getenv('ALLOW_SOFTWARE_FALLBACK', 'true').lower() == 'true'

# Load memory tracing settings
enable_memory_tracing = os.getenv('ENABLE_MEMORY_TRACING', 'false').lower() == 'true'

# Global variables for hardware acceleration (disabled on Windows)
hardware_acceleration_available = False

# Global variables for memory management
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

# Setup for FFmpeg error detection
ffmpeg_http_errors = []
youtube_connection_status = {
    'last_success': 0,
    'last_error': 0,
    'error_count': 0,
    'retry_count': 0,
    'current_host': ''
}

def log_youtube_connection_status(status, host='', error=''):
    """Log YouTube connection status with reduced verbosity."""
    global youtube_connection_status
    current_time = time.time()
    
    if status == 'connected':
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
        if youtube_connection_status['error_count'] <= 3 and error:
            logging.warning(f"YouTube connection error: {error}")
        elif youtube_connection_status['error_count'] == 4:
            logging.warning("Multiple YouTube connection errors - suppressing further error messages")
    elif status == 'retry':
        youtube_connection_status['retry_count'] += 1
        if youtube_connection_status['retry_count'] % 5 == 1:
            logging.info(f"Retrying YouTube connection (attempt {youtube_connection_status['retry_count']})")

# Frame buffer for continuous streaming
frame_buffer = None
frame_buffer_lock = None
frame_reader_thread = None
frame_buffer_size = 60
should_stop_frame_reader = False
current_stream_url = None
last_buffer_warning_time = 0
frame_drop_threshold = 0.8
stream_url_refresh_interval = 5 * 60
last_url_refresh_time = 0

def refresh_youtube_url(original_url):
    """Refresh the YouTube streaming URL."""
    try:
        logging.info(f"Refreshing YouTube stream URL for: {original_url}")
        ts = int(time.time())
        query_url = f"{original_url}{'&' if '?' in original_url else '?'}_ts={ts}"
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
    """Start a background thread to read frames from the video source."""
    global frame_buffer, frame_buffer_lock, frame_reader_thread, should_stop_frame_reader
    global current_stream_url, frame_drop_threshold, last_url_refresh_time
    
    current_stream_url = stream_url
    last_url_refresh_time = time.time()
    
    frame_buffer = queue.Queue(maxsize=buffer_size)
    frame_buffer_lock = threading.Lock()
    should_stop_frame_reader = False
    
    def frame_reader_worker():
        logging.info("Frame reader thread started")
        global current_stream_url, last_url_refresh_time, stream_url_refresh_interval
        frames_read = 0
        frames_dropped = 0
        consecutive_failures = 0
        max_consecutive_failures = 5
        target_fps = 30
        frame_interval = 1.0 / target_fps
        last_frame_time = time.time()
        reconnection_backoff = 1.0
        max_reconnection_backoff = 15.0
        
        width = int(video_source.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        thread_cap = None
        last_reconnect_attempt = 0
        reconnect_interval = 3
        
        http_errors_count = 0
        http_error_threshold = 3
        last_error_time = 0
        error_reset_interval = 60
        
        def open_capture(url=None):
            nonlocal thread_cap, consecutive_failures, reconnection_backoff, last_reconnect_attempt
            # Use current_stream_url which is a global variable set in start_frame_reader_thread
            global current_stream_url
            capture_url = url if url is not None else current_stream_url
            
            if thread_cap is not None:
                thread_cap.release()
                time.sleep(0.5)
                thread_cap = None
                
            if isinstance(capture_url, str) and capture_url:
                logging.info(f"Frame reader opening stream (backoff: {reconnection_backoff:.1f}s)")
                ffmpeg_options = {
                    "rtsp_transport": "tcp",
                    "fflags": "nobuffer",
                    "flags": "low_delay",
                    "stimeout": "5000000",
                    "reconnect": "1",
                    "reconnect_streamed": "1",
                    "reconnect_delay_max": "5",
                    "multiple_requests": "0",
                    "reuse_socket": "0",
                    "http_persistent": "0"
                }
                ffmpeg_opt_str = ' '.join([f"-{k} {v}" for k, v in ffmpeg_options.items()])
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = ffmpeg_opt_str
                
                thread_cap = cv2.VideoCapture(capture_url, cv2.CAP_FFMPEG)
                
                if thread_cap.isOpened():
                    thread_cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    thread_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    thread_cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                    thread_cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
                    thread_cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                    try:
                        thread_cap.set(cv2.CAP_PROP_FPS, target_fps)
                    except:
                        pass
                    consecutive_failures = 0
                    reconnection_backoff = 1.0
                    last_reconnect_attempt = time.time()
                    
                    from urllib.parse import urlparse
                    host = urlparse(capture_url).netloc if capture_url else "unknown"
                    log_youtube_connection_status('connected', host=host)
                    return True
                else:
                    log_youtube_connection_status('error', error="Failed to open video capture")
                    logging.error("Failed to open video capture in frame reader thread")
            
            reconnection_backoff = min(reconnection_backoff * 1.5, max_reconnection_backoff)
            return False
            
        # Initialize capture after all variables are defined
        if not open_capture():
            logging.error("Frame reader could not open video source")
        
        buffer_filled = False
        
        while not should_stop_frame_reader:
            current_time = time.time()
            
            if http_errors_count > 0 and current_time - last_error_time > error_reset_interval:
                http_errors_count = 0
                logging.debug("HTTP error count reset after error-free period")
            
            need_refresh = False
            if original_youtube_url:
                if current_time - last_url_refresh_time > stream_url_refresh_interval:
                    need_refresh = True
                    logging.debug(f"Scheduled URL refresh after {stream_url_refresh_interval/60:.1f} minutes")
                elif http_errors_count >= http_error_threshold:
                    need_refresh = True
                    http_errors_count = 0
                    logging.debug(f"Forcing URL refresh after {http_error_threshold} connection errors")
                
                if need_refresh:
                    new_url = refresh_youtube_url(original_youtube_url)
                    if new_url and new_url != current_stream_url:
                        logging.debug("Stream URL refreshed successfully")
                        current_stream_url = new_url
                        if open_capture(new_url):
                            logging.debug("Reconnected with fresh YouTube URL")
                    last_url_refresh_time = current_time
            
            if thread_cap is None or not thread_cap.isOpened():
                if current_time - last_reconnect_attempt > reconnection_backoff:
                    log_youtube_connection_status('retry')
                    logging.debug(f"Video source disconnected, reconnecting (backoff: {reconnection_backoff:.1f}s)...")
                    open_capture()
                    last_reconnect_attempt = current_time
                time.sleep(0.5)
                continue
            
            elapsed = current_time - last_frame_time
            buffer_fullness = frame_buffer.qsize() / buffer_size
            
            if buffer_fullness > frame_drop_threshold and elapsed < frame_interval:
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            try:
                ret, frame = thread_cap.read()
                last_frame_time = time.time()
            except Exception as e:
                error_str = str(e).lower()
                if "http" in error_str or "connect" in error_str or "host" in error_str or "network" in error_str:
                    log_youtube_connection_status('error', error="HTTP connection error")
                    http_errors_count += 1
                    last_error_time = current_time
                else:
                    log_youtube_connection_status('error', error=f"Read error: {type(e).__name__}")
                ret, frame = False, None
                consecutive_failures = max_consecutive_failures
            
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= 2:
                    current_time = time.time()
                    if current_time - last_error_time > 5:
                        http_errors_count += 1
                        last_error_time = current_time
                        log_youtube_connection_status('error', error="Connection interrupted")
                
                if consecutive_failures >= max_consecutive_failures:
                    log_youtube_connection_status('retry')
                    logging.warning(f"Failed to read frames from video source ({consecutive_failures} consecutive failures)")
                    consecutive_failures = 0
                    if current_time - last_reconnect_attempt > reconnect_interval:
                        logging.warning("Too many consecutive failures, attempting to reconnect...")
                        if original_youtube_url and current_time - last_url_refresh_time > 60:
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
                time.sleep(min(0.1, reconnection_backoff / 2))
                continue
            
            consecutive_failures = 0
            frames_read += 1
            
            try:
                buffer_fullness = frame_buffer.qsize() / buffer_size
                if buffer_fullness > 0.95:
                    frames_dropped += 1
                    continue
                
                if frame_buffer.full():
                    with frame_buffer_lock:
                        try:
                            frame_buffer.get_nowait()
                            frames_dropped += 1
                        except queue.Empty:
                            pass
                
                with frame_buffer_lock:
                    frame_buffer.put(frame.copy(), block=False)
                    current_size = frame_buffer.qsize()
                    if not buffer_filled and current_size >= buffer_size * 0.5:
                        buffer_filled = True
                        logging.info(f"Frame buffer filled to {current_size}/{buffer_size} frames")
                
                if frames_read % 300 == 0:
                    logging.info(f"Frame reader stats: Read {frames_read}, Dropped {frames_dropped}, Buffer size {frame_buffer.qsize()}/{buffer_size}")
                    
            except queue.Full:
                frames_dropped += 1
                continue
            except Exception as e:
                logging.error(f"Error in frame reader thread: {e}")
                continue
                
            if buffer_fullness < 0.2:
                pass
            elif buffer_fullness < 0.5:
                time.sleep(0.001)
            else:
                time.sleep(0.005)
        
        if thread_cap is not None:
            thread_cap.release()
            
        logging.info(f"Frame reader thread stopped. Total frames read: {frames_read}, dropped: {frames_dropped}")
    
    frame_reader_thread = threading.Thread(target=frame_reader_worker, name="FrameReader", daemon=True)
    frame_reader_thread.start()
    return frame_reader_thread

def get_latest_frame():
    """Get the latest frame from the buffer."""
    global frame_buffer, frame_buffer_lock, last_buffer_warning_time
    
    if frame_buffer is None or frame_buffer.empty():
        current_time = time.time()
        if current_time - last_buffer_warning_time > 1.0:
            logging.warning("No frames available from buffer")
            last_buffer_warning_time = current_time
        return None
        
    with frame_buffer_lock:
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
            
    if frame_buffer is not None:
        with frame_buffer_lock:
            while not frame_buffer.empty():
                try:
                    frame_buffer.get_nowait()
                except queue.Empty:
                    break

# Global variables
grid_size = 2
depth = 1
running = True
debug_mode = False
show_info = True
info_hidden_time = 0
mode = "fractal_depth"
fractal_grid_size = 3
fractal_debug = False
fractal_source_position = 2
prev_output_frame = None
fractal_depth = 1

# Stats
frame_count = 0
processed_count = 0
displayed_count = 0
dropped_count = 0

# Performance tracking
downsample_times = []
downsample_sizes = []
max_tracked_samples = 50

def create_fractal_grid(live_frame, prev_output, grid_size, source_position=1):
    """
    Create an NxN grid for the infinite fractal effect.
    
    Args:
        live_frame (np.ndarray): Current frame from the video stream.
        prev_output (np.ndarray): Previous output frame for recursion.
        grid_size (int): Size of the grid (e.g., 2 for 2x2).
        source_position (int): Position of the source frame (1=top-left, 2=center, 3=top-right).
    
    Returns:
        np.ndarray: Resulting NxN grid frame.
    """
    if prev_output is None:
        prev_output = np.zeros_like(live_frame)
    
    h, w = live_frame.shape[:2]
    cell_h = h // grid_size
    cell_w = w // grid_size
    
    grid_frame = np.zeros_like(live_frame)
    
    source_i, source_j = 0, 0
    if source_position == 2:
        if grid_size == 2:
            source_i, source_j = 0, 1
        elif grid_size % 2 == 1:
            source_i = source_j = grid_size // 2
        else:
            center = grid_size / 2 - 0.5
            source_i = source_j = int(center)
    elif source_position == 3:
        source_i, source_j = 0, grid_size - 1
    
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
                    grid_frame[y_start:y_end, x_start:x_end] = cv2.resize(live_frame, (cell_width, cell_height))
                else:
                    grid_frame[y_start:y_end, x_start:x_end] = cv2.resize(prev_output, (cell_width, cell_height))
        
        return grid_frame
    except Exception as e:
        logging.error(f"Error in create_fractal_grid: {e}")
        return live_frame.copy()

def compute_fractal_depth(live_frame, depth):
    """Compute fractal output for a given depth."""
    if depth == 0:
        return np.zeros_like(live_frame)
    else:
        prev = compute_fractal_depth(live_frame, depth - 1)
        return create_fractal_grid(live_frame, prev, 2, 3)

def handle_keyboard_event(key_name, mod=None):
    """Handle keyboard inputs for adjusting settings."""
    global grid_size, depth, debug_mode, show_info, info_hidden_time, mode, fractal_grid_size, fractal_debug
    global fractal_source_position, fractal_depth, prev_frames

    old_grid_size = grid_size
    old_depth = depth
    old_fractal_grid_size = fractal_grid_size
    old_fractal_source_position = fractal_source_position
    
    try:
        is_repeat = key_name in ['up', 'down'] and key_pressed.get(key_name, False) and time.time() - key_press_start.get(key_name, 0) > key_repeat_delay
        
        if key_name == '4' and mode == "fractal" and (not mod or not (mod & pygame.KMOD_SHIFT)):
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
                if old_fractal_depth > fractal_depth:
                    for i in range(fractal_depth, len(prev_frames)):
                        prev_frames[i] = None
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
                resolution_text = "" if not is_resolution_limited else f" (at {frame_width}x{frame_height} resolution, limited by {min(frame_width, frame_height) / (fractal_grid_size ** max_level):.2f} pixels per cell)"
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
                        fractal_grid_size -= 1 if fractal_grid_size % 2 == 0 else 2
                else:
                    fractal_grid_size = max(1, fractal_grid_size - 1)
                frame_width, frame_height = 1280, 720
                visible_screens, is_resolution_limited, max_level, screens_by_category = calculate_visible_screens(fractal_grid_size, frame_width, frame_height)
                visible_recursive_cells = calculate_visible_recursive_cells(fractal_grid_size, frame_width, frame_height)
                resolution_text = "" if not is_resolution_limited else f" (at {frame_width}x{frame_height} resolution, limited by {min(frame_width, frame_height) / (fractal_grid_size ** max_level):.2f} pixels per cell)"
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
            if mode in ["fractal", "fractal_depth"]:
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
    """Handle keyboard interrupts and signals."""
    global running
    try:
        signal_name = signal.Signals(sig).name
        logging.info(f"Received signal {signal_name} ({sig}), shutting down gracefully...")
    except (ValueError, AttributeError):
        # Windows may not recognize all signal names
        logging.info(f"Received signal {sig}, shutting down gracefully...")
    running = False

def generate_grid_frame(previous_frame, grid_size, current_depth):
    """Generate a grid frame using OpenCV."""
    global mode
    
    try:
        if previous_frame is None or previous_frame.size == 0:
            logging.error(f"Depth {current_depth}: Previous frame is None or empty" if mode == "grid" else "Previous frame is None or empty")
            return None
            
        h, w = previous_frame.shape[:2]
        if h <= 0 or w <= 0:
            logging.error(f"Depth {current_depth}: Invalid frame dimensions: {w}x{h}" if mode == "grid" else f"Invalid frame dimensions: {w}x{h}")
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
        interpolation = cv2.INTER_AREA if (small_h < h or small_w < w) else cv2.INTER_LINEAR
        small_frame = cv2.resize(previous_frame, (small_w, small_h), interpolation=interpolation)
        
        downsample_time = time.time() - downsample_start_time
        downsample_times.append(downsample_time)
        downsample_sizes.append((h * w, small_h * small_w))
        if len(downsample_times) > max_tracked_samples:
            downsample_times.pop(0)
            downsample_sizes.pop(0)
        
        for i in range(grid_size):
            for j in range(grid_size):
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
                    logging.error(f"Depth {current_depth}: Error processing cell {i}x{j}: {e}" if mode == "grid" else f"Error processing cell {i}x{j}: {e}")
                    result[y_start:y_end, x_start:x_end] = 0
            
        return result
    except Exception as e:
        logging.error(f"Depth {current_depth}: Error in generate_grid_frame: {e}\n{traceback.format_exc()}" if mode == "grid" else f"Error in generate_grid_frame: {e}\n{traceback.format_exc()}")
        return None
    finally:
        gc.collect()

def apply_grid_effect(frame, grid_size, depth):
    """Apply the grid effect using OpenCV."""
    global mode
    
    try:
        if frame is None or frame.size == 0:
            logging.error("Input frame is None or empty")
            return None
        
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            logging.error(f"Invalid frame dimensions: {w}x{h}")
            return None
        
        previous_frame = frame.copy()
        result = None
        max_failed_depths = 3
        failed_depths = 0
        
        for d in range(1, depth + 1):
            new_frame = generate_grid_frame(previous_frame, grid_size, d)
            
            if new_frame is None:
                logging.error(f"Depth {d}: Failed to generate grid frame (returned None)" if mode == "grid" else "Failed to generate grid frame (returned None)")
                failed_depths += 1
                if failed_depths > max_failed_depths:
                    logging.error(f"Too many failures ({failed_depths}), aborting grid effect" if mode == "grid" else f"Too many failures ({failed_depths}), aborting effect")
                    result = previous_frame.copy()
                    del previous_frame
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
        gc.collect()
        return result
    except Exception as e:
        logging.error(f"Error in apply_grid_effect: {e}\n{traceback.format_exc()}")
        if 'previous_frame' in locals() and id(previous_frame) != id(frame):
            del previous_frame
        return frame.copy() if frame is not None else None
    finally:
        if 'previous_frame' in locals() and id(previous_frame) != id(result) and id(previous_frame) != id(frame):
            del previous_frame
        gc.collect()

def get_stream_url(url):
    """Retrieve streaming URL using yt-dlp."""
    try:
        if not url or url.strip() == "":
            logging.error("Empty URL provided to get_stream_url function")
            return None
            
        logging.info(f"Attempting to get stream URL for: {url}")
        
        yt_dlp_args = [
            "yt-dlp",
            "-f", "best",
            "--get-url",
            "--no-check-certificate",
            "--no-cache-dir",
            "--force-ipv4",
            "--no-playlist",
            "--extractor-retries", "3",
            "--referer", "https://www.youtube.com/",
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
            "--socket-timeout", "5",
            url
        ]
        
        result = subprocess.run(yt_dlp_args, capture_output=True, text=True, check=True)
        
        stream_url = result.stdout.strip()
        if not stream_url:
            logging.error("yt-dlp returned an empty stream URL")
            return None
            
        if not stream_url.startswith(('http://', 'https://')):
            logging.error(f"yt-dlp returned an invalid URL format: {stream_url[:30]}...")
            return None
            
        logging.info("Successfully retrieved stream URL")
        return stream_url
    except subprocess.CalledProcessError as e:
        logging.error(f"yt-dlp failed with return code {e.returncode}: {e.stderr}")
        try:
            logging.info("Retrying with simplified options...")
            result = subprocess.run(["yt-dlp", "-f", "best", "--get-url", url], capture_output=True, text=True, check=True)
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
    """Generate breakdown of screens at each depth level."""
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
        (1e21, "sextillion"), (1e24, "septillion"), (1e27, "octillion"), (1e30, "nonillion"), (1e33, "decillion")
    ]
    formatted_total = f"{total_screens:,}"
    for threshold, unit in reversed(number_units):
        if total_screens >= threshold:
            value = total_screens / threshold
            formatted_total = f"{total_screens:,} ({value:.2f} {unit})"
            break
    return layer_info, total_screens, formatted_total

def get_fractal_depth_breakdown(max_depth):
    """Generate breakdown of fractal depth levels."""
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

def calculate_visible_screens(fractal_grid_size, frame_width, frame_height):
    """Calculate visible screens in fractal mode."""
    if fractal_grid_size < 2:
        return 1, False, 0, {"larger_than_1px": 1, "exactly_1px": 0, "smaller_than_1px": False}
    
    min_dim = min(frame_width, frame_height)
    max_level = int(np.floor(np.log(min_dim) / np.log(fractal_grid_size)))
    r = fractal_grid_size ** 2 - 1
    
    screens_by_category = {"larger_than_1px": 0, "exactly_1px": 0, "smaller_than_1px": False}
    visible_screens = 0
    
    one_pixel_level = np.log(min_dim) / np.log(fractal_grid_size)
    exact_pixel_level = int(np.floor(one_pixel_level)) if abs(min_dim / (fractal_grid_size ** int(np.floor(one_pixel_level))) - 1.0) < abs(min_dim / (fractal_grid_size ** int(np.ceil(one_pixel_level))) - 1.0) else int(np.ceil(one_pixel_level))
    
    for k in range(max_level + 1):
        screens_at_level = r ** k
        cell_size = min_dim / (fractal_grid_size ** k)
        
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
    
    return visible_screens, is_resolution_limited, max_level, screens_by_category

def calculate_visible_recursive_cells(fractal_grid_size, frame_width, frame_height):
    """Calculate visible recursive cells in fractal mode."""
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
    """Main function for Windows-compatible video processing."""
    global running, grid_size, depth, debug_mode, show_info, frame_count, processed_count, displayed_count, dropped_count
    global hardware_acceleration_available, cap, mode, fractal_grid_size, fractal_debug, fractal_source_position, prev_output_frame
    global enable_memory_tracing, fractal_depth, current_stream_url, last_buffer_warning_time, frame_drop_threshold
    global key_pressed, key_press_start, key_last_repeat, last_url_refresh_time, stream_url_refresh_interval, prev_frames
    
    key_pressed = {}
    key_press_start = {}
    key_last_repeat = {}
    
    screen = None
    cap = None
    frame = None
    processed_frame = None
    frame_surface = None
    font = None
    
    prev_frames = [None] * fractal_depth
    last_fractal_cleanup_time = time.time()
    fractal_cleanup_interval = 10.0
    
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
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        load_dotenv()
        youtube_url = os.getenv('YOUTUBE_URL', "https://www.youtube.com/watch?v=ZzWBpGwKoaI")
        if not os.getenv('YOUTUBE_URL'):
            logging.warning("No YOUTUBE_URL found in .env file. Using default demo URL.")
        
        parser = argparse.ArgumentParser(description='Recursive Video Grid')
        parser.add_argument('--grid-size', type=int, default=3, help='Grid size')
        parser.add_argument('--depth', type=int, default=1)
        parser.add_argument('--youtube-url', type=str, default=youtube_url)
        parser.add_argument('--log-level', type=str, default='INFO')
        parser.add_argument('--debug', action='store_true')
        parser.add_argument('--mode', type=str, choices=['grid', 'fractal', 'fractal_depth'], default='fractal_depth')
        parser.add_argument('--fractal-source', type=int, choices=[1, 2, 3], default=2)
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
        
        logging.info(f"Starting in mode: {mode.upper()}")
        if mode == "fractal_depth":
            depth_info = get_fractal_depth_breakdown(fractal_depth)
            for info in depth_info:
                logging.info(f"  {info}")
            logging.info("Use UP/DOWN arrow keys to increase/decrease depth level (1-100)")
        elif mode == "fractal":
            logging.info("Press 1-3 to change source position, 4 to switch to fractal depth mode")
        
        logging.info(f"Starting with debug={debug_mode} (toggle with 'd')")
        
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
        screen.blit(loading_text, (screen.get_width() // 2 - loading_text.get_width() // 2, screen.get_height() // 2 - loading_text.get_height() // 2))
        pygame.display.flip()

        stream_url = get_stream_url(args.youtube_url)
        if not stream_url:
            logging.error(f"Failed to get stream URL for {args.youtube_url}")
            logging.error("Please check that your YOUTUBE_URL in .env is valid and accessible")
            return

        screen.fill((255, 255, 255))
        connecting_text = font.render("Connecting to Video Stream...", True, (0, 0, 0))
        please_wait_text = font.render("Please wait...", True, (0, 0, 0))
        screen.blit(connecting_text, (screen.get_width() // 2 - connecting_text.get_width() // 2, screen.get_height() // 2 - connecting_text.get_height() // 2))
        screen.blit(please_wait_text, (screen.get_width() // 2 - please_wait_text.get_width() // 2, screen.get_height() // 2 + 30))
        pygame.display.flip()

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
        
        start_frame_reader_thread(cap, stream_url, args.youtube_url, buffer_size=frame_buffer_size)
        logging.info("Started background frame reader thread")

        last_status_time = time.time()
        last_process_time = 0
        frames_since_last_log = 0
        successful_grids_since_last_log = 0
        min_process_time = float('inf')
        max_process_time = 0
        total_process_time = 0
        frame_counter = 0

        buffer_wait_start = time.time()
        buffer_wait_timeout = 5.0
        while time.time() - buffer_wait_start < buffer_wait_timeout:
            if frame_buffer is not None and not frame_buffer.empty():
                logging.info(f"Buffer started filling after {time.time() - buffer_wait_start:.1f} seconds")
                break
            time.sleep(0.1)
            screen.fill((0, 0, 0))
            if font is None:
                font = pygame.font.SysFont('Arial', 24)
            loading_text = font.render(f"Loading stream... ({int(time.time() - buffer_wait_start)}s)", True, (255, 255, 255))
            screen.blit(loading_text, (screen.get_width() // 2 - loading_text.get_width() // 2, screen.get_height() // 2 - loading_text.get_height() // 2))
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
                    if key in ['up', 'down', 'd', 's', 'f'] or (key in '0123456789' and mode != "fractal_depth"):
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

            frame = get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            frame_count += 1

            process_start_time = time.time()
            if mode == "grid":
                processed_frame = frame.copy() if debug_mode else apply_grid_effect(frame, grid_size, depth)
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
                    if current_time - last_fractal_cleanup_time > fractal_cleanup_interval:
                        logging.debug("Performing periodic cleanup of fractal depth memory")
                        for i in range(len(prev_frames)):
                            if prev_frames[i] is not None and i >= fractal_depth:
                                prev_frames[i] = None
                        gc.collect()
                        last_fractal_cleanup_time = current_time
                        
                    temp = frame.copy()
                    if len(prev_frames) < fractal_depth:
                        prev_frames.extend([None] * (fractal_depth - len(prev_frames)))
                    
                    for d in range(fractal_depth):
                        if prev_frames[d] is None:
                            prev_frames[d] = np.zeros_like(frame)
                        new_frame = create_fractal_grid(temp, prev_frames[d], 2, 3)
                        old_frame = prev_frames[d]
                        prev_frames[d] = new_frame.copy()
                        old_temp = temp
                        temp = new_frame
                        del old_frame
                        if d > 0:
                            del old_temp
                        del new_frame
                        if d > 50 and d % 10 == 0:
                            gc.collect()
                    processed_frame = temp
            
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
                
                if frame_count % 10 == 0:
                    gc.collect()
                
                if show_info:
                    hw_status = "Disabled"
                    texts = [
                        f"Mode: {mode.upper()}",
                        f"Grid: {grid_size if mode == 'grid' else fractal_grid_size if mode == 'fractal' else 2}x{grid_size if mode == 'grid' else fractal_grid_size if mode == 'fractal' else 2}",
                        f"Depth: {depth if mode == 'grid' else fractal_depth if mode == 'fractal_depth' else 'N/A'}"
                    ]
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
                            alpha = int(128 * (1.0 - (time_since_hidden - 4.0)))
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

            gc.collect()
            for var in ['frame', 'processed_frame', 'rgb_frame', 'temp', 'new_frame', 'old_frame', 'old_temp']:
                if var in locals():
                    del locals()[var]

            if mode == "fractal_depth" and frame_counter % 30 == 0:
                if len(prev_frames) > fractal_depth:
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
                        f"Configuration: {'Grid' if mode == 'grid' else 'Fractal' if mode == 'fractal' else 'Fractal Depth'} {grid_size if mode == 'grid' else fractal_grid_size if mode == 'fractal' else 2}x{grid_size if mode == 'grid' else fractal_grid_size if mode == 'fractal' else 2}, Depth {depth if mode == 'grid' else fractal_depth if mode == 'fractal_depth' else 'N/A'}, Mode: {mode.upper()}",
                        f"Frames: Captured {frame_count}, Displayed {displayed_count}, Dropped {dropped_count}",
                        f"Processing: Min {min_process_time * 1000:.1f}ms, Max {max_process_time * 1000:.1f}ms, Avg {avg_process_time:.1f}ms",
                        f"Success Rate: {successful_grids_since_last_log}/{frames_since_last_log} frames processed successfully",
                        f"FPS: {int(clock.get_fps())}",
                        "Hardware Acceleration: Disabled"
                    ]
                    if downsample_times and downsample_sizes:
                        avg_downsample_time = sum(downsample_times) / len(downsample_times) * 1000
                        last_original, last_small = downsample_sizes[-1]
                        reduction_ratio = last_original / max(1, last_small)
                        summary_lines.append(f"Downsampling: {avg_downsample_time:.1f}ms, Reduction Ratio: {reduction_ratio:.1f}x")
                    memory_usage = process.memory_info().rss / 1024 / 1024
                    summary_lines.append(f"Current Process Memory Usage: {memory_usage:.2f} MB")
                    if not debug_mode and depth > 0 and mode == "grid":
                        summary_lines.append("\nGrid Layer Breakdown:")
                        layer_info, total_screens, formatted_total = get_grid_layer_breakdown(grid_size, depth)
                        for layer in layer_info:
                            summary_lines.append(f"  {layer}")
                        summary_lines.append(f"  Total screens = {formatted_total}")
                    if enable_memory_tracing:
                        snapshot = tracemalloc.take_snapshot()
                        top_stats = snapshot.statistics('lineno')
                        summary_lines.append("\nTop 5 Memory Allocations:")
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
            frame_processing_time = time.time() - frame_start_time
            target_frame_time = 1.0 / 30
            if frame_processing_time < target_frame_time:
                time.sleep(target_frame_time - frame_processing_time)

        logging.info("Shutting down frame reader thread...")
        stop_frame_reader_thread()
        cap.release()
        pygame.quit()
        logging.info("Shutdown complete")
    except Exception as e:
        logging.error(f"Main function crashed: {e}\n{traceback.format_exc()}")
    finally:
        if cap is not None:
            cap.release()
        for var in ['frame', 'processed_frame', 'frame_surface', 'screen', 'font']:
            if var in locals() and locals()[var] is not None:
                locals()[var] = None
        pygame.quit()
        gc.collect()
        logging.info("Shutdown complete")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Program crashed: {e}\n{traceback.format_exc()}")