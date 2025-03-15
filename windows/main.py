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
import atexit
from dotenv import load_dotenv
import traceback
import gc
import pygame
import tracemalloc
import psutil
import queue
import requests
import sys
from pathlib import Path
import re

# Windows-specific version marker - DO NOT REMOVE - used by installer to verify correct version
__windows_specific_version__ = True

# Setup version checking mechanism
CURRENT_VERSION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "version.txt")
REMOTE_VERSION_URL = "https://raw.githubusercontent.com/dreamworks2050/multi-max/main/windows/version.txt"
VERSION_CHECK_TIMEOUT = 5  # seconds

def get_local_version():
    """Read the local version number from version.txt file."""
    try:
        if os.path.exists(CURRENT_VERSION_FILE):
            with open(CURRENT_VERSION_FILE, 'r') as f:
                version = f.read().strip()
                return version
        else:
            logging.warning(f"Local version file not found at {CURRENT_VERSION_FILE}")
            return "0.0.0"  # Default version if file doesn't exist
    except Exception as e:
        logging.error(f"Error reading local version: {e}")
        return "0.0.0"
        
def get_remote_version():
    """Fetch the latest version number from the repository."""
    try:
        response = requests.get(REMOTE_VERSION_URL, timeout=VERSION_CHECK_TIMEOUT)
        if response.status_code == 200:
            return response.text.strip()
        else:
            logging.warning(f"Failed to fetch remote version. Status code: {response.status_code}")
            return None
    except requests.RequestException as e:
        logging.warning(f"Network error while checking for updates: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error checking for updates: {e}")
        return None

def versions_need_update(local_version, remote_version):
    """Compare version numbers to determine if an update is needed."""
    if not remote_version:
        return False
        
    # Normalize versions - handle any unexpected characters or formats
    def normalize_version(version_str):
        if not version_str:
            return "0.0.0"
            
        # Remove any whitespace
        version_str = version_str.strip()
        
        # Keep only digits and dots
        version_str = re.sub(r'[^\d.]', '', version_str)
        
        # Ensure at least one dot
        if '.' not in version_str:
            version_str = version_str + '.0'
            
        # Handle consecutive dots
        while '..' in version_str:
            version_str = version_str.replace('..', '.0.')
            
        # Remove trailing dot if any
        if version_str.endswith('.'):
            version_str = version_str + '0'
            
        # Handle empty version
        if not version_str or version_str == '.':
            return "0.0.0"
            
        return version_str
    
    try:
        local_version = normalize_version(local_version)
        remote_version = normalize_version(remote_version)
        
        logging.debug(f"Comparing versions - local: {local_version}, remote: {remote_version}")
        
        # If version strings are identical, no update needed
        if local_version == remote_version:
            return False
            
        # Split versions into components and convert to integers for comparison
        local_parts = [int(p) for p in local_version.split('.')]
        remote_parts = [int(p) for p in remote_version.split('.')]
        
        # Pad shorter version with zeros
        while len(local_parts) < len(remote_parts):
            local_parts.append(0)
        while len(remote_parts) < len(local_parts):
            remote_parts.append(0)
            
        # Compare each part
        for local, remote in zip(local_parts, remote_parts):
            if remote > local:
                return True
            if local > remote:
                return False
                
        # If we get here, versions are equal
        return False
    except (ValueError, AttributeError, IndexError) as e:
        logging.error(f"Error comparing versions ({local_version} vs {remote_version}): {e}")
        return False

def perform_auto_update():
    """Trigger the automatic update process."""
    try:
        windows_dir = os.path.dirname(os.path.abspath(__file__))
        installer_path = os.path.join(windows_dir, "Install-Windows.bat")
        
        if not os.path.exists(installer_path):
            logging.error(f"Installer not found at {installer_path}")
            # Try to find the installer using an absolute path search
            alternate_paths = [
                os.path.join(os.path.dirname(windows_dir), "Install-Windows.bat"),
                os.path.join(os.path.dirname(windows_dir), "windows", "Install-Windows.bat"),
                os.path.join(os.path.dirname(os.path.dirname(windows_dir)), "Install-Windows.bat"),
                os.path.join(os.path.dirname(os.path.dirname(windows_dir)), "windows", "Install-Windows.bat")
            ]
            
            for alt_path in alternate_paths:
                if os.path.exists(alt_path):
                    logging.info(f"Found installer at alternative location: {alt_path}")
                    installer_path = alt_path
                    break
                    
            if not os.path.exists(installer_path):
                logging.error("Could not find the installer at any expected location")
                # Just return without trying to update
                return False
            
        logging.info(f"Starting automatic update process with installer: {installer_path}")
        
        # Create and display an update notification window using a simpler approach
        try:
            # Initialize only the necessary components
            pygame.init()
            
            # Use a more basic approach for font creation that's less likely to fail
            try:
                font = pygame.font.SysFont('Arial', 18)
                if not font:
                    # Sometimes SysFont fails silently
                    fonts = pygame.font.get_fonts()
                    if fonts:
                        font = pygame.font.SysFont(fonts[0], 18)
                    else:
                        # Last resort default font
                        font = pygame.font.Font(None, 18)
            except:
                # If all font attempts fail, use the default font
                font = pygame.font.Font(None, 18)
                
            # Create display with more error handling
            try:
                info_screen = pygame.display.set_mode((600, 200))
                pygame.display.set_caption("Multi-Max Update")
            except:
                # Use a smaller size if the display creation fails
                info_screen = pygame.display.set_mode((400, 150))
                pygame.display.set_caption("Update")
            
            info_screen.fill((0, 0, 0))
            
            # Render and display text with error handling
            try:
                lines = [
                    "A new version of Multi-Max is available!",
                    "The application will now update automatically.",
                    "Please wait while the installer runs..."
                ]
                
                y_position = 50
                for line in lines:
                    try:
                        text = font.render(line, True, (255, 255, 255))
                        info_screen.blit(text, (50, y_position))
                        y_position += 30
                    except:
                        # Skip any lines that fail to render
                        pass
                        
                pygame.display.flip()
                
                # Wait a moment for the user to read the message
                time.sleep(3)
            except:
                # Continue with update even if text rendering fails
                pass
                
            # Close pygame properly before launching the installer
            pygame.quit()
        except Exception as e:
            logging.error(f"Error showing update notification: {e}")
            # Continue with update even if notification fails
            
        # Launch the installer in a new process and exit this one
        try:
            logging.info(f"Launching installer from: {installer_path}")
            
            # Use a more robust Windows-specific approach
            if platform.system() == 'Windows':
                try:
                    # First try with the standard approach
                    creation_flags = 0x00000010  # CREATE_NEW_CONSOLE flag
                    subprocess.Popen(['cmd.exe', '/c', 'start', installer_path], 
                                   creationflags=creation_flags)
                except:
                    # Fall back to simpler approach if the first one fails
                    subprocess.Popen(['cmd.exe', '/c', 'start', installer_path], shell=True)
            else:
                # This should never happen on Windows version
                subprocess.Popen([installer_path], shell=True)
            
            logging.info("Update initiated. Exiting current instance.")
            time.sleep(1)  # Give the subprocess a moment to start
            
            # Exit cleanly
            try:
                pygame.quit()
            except:
                pass
                
            # Exit
            sys.exit(0)
            
        except Exception as e:
            logging.error(f"Error launching installer: {e}")
            logging.error(traceback.format_exc())
            return False
        
    except Exception as e:
        logging.error(f"Failed to perform automatic update: {e}")
        logging.error(traceback.format_exc())
        return False

def check_for_updates():
    """Check if an update is available and trigger the update process if needed."""
    logging.info("Checking for updates...")
    
    try:
        local_version = get_local_version()
        logging.info(f"Current version: {local_version}")
        
        # Ensure we have a valid local version
        if not local_version or local_version.strip() == "":
            logging.warning("Could not determine local version, will skip update check")
            return False
        
        # Add retry logic for network operations
        max_retries = 3
        retry_delay = 1.0
        remote_version = None
        
        for retry in range(max_retries):
            try:
                remote_version = get_remote_version()
                if remote_version:
                    break
                    
                logging.warning(f"Failed to get remote version (attempt {retry+1}/{max_retries})")
                if retry < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 1.5  # Exponential backoff
            except Exception as e:
                logging.error(f"Error during update check attempt {retry+1}: {e}")
                if retry < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 1.5
        
        if remote_version:
            logging.info(f"Latest available version: {remote_version}")
            
            if versions_need_update(local_version, remote_version):
                logging.info(f"New version available: {remote_version} (current: {local_version})")
                # Ask user before updating
                try:
                    pygame.init()
                    info_screen = pygame.display.set_mode((600, 250))
                    pygame.display.set_caption("Multi-Max Update Available")
                    font = pygame.font.SysFont('Arial', 18)
                    button_font = pygame.font.SysFont('Arial', 16)
                    
                    info_screen.fill((0, 0, 0))
                    update_text1 = font.render(f"A new version is available: {remote_version}", True, (255, 255, 255))
                    update_text2 = font.render(f"Current version: {local_version}", True, (255, 255, 255))
                    update_text3 = font.render("Would you like to update now?", True, (255, 255, 255))
                    
                    yes_button = pygame.Rect(150, 150, 100, 40)
                    no_button = pygame.Rect(350, 150, 100, 40)
                    
                    info_screen.blit(update_text1, (50, 50))
                    info_screen.blit(update_text2, (50, 80))
                    info_screen.blit(update_text3, (50, 110))
                    
                    pygame.draw.rect(info_screen, (0, 128, 0), yes_button)
                    pygame.draw.rect(info_screen, (128, 0, 0), no_button)
                    
                    yes_text = button_font.render("Yes", True, (255, 255, 255))
                    no_text = button_font.render("No", True, (255, 255, 255))
                    
                    info_screen.blit(yes_text, (yes_button.x + (yes_button.width - yes_text.get_width()) // 2, 
                                               yes_button.y + (yes_button.height - yes_text.get_height()) // 2))
                    info_screen.blit(no_text, (no_button.x + (no_button.width - no_text.get_width()) // 2,
                                              no_button.y + (no_button.height - no_text.get_height()) // 2))
                    
                    pygame.display.flip()
                    
                    # Wait for user response
                    waiting_for_response = True
                    should_update = False
                    
                    try:
                        while waiting_for_response:
                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    waiting_for_response = False
                                elif event.type == pygame.MOUSEBUTTONDOWN:
                                    if yes_button.collidepoint(event.pos):
                                        should_update = True
                                        waiting_for_response = False
                                    elif no_button.collidepoint(event.pos):
                                        waiting_for_response = False
                                elif event.type == pygame.KEYDOWN:
                                    if event.key == pygame.K_RETURN or event.key == pygame.K_y:
                                        should_update = True
                                        waiting_for_response = False
                                    elif event.key == pygame.K_ESCAPE or event.key == pygame.K_n:
                                        waiting_for_response = False
                    except Exception as e:
                        logging.error(f"Error handling user input for update: {e}")
                        should_update = False
                        
                    try:
                        pygame.quit()
                    except:
                        pass
                    
                    if should_update:
                        return perform_auto_update()
                except Exception as e:
                    logging.error(f"Error showing update dialog: {e}")
            else:
                logging.info("You have the latest version.")
        else:
            logging.warning("Could not check for updates. Will continue with current version.")
    except Exception as e:
        logging.error(f"Error in update check process: {e}")
        logging.error(traceback.format_exc())
    
    return False

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
        
        # Add retry logic for more robust handling
        max_retries = 3
        retry_delay = 1.0
        
        for retry in range(max_retries):
            try:
                new_url = get_stream_url(query_url)
                if new_url:
                    logging.info("Successfully obtained fresh YouTube stream URL")
                    return new_url
                    
                if retry < max_retries - 1:
                    logging.warning(f"Failed to get URL, retrying in {retry_delay}s (attempt {retry+1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 1.5  # Exponential backoff
                else:
                    logging.warning("Maximum retries reached, using original URL")
                    return original_url
            except Exception as retry_e:
                logging.error(f"Error during URL refresh attempt {retry+1}: {retry_e}")
                if retry < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 1.5
                    
        return None
    except Exception as e:
        logging.error(f"Failed to refresh YouTube URL: {e}")
        return None

def start_frame_reader_thread(video_source, stream_url, original_youtube_url, buffer_size=60):
    """Start a background thread to read frames from the video source."""
    global frame_buffer, frame_buffer_lock, frame_reader_thread, should_stop_frame_reader
    global current_stream_url, last_url_refresh_time
    
    current_stream_url = stream_url
    last_url_refresh_time = time.time()
    
    # Initialize with a new buffer - make sure we clear any old buffer first
    if frame_buffer is not None:
        try:
            with frame_buffer_lock:
                while not frame_buffer.empty():
                    try:
                        frame_buffer.get_nowait()
                    except queue.Empty:
                        break
        except Exception as e:
            logging.error(f"Error clearing existing frame buffer: {e}")
    
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
        reconnection_backoff = 1.0
        max_reconnection_backoff = 15.0
        
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
        http_error_threshold = 3
        last_error_time = 0
        error_reset_interval = 60
        
        while not should_stop_frame_reader:
            try:
                current_time = time.time()
                
                # Handle initial setup and reconnection logic
                if thread_cap is None or (consecutive_failures > 0 and current_time - last_reconnect_attempt > reconnection_backoff):
                    try:
                        # Clean up previous capture if it exists
                        if thread_cap is not None:
                            thread_cap.release()
                            thread_cap = None
                            gc.collect()  # Help prevent memory leaks
                        
                        # Check if we need to refresh the URL due to errors or time elapsed
                        url_refresh_needed = (http_errors_count >= http_error_threshold or 
                                            current_time - last_url_refresh_time > stream_url_refresh_interval)
                        
                        if url_refresh_needed:
                            logging.info("Refreshing YouTube URL due to connection issues or timeout")
                            fresh_url = refresh_youtube_url(original_youtube_url)
                            if fresh_url:
                                current_stream_url = fresh_url
                                last_url_refresh_time = current_time
                                http_errors_count = 0
                                logging.info("Using fresh YouTube URL")
                        
                        logging.info(f"Opening video capture in thread with URI: {current_stream_url[:100]}...")
                        thread_cap = cv2.VideoCapture(current_stream_url, cv2.CAP_FFMPEG)
                        thread_cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                        thread_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                        
                        if not thread_cap.isOpened():
                            raise Exception("Failed to open video capture")
                            
                        last_reconnect_attempt = current_time
                        if consecutive_failures > 0:
                            logging.info("Successfully reconnected to video stream")
                            consecutive_failures = 0
                            # Reset backoff after successful connection
                            reconnection_backoff = 1.0
                        
                    except Exception as e:
                        consecutive_failures += 1
                        logging.error(f"Error setting up video capture: {e}")
                        
                        # Apply exponential backoff for reconnection attempts
                        reconnection_backoff = min(reconnection_backoff * 1.5, max_reconnection_backoff)
                        logging.info(f"Will retry reconnection in {reconnection_backoff:.1f} seconds")
                        
                        # Sleep briefly to avoid tight loop
                        time.sleep(0.5)
                        continue
                
                # Skip if not enough time has passed since last frame (maintain target FPS)
                if current_time - last_frame_time < frame_interval:
                    time.sleep(0.001)  # Short sleep to avoid CPU spinning
                    continue
                
                # Read a frame from the video capture
                ret, frame = None, None
                if thread_cap is not None and thread_cap.isOpened():
                    ret, frame = thread_cap.read()
                    
                if ret and frame is not None:
                    frames_read += 1
                    consecutive_failures = 0
                    
                    # If buffer is getting full (over threshold), drop frames to avoid lag
                    drop_frame = False
                    with frame_buffer_lock:
                        buffer_fullness = frame_buffer.qsize() / frame_buffer.maxsize
                        drop_frame = buffer_fullness > frame_drop_threshold
                    
                    if drop_frame:
                        frames_dropped += 1
                        if frames_dropped % 30 == 0:  # Log only occasionally to avoid spam
                            logging.debug(f"Dropped frame to prevent buffer overflow (dropped {frames_dropped} total)")
                    else:
                        # Add the frame to the buffer
                        try:
                            with frame_buffer_lock:
                                if not frame_buffer.full():
                                    frame_buffer.put_nowait(frame)
                                else:
                                    # If buffer is full despite our throttling, log it
                                    logging.warning("Buffer full, dropped frame")
                                    frames_dropped += 1
                        except Exception as e:
                            logging.error(f"Error adding frame to buffer: {e}")
                    
                    # Update timing for frame rate control
                    last_frame_time = current_time
                    
                    # Reset error counters after successful reads
                    if current_time - last_error_time > error_reset_interval:
                        http_errors_count = 0
                    
                else:
                    consecutive_failures += 1
                    
                    # Consider this likely to be an HTTP error after multiple consecutive failures
                    if consecutive_failures >= 2:
                        current_time = time.time()
                        if current_time - last_error_time > 5:
                            http_errors_count += 1
                            last_error_time = current_time
                            logging.warning(f"Connection interrupted, counted as HTTP error #{http_errors_count}")
                    
                    if consecutive_failures > max_consecutive_failures:
                        # Force URL refresh on persistent failures
                        http_errors_count = http_error_threshold
                        logging.warning(f"Persistent connection failure, will force URL refresh")
                    
                    # Backoff on failures to avoid hammering the server
                    time.sleep(0.1 * min(consecutive_failures, 10))
            
            except Exception as e:
                logging.error(f"Error in frame reader worker: {e}")
                consecutive_failures += 1
                time.sleep(0.5)  # Sleep to avoid tight loop on errors
        
        # Clean up before exiting thread
        if thread_cap is not None:
            thread_cap.release()
        
        logging.info(f"Frame reader thread stopped. Read {frames_read} frames, dropped {frames_dropped}")
    
    # Create and start the thread
    frame_reader_thread = threading.Thread(target=frame_reader_worker, daemon=True)
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
        logging.info("Waiting for frame reader thread to terminate...")
        frame_reader_thread.join(timeout=2.0)
        if frame_reader_thread.is_alive():
            logging.warning("Frame reader thread did not terminate gracefully")
            
    if frame_buffer is not None:
        try:
            with frame_buffer_lock:
                while not frame_buffer.empty():
                    try:
                        frame_buffer.get_nowait()
                    except queue.Empty:
                        break
        except Exception as e:
            logging.error(f"Error clearing frame buffer: {e}")

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
is_fullscreen = False  # Track fullscreen state
window_size = (1280, 720)  # Store original window size
frame_display_rect = None  # Rectangle for preserving aspect ratio
ASPECT_RATIO = 16/9  # Standard 16:9 aspect ratio for videos

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
    Create a fractal grid from the live frame and previous output.
    
    Args:
        live_frame: Current video frame
        prev_output: Previous fractal output
        grid_size: Size of the grid (e.g., 2 = 2x2 grid)
        source_position: Position of the source (1=top-left, 2=center, 3=top-right)
    
    Returns:
        Fractal grid frame
    """
    if live_frame is None or prev_output is None:
        return None
        
    h, w = live_frame.shape[:2]
    cell_h = h // grid_size
    cell_w = w // grid_size
    
    # Handle source position
    if source_position == 1:
        # Top-left
        source_i, source_j = 0, 0
    elif source_position == 3:
        # Top-right
        source_i, source_j = 0, grid_size - 1
    else:
        # Center (position 2)
        if grid_size == 2:
            # Special case for 2x2 grid - use top-right instead of center
            source_i, source_j = 0, 1
        else:
            # For grids 3x3 or larger with odd sizes, use true center
            # For even-sized grids, use the cell just to the top-left of center
            source_i = (grid_size - 1) // 2
            source_j = (grid_size - 1) // 2
    
    # Create output frame with same dimensions as live frame
    grid_frame = np.zeros_like(live_frame)
    
    # Pre-allocate buffers for resized frames
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
    """Compute fractal output for a given depth."""
    if depth == 0:
        return np.zeros_like(live_frame)
    else:
        prev = compute_fractal_depth(live_frame, depth - 1)
        return create_fractal_grid(live_frame, prev, 2, 3)

def calculate_display_rect(screen_width, screen_height, aspect_ratio=None):
    """
    Calculate the largest rectangle with the given aspect ratio that fits in the screen.
    Returns (x, y, width, height) for positioning the frame centered on screen with letterbox/pillarbox as needed.
    """
    # Default to 16:9 if no aspect ratio provided
    if aspect_ratio is None:
        aspect_ratio = ASPECT_RATIO
    
    # Avoid division by zero    
    if screen_height <= 0 or screen_width <= 0:
        logging.warning(f"Invalid screen dimensions: {screen_width}x{screen_height}")
        return (0, 0, max(1, screen_width), max(1, screen_height))
        
    screen_aspect = screen_width / screen_height
    
    # Allow a small tolerance (e.g., if screen is 16:10 but close enough to 16:9)
    if abs(screen_aspect - aspect_ratio) < 0.05:
        # If screen is already very close to 16:9, use full screen
        return (0, 0, screen_width, screen_height)
    
    if screen_aspect > aspect_ratio:
        # Screen is wider than target - pillarbox (black bars on sides)
        display_height = screen_height
        display_width = int(display_height * aspect_ratio)
        x_offset = (screen_width - display_width) // 2
        return (x_offset, 0, display_width, display_height)
    else:
        # Screen is taller than target - letterbox (black bars on top/bottom)
        display_width = screen_width
        display_height = int(display_width / aspect_ratio)
        y_offset = (screen_height - display_height) // 2
        return (0, y_offset, display_width, display_height)

def get_optimal_fullscreen_dimensions():
    """
    Get the optimal fullscreen dimensions for the current display.
    This queries the actual hardware display size rather than relying on window dimensions.
    """
    try:
        display_info = pygame.display.Info()
        # Get the true display size
        true_width, true_height = display_info.current_w, display_info.current_h
        
        # Handle edge case of invalid dimensions
        if true_width <= 0 or true_height <= 0:
            logging.warning("Could not determine display dimensions, using defaults")
            true_width = 1280
            true_height = 720
            
        logging.info(f"Detected display dimensions: {true_width}x{true_height}")
        return true_width, true_height
    except Exception as e:
        logging.error(f"Error getting display dimensions: {e}")
        return 1280, 720  # Safe fallback

def handle_keyboard_event(key_name, mod=None):
    """Handle keyboard inputs for adjusting settings."""
    global grid_size, depth, debug_mode, show_info, info_hidden_time, mode, fractal_grid_size, fractal_debug
    global fractal_source_position, fractal_depth, prev_frames, is_fullscreen, window_size, screen, frame_display_rect

    old_grid_size = grid_size
    old_depth = depth
    old_fractal_grid_size = fractal_grid_size
    old_fractal_source_position = fractal_source_position
    
    try:
        is_repeat = key_name in ['up', 'down'] and key_pressed.get(key_name, False) and time.time() - key_press_start.get(key_name, 0) > key_repeat_delay
        
        if key_name == 'w' or key_name == 'escape':
            # For 'escape', only handle toggle if already in fullscreen mode
            if key_name == 'escape' and not is_fullscreen:
                return  # Let the main event loop handle escape key when not in fullscreen
                
            # Toggle fullscreen state
            is_fullscreen = not is_fullscreen if key_name == 'w' else False
            
            if is_fullscreen:
                # Store current window position and size before switching
                window_size = screen.get_size()
                
                try:
                    # Get the true screen dimensions from the hardware
                    true_width, true_height = get_optimal_fullscreen_dimensions()
                    
                    # Switch to fullscreen using the detected resolution
                    screen = pygame.display.set_mode((true_width, true_height), pygame.FULLSCREEN)
                    
                    # Precalculate display rectangle for aspect ratio preservation
                    frame_display_rect = calculate_display_rect(true_width, true_height)
                    
                    logging.info(f"Full screen mode enabled with display area: {frame_display_rect}")
                except Exception as e:
                    logging.error(f"Error switching to fullscreen: {e}")
                    # Fall back to windowed mode in case of error
                    is_fullscreen = False
                    try:
                        screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
                        frame_display_rect = calculate_display_rect(window_size[0], window_size[1])
                    except Exception as fallback_e:
                        logging.error(f"Error in fallback to window mode: {fallback_e}")
                        # Last resort fallback with standard dimensions
                        try:
                            screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
                            window_size = (1280, 720)
                            frame_display_rect = calculate_display_rect(1280, 720)
                        except Exception as last_e:
                            logging.error(f"Critical display error: {last_e}")
            else:
                try:
                    # Return to windowed mode with the original size
                    screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
                    # Calculate aspect ratio preserving rect for window mode as well
                    frame_display_rect = calculate_display_rect(window_size[0], window_size[1])
                    logging.info(f"Window mode enabled with size: {window_size}")
                except Exception as e:
                    logging.error(f"Error switching to windowed mode: {e}")
                    # Fallback
                    try:
                        screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
                        frame_display_rect = calculate_display_rect(1280, 720)
                        window_size = (1280, 720)
                    except Exception as fallback_e:
                        logging.error(f"Critical display error in windowed fallback: {fallback_e}")
        
        elif key_name == '4' and mode == "fractal" and (not mod or not (mod & pygame.KMOD_SHIFT)):
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
        # Avoid getting stuck in fullscreen if there was an error
        if is_fullscreen and key_name == 'escape':
            try:
                is_fullscreen = False
                screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
                window_size = (1280, 720)
                frame_display_rect = calculate_display_rect(1280, 720)
            except Exception as fallback_error:
                logging.error(f"Emergency fallback to windowed mode failed: {fallback_error}")
                pass

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
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
            "--socket-timeout", "10",         # Increased socket timeout to 10 seconds
            "--retries", "5",                 # Retry 5 times if download fails
            "--file-access-retries", "5",     # Retry 5 times if file access fails
            url
        ]
        
        # Run yt-dlp with enhanced options
        logging.debug(f"Running yt-dlp with args: {' '.join(yt_dlp_args)}")
        
        try:
            result = subprocess.run(yt_dlp_args, capture_output=True, text=True, check=True, timeout=30)
            
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
            # Add more detailed error information to help debugging YouTube issues
            error_msg = e.stderr.lower() if e.stderr else ""
            if "403" in error_msg:
                logging.error("YouTube returned 403 Forbidden - you might be rate limited or IP blocked")
            elif "404" in error_msg:
                logging.error("YouTube returned 404 Not Found - the video might be private or removed")
            elif "unavailable" in error_msg:
                logging.error("YouTube video unavailable - it might be region-restricted or age-restricted")
                
            # Retry with simplified options if the initial request failed
            try:
                logging.info("Retrying with simplified options...")
                result = subprocess.run([
                    "yt-dlp", 
                    "-f", "best", 
                    "--get-url",
                    "--no-check-certificate",
                    "--retries", "3",
                    url
                ], capture_output=True, text=True, check=True, timeout=30)
                
                stream_url = result.stdout.strip()
                if stream_url and stream_url.startswith(('http://', 'https://')):
                    logging.info("Successfully retrieved stream URL on retry")
                    return stream_url
                logging.error("Retry attempt failed to return a valid URL")
            except subprocess.SubprocessError as retry_error:
                logging.error(f"Retry attempt failed: {retry_error}")
            except Exception as retry_error:
                logging.error(f"Unexpected error during retry: {retry_error}")
            return None
    except subprocess.TimeoutExpired:
        logging.error("yt-dlp process timed out after 30 seconds")
        return None
    except FileNotFoundError:
        logging.error("yt-dlp not found. Please install yt-dlp using 'pip install yt-dlp'")
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
    # Calculate which level has closest to 1px cells
    floor_level = int(np.floor(one_pixel_level))
    ceil_level = int(np.ceil(one_pixel_level))
    floor_diff = abs(min_dim / (fractal_grid_size ** floor_level) - 1.0)
    ceil_diff = abs(min_dim / (fractal_grid_size ** ceil_level) - 1.0)
    exact_pixel_level = floor_level if floor_diff < ceil_diff else ceil_level
    
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

def shutdown_application():
    """Perform a graceful shutdown, cleaning up all resources."""
    global running, frame_reader_thread, should_stop_frame_reader, cap, screen
    
    logging.info("Performing graceful shutdown...")
    
    # Stop the main loop
    running = False
    
    # Capture local refs to these objects to avoid AttributeError if they're set to None during cleanup
    current_cap = cap if 'cap' in globals() else None
    current_frame_reader_thread = frame_reader_thread if 'frame_reader_thread' in globals() else None
    current_frame_buffer = frame_buffer if 'frame_buffer' in globals() else None
    current_frame_buffer_lock = frame_buffer_lock if 'frame_buffer_lock' in globals() else None
    
    # Stop the frame reader thread
    if current_frame_reader_thread is not None and current_frame_reader_thread.is_alive():
        logging.info("Shutting down frame reader thread...")
        should_stop_frame_reader = True
        try:
            current_frame_reader_thread.join(timeout=3.0)
            if current_frame_reader_thread.is_alive():
                logging.warning("Frame reader thread did not terminate within timeout")
        except Exception as e:
            logging.error(f"Error while stopping frame reader thread: {e}")
    
    # Release video capture resources
    if current_cap is not None:
        try:
            logging.info("Releasing video capture resources...")
            current_cap.release()
            cap = None
        except Exception as e:
            logging.error(f"Error while releasing video capture: {e}")
    
    # Clear the frame buffer
    if current_frame_buffer is not None and current_frame_buffer_lock is not None:
        try:
            logging.info("Clearing frame buffer...")
            with current_frame_buffer_lock:
                while not current_frame_buffer.empty():
                    try:
                        current_frame_buffer.get_nowait()
                    except queue.Empty:
                        break
        except Exception as e:
            logging.error(f"Error while clearing frame buffer: {e}")
    
    # Clean up pygame resources
    try:
        logging.info("Cleaning up pygame resources...")
        pygame.mixer.quit()
        pygame.quit()
    except Exception as e:
        logging.error(f"Error while quitting pygame: {e}")
    
    # Release any numpy arrays in global space
    try:
        for var_name in ['prev_output_frame', 'prev_frames']:
            if var_name in globals() and globals()[var_name] is not None:
                logging.info(f"Releasing {var_name}...")
                if isinstance(globals()[var_name], list):
                    for i in range(len(globals()[var_name])):
                        globals()[var_name][i] = None
                else:
                    globals()[var_name] = None
    except Exception as e:
        logging.error(f"Error while releasing numpy arrays: {e}")
    
    # Final garbage collection
    try:
        logging.info("Performing final memory cleanup...")
        gc.collect()
    except Exception as e:
        logging.error(f"Error during final garbage collection: {e}")
    
    logging.info("Shutdown complete. Goodbye!")

def main():
    """Main function for Windows-compatible video processing."""
    global running, grid_size, depth, debug_mode, show_info, frame_count, processed_count, displayed_count, dropped_count
    global hardware_acceleration_available, cap, mode, fractal_grid_size, fractal_debug, fractal_source_position, prev_output_frame
    global enable_memory_tracing, fractal_depth, current_stream_url, last_buffer_warning_time, frame_drop_threshold
    global key_pressed, key_press_start, key_last_repeat, last_url_refresh_time, stream_url_refresh_interval, prev_frames
    global is_fullscreen, window_size, screen, frame_display_rect, ASPECT_RATIO
    
    # Register clean shutdown with atexit
    atexit.register(shutdown_application)
    
    key_pressed = {}
    key_press_start = {}
    key_last_repeat = {}
    
    screen = None
    cap = None
    frame = None
    processed_frame = None
    frame_surface = None
    font = None
    
    prev_frames = [None] * fractal_depth  # Initialize with the correct size
    last_fractal_cleanup_time = time.time()
    fractal_cleanup_interval = 10.0
    
    try:
        # Initialize pygame with simplified error handling for Windows compatibility
        try:
            pygame.init()
            logging.info("Pygame initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing pygame: {e}")
            # Try individual components that we absolutely need
            try:
                pygame.display.init()
                logging.info("Pygame display initialized successfully")
            except Exception as e2:
                logging.error(f"Failed to initialize pygame display: {e2}")
                return
                
            try:
                pygame.font.init()
                logging.info("Pygame font initialized successfully")
            except Exception as e2:
                logging.warning(f"Failed to initialize pygame font: {e2}")
                # Continue without fonts, we'll handle this later
                
        # Try to quit mixer to avoid sound issues (commonly causes problems)
        try:
            pygame.mixer.quit()
        except:
            pass
            
        # Check for updates before proceeding
        if check_for_updates():
            # If update process was triggered, the function won't return
            # But just in case it does return True, we should exit
            try:
                pygame.quit()
            except:
                pass
            return
        
        # Continue with normal initialization
        pygame.display.set_caption("MultiMax Grid")
        
        try:
            display_info = pygame.display.Info()
            screen_width = min(1280, display_info.current_w - 100)
            screen_height = min(720, display_info.current_h - 100)
        except:
            # Fallback if display info fails
            screen_width = 1280
            screen_height = 720
            
        try:
            screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
            logging.info(f"Display initialized with size: {screen_width}x{screen_height}")
        except Exception as e:
            logging.error(f"Failed to create display: {e}")
            try:
                # Fallback to a smaller size
                screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE)
                screen_width, screen_height = 800, 600
                logging.info("Falling back to 800x600 display")
            except Exception as e2:
                logging.error(f"Critical display error: {e2}")
                return
                
        # Calculate initial display rectangle
        frame_display_rect = calculate_display_rect(screen_width, screen_height)
        
        screen.fill((0, 0, 0))  # Use black background instead of white
        pygame.display.flip()
        
        # Set up signal handlers for graceful termination
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
        parser.add_argument('--fractal-source', type=int, choices=[1, 2, 3], default=3, help='Source position in fractal mode')
        
        args = parser.parse_args()
        
        # Define a fallback clock in case pygame clock fails
        class FallbackClock:
            """Fallback clock implementation if pygame clock fails."""
            def __init__(self):
                self.last_tick = time.time()
                
            def tick(self, fps):
                """Simulate pygame clock tick."""
                now = time.time()
                elapsed = now - self.last_tick
                target_elapsed = 1.0 / fps
                if elapsed < target_elapsed:
                    time.sleep(target_elapsed - elapsed)
                self.last_tick = time.time()
                return int((1.0 / max(elapsed, 1e-6)) + 0.5)
                
            def get_fps(self):
                """Return estimated FPS."""
                return 30.0
        
        # Initialize pygame clock for FPS control
        try:
            clock = pygame.time.Clock()
        except Exception as e:
            logging.error(f"Error creating pygame clock: {e}")
            clock = FallbackClock()
            
        grid_size, depth = args.grid_size, args.depth
        mode = args.mode
        # Fix: properly initialize fractal_source_position from command line args
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
            position_desc = {1: "top-left", 2: "center", 3: "top-right"}
            actual_position = position_desc[fractal_source_position]
            if fractal_source_position == 2 and fractal_grid_size == 2:
                actual_position = "top-right (special 2x2 case)"
            logging.info(f"Fractal source position: {actual_position}")
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
        
        # Improve font creation to be more robust
        try:
            font = pygame.font.SysFont("Arial", 24)
            # Quick test to see if font works
            test_text = font.render("Test", True, (255, 255, 255))
        except Exception as e:
            logging.warning(f"Could not create Arial font: {e}")
            try:
                # Try available system fonts
                available_fonts = pygame.font.get_fonts()
                if available_fonts:
                    for test_font in available_fonts[:3]:  # Try the first few fonts
                        try:
                            font = pygame.font.SysFont(test_font, 24)
                            test_text = font.render("Test", True, (255, 255, 255))
                            logging.info(f"Using alternative font: {test_font}")
                            break
                        except:
                            continue
                    else:  # This runs if no font in the loop worked
                        raise Exception("No system fonts worked")
                else:
                    raise Exception("No system fonts available")
            except Exception as e2:
                logging.error(f"Could not create any system font: {e2}")
                # Last resort
                try:
                    font = pygame.font.Font(None, 24)  # Default pygame font
                    logging.info("Using pygame default font")
                except:
                    logging.error("Critical: All font creation methods failed")
                    # Continue without a font - we'll handle rendering carefully
        
        screen.fill((0, 0, 0))
        
        # Draw loading message with extra error checking
        try:
            if font:
                loading_text = font.render("Loading Video...", True, (255, 255, 255))
                center_x = screen.get_width() // 2 - loading_text.get_width() // 2
                center_y = screen.get_height() // 2 - loading_text.get_height() // 2
                screen.blit(loading_text, (center_x, center_y))
            else:
                # Fallback if no font
                pygame.draw.rect(screen, (255, 255, 255), (screen.get_width() // 2 - 100, 
                                                         screen.get_height() // 2 - 10, 200, 20))
        except Exception as e:
            logging.error(f"Error drawing loading text: {e}")
            # Just show a white rectangle as a visual indicator
            try:
                pygame.draw.rect(screen, (255, 255, 255), (screen.get_width() // 2 - 100, 
                                                         screen.get_height() // 2 - 10, 200, 20))
            except:
                pass
                
        pygame.display.flip()
        
        stream_url = get_stream_url(args.youtube_url)
        if not stream_url:
            logging.error(f"Failed to get stream URL for {args.youtube_url}")
            logging.error("Please check that your YOUTUBE_URL in .env is valid and accessible")
            logging.error("Attempting to use the default demo URL as fallback...")
            
            # Try the default URL as a fallback
            fallback_url = "https://www.youtube.com/watch?v=ZzWBpGwKoaI"
            if args.youtube_url != fallback_url:
                logging.info(f"Trying fallback URL: {fallback_url}")
                stream_url = get_stream_url(fallback_url)
                
            if not stream_url:
                logging.error("Could not get any valid stream URL, exiting")
                return

        # Update the display with connecting message, with robust error handling
        try:
            screen.fill((0, 0, 0))
            
            try:
                if font:
                    # First line
                    connecting_text = font.render("Connecting to Video Stream...", True, (255, 255, 255))
                    center_x = screen.get_width() // 2 - connecting_text.get_width() // 2
                    center_y = screen.get_height() // 2 - connecting_text.get_height()
                    screen.blit(connecting_text, (center_x, center_y))
                    
                    # Second line
                    please_wait_text = font.render("Please wait...", True, (255, 255, 255))
                    center_x = screen.get_width() // 2 - please_wait_text.get_width() // 2
                    center_y = screen.get_height() // 2 + 30
                    screen.blit(please_wait_text, (center_x, center_y))
                else:
                    # Visual indicator if no font
                    pygame.draw.rect(screen, (255, 255, 255), (screen.get_width() // 2 - 100, 
                                                             screen.get_height() // 2 - 10, 200, 5))
                    pygame.draw.rect(screen, (200, 200, 200), (screen.get_width() // 2 - 50, 
                                                             screen.get_height() // 2 + 15, 100, 5))
            except Exception as text_error:
                logging.error(f"Error rendering connecting text: {text_error}")
                # Fallback visual indicator
                try:
                    pygame.draw.rect(screen, (255, 255, 255), (screen.get_width() // 2 - 100, 
                                                             screen.get_height() // 2 - 10, 200, 5))
                except:
                    pass
                    
            pygame.display.flip()
        except Exception as e:
            logging.error(f"Error updating connecting screen: {e}")

        logging.info(f"Initializing video capture for stream: {stream_url}")
        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            logging.error(f"Failed to open video stream: {stream_url}")
            logging.error("Please check your internet connection and YouTube URL validity")
            
            # Try one more time with a small delay
            time.sleep(2.0)
            logging.info("Attempting to reconnect to the stream...")
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            
            if not cap.isOpened():
                logging.error("Second connection attempt failed, exiting")
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

        # Buffer waiting loop with improved robustness
        buffer_wait_start = time.time()
        buffer_wait_timeout = 5.0
        dots = 0
        
        while time.time() - buffer_wait_start < buffer_wait_timeout:
            if frame_buffer is not None and not frame_buffer.empty():
                logging.info(f"Buffer started filling after {time.time() - buffer_wait_start:.1f} seconds")
                break
                
            # Update dots animation
            dots = (dots + 1) % 4
            dots_str = "." * dots
            elapsed = int(time.time() - buffer_wait_start)
            
            time.sleep(0.1)
            
            try:
                screen.fill((0, 0, 0))
                
                try:
                    if font:
                        loading_message = f"Loading stream{dots_str} ({elapsed}s)"
                        loading_text = font.render(loading_message, True, (255, 255, 255))
                        center_x = screen.get_width() // 2 - loading_text.get_width() // 2
                        center_y = screen.get_height() // 2 - loading_text.get_height() // 2
                        screen.blit(loading_text, (center_x, center_y))
                    else:
                        # Visual progress indicator if no font
                        progress = min(elapsed / buffer_wait_timeout, 1.0)
                        width = int(200 * progress)
                        pygame.draw.rect(screen, (100, 100, 100), (screen.get_width() // 2 - 100, 
                                                                 screen.get_height() // 2 - 10, 200, 20))
                        pygame.draw.rect(screen, (255, 255, 255), (screen.get_width() // 2 - 100, 
                                                                 screen.get_height() // 2 - 10, width, 20))
                except Exception as text_error:
                    # Last resort visual indicator
                    try:
                        progress = min(elapsed / buffer_wait_timeout, 1.0)
                        width = int(200 * progress)
                        pygame.draw.rect(screen, (100, 100, 100), (screen.get_width() // 2 - 100, 
                                                                 screen.get_height() // 2 - 10, 200, 20))
                        pygame.draw.rect(screen, (255, 255, 255), (screen.get_width() // 2 - 100, 
                                                                 screen.get_height() // 2 - 10, width, 20))
                    except:
                        pass
                
                # Try to update the display
                try:
                    pygame.display.flip()
                except Exception as flip_error:
                    logging.error(f"Failed to update display: {flip_error}")
                
            except Exception as outer_error:
                logging.error(f"Error in buffer wait loop: {outer_error}")

        while running:
            current_time = time.time()
            frame_start_time = time.time()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    logging.info("Window close event detected, initiating shutdown...")
                    running = False
                    break
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    if is_fullscreen:
                        # If in fullscreen mode, ESC toggles back to windowed mode
                        logging.info("Escape key pressed, toggling back to windowed mode...")
                        handle_keyboard_event('escape')
                    else:
                        # Only exit if not in fullscreen mode
                        logging.info("Escape key pressed, initiating shutdown...")
                        running = False
                        break
                elif event.type == pygame.KEYDOWN:
                    key = pygame.key.name(event.key)
                    mod = event.mod
                    if key in ['up', 'down', 'd', 's', 'f', 'w'] or (key in '0123456789' and mode != "fractal_depth"):
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
                elif event.type == pygame.VIDEORESIZE:
                    # Handle window resize events
                    if not is_fullscreen:
                        window_size = (event.w, event.h)
                        screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
                        # Recalculate the display rectangle
                        frame_display_rect = calculate_display_rect(event.w, event.h)
                        logging.debug(f"Window resized to {window_size}, display area: {frame_display_rect}")

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
                    # Create a new copy to avoid reference issues
                    old_prev_output = prev_output_frame
                    prev_output_frame = processed_frame.copy()
                    # Clean up old frame
                    if old_prev_output is not None and id(old_prev_output) != id(processed_frame):
                        del old_prev_output
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
                        
                        # Clean up as we go
                        if id(old_frame) != id(prev_frames[d]):
                            del old_frame
                        if d > 0 and id(old_temp) != id(temp):
                            del old_temp
                        del new_frame
                        
                        # More frequent garbage collection for deeper levels
                        if d > 50 and d % 10 == 0:
                            gc.collect()
                    processed_frame = temp
                    # temp will be cleaned up at the end of the loop
            
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
                
                # Handle display based on fullscreen status
                screen.fill((0, 0, 0))  # Fill with black for letterboxing/pillarboxing
                
                if frame_display_rect:
                    # Use calculated display rectangle to maintain aspect ratio
                    x, y, width, height = frame_display_rect
                    scaled_surface = pygame.transform.scale(frame_surface, (width, height))
                    screen.blit(scaled_surface, (x, y))
                else:
                    # Fallback - should not normally reach here
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
                        texts.append("Press W to toggle full screen mode, ESC to exit fullscreen")
                    elif mode == "fractal":
                        texts.append("Press s to show/hide this info, d to debug, up/down grid size, f to switch modes")
                        texts.append("Press 1-3 to change source position: 1=top-left, 2=center (odd grids + 2x2 special case), 3=top-right")
                        texts.append("Press 4 to switch to fractal depth mode")
                        texts.append("Press W to toggle full screen mode, ESC to exit fullscreen")
                    elif mode == "fractal_depth":
                        texts.append("Press s to show/hide this info, d to debug, f to switch modes")
                        texts.append("Press UP/DOWN arrow keys to increase/decrease depth level (1-100)")
                        texts.append("Press W to toggle full screen mode, ESC to exit fullscreen")
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

            # Clean up variables explicitly to help with memory management
            frame_processing_time = time.time() - frame_start_time
            
            # Explicit cleanup
            local_vars_to_clean = ['frame', 'processed_frame', 'rgb_frame', 'rgb_frame_swapped', 'temp', 'new_frame', 'old_frame', 'old_temp', 'scaled_surface']
            for var in local_vars_to_clean:
                if var in locals() and locals()[var] is not None:
                    del locals()[var]
                    
            # Manage fractal depth frames periodically
            if mode == "fractal_depth" and frame_counter % 30 == 0:
                if len(prev_frames) > fractal_depth:
                    for i in range(fractal_depth, len(prev_frames)):
                        prev_frames[i] = None
                    prev_frames[fractal_depth:] = []
                    gc.collect()
                    
            frame_counter += 1

            # Monitor memory usage
            memory_usage = process.memory_info().rss / 1024 / 1024
            if memory_usage > 2000:
                logging.debug(f"High memory usage detected: {memory_usage:.2f} MB")

            # Log performance stats periodically
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
            
            # Control frame rate
            clock.tick(30)
            
            # Maintain target framerate
            target_frame_time = 1.0 / 30
            if frame_processing_time < target_frame_time:
                time.sleep(target_frame_time - frame_processing_time)
                
        # End of main loop
        logging.info("Main loop exited, beginning cleanup...")
        
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt detected")
    except Exception as e:
        logging.error(f"Main function crashed: {e}\n{traceback.format_exc()}")
    finally:
        # Perform cleanup
        shutdown_application()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Program crashed: {e}\n{traceback.format_exc()}")
        # Ensure shutdown happens even after uncaught exceptions
        if 'shutdown_application' in globals():
            try:
                shutdown_application()
            except:
                pass