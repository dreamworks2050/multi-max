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
from memory_profiler import profile

# Windows-specific version marker - DO NOT REMOVE - used by installer to verify correct version
__windows_specific_version__ = True

# Global variables section
os.environ['OPENCV_FFMPEG_DEBUG'] = '0'  # Disable verbose FFmpeg output

# Load environment variables - prioritize windows/.env if it exists
if os.path.exists(os.path.join('windows', '.env')):
    load_dotenv(os.path.join('windows', '.env'))
else:
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

# Load hardware acceleration settings - Windows version always uses software mode
force_hardware_acceleration = False
allow_software_fallback = True

# Load memory tracing settings
enable_memory_tracing = os.getenv('ENABLE_MEMORY_TRACING', 'false').lower() == 'true'

# Global variables for hardware acceleration
is_windows = platform.system() == 'Windows'
hardware_acceleration_available = False  # Always false for Windows version

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
frame_buffer_size = int(os.getenv('FRAME_BUFFER_SIZE', '60'))  # Get buffer size from env or use default
should_stop_frame_reader = False
current_stream_url = None  # Store the stream URL globally
last_buffer_warning_time = 0  # Track when we last issued a warning
frame_drop_threshold = 0.8  # Drop frames if buffer is more than 80% full
stream_url_refresh_interval = 5 * 60  # Refresh YouTube URL every 5 minutes
last_url_refresh_time = 0  # Last time the URL was refreshed

# Windows-specific image processing functions
def process_image_software(cv_img):
    """
    Process image using OpenCV software rendering for Windows.
    This is a replacement for the Mac-specific Quartz hardware acceleration.
    
    Args:
        cv_img: OpenCV image to process
        
    Returns:
        Processed OpenCV image
    """
    try:
        if cv_img is None or cv_img.size == 0:
            logging.error("Input image to process_image_software is None or empty")
            return None
        
        # Convert to RGBA if needed
        if len(cv_img.shape) < 3:
            cv_img_rgba = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGRA)
        elif cv_img.shape[2] == 3:
            cv_img_rgba = cv2.cvtColor(cv_img, cv2.COLOR_BGR2BGRA)
        else:
            cv_img_rgba = cv_img.copy()
        
        # Ensure memory is contiguous
        if not cv_img_rgba.flags['C_CONTIGUOUS']:
            cv_img_rgba = np.ascontiguousarray(cv_img_rgba)
        
        # Apply any desired image processing here
        # This is a simple pass-through for now, but you can add filters as needed
        
        return cv_img_rgba
    except Exception as e:
        logging.error(f"Error in process_image_software: {e}\n{traceback.format_exc()}")
        return None

def apply_filter(cv_img, filter_name, filter_params=None):
    """
    Apply a filter to an OpenCV image using software processing.
    This is a replacement for the CoreImage filters on Mac.
    
    Args:
        cv_img: OpenCV image to filter
        filter_name: Name of the filter to apply
        filter_params: Optional parameters for the filter
        
    Returns:
        Filtered OpenCV image
    """
    if cv_img is None:
        return None
    
    if filter_params is None:
        filter_params = {}
    
    try:
        result = cv_img.copy()
        
        if filter_name == 'blur':
            # Gaussian blur
            blur_radius = filter_params.get('radius', 5)
            if blur_radius > 0:
                result = cv2.GaussianBlur(result, (blur_radius*2+1, blur_radius*2+1), 0)
        
        elif filter_name == 'saturation':
            # Adjust saturation
            saturation = filter_params.get('amount', 1.5)
            if len(result.shape) == 3:  # Color image
                hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 1] = hsv[:, :, 1] * saturation
                hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
                result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        elif filter_name == 'brightness':
            # Adjust brightness
            brightness = filter_params.get('amount', 1.2)
            result = cv2.convertScaleAbs(result, alpha=brightness, beta=0)
        
        elif filter_name == 'contrast':
            # Adjust contrast
            contrast = filter_params.get('amount', 1.2)
            result = cv2.convertScaleAbs(result, alpha=contrast, beta=0)
        
        elif filter_name == 'edge':
            # Edge detection
            result = cv2.Canny(result, 100, 200)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        # Add more filters as needed
        
        return result
    
    except Exception as e:
        logging.error(f"Error applying filter {filter_name}: {e}")
        return cv_img  # Return original on error

def composite_images(base_img, overlay_img, x, y, opacity=1.0):
    """
    Composite an overlay image onto a base image at position (x, y) with given opacity.
    This is a replacement for CoreImage compositing on Mac.
    
    Args:
        base_img: Base OpenCV image
        overlay_img: Overlay OpenCV image
        x, y: Position to place overlay
        opacity: Opacity of overlay (0.0-1.0)
        
    Returns:
        Composited OpenCV image
    """
    try:
        if base_img is None or overlay_img is None:
            return base_img if base_img is not None else overlay_img
        
        # Create a region of interest
        rows, cols = overlay_img.shape[:2]
        roi = base_img[y:y+rows, x:x+cols]
        
        # Check if ROI is valid
        if roi.shape[0] <= 0 or roi.shape[1] <= 0:
            return base_img
        
        # Create a mask of the overlay and its inverse
        if len(overlay_img.shape) == 3 and overlay_img.shape[2] == 4:
            # If overlay has alpha channel
            overlay_rgba = overlay_img
            alpha = overlay_rgba[:, :, 3] / 255.0 * opacity
            
            # Resize alpha if needed to match ROI
            if alpha.shape[:2] != roi.shape[:2]:
                alpha = cv2.resize(alpha, (roi.shape[1], roi.shape[0]))
            
            alpha = np.stack([alpha, alpha, alpha], axis=2)
            
            # Resize overlay_rgb if needed
            overlay_rgb = overlay_rgba[:, :, :3]
            if overlay_rgb.shape[:2] != roi.shape[:2]:
                overlay_rgb = cv2.resize(overlay_rgb, (roi.shape[1], roi.shape[0]))
            
            # Blend images
            blended = cv2.addWeighted(roi, 1 - alpha, overlay_rgb, alpha, 0)
            
            # Copy the blended image back to the base image
            result = base_img.copy()
            result[y:y+rows, x:x+cols] = blended
            return result
        else:
            # If overlay doesn't have alpha, just use standard alpha blending
            result = base_img.copy()
            blended = cv2.addWeighted(roi, 1-opacity, overlay_img, opacity, 0)
            result[y:y+rows, x:x+cols] = blended
            return result
    
    except Exception as e:
        logging.error(f"Error in composite_images: {e}")
        return base_img  # Return original on error

# Replace Mac-specific conversion functions with Windows versions
def cv_to_ci_image(cv_img):
    """
    Windows replacement for Mac's cv_to_ci_image function.
    Simply processes the image with software rendering and returns it.
    
    Args:
        cv_img: OpenCV image
        
    Returns:
        Processed OpenCV image (not a CoreImage as in Mac version)
    """
    return process_image_software(cv_img)

def ci_to_cv_image(ci_img, width, height):
    """
    Windows replacement for Mac's ci_to_cv_image function.
    Since ci_img is already an OpenCV image in this version, this just returns it.
    
    Args:
        ci_img: "CoreImage" (actually OpenCV in Windows version)
        width, height: Image dimensions
        
    Returns:
        OpenCV image
    """
    return ci_img

# Continue with the rest of the code...
# The key is to maintain the same function names and interfaces
# but replace Mac-specific implementations with Windows ones

def main():
    """Main function to run the application."""
    logging.info("Starting Multi-Max in Windows mode with software rendering")
    # ... rest of the main function will be retained

if __name__ == "__main__":
    main() 