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
import Quartz  # For Core Graphics functions

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def configure_logging(level_name):
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.getLogger().setLevel(level)

# Load hardware acceleration settings from environment
force_hardware_acceleration = os.getenv('FORCE_HARDWARE_ACCELERATION', 'true').lower() == 'true'
allow_software_fallback = os.getenv('ALLOW_SOFTWARE_FALLBACK', 'false').lower() == 'true'

# Check for Apple Silicon and initialize hardware acceleration
is_apple_silicon = platform.system() == 'Darwin' and platform.machine().startswith('arm')
hardware_acceleration_available = False

if is_apple_silicon:
    try:
        import objc
        from Foundation import NSData, NSMutableData
        from Quartz import CIContext, CIImage, CIFilter, kCIFormatRGBA8, kCIContextUseSoftwareRenderer, CIVector
        from CoreFoundation import CFDataCreate
        
        # Create an optimized Core Image context specifically for Apple Silicon
        # Setting advanced options for maximum hardware acceleration
        context_options = {
            kCIContextUseSoftwareRenderer: False,              # Force GPU rendering
            "kCIContextOutputColorSpace": None,                # Use device color space
            "kCIContextWorkingColorSpace": None,               # Use device color space
            "kCIContextHighQualityDownsample": False,          # Prioritize speed over quality
            "kCIContextOutputPremultiplied": True,             # Pre-multiplied alpha for speed
            "kCIContextCacheIntermediates": True,              # Cache intermediate results
            "kCIContextPriorityRequestLow": False              # High priority processing
        }
        
        # Create the optimized context
        ci_context = CIContext.contextWithOptions_(context_options)
        hardware_acceleration_available = True
        logging.info("Hardware acceleration enabled and optimized for Apple Silicon")
    except ImportError as e:
        if force_hardware_acceleration and not allow_software_fallback:
            logging.error(f"Hardware acceleration required but unavailable: {e}")
            raise RuntimeError("Hardware acceleration required but unavailable. Set ALLOW_SOFTWARE_FALLBACK=true to allow software processing.")
        logging.warning(f"Hardware acceleration unavailable, falling back to software processing: {e}")
else:
    if force_hardware_acceleration and not allow_software_fallback:
        logging.error("Hardware acceleration required but not running on Apple Silicon")
        raise RuntimeError("Hardware acceleration required but not running on Apple Silicon. Set ALLOW_SOFTWARE_FALLBACK=true to allow software processing.")
    logging.info("Not running on Apple Silicon, using software processing")

# Global variables
grid_size = 3
depth = 1
running = True
debug_mode = True  # Start in debug mode
show_info = True   # Whether to show the information overlay
info_hidden_time = 0  # Timestamp when info was hidden

# Stats
frame_count = 0
processed_count = 0
displayed_count = 0
dropped_count = 0

# Performance tracking
downsample_times = []
downsample_sizes = []
max_tracked_samples = 50  # Keep tracking data for the last 50 frames

# Keyboard controls
def handle_keyboard_event(key_name):
    global grid_size, depth, debug_mode, show_info, info_hidden_time
    
    old_grid_size = grid_size
    old_depth = depth
    old_debug_mode = debug_mode
    
    if key_name == 'up':
        grid_size += 1
        logging.info(f"Grid size changed: {old_grid_size}x{old_grid_size} → {grid_size}x{grid_size}")
    elif key_name == 'down' and grid_size > 1:
        grid_size -= 1
        logging.info(f"Grid size changed: {old_grid_size}x{old_grid_size} → {grid_size}x{grid_size}")
    elif key_name in '1234567890':
        try:
            depth = int(key_name) if key_name != '0' else 10
            logging.info(f"Recursion depth changed: {old_depth} → {depth}")
        except ValueError:
            logging.warning(f"Invalid depth value: {key_name}")
    elif key_name == 'd':
        debug_mode = not debug_mode
        mode_text = "enabled" if debug_mode else "disabled"
        old_mode_text = "enabled" if old_debug_mode else "disabled"
        logging.info(f"Debug mode changed: {old_mode_text} → {mode_text}")
    elif key_name == 's':
        show_info = not show_info
        logging.info(f"Info overlay: {'shown' if show_info else 'hidden'}")
        
        # Record the time when info was hidden
        if not show_info:
            info_hidden_time = time.time()
        
    # Log detailed grid information after change
    if old_grid_size != grid_size or old_depth != depth:
        # Calculate expected cell sizes for the new configuration
        frame_h, frame_w = 720, 1280  # Assume standard HD resolution
        cell_h = frame_h // grid_size
        cell_w = frame_w // grid_size
        
        # Calculate memory requirements and potential challenges
        frame_size_mb = (frame_h * frame_w * 3) / (1024 * 1024)  # Size in MB (assuming 3 channels)
        cell_size_mb = frame_size_mb / (grid_size * grid_size)
        
        # Log configuration details
        logging.info(f"Configuration details for Grid {grid_size}x{grid_size}, Depth {depth}:")
        logging.info(f"  - Cell size: ~{cell_w}x{cell_h} pixels")
        logging.info(f"  - Total cells: {grid_size * grid_size} per frame")
        logging.info(f"  - Estimated memory per frame: {frame_size_mb:.2f} MB")
        logging.info(f"  - Estimated memory per cell: {cell_size_mb:.4f} MB")

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    global running
    print("\nShutting down...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

# Hardware-accelerated frame conversion functions
def cv_to_ci_image(cv_img):
    # Check if image is already in BGRA format to avoid unnecessary conversion
    if cv_img.shape[2] == 3:
        cv_img_rgba = cv2.cvtColor(cv_img, cv2.COLOR_BGR2BGRA)
    else:
        cv_img_rgba = cv_img.copy()  # Already in BGRA format
        
    height, width = cv_img_rgba.shape[:2]
    
    # Use contiguous C-style memory layout for better performance
    if not cv_img_rgba.flags['C_CONTIGUOUS']:
        cv_img_rgba = np.ascontiguousarray(cv_img_rgba)
    
    data = cv_img_rgba.tobytes()
    data_provider = CFDataCreate(None, data, len(data))
    
    # Use optimized format for Apple Silicon
    ci_img = CIImage.imageWithBitmapData_bytesPerRow_size_format_colorSpace_(
        data_provider, width * 4, Quartz.CGSizeMake(width, height), kCIFormatRGBA8, None)
        
    # Explicitly hint for hardware processing by setting properties
    ci_img = ci_img.imageBySettingProperties_({
        "CIImageAppleM1Optimized": True
    })
    
    return ci_img

def ci_to_cv_image(ci_img, width, height):
    # Create optimized buffer with proper alignment for Apple Silicon (16-byte alignment)
    buffer_size = height * width * 4
    aligned_size = ((buffer_size + 15) // 16) * 16  # Align to 16 bytes
    output_data = NSMutableData.dataWithLength_(aligned_size)
    
    # Use low-overhead rendering mode
    ci_context.render_toBitmap_rowBytes_bounds_format_colorSpace_(
        ci_img, output_data.mutableBytes(), width * 4, Quartz.CGRectMake(0, 0, width, height),
        kCIFormatRGBA8, None)
    
    try:
        # Create numpy array without copying data
        buffer = np.frombuffer(output_data, dtype=np.uint8)
        
        # Check if the buffer size matches the expected dimensions
        expected_size = height * width * 4
        actual_size = buffer.size
        
        if actual_size != expected_size:
            # Create a more detailed log message for buffer size mismatch
            mismatch_details = (
                f"Buffer size mismatch: expected {expected_size} ({width}x{height}x4), "
                f"got {actual_size} (diff: {actual_size - expected_size} bytes). "
                f"This typically happens with grid size transitions."
            )
            
            # Log more details about the dimensions
            if actual_size < expected_size:
                logging.warning(f"{mismatch_details} Buffer is too small - will adjust dimensions.")
            else:
                logging.warning(f"{mismatch_details} Buffer is too large - will adjust dimensions.")
            
            # Calculate the closest dimensions that match the buffer size
            # Try to maintain aspect ratio as much as possible
            adjusted_height = int(np.sqrt((actual_size / 4) * (height / width)))
            adjusted_width = int(actual_size / (4 * adjusted_height))
            
            logging.debug(f"Initial adjustment: {width}x{height} → {adjusted_width}x{adjusted_height}")
            
            # Make a final adjustment to ensure exact size match
            adjustment_iterations = 0
            max_iterations = 10  # Prevent infinite loops
            while (adjusted_height * adjusted_width * 4) != actual_size and adjustment_iterations < max_iterations:
                adjustment_iterations += 1
                old_width, old_height = adjusted_width, adjusted_height
                
                if (adjusted_height * adjusted_width * 4) < actual_size:
                    adjusted_width += 1
                else:
                    adjusted_width -= 1
                
                # If width adjustment doesn't work, try height
                if (adjusted_height * adjusted_width * 4) != actual_size:
                    if (adjusted_height * adjusted_width * 4) < actual_size:
                        adjusted_height += 1
                    else:
                        adjusted_height -= 1
                
                logging.debug(f"Adjustment iteration {adjustment_iterations}: {old_width}x{old_height} → {adjusted_width}x{adjusted_height}")
            
            # Update the dimensions
            height, width = adjusted_height, adjusted_width
            logging.debug(f"Final dimensions: {width}x{height} (buffer size: {actual_size}, expected: {height * width * 4})")
        
        # Reshape with the potentially adjusted dimensions
        buffer = buffer[:height * width * 4].reshape(height, width, 4)
        
        # For BGR output (3 channels), perform efficient conversion
        if buffer.shape[2] == 4:
            cv_img = cv2.cvtColor(buffer, cv2.COLOR_BGRA2BGR)
        else:
            cv_img = buffer[:,:,:3]  # Just take the first 3 channels if already BGR
            
        return cv_img
        
    except Exception as e:
        # Enhanced error reporting for reshape errors
        error_message = str(e)
        if "reshape" in error_message:
            logging.error(
                f"Reshape error: {error_message}. Buffer size: {buffer.size}, "
                f"Target shape: ({height}, {width}, 4), Required size: {height * width * 4}. "
                f"This typically happens with grid size transitions or memory alignment issues."
            )
        else:
            logging.warning(f"Image conversion error: {e}. Falling back to copy-based approach.")
        
        # Copy the data to a new buffer with exact size
        raw_data = np.array(output_data)
        if raw_data.size >= width * height * 4:
            # Truncate if necessary
            raw_data = raw_data[:width * height * 4]
            # Reshape and convert
            buffer = raw_data.reshape(height, width, 4)
            cv_img = cv2.cvtColor(buffer, cv2.COLOR_BGRA2BGR)
            return cv_img
        else:
            # If all else fails, create a black image
            logging.error(f"Buffer size {raw_data.size} is too small for {width}x{height}x4. Creating black image.")
            return np.zeros((height, width, 3), dtype=np.uint8)

# Generate a new grid frame from the entire previous frame
def generate_grid_frame(previous_frame, grid_size, current_depth):
    global downsample_times, downsample_sizes
    
    if previous_frame is None:
        logging.error(f"Depth {current_depth}: Previous frame is None")
        return None
    h, w = previous_frame.shape[:2]
    base_cell_h = h // grid_size
    base_cell_w = w // grid_size
    remainder_h = h % grid_size
    remainder_w = w % grid_size

    # Create a blank result frame
    result = np.zeros((h, w, 3), dtype=np.uint8)

    # Frame aspect ratio (16:9)
    frame_aspect = 16 / 9
    
    # Calculate the most common cell size (ignoring remainder distribution)
    # This optimization allows us to generate a single downsampled image once
    common_cell_h = base_cell_h
    common_cell_w = base_cell_w
    
    # Calculate optimal downsampling size to maintain aspect ratio
    # We'll generate one small version of the image with correct aspect ratio
    if frame_aspect > common_cell_w / common_cell_h:
        # Frame is wider than cell, scale based on height
        small_h = int(common_cell_h)
        small_w = int(small_h * frame_aspect)
    else:
        # Frame is taller than cell, scale based on width
        small_w = int(common_cell_w)
        small_h = int(small_w / frame_aspect)

    # Ensure minimum dimensions
    small_w = max(1, small_w)
    small_h = max(1, small_h)

    # Measure downsampling performance
    downsample_start_time = time.time()
    
    # Downsample the input frame once using hardware acceleration
    if is_apple_silicon and hardware_acceleration_available:
        try:
            # Use Core Image for hardware-accelerated downsampling
            ci_img = cv_to_ci_image(previous_frame)
            scale_filter = CIFilter.filterWithName_("CILanczosScaleTransform")
            scale_filter.setValue_forKey_(ci_img, "inputImage")
            
            # Determine scale factor based on the smaller dimension
            scale_factor = min(small_h / h, small_w / w)
            scale_filter.setValue_forKey_(scale_factor, "inputScale")
            scale_filter.setValue_forKey_(1.0, "inputAspectRatio")
            
            # Get the downsampled image
            small_ci = scale_filter.valueForKey_("outputImage")
            
            # Ensure the dimensions are aligned to 4 bytes for better buffer handling
            # This helps avoid reshape errors by ensuring buffer sizes align properly
            exact_w = int(w * scale_factor)
            exact_h = int(h * scale_factor)
            
            # Adjust dimensions to ensure they're aligned to 4 pixels
            # This helps prevent buffer size mismatches with Core Image
            exact_w = (exact_w + 3) & ~3  # Round up to multiple of 4
            exact_h = (exact_h + 3) & ~3  # Round up to multiple of 4
            
            # Update our working dimensions
            small_w = exact_w
            small_h = exact_h
            
            # Log the exact dimensions we're using
            logging.debug(f"Using aligned dimensions: {small_w}x{small_h}")
            
            # Create the downsampled frame with explicit dimensions
            small_frame = ci_to_cv_image(small_ci, exact_w, exact_h)
            
            # Clean up Core Image objects
            del ci_img, small_ci
            logging.debug(f"Depth {current_depth}: Hardware downsampled frame to {small_w}x{small_h}")
        except Exception as e:
            if force_hardware_acceleration and not allow_software_fallback:
                logging.error(f"Hardware downsampling failed: {e}")
                raise RuntimeError(f"Hardware acceleration required but failed: {e}")
            else:
                logging.warning(f"Hardware downsampling failed, falling back to software: {e}")
                # Try software fallback with explicit integer dimensions
                small_frame = cv2.resize(previous_frame, (int(small_w), int(small_h)), interpolation=cv2.INTER_AREA)
    elif force_hardware_acceleration and not allow_software_fallback:
        logging.error("Hardware acceleration required but not available")
        raise RuntimeError("Hardware acceleration required but not available")
    else:
        # Software fallback
        # Ensure dimensions are integers
        small_w_int = int(small_w)
        small_h_int = int(small_h)
        small_frame = cv2.resize(previous_frame, (small_w_int, small_h_int), interpolation=cv2.INTER_AREA)
        # Update dimensions to what was actually used
        small_w = small_w_int
        small_h = small_h_int
        logging.debug(f"Depth {current_depth}: Software downsampled frame to {small_w}x{small_h}")
    
    # Ensure we have a valid downsampled frame before proceeding
    if small_frame is None or small_frame.size == 0:
        logging.error(f"Depth {current_depth}: Failed to create valid downsampled frame")
        return None

    # Verify the shape and dimensions
    small_h, small_w = small_frame.shape[:2]
    if small_h <= 0 or small_w <= 0:
        logging.error(f"Depth {current_depth}: Invalid downsampled frame dimensions: {small_w}x{small_h}")
        return None

    # Log actual dimensions for debugging
    logging.debug(f"Depth {current_depth}: Using downsampled frame of size {small_w}x{small_h}")

    # Record performance metrics
    downsample_time = time.time() - downsample_start_time
    downsample_times.append(downsample_time)
    downsample_sizes.append((w * h, small_w * small_h))  # Original and downsampled sizes
    
    # Keep only the last N samples
    if len(downsample_times) > max_tracked_samples:
        downsample_times.pop(0)
        downsample_sizes.pop(0)
    
    # Now iterate through grid and place the downsampled image in each cell
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate actual cell size, distributing remainder pixels
            cell_h = base_cell_h + (1 if i < remainder_h else 0)
            cell_w = base_cell_w + (1 if j < remainder_w else 0)

            # Calculate exact cell position to avoid gaps
            # This ensures cells are placed edge-to-edge with no gaps
            y_start = 0
            for k in range(i):
                y_start += base_cell_h + (1 if k < remainder_h else 0)

            y_end = y_start + cell_h

            x_start = 0
            for k in range(j):
                x_start += base_cell_w + (1 if k < remainder_w else 0)

            x_end = x_start + cell_w

            # Verify we're not exceeding frame boundaries
            y_end = min(y_end, h)
            x_end = min(x_end, w)

            # Double-check cell dimensions after boundary adjustment
            cell_h = y_end - y_start
            cell_w = x_end - x_start

            if cell_h < 1 or cell_w < 1:
                # If cell is too small, fill with average color
                avg_color = cv2.mean(small_frame)[:3]
                result[y_start:y_end, x_start:x_end] = avg_color
                continue
            
            # Crop the downsampled image to fit this cell (might need slight adjustments)
            if frame_aspect > cell_w / cell_h:
                # Frame is wider than cell, crop horizontally
                # Ensure all slice indices are integers
                # Calculate what width the downsampled image should be to maintain aspect ratio
                required_width = int(small_h * (cell_w / cell_h))
                crop_x = int((small_w - required_width) / 2)  # Center crop
                # Ensure we don't have negative indices
                crop_x = max(0, crop_x)
                crop_w = min(required_width, small_w - crop_x)
                
                # Ensure all values are valid integers
                crop_x = int(crop_x)
                crop_w = int(crop_w)
                
                if crop_x + crop_w <= small_w:
                    cell_img = small_frame[:, crop_x:crop_x + crop_w]
                else:
                    # In case of any edge case, just take what we can
                    cell_img = small_frame[:, :small_w]
            else:
                # Frame is taller than cell, crop vertically
                # Ensure all slice indices are integers
                # Calculate what height the downsampled image should be to maintain aspect ratio
                required_height = int(small_w * (cell_h / cell_w))
                crop_y = int((small_h - required_height) / 2)  # Center crop
                # Ensure we don't have negative indices
                crop_y = max(0, crop_y)
                crop_h = min(required_height, small_h - crop_y)
                
                # Ensure all values are valid integers
                crop_y = int(crop_y)
                crop_h = int(crop_h)
                
                if crop_y + crop_h <= small_h:
                    cell_img = small_frame[crop_y:crop_y + crop_h, :]
                else:
                    # In case of any edge case, just take what we can
                    cell_img = small_frame[:small_h, :]
            
            # Resize to exact cell dimensions if needed (should be minimal adjustment)
            if cell_img.shape[0] != cell_h or cell_img.shape[1] != cell_w:
                # Use INTER_LINEAR for better quality when upscaling slightly
                interpolation = cv2.INTER_LINEAR if (cell_img.shape[0] < cell_h or cell_img.shape[1] < cell_w) else cv2.INTER_AREA
                
                # Ensure we're resizing to exact dimensions to avoid gaps
                cell_img = cv2.resize(cell_img, (cell_w, cell_h), interpolation=interpolation)
                
                # Verify the resized image has the exact dimensions we need
                if cell_img.shape[0] != cell_h or cell_img.shape[1] != cell_w:
                    logging.warning(f"Resize failed to produce exact dimensions: got {cell_img.shape[:2]}, needed {cell_h}x{cell_w}")
                    # Force exact dimensions by creating a new image and copying
                    exact_cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                    h_copy = min(cell_img.shape[0], cell_h)
                    w_copy = min(cell_img.shape[1], cell_w)
                    exact_cell[:h_copy, :w_copy] = cell_img[:h_copy, :w_copy]
                    cell_img = exact_cell
            
            # Place the cell in the result frame
            result[y_start:y_end, x_start:x_end] = cell_img
    
    # Clean up the downsampled frame
    del small_frame
    gc.collect()  # Force garbage collection
    
    # Verify there are no black pixels in the result (all zeros)
    black_pixels = np.all(result == 0, axis=2)
    if np.any(black_pixels):
        # Count and log the number of black pixels
        num_black = np.count_nonzero(black_pixels)
        percent_black = (num_black / (h * w)) * 100
        logging.debug(f"Depth {current_depth}: Found {num_black} black pixels ({percent_black:.4f}% of frame)")
        
        # Fill black pixels with nearest non-black neighbor
        if percent_black > 0.01:  # More than 0.01% of the frame is black
            if grid_size <= 10:  # For smaller grids, use pixel-by-pixel approach
                # Simple approach: for each black pixel, find a non-black neighbor
                y_coords, x_coords = np.where(black_pixels)
                for y, x in zip(y_coords, x_coords):
                    # Check 8-connected neighbors
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = y + dy, x + dx
                            if (0 <= ny < h and 0 <= nx < w and not black_pixels[ny, nx]):
                                result[y, x] = result[ny, nx]
                                break
                        else:
                            continue
                        break
            else:
                # For larger grids, use a more efficient morphological approach
                logging.debug(f"Using morphological operations to fill black lines")
                # Create a mask of non-black pixels
                mask = (~black_pixels).astype(np.uint8) * 255
                # Dilate the mask to fill small gaps
                kernel = np.ones((3, 3), np.uint8)
                dilated = cv2.dilate(mask, kernel, iterations=1)
                # Create a mask of pixels to fill (black pixels that are next to non-black pixels)
                fill_mask = (dilated > 0) & black_pixels
                
                # For each channel, use inpainting to fill the black areas
                for c in range(3):
                    channel = result[:, :, c]
                    # Simple inpainting: dilate and use the dilated values for black pixels
                    dilated_channel = cv2.dilate(channel, kernel, iterations=1)
                    channel[fill_mask] = dilated_channel[fill_mask]

    logging.debug(f"Depth {current_depth}: Generated grid frame {w}x{h} with grid {grid_size}x{grid_size}")
    return result

# Apply grid effect across multiple frames with memory management
def apply_grid_effect(frame, grid_size, depth):
    if frame is None:
        return None
    if debug_mode or depth == 0:
        logging.debug("Depth 0: Returning original frame (debug mode or depth=0)")
        return frame.copy()
    
    try:
        previous_frame = frame.copy()  # Start with the original frame
        
        # Initialize a counter for failed attempts
        failed_depths = 0
        max_failed_depths = 2  # Allow at most 2 failures before giving up
        
        for d in range(1, depth + 1):
            try:
                new_frame = generate_grid_frame(previous_frame, grid_size, d)
                if new_frame is None:
                    logging.error(f"Depth {d}: Failed to generate grid frame (returned None)")
                    failed_depths += 1
                    if failed_depths > max_failed_depths:
                        logging.error(f"Too many failures ({failed_depths}), aborting grid effect")
                        break
                    # Use previous frame and continue
                    new_frame = previous_frame.copy()
                else:
                    # Reset failed depth counter on success
                    failed_depths = 0
                    
                # Replace previous frame with the new one and delete the old one
                del previous_frame
                previous_frame = new_frame
                gc.collect()  # Force garbage collection after each depth
            except Exception as e:
                logging.error(f"Depth {d}: Error during grid generation: {e}")
                failed_depths += 1
                if failed_depths > max_failed_depths:
                    logging.error(f"Too many failures ({failed_depths}), aborting grid effect")
                    break
                # Just continue with previous frame
                
        # Return the last successfully generated frame
        return previous_frame
    except Exception as e:
        logging.error(f"Grid effect pipeline error: {e}")
        # Return original frame as fallback
        return frame.copy()

# Fetch YouTube stream URL
def get_stream_url(url):
    try:
        result = subprocess.run(["yt-dlp", "-f", "best", "--get-url", url],
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception as e:
        logging.error(f"Error getting YouTube stream URL: {e}")
        return None

# Update the get_grid_layer_breakdown function to include more units for large numbers
def get_grid_layer_breakdown(grid_size, max_depth):
    """
    Calculate and format information about each grid layer.
    
    Args:
        grid_size: The size of the grid (NxN)
        max_depth: The maximum recursion depth
        
    Returns:
        tuple: (list of layer descriptions, total screens, formatted total)
    """
    if max_depth <= 0 or grid_size <= 0:
        return [], 0, "0"
    
    layer_info = []
    total_screens = 0
    
    # Calculate screens at each layer
    for d in range(1, max_depth + 1):
        # For depth d, the number of screens is grid_size^(2*d)
        # Each cell contains grid_size^2 subcells, and this compounds with depth
        screens_at_depth = grid_size ** (2 * d)
        total_screens += screens_at_depth
        
        # Format the layer information
        if d == 1:
            layer_desc = f"Grid frame {d} - depth {d} - {grid_size}x{grid_size} (total {screens_at_depth:,} screens)"
        else:
            prev_screens = grid_size ** (2 * (d-1))
            layer_desc = f"Grid frame {d} - depth {d} - {grid_size}x{grid_size} of frame {d-1} = (total {screens_at_depth:,} screens)"
        
        layer_info.append(layer_desc)
    
    # Format the total with appropriate scale using an expanded set of units
    # Define the units and their corresponding powers of 10
    number_units = [
        (1e6, "million"),
        (1e9, "billion"),
        (1e12, "trillion"),
        (1e15, "quadrillion"),
        (1e18, "quintillion"),
        (1e21, "sextillion"),
        (1e24, "septillion"),
        (1e27, "octillion"),
        (1e30, "nonillion"),
        (1e33, "decillion"),
        (1e36, "undecillion"),
        (1e39, "duodecillion"),
        (1e42, "tredecillion"),
        (1e45, "quattuordecillion"),
        (1e48, "quindecillion"),
        (1e51, "sexdecillion"),
        (1e54, "septendecillion"),
        (1e57, "octodecillion"),
        (1e60, "novemdecillion"),
        (1e63, "vigintillion"),
        (1e66, "unvigintillion"),
        (1e69, "duovigintillion"),
        (1e72, "trevigintillion"),
        (1e75, "quattuorvigintillion"),
        (1e78, "quinvigintillion"),
        (1e81, "sexvigintillion"),
        (1e84, "septenvigintillion"),
        (1e87, "octovigintillion"),
        (1e90, "novemvigintillion"),
        (1e93, "trigintillion"),
        (1e96, "untrigintillion"),
        (1e99, "duotrigintillion"),
        (1e100, "googol"),
        (1e303, "centillion")
    ]
    
    # Start with the basic formatted number
    formatted_total = f"{total_screens:,}"
    
    # Find the appropriate unit
    for threshold, unit in reversed(number_units):
        if total_screens >= threshold:
            value = total_screens / threshold
            formatted_total = f"{total_screens:,} ({value:.2f} {unit})"
            break
    
    # Special handling for absolutely massive numbers
    if total_screens >= 1e303:
        # Calculate the approximate power of 10
        power = int(np.log10(total_screens))
        if power >= 303:
            formatted_total += f" (approximately 10^{power})"
    
    return layer_info, total_screens, formatted_total

# Main application loop
def main():
    global running, grid_size, depth, debug_mode, show_info, frame_count, processed_count, displayed_count, dropped_count
    global force_hardware_acceleration, allow_software_fallback
    
    parser = argparse.ArgumentParser(description='Recursive Video Grid')
    parser.add_argument('--grid-size', type=int, default=3)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--youtube-url', type=str, default=os.getenv('YOUTUBE_URL'))
    parser.add_argument('--log-level', type=str, default='INFO', 
                        help='Logging level: DEBUG for all logs, INFO for summaries only (default: INFO)')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--force-hardware', action='store_true', help='Force hardware acceleration')
    parser.add_argument('--allow-software', action='store_true', help='Allow software fallback if hardware acceleration fails')
    
    args = parser.parse_args()
    grid_size, depth = args.grid_size, args.depth
    configure_logging(args.log_level)
    debug_mode = args.debug
    show_info = True   # Whether to show the information overlay
    
    # Override environment settings with command-line options if provided
    if args.force_hardware:
        force_hardware_acceleration = True
        logging.info("Hardware acceleration forced by command-line option")
    if args.allow_software:
        allow_software_fallback = True
        logging.info("Software fallback allowed by command-line option")
    
    logging.info(f"Starting with debug={debug_mode} (toggle with 'd')")
    logging.info(f"Hardware acceleration: forced={force_hardware_acceleration}, software fallback={allow_software_fallback}")
    logging.info("Logging set to summary mode - detailed logs will appear every 5 seconds")
    logging.info("For more detailed logs, set --log-level=DEBUG")

    pygame.init()
    screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
    pygame.display.set_caption("Recursive Grid Livestream")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24)

    stream_url = get_stream_url(args.youtube_url)
    if not stream_url:
        logging.error("Failed to get stream URL")
        return

    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logging.error("Failed to open video stream")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    logging.debug(f"Capture resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    # Loading screen
    screen.fill((0, 0, 0))
    loading_text = font.render("Loading video stream...", True, (255, 255, 255))
    screen.blit(loading_text, (screen.get_width() // 2 - loading_text.get_width() // 2,
                               screen.get_height() // 2 - loading_text.get_height() // 2))
    pygame.display.flip()

    # Initialize variables for logging summary
    last_frame_time = time.time()
    last_gc_time = time.time()
    last_status_time = time.time()
    last_process_time = 0
    
    # Stats for summarized logging
    frames_since_last_log = 0
    successful_grids_since_last_log = 0
    failed_grids_since_last_log = 0
    min_process_time = float('inf')
    max_process_time = 0
    total_process_time = 0
    
    while running:
        current_time = time.time()
        loop_start_time = current_time

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN:
                key = pygame.key.name(event.key)
                if key in ['up', 'down', 'd', 's'] or key in '0123456789':
                    handle_keyboard_event(key)

        # Capture frame
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to read frame")
            time.sleep(0.1)
            continue
        frame_count += 1

        # Process frame
        process_start_time = time.time()
        try:
            processed_frame = apply_grid_effect(frame, grid_size, depth)
            processed_count += 1
            last_process_time = time.time() - process_start_time
            
            # Update processing time stats
            frames_since_last_log += 1
            successful_grids_since_last_log += 1
            min_process_time = min(min_process_time, last_process_time)
            max_process_time = max(max_process_time, last_process_time)
            total_process_time += last_process_time
            
        except Exception as e:
            logging.error(f"Error applying effect: {e}")
            processed_frame = frame.copy()
            dropped_count += 1
            failed_grids_since_last_log += 1

        # Display frame
        try:
            # Make sure we have a valid processed frame
            if processed_frame is None or processed_frame.size == 0 or not hasattr(processed_frame, 'shape'):
                logging.warning("Invalid processed frame, using original frame instead")
                processed_frame = frame.copy()
            
            # Verify frame has valid dimensions
            h, w = processed_frame.shape[:2]
            if h <= 0 or w <= 0 or processed_frame.ndim < 2:
                logging.warning(f"Processed frame has invalid dimensions: {processed_frame.shape}, using original frame")
                processed_frame = frame.copy()
            
            # If we have hardware acceleration, use it for color conversion too
            if is_apple_silicon and hardware_acceleration_available:
                try:
                    # Convert BGR to RGB using Core Image
                    ci_img = cv_to_ci_image(processed_frame)
                    color_filter = CIFilter.filterWithName_("CIColorMatrix")
                    color_filter.setValue_forKey_(ci_img, "inputImage")
                    
                    # BGR to RGB conversion matrix (swap red and blue channels)
                    color_filter.setValue_forKey_(CIVector.vectorWithX_Y_Z_W_(0, 0, 1, 0), "inputRVector")
                    color_filter.setValue_forKey_(CIVector.vectorWithX_Y_Z_W_(0, 1, 0, 0), "inputGVector")
                    color_filter.setValue_forKey_(CIVector.vectorWithX_Y_Z_W_(1, 0, 0, 0), "inputBVector")
                    
                    rgb_ci = color_filter.valueForKey_("outputImage")
                    h, w = processed_frame.shape[:2]
                    rgb_frame = ci_to_cv_image(rgb_ci, w, h)
                    
                    # Clean up Core Image objects
                    del ci_img, rgb_ci
                except Exception as e:
                    if force_hardware_acceleration and not allow_software_fallback:
                        logging.error(f"Hardware color conversion failed: {e}")
                        raise RuntimeError(f"Hardware acceleration required but failed: {e}")
                    else:
                        logging.warning(f"Hardware color conversion failed, falling back to software: {e}")
                        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            else:
                # Software fallback
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
            surface = pygame.surfarray.make_surface(rgb_frame.swapaxes(0, 1))
            screen.blit(pygame.transform.scale(surface, screen.get_size()), (0, 0))
            displayed_count += 1

            # Only show the overlay if show_info is True
            if show_info:
                # Overlay stats - first prepare all text items
                hw_status = "Enabled" if hardware_acceleration_available else "Disabled"
                if is_apple_silicon and force_hardware_acceleration:
                    hw_status += " (Forced)"
                if not hardware_acceleration_available and allow_software_fallback:
                    hw_status += " (Software Fallback)"
                
                texts = [
                    f"Grid: {grid_size}x{grid_size}, Depth: {depth}",
                    f"Captured: {frame_count}, Displayed: {displayed_count}",
                    f"Processing: {last_process_time * 1000:.1f}ms",
                    f"FPS: {int(clock.get_fps())}",
                    f"Hardware: {hw_status}",
                ]

                # Add downsampling performance metrics if available
                if downsample_times:
                    avg_time = sum(downsample_times) / len(downsample_times) * 1000  # Convert to ms
                    if len(downsample_sizes) > 0:
                        last_original, last_small = downsample_sizes[-1]
                        reduction_ratio = last_original / max(1, last_small)
                        texts.append(f"Downsampling: {avg_time:.1f}ms, Reduction: {reduction_ratio:.1f}x")

                # Add grid layer breakdown if not in debug mode and depth > 0
                if not debug_mode and depth > 0:
                    texts.append("")  # Add a blank line for spacing
                    texts.append("Grid Layer Breakdown:")
                    layer_info, total_screens, formatted_total = get_grid_layer_breakdown(grid_size, depth)
                    for layer in layer_info:
                        texts.append(f"  {layer}")
                    texts.append(f"  Total screens = {formatted_total}")

                texts.append(f"Mode: {'DEBUG' if debug_mode else 'GRID'} (press 'd' to toggle)")
                
                # Add blank lines for spacing
                texts.append("")
                texts.append("")
                
                # Add the help text at the bottom
                texts.append("Press s to show/hide this info, d to debug, up/down grid size, 1 - 0 multiply depth")

                # Calculate required height based on number of text lines
                line_height = 25  # Height per line of text
                padding = 20      # Padding at top and bottom
                required_height = len(texts) * line_height + padding * 2

                # Create a surface with the calculated height
                info_surface = pygame.Surface((screen.get_width(), required_height), pygame.SRCALPHA)
                info_surface.fill((0, 0, 0, 128))

                # Render all text items
                for i, text in enumerate(texts):
                    text_surface = font.render(text, True, (255, 255, 255))
                    info_surface.blit(text_surface, (10, padding + i * line_height))
                
                screen.blit(info_surface, (0, 0))
            else:
                # When info is hidden, show a temporary help message
                current_time = time.time()
                time_since_hidden = current_time - info_hidden_time
                
                # Only show the message for 5 seconds
                if time_since_hidden < 5.0:
                    # Calculate fade-out effect for the last second
                    alpha = 128  # Default opacity
                    if time_since_hidden > 4.0:
                        # Linear fade from 128 to 0 during the last second
                        fade_progress = time_since_hidden - 4.0  # 0.0 to 1.0
                        alpha = int(128 * (1.0 - fade_progress))
                        alpha = max(0, min(128, alpha))  # Clamp between 0-128
                    
                    # Create a small overlay at the bottom of the screen
                    min_height = 40
                    min_info = pygame.Surface((screen.get_width(), min_height), pygame.SRCALPHA)
                    min_info.fill((0, 0, 0, alpha))
                    
                    # Make text opacity match the background
                    text_color = (255, 255, 255, min(255, alpha * 2))  # Text stays more visible longer
                    help_text = font.render("Press 's' to show info", True, text_color)
                    min_info.blit(help_text, (10, 10))
                    
                    # Position at the bottom of the screen
                    screen.blit(min_info, (0, screen.get_height() - min_height))

            pygame.display.flip()
        except Exception as e:
            logging.error(f"Display error: {e}")
            dropped_count += 1

        # Periodic garbage collection and status logging
        if current_time - last_gc_time > 10.0:
            gc.collect()
            last_gc_time = current_time
            
        # Summarized logging every 5 seconds
        if current_time - last_status_time > 5.0:
            if frames_since_last_log > 0:
                avg_process_time = total_process_time / frames_since_last_log * 1000
                
                # Build a comprehensive summary
                summary_lines = [
                    f"===== PERFORMANCE SUMMARY =====",
                    f"Configuration: Grid {grid_size}x{grid_size}, Depth {depth}, Mode: {'DEBUG' if debug_mode else 'GRID'}",
                    f"Frames: Captured {frame_count}, Displayed {displayed_count}, Dropped {dropped_count}",
                    f"Processing: Min {min_process_time * 1000:.1f}ms, Max {max_process_time * 1000:.1f}ms, Avg {avg_process_time:.1f}ms",
                    f"Success Rate: {successful_grids_since_last_log}/{frames_since_last_log} frames processed successfully",
                    f"FPS: {int(clock.get_fps())}"
                ]
                
                # Add hardware acceleration status
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
                
                # Add downsampling info if available
                if downsample_times and downsample_sizes:
                    avg_downsample_time = sum(downsample_times) / len(downsample_times) * 1000
                    last_original, last_small = downsample_sizes[-1]
                    reduction_ratio = last_original / max(1, last_small)
                    summary_lines.append(f"Downsampling: {avg_downsample_time:.1f}ms, Reduction Ratio: {reduction_ratio:.1f}x")
                
                # Add grid layer breakdown
                if not debug_mode and depth > 0:
                    summary_lines.append("\nGrid Layer Breakdown:")
                    layer_info, total_screens, formatted_total = get_grid_layer_breakdown(grid_size, depth)
                    for layer in layer_info:
                        summary_lines.append(f"  {layer}")
                    summary_lines.append(f"  Total screens = {formatted_total}")
                
                # Log the summary
                for line in summary_lines:
                    logging.info(line)
            
            # Reset summary stats
            frames_since_last_log = 0
            successful_grids_since_last_log = 0
            failed_grids_since_last_log = 0
            min_process_time = float('inf')
            max_process_time = 0
            total_process_time = 0
            last_status_time = current_time
        
        clock.tick(30)

    cap.release()
    pygame.quit()
    logging.info("Shutdown complete")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Main crashed: {e}\n{traceback.format_exc()}")