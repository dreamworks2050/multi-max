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
from memory_profiler import profile

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def configure_logging(level_name):
    """Configure logging level based on input string."""
    try:
        level = getattr(logging, level_name.upper(), logging.INFO)
        logging.getLogger().setLevel(level)
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
grid_size = 3
depth = 1
running = True
debug_mode = True
show_info = True
info_hidden_time = 0

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

def handle_keyboard_event(key_name):
    """Handle keyboard inputs for adjusting grid settings."""
    global grid_size, depth, debug_mode, show_info, info_hidden_time
    
    old_grid_size = grid_size
    old_depth = depth
    old_debug_mode = debug_mode
    
    try:
        if key_name == 'up':
            grid_size += 1
            logging.info(f"Grid size changed: {old_grid_size}x{old_grid_size} → {grid_size}x{grid_size}")
        elif key_name == 'down' and grid_size > 1:
            grid_size -= 1
            logging.info(f"Grid size changed: {old_grid_size}x{old_grid_size} → {grid_size}x{grid_size}")
        elif key_name in '1234567890':
            depth = int(key_name) if key_name != '0' else 10
            logging.info(f"Recursion depth changed: {old_depth} → {depth}")
        elif key_name == 'd':
            debug_mode = not debug_mode
            mode_text = "enabled" if debug_mode else "disabled"
            old_mode_text = "enabled" if old_debug_mode else "disabled"
            logging.info(f"Debug mode changed: {old_mode_text} → {mode_text}")
        elif key_name == 's':
            show_info = not show_info
            logging.info(f"Info overlay: {'shown' if show_info else 'hidden'}")
            if not show_info:
                info_hidden_time = time.time()
        
        if old_grid_size != grid_size or old_depth != depth:
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

@conditional_profile
def generate_grid_frame(previous_frame, grid_size, current_depth):
    """Generate a grid frame recursively with direct assignment to reduce memory usage."""
    global downsample_times, downsample_sizes
    
    ci_img = None
    scale_filter = None
    small_ci = None
    small_frame = None
    
    try:
        if previous_frame is None or previous_frame.size == 0:
            logging.error(f"Depth {current_depth}: Previous frame is None or empty")
            return None
        
        h, w = previous_frame.shape[:2]
        if h <= 0 or w <= 0:
            logging.error(f"Depth {current_depth}: Invalid frame dimensions: {w}x{h}")
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
        
        if is_apple_silicon and hardware_acceleration_available:
            try:
                import objc
                with objc.autorelease_pool():
                    ci_img = cv_to_ci_image(previous_frame)
                    if ci_img is None:
                        logging.error(f"Depth {current_depth}: Failed to convert frame to CIImage")
                        return None
                    
                    scale_filter = CIFilter.filterWithName_("CILanczosScaleTransform")
                    if scale_filter is None:
                        logging.error(f"Depth {current_depth}: Failed to create CILanczosScaleTransform filter")
                        del ci_img
                        ci_img = None
                        return None
                    
                    scale_filter.setValue_forKey_(ci_img, "inputImage")
                    scale_factor = min(small_h / h, small_w / w)
                    scale_filter.setValue_forKey_(scale_factor, "inputScale")
                    scale_filter.setValue_forKey_(1.0, "inputAspectRatio")
                    
                    small_ci = scale_filter.valueForKey_("outputImage")
                    
                    del ci_img
                    ci_img = None
                    del scale_filter
                    scale_filter = None
                    
                    gc.collect()
                    
                    if small_ci is None:
                        logging.error(f"Depth {current_depth}: Failed to apply scale filter")
                        return None
                    
                    exact_w = int(w * scale_factor)
                    exact_h = int(h * scale_factor)
                    exact_w = (exact_w + 3) & ~3
                    exact_h = (exact_h + 3) & ~3
                    small_w = exact_w
                    small_h = exact_h
                    logging.debug(f"Depth {current_depth}: Hardware downsampled frame to {small_w}x{small_h}")
                    
                    small_frame = ci_to_cv_image(small_ci, exact_w, exact_h)
                    
                    del small_ci
                    small_ci = None
                    
                    ci_context.clearCaches()
                    logging.debug("Cleared Core Image cache after downsampling")
                
                if small_frame is None:
                    logging.error(f"Depth {current_depth}: Failed to convert CIImage back to CV image")
                    return None
                
                gc.collect()
            except Exception as e:
                logging.error(f"Depth {current_depth}: Error in hardware downsampling: {e}\n{traceback.format_exc()}")
                
                if ci_img is not None:
                    del ci_img
                    ci_img = None
                if small_ci is not None:
                    del small_ci
                    small_ci = None
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
            logging.error(f"Depth {current_depth}: Failed to create valid downsampled frame")
            return None

        small_h, small_w = small_frame.shape[:2]
        if small_h <= 0 or small_w <= 0:
            logging.error(f"Depth {current_depth}: Invalid downsampled frame dimensions: {small_w}x{small_h}")
            return None

        logging.debug(f"Depth {current_depth}: Using downsampled frame of size {small_w}x{small_h}")

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
                    logging.warning(f"Depth {current_depth}: Cell too small: {cell_w}x{cell_h}, using average color")
                    avg_color = cv2.mean(small_frame)[:3]
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
                    logging.error(f"Error processing cell {i}x{j} at depth {current_depth}: {e}")
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

        logging.debug(f"Depth {current_depth}: Generated grid frame {w}x{h} with grid {grid_size}x{grid_size}")
        
        return result
    except Exception as e:
        logging.error(f"Error in generate_grid_frame at depth {current_depth}: {e}\n{traceback.format_exc()}")
        return None

@conditional_profile
def apply_grid_effect(frame, grid_size, depth):
    """Apply recursive grid effect to the frame."""
    previous_frame = None
    result = None
    
    try:
        if frame is None or frame.size == 0:
            logging.warning("apply_grid_effect received None or empty frame")
            return None
        if debug_mode or depth == 0:
            logging.debug("Depth 0: Returning original frame (debug mode or depth=0)")
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
                        logging.error(f"Depth {d}: Failed to generate grid frame (returned None)")
                        failed_depths += 1
                        if failed_depths > max_failed_depths:
                            logging.error(f"Too many failures ({failed_depths}), aborting grid effect")
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
                            logging.debug(f"Depth {d}: Cleared Core Image cache after processing")
                        except Exception as e:
                            logging.error(f"Depth {d}: Failed to clear Core Image cache: {e}")
                
                gc.collect()
            else:
                new_frame = generate_grid_frame(previous_frame, grid_size, d)
                
                if new_frame is None:
                    logging.error(f"Depth {d}: Failed to generate grid frame (returned None)")
                    failed_depths += 1
                    if failed_depths > max_failed_depths:
                        logging.error(f"Too many failures ({failed_depths}), aborting grid effect")
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
    """Retrieve streaming URL using yt-dlp."""
    try:
        if not url or url.strip() == "":
            logging.error("Empty URL provided to get_stream_url function")
            return None
            
        logging.info(f"Attempting to get stream URL for: {url}")
        result = subprocess.run(["yt-dlp", "-f", "best", "--get-url", url],
                                capture_output=True, text=True, check=True)
        stream_url = result.stdout.strip()
        if not stream_url:
            logging.error("yt-dlp returned an empty stream URL")
            return None
            
        logging.info(f"Successfully retrieved stream URL")
        return stream_url
    except subprocess.CalledProcessError as e:
        logging.error(f"yt-dlp failed with return code {e.returncode}: {e.stderr}")
        return None
    except FileNotFoundError:
        logging.error("yt-dlp not found. Please install yt-dlp.")
        return None
    except Exception as e:
        logging.error(f"Error getting stream URL: {e}\n{traceback.format_exc()}")
        return None

def get_grid_layer_breakdown(grid_size, max_depth):
    """Calculate grid layer breakdown."""
    try:
        if max_depth <= 0 or grid_size <= 0:
            return [], 0, "0"
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
    except Exception as e:
        logging.error(f"Error in get_grid_layer_breakdown: {e}\n{traceback.format_exc()}")
        return [], 0, "0"

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
        global cleanup_thread_running
        cleanup_thread_running = True
        cleanup_interval = 2.0
        memory_history = []
        history_size = 5
        
        try:
            current_process = psutil.Process(os.getpid())
        except Exception as e:
            logging.error(f"Failed to create process object for memory monitoring: {e}")
            current_process = None
        
        try:
            logging.info("Starting periodic memory cleanup thread")
            while cleanup_thread_running:
                time.sleep(cleanup_interval)
                
                try:
                    if is_apple_silicon and hardware_acceleration_available and ci_context is not None:
                        try:
                            with objc.autorelease_pool():
                                ci_context.clearCaches()
                            logging.debug("Periodic Core Image cache cleanup")
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
                            
                            if len(memory_history) > history_size:
                                memory_history.pop(0)
                            
                            growth_rate = 0
                            if len(memory_history) >= 2:
                                growth_rate = memory_history[-1] - memory_history[0]
                                
                            if memory_usage > 1500 or growth_rate > 100:
                                cleanup_interval = 1.0
                                if growth_rate > 200:
                                    logging.warning(f"Memory growing rapidly ({growth_rate:.2f} MB), performing extra cleanup")
                                    if ci_context is not None:
                                        try:
                                            ci_context.clearCaches()
                                        except Exception:
                                            pass
                                    try:
                                        import sys
                                        sys.modules.get('gc', {}).get('garbage', []).clear()
                                    except Exception:
                                        pass
                                    for _ in range(3):
                                        gc.collect()
                            elif memory_usage > 1000:
                                cleanup_interval = 1.5
                            else:
                                cleanup_interval = 2.0
                    except Exception as e:
                        logging.error(f"Error checking memory usage in cleanup thread: {e}")
                        cleanup_interval = 2.0
                        
                except Exception as e:
                    logging.error(f"Error in memory cleanup thread: {e}")
                    time.sleep(1.0)
        finally:
            cleanup_thread_running = False
            logging.info("Memory cleanup thread stopped")
    
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
            logging.info("Memory cleanup thread stopped")
        else:
            logging.debug("Memory cleanup thread not running or already stopped")
    except Exception as e:
        logging.error(f"Error stopping memory cleanup thread: {e}")
        cleanup_thread_running = False
        memory_cleanup_thread = None

def main():
    """Main function with fixed-size frame surface and scaling for display."""
    global running, grid_size, depth, debug_mode, show_info, frame_count, processed_count, displayed_count, dropped_count
    global force_hardware_acceleration, allow_software_fallback, enable_memory_tracing, ci_context, context_options
    
    screen = None
    cap = None
    frame = None
    processed_frame = None
    frame_surface = None
    font = None
    
    try:
        pygame.init()
        pygame.mixer.quit()
        pygame.display.set_caption("MultiMax Grid")
        display_info = pygame.display.Info()
        screen_width = min(1280, display_info.current_w - 100)
        screen_height = min(720, display_info.current_h - 100)
        screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
        
        if is_apple_silicon and hardware_acceleration_available:
            start_memory_cleanup_thread()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        load_dotenv()
        
        youtube_url = os.getenv('YOUTUBE_URL')
        if not youtube_url:
            logging.error("No YouTube URL found in .env file. Please set YOUTUBE_URL in .env file.")
            return
            
        parser = argparse.ArgumentParser(description='Recursive Video Grid')
        parser.add_argument('--grid-size', type=int, default=3)
        parser.add_argument('--depth', type=int, default=1)
        parser.add_argument('--youtube-url', type=str, default=youtube_url)
        parser.add_argument('--log-level', type=str, default='INFO')
        parser.add_argument('--debug', action='store_true')
        parser.add_argument('--force-hardware', action='store_true')
        parser.add_argument('--allow-software', action='store_true')
        parser.add_argument('--enable-memory-tracing', action='store_true')
        parser.add_argument('--test-hardware-accel', action='store_true')
        
        args = parser.parse_args()
        grid_size, depth = args.grid_size, args.depth
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

        stream_url = get_stream_url(args.youtube_url)
        if not stream_url:
            logging.error(f"Failed to get stream URL for {args.youtube_url}")
            logging.error("Please check that your YOUTUBE_URL in .env is valid and accessible")
            logging.error("Exiting program due to missing stream URL")
            return

        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                logging.info(f"Attempting to open video stream (attempt {retry_count+1}/{max_retries})")
                cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                if cap.isOpened():
                    logging.info("Successfully opened video stream")
                    break
                else:
                    logging.warning(f"Failed to open video stream on attempt {retry_count+1}")
                    retry_count += 1
                    time.sleep(1)
            except Exception as e:
                logging.error(f"Error opening video stream: {e}")
                retry_count += 1
                time.sleep(1)
                
        if cap is None or not cap.isOpened():
            logging.error(f"Failed to open video stream after {max_retries} attempts: {stream_url}")
            logging.error("Please check your internet connection and YouTube URL validity")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        logging.debug(f"Capture resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

        # Initialize frame_surface with video frame dimensions
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_surface = pygame.Surface((frame_width, frame_height))

        screen.fill((0, 0, 0))
        loading_text = font.render("Loading video stream...", True, (255, 255, 255))
        screen.blit(loading_text, (screen.get_width() // 2 - loading_text.get_width() // 2,
                                   screen.get_height() // 2 - loading_text.get_height() // 2))
        pygame.display.flip()

        last_frame_time = time.time()
        last_status_time = time.time()
        last_process_time = 0
        frames_since_last_log = 0
        successful_grids_since_last_log = 0
        min_process_time = float('inf')
        max_process_time = 0
        total_process_time = 0
        frame_counter = 0

        while running:
            current_time = time.time()

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                elif event.type == pygame.KEYDOWN:
                    key = pygame.key.name(event.key)
                    if key in ['up', 'down', 'd', 's'] or key in '0123456789':
                        handle_keyboard_event(key)

            ret, frame = cap.read()
            if not ret or frame is None:
                logging.warning(f"Failed to read frame from stream: {stream_url}")
                time.sleep(0.1)
                continue
            frame_count += 1

            process_start_time = time.time()
            processed_frame = apply_grid_effect(frame, grid_size, depth)
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
                
                # Periodic cache clearing every 10 frames
                if frame_count % 10 == 0 and ci_context is not None:
                    ci_context.clearCaches()
                
                if show_info:
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
                    if downsample_times:
                        avg_time = sum(downsample_times) / len(downsample_times) * 1000
                        if len(downsample_sizes) > 0:
                            last_original, last_small = downsample_sizes[-1]
                            reduction_ratio = last_original / max(1, last_small)
                            texts.append(f"Downsampling: {avg_time:.1f}ms, Reduction: {reduction_ratio:.1f}x")
                    if not debug_mode and depth > 0:
                        texts.append("")
                        texts.append("Grid Layer Breakdown:")
                        layer_info, total_screens, formatted_total = get_grid_layer_breakdown(grid_size, depth)
                        for layer in layer_info:
                            texts.append(f"  {layer}")
                        texts.append(f"  Total screens = {formatted_total}")
                    texts.append(f"Mode: {'DEBUG' if debug_mode else 'GRID'} (press 'd' to toggle)")
                    texts.append("")
                    texts.append("")
                    texts.append("Press s to show/hide this info, d to debug, up/down grid size, 1 - 0 multiply depth")
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
                        logging.info(f"Medium memory usage detected: {memory_usage:.2f} MB - performing cleanup")
                        gc.collect()
                        ci_context.clearCaches()
                        
                        for var_name in list(locals().keys()):
                            if var_name not in ['ci_context', 'context_options', 'process', 'memory_usage'] and var_name[0] != '_':
                                if var_name in locals():
                                    del locals()[var_name]
                    
                    if memory_usage > 1800:
                        logging.warning(f"High memory usage detected: {memory_usage:.2f} MB - performing context recreation")
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
                                        try:
                                            import sys
                                            if hasattr(sys.intern, 'clear'):
                                                sys.intern.clear()
                                        except Exception as e:
                                            logging.debug(f"Could not clear interned strings: {e}")
                                        gc.collect()
                                        logging.info("Recreated CIContext to free memory")
                                        memory_to_free = memory_usage - 1500
                                        if memory_to_free > 0:
                                            logging.warning(f"Attempting to free {memory_to_free:.2f} MB of memory")
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

            frame_counter += 1

            memory_usage = process.memory_info().rss / 1024 / 1024
            if memory_usage > 2000:
                logging.warning(f"High memory usage detected: {memory_usage:.2f} MB")

            if current_time - last_status_time > 5.0:
                if frames_since_last_log > 0:
                    avg_process_time = total_process_time / frames_since_last_log * 1000
                    summary_lines = [
                        f"===== PERFORMANCE SUMMARY =====",
                        f"Configuration: Grid {grid_size}x{grid_size}, Depth {depth}, Mode: {'DEBUG' if debug_mode else 'GRID'}",
                        f"Frames: Captured {frame_count}, Displayed {displayed_count}, Dropped {dropped_count}",
                        f"Processing: Min {min_process_time * 1000:.1f}ms, Max {max_process_time * 1000:.1f}ms, Avg {avg_process_time:.1f}ms",
                        f"Success Rate: {successful_grids_since_last_log}/{frames_since_last_log} frames processed successfully",
                        f"FPS: {int(clock.get_fps())}"
                    ]
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
                    if not debug_mode and depth > 0:
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
                    
                    memory_usage = process.memory_info().rss / 1024 / 1024
                    summary_lines.append(f"Current Process Memory Usage (Hardware-level): {memory_usage:.2f} MB")
                    for line in summary_lines:
                        logging.info(line)
                frames_since_last_log = 0
                successful_grids_since_last_log = 0
                min_process_time = float('inf')
                max_process_time = 0
                total_process_time = 0
                last_status_time = current_time
            
            clock.tick(30)

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