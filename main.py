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

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def configure_logging(level_name):
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.getLogger().setLevel(level)

# Check for Apple Silicon
is_apple_silicon = platform.system() == 'Darwin' and platform.machine().startswith('arm')
hardware_acceleration_available = False

if is_apple_silicon:
    try:
        import objc
        from Foundation import NSData, NSMutableData
        from Quartz import CIContext, CIImage, CIFilter, kCIFormatRGBA8, kCIContextUseSoftwareRenderer, CGRect, CGPoint, CGSize
        from CoreFoundation import CFDataCreate
        # Create a hardware-accelerated context for Apple Silicon
        ci_context = CIContext.contextWithOptions_({kCIContextUseSoftwareRenderer: False})
        hardware_acceleration_available = True
        logging.info("Hardware acceleration enabled on Apple Silicon")
    except ImportError as e:
        logging.warning(f"Hardware acceleration unavailable: {e}")
else:
    logging.info("Not running on Apple Silicon, using software processing")

# Global variables
grid_size = 3
depth = 1
running = True
debug_mode = True  # Start in debug mode to ensure functionality

# Stats
frame_count = 0
processed_count = 0
displayed_count = 0
dropped_count = 0

# For keyboard control
def handle_keyboard_event(key_name):
    global grid_size, depth, debug_mode
    
    if key_name == 'up':
        grid_size += 1
        logging.info(f"Grid size increased to {grid_size}")
    elif key_name == 'down' and grid_size > 1:
        grid_size -= 1
        logging.info(f"Grid size decreased to {grid_size}")
    elif key_name in '1234567890':
        depth = int(key_name) if key_name != '0' else 10
        logging.info(f"Recursion depth set to {depth}")
    elif key_name == 'd':
        debug_mode = not debug_mode
        logging.info(f"Debug mode {'enabled' if debug_mode else 'disabled'}")

# Signal handler
def signal_handler(sig, frame):
    global running
    print("\nShutting down...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

# Grid effect functions with Apple Silicon hardware acceleration
def cv_to_ci_image(cv_img):
    """Convert OpenCV image to CoreImage image using hardware acceleration"""
    cv_img_rgba = cv2.cvtColor(cv_img, cv2.COLOR_BGR2BGRA)
    height, width = cv_img_rgba.shape[:2]
    data = cv_img_rgba.tobytes()
    data_provider = CFDataCreate(None, data, len(data))
    ci_img = CIImage.imageWithBitmapData_bytesPerRow_size_format_colorSpace_(
        data_provider, width * 4, CGSize(width, height), kCIFormatRGBA8, None)
    return ci_img

def ci_to_cv_image(ci_img, width, height):
    """Convert CoreImage image to OpenCV image using hardware acceleration"""
    output_data = NSMutableData.dataWithLength_(height * width * 4)
    ci_context.render_toBitmap_rowBytes_bounds_format_colorSpace_(
        ci_img, output_data.mutableBytes(), width * 4, CGRect(CGPoint(0, 0), CGSize(width, height)),
        kCIFormatRGBA8, None)
    buffer = np.frombuffer(output_data, dtype=np.uint8).reshape(height, width, 4)
    cv_img = cv2.cvtColor(buffer, cv2.COLOR_BGRA2BGR)
    return cv_img

def hardware_grid_arrange(frame, n, current_d):
    """Apply grid arrangement using Apple Silicon hardware acceleration"""
    if current_d == 0 or frame.shape[0] // n < 4 or frame.shape[1] // n < 4:
        return frame
    try:
        # Convert to CoreImage for hardware acceleration
        ci_frame = cv_to_ci_image(frame)
        h, w = frame.shape[:2]
        result = np.zeros((h, w, 3), dtype=np.uint8)
        cell_h, cell_w = h // n, w // n
        target_ratio = 16 / 9
        frame_ratio = w / h

        for i in range(n):
            for j in range(n):
                y, x = i * cell_h, j * cell_w
                # Use hardware-accelerated CIFilter for scaling
                scale_filter = CIFilter.filterWithName_("CILanczosScaleTransform")
                scale_filter.setValue_forKey_(ci_frame, "inputImage")

                if frame_ratio == target_ratio:
                    scale = cell_w / w
                    scale_filter.setValue_forKey_(scale, "inputScale")
                    scale_filter.setValue_forKey_(1.0, "inputAspectRatio")
                    cell_ci = scale_filter.valueForKey_("outputImage")
                elif frame_ratio > target_ratio:
                    scale = cell_h / h
                    scaled_width = w * scale
                    scale_filter.setValue_forKey_(scale, "inputScale")
                    scale_filter.setValue_forKey_(1.0, "inputAspectRatio")
                    scaled_ci = scale_filter.valueForKey_("outputImage")
                    crop_x = (scaled_width - cell_w) / 2.0
                    # Use hardware-accelerated cropping
                    crop_filter = CIFilter.filterWithName_("CICrop")
                    crop_filter.setValue_forKey_(scaled_ci, "inputImage")
                    crop_filter.setValue_forKey_([crop_x, 0, cell_w, cell_h], "inputRectangle")
                    cell_ci = crop_filter.valueForKey_("outputImage")
                else:
                    scale = cell_w / w
                    scaled_height = h * scale
                    scale_filter.setValue_forKey_(scale, "inputScale")
                    scale_filter.setValue_forKey_(1.0, "inputAspectRatio")
                    scaled_ci = scale_filter.valueForKey_("outputImage")
                    crop_y = (scaled_height - cell_h) / 2.0
                    # Use hardware-accelerated cropping
                    crop_filter = CIFilter.filterWithName_("CICrop")
                    crop_filter.setValue_forKey_(scaled_ci, "inputImage")
                    crop_filter.setValue_forKey_([0, crop_y, cell_w, cell_h], "inputRectangle")
                    cell_ci = crop_filter.valueForKey_("outputImage")

                # Convert back to CV image
                cell = ci_to_cv_image(cell_ci, cell_w, cell_h)
                
                # Recursively apply grid effect if depth > 1
                if current_d > 1:
                    cell = hardware_grid_arrange(cell, n, current_d - 1)
                    
                # Place the cell in the result image
                result[y:y+cell_h, x:x+cell_w] = cell
        return result
    except Exception as e:
        logging.debug(f"Hardware grid failed: {e}, falling back to software implementation")
        return sw_grid_arrange(frame, n, current_d)

# Software grid arrangement (fallback)
def sw_grid_arrange(frame, n, current_d):
    """Software implementation of grid arrangement (fallback for non-Apple Silicon)"""
    if debug_mode:
        # For debugging - just pass through the frame without grid effect
        return frame.copy()
            
    if current_d == 0:
        return frame
        
    h, w = frame.shape[:2]
    cell_h, cell_w = h // n, w // n
    if cell_h < 1 or cell_w < 1:
        logging.warning(f"Cell size too small: {cell_h}x{cell_w}, filling with average color")
        return np.full((h, w, 3), cv2.mean(frame)[:3], dtype=np.uint8)
    
    result = np.zeros((h, w, 3), dtype=np.uint8)
    target_ratio = 16 / 9
    frame_ratio = w / h
    for i in range(n):
        for j in range(n):
            y, x = i * cell_h, j * cell_w
            if frame_ratio > target_ratio:
                resize_h = max(cell_h, 1)
                resize_w = max(int(resize_h * frame_ratio), 1)
                cell = cv2.resize(frame, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
                start_x = max(0, (resize_w - cell_w) // 2)
                end_x = min(resize_w, start_x + cell_w)
                cropped_cell = cell[:, start_x:end_x]
                if cropped_cell.shape[1] != cell_w or cropped_cell.shape[0] != cell_h:
                    cropped_cell = cv2.resize(cropped_cell, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
                cell = cropped_cell
            else:
                resize_w = max(cell_w, 1)
                resize_h = max(int(resize_w / frame_ratio), 1)
                cell = cv2.resize(frame, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
                start_y = max(0, (resize_h - cell_h) // 2)
                end_y = min(resize_h, start_y + cell_h)
                cropped_cell = cell[start_y:end_y, :]
                if cropped_cell.shape[0] != cell_h or cropped_cell.shape[1] != cell_w:
                    cropped_cell = cv2.resize(cropped_cell, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
                cell = cropped_cell
                
            # Recursively apply grid effect if depth > 1
            if current_d > 1:
                cell = sw_grid_arrange(cell, n, current_d - 1)
                
            result[y:y+cell_h, x:x+cell_w] = cell
    return result

# Choose the appropriate grid function based on hardware
grid_arrange = sw_grid_arrange
if is_apple_silicon and hardware_acceleration_available:
    grid_arrange = hardware_grid_arrange
    logging.info("Using hardware-accelerated grid effect")
else:
    logging.info("Using software grid effect implementation")

def apply_grid_effect(frame, n, d):
    """Apply the grid effect to a frame, using hardware acceleration if available"""
    if frame is None:
        return None
    
    if debug_mode:
        # For debugging: just pass through the frame
        return frame.copy()
    
    try:
        if d == 0:
            return frame.copy()
            
        # Apply actual grid effect
        result = grid_arrange(frame.copy(), n, d)
        
        # Clean up memory
        del frame
        return result
    except Exception as e:
        logging.error(f"Grid effect error: {e}")
        if frame is not None:
            result = frame.copy()
            del frame
            return result
        return None

def get_stream_url(url):
    """Get the real streaming URL from YouTube"""
    try:
        result = subprocess.run(["yt-dlp", "-f", "best", "--get-url", url],
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception as e:
        logging.error(f"Error getting YouTube stream URL: {e}")
        return None

def main():
    global running, grid_size, depth, debug_mode, frame_count, processed_count, displayed_count, dropped_count
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Recursive Video Grid')
    parser.add_argument('--grid-size', type=int, default=3)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--youtube-url', type=str, default=os.getenv('YOUTUBE_URL'))
    parser.add_argument('--log-level', type=str, default='INFO')
    parser.add_argument('--debug', action='store_true', help='Start in debug/passthrough mode')
    args = parser.parse_args()
    
    # Initialize parameters
    grid_size, depth = args.grid_size, args.depth
    configure_logging(args.log_level)
    
    # Set debug mode if requested or by default
    debug_mode = True  # Start in debug mode
    logging.info("Starting in debug mode for reliable playback (press 'd' to toggle grid effect)")

    # Initialize Pygame
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
    pygame.display.set_caption("Recursive Grid Livestream")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24)
    
    # Get the stream URL
    logging.info(f"Getting YouTube stream URL: {args.youtube_url}")
    stream_url = get_stream_url(args.youtube_url)
    if stream_url is None:
        logging.error("Failed to get stream URL")
        return
    
    # Initialize video capture
    logging.info("Initializing video capture")
    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logging.error("Failed to open video stream")
        return
        
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    logging.info(f"Actual capture resolution: {actual_w}x{actual_h}")
    
    # Show loading screen
    loading_surface = pygame.Surface(screen.get_size())
    loading_surface.fill((0, 0, 0))
    loading_text = font.render("Loading video stream...", True, (255, 255, 255))
    loading_surface.blit(loading_text, (screen.get_width() // 2 - loading_text.get_width() // 2, 
                                      screen.get_height() // 2 - loading_text.get_height() // 2))
    screen.blit(loading_surface, (0, 0))
    pygame.display.flip()
    
    # Variables for tracking
    last_frame_time = time.time()
    last_gc_time = time.time()
    last_status_time = time.time()
    last_effect_change_time = time.time()
    
    # Main loop
    logging.info("Starting main loop")
    prev_debug_mode = debug_mode
    prev_grid_size = grid_size
    prev_depth = depth
    
    while running:
        current_time = time.time()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_UP:
                    handle_keyboard_event('up')
                elif event.key == pygame.K_DOWN:
                    handle_keyboard_event('down')
                elif event.key == pygame.K_d:
                    handle_keyboard_event('d')
                elif event.unicode in '0123456789':
                    handle_keyboard_event(event.unicode)
        
        # Log when effect settings change
        if (prev_debug_mode != debug_mode or prev_grid_size != grid_size or 
            prev_depth != depth) and current_time - last_effect_change_time > 0.5:
            mode_text = "DEBUG/PASSTHROUGH" if debug_mode else "NORMAL/GRID"
            logging.info(f"Effect settings changed - Mode: {mode_text}, Grid: {grid_size}x{grid_size}, Depth: {depth}")
            prev_debug_mode = debug_mode
            prev_grid_size = grid_size
            prev_depth = depth
            last_effect_change_time = current_time
        
        # Capture new frame
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to read frame")
            time.sleep(0.1)
            continue
            
        # Update counters
        frame_count += 1
        
        # Apply grid effect if not in debug mode
        try:
            if not debug_mode:
                # Apply hardware-accelerated grid effect
                processed_frame = apply_grid_effect(frame.copy(), grid_size, depth)
                processed_count += 1
            else:
                # Just pass through the frame in debug mode
                processed_frame = frame.copy()
        except Exception as e:
            logging.error(f"Error applying effect: {e}")
            processed_frame = frame.copy()
            dropped_count += 1
        
        # Convert to pygame surface and display
        try:
            # Convert from BGR (OpenCV) to RGB (Pygame)
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Create surface and scale to screen size
            surface = pygame.surfarray.make_surface(rgb_frame.swapaxes(0, 1))
            screen.blit(pygame.transform.scale(surface, screen.get_size()), (0, 0))
            displayed_count += 1
            
            # Display stats overlay
            info_surface = pygame.Surface((screen.get_width(), 150), pygame.SRCALPHA)
            info_surface.fill((0, 0, 0, 128))
            
            # Prepare stats text
            grid_text = f"Grid: {grid_size}x{grid_size}, Depth: {depth}"
            frames_text = f"Captured: {frame_count}, Displayed: {displayed_count}"
            process_text = f"Processed: {processed_count}, Dropped: {dropped_count}"
            fps_text = f"FPS: {int(clock.get_fps())}"
            hw_text = f"Hardware: {'Enabled' if hardware_acceleration_available else 'Disabled'}"
            debug_text = f"Mode: {'DEBUG/PASSTHROUGH' if debug_mode else 'NORMAL/GRID'} (press 'd' to toggle)"
            
            # Draw stats text
            texts = [grid_text, frames_text, process_text, fps_text, hw_text, debug_text]
            y_pos = 10
            for text in texts:
                text_surface = font.render(text, True, (255, 255, 255))
                info_surface.blit(text_surface, (10, y_pos))
                y_pos += 20
                
            screen.blit(info_surface, (0, 0))
            pygame.display.flip()
            
            # Clean up memory
            del processed_frame
            
        except Exception as e:
            logging.error(f"Error displaying frame: {e}")
            dropped_count += 1
        
        # Garbage collection periodically
        if current_time - last_gc_time > 10.0:
            gc.collect()
            last_gc_time = current_time
            
        # Log status periodically
        if current_time - last_status_time > 5.0:
            mode_text = "DEBUG/PASSTHROUGH" if debug_mode else "NORMAL/GRID"
            logging.info(f"Status: {frame_count} captured, {displayed_count} displayed, FPS: {int(clock.get_fps())}, Mode: {mode_text}")
            last_status_time = current_time
            
        # Limit frame rate
        clock.tick(30)
    
    # Cleanup
    cap.release()
    pygame.quit()
    logging.info("Application shutdown complete")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Main crashed: {e}\n{traceback.format_exc()}")