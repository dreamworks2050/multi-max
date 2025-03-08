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

# Global variables
grid_size = 3
depth = 1
running = True
debug_mode = True  # Start in debug mode

# Stats
frame_count = 0
processed_count = 0
displayed_count = 0
dropped_count = 0

# Keyboard controls
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

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    global running
    print("\nShutting down...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

# Hardware-accelerated frame conversion functions
def cv_to_ci_image(cv_img):
    cv_img_rgba = cv2.cvtColor(cv_img, cv2.COLOR_BGR2BGRA)
    height, width = cv_img_rgba.shape[:2]
    data = cv_img_rgba.tobytes()
    data_provider = CFDataCreate(None, data, len(data))
    ci_img = CIImage.imageWithBitmapData_bytesPerRow_size_format_colorSpace_(
        data_provider, width * 4, Quartz.CGSizeMake(width, height), kCIFormatRGBA8, None)
    return ci_img

def ci_to_cv_image(ci_img, width, height):
    output_data = NSMutableData.dataWithLength_(height * width * 4)
    ci_context.render_toBitmap_rowBytes_bounds_format_colorSpace_(
        ci_img, output_data.mutableBytes(), width * 4, Quartz.CGRectMake(0, 0, width, height),
        kCIFormatRGBA8, None)
    buffer = np.frombuffer(output_data, dtype=np.uint8).reshape(height, width, 4)
    cv_img = cv2.cvtColor(buffer, cv2.COLOR_BGRA2BGR)
    return cv_img

# Generate a new grid frame from the entire previous frame
def generate_grid_frame(previous_frame, grid_size, current_depth):
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

    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate cell size, distributing remainder pixels
            cell_h = base_cell_h + (1 if i < remainder_h else 0)
            cell_w = base_cell_w + (1 if j < remainder_w else 0)

            if cell_h < 1 or cell_w < 1:
                # If cell is too small, fill with average color
                avg_color = cv2.mean(previous_frame)[:3]
                result[i * base_cell_h:(i + 1) * base_cell_h, j * base_cell_w:(j + 1) * base_cell_w] = avg_color
                continue

            # Calculate the aspect ratio of the cell
            cell_aspect = cell_w / cell_h

            # Resize frame to maintain 16:9 aspect ratio, then crop to fit cell
            if frame_aspect > cell_aspect:
                # Frame is wider than cell, scale based on height
                scaled_h = cell_h
                scaled_w = int(scaled_h * frame_aspect)
                crop_x = (scaled_w - cell_w) // 2
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
            else:
                # Frame is taller than cell, scale based on width
                scaled_w = cell_w
                scaled_h = int(scaled_w / frame_aspect)
                crop_y = (scaled_h - cell_h) // 2
                if hardware_acceleration_available:
                    ci_img = cv_to_ci_image(previous_frame)
                    scale_filter = CIFilter.filterWithName_("CILanczosScaleTransform")
                    scale_filter.setValue_forKey_(ci_img, "inputImage")
                    scale_filter.setValue_forKey_(scaled_w / w, "inputScale")
                    scale_filter.setValue_forKey_(1.0, "inputAspectRatio")
                    scaled_ci = scale_filter.valueForKey_("outputImage")
                    crop_filter = CIFilter.filterWithName_("CICrop")
                    crop_filter.setValue_forKey_(scaled_ci, "inputImage")
                    rect_vector = CIVector.vectorWithX_Y_Z_W_(0, crop_y, cell_w, cell_h)
                    crop_filter.setValue_forKey_(rect_vector, "inputRectangle")
                    cell_ci = crop_filter.valueForKey_("outputImage")
                    cell = ci_to_cv_image(cell_ci, cell_w, cell_h)
                    # Clean up Core Image objects
                    del ci_img, scaled_ci, cell_ci
                else:
                    scaled = cv2.resize(previous_frame, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
                    cell = scaled[crop_y:crop_y + cell_h, :]
                    del scaled  # Clean up scaled frame

            # Place the cell in the result frame
            y_start = i * base_cell_h + min(i, remainder_h)
            y_end = y_start + cell_h
            x_start = j * base_cell_w + min(j, remainder_w)
            x_end = x_start + cell_w
            result[y_start:y_end, x_start:x_end] = cell

            # Clean up cell frame
            del cell
            gc.collect()  # Force garbage collection after each cell

    logging.info(f"Depth {current_depth}: Generated grid frame {w}x{h} with grid {grid_size}x{grid_size}")
    return result

# Apply grid effect across multiple frames with memory management
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

# Fetch YouTube stream URL
def get_stream_url(url):
    try:
        result = subprocess.run(["yt-dlp", "-f", "best", "--get-url", url],
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception as e:
        logging.error(f"Error getting YouTube stream URL: {e}")
        return None

# Main application loop
def main():
    global running, grid_size, depth, debug_mode, frame_count, processed_count, displayed_count, dropped_count
    parser = argparse.ArgumentParser(description='Recursive Video Grid')
    parser.add_argument('--grid-size', type=int, default=3)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--youtube-url', type=str, default=os.getenv('YOUTUBE_URL'))
    parser.add_argument('--log-level', type=str, default='INFO')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    grid_size, depth = args.grid_size, args.depth
    configure_logging(args.log_level)
    debug_mode = args.debug
    logging.info(f"Starting with debug={debug_mode} (toggle with 'd')")

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
    logging.info(f"Capture resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    # Loading screen
    screen.fill((0, 0, 0))
    loading_text = font.render("Loading video stream...", True, (255, 255, 255))
    screen.blit(loading_text, (screen.get_width() // 2 - loading_text.get_width() // 2,
                               screen.get_height() // 2 - loading_text.get_height() // 2))
    pygame.display.flip()

    last_frame_time = time.time()
    last_gc_time = time.time()
    last_status_time = time.time()
    last_process_time = 0

    while running:
        current_time = time.time()
        loop_start_time = current_time

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN:
                key = pygame.key.name(event.key)
                if key in ['up', 'down', 'd'] or key in '0123456789':
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
        except Exception as e:
            logging.error(f"Error applying effect: {e}")
            processed_frame = frame.copy()
            dropped_count += 1

        # Display frame
        try:
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            surface = pygame.surfarray.make_surface(rgb_frame.swapaxes(0, 1))
            screen.blit(pygame.transform.scale(surface, screen.get_size()), (0, 0))
            displayed_count += 1

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
            pygame.display.flip()
        except Exception as e:
            logging.error(f"Display error: {e}")
            dropped_count += 1

        # Periodic garbage collection and status logging
        if current_time - last_gc_time > 10.0:
            gc.collect()
            last_gc_time = current_time
        if current_time - last_status_time > 5.0:
            logging.info(f"Status: {frame_count} captured, {displayed_count} displayed, "
                         f"FPS: {int(clock.get_fps())}, Process: {last_process_time * 1000:.1f}ms")
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