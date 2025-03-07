import cv2
import numpy as np
import subprocess
from pynput import keyboard
import threading
import time

# Global variables for grid size and recursion depth
n = 3  # Initial grid size (n x n)
d = 1  # Initial recursion depth

# Lock for thread-safe updates to n and d
lock = threading.Lock()

def grid_arrange(frame, n, current_d):
    """
    Recursively arrange a frame into an n x n grid for current_d levels.

    Args:
        frame: Input frame (numpy array)
        n: Grid size (e.g., 3 for 3x3)
        current_d: Current recursion depth

    Returns:
        Processed frame after grid arrangement
    """
    if current_d == 0:
        return frame

    # Create an empty output frame using NumPy for efficiency
    output = np.zeros_like(frame)
    cell_width = frame.shape[1] / n  # Width of each cell
    cell_height = frame.shape[0] / n  # Height of each cell

    # Fill the grid with resized copies of the frame
    for i in range(n):
        for j in range(n):
            x_start = int(round(j * cell_width))
            x_end = int(round((j + 1) * cell_width))
            y_start = int(round(i * cell_height))
            y_end = int(round((i + 1) * cell_height))
            cw = x_end - x_start
            ch = y_end - y_start
            if cw > 0 and ch > 0:  # Ensure cell has valid size
                # Use INTER_AREA for downscaling efficiency
                small_frame = cv2.resize(frame, (cw, ch), interpolation=cv2.INTER_AREA)
                output[y_start:y_end, x_start:x_end] = small_frame

    # Recursively apply the grid arrangement
    return grid_arrange(output, n, current_d - 1)

def on_press(key):
    """
    Handle keyboard inputs to adjust grid size (n) and recursion depth (d).
    """
    global n, d
    try:
        with lock:
            if key == keyboard.Key.up:
                n += 1  # Increase grid size
            elif key == keyboard.Key.down:
                if n > 1:
                    n -= 1  # Decrease grid size, minimum 1
            elif hasattr(key, 'char') and key.char in '1234567890':
                d = int(key.char) if key.char != '0' else 10  # Set d from 1 to 10
    except AttributeError:
        pass

def main():
    # Set up keyboard listener in a separate thread
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Replace with your YouTube livestream URL
    youtube_url = "https://www.youtube.com/watch?v=ZzWBpGwKoaI"  # Example: "https://www.youtube.com/watch?v=your_livestream_id"

    # Get the direct stream URL using yt-dlp
    try:
        result = subprocess.run(
            ["yt-dlp", "-f", "best", "--get-url", youtube_url],
            capture_output=True,
            text=True,
            check=True
        )
        stream_url = result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error fetching stream URL: {e}")
        return

    # Open the video stream with OpenCV, enabling hardware acceleration
    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Error: Could not open video stream. Ensure FFmpeg is installed and the URL is valid.")
        return

    # Set a reasonable resolution to balance quality and performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Target 15 FPS (66 ms per frame)
    target_delay = 1 / 15  # 66.67 ms in seconds

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("Stream ended or error occurred.")
            break

        # Get current values of n and d thread-safely
        with lock:
            current_n = n
            current_d = d

        # Check if the effective cell size is less than 1 pixel
        effective_factor = current_n ** current_d
        is_averaged = False
        if frame.shape[1] / effective_factor < 1 or frame.shape[0] / effective_factor < 1:
            # Average the frame to a single color for efficiency
            average_color = tuple(int(x) for x in cv2.mean(frame)[:3])
            processed_frame = np.full(frame.shape, average_color, dtype=np.uint8)
            is_averaged = True
        else:
            # Process the frame with recursive grid arrangement
            processed_frame = grid_arrange(frame, current_n, current_d)

        # Add text overlay with current settings
        total_screens = current_n ** (2 * current_d)  # Number of "screens" = n^(2d)
        text = f"n: {current_n}, d: {current_d}, total screens: {total_screens}"
        if is_averaged:
            text += " (1 frame - 1 pixel)"
        cv2.putText(
            processed_frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),  # White text
            2,
            cv2.LINE_AA  # Anti-aliased text for clarity
        )

        # Display the frame
        cv2.imshow("Recursive Grid Livestream", processed_frame)

        # Calculate elapsed time and adjust delay to maintain 15 FPS
        elapsed_time = time.time() - start_time
        remaining_delay = max(0, target_delay - elapsed_time)
        if cv2.waitKey(int(remaining_delay * 1000)) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    listener.stop()

if __name__ == "__main__":
    main()