import cv2
import numpy as np

def create_fractal_grid(live_frame, prev_output, grid_size, source_position=1):
    """
    Create an NxN grid for the infinite fractal effect:
    - Source position 1: Live frame in top-left cell (default)
    - Source position 2: Live frame in center cell (or nearest to center)
    - Source position 3: Live frame in top-right cell
    
    Args:
        live_frame (np.ndarray): The current frame from the YouTube livestream.
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
    
    grid_frame = np.zeros_like(live_frame)
    
    # Determine the position of the source frame based on source_position
    source_i, source_j = 0, 0  # Default: top-left (position 1)
    
    if source_position == 2:  # Center
        # Special case for 2x2 grid with position 2: place in top-right
        if grid_size == 2:
            source_i = 0
            source_j = 1  # top-right in a 2x2 grid
        # For position 2, we should only have odd grid sizes from the handler
        # But let's ensure it anyway - for odd grid sizes, there's a clear center
        elif grid_size % 2 == 1:
            source_i = source_j = grid_size // 2
        else:
            # This case should not happen with our updated handlers, but just in case
            # Force to center-ish by adjusting the grid size calculation
            center = grid_size / 2 - 0.5
            source_i = int(center)
            source_j = int(center)
    elif source_position == 3:  # Top-right
        source_i = 0
        source_j = grid_size - 1
    
    for i in range(grid_size):
        for j in range(grid_size):
            y_start = i * cell_h
            y_end = (i + 1) * cell_h if i < grid_size - 1 else h
            x_start = j * cell_w
            x_end = (j + 1) * cell_w if j < grid_size - 1 else w
            
            if i == source_i and j == source_j:
                # Source cell: Live YouTube frame
                grid_frame[y_start:y_end, x_start:x_end] = cv2.resize(live_frame, (x_end - x_start, y_end - y_start))
            else:
                # Other cells: Previous output
                grid_frame[y_start:y_end, x_start:x_end] = cv2.resize(prev_output, (x_end - x_start, y_end - y_start))
    
    return grid_frame