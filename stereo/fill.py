import numpy as np
from .debug import logger 

def fill_gaps(disp, valid, max_iter=5, max_radius=3):
    """
    Fill gaps in the disparity map using neighboring valid pixels.

    Args:
        disp (np.ndarray): Disparity map with gaps.
        valid (np.ndarray): Boolean mask indicating valid pixels.
        max_iter (int): Maximum number of iterations for gap filling.
        max_radius (int): Maximum radius to search for valid neighbors.

    Returns:
        np.ndarray: Disparity map with gaps filled.
    """
    filled = disp.copy()
    h, w = disp.shape

    for iter_num in range(max_iter):
        changed = False
        valid_prev = valid.copy()

        for y in range(h):
            for x in range(w):
                if valid[y, x]:
                    continue  # Skip already valid pixels

                # Search for valid neighbors within a growing radius
                for r in range(1, max_radius + 1):
                    ys, ye = max(0, y - r), min(h - 1, y + r)
                    xs, xe = max(0, x - r), min(w - 1, x + r)
                    vals = filled[ys:ye + 1, xs:xe + 1][valid[ys:ye + 1, xs:xe + 1]]

                    if vals.size >= 5 or (r == max_radius and vals.size > 0):
                        filled[y, x] = vals.mean()
                        valid[y, x] = True
                        changed = True
                        break  # Stop searching once a valid neighbor is found

        # Log the number of gaps filled in this iteration
        gaps_filled = (~valid_prev & valid).sum()
        logger.info("gap-filling iter=%d filled=%d remaining=%d", 
                    iter_num, gaps_filled, (~valid).sum())

        if not changed:
            logger.info("No more gaps filled. Stopping early at iteration %d.", iter_num)
            break

    return filled