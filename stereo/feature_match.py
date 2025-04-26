
"""Feature‑based matching using Harris response (R) as descriptor."""
import numpy as np
import cv2 as cv
from .utils import to_gray
from .debug import logger

def harris_response(gray, blockSize=3, ksize=5, k=0.0):
    gray = gray.astype(np.float32)
    return cv.cornerHarris(gray, blockSize, ksize, k)

def detect_corners(R, thresh_ratio=0.001):
    R_dil = cv.dilate(R, None)
    pts = np.argwhere((R == R_dil) & (R > thresh_ratio*R.max()))
    # return list of (y,x,Rval)
    return [(int(y), int(x), float(R[y,x])) for y,x in pts]

def match_features_R(left, right, max_disp, metric='SSD'):
    left_g = to_gray(left); right_g = to_gray(right)
    R_left = harris_response(left_g); R_right = harris_response(right_g)
    corners_L = detect_corners(R_left)
    corners_R = detect_corners(R_right)

    # Log the number of detected corners
    logger.info("Harris corners L=%d  R=%d", len(corners_L), len(corners_R))

    # Debug: Visualize detected corners
    corner_img_left = left.copy()
    for y, x, _ in corners_L:
        cv.circle(corner_img_left, (x, y), 2, (0, 255, 0), -1)
    cv.imwrite("debug_corners_left.png", corner_img_left)

    corner_img_right = right.copy()
    for y, x, _ in corners_R:
        cv.circle(corner_img_right, (x, y), 2, (0, 255, 0), -1)
    cv.imwrite("debug_corners_right.png", corner_img_right)

    # Build row dict for right corners
    row_dict = {}
    for y, x, Rv in corners_R:
        row_dict.setdefault(y, []).append((x, Rv))

    disp_map = np.zeros_like(left_g, dtype=np.float32)
    matches = []
    for y, x, RvL in corners_L:
        best_score = float('inf') if metric != 'NCC' else -1
        best_xr = None
        cand_list = row_dict.get(y, [])
        for xr, RvR in cand_list:
            if abs(x - xr) > max_disp:
                continue
            if metric == 'SAD':
                score = abs(RvL - RvR)
                better = score < best_score
            elif metric == 'SSD':
                score = (RvL - RvR) ** 2
                better = score < best_score
            elif metric == 'NCC':
                score = (RvL * RvR) / (abs(RvL * RvR) + 1e-6)  # Maximize NCC
                better = score > best_score
            else:
                raise ValueError(metric)
            if better:
                best_score = score
                best_xr = xr
        if best_xr is not None:
            d = x - best_xr
            disp_map[y, x] = d
            matches.append(((x, y), d))

    # Debug: Log matches
    logger.info("Number of matches: %d", len(matches))
    for match in matches[:10]:  # Log first 10 matches for brevity
        logger.debug("Match: %s", match)

    return disp_map, matches