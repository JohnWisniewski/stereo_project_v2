
"""Region‑based block matching with optional multi‑resolution refinement."""
import numpy as np
import cv2 as cv
from .utils import to_gray
from .debug import logger, dump_image

def _cost(p1, p2, mode):
    if mode == 'SAD':
        return np.sum(np.abs(p1 - p2))
    elif mode == 'SSD':
        diff = p1 - p2
        return np.sum(diff*diff)
    elif mode == 'NCC':
        p1 = p1.astype(np.float32); p2 = p2.astype(np.float32)
        p1 -= p1.mean(); p2 -= p2.mean()
        denom = (np.linalg.norm(p1)*np.linalg.norm(p2)+1e-6)
        return 1 - (p1*p2).sum()/denom
    else:
        raise ValueError(mode)

def disparity_block_single(left_g, right_g, max_disp, tpl_w, tpl_h, mode,dbg_name="lvl0"):
    h, w = left_g.shape
    pad_x, pad_y = tpl_w//2, tpl_h//2
    disp = np.zeros((h, w), dtype=np.float32)
    left_p = cv.copyMakeBorder(left_g, pad_y, pad_y, pad_x, pad_x, cv.BORDER_REFLECT)
    right_p = cv.copyMakeBorder(right_g, pad_y, pad_y, pad_x+max_disp, pad_x, cv.BORDER_REFLECT)
    logger.info("single-res BM  %s  %sx%s  max_disp=%d  tpl=%dx%d",
                dbg_name, w, h, max_disp, tpl_w, tpl_h)    
    for y in range(pad_y, h+pad_y):
        for x in range(pad_x, w+pad_x):
            tpl = left_p[y-pad_y:y+pad_y+1, x-pad_x:x+pad_x+1]
            best_c = float('inf')
            best_d = 0
            for d in range(max_disp+1):
                xr = x - d
                if xr - pad_x < 0:
                    break
                cand = right_p[y-pad_y:y+pad_y+1, xr-pad_x:xr+pad_x+1]
                c = _cost(tpl, cand, mode)
                if c < best_c:
                    best_c, best_d = c, d
            disp[y-pad_y, x-pad_x] = best_d
            # Log debug information for every 50th pixel
            if (x % 50 == 0) and (y % 50 == 0):
                logger.debug("(%d,%d) best_d=%d best_c=%.3f", x, y, best_d, best_c)

    # Dump the disparity image for debugging purposes
    if dbg_name:
        dump_image(f"disp_{dbg_name}_raw", disp, "debug")

    return disp

def disparity_block_multires(left, right, max_disp, tpl_w, tpl_h, mode='SAD', levels=3):
    """Multi‑resolution block matching.
    Coarse disparity estimated at lower resolution then refined ±1 pixel at finer scales.
    """
    left_pyr = [to_gray(left)]
    right_pyr = [to_gray(right)]
    for _ in range(1, levels):
        left_pyr.insert(0, cv.pyrDown(left_pyr[0]))
        right_pyr.insert(0, cv.pyrDown(right_pyr[0]))

    scale = 2**(levels-1)
    disp_coarse = disparity_block_single(left_pyr[0], right_pyr[0],
                                         max_disp//scale, tpl_w, tpl_h, mode)
    # iterative refinement

    
    for lvl in range(1, levels):
        # upscale disparity
        disp_up = cv.resize(disp_coarse, (left_pyr[lvl].shape[1], left_pyr[lvl].shape[0]),
                            interpolation=cv.INTER_NEAREST)*2
        # refine each pixel by searching ±1 around prediction
        h, w = left_pyr[lvl].shape
        pad_x, pad_y = tpl_w//2, tpl_h//2
        left_g = left_pyr[lvl]; right_g = right_pyr[lvl]
        left_pad = cv.copyMakeBorder(left_g, pad_y, pad_y, pad_x, pad_x, cv.BORDER_REFLECT)
        right_pad = cv.copyMakeBorder(right_g, pad_y, pad_y, pad_x+max_disp, pad_x, cv.BORDER_REFLECT)
        disp_ref = np.zeros_like(left_g, dtype=np.float32)
        for y in range(pad_y, h+pad_y):
            for x in range(pad_x, w+pad_x):
                pred = int(disp_up[y-pad_y, x-pad_x])
                tpl = left_pad[y-pad_y:y+pad_y+1, x-pad_x:x+pad_x+1]
                best_c = float('inf')
                best_d = pred
                for d in range(max(0, pred-1), min(max_disp, pred+1)+1):
                    xr = x - d
                    if xr - pad_x < 0:
                        continue
                    cand = right_pad[y-pad_y:y+pad_y+1, xr-pad_x:xr+pad_x+1]
                    c = _cost(tpl, cand, mode)
                    if c < best_c:
                        best_c, best_d = c, d
                disp_ref[y-pad_y, x-pad_x] = best_d
        disp_coarse = disp_ref  # becomes input for next level
    return disp_coarse

def disparity_block(left, right, max_disp, tpl_w, tpl_h, mode='SAD', multires=False, levels=3):
    if multires:
        return disparity_block_multires(left, right, max_disp, tpl_w, tpl_h, mode, levels)
    return disparity_block_single(to_gray(left), to_gray(right), max_disp, tpl_w, tpl_h, mode)

