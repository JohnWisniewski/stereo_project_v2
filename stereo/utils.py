
import cv2 as cv
import numpy as np

def to_gray(img):
    if img.ndim == 2:
        return img
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def read_image(path):
    img = cv.imread(path)
    if img is None:
        raise IOError(f'Cannot read image {path}')
    return img

def normalize_disparity(disp):
    valid = disp > 0
    if not np.any(valid):
        return np.zeros_like(disp, dtype=np.uint8)
    dmin = disp[valid].min()
    dmax = disp.max()
    if dmax == dmin:
        return np.zeros_like(disp, dtype=np.uint8)
    return (255*(disp-dmin)/(dmax-dmin)).astype(np.uint8)
