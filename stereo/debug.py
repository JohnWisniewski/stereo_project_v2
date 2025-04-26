# stereo/debug.py
import logging, os, cv2 as cv, numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("stereo")

def dump_image(tag: str, arr: np.ndarray, outdir: str):
    """Helper to write uint8 or float arrays for visual inspection."""
    os.makedirs(outdir, exist_ok=True)
    if arr.dtype != np.uint8:
        a = arr.copy()
        a -= a.min()
        a = (255 * a / (a.max() + 1e-6)).astype(np.uint8)
    else:
        a = arr
    cv.imwrite(os.path.join(outdir, f"{tag}.png"), a)
    logger.debug("wrote %s.png  min=%.2f max=%.2f", tag, arr.min(), arr.max())
