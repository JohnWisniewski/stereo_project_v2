
import numpy as np
from .debug import logger
def lr_consistency(dL, dR, tol=5):
    h,w = dL.shape
    mask = np.ones_like(dL, dtype=bool)
    for y in range(h):
        for x in range(w):
            d=int(dL[y,x])
            if d==0:
                mask[y,x]=False; continue
            xr = x-d
            if xr<0 or xr>=w:
                mask[y,x]=False; continue
            d_back=int(dR[y,xr])
            if abs(d-d_back)>tol or d_back==0:
                mask[y,x]=False

    # Log the number of pixels kept and dropped
    logger.info("LR-consistency  kept=%d  dropped=%d",
                mask.sum(), mask.size - mask.sum())


    return mask
