
import argparse, os, numpy as np, cv2 as cv
from .utils import read_image, normalize_disparity
from .block_match import disparity_block
from .feature_match import match_features_R
from .validate import lr_consistency
from .fill import fill_gaps
from .debug import dump_image 
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--left', required=True); p.add_argument('--right', required=True)
    p.add_argument('--method', choices=['region','feature'], default='region')
    p.add_argument('--metric', choices=['SAD','SSD','NCC'], default='SAD')
    p.add_argument('--block', type=int, default=9)
    p.add_argument('--max_disp', type=int, default=64)
    p.add_argument('--multires', action='store_true', help='Enable multi‑resolution (region only)')
    p.add_argument('--outdir', default='results')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    left = read_image(args.left); right = read_image(args.right)

    if args.method=='region':
        disp_left = disparity_block(left, right, args.max_disp, args.block, args.block,
                                    args.metric, multires=args.multires)
        disp_right = disparity_block(right, left, args.max_disp, args.block, args.block,
                                     args.metric, multires=args.multires)
    else:
        disp_left,_ = match_features_R(left, right, args.max_disp, args.metric)
        disp_right,_ = match_features_R(right, left, args.max_disp, args.metric)
    # Save raw disparity maps for debugging
    cv.imwrite(os.path.join(args.outdir, "disp_left_raw.png"), normalize_disparity(disp_left))
    cv.imwrite(os.path.join(args.outdir, "disp_right_raw.png"), normalize_disparity(disp_right))

    mask = lr_consistency(disp_left, disp_right)
    disp_valid = np.where(mask, disp_left, 0)
    disp_filled = fill_gaps(disp_valid, mask.copy())

    base = os.path.splitext(os.path.basename(args.left))[0]
    cv.imwrite(os.path.join(args.outdir, f'{base}_{args.method}_raw.png'),
               normalize_disparity(disp_left))
    cv.imwrite(os.path.join(args.outdir, f'{base}_{args.method}_valid.png'),
               normalize_disparity(disp_valid))
    cv.imwrite(os.path.join(args.outdir, f'{base}_{args.method}_filled.png'),
               normalize_disparity(disp_filled))
    print('Done. Results in', args.outdir)

if __name__=='__main__':
    main()
