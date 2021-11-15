# Copyright 2021 Alex Yu
# Eval

import torch
import svox2
import svox2.utils
import math
import argparse
import numpy as np
import os
from os import path
from util.dataset import datasets
from util.util import Timing, compute_ssim, viridis_cmap
from util import config_util

import imageio
import cv2
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('ckpt', type=str)

config_util.define_common_args(parser)

parser.add_argument('--n_eval', '-n', type=int, default=100000, help='images to evaluate (equal interval), at most evals every image')
parser.add_argument('--train', action='store_true', default=False, help='render train set')
parser.add_argument('--render_path',
                    action='store_true',
                    default=False,
                    help="Render path instead of test images (no metrics will be given)")
parser.add_argument('--timing',
                    action='store_true',
                    default=False,
                    help="Run only for timing (do not save images or use LPIPS/SSIM; "
                    "still computes PSNR to make sure images are being generated)")
parser.add_argument('--no_lpips',
                    action='store_true',
                    default=False,
                    help="Disable LPIPS (faster load)")
parser.add_argument('--no_vid',
                    action='store_true',
                    default=False,
                    help="Disable video generation")
parser.add_argument('--fps',
                    type=int,
                    default=30,
                    help="FPS of video")

# Camera adjustment
parser.add_argument('--xy_shrink',
                    type=float,
                    default=1.0,
                    help="Multiply xy poses")
parser.add_argument('--z_shift',
                    type=float,
                    default=0.0,
                    help="Amount of z to add to poses")
parser.add_argument('--near_clip',
                    type=float,
                    default=0.0,
                    help="Near clip of poses (in voxels)")

# Foreground/background only
parser.add_argument('--nofg',
                    action='store_true',
                    default=False,
                    help="Do not render foreground (if using BG model)")
parser.add_argument('--nobg',
                    action='store_true',
                    default=False,
                    help="Do not render background (if using BG model)")

# Random debugging features
parser.add_argument('--blackbg',
                    action='store_true',
                    default=False,
                    help="Force a black BG (behind BG model) color; useful for debugging 'clouds'")
parser.add_argument('--ray_len',
                    action='store_true',
                    default=False,
                    help="Render the ray lengths")

args = parser.parse_args()
config_util.maybe_merge_config_file(args, allow_invalid=True)
device = 'cuda:0'

if args.timing:
    args.no_lpips = True
    args.no_vid = True
    args.ray_len = False

if not args.no_lpips:
    import lpips
    lpips_vgg = lpips.LPIPS(net="vgg").eval().to(device)

render_dir = path.join(path.dirname(args.ckpt),
            'train_renders' if args.train else 'test_renders')
want_metrics = True
if args.render_path:
    assert not args.train
    render_dir += '_path'
    want_metrics = False

if args.z_shift != 0:
    render_dir += f'_zshift{args.z_shift}'
if args.near_clip != 0:
    render_dir += f'_nclip{args.near_clip}'
if args.xy_shrink != 1.0:
    render_dir += f'_xy_shrink{args.xy_shrink}'
if args.ray_len:
    render_dir += f'_raylen'
    want_metrics = False

dset = datasets[args.dataset_type](args.data_dir, split="test_train" if args.train else "test",
                                    **config_util.build_data_options(args))

grid = svox2.SparseGrid.load(args.ckpt, device=device)

if grid.use_background:
    if args.nobg:
        #  grid.background_cubemap.data = grid.background_cubemap.data.cuda()
        grid.background_data.data[..., -1] = 0.0
        render_dir += '_nobg'
    if args.nofg:
        grid.density_data.data[:] = 0.0
        #  grid.sh_data.data[..., 0] = 1.0 / svox2.utils.SH_C0
        #  grid.sh_data.data[..., 9] = 1.0 / svox2.utils.SH_C0
        #  grid.sh_data.data[..., 18] = 1.0 / svox2.utils.SH_C0
        render_dir += '_nofg'

config_util.setup_render_opts(grid.opt, args)

if args.blackbg:
    print('Forcing black bg')
    render_dir += '_blackbg'
    grid.opt.background_brightness = 0.0

print('Writing to', render_dir)
os.makedirs(render_dir, exist_ok=True)

# NOTE: no_grad enables the fast image-level rendering kernel for cuvol backend only
# other backends will manually generate rays per frame (slow)
with torch.no_grad():
    im_size = dset.h * dset.w
    n_images = dset.render_c2w.size(0) if args.render_path else dset.n_images
    img_eval_interval = max(n_images // args.n_eval, 1)
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_lpips = 0.0
    n_images_gen = 0
    cam = svox2.Camera(torch.tensor(0), dset.intrins.fx, dset.intrins.fy,
                       dset.intrins.cx, dset.intrins.cy,
                       dset.w, dset.h,
                       ndc_coeffs=dset.ndc_coeffs)
    c2ws = dset.render_c2w.to(device=device) if args.render_path else dset.c2w.to(device=device)
    if args.z_shift != 0.0:
        c2ws[:, 2, 3] += args.z_shift
    if args.xy_shrink != 1.0:
        c2ws[:, :2, 3] *= args.xy_shrink
    if args.near_clip != 0.0:
        grid.opt.near_clip = args.near_clip

    frames = []
    im_gt_all = dset.gt.to(device=device)

    for img_id in tqdm(range(0, n_images, img_eval_interval)):
        cam.c2w = c2ws[img_id]
        im = grid.volume_render_image(cam, use_kernel=True, return_raylen=args.ray_len)
        if args.ray_len:
            minv, meanv, maxv = im.min().item(), im.mean().item(), im.max().item()
            im = viridis_cmap(im.cpu().numpy())
            cv2.putText(im, f"{minv=:.4f} {meanv=:.4f} {maxv=:.4f}", (10, 20),
                        0, 0.5, [255, 0, 0])
            im = torch.from_numpy(im).to(device=device)
        im.clamp_(0.0, 1.0)

        if not args.render_path:
            im_gt = im_gt_all[img_id]
            mse = (im - im_gt) ** 2
            mse_num : float = mse.mean().item()
            psnr = -10.0 * math.log10(mse_num)
            avg_psnr += psnr
            if not args.timing:
                ssim = compute_ssim(im_gt, im).item()
                avg_ssim += ssim
                if not args.no_lpips:
                    lpips_i = lpips_vgg(im_gt.permute([2, 0, 1]).contiguous(),
                            im.permute([2, 0, 1]).contiguous(), normalize=True).item()
                    avg_lpips += lpips_i
                    print(img_id, 'PSNR', psnr, 'SSIM', ssim, 'LPIPS', lpips_i)
                else:
                    print(img_id, 'PSNR', psnr, 'SSIM', ssim)
        img_path = path.join(render_dir, f'{img_id:04d}.png');
        im = im.cpu().numpy()
        if not args.render_path:
            im_gt = dset.gt[img_id].numpy()
            im = np.concatenate([im_gt, im], axis=1)
        if not args.timing:
            im = (im * 255).astype(np.uint8)
            imageio.imwrite(img_path,im)
            if not args.no_vid:
                frames.append(im)
        im = None
        n_images_gen += 1
    if want_metrics:
        print('AVERAGES')

        avg_psnr /= n_images_gen
        with open(path.join(render_dir, 'psnr.txt'), 'w') as f:
            f.write(str(avg_psnr))
        print('PSNR:', avg_psnr)
        if not args.timing:
            avg_ssim /= n_images_gen
            print('SSIM:', avg_ssim)
            with open(path.join(render_dir, 'ssim.txt'), 'w') as f:
                f.write(str(avg_ssim))
            if not args.no_lpips:
                avg_lpips /= n_images_gen
                print('LPIPS:', avg_lpips)
                with open(path.join(render_dir, 'lpips.txt'), 'w') as f:
                    f.write(str(avg_lpips))
    if not args.no_vid and len(frames):
        vid_path = render_dir + '.mp4'
        imageio.mimwrite(vid_path, frames, fps=args.fps)  # pip install imageio-ffmpeg


