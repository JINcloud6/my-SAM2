import argparse
import os
import time
from contextlib import nullcontext

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .data_manager import VolumeManager
from .sam2_baseline.predict_utils import map_local_point


COLORS = [
    (1.0, 0.1, 0.1),
    (0.1, 1.0, 0.1),
    (0.1, 0.4, 1.0),
]


def get_args():
    parser = argparse.ArgumentParser(
        description="Visualize 3 SAM2 candidate masks on nearby slices around a seed."
    )
    parser.add_argument("--sam2_checkpoint", required=True, help="Path to SAM2 checkpoint")
    parser.add_argument("--sam2_model_cfg", required=True, help="Path to SAM2 model config")
    parser.add_argument("--volume_path", required=True, help="Path to .h5 or .nii/.nii.gz file")
    parser.add_argument("--dataset_key", default="main", help="Key name in h5 file")
    parser.add_argument("--seed", required=True, help="Seed in z,y,x format, e.g. 120,300,280")
    parser.add_argument(
        "--axis",
        type=int,
        default=-1,
        choices=[-1, 0, 1, 2],
        help="Tracking axis. -1 means auto-select on seed slice by min-area mask.",
    )
    parser.add_argument(
        "--half_window",
        type=int,
        default=8,
        help="Number of slices above and below seed index to visualize.",
    )
    parser.add_argument("--output_dir", default="./candidate_mask_viz", help="Output directory")
    parser.add_argument("--alpha", type=float, default=0.35, help="Mask overlay alpha")
    parser.add_argument(
        "--crop_size",
        type=int,
        default=384,
        help="Crop size around seed on each slice for faster visualization; <=0 means full slice.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def parse_seed(seed_text):
    parts = [int(p.strip()) for p in seed_text.split(",")]
    if len(parts) != 3:
        raise ValueError("--seed must be z,y,x")
    return tuple(parts)


def _clamp_crop(center, size, limit):
    half = size // 2
    start = max(0, center - half)
    end = min(limit, center + half)
    return start, end


def get_slice_rgb_and_box(vol_man, axis, idx, seed, crop_size):
    if axis == 0:
        y0, y1 = (0, vol_man.shape[1]) if crop_size <= 0 else _clamp_crop(seed[1], crop_size, vol_man.shape[1])
        x0, x1 = (0, vol_man.shape[2]) if crop_size <= 0 else _clamp_crop(seed[2], crop_size, vol_man.shape[2])
        img = vol_man.vol[idx, y0:y1, x0:x1]
        box = (idx, y0, y1, x0, x1)
    elif axis == 1:
        z0, z1 = (0, vol_man.shape[0]) if crop_size <= 0 else _clamp_crop(seed[0], crop_size, vol_man.shape[0])
        x0, x1 = (0, vol_man.shape[2]) if crop_size <= 0 else _clamp_crop(seed[2], crop_size, vol_man.shape[2])
        img = vol_man.vol[z0:z1, idx, x0:x1]
        box = (idx, z0, z1, x0, x1)
    else:
        z0, z1 = (0, vol_man.shape[0]) if crop_size <= 0 else _clamp_crop(seed[0], crop_size, vol_man.shape[0])
        y0, y1 = (0, vol_man.shape[1]) if crop_size <= 0 else _clamp_crop(seed[1], crop_size, vol_man.shape[1])
        img = vol_man.vol[z0:z1, y0:y1, idx]
        box = (idx, z0, z1, y0, y1)
    return np.stack([img] * 3, axis=-1), box


def predict_three_masks(img_predictor, image, local_point):
    img_predictor.set_image(image)
    masks, scores, _ = img_predictor.predict(
        point_coords=np.array([local_point]),
        point_labels=np.array([1]),
        multimask_output=True,
    )
    order = np.argsort(scores)[::-1]
    masks = masks[order]
    scores = scores[order]
    return masks.astype(np.uint8), scores


def auto_choose_axis(img_predictor, vol_man, seed, crop_size):
    best_axis = -1
    best_area = float("inf")

    for axis in [0, 1, 2]:
        idx = seed[axis]
        rgb, box = get_slice_rgb_and_box(vol_man, axis, idx, seed, crop_size)
        local_pt = map_local_point(seed, axis, box)
        masks, _ = predict_three_masks(img_predictor, rgb, local_pt)
        if len(masks) == 0:
            continue
        min_area = min(int(m.sum()) for m in masks)
        if min_area < best_area:
            best_area = min_area
            best_axis = axis

    if best_axis < 0:
        raise RuntimeError("Failed to auto-select axis; SAM2 returned no masks.")
    return best_axis


def overlay_and_save(image, masks, scores, seed_xy, out_path, title, alpha):
    base = image.astype(np.float32) / 255.0
    canvas = base.copy()

    for i, mask in enumerate(masks[:3]):
        color = np.array(COLORS[i], dtype=np.float32)
        m = mask.astype(bool)
        if not np.any(m):
            continue
        canvas[m] = canvas[m] * (1 - alpha) + color * alpha

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(canvas)
    ax.scatter([seed_xy[0]], [seed_xy[1]], c="yellow", s=24, marker="x", linewidths=1.0)
    score_text = " | ".join([f"m{i}:{scores[i]:.3f}" for i in range(min(3, len(scores)))])
    ax.set_title(f"{title}\n{score_text}")
    ax.axis("off")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = get_args()
    seed = parse_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable, fallback to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    vol_man = VolumeManager(args.volume_path, key=args.dataset_key)

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module("use_sam2", version_base="1.2")

    sam2_img_model = build_sam2(args.sam2_model_cfg, args.sam2_checkpoint, device=device)
    img_predictor = SAM2ImagePredictor(sam2_img_model)

    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if args.device.startswith("cuda") else nullcontext()
    with torch.inference_mode(), autocast_ctx:
        axis = args.axis
        if axis == -1:
            axis = auto_choose_axis(img_predictor, vol_man, seed, args.crop_size)
            print(f"Auto-selected axis: {axis}")

        max_idx = vol_man.shape[axis] - 1
        start = max(0, seed[axis] - args.half_window)
        end = min(max_idx, seed[axis] + args.half_window)

        idxs = list(range(start, end + 1))
        print(f"Visualizing {len(idxs)} slices on axis={axis} (crop_size={args.crop_size}).")
        t0 = time.time()
        for idx in tqdm(idxs, desc="SAM2 candidate viz"):
            rgb, box = get_slice_rgb_and_box(vol_man, axis, idx, seed, args.crop_size)
            local_pt = map_local_point(seed, axis, box)
            masks, scores = predict_three_masks(img_predictor, rgb, local_pt)

            out_name = f"axis{axis}_slice{idx:04d}.png"
            out_path = os.path.join(args.output_dir, out_name)
            overlay_and_save(
                image=rgb,
                masks=masks,
                scores=scores,
                seed_xy=local_pt,
                out_path=out_path,
                title=f"axis={axis}, slice={idx}, seed={seed}",
                alpha=args.alpha,
            )
        print(f"Elapsed: {time.time() - t0:.2f}s")

    print(f"Done. Saved visualizations to: {args.output_dir}")


if __name__ == "__main__":
    main()
