import argparse
import os
from contextlib import nullcontext

import hydra
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .data_manager import VolumeManager
from .sam2_baseline.predict_utils import map_local_point


EPS = 1e-6


def get_args():
    parser = argparse.ArgumentParser(
        description="Baseline joint inference over multi-slice SAM2 candidates via DP/Viterbi."
    )
    parser.add_argument("--sam2_checkpoint", required=True, help="Path to SAM2 checkpoint")
    parser.add_argument("--sam2_model_cfg", required=True, help="Path to SAM2 model config")
    parser.add_argument("--volume_path", required=True, help="Path to .h5 or .nii/.nii.gz file")
    parser.add_argument("--dataset_key", default="main", help="Key name in h5 file")
    parser.add_argument("--seed", required=True, help="Seed in z,y,x format, e.g. 120,300,280")
    parser.add_argument("--axis", type=int, default=-1, choices=[-1, 0, 1, 2], help="-1: auto axis")
    parser.add_argument("--half_window", type=int, default=8, help="Use slices [seed-W, seed+W]")

    parser.add_argument("--top_k", type=int, default=8, help="Keep top-K candidates per slice")
    parser.add_argument("--num_point_jitters", type=int, default=4, help="Number of random point jitters")
    parser.add_argument("--jitter_radius", type=float, default=8.0, help="Point jitter std in pixels")
    parser.add_argument("--crop_size", type=int, default=384, help="ROI crop around seed, <=0 full slice")

    parser.add_argument("--w_score", type=float, default=1.0)
    parser.add_argument("--w_iou", type=float, default=3.0)
    parser.add_argument("--w_centroid", type=float, default=0.03)
    parser.add_argument("--w_area", type=float, default=0.6)
    parser.add_argument("--empty_mask_penalty", type=float, default=6.0)

    parser.add_argument("--invalid_empty_ratio", type=float, default=0.5)
    parser.add_argument("--invalid_mean_iou", type=float, default=0.1)

    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--output_dir", default="./joint_inference_baseline")
    parser.add_argument("--save_mask_volume", action="store_true", help="Save selected window mask as nii.gz")
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def parse_seed(seed_text):
    parts = [int(p.strip()) for p in seed_text.split(",")]
    if len(parts) != 3:
        raise ValueError("--seed must be z,y,x")
    return tuple(parts)


def clamp_crop(center, size, limit):
    half = size // 2
    start = max(0, center - half)
    end = min(limit, center + half)
    return start, end


def get_slice_rgb_and_box(vol_man, axis, idx, seed, crop_size):
    if axis == 0:
        y0, y1 = (0, vol_man.shape[1]) if crop_size <= 0 else clamp_crop(seed[1], crop_size, vol_man.shape[1])
        x0, x1 = (0, vol_man.shape[2]) if crop_size <= 0 else clamp_crop(seed[2], crop_size, vol_man.shape[2])
        img = vol_man.vol[idx, y0:y1, x0:x1]
        box = (idx, y0, y1, x0, x1)
    elif axis == 1:
        z0, z1 = (0, vol_man.shape[0]) if crop_size <= 0 else clamp_crop(seed[0], crop_size, vol_man.shape[0])
        x0, x1 = (0, vol_man.shape[2]) if crop_size <= 0 else clamp_crop(seed[2], crop_size, vol_man.shape[2])
        img = vol_man.vol[z0:z1, idx, x0:x1]
        box = (idx, z0, z1, x0, x1)
    else:
        z0, z1 = (0, vol_man.shape[0]) if crop_size <= 0 else clamp_crop(seed[0], crop_size, vol_man.shape[0])
        y0, y1 = (0, vol_man.shape[1]) if crop_size <= 0 else clamp_crop(seed[1], crop_size, vol_man.shape[1])
        img = vol_man.vol[z0:z1, y0:y1, idx]
        box = (idx, z0, z1, y0, y1)
    return np.stack([img] * 3, axis=-1), box


def compute_centroid(mask):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    return np.array([ys.mean(), xs.mean()], dtype=np.float32)


def get_point_ensemble(local_pt, h, w, num_jitters, jitter_radius, rng):
    points = [np.array(local_pt, dtype=np.float32)]
    for _ in range(num_jitters):
        jitter = rng.normal(0.0, jitter_radius, size=(2,))
        p = np.array(local_pt, dtype=np.float32) + jitter
        p[0] = np.clip(p[0], 0, max(0, w - 1))
        p[1] = np.clip(p[1], 0, max(0, h - 1))
        points.append(p)
    return points


def build_candidates_for_slice(img_predictor, rgb, local_pt, top_k, num_jitters, jitter_radius, rng):
    img_predictor.set_image(rgb)
    h, w = rgb.shape[:2]
    points = get_point_ensemble(local_pt, h, w, num_jitters, jitter_radius, rng)

    candidates = []
    for p in points:
        masks, scores, _ = img_predictor.predict(
            point_coords=np.array([p]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        order = np.argsort(scores)[::-1]
        for j in order:
            mask = masks[j].astype(np.uint8)
            area = int(mask.sum())
            centroid = compute_centroid(mask)
            candidates.append(
                {
                    "mask": mask,
                    "score": float(scores[j]),
                    "area": area,
                    "centroid": centroid,
                    "point": p.copy(),
                }
            )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    candidates = candidates[:top_k]

    # 保证每层固定K个节点，方便DP张量化
    while len(candidates) < top_k:
        empty = np.zeros((h, w), dtype=np.uint8)
        candidates.append(
            {
                "mask": empty,
                "score": 0.0,
                "area": 0,
                "centroid": None,
                "point": np.array(local_pt, dtype=np.float32),
            }
        )
    return candidates


def overlay_candidate_set(rgb, candidates, local_pt, out_path, alpha=0.35):
    base = rgb.astype(np.float32) / 255.0
    canvas = base.copy()
    cmap = plt.get_cmap("tab20")

    for k, cand in enumerate(candidates):
        m = cand["mask"].astype(bool)
        if not np.any(m):
            continue
        color = np.array(cmap(k % 20)[:3], dtype=np.float32)
        canvas[m] = canvas[m] * (1 - alpha) + color * alpha

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(canvas)
    ax.scatter([local_pt[0]], [local_pt[1]], c="yellow", s=20, marker="x", linewidths=1)
    info = " ".join([f"#{k}:{c['score']:.3f}" for k, c in enumerate(candidates[:6])])
    ax.set_title(f"All candidates (top shown): {info}")
    ax.axis("off")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def overlay_selected(rgb, cand, selected_idx, out_path, title, alpha=0.45):
    base = rgb.astype(np.float32) / 255.0
    canvas = base.copy()
    m = cand["mask"].astype(bool)
    if np.any(m):
        color = np.array([1.0, 0.2, 0.2], dtype=np.float32)
        canvas[m] = canvas[m] * (1 - alpha) + color * alpha

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(canvas)
    ax.set_title(title + f" | selected={selected_idx}, score={cand['score']:.3f}, area={cand['area']}")
    ax.axis("off")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def iou(mask_a, mask_b):
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def unary_cost(cand, w_score, empty_mask_penalty):
    score_cost = -np.log(cand["score"] + EPS)
    empty_cost = empty_mask_penalty if cand["area"] == 0 else 0.0
    return w_score * score_cost + empty_cost


def pairwise_cost(c_prev, c_curr, w_iou, w_centroid, w_area):
    iou_term = 1.0 - iou(c_prev["mask"], c_curr["mask"])

    if c_prev["centroid"] is None or c_curr["centroid"] is None:
        centroid_term = 100.0
    else:
        centroid_term = float(np.linalg.norm(c_curr["centroid"] - c_prev["centroid"]))

    area_term = abs(np.log(c_curr["area"] + EPS) - np.log(c_prev["area"] + EPS))

    return w_iou * iou_term + w_centroid * centroid_term + w_area * area_term


def dp_viterbi(candidate_layers, args):
    n = len(candidate_layers)
    k = len(candidate_layers[0])

    dp = np.full((n, k), np.inf, dtype=np.float64)
    parent = np.full((n, k), -1, dtype=np.int32)

    for j in range(k):
        dp[0, j] = unary_cost(candidate_layers[0][j], args.w_score, args.empty_mask_penalty)

    for t in range(1, n):
        for j in range(k):
            u = unary_cost(candidate_layers[t][j], args.w_score, args.empty_mask_penalty)
            best_val = np.inf
            best_i = -1
            for i in range(k):
                p = pairwise_cost(
                    candidate_layers[t - 1][i],
                    candidate_layers[t][j],
                    args.w_iou,
                    args.w_centroid,
                    args.w_area,
                )
                v = dp[t - 1, i] + u + p
                if v < best_val:
                    best_val = v
                    best_i = i
            dp[t, j] = best_val
            parent[t, j] = best_i

    path = [-1] * n
    path[-1] = int(np.argmin(dp[-1]))
    for t in range(n - 1, 0, -1):
        path[t - 1] = int(parent[t, path[t]])
    return path, float(dp[-1, path[-1]])


def auto_choose_axis(img_predictor, vol_man, seed, crop_size):
    best_axis, best_area = -1, float("inf")
    for axis in [0, 1, 2]:
        idx = seed[axis]
        rgb, box = get_slice_rgb_and_box(vol_man, axis, idx, seed, crop_size)
        local_pt = map_local_point(seed, axis, box)
        img_predictor.set_image(rgb)
        masks, scores, _ = img_predictor.predict(
            point_coords=np.array([local_pt]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        if len(masks) == 0:
            continue
        area = min(int(m.sum()) for m in masks)
        if area < best_area:
            best_area = area
            best_axis = axis
    if best_axis < 0:
        raise RuntimeError("Failed to auto-select axis")
    return best_axis


def save_selected_mask_volume(vol_man, axis, slice_indices, selected_candidates, out_path):
    vol = np.zeros(vol_man.shape, dtype=np.uint8)
    for idx, cand in zip(slice_indices, selected_candidates):
        mask = cand["mask"].astype(np.uint8)
        if mask.sum() == 0:
            continue
        if axis == 0:
            vol[idx, :, :] = np.maximum(vol[idx, :, :], mask)
        elif axis == 1:
            vol[:, idx, :] = np.maximum(vol[:, idx, :], mask)
        else:
            vol[:, :, idx] = np.maximum(vol[:, :, idx], mask)

    affine = vol_man.affine if vol_man.affine is not None else np.eye(4)
    nib.save(nib.Nifti1Image(vol, affine), out_path)


def main():
    args = get_args()
    np.random.seed(args.random_seed)
    rng = np.random.default_rng(args.random_seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable, fallback to CPU.")
        args.device = "cpu"

    os.makedirs(args.output_dir, exist_ok=True)
    seed = parse_seed(args.seed)

    vol_man = VolumeManager(args.volume_path, key=args.dataset_key)

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module("use_sam2", version_base="1.2")

    device = torch.device(args.device)
    sam2_img_model = build_sam2(args.sam2_model_cfg, args.sam2_checkpoint, device=device)
    img_predictor = SAM2ImagePredictor(sam2_img_model)

    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if args.device.startswith("cuda") else nullcontext()
    with torch.inference_mode(), autocast_ctx:
        axis = args.axis if args.axis != -1 else auto_choose_axis(img_predictor, vol_man, seed, args.crop_size)

        max_idx = vol_man.shape[axis] - 1
        start = max(0, seed[axis] - args.half_window)
        end = min(max_idx, seed[axis] + args.half_window)
        slice_indices = list(range(start, end + 1))

        print(f"Axis={axis}, slices={len(slice_indices)}, top_k={args.top_k}")

        candidate_layers = []
        slice_rgbs = []
        for idx in tqdm(slice_indices, desc="Build candidates"):
            rgb, box = get_slice_rgb_and_box(vol_man, axis, idx, seed, args.crop_size)
            local_pt = map_local_point(seed, axis, box)
            cands = build_candidates_for_slice(
                img_predictor=img_predictor,
                rgb=rgb,
                local_pt=local_pt,
                top_k=args.top_k,
                num_jitters=args.num_point_jitters,
                jitter_radius=args.jitter_radius,
                rng=rng,
            )
            candidate_layers.append(cands)
            slice_rgbs.append((rgb, local_pt))

            overlay_candidate_set(
                rgb,
                cands,
                local_pt,
                out_path=os.path.join(args.output_dir, "all_candidates", f"axis{axis}_slice{idx:04d}.png"),
                alpha=args.alpha,
            )

        best_path, total_energy = dp_viterbi(candidate_layers, args)
        print(f"DP done. Total energy={total_energy:.4f}")

        selected = []
        empty_count = 0
        ious = []
        for t, idx in enumerate(slice_indices):
            sel_idx = best_path[t]
            sel = candidate_layers[t][sel_idx]
            selected.append(sel)
            if sel["area"] == 0:
                empty_count += 1

            if t > 0:
                ious.append(iou(selected[t - 1]["mask"], selected[t]["mask"]))

            rgb, _ = slice_rgbs[t]
            overlay_selected(
                rgb,
                sel,
                sel_idx,
                out_path=os.path.join(args.output_dir, "joint_selected", f"axis{axis}_slice{idx:04d}.png"),
                title=f"axis={axis}, slice={idx}",
                alpha=args.alpha,
            )

        empty_ratio = empty_count / max(1, len(selected))
        mean_iou = float(np.mean(ious)) if len(ious) > 0 else 0.0
        print(f"Path stats: empty_ratio={empty_ratio:.3f}, mean_adj_iou={mean_iou:.3f}")
        if empty_ratio > args.invalid_empty_ratio or mean_iou < args.invalid_mean_iou:
            print("[Warning] Seed may be unreliable for joint tube inference in this window.")

        if args.save_mask_volume:
            vol_path = os.path.join(args.output_dir, f"joint_selected_axis{axis}.nii.gz")
            save_selected_mask_volume(vol_man, axis, slice_indices, selected, vol_path)
            print(f"Saved selected mask volume: {vol_path}")

    print(f"Done. Outputs at: {args.output_dir}")


if __name__ == "__main__":
    main()
