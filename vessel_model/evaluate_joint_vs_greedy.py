import argparse
from contextlib import nullcontext

import hydra
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
    parser = argparse.ArgumentParser(description="Evaluate joint-inference vs greedy SAM2 on random GT seeds.")
    parser.add_argument("--sam2_checkpoint", required=True)
    parser.add_argument("--sam2_model_cfg", required=True)
    parser.add_argument("--volume_path", required=True)
    parser.add_argument("--gt_path", required=True, help="Ground-truth nii/nii.gz")
    parser.add_argument("--dataset_key", default="main")

    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--half_window", type=int, default=8)
    parser.add_argument("--axis", type=int, default=-1, choices=[-1, 0, 1, 2])
    parser.add_argument("--crop_size", type=int, default=384)

    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--num_point_jitters", type=int, default=4)
    parser.add_argument("--jitter_radius", type=float, default=8.0)

    parser.add_argument("--w_score", type=float, default=1.0)
    parser.add_argument("--w_iou", type=float, default=3.0)
    parser.add_argument("--w_centroid", type=float, default=0.03)
    parser.add_argument("--w_area", type=float, default=0.6)
    parser.add_argument("--empty_mask_penalty", type=float, default=6.0)

    parser.add_argument("--max_init_mask_area", type=int, default=12000)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def clamp_crop(center, size, limit):
    half = size // 2
    return max(0, center - half), min(limit, center + half)


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


def iou_mask(a, b):
    ab = a.astype(bool)
    bb = b.astype(bool)
    inter = np.logical_and(ab, bb).sum()
    union = np.logical_or(ab, bb).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def unary_cost(c, args):
    return args.w_score * (-np.log(c["score"] + EPS)) + (args.empty_mask_penalty if c["area"] == 0 else 0.0)


def pairwise_cost(c0, c1, args):
    iou_term = 1.0 - iou_mask(c0["mask"], c1["mask"])
    if c0["centroid"] is None or c1["centroid"] is None:
        center_term = 100.0
    else:
        center_term = float(np.linalg.norm(c1["centroid"] - c0["centroid"]))
    area_term = abs(np.log(c1["area"] + EPS) - np.log(c0["area"] + EPS))
    return args.w_iou * iou_term + args.w_centroid * center_term + args.w_area * area_term


def select_seed_slice_mask(masks, scores, max_area):
    if len(masks) == 0:
        return None
    order = np.argsort(scores)[::-1]
    for j in order:
        m = masks[j].astype(np.uint8)
        area = int(m.sum())
        if area <= 0:
            continue
        if area > max_area:
            continue
        return m
    return None


def build_candidates_for_slice(img_predictor, rgb, local_pt, args, rng):
    img_predictor.set_image(rgb)
    h, w = rgb.shape[:2]
    pts = [np.array(local_pt, dtype=np.float32)]
    for _ in range(args.num_point_jitters):
        p = np.array(local_pt, dtype=np.float32) + rng.normal(0.0, args.jitter_radius, size=(2,))
        p[0] = np.clip(p[0], 0, max(0, w - 1))
        p[1] = np.clip(p[1], 0, max(0, h - 1))
        pts.append(p)

    cands = []
    for p in pts:
        masks, scores, _ = img_predictor.predict(
            point_coords=np.array([p]), point_labels=np.array([1]), multimask_output=True
        )
        order = np.argsort(scores)[::-1]
        for j in order:
            m = masks[j].astype(np.uint8)
            cands.append({
                "mask": m,
                "score": float(scores[j]),
                "area": int(m.sum()),
                "centroid": compute_centroid(m),
            })

    cands.sort(key=lambda x: x["score"], reverse=True)
    cands = cands[:args.top_k]
    while len(cands) < args.top_k:
        em = np.zeros((h, w), dtype=np.uint8)
        cands.append({"mask": em, "score": 0.0, "area": 0, "centroid": None})
    return cands


def dp_select_path(layers, args):
    n, k = len(layers), len(layers[0])
    dp = np.full((n, k), np.inf, dtype=np.float64)
    parent = np.full((n, k), -1, dtype=np.int32)

    for j in range(k):
        dp[0, j] = unary_cost(layers[0][j], args)
    for t in range(1, n):
        for j in range(k):
            u = unary_cost(layers[t][j], args)
            best, bi = np.inf, -1
            for i in range(k):
                v = dp[t - 1, i] + u + pairwise_cost(layers[t - 1][i], layers[t][j], args)
                if v < best:
                    best, bi = v, i
            dp[t, j], parent[t, j] = best, bi

    path = [-1] * n
    path[-1] = int(np.argmin(dp[-1]))
    for t in range(n - 1, 0, -1):
        path[t - 1] = int(parent[t, path[t]])
    return path


def get_gt_crop(gt, axis, idx, box):
    if axis == 0:
        return gt[idx, box[1]:box[2], box[3]:box[4]]
    if axis == 1:
        return gt[box[1]:box[2], idx, box[3]:box[4]]
    return gt[box[1]:box[2], box[3]:box[4], idx]


def bin_metrics(pred, gt):
    p = pred.astype(bool)
    g = gt.astype(bool)
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, np.logical_not(g)).sum()
    fn = np.logical_and(np.logical_not(p), g).sum()
    tn = np.logical_and(np.logical_not(p), np.logical_not(g)).sum()

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(prec),
        "recall": float(rec),
        "acc": float(acc),
    }


def auto_choose_axis(img_predictor, vol_man, seed, crop_size, max_init_mask_area):
    best_axis, best_area = -1, float("inf")
    for axis in [0, 1, 2]:
        rgb, box = get_slice_rgb_and_box(vol_man, axis, seed[axis], seed, crop_size)
        local_pt = map_local_point(seed, axis, box)
        img_predictor.set_image(rgb)
        masks, scores, _ = img_predictor.predict(
            point_coords=np.array([local_pt]), point_labels=np.array([1]), multimask_output=True
        )
        m = select_seed_slice_mask(masks, scores, max_init_mask_area)
        if m is None:
            continue
        area = int(m.sum())
        if area < best_area:
            best_axis, best_area = axis, area
    return best_axis


def main():
    args = get_args()
    rng = np.random.default_rng(args.random_seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable, fallback to CPU")
        args.device = "cpu"

    gt = nib.load(args.gt_path).get_fdata()
    gt = (gt > 0).astype(np.uint8)

    vol_man = VolumeManager(args.volume_path, key=args.dataset_key)
    if tuple(vol_man.shape) != tuple(gt.shape):
        raise ValueError(f"Shape mismatch: volume={vol_man.shape}, gt={gt.shape}")

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('use_sam2', version_base='1.2')

    device = torch.device(args.device)
    sam2_img_model = build_sam2(args.sam2_model_cfg, args.sam2_checkpoint, device=device)
    img_predictor = SAM2ImagePredictor(sam2_img_model)

    pos = np.argwhere(gt > 0)
    if len(pos) == 0:
        raise RuntimeError("GT has no positive voxels")

    choose_n = min(args.num_seeds, len(pos))
    pick_ids = rng.choice(len(pos), size=choose_n, replace=False)
    seeds = [tuple(map(int, pos[i])) for i in pick_ids]

    joint_metrics_all = []
    greedy_metrics_all = []

    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if args.device.startswith("cuda") else nullcontext()
    with torch.inference_mode(), autocast_ctx:
        for seed in tqdm(seeds, desc="Evaluate seeds"):
            axis = args.axis if args.axis != -1 else auto_choose_axis(
                img_predictor, vol_man, seed, args.crop_size, args.max_init_mask_area
            )
            if axis == -1:
                continue

            start = max(0, seed[axis] - args.half_window)
            end = min(vol_man.shape[axis] - 1, seed[axis] + args.half_window)
            idxs = list(range(start, end + 1))

            layers = []
            greedy_stack = []
            gt_stack = []

            for idx in idxs:
                rgb, box = get_slice_rgb_and_box(vol_man, axis, idx, seed, args.crop_size)
                local_pt = map_local_point(seed, axis, box)

                # greedy per-slice (score-first, area constrained)
                img_predictor.set_image(rgb)
                masks, scores, _ = img_predictor.predict(
                    point_coords=np.array([local_pt]), point_labels=np.array([1]), multimask_output=True
                )
                gm = select_seed_slice_mask(masks, scores, args.max_init_mask_area)
                if gm is None:
                    gm = np.zeros(rgb.shape[:2], dtype=np.uint8)
                greedy_stack.append(gm)

                # joint candidates
                cands = build_candidates_for_slice(img_predictor, rgb, local_pt, args, rng)
                layers.append(cands)

                gt_stack.append(get_gt_crop(gt, axis, idx, box).astype(np.uint8))

            path = dp_select_path(layers, args)
            joint_stack = [layers[t][path[t]]["mask"].astype(np.uint8) for t in range(len(path))]

            gt_vol = np.stack(gt_stack, axis=0)
            greedy_vol = np.stack(greedy_stack, axis=0)
            joint_vol = np.stack(joint_stack, axis=0)

            greedy_metrics_all.append(bin_metrics(greedy_vol, gt_vol))
            joint_metrics_all.append(bin_metrics(joint_vol, gt_vol))

    if len(joint_metrics_all) == 0:
        raise RuntimeError("No valid seeds evaluated.")

    def avg(ms, key):
        return float(np.mean([m[key] for m in ms]))

    keys = ["dice", "iou", "precision", "recall", "acc"]
    print("\n=== Average metrics over evaluated seeds ===")
    for k in keys:
        jv = avg(joint_metrics_all, k)
        gv = avg(greedy_metrics_all, k)
        print(f"{k:>9s} | joint: {jv:.4f} | greedy: {gv:.4f} | delta: {jv-gv:+.4f}")


if __name__ == "__main__":
    main()
