from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..sam2_baseline.image_utils import get_slice, to_uint8_rgb
from ..sam2_baseline.predict_utils import map_local_point
from .common import select_mask_with_constraints

EPS = 1e-6


@dataclass
class JointInitResult:
    seed: Tuple[int, int, int]
    axis: int
    box: Tuple[int, int, int, int, int]
    init_masks_by_global_idx: Dict[int, np.ndarray]
    best_energy: float
    energy_std: float
    energy_uniformity: float
    robustness_energy_cv: float
    robustness_path_iou: float
    avg_area: float
    is_trustworthy: bool


def compute_centroid(mask):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    return np.array([ys.mean(), xs.mean()], dtype=np.float32)


def iou(mask_a, mask_b):
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return float(inter) / float(uni) if uni > 0 else 0.0


def unary_cost(c, w_score, empty_mask_penalty):
    return w_score * (-np.log(c["score"] + EPS)) + (empty_mask_penalty if c["area"] == 0 else 0.0)


def pairwise_cost(a, b, w_iou, w_centroid, w_area):
    iou_term = 1.0 - iou(a["mask"], b["mask"])
    if a["centroid"] is None or b["centroid"] is None:
        centroid_term = 100.0
    else:
        centroid_term = float(np.linalg.norm(b["centroid"] - a["centroid"]))
    area_term = abs(np.log(a["area"] + EPS) - np.log(b["area"] + EPS))
    return w_iou * iou_term + w_centroid * centroid_term + w_area * area_term


def build_candidates_for_slice(img_predictor, rgb, local_pt, top_k, num_point_jitters, jitter_radius, rng):
    img_predictor.set_image(rgb)
    h, w = rgb.shape[:2]
    pts = [np.array(local_pt, dtype=np.float32)]
    for _ in range(num_point_jitters):
        p = np.array(local_pt, dtype=np.float32) + rng.normal(0.0, jitter_radius, size=(2,))
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
            cands.append(
                {
                    "mask": m,
                    "score": float(scores[j]),
                    "area": int(m.sum()),
                    "centroid": compute_centroid(m),
                }
            )

    cands.sort(key=lambda x: x["score"], reverse=True)
    cands = cands[:top_k]
    while len(cands) < top_k:
        em = np.zeros((h, w), dtype=np.uint8)
        cands.append({"mask": em, "score": 0.0, "area": 0, "centroid": None})
    return cands


def dp_best_path_with_energy(layers, args):
    n, k = len(layers), len(layers[0])
    dp = np.full((n, k), np.inf, dtype=np.float64)
    parent = np.full((n, k), -1, dtype=np.int32)
    unary_table = np.zeros((n, k), dtype=np.float64)

    for t in range(n):
        for j in range(k):
            unary_table[t, j] = unary_cost(layers[t][j], args.w_score, args.empty_mask_penalty)

    for j in range(k):
        dp[0, j] = unary_table[0, j]

    for t in range(1, n):
        for j in range(k):
            u = unary_table[t, j]
            best_val, best_i = np.inf, -1
            for i in range(k):
                p = pairwise_cost(layers[t - 1][i], layers[t][j], args.w_iou, args.w_centroid, args.w_area)
                v = dp[t - 1, i] + u + p
                if v < best_val:
                    best_val, best_i = v, i
            dp[t, j] = best_val
            parent[t, j] = best_i

    path = [-1] * n
    path[-1] = int(np.argmin(dp[-1]))
    for t in range(n - 1, 0, -1):
        path[t - 1] = int(parent[t, path[t]])

    best_energy = float(dp[-1, path[-1]])
    per_step_energy = []
    for t in range(n):
        j = path[t]
        u = unary_table[t, j]
        if t == 0:
            p = 0.0
        else:
            i = path[t - 1]
            p = pairwise_cost(layers[t - 1][i], layers[t][j], args.w_iou, args.w_centroid, args.w_area)
        per_step_energy.append(float(u + p))

    return {"path": path, "best_energy": best_energy, "per_step_energy": per_step_energy}


def select_best_axis_seed_mask(img_predictor, crops, seed, max_init_mask_area, autocast_device):
    best_axis, best_mask, min_area = -1, None, float("inf")
    with torch.inference_mode(), torch.autocast(autocast_device, dtype=torch.bfloat16):
        for axis in [0, 1, 2]:
            img, box = crops[axis]
            local_pt = map_local_point(seed, axis, box)
            img_predictor.set_image(img)
            masks, scores, _ = img_predictor.predict(
                point_coords=np.array([local_pt]), point_labels=np.array([1]), multimask_output=True
            )
            if len(masks) == 0:
                continue
            order = np.argsort(scores)[::-1]
            picked = None
            for j in order:
                m = masks[j].astype(np.uint8)
                area = int(m.sum())
                if area <= 0:
                    continue
                if area > max_init_mask_area:
                    continue
                picked = m
                break
            if picked is None:
                for j in order:
                    m = masks[j].astype(np.uint8)
                    if int(m.sum()) > 0:
                        picked = m
                        break
            if picked is None:
                continue
            area = int(picked.sum())
            if area < min_area:
                min_area, best_axis, best_mask = area, axis, picked
    return best_axis, best_mask


def build_layers_for_seed(img_predictor, vol_man, seed, axis, box, args, rng):
    center = seed[axis]
    start = max(0, center - args.init_half_window)
    end = min(vol_man.shape[axis] - 1, center + args.init_half_window)
    idxs = list(range(start, end + 1))

    local_pt = map_local_point(seed, axis, box)
    layers = []
    for idx in idxs:
        sl = get_slice(vol_man.vol, axis, idx, box)
        rgb = to_uint8_rgb(sl)
        if rgb is None:
            rgb = np.zeros((box[2] - box[1], box[4] - box[3], 3), dtype=np.uint8)
        cands = build_candidates_for_slice(
            img_predictor, rgb, local_pt, args.top_k, args.num_point_jitters, args.jitter_radius, rng
        )
        layers.append(cands)
    return idxs, layers


def path_masks_from_layers(layers, path):
    return [layers[t][path[t]]["mask"].astype(np.uint8) for t in range(len(path))]


def mean_path_iou(path_a_masks, path_b_masks):
    n = min(len(path_a_masks), len(path_b_masks))
    if n == 0:
        return 0.0
    vals = [iou(path_a_masks[t], path_b_masks[t]) for t in range(n)]
    return float(np.mean(vals))


def build_init_masks_dict(idxs, masks):
    return {idxs[t]: masks[t].astype(np.uint8) for t in range(len(idxs))}


def evaluate_axis_joint_init(img_predictor, vol_man, seed, axis, box, args, axis_rng) -> Optional[JointInitResult]:
    idxs, layers = build_layers_for_seed(img_predictor, vol_man, seed, axis, box, args, axis_rng)
    if len(layers) == 0:
        return None

    baseline = dp_best_path_with_energy(layers, args)
    baseline_masks = path_masks_from_layers(layers, baseline["path"])
    center_local_idx = min(max(seed[axis] - idxs[0], 0), len(baseline_masks) - 1)
    center_mask = baseline_masks[center_local_idx] if len(baseline_masks) > 0 else None
    if center_mask is None or int(center_mask.sum()) == 0:
        return None
    if int(center_mask.sum()) > args.max_init_mask_area:
        return None

    avg_area = float(np.mean([int(m.sum()) for m in baseline_masks])) if len(baseline_masks) > 0 else 0.0
    if avg_area > args.max_joint_avg_area:
        return None

    per_step = np.array(baseline["per_step_energy"], dtype=np.float64)
    energy_std = float(np.std(per_step)) if len(per_step) > 0 else np.inf
    energy_mean = float(np.mean(per_step)) if len(per_step) > 0 else np.inf
    energy_uniformity = float(energy_std / (energy_mean + EPS))

    robust_energies = []
    robust_path_ious = []
    for _ in range(args.robust_trials):
        trial_rng = np.random.default_rng(int(axis_rng.integers(0, 2**31 - 1)))
        _, layers_t = build_layers_for_seed(img_predictor, vol_man, seed, axis, box, args, trial_rng)
        trial = dp_best_path_with_energy(layers_t, args)
        robust_energies.append(float(trial["best_energy"]))
        trial_masks = path_masks_from_layers(layers_t, trial["path"])
        robust_path_ious.append(mean_path_iou(baseline_masks, trial_masks))

    robust_energies = np.array(robust_energies, dtype=np.float64)
    robust_energy_cv = float(np.std(robust_energies) / (np.mean(robust_energies) + EPS))
    robust_path_iou = float(np.mean(robust_path_ious)) if len(robust_path_ious) > 0 else 0.0

    is_trustworthy = True
    if getattr(args, "enable_seed_judge", False):
        is_trustworthy = (
            baseline["best_energy"] <= args.thr_best_energy
            and energy_std <= args.thr_energy_std
            and energy_uniformity <= args.thr_energy_uniformity
            and robust_energy_cv <= args.thr_robust_energy_cv
            and robust_path_iou >= args.thr_robust_path_iou
            and avg_area >= args.thr_avg_area_min
            and avg_area <= args.thr_avg_area_max
        )

    return JointInitResult(
        seed=seed,
        axis=axis,
        box=box,
        init_masks_by_global_idx=build_init_masks_dict(idxs, baseline_masks),
        best_energy=float(baseline["best_energy"]),
        energy_std=energy_std,
        energy_uniformity=energy_uniformity,
        robustness_energy_cv=robust_energy_cv,
        robustness_path_iou=robust_path_iou,
        avg_area=avg_area,
        is_trustworthy=bool(is_trustworthy),
    )


def prepare_seed_init(img_predictor, vol_man, seed, args, base_rng) -> Optional[JointInitResult]:
    crops = vol_man.get_triplane_crops(seed)
    axis_results: List[JointInitResult] = []

    for axis in [0, 1, 2]:
        _, box = crops[axis]
        axis_seed = int(base_rng.integers(0, 2**31 - 1))
        axis_rng = np.random.default_rng(axis_seed)
        axis_result = evaluate_axis_joint_init(
            img_predictor=img_predictor,
            vol_man=vol_man,
            seed=seed,
            axis=axis,
            box=box,
            args=args,
            axis_rng=axis_rng,
        )
        if axis_result is None:
            continue
        axis_results.append(axis_result)

    if len(axis_results) == 0:
        return None

    axis_results.sort(key=lambda x: (x.best_energy, x.avg_area, x.axis))
    return axis_results[0]


def prepare_basic_seed_init(img_predictor, vol_man, seed, args) -> Optional[JointInitResult]:
    crops = vol_man.get_triplane_crops(seed)
    best_axis, best_mask, min_area = -1, None, float("inf")

    with torch.inference_mode(), torch.autocast(str(vol_man.device), dtype=torch.bfloat16):
        for axis in [0, 1, 2]:
            img, box = crops[axis]
            local_pt = map_local_point(seed, axis, box)
            img_predictor.set_image(img)
            masks, scores, _ = img_predictor.predict(
                point_coords=np.array([local_pt]),
                point_labels=np.array([1]),
                multimask_output=True,
            )
            mask, _ = select_mask_with_constraints(masks, scores, args.max_init_mask_area)
            if mask is None:
                continue
            area = int(mask.sum())
            if area < min_area:
                min_area = area
                best_axis = axis
                best_mask = mask

    if best_axis == -1 or best_mask is None or int(best_mask.sum()) == 0:
        return None
    if int(best_mask.sum()) > args.max_init_mask_area:
        return None

    _, box = crops[best_axis]
    start_idx = seed[best_axis]
    init_masks_by_global_idx = {int(start_idx): best_mask.astype(np.uint8)}
    return JointInitResult(
        seed=seed,
        axis=best_axis,
        box=box,
        init_masks_by_global_idx=init_masks_by_global_idx,
        best_energy=0.0,
        energy_std=0.0,
        energy_uniformity=0.0,
        robustness_energy_cv=0.0,
        robustness_path_iou=1.0,
        avg_area=float(best_mask.sum()),
        is_trustworthy=True,
    )
