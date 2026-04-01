# 基于 sam2_main4_joint_init 的“seed可信度判别”实验脚本
# 功能：
# 1) 读取 seed_file 或自动生成 init_seg seeds
# 2) 在局部窗口内做多切片多候选联合路径搜索
# 3) 用路径能量、能量均匀性、扰动鲁棒性等指标判断该 seed 是否可信（更像血管）
# 4) 仅打印/保存判别结果，不做 SAM2 视频追踪
import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import h5py
import hydra
import numpy as np
import torch
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .data_manager import VolumeManager
from .preprocessing import get_multi_axis_init_seg, get_seeds_from_init_seg, get_seg
from .sam2_baseline.image_utils import get_slice, to_uint8_rgb
from .sam2_baseline.predict_utils import map_local_point

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
EPS = 1e-6


@dataclass
class SeedJudgeResult:
    seed: Tuple[int, int, int]
    axis: int
    best_energy: float
    mean_unary: float
    mean_pairwise: float
    energy_std: float
    energy_uniformity: float
    robustness_energy_cv: float
    robustness_path_iou: float
    avg_area: float
    is_trustworthy: bool


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2_checkpoint", required=True)
    parser.add_argument("--sam2_model_cfg", required=True)
    parser.add_argument("--volume_path", required=True)
    parser.add_argument("--output_dir", default="./bv_seed_judge_output")
    parser.add_argument("--dataset_key", default="main")
    parser.add_argument("--seed_file", default=None)
    parser.add_argument("--init_seg_path", default=None, help="Optional cached init-seg h5 path")
    parser.add_argument("--axis", type=int, default=3)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--gaussian_kernel", type=int, default=5)
    parser.add_argument("--min_bright", type=int, default=40)
    parser.add_argument("--remove_portion", type=float, default=0.98)
    parser.add_argument("--device", default="cuda")

    # 联合初始化候选参数（和 sam2_main4_joint_init 对齐）
    parser.add_argument("--init_half_window", type=int, default=6)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--num_point_jitters", type=int, default=3)
    parser.add_argument("--jitter_radius", type=float, default=6.0)
    parser.add_argument("--w_score", type=float, default=1.0)
    parser.add_argument("--w_iou", type=float, default=3.0)
    parser.add_argument("--w_centroid", type=float, default=0.03)
    parser.add_argument("--w_area", type=float, default=0.6)
    parser.add_argument("--empty_mask_penalty", type=float, default=6.0)
    parser.add_argument("--max_init_mask_area", type=int, default=12000)

    # 可信度判别参数
    parser.add_argument("--robust_trials", type=int, default=5,
                        help="扰动鲁棒性试验次数（每次随机抖动点并重算最优路径）")
    parser.add_argument("--thr_best_energy", type=float, default=38.0,
                        help="最佳路径总能量阈值（越小越好）")
    parser.add_argument("--thr_energy_std", type=float, default=2.2,
                        help="路径逐层局部能量标准差阈值（越小越均匀）")
    parser.add_argument("--thr_energy_uniformity", type=float, default=0.35,
                        help="能量均匀性阈值=std/mean（越小越均匀）")
    parser.add_argument("--thr_robust_energy_cv", type=float, default=0.18,
                        help="鲁棒试验最佳能量变异系数阈值（越小越稳）")
    parser.add_argument("--thr_robust_path_iou", type=float, default=0.58,
                        help="鲁棒试验与基线路径mask平均IoU阈值（越大越稳）")
    parser.add_argument("--thr_avg_area_min", type=float, default=20.0,
                        help="路径平均mask面积下限")
    parser.add_argument("--thr_avg_area_max", type=float, default=12000.0,
                        help="路径平均mask面积上限")

    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--save_csv", action="store_true", help="将判别结果保存到csv")
    return parser.parse_args()


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
            cands.append({
                "mask": m,
                "score": float(scores[j]),
                "area": int(m.sum()),
                "centroid": compute_centroid(m),
            })

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

    # 记录分解能量，便于后续分析“是否平滑均匀”
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

    # 计算路径每一层的局部能量（unary + 与前一层pairwise）
    per_step_energy = []
    unary_terms = []
    pairwise_terms = []
    for t in range(n):
        j = path[t]
        u = unary_table[t, j]
        unary_terms.append(float(u))
        if t == 0:
            p = 0.0
        else:
            i = path[t - 1]
            p = pairwise_cost(layers[t - 1][i], layers[t][j], args.w_iou, args.w_centroid, args.w_area)
        pairwise_terms.append(float(p))
        per_step_energy.append(float(u + p))

    return {
        "path": path,
        "best_energy": best_energy,
        "per_step_energy": per_step_energy,
        "unary_terms": unary_terms,
        "pairwise_terms": pairwise_terms,
    }


def select_best_axis_seed_mask(img_predictor, crops, seed, max_init_mask_area):
    best_axis, best_mask, min_area = -1, None, float("inf")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for axis in [0, 1, 2]:
            img, box = crops[axis]
            local_pt = map_local_point(seed, axis, box)
            img_predictor.set_image(img)
            masks, scores, _ = img_predictor.predict(
                point_coords=np.array([local_pt]),
                point_labels=np.array([1]),
                multimask_output=True,
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


def judge_seed_once(img_predictor, vol_man, seed, args, base_rng):
    # 1) 先找best axis（与 joint_init 同思路）
    crops = vol_man.get_triplane_crops(seed)
    best_axis, best_mask = select_best_axis_seed_mask(img_predictor, crops, seed, args.max_init_mask_area)
    if best_axis == -1 or best_mask is None or int(best_mask.sum()) == 0:
        return None
    if int(best_mask.sum()) > args.max_init_mask_area:
        return None

    _, box = crops[best_axis]

    # 2) 基线路径（固定随机种子）
    idxs, layers = build_layers_for_seed(img_predictor, vol_man, seed, best_axis, box, args, base_rng)
    if len(layers) == 0:
        return None
    baseline = dp_best_path_with_energy(layers, args)
    baseline_masks = path_masks_from_layers(layers, baseline["path"])

    per_step = np.array(baseline["per_step_energy"], dtype=np.float64)
    energy_std = float(np.std(per_step)) if len(per_step) > 0 else np.inf
    energy_mean = float(np.mean(per_step)) if len(per_step) > 0 else np.inf
    energy_uniformity = float(energy_std / (energy_mean + EPS))

    unary_terms = np.array(baseline["unary_terms"], dtype=np.float64)
    pair_terms = np.array(baseline["pairwise_terms"], dtype=np.float64)

    avg_area = float(np.mean([int(m.sum()) for m in baseline_masks])) if len(baseline_masks) > 0 else 0.0

    # 3) 多扰动鲁棒性：重复构图+DP，比较最优能量稳定性与路径IoU稳定性
    robust_energies = []
    robust_path_ious = []
    for _ in range(args.robust_trials):
        trial_rng = np.random.default_rng(int(base_rng.integers(0, 2**31 - 1)))
        _, layers_t = build_layers_for_seed(img_predictor, vol_man, seed, best_axis, box, args, trial_rng)
        trial = dp_best_path_with_energy(layers_t, args)
        robust_energies.append(float(trial["best_energy"]))
        trial_masks = path_masks_from_layers(layers_t, trial["path"])
        robust_path_ious.append(mean_path_iou(baseline_masks, trial_masks))

    robust_energies = np.array(robust_energies, dtype=np.float64)
    robust_energy_cv = float(np.std(robust_energies) / (np.mean(robust_energies) + EPS))
    robust_path_iou = float(np.mean(robust_path_ious)) if len(robust_path_ious) > 0 else 0.0

    # 4) 规则判别（先用可解释的多条件AND，后续可替换成学习器）
    is_trustworthy = (
        baseline["best_energy"] <= args.thr_best_energy
        and energy_std <= args.thr_energy_std
        and energy_uniformity <= args.thr_energy_uniformity
        and robust_energy_cv <= args.thr_robust_energy_cv
        and robust_path_iou >= args.thr_robust_path_iou
        and avg_area >= args.thr_avg_area_min
        and avg_area <= args.thr_avg_area_max
    )

    return SeedJudgeResult(
        seed=seed,
        axis=best_axis,
        best_energy=float(baseline["best_energy"]),
        mean_unary=float(np.mean(unary_terms)) if len(unary_terms) > 0 else np.inf,
        mean_pairwise=float(np.mean(pair_terms)) if len(pair_terms) > 0 else np.inf,
        energy_std=energy_std,
        energy_uniformity=energy_uniformity,
        robustness_energy_cv=robust_energy_cv,
        robustness_path_iou=robust_path_iou,
        avg_area=avg_area,
        is_trustworthy=bool(is_trustworthy),
    )


def load_or_build_seeds(args, vol_man):
    if args.seed_file:
        seeds = []
        with open(args.seed_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = [int(p) for p in line.strip().split(",")]
                if len(parts) != 3:
                    raise ValueError(f"Invalid seed line: {line}")
                seeds.append(tuple(parts))
        return seeds

    default_init_seg_name = f"init_seg_axis{args.axis}_s{args.stride}_t{args.remove_portion}.h5"
    init_seg_path = args.init_seg_path or os.path.join(args.output_dir, default_init_seg_name)

    if os.path.exists(init_seg_path):
        print(f"Loading existing init_seg from {init_seg_path}...")
        with h5py.File(init_seg_path, "r") as f:
            init_seg = f["main"][:]
    else:
        if args.axis in (0, 1, 2):
            axis = args.axis
            init_seg = np.zeros_like(vol_man.vol, dtype=np.uint8)
            if axis == 0:
                for m in tqdm(range(0, vol_man.shape[0], args.stride), desc="Axis 0 (Z)"):
                    image = vol_man.vol[m, :, :]
                    temp_seg = get_seg(image, remove_portion=args.remove_portion,
                                       gaussian_kernel=args.gaussian_kernel, min_bright=args.min_bright)
                    init_seg[m, :, :][temp_seg > 0] = 1
            elif axis == 1:
                for m in tqdm(range(0, vol_man.shape[1], args.stride), desc="Axis 1 (Y)"):
                    image = vol_man.vol[:, m, :]
                    temp_seg = get_seg(image, remove_portion=args.remove_portion,
                                       gaussian_kernel=args.gaussian_kernel, min_bright=args.min_bright)
                    init_seg[:, m, :][temp_seg > 0] = 1
            else:
                for m in tqdm(range(0, vol_man.shape[2], args.stride), desc="Axis 2 (X)"):
                    image = vol_man.vol[:, :, m]
                    temp_seg = get_seg(image, remove_portion=args.remove_portion,
                                       gaussian_kernel=args.gaussian_kernel, min_bright=args.min_bright)
                    init_seg[:, :, m][temp_seg > 0] = 1
        else:
            init_seg = get_multi_axis_init_seg(
                vol_man.vol,
                stride=args.stride,
                thr=args.remove_portion,
                gaussian_kernel=args.gaussian_kernel,
                min_bright=args.min_bright,
            )
        print(f"Saving init_seg to {init_seg_path}...")
        with h5py.File(init_seg_path, "w") as f:
            f.create_dataset("main", data=init_seg, compression="gzip")

    return get_seeds_from_init_seg(init_seg)


def save_results_csv(results: List[SeedJudgeResult], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed_z", "seed_y", "seed_x", "axis",
            "best_energy", "mean_unary", "mean_pairwise",
            "energy_std", "energy_uniformity",
            "robustness_energy_cv", "robustness_path_iou",
            "avg_area", "is_trustworthy",
        ])
        for r in results:
            writer.writerow([
                r.seed[0], r.seed[1], r.seed[2], r.axis,
                f"{r.best_energy:.6f}", f"{r.mean_unary:.6f}", f"{r.mean_pairwise:.6f}",
                f"{r.energy_std:.6f}", f"{r.energy_uniformity:.6f}",
                f"{r.robustness_energy_cv:.6f}", f"{r.robustness_path_iou:.6f}",
                f"{r.avg_area:.6f}", int(r.is_trustworthy),
            ])


def run_seed_judgement():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.random_seed)

    vol_man = VolumeManager(args.volume_path, key=args.dataset_key)

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('use_sam2', version_base='1.2')

    sam2_img_model = build_sam2(args.sam2_model_cfg, args.sam2_checkpoint, device=device)
    img_predictor = SAM2ImagePredictor(sam2_img_model)

    seeds = load_or_build_seeds(args, vol_man)
    if not seeds:
        raise RuntimeError("No seeds available for judgement.")

    print(f"Total seeds: {len(seeds)}")

    results: List[SeedJudgeResult] = []
    for seed in tqdm(seeds, desc="Seed judging"):
        r = judge_seed_once(img_predictor, vol_man, seed, args, rng)
        if r is None:
            print(f"[REJECT] seed={seed} reason=invalid_init_or_empty")
            continue

        tag = "TRUST" if r.is_trustworthy else "REJECT"
        print(
            f"[{tag}] seed={seed} axis={r.axis} "
            f"E={r.best_energy:.3f} E_std={r.energy_std:.3f} U={r.energy_uniformity:.3f} "
            f"RobCV={r.robustness_energy_cv:.3f} RobIoU={r.robustness_path_iou:.3f} "
            f"Area={r.avg_area:.1f}"
        )
        results.append(r)

    trust_cnt = sum(int(r.is_trustworthy) for r in results)
    total_eval = len(results)
    print("=" * 80)
    print(f"Evaluated seeds: {total_eval}")
    print(f"Trustworthy seeds: {trust_cnt}")
    print(f"Rejected seeds: {total_eval - trust_cnt}")
    if total_eval > 0:
        print(f"Trust ratio: {trust_cnt / total_eval:.4f}")

    if args.save_csv:
        csv_path = os.path.join(args.output_dir, "seed_judgement.csv")
        save_results_csv(results, csv_path)
        print(f"Saved csv: {csv_path}")


if __name__ == "__main__":
    run_seed_judgement()
