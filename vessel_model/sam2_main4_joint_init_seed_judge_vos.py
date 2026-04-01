# 基于 sam2_main4_joint_init_seed_judge 的完整实验脚本：
# - 先对每个 seed 做可信度判别
# - REJECT seed 直接忽略
# - TRUST seed 使用“联合选择的候选mask路径”作为多帧初始提示，执行 VOS（同 sam2_main4_joint_init 思路）
import argparse
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import h5py
import hydra
import numpy as np
import torch
from tqdm import tqdm

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .data_manager import VolumeManager
from .preprocessing import get_multi_axis_init_seg, get_seeds_from_init_seg, get_seg
from .sam2_baseline.image_utils import get_slice, to_uint8_rgb, write_jpeg_frames
from .sam2_baseline.predict_utils import map_local_point

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
EPS = 1e-6


@dataclass
class SeedJudgeResult:
    seed: Tuple[int, int, int]
    axis: int
    box: Tuple[int, int, int, int, int]
    best_energy: float
    energy_std: float
    energy_uniformity: float
    robustness_energy_cv: float
    robustness_path_iou: float
    avg_area: float
    init_masks_by_global_idx: Dict[int, np.ndarray]
    is_trustworthy: bool


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2_checkpoint", required=True)
    parser.add_argument("--sam2_model_cfg", required=True)
    parser.add_argument("--volume_path", required=True)
    parser.add_argument("--output_dir", default="./bv_seg_output_joint_seed_judge")
    parser.add_argument("--output_filename", default="segmentation.nii.gz")
    parser.add_argument("--dataset_key", default="main")
    parser.add_argument("--seed_file", default=None)
    parser.add_argument("--init_seg_path", default=None, help="Optional cached init-seg h5 path")
    parser.add_argument("--axis", type=int, default=3)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--gaussian_kernel", type=int, default=5)
    parser.add_argument("--min_bright", type=int, default=40)
    parser.add_argument("--remove_portion", type=float, default=0.98)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--need_transpose", default="False")
    parser.add_argument("--max_track_distance", type=int, default=2000)

    # 联合初始化候选参数（对齐 sam2_main4_joint_init）
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
    parser.add_argument("--max_joint_avg_area", type=float, default=12000.0,
                        help="跳过平均面积过大的联合路径")

    # 可信度判别参数
    parser.add_argument("--robust_trials", type=int, default=5)
    parser.add_argument("--thr_best_energy", type=float, default=38.0)
    parser.add_argument("--thr_energy_std", type=float, default=2.2)
    parser.add_argument("--thr_energy_uniformity", type=float, default=0.35)
    parser.add_argument("--thr_robust_energy_cv", type=float, default=0.18)
    parser.add_argument("--thr_robust_path_iou", type=float, default=0.58)
    parser.add_argument("--thr_avg_area_min", type=float, default=20.0)
    parser.add_argument("--thr_avg_area_max", type=float, default=12000.0)

    parser.add_argument("--vos_offload_video_to_cpu", action="store_true")
    parser.add_argument("--keep_tmp_vos_frames", action="store_true")
    parser.add_argument("--random_seed", type=int, default=123)
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

    return {
        "path": path,
        "best_energy": best_energy,
        "per_step_energy": per_step_energy,
    }


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


def judge_seed_once(img_predictor, vol_man, seed, args, base_rng):
    crops = vol_man.get_triplane_crops(seed)
    autocast_device = str(vol_man.device) if hasattr(vol_man, "device") else args.device
    best_axis, best_mask = select_best_axis_seed_mask(
        img_predictor, crops, seed, args.max_init_mask_area, autocast_device
    )
    if best_axis == -1 or best_mask is None or int(best_mask.sum()) == 0:
        return None
    if int(best_mask.sum()) > args.max_init_mask_area:
        return None

    _, box = crops[best_axis]

    # 基线路径
    idxs, layers = build_layers_for_seed(img_predictor, vol_man, seed, best_axis, box, args, base_rng)
    if len(layers) == 0:
        return None
    baseline = dp_best_path_with_energy(layers, args)
    baseline_masks = path_masks_from_layers(layers, baseline["path"])
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
        trial_rng = np.random.default_rng(int(base_rng.integers(0, 2**31 - 1)))
        _, layers_t = build_layers_for_seed(img_predictor, vol_man, seed, best_axis, box, args, trial_rng)
        trial = dp_best_path_with_energy(layers_t, args)
        robust_energies.append(float(trial["best_energy"]))
        trial_masks = path_masks_from_layers(layers_t, trial["path"])
        robust_path_ious.append(mean_path_iou(baseline_masks, trial_masks))

    robust_energies = np.array(robust_energies, dtype=np.float64)
    robust_energy_cv = float(np.std(robust_energies) / (np.mean(robust_energies) + EPS))
    robust_path_iou = float(np.mean(robust_path_ious)) if len(robust_path_ious) > 0 else 0.0

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
        box=box,
        best_energy=float(baseline["best_energy"]),
        energy_std=energy_std,
        energy_uniformity=energy_uniformity,
        robustness_energy_cv=robust_energy_cv,
        robustness_path_iou=robust_path_iou,
        avg_area=avg_area,
        init_masks_by_global_idx=build_init_masks_dict(idxs, baseline_masks),
        is_trustworthy=bool(is_trustworthy),
    )


def vos_track_with_multi_init(
    video_predictor,
    vol_man,
    axis,
    box,
    idx_list,
    init_masks_by_global_idx,
    global_update_axis,
    vos_tmp_root,
    offload_video_to_cpu,
):
    if len(idx_list) == 0:
        return

    frames_rgb = []
    for vidx in idx_list:
        sl = get_slice(vol_man.vol, axis, vidx, box)
        if sl.size == 0:
            break
        rgb = to_uint8_rgb(sl)
        if rgb is None:
            break
        frames_rgb.append(rgb)
    if len(frames_rgb) == 0:
        return

    tmp_dir = tempfile.mkdtemp(prefix="sam2_vos_joint_judge_", dir=vos_tmp_root)
    write_jpeg_frames(frames_rgb, tmp_dir)

    try:
        with torch.inference_mode(), torch.autocast(str(vol_man.device), dtype=torch.bfloat16):
            state = video_predictor.init_state(
                video_path=tmp_dir,
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=False,
                async_loading_frames=False,
            )

            idx_to_frame = {gidx: fi for fi, gidx in enumerate(idx_list)}
            used = 0
            for gidx in idx_list:
                if gidx not in init_masks_by_global_idx:
                    continue
                m = init_masks_by_global_idx[gidx].astype(bool)
                if m.sum() == 0:
                    continue
                fidx = idx_to_frame[gidx]
                video_predictor.add_new_mask(state, frame_idx=fidx, obj_id=1, mask=m)
                vol_man.update_global_mask(m.astype(np.uint8), global_update_axis, (gidx, *box[1:]))
                used += 1

            if used == 0:
                g0 = idx_list[0]
                m0 = init_masks_by_global_idx.get(g0, None)
                if m0 is None or m0.sum() == 0:
                    return
                video_predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=m0.astype(bool))
                vol_man.update_global_mask(m0.astype(np.uint8), global_update_axis, (g0, *box[1:]))

            for f_idx, _obj_ids, masks in video_predictor.propagate_in_video(state):
                if f_idx < 0 or f_idx >= len(idx_list):
                    continue
                mm = masks[0, 0]
                if torch.is_tensor(mm):
                    mm = (mm > 0).to(torch.uint8).cpu().numpy()
                else:
                    mm = (mm > 0).astype(np.uint8)
                if mm.sum() == 0:
                    continue
                vol_man.update_global_mask(mm, global_update_axis, (idx_list[f_idx], *box[1:]))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


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


def run_segmentation_with_seed_judge():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.random_seed)

    vol_man = VolumeManager(args.volume_path, key=args.dataset_key)
    if not hasattr(vol_man, "device"):
        vol_man.device = device

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('use_sam2', version_base='1.2')

    sam2_img_model = build_sam2(args.sam2_model_cfg, args.sam2_checkpoint, device=device)
    img_predictor = SAM2ImagePredictor(sam2_img_model)
    video_predictor = build_sam2_video_predictor(args.sam2_model_cfg, args.sam2_checkpoint, device=device)

    seeds = load_or_build_seeds(args, vol_man)
    if not seeds:
        raise RuntimeError("No seeds available.")

    print(f"Total seeds: {len(seeds)}")

    vos_tmp_root = os.path.join(args.output_dir, "_tmp_sam2_vos")
    os.makedirs(vos_tmp_root, exist_ok=True)

    trust_cnt = 0
    reject_cnt = 0

    with tqdm(total=len(seeds), desc="Seed judge + VOS") as pbar:
        for seed in seeds:
            pbar.update(1)
            z, y, x = seed
            if vol_man.global_mask[z, y, x] > 0:
                continue

            result = judge_seed_once(img_predictor, vol_man, seed, args, rng)
            if result is None:
                reject_cnt += 1
                print(f"[REJECT] seed={seed} reason=invalid_init_or_path")
                continue

            if not result.is_trustworthy:
                reject_cnt += 1
                print(
                    f"[REJECT] seed={seed} axis={result.axis} "
                    f"E={result.best_energy:.3f} E_std={result.energy_std:.3f} "
                    f"U={result.energy_uniformity:.3f} RobCV={result.robustness_energy_cv:.3f} "
                    f"RobIoU={result.robustness_path_iou:.3f} Area={result.avg_area:.1f}"
                )
                continue

            trust_cnt += 1
            print(
                f"[TRUST] seed={seed} axis={result.axis} "
                f"E={result.best_energy:.3f} E_std={result.energy_std:.3f} "
                f"U={result.energy_uniformity:.3f} RobCV={result.robustness_energy_cv:.3f} "
                f"RobIoU={result.robustness_path_iou:.3f} Area={result.avg_area:.1f}"
            )

            best_axis = result.axis
            box = result.box
            init_masks = result.init_masks_by_global_idx

            track_dim = [0, 1, 2][best_axis]
            start_idx = seed[track_dim]
            max_dist = args.max_track_distance
            forward_idxs = list(range(start_idx, min(start_idx + max_dist, vol_man.shape[track_dim])))
            backward_idxs = list(range(start_idx, max(start_idx - max_dist, -1), -1))

            vos_track_with_multi_init(
                video_predictor=video_predictor,
                vol_man=vol_man,
                axis=best_axis,
                box=box,
                idx_list=forward_idxs,
                init_masks_by_global_idx=init_masks,
                global_update_axis=best_axis,
                vos_tmp_root=vos_tmp_root,
                offload_video_to_cpu=args.vos_offload_video_to_cpu,
            )
            vos_track_with_multi_init(
                video_predictor=video_predictor,
                vol_man=vol_man,
                axis=best_axis,
                box=box,
                idx_list=backward_idxs,
                init_masks_by_global_idx=init_masks,
                global_update_axis=best_axis,
                vos_tmp_root=vos_tmp_root,
                offload_video_to_cpu=args.vos_offload_video_to_cpu,
            )

    print("=" * 80)
    print(f"Trust seeds used for VOS: {trust_cnt}")
    print(f"Rejected seeds: {reject_cnt}")

    if (not args.keep_tmp_vos_frames) and os.path.isdir(vos_tmp_root):
        shutil.rmtree(vos_tmp_root, ignore_errors=True)

    final_path = os.path.join(args.output_dir, args.output_filename)
    flag = (args.need_transpose != "False")
    vol_man.save(final_path, flag)
    print(f"Done! Saved to {final_path}")


if __name__ == "__main__":
    run_segmentation_with_seed_judge()
