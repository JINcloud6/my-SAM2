# 混合版本：
# - 联合多切片推断初始分割
# - 可选 seed judge
# - 分割过程中 respawn 新 seed
# - long-term memory + working memory + global memory injection
import argparse
import os
import shutil
import time
import uuid
from collections import deque
from typing import Deque, Set, Tuple

import hydra
import numpy as np
import torch
from tqdm import tqdm

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .data_manager import VolumeManager
from .global_memory_pool import GlobalMemoryPool
from .sam2_hybrid.common import SeedTask, load_or_build_seeds
from .sam2_hybrid.joint_init import prepare_basic_seed_init, prepare_seed_init
from .sam2_hybrid.tracking import track_one_direction_hybrid

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2_checkpoint", required=True)
    parser.add_argument("--sam2_model_cfg", required=True)
    parser.add_argument("--volume_path", required=True)
    parser.add_argument("--output_dir", default="./bv_seg_output_hybrid")
    parser.add_argument("--output_filename", default="segmentation.nii.gz")
    parser.add_argument("--dataset_key", default="main")
    parser.add_argument("--seed_file", default=None)
    parser.add_argument("--init_seg_path", default=None)
    parser.add_argument("--axis", type=int, default=3)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--gaussian_kernel", type=int, default=5)
    parser.add_argument("--min_bright", type=int, default=40)
    parser.add_argument("--remove_portion", type=float, default=0.98)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--need_transpose", default="False")
    parser.add_argument("--max_track_distance", type=int, default=2000)
    parser.add_argument("--max_init_mask_area", type=int, default=12000)
    parser.add_argument("--max_segmented_seeds", type=int, default=200)

    parser.add_argument("--enable_seed_judge", action="store_true")
    parser.add_argument("--disable_joint_init", action="store_true")
    parser.add_argument("--init_half_window", type=int, default=6)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--num_point_jitters", type=int, default=3)
    parser.add_argument("--jitter_radius", type=float, default=6.0)
    parser.add_argument("--w_score", type=float, default=1.0)
    parser.add_argument("--w_iou", type=float, default=3.0)
    parser.add_argument("--w_centroid", type=float, default=0.03)
    parser.add_argument("--w_area", type=float, default=0.6)
    parser.add_argument("--empty_mask_penalty", type=float, default=6.0)
    parser.add_argument("--max_joint_avg_area", type=float, default=12000.0)
    parser.add_argument("--robust_trials", type=int, default=5)
    parser.add_argument("--thr_best_energy", type=float, default=38.0)
    parser.add_argument("--thr_energy_std", type=float, default=2.2)
    parser.add_argument("--thr_energy_uniformity", type=float, default=0.35)
    parser.add_argument("--thr_robust_energy_cv", type=float, default=0.18)
    parser.add_argument("--thr_robust_path_iou", type=float, default=0.58)
    parser.add_argument("--thr_avg_area_min", type=float, default=20.0)
    parser.add_argument("--thr_avg_area_max", type=float, default=12000.0)
    parser.add_argument("--random_seed", type=int, default=123)

    parser.add_argument("--enable_respawn", action="store_true")
    parser.add_argument("--disable_longterm_memory", action="store_true")
    parser.add_argument("--segment_len", type=int, default=15)
    parser.add_argument("--min_segment_frames", type=int, default=5)
    parser.add_argument("--axis_ratio_thr", type=float, default=1.1)
    parser.add_argument("--segment_respawn_num_seeds", type=int, default=4)
    parser.add_argument("--segment_respawn_min_distance", type=float, default=20.0)
    parser.add_argument("--max_segment_respawn_candidates", type=int, default=2048)
    parser.add_argument("--max_untrusted_segments_per_lineage", type=int, default=1)
    parser.add_argument("--min_respawn_mask_area", type=int, default=20)
    parser.add_argument("--respawn_num_seeds", type=int, default=2)
    parser.add_argument("--respawn_min_point_distance", type=float, default=24.0)
    parser.add_argument("--max_respawn_candidates", type=int, default=512)

    parser.add_argument("--longterm_segment_len", type=int, default=12)
    parser.add_argument("--longterm_quality_thr", type=float, default=0.8)
    parser.add_argument("--max_longterm_segments", type=int, default=5)
    parser.add_argument("--working_window", type=int, default=24)
    parser.add_argument("--max_global_inject_per_seed", type=int, default=0)

    parser.add_argument("--vos_offload_video_to_cpu", action="store_true")
    parser.add_argument("--keep_tmp_vos_frames", action="store_true")
    return parser.parse_args()


def run_segmentation():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.random_seed)

    vol_man = VolumeManager(args.volume_path, key=args.dataset_key)
    if not hasattr(vol_man, "device"):
        vol_man.device = device

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module("use_sam2", version_base="1.2")

    sam2_img_model = build_sam2(args.sam2_model_cfg, args.sam2_checkpoint, device=device)
    img_predictor = SAM2ImagePredictor(sam2_img_model)
    video_predictor = build_sam2_video_predictor(args.sam2_model_cfg, args.sam2_checkpoint, device=device)

    initial_seeds = load_or_build_seeds(args, vol_man)
    if not initial_seeds:
        raise RuntimeError("No seeds available for SAM2 tracking.")

    pending: Deque[SeedTask] = deque([SeedTask(seed=s, is_original=True) for s in initial_seeds])
    seen: Set[Tuple[int, int, int]] = set(initial_seeds)
    global_memory_pool = GlobalMemoryPool()
    covered_mask = np.zeros_like(vol_man.vol, dtype=np.uint8)
    trusted_mask = np.zeros_like(vol_man.vol, dtype=np.uint8)

    run_tag = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    vos_tmp_root = os.path.join(args.output_dir, f"_tmp_sam2_hybrid_{run_tag}")
    os.makedirs(vos_tmp_root, exist_ok=True)

    segmented_count = 0
    rejected_count = 0
    respawned_count = 0
    promoted_longterm = 0
    trusted_segment_count = 0
    untrusted_segment_count = 0

    pbar = tqdm(total=len(pending), desc="Tracking (SAM2 hybrid)")
    while len(pending) > 0:
        task = pending.popleft()
        pbar.update(1)

        if segmented_count >= args.max_segmented_seeds:
            print(f"[STOP] segmented seeds reached max_segmented_seeds={args.max_segmented_seeds}")
            break

        seed = task.seed
        z, y, x = seed
        if args.enable_respawn:
            if task.is_original:
                if covered_mask[z, y, x] > 0:
                    continue
            else:
                if trusted_mask[z, y, x] > 0:
                    continue
        else:
            if vol_man.global_mask[z, y, x] > 0:
                continue

        if args.disable_joint_init:
            init_result = prepare_basic_seed_init(img_predictor, vol_man, seed, args)
        else:
            init_result = prepare_seed_init(img_predictor, vol_man, seed, args, rng)
        if init_result is None:
            rejected_count += 1
            continue
        if not init_result.is_trustworthy:
            rejected_count += 1
            continue

        segmented_count += 1
        track_dim = [0, 1, 2][init_result.axis]
        start_idx = seed[track_dim]
        max_dist = args.max_track_distance
        forward_idxs = list(range(start_idx, min(start_idx + max_dist, vol_man.shape[track_dim])))
        backward_idxs = list(range(start_idx, max(start_idx - max_dist, -1), -1))

        new_fw, stat_fw = track_one_direction_hybrid(
            video_predictor=video_predictor,
            vol_man=vol_man,
            axis=init_result.axis,
            box=init_result.box,
            idx_list=forward_idxs,
            init_masks_by_global_idx=init_result.init_masks_by_global_idx,
            vos_tmp_root=vos_tmp_root,
            offload_video_to_cpu=args.vos_offload_video_to_cpu,
            global_memory_pool=global_memory_pool,
            seed=seed,
            direction="forward",
            covered_mask=covered_mask,
            trusted_mask=trusted_mask,
            start_untrusted_count=task.untrusted_count,
            args=args,
        )
        new_bw, stat_bw = track_one_direction_hybrid(
            video_predictor=video_predictor,
            vol_man=vol_man,
            axis=init_result.axis,
            box=init_result.box,
            idx_list=backward_idxs,
            init_masks_by_global_idx=init_result.init_masks_by_global_idx,
            vos_tmp_root=vos_tmp_root,
            offload_video_to_cpu=args.vos_offload_video_to_cpu,
            global_memory_pool=global_memory_pool,
            seed=seed,
            direction="backward",
            covered_mask=covered_mask,
            trusted_mask=trusted_mask,
            start_untrusted_count=int(stat_fw["final_untrusted_count"]),
            args=args,
        )

        promoted_longterm += int(stat_fw["longterm_segments"]) + int(stat_bw["longterm_segments"])
        trusted_segment_count += int(stat_fw["trusted_segments"]) + int(stat_bw["trusted_segments"])
        untrusted_segment_count += int(stat_fw["untrusted_segments"]) + int(stat_bw["untrusted_segments"])

        for child in (new_fw + new_bw):
            if child.seed in seen:
                continue
            if args.enable_respawn:
                if trusted_mask[child.seed] > 0:
                    continue
            elif vol_man.global_mask[child.seed] > 0:
                continue
            seen.add(child.seed)
            pending.append(child)
            respawned_count += 1
            pbar.total += 1
            pbar.refresh()

    pbar.close()

    if (not args.keep_tmp_vos_frames) and os.path.isdir(vos_tmp_root):
        shutil.rmtree(vos_tmp_root, ignore_errors=True)

    final_path = os.path.join(args.output_dir, args.output_filename)
    flag = (args.need_transpose != "False")
    vol_man.save(final_path, flag)

    print("=" * 80)
    print(f"Segmented seeds actually run: {segmented_count}")
    print(f"Rejected seeds: {rejected_count}")
    print(f"Respawned seeds queued: {respawned_count}")
    print(f"Promoted long-term segments: {promoted_longterm}")
    if args.enable_respawn:
        print(f"Trusted segments: {trusted_segment_count}")
        print(f"Untrusted segments: {untrusted_segment_count}")
    print(f"Done! Saved to {final_path}")


if __name__ == "__main__":
    run_segmentation()
