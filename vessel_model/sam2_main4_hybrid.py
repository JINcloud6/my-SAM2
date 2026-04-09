# 混合版本：
# - 联合多切片推断初始分割
# - 可选 seed judge
# - 分割过程中 respawn 新 seed
# - long-term memory + working memory + global memory injection
import argparse
import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Set, Tuple

import hydra
import numpy as np
import torch
from tqdm import tqdm

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .data_manager import VolumeManager
from .global_memory_pool import GlobalMemoryPool
from .sam2_hybrid.axis_cache import (
    build_all_axis_sequence_caches,
    enable_precomputed_feature_cache,
    prepare_axis_sequence_cache,
    release_axis_sequence_cache,
)
from .sam2_hybrid.common import SeedTask, load_or_build_seeds
from .sam2_hybrid.joint_init import prepare_basic_seed_init, prepare_seed_init
from .sam2_hybrid.tracking import track_one_direction_hybrid


@dataclass
class PreparedSeedTask:
    task: SeedTask
    init_result: object


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda_device",
        type=int,
        default=0,
        help="Physical CUDA device index to use when --device starts with 'cuda'",
    )
    parser.add_argument("--sam2_checkpoint", required=True)
    parser.add_argument("--sam2_model_cfg", required=True)
    parser.add_argument("--volume_path", required=True)
    parser.add_argument("--output_dir", default="./bv_seg_output_hybrid")
    parser.add_argument("--output_filename", default="segmentation.nii.gz")
    parser.add_argument("--axis_sequence_cache_root", default=None)
    parser.add_argument(
        "--sequential_axis_cache",
        action="store_true",
        help="Only keep one axis image/feature cache in memory at a time.",
    )
    parser.add_argument("--enable_axis_feature_cache", action="store_true")
    parser.add_argument("--feature_cache_device", default="cuda", choices=["cpu", "cuda"])
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
    parser.add_argument("--max_slice_mask_area", type=int, default=12000)
    parser.add_argument("--max_slice_mask_ratio", type=float, default=0.45)

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


def resolve_torch_device(args):
    if str(args.device).lower() == "cpu":
        return torch.device("cpu")
    if str(args.device).startswith("cuda:"):
        return torch.device(args.device)
    if str(args.device).startswith("cuda"):
        return torch.device(f"cuda:{args.cuda_device}")
    return torch.device(args.device)


def seed_should_skip(task, args, vol_man, covered_mask, trusted_mask) -> bool:
    z, y, x = task.seed
    if args.enable_respawn:
        if task.is_original:
            return bool(covered_mask[z, y, x] > 0)
        return bool(trusted_mask[z, y, x] > 0)
    return bool(vol_man.global_mask[z, y, x] > 0)


def prepare_seed_task(task, img_predictor, vol_man, args, rng) -> Optional[PreparedSeedTask]:
    if args.disable_joint_init:
        init_result = prepare_basic_seed_init(img_predictor, vol_man, task.seed, args)
    else:
        init_result = prepare_seed_init(img_predictor, vol_man, task.seed, args, rng)
    if init_result is None or (not init_result.is_trustworthy):
        return None
    return PreparedSeedTask(task=task, init_result=init_result)


def run_segmentation():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = resolve_torch_device(args)
    rng = np.random.default_rng(args.random_seed)

    vol_man = VolumeManager(args.volume_path, key=args.dataset_key)
    if not hasattr(vol_man, "device"):
        vol_man.device = device

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module("use_sam2", version_base="1.2")

    sam2_img_model = build_sam2(args.sam2_model_cfg, args.sam2_checkpoint, device=device)
    img_predictor = SAM2ImagePredictor(sam2_img_model)
    video_predictor = build_sam2_video_predictor(args.sam2_model_cfg, args.sam2_checkpoint, device=device)
    enable_precomputed_feature_cache(video_predictor)
    axis_cache_root = args.axis_sequence_cache_root or os.path.join(args.output_dir, "_axis_sequence_cache")

    axis_sequence_caches: Optional[Dict[int, object]] = None
    if not args.sequential_axis_cache:
        axis_sequence_caches = build_all_axis_sequence_caches(
            volume=vol_man.vol,
            cache_root=axis_cache_root,
            image_size=video_predictor.image_size,
            offload_video_to_cpu=args.vos_offload_video_to_cpu,
            compute_device=device,
            video_predictor=video_predictor,
            enable_feature_cache=args.enable_axis_feature_cache,
            feature_cache_device=args.feature_cache_device,
        )

    initial_seeds = load_or_build_seeds(args, vol_man)
    if not initial_seeds:
        final_path = os.path.join(args.output_dir, args.output_filename)
        flag = (args.need_transpose != "False")
        print("No seeds available for SAM2 tracking. Saving empty mask.")
        vol_man.save(final_path, flag)
        print(f"Done! Saved empty mask to {final_path}")
        return

    pending: Deque[SeedTask] = deque([SeedTask(seed=s, is_original=True) for s in initial_seeds])
    prepared_by_axis = {0: deque(), 1: deque(), 2: deque()}
    seen: Set[Tuple[int, int, int]] = set(initial_seeds)
    global_memory_pool = GlobalMemoryPool()
    covered_mask = np.zeros_like(vol_man.vol, dtype=np.uint8)
    trusted_mask = np.zeros_like(vol_man.vol, dtype=np.uint8)

    segmented_count = 0
    rejected_count = 0
    respawned_count = 0
    promoted_longterm = 0
    trusted_segment_count = 0
    untrusted_segment_count = 0
    stop_printed = False

    pbar = tqdm(total=len(pending), desc="Tracking (SAM2 hybrid)")
    while (len(pending) > 0) or any(len(q) > 0 for q in prepared_by_axis.values()):
        while (len(pending) > 0) and (segmented_count < args.max_segmented_seeds):
            task = pending.popleft()
            pbar.update(1)

            if seed_should_skip(task, args, vol_man, covered_mask, trusted_mask):
                continue

            prepared = prepare_seed_task(task, img_predictor, vol_man, args, rng)
            if prepared is None:
                rejected_count += 1
                continue

            segmented_count += 1
            prepared_by_axis[int(prepared.init_result.axis)].append(prepared)

        if (segmented_count >= args.max_segmented_seeds) and (not stop_printed):
            print(f"[STOP] segmented seeds reached max_segmented_seeds={args.max_segmented_seeds}")
            stop_printed = True

        processed_axis_this_round = False
        for axis in (0, 1, 2):
            axis_queue = prepared_by_axis[axis]
            if len(axis_queue) == 0:
                continue
            processed_axis_this_round = True

            if args.sequential_axis_cache:
                axis_sequence_cache = prepare_axis_sequence_cache(
                    volume=vol_man.vol,
                    axis=axis,
                    cache_root=axis_cache_root,
                    image_size=video_predictor.image_size,
                    offload_video_to_cpu=args.vos_offload_video_to_cpu,
                    compute_device=device,
                    video_predictor=video_predictor,
                    enable_feature_cache=args.enable_axis_feature_cache,
                    feature_cache_device=args.feature_cache_device,
                )
            else:
                axis_sequence_cache = axis_sequence_caches[axis]

            try:
                while len(axis_queue) > 0:
                    prepared = axis_queue.popleft()
                    task = prepared.task
                    init_result = prepared.init_result
                    seed = task.seed

                    if seed_should_skip(task, args, vol_man, covered_mask, trusted_mask):
                        continue

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
                        offload_video_to_cpu=args.vos_offload_video_to_cpu,
                        global_memory_pool=global_memory_pool,
                        axis_sequence_cache=axis_sequence_cache,
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
                        offload_video_to_cpu=args.vos_offload_video_to_cpu,
                        global_memory_pool=global_memory_pool,
                        axis_sequence_cache=axis_sequence_cache,
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
            finally:
                if args.sequential_axis_cache:
                    release_axis_sequence_cache(axis_sequence_cache)

        if (not processed_axis_this_round) and (segmented_count >= args.max_segmented_seeds):
            break

    pbar.close()

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
