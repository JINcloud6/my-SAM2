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
from typing import Deque, Dict, List, Optional, Set, Tuple

import hydra
import numpy as np
import torch
from tqdm import tqdm

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .data_manager import VolumeManager
from .global_memory_pool import GlobalMemoryPool
from .sam2_hybrid.axis_cache import build_all_axis_sequence_caches, enable_precomputed_feature_cache
from .sam2_hybrid.common import SeedTask, load_or_build_seeds
from .sam2_hybrid.joint_init import prepare_basic_seed_init, prepare_seed_init
from .sam2_hybrid.tracking import track_one_direction_hybrid
from .sam2_hybrid.unified_segment_classifier import save_segment_label_volume


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
    parser.add_argument(
        "--joint_init_axis_mode",
        default="preselect",
        choices=["preselect", "joint_energy"],
        help=(
            "How to choose the axis for MSJI. 'preselect' first chooses one axis using "
            "single-slice SAM initialization and then runs MSJI only on that axis. "
            "'joint_energy' runs MSJI on all three axes and picks the lowest-energy path."
        ),
    )
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
    parser.add_argument(
        "--segment_len_diameter_multiplier",
        type=float,
        default=0.0,
        help="If > 0, use ceil(first-frame mask diameter * this multiplier) as the current segment decision length; otherwise use fixed --segment_len.",
    )
    parser.add_argument("--min_segment_frames", type=int, default=5)
    parser.add_argument("--axis_ratio_thr", type=float, default=1.1)
    parser.add_argument("--segment_respawn_num_seeds", type=int, default=4)
    parser.add_argument("--segment_respawn_min_distance", type=float, default=20.0)
    parser.add_argument("--max_segment_respawn_candidates", type=int, default=2048)
    parser.add_argument(
        "--enable_skeleton_respawn",
        action="store_true",
        help="For segments already classified as complex, use skeleton endpoint extrapolation instead of the legacy FPS respawn.",
    )
    parser.add_argument(
        "--disable_legacy_segment_trust_respawn",
        action="store_true",
        help="Disable legacy direction-based trust/untrust judgment and legacy FPS respawn; use unified segment classification instead.",
    )
    parser.add_argument(
        "--skeleton_respawn_offset",
        type=float,
        default=12.0,
        help="Distance to extend outward from a local 3D skeleton endpoint when proposing skeleton-based respawn seeds.",
    )
    parser.add_argument(
        "--max_skeleton_respawn_seeds",
        type=int,
        default=4,
        help="Maximum number of skeleton-based respawn seeds to return for one complex segment.",
    )
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
    parser.add_argument("--enable_segment_classifier_labels", action="store_true")
    parser.add_argument("--segment_label_output_filename", default="mousep4_segment_class_labels.nii.gz")
    parser.add_argument("--segment_classifier_stable_quality_thr", type=float, default=0.80)
    parser.add_argument("--segment_classifier_stable_iou_thr", type=float, default=0.65)
    parser.add_argument("--segment_classifier_stable_empty_rate_thr", type=float, default=0.05)
    parser.add_argument("--segment_classifier_stable_axis_ratio_thr", type=float, default=1.10)
    parser.add_argument(
        "--segment_classifier_boundary_tail_max_frames",
        type=int,
        default=6,
        help="If a non-failure segment ends at the dataset boundary within this many frames, classify it as stable.",
    )
    parser.add_argument(
        "--segment_classifier_complex_tube_score_thr",
        type=float,
        default=2.5,
        help="Rescue an initially complex segment to stable if its PCA tube score lambda1/(lambda2+lambda3+eps) reaches this value.",
    )
    parser.add_argument(
        "--disable_segment_classifier_dominant_axis_check",
        action="store_true",
        help="Do not require dominant_axis == track_axis when deciding whether a segment is stable.",
    )
    parser.add_argument("--segment_classifier_failure_quality_thr", type=float, default=0.55)
    parser.add_argument("--segment_classifier_failure_iou_thr", type=float, default=0.20)
    parser.add_argument("--segment_classifier_failure_empty_rate_thr", type=float, default=0.25)
    parser.add_argument("--segment_classifier_failure_min_frames", type=int, default=3)
    parser.add_argument(
        "--print_segment_classifier_details",
        action="store_true",
        help="Print per-segment stable rule violations with actual metric values and thresholds.",
    )
    parser.add_argument(
        "--enable_segmented_seed_logging",
        action="store_true",
        help="Print how many actually segmented seeds are original vs respawned, and save all actually segmented seed positions with their type.",
    )
    parser.add_argument(
        "--segmented_seed_log_filename",
        default="segmented_seeds.csv",
        help="CSV filename used when --enable_segmented_seed_logging is on.",
    )
    parser.add_argument(
        "--print_total_runtime",
        action="store_true",
        help="Print total end-to-end segmentation runtime when the program finishes.",
    )

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


def prepare_init_result(task: SeedTask, img_predictor, vol_man, args, rng) -> Optional[object]:
    if args.disable_joint_init:
        return prepare_basic_seed_init(img_predictor, vol_man, task.seed, args)
    return prepare_seed_init(img_predictor, vol_man, task.seed, args, rng)


def save_segmented_seed_records(path: str, records: List[Dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("z,y,x,seed_type,untrusted_count,selected_axis\n")
        for item in records:
            f.write(
                f"{int(item['z'])},{int(item['y'])},{int(item['x'])},"
                f"{item['seed_type']},{int(item['untrusted_count'])},{int(item['selected_axis'])}\n"
            )


def run_segmentation():
    args = get_args()
    run_start_time = time.perf_counter()
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
        if args.enable_segmented_seed_logging:
            segmented_seed_log_path = os.path.join(args.output_dir, args.segmented_seed_log_filename)
            save_segmented_seed_records(segmented_seed_log_path, [])
        if args.enable_segment_classifier_labels:
            empty_label_mask = np.zeros_like(vol_man.vol, dtype=np.uint8)
            segment_label_path = os.path.join(args.output_dir, args.segment_label_output_filename)
            save_segment_label_volume(
                segment_label_path,
                empty_label_mask,
                vol_man.affine,
                flag,
            )
        print(f"Done! Saved empty mask to {final_path}")
        if args.print_total_runtime:
            elapsed_sec = time.perf_counter() - run_start_time
            print(f"Total runtime: {elapsed_sec:.2f}s")
        return

    pending: Deque[SeedTask] = deque([SeedTask(seed=s, is_original=True) for s in initial_seeds])
    seen: Set[Tuple[int, int, int]] = set(initial_seeds)
    global_memory_pool = GlobalMemoryPool()
    covered_mask = np.zeros_like(vol_man.vol, dtype=np.uint8)
    trusted_mask = np.zeros_like(vol_man.vol, dtype=np.uint8)
    segment_label_mask = (
        np.zeros_like(vol_man.vol, dtype=np.uint8) if args.enable_segment_classifier_labels else None
    )

    run_tag = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    vos_tmp_root = os.path.join(args.output_dir, f"_tmp_sam2_hybrid_{run_tag}")
    os.makedirs(vos_tmp_root, exist_ok=True)

    segmented_count = 0
    rejected_count = 0
    respawned_count = 0
    promoted_longterm = 0
    trusted_segment_count = 0
    untrusted_segment_count = 0
    stable_segment_count = 0
    complex_segment_count = 0
    failure_segment_count = 0
    complex_reason_counts: Dict[str, int] = {}
    complex_reason_combo_counts: Dict[str, int] = {}
    segmented_original_count = 0
    segmented_respawn_count = 0
    segmented_seed_records: List[Dict[str, object]] = []

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

        init_result = prepare_init_result(task, img_predictor, vol_man, args, rng)
        if init_result is None:
            rejected_count += 1
            continue
        if not init_result.is_trustworthy:
            rejected_count += 1
            continue

        segmented_count += 1
        if task.is_original:
            segmented_original_count += 1
            seed_type = "original"
        else:
            segmented_respawn_count += 1
            seed_type = "respawn"
        if args.enable_segmented_seed_logging:
            segmented_seed_records.append(
                {
                    "z": int(seed[0]),
                    "y": int(seed[1]),
                    "x": int(seed[2]),
                    "seed_type": seed_type,
                    "untrusted_count": int(task.untrusted_count),
                    "selected_axis": int(init_result.axis),
                }
            )
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
            axis_sequence_cache=axis_sequence_caches[init_result.axis],
            seed=seed,
            direction="forward",
            covered_mask=covered_mask,
            trusted_mask=trusted_mask,
            segment_label_mask=segment_label_mask,
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
            axis_sequence_cache=axis_sequence_caches[init_result.axis],
            seed=seed,
            direction="backward",
            covered_mask=covered_mask,
            trusted_mask=trusted_mask,
            segment_label_mask=segment_label_mask,
            start_untrusted_count=int(stat_fw["final_untrusted_count"]),
            args=args,
        )

        promoted_longterm += int(stat_fw["longterm_segments"]) + int(stat_bw["longterm_segments"])
        trusted_segment_count += int(stat_fw["trusted_segments"]) + int(stat_bw["trusted_segments"])
        untrusted_segment_count += int(stat_fw["untrusted_segments"]) + int(stat_bw["untrusted_segments"])
        stable_segment_count += int(stat_fw["stable_segments"]) + int(stat_bw["stable_segments"])
        complex_segment_count += int(stat_fw["complex_segments"]) + int(stat_bw["complex_segments"])
        failure_segment_count += int(stat_fw["failure_segments"]) + int(stat_bw["failure_segments"])
        for key, value in stat_fw["complex_reason_counts"].items():
            complex_reason_counts[key] = int(complex_reason_counts.get(key, 0)) + int(value)
        for key, value in stat_bw["complex_reason_counts"].items():
            complex_reason_counts[key] = int(complex_reason_counts.get(key, 0)) + int(value)
        for key, value in stat_fw["complex_reason_combo_counts"].items():
            complex_reason_combo_counts[key] = int(complex_reason_combo_counts.get(key, 0)) + int(value)
        for key, value in stat_bw["complex_reason_combo_counts"].items():
            complex_reason_combo_counts[key] = int(complex_reason_combo_counts.get(key, 0)) + int(value)

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
    if args.enable_segment_classifier_labels and segment_label_mask is not None:
        segment_label_path = os.path.join(args.output_dir, args.segment_label_output_filename)
        saved_label_path = save_segment_label_volume(
            segment_label_path,
            segment_label_mask,
            vol_man.affine,
            flag,
        )
    else:
        saved_label_path = None

    print("=" * 80)
    print(f"Segmented seeds actually run: {segmented_count}")
    if args.enable_segmented_seed_logging:
        print(f"Segmented original seeds: {segmented_original_count}")
        print(f"Segmented respawn seeds: {segmented_respawn_count}")
    print(f"Rejected seeds: {rejected_count}")
    print(f"Respawned seeds queued: {respawned_count}")
    print(f"Promoted long-term segments: {promoted_longterm}")
    if args.enable_respawn:
        print(f"Trusted segments: {trusted_segment_count}")
        print(f"Untrusted segments: {untrusted_segment_count}")
    if args.enable_segment_classifier_labels:
        print(f"Stable segments: {stable_segment_count}")
        print(f"Complex segments: {complex_segment_count}")
        print(f"Failure segments: {failure_segment_count}")
        if complex_segment_count > 0:
            print("Complex segment reasons:")
            for key, value in sorted(complex_reason_counts.items(), key=lambda x: (-x[1], x[0])):
                print(f"  {key}: {value}")
            print("Complex segment reason combos:")
            for key, value in sorted(complex_reason_combo_counts.items(), key=lambda x: (-x[1], x[0])):
                print(f"  {key}: {value}")
    print(f"Done! Saved to {final_path}")
    if saved_label_path is not None:
        print(f"Segment class labels saved to {saved_label_path}")
    if args.enable_segmented_seed_logging:
        segmented_seed_log_path = os.path.join(args.output_dir, args.segmented_seed_log_filename)
        save_segmented_seed_records(segmented_seed_log_path, segmented_seed_records)
        print(f"Segmented seed log saved to {segmented_seed_log_path}")
    if args.print_total_runtime:
        elapsed_sec = time.perf_counter() - run_start_time
        print(f"Total runtime: {elapsed_sec:.2f}s")


if __name__ == "__main__":
    run_segmentation()
