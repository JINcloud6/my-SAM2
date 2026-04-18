import argparse
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import hydra
import numpy as np
import torch
from tqdm import tqdm

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .data_manager import VolumeManager
from .sam2_baseline.predict_utils import map_local_point
from .sam2_hybrid.axis_cache import (
    build_all_axis_sequence_caches,
    enable_precomputed_feature_cache,
    init_state_from_axis_cache,
)
from .sam2_hybrid.common import SeedTask, load_or_build_seeds, select_mask_with_constraints
from .sam2_hybrid.joint_init import JointInitResult, evaluate_axis_joint_init
from .sam2_hybrid.tracking import mask_score_from_logits, propagate_in_video_optional_scores, siou_from_score_info


@dataclass
class PreparedSeedTask:
    task: SeedTask
    init_result: JointInitResult


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch multiple seeds that lie on the same plane into one SAM2 video state. "
            "This script focuses on speeding up the VOS stage for many original seeds by "
            "forcing a shared tracking axis and propagating multiple object prompts together."
        )
    )
    parser.add_argument(
        "--cuda_device",
        type=int,
        default=0,
        help="Physical CUDA device index to use when --device starts with 'cuda'",
    )
    parser.add_argument("--sam2_checkpoint", required=True)
    parser.add_argument("--sam2_model_cfg", required=True)
    parser.add_argument("--volume_path", required=True)
    parser.add_argument("--output_dir", default="./bv_seg_output_hybrid_multiseed")
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
    parser.add_argument(
        "--min_frame_mask_score",
        type=float,
        default=0.0,
        help=(
            "If > 0, terminate an object before writing a frame when the mean sigmoid score "
            "inside its predicted mask is below this threshold."
        ),
    )
    parser.add_argument(
        "--min_frame_siou",
        type=float,
        default=0.0,
        help=(
            "If > 0, request per-frame SAM2 decoder s_iou scores and terminate an object "
            "before writing a frame when s_iou is below this threshold."
        ),
    )

    parser.add_argument("--enable_seed_judge", action="store_true")
    parser.add_argument("--disable_joint_init", action="store_true")
    parser.add_argument("--init_half_window", type=int, default=6)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--num_point_jitters", type=int, default=3)
    parser.add_argument("--jitter_radius", type=float, default=6.0)
    parser.add_argument("--point_sample_radius", type=float, default=None)
    parser.add_argument("--joint_energy_mode", default="legacy_weighted", choices=["legacy_weighted", "unweighted_terms"])
    parser.add_argument("--joint_energy_terms", default="score,iou,centroid_radius,area_log")
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

    parser.add_argument(
        "--forced_track_axis",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Force all seeds in this run to use the same tracking axis. Seeds are grouped by the plane index on this axis.",
    )
    parser.add_argument(
        "--max_group_seeds_per_plane",
        type=int,
        default=8,
        help="Maximum number of seeds to batch together in one plane group.",
    )
    parser.add_argument(
        "--print_total_runtime",
        action="store_true",
        help="Print total end-to-end runtime when the program finishes.",
    )
    parser.add_argument(
        "--enable_segmented_seed_logging",
        action="store_true",
        help="Save actually segmented seed positions for this batched multi-seed run.",
    )
    parser.add_argument(
        "--segmented_seed_log_filename",
        default="segmented_seeds.csv",
        help="CSV filename used when --enable_segmented_seed_logging is on.",
    )
    parser.add_argument("--vos_offload_video_to_cpu", action="store_true")
    args, unknown = parser.parse_known_args()
    if len(unknown) > 0:
        print(
            "[WARN] Ignoring unsupported extra arguments for the batched multi-seed script: "
            f"{unknown}"
        )
    return args


def resolve_torch_device(args):
    if str(args.device).lower() == "cpu":
        return torch.device("cpu")
    if str(args.device).startswith("cuda:"):
        return torch.device(args.device)
    if str(args.device).startswith("cuda"):
        return torch.device(f"cuda:{args.cuda_device}")
    return torch.device(args.device)


def save_segmented_seed_records(path: str, records: List[Dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("z,y,x,seed_type,forced_axis,plane_idx,batch_size\n")
        for item in records:
            f.write(
                f"{int(item['z'])},{int(item['y'])},{int(item['x'])},"
                f"{item['seed_type']},{int(item['forced_axis'])},"
                f"{int(item['plane_idx'])},{int(item['batch_size'])}\n"
            )


def prepare_forced_axis_basic_init(
    img_predictor,
    vol_man,
    seed: Tuple[int, int, int],
    forced_axis: int,
    args,
) -> Optional[JointInitResult]:
    crops = vol_man.get_triplane_crops(seed)
    img, box = crops[forced_axis]
    local_pt = map_local_point(seed, forced_axis, box)

    with torch.inference_mode(), torch.autocast(str(vol_man.device), dtype=torch.bfloat16):
        img_predictor.set_image(img)
        masks, scores, _ = img_predictor.predict(
            point_coords=np.array([local_pt]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        mask, _ = select_mask_with_constraints(masks, scores, args.max_init_mask_area)

    if mask is None or int(mask.sum()) == 0:
        return None
    if int(mask.sum()) > args.max_init_mask_area:
        return None

    start_idx = int(seed[forced_axis])
    return JointInitResult(
        seed=seed,
        axis=forced_axis,
        box=box,
        init_masks_by_global_idx={start_idx: mask.astype(np.uint8)},
        best_energy=0.0,
        energy_std=0.0,
        energy_uniformity=0.0,
        robustness_energy_cv=0.0,
        robustness_path_iou=1.0,
        avg_area=float(mask.sum()),
        is_trustworthy=True,
    )


def prepare_forced_axis_init_result(
    task: SeedTask,
    img_predictor,
    vol_man,
    forced_axis: int,
    args,
    base_rng,
) -> Optional[JointInitResult]:
    if args.disable_joint_init:
        return prepare_forced_axis_basic_init(img_predictor, vol_man, task.seed, forced_axis, args)

    crops = vol_man.get_triplane_crops(task.seed)
    _, box = crops[forced_axis]
    axis_seed = int(base_rng.integers(0, 2**31 - 1))
    axis_rng = np.random.default_rng(axis_seed)
    return evaluate_axis_joint_init(
        img_predictor=img_predictor,
        vol_man=vol_man,
        seed=task.seed,
        axis=forced_axis,
        box=box,
        args=args,
        axis_rng=axis_rng,
    )


def split_group(items: Sequence[PreparedSeedTask], max_group_size: int) -> List[List[PreparedSeedTask]]:
    groups: List[List[PreparedSeedTask]] = []
    for start in range(0, len(items), max_group_size):
        groups.append(list(items[start : start + max_group_size]))
    return groups


def warn_unsupported_runtime_options(args) -> None:
    unsupported = []
    for flag_name in (
        "enable_respawn",
        "enable_skeleton_respawn",
        "disable_legacy_segment_trust_respawn",
        "enable_segment_classifier_labels",
        "print_segment_classifier_details",
        "disable_longterm_memory",
    ):
        if getattr(args, flag_name, False):
            unsupported.append(flag_name)
    if unsupported:
        print(
            "[WARN] This batched multi-seed script focuses on grouped original-seed VOS propagation. "
            f"The following hybrid features are not implemented here and will be ignored if passed: {unsupported}"
        )


def add_group_init_masks_to_state(
    video_predictor,
    state,
    vol_man,
    group_items: Sequence[PreparedSeedTask],
    axis: int,
    valid_gidx_set,
) -> Dict[int, PreparedSeedTask]:
    active: Dict[int, PreparedSeedTask] = {}
    for obj_id, item in enumerate(group_items, start=1):
        used = False
        box = item.init_result.box
        for gidx, mask in item.init_result.init_masks_by_global_idx.items():
            gidx = int(gidx)
            if gidx not in valid_gidx_set:
                continue
            if mask is None or int(mask.sum()) == 0:
                continue
            video_predictor.add_new_mask(state, frame_idx=gidx, obj_id=obj_id, mask=mask.astype(bool))
            vol_man.update_global_mask(mask.astype(np.uint8), axis, (gidx, *box[1:]))
            used = True
        if used:
            active[obj_id] = item
    return active


def track_seed_group_one_direction(
    video_predictor,
    vol_man,
    axis: int,
    idx_list: Sequence[int],
    group_items: Sequence[PreparedSeedTask],
    axis_sequence_cache,
    offload_video_to_cpu: bool,
    args,
) -> Dict[str, int]:
    if len(idx_list) == 0 or len(group_items) == 0:
        return {
            "tracked_objects": 0,
            "terminated_large_mask_objects": 0,
            "terminated_low_score_mask_objects": 0,
            "terminated_low_siou_mask_objects": 0,
        }

    valid_gidx = [int(vidx) for vidx in idx_list if 0 <= int(vidx) < int(axis_sequence_cache.num_frames)]
    if len(valid_gidx) == 0:
        return {
            "tracked_objects": 0,
            "terminated_large_mask_objects": 0,
            "terminated_low_score_mask_objects": 0,
            "terminated_low_siou_mask_objects": 0,
        }
    valid_gidx_set = set(valid_gidx)

    tracked_objects = 0
    terminated_large_mask_objects = 0
    terminated_low_score_mask_objects = 0
    terminated_low_siou_mask_objects = 0

    with torch.inference_mode(), torch.autocast(str(vol_man.device), dtype=torch.bfloat16):
        state = init_state_from_axis_cache(
            video_predictor=video_predictor,
            seq_cache=axis_sequence_cache,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=False,
        )

        active_objects = add_group_init_masks_to_state(
            video_predictor=video_predictor,
            state=state,
            vol_man=vol_man,
            group_items=group_items,
            axis=axis,
            valid_gidx_set=valid_gidx_set,
        )
        if len(active_objects) == 0:
            return {
                "tracked_objects": 0,
                "terminated_large_mask_objects": 0,
                "terminated_low_score_mask_objects": 0,
                "terminated_low_siou_mask_objects": 0,
            }

        tracked_objects = len(active_objects)
        terminated_obj_ids = set()
        obj_id_to_mask_index = {}
        min_frame_siou = float(getattr(args, "min_frame_siou", 0.0))
        need_siou_scores = min_frame_siou > 0.0
        for propagation_out in propagate_in_video_optional_scores(
            video_predictor,
            state,
            need_scores=need_siou_scores,
        ):
            if need_siou_scores:
                f_idx, obj_ids, masks, score_info = propagation_out
            else:
                f_idx, obj_ids, masks = propagation_out
                score_info = None
            gidx = int(f_idx)
            if gidx not in valid_gidx_set:
                continue

            obj_id_to_mask_index.clear()
            for mask_idx, obj_id in enumerate(obj_ids):
                obj_id_to_mask_index[int(obj_id)] = int(mask_idx)

            for obj_id, item in active_objects.items():
                if obj_id in terminated_obj_ids:
                    continue
                mask_idx = obj_id_to_mask_index.get(int(obj_id), None)
                if mask_idx is None:
                    continue
                mm_logits = masks[mask_idx, 0]
                if torch.is_tensor(mm_logits):
                    mm = (mm_logits > 0).to(torch.uint8).cpu().numpy()
                else:
                    mm = (mm_logits > 0).astype(np.uint8)

                if int(mm.sum()) == 0:
                    terminated_obj_ids.add(int(obj_id))
                    continue

                mask_area = int(mm.sum())
                if mask_area > int(args.max_slice_mask_area):
                    terminated_obj_ids.add(int(obj_id))
                    terminated_large_mask_objects += 1
                    continue

                if min_frame_siou > 0.0:
                    frame_siou = siou_from_score_info(score_info, obj_index=mask_idx)
                    if frame_siou is None:
                        raise RuntimeError("SAM2 propagation did not return a valid siou score.")
                    if frame_siou < min_frame_siou:
                        terminated_obj_ids.add(int(obj_id))
                        terminated_low_siou_mask_objects += 1
                        continue

                min_frame_mask_score = float(getattr(args, "min_frame_mask_score", 0.0))
                if min_frame_mask_score > 0.0:
                    frame_mask_score = mask_score_from_logits(mm_logits, mm)
                    if frame_mask_score < min_frame_mask_score:
                        terminated_obj_ids.add(int(obj_id))
                        terminated_low_score_mask_objects += 1
                        continue

                vol_man.update_global_mask(mm.astype(np.uint8), axis, (gidx, *item.init_result.box[1:]))

            if len(terminated_obj_ids) >= len(active_objects):
                break

    return {
        "tracked_objects": tracked_objects,
        "terminated_large_mask_objects": terminated_large_mask_objects,
        "terminated_low_score_mask_objects": terminated_low_score_mask_objects,
        "terminated_low_siou_mask_objects": terminated_low_siou_mask_objects,
    }


def run_segmentation():
    args = get_args()
    run_start_time = time.perf_counter()
    os.makedirs(args.output_dir, exist_ok=True)
    warn_unsupported_runtime_options(args)
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
        print(f"Done! Saved empty mask to {final_path}")
        if args.print_total_runtime:
            elapsed_sec = time.perf_counter() - run_start_time
            print(f"Total runtime: {elapsed_sec:.2f}s")
        return

    forced_axis = int(args.forced_track_axis)
    plane_to_tasks: Dict[int, List[SeedTask]] = defaultdict(list)
    for seed in initial_seeds:
        plane_to_tasks[int(seed[forced_axis])].append(SeedTask(seed=seed, is_original=True))

    total_raw_seeds = sum(len(v) for v in plane_to_tasks.values())
    segmented_count = 0
    rejected_count = 0
    skipped_covered_count = 0
    terminated_large_mask_objects = 0
    terminated_low_score_mask_objects = 0
    terminated_low_siou_mask_objects = 0
    segmented_seed_records: List[Dict[str, object]] = []

    prep_pbar = tqdm(total=total_raw_seeds, desc="Prepare grouped seeds")
    for plane_idx in sorted(plane_to_tasks.keys()):
        raw_tasks = plane_to_tasks[plane_idx]
        prepared_plane_items: List[PreparedSeedTask] = []
        for task in raw_tasks:
            prep_pbar.update(1)
            z, y, x = task.seed
            if vol_man.global_mask[z, y, x] > 0:
                skipped_covered_count += 1
                continue
            init_result = prepare_forced_axis_init_result(
                task=task,
                img_predictor=img_predictor,
                vol_man=vol_man,
                forced_axis=forced_axis,
                args=args,
                base_rng=rng,
            )
            if init_result is None or not init_result.is_trustworthy:
                rejected_count += 1
                continue
            prepared_plane_items.append(PreparedSeedTask(task=task, init_result=init_result))
        if len(prepared_plane_items) == 0:
            continue

        plane_batches = split_group(prepared_plane_items, int(args.max_group_seeds_per_plane))
        for batch_items in plane_batches:
            if segmented_count >= args.max_segmented_seeds:
                print(f"[STOP] segmented seeds reached max_segmented_seeds={args.max_segmented_seeds}")
                break

            batch_items = [item for item in batch_items if vol_man.global_mask[item.task.seed] <= 0]
            if len(batch_items) == 0:
                continue

            remaining = int(args.max_segmented_seeds) - segmented_count
            batch_items = batch_items[: max(0, remaining)]
            if len(batch_items) == 0:
                continue

            segmented_count += len(batch_items)
            batch_size = len(batch_items)
            for item in batch_items:
                segmented_seed_records.append(
                    {
                        "z": int(item.task.seed[0]),
                        "y": int(item.task.seed[1]),
                        "x": int(item.task.seed[2]),
                        "seed_type": "original",
                        "forced_axis": int(forced_axis),
                        "plane_idx": int(plane_idx),
                        "batch_size": int(batch_size),
                    }
                )

            start_idx = int(plane_idx)
            max_dist = int(args.max_track_distance)
            forward_idxs = list(range(start_idx, min(start_idx + max_dist, vol_man.shape[forced_axis])))
            backward_idxs = list(range(start_idx, max(start_idx - max_dist, -1), -1))

            stat_fw = track_seed_group_one_direction(
                video_predictor=video_predictor,
                vol_man=vol_man,
                axis=forced_axis,
                idx_list=forward_idxs,
                group_items=batch_items,
                axis_sequence_cache=axis_sequence_caches[forced_axis],
                offload_video_to_cpu=args.vos_offload_video_to_cpu,
                args=args,
            )
            stat_bw = track_seed_group_one_direction(
                video_predictor=video_predictor,
                vol_man=vol_man,
                axis=forced_axis,
                idx_list=backward_idxs,
                group_items=batch_items,
                axis_sequence_cache=axis_sequence_caches[forced_axis],
                offload_video_to_cpu=args.vos_offload_video_to_cpu,
                args=args,
            )
            terminated_large_mask_objects += int(stat_fw["terminated_large_mask_objects"])
            terminated_large_mask_objects += int(stat_bw["terminated_large_mask_objects"])
            terminated_low_score_mask_objects += int(stat_fw["terminated_low_score_mask_objects"])
            terminated_low_score_mask_objects += int(stat_bw["terminated_low_score_mask_objects"])
            terminated_low_siou_mask_objects += int(stat_fw["terminated_low_siou_mask_objects"])
            terminated_low_siou_mask_objects += int(stat_bw["terminated_low_siou_mask_objects"])

        if segmented_count >= args.max_segmented_seeds:
            break
    prep_pbar.close()

    final_path = os.path.join(args.output_dir, args.output_filename)
    flag = (args.need_transpose != "False")
    vol_man.save(final_path, flag)

    print("=" * 80)
    print("Multi-seed plane batching summary")
    print(f"Forced track axis: {forced_axis}")
    print(f"Input seeds: {len(initial_seeds)}")
    print(f"Segmented seeds actually run: {segmented_count}")
    print(f"Rejected seeds: {rejected_count}")
    print(f"Skipped already covered seeds: {skipped_covered_count}")
    print(f"Objects terminated by large mask filter: {terminated_large_mask_objects}")
    if args.min_frame_mask_score > 0.0:
        print(f"Objects terminated by low mask score: {terminated_low_score_mask_objects}")
    if args.min_frame_siou > 0.0:
        print(f"Objects terminated by low s_iou: {terminated_low_siou_mask_objects}")
    print(f"Done! Saved to {final_path}")
    if args.enable_segmented_seed_logging:
        segmented_seed_log_path = os.path.join(args.output_dir, args.segmented_seed_log_filename)
        save_segmented_seed_records(segmented_seed_log_path, segmented_seed_records)
        print(f"Segmented seed log saved to {segmented_seed_log_path}")
    if args.print_total_runtime:
        elapsed_sec = time.perf_counter() - run_start_time
        print(f"Total runtime: {elapsed_sec:.2f}s")


if __name__ == "__main__":
    run_segmentation()
