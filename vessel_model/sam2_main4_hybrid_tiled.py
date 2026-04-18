import argparse
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np

from .data_manager import VolumeManager


@dataclass(frozen=True)
class ChunkSpec:
    chunk_id: int
    z0: int
    z1: int
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.z1 - self.z0, self.y1 - self.y0, self.x1 - self.x0)

    @property
    def tag(self) -> str:
        return (
            f"chunk_{self.chunk_id:04d}"
            f"_z{self.z0}_{self.z1}"
            f"_y{self.y0}_{self.y1}"
            f"_x{self.x0}_{self.x1}"
        )


def parse_seed_line(line: str) -> Optional[Tuple[int, int, int]]:
    line = line.strip()
    if not line:
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Invalid seed line: {line}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def load_seed_file(path: str) -> List[Tuple[int, int, int]]:
    seeds: List[Tuple[int, int, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            seed = parse_seed_line(line)
            if seed is not None:
                seeds.append(seed)
    return seeds


def save_seed_file(path: str, seeds: Sequence[Tuple[int, int, int]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for z, y, x in seeds:
            f.write(f"{z},{y},{x}\n")


def build_chunks(shape: Tuple[int, int, int], chunk_size: int) -> List[ChunkSpec]:
    chunks: List[ChunkSpec] = []
    chunk_id = 0
    for z0 in range(0, shape[0], chunk_size):
        z1 = min(z0 + chunk_size, shape[0])
        for y0 in range(0, shape[1], chunk_size):
            y1 = min(y0 + chunk_size, shape[1])
            for x0 in range(0, shape[2], chunk_size):
                x1 = min(x0 + chunk_size, shape[2])
                chunks.append(ChunkSpec(chunk_id, z0, z1, y0, y1, x0, x1))
                chunk_id += 1
    return chunks


def save_nifti(path: str, vol: np.ndarray, affine: Optional[np.ndarray], need_transpose: bool) -> None:
    data = vol.astype(np.uint8)
    if need_transpose:
        data = np.transpose(data, (2, 1, 0))
    if affine is None:
        affine = np.eye(4)
    nib.save(nib.Nifti1Image(data, affine), path)


def slice_volume(vol: np.ndarray, chunk: ChunkSpec) -> np.ndarray:
    return vol[chunk.z0:chunk.z1, chunk.y0:chunk.y1, chunk.x0:chunk.x1]


def filter_local_seeds(
    seeds: Sequence[Tuple[int, int, int]], chunk: ChunkSpec
) -> List[Tuple[int, int, int]]:
    local: List[Tuple[int, int, int]] = []
    for z, y, x in seeds:
        if chunk.z0 <= z < chunk.z1 and chunk.y0 <= y < chunk.y1 and chunk.x0 <= x < chunk.x1:
            local.append((z - chunk.z0, y - chunk.y0, x - chunk.x0))
    return local


def load_mask(path: str) -> np.ndarray:
    return (nib.load(path).get_fdata() > 0).astype(np.uint8)


def default_init_seg_basename(args) -> str:
    return (
        f"init_seg_axis{args.axis}"
        f"_s{args.stride}"
        f"_t{args.remove_portion}"
        f"_g{args.gaussian_kernel}.h5"
    )


def shared_auto_init_seg_basename(args) -> str:
    return (
        f"init_seg_axis{args.axis}"
        f"_s{args.stride}"
        f"_t{args.remove_portion}"
        f"_g{args.gaussian_kernel}"
        f"_mb{args.min_bright}.h5"
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2_checkpoint", required=True)
    parser.add_argument("--sam2_model_cfg", required=True)
    parser.add_argument("--volume_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_filename", default="segmentation_merged.nii.gz")
    parser.add_argument("--dataset_key", default="main")
    parser.add_argument("--seed_file", default=None)
    parser.add_argument("--init_seg_path", default=None)
    parser.add_argument("--need_transpose", default="False")
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--axis", type=int, default=3)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--gaussian_kernel", type=int, default=5)
    parser.add_argument("--min_bright", type=int, default=40)
    parser.add_argument("--remove_portion", type=float, default=0.98)
    parser.add_argument("--max_track_distance", type=int, default=2000)
    parser.add_argument("--max_init_mask_area", type=int, default=12000)
    parser.add_argument("--max_segmented_seeds", type=int, default=200)
    parser.add_argument("--max_slice_mask_area", type=int, default=12000)
    parser.add_argument("--max_slice_mask_ratio", type=float, default=0.45)
    parser.add_argument("--min_frame_mask_score", type=float, default=0.0)
    parser.add_argument("--min_frame_siou", type=float, default=0.0)
    parser.add_argument("--enable_seed_judge", action="store_true")
    parser.add_argument("--disable_joint_init", action="store_true")
    parser.add_argument(
        "--joint_init_axis_mode",
        default="preselect",
        choices=["preselect", "joint_energy"],
    )
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
    parser.add_argument("--enable_respawn", action="store_true")
    parser.add_argument("--disable_longterm_memory", action="store_true")
    parser.add_argument("--segment_len", type=int, default=15)
    parser.add_argument("--segment_len_diameter_multiplier", type=float, default=0.0)
    parser.add_argument("--min_segment_frames", type=int, default=5)
    parser.add_argument("--axis_ratio_thr", type=float, default=1.1)
    parser.add_argument("--segment_respawn_num_seeds", type=int, default=4)
    parser.add_argument("--segment_respawn_min_distance", type=float, default=20.0)
    parser.add_argument("--max_segment_respawn_candidates", type=int, default=2048)
    parser.add_argument("--enable_skeleton_respawn", action="store_true")
    parser.add_argument("--disable_legacy_segment_trust_respawn", action="store_true")
    parser.add_argument("--skeleton_respawn_offset", type=float, default=12.0)
    parser.add_argument("--max_skeleton_respawn_seeds", type=int, default=4)
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
    parser.add_argument("--segment_classifier_boundary_tail_max_frames", type=int, default=6)
    parser.add_argument("--segment_classifier_complex_tube_score_thr", type=float, default=2.5)
    parser.add_argument("--disable_segment_classifier_dominant_axis_check", action="store_true")
    parser.add_argument("--segment_classifier_failure_quality_thr", type=float, default=0.55)
    parser.add_argument("--segment_classifier_failure_iou_thr", type=float, default=0.20)
    parser.add_argument("--segment_classifier_failure_empty_rate_thr", type=float, default=0.25)
    parser.add_argument("--segment_classifier_failure_min_frames", type=int, default=3)
    parser.add_argument("--print_segment_classifier_details", action="store_true")
    parser.add_argument("--enable_segmented_seed_logging", action="store_true")
    parser.add_argument("--segmented_seed_log_filename", default="segmented_seeds.csv")
    parser.add_argument("--print_total_runtime", action="store_true")
    parser.add_argument("--vos_offload_video_to_cpu", action="store_true")
    parser.add_argument("--keep_tmp_vos_frames", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--chunks_subdir", default="chunks")
    parser.add_argument("--merged_subdir", default="merged")
    parser.add_argument("--enable_axis_feature_cache", action="store_true")
    parser.add_argument("--feature_cache_device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument(
        "--shared_cache_root",
        default=None,
        help="Shared cache root for chunk volumes and init_seg files. Defaults to <output_dir>/shared_chunk_cache.",
    )
    parser.add_argument(
        "--run_prefix",
        default=None,
        help="Optional unique prefix for one tiled run. If omitted, an automatic unique prefix is generated.",
    )
    parser.add_argument("--keep_chunk_volumes", action="store_true")
    parser.add_argument("--keep_chunk_init_seg", action="store_true")
    parser.add_argument("--skip_existing_chunks", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_known_args()


def append_value_arg(cmd: List[str], flag: str, value) -> None:
    cmd.extend([flag, str(value)])


def append_flag_arg(cmd: List[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def build_child_cmd(
    args,
    passthrough_args: Sequence[str],
    chunk_volume_path: str,
    chunk_output_dir: str,
    chunk_output_filename: str,
    chunk_seed_path: Optional[str],
    chunk_init_seg_path: Optional[str],
    axis_sequence_cache_root: str,
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "vessel_model.sam2_main4_hybrid",
    ]
    append_value_arg(cmd, "--sam2_checkpoint", args.sam2_checkpoint)
    append_value_arg(cmd, "--sam2_model_cfg", args.sam2_model_cfg)
    append_value_arg(cmd, "--volume_path", chunk_volume_path)
    append_value_arg(cmd, "--output_dir", chunk_output_dir)
    append_value_arg(cmd, "--output_filename", chunk_output_filename)
    append_value_arg(cmd, "--axis_sequence_cache_root", axis_sequence_cache_root)
    append_value_arg(cmd, "--feature_cache_device", args.feature_cache_device)
    append_value_arg(cmd, "--dataset_key", args.dataset_key)
    append_value_arg(cmd, "--axis", args.axis)
    append_value_arg(cmd, "--stride", args.stride)
    append_value_arg(cmd, "--gaussian_kernel", args.gaussian_kernel)
    append_value_arg(cmd, "--min_bright", args.min_bright)
    append_value_arg(cmd, "--remove_portion", args.remove_portion)
    append_value_arg(cmd, "--cuda_device", args.cuda_device)
    append_value_arg(cmd, "--device", args.device)
    append_value_arg(cmd, "--need_transpose", "False")

    value_arg_names = [
        "max_track_distance",
        "max_init_mask_area",
        "max_segmented_seeds",
        "max_slice_mask_area",
        "max_slice_mask_ratio",
        "min_frame_mask_score",
        "min_frame_siou",
        "joint_init_axis_mode",
        "init_half_window",
        "top_k",
        "num_point_jitters",
        "jitter_radius",
        "point_sample_radius",
        "joint_energy_mode",
        "joint_energy_terms",
        "w_score",
        "w_iou",
        "w_centroid",
        "w_area",
        "empty_mask_penalty",
        "max_joint_avg_area",
        "robust_trials",
        "thr_best_energy",
        "thr_energy_std",
        "thr_energy_uniformity",
        "thr_robust_energy_cv",
        "thr_robust_path_iou",
        "thr_avg_area_min",
        "thr_avg_area_max",
        "random_seed",
        "segment_len",
        "segment_len_diameter_multiplier",
        "min_segment_frames",
        "axis_ratio_thr",
        "segment_respawn_num_seeds",
        "segment_respawn_min_distance",
        "max_segment_respawn_candidates",
        "skeleton_respawn_offset",
        "max_skeleton_respawn_seeds",
        "max_untrusted_segments_per_lineage",
        "min_respawn_mask_area",
        "respawn_num_seeds",
        "respawn_min_point_distance",
        "max_respawn_candidates",
        "longterm_segment_len",
        "longterm_quality_thr",
        "max_longterm_segments",
        "working_window",
        "max_global_inject_per_seed",
        "segment_label_output_filename",
        "segment_classifier_stable_quality_thr",
        "segment_classifier_stable_iou_thr",
        "segment_classifier_stable_empty_rate_thr",
        "segment_classifier_stable_axis_ratio_thr",
        "segment_classifier_boundary_tail_max_frames",
        "segment_classifier_complex_tube_score_thr",
        "segment_classifier_failure_quality_thr",
        "segment_classifier_failure_iou_thr",
        "segment_classifier_failure_empty_rate_thr",
        "segment_classifier_failure_min_frames",
        "segmented_seed_log_filename",
    ]
    for name in value_arg_names:
        append_value_arg(cmd, f"--{name}", getattr(args, name))

    flag_arg_names = [
        "enable_axis_feature_cache",
        "enable_seed_judge",
        "disable_joint_init",
        "enable_respawn",
        "disable_longterm_memory",
        "enable_skeleton_respawn",
        "disable_legacy_segment_trust_respawn",
        "enable_segment_classifier_labels",
        "disable_segment_classifier_dominant_axis_check",
        "print_segment_classifier_details",
        "enable_segmented_seed_logging",
        "print_total_runtime",
        "vos_offload_video_to_cpu",
        "keep_tmp_vos_frames",
    ]
    for name in flag_arg_names:
        append_flag_arg(cmd, f"--{name}", bool(getattr(args, name)))

    if chunk_seed_path is not None:
        cmd.extend(["--seed_file", chunk_seed_path])
    if chunk_init_seg_path is not None:
        cmd.extend(["--init_seg_path", chunk_init_seg_path])
    cmd.extend(passthrough_args)
    return cmd


def main():
    run_start_time = time.perf_counter()
    args, passthrough_args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_prefix = args.run_prefix or (time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    chunks_root = os.path.join(args.output_dir, args.chunks_subdir, run_prefix)
    merged_root = os.path.join(args.output_dir, args.merged_subdir, run_prefix)
    shared_cache_root = args.shared_cache_root or os.path.join(args.output_dir, "shared_chunk_cache")
    shared_cache_root = os.path.join(shared_cache_root, f"chunk_size_{args.chunk_size}")
    os.makedirs(chunks_root, exist_ok=True)
    os.makedirs(merged_root, exist_ok=True)
    os.makedirs(shared_cache_root, exist_ok=True)

    vol_man = VolumeManager(args.volume_path, key=args.dataset_key)
    full_vol = vol_man.vol
    affine = vol_man.affine
    full_shape = full_vol.shape

    all_seeds = load_seed_file(args.seed_file) if args.seed_file else None
    init_seg = load_mask(args.init_seg_path) if args.init_seg_path else None
    if init_seg is not None and init_seg.shape != full_shape:
        raise ValueError(
            f"init_seg shape {init_seg.shape} does not match volume shape {full_shape}"
        )

    merged_mask = np.zeros(full_shape, dtype=np.uint8)
    chunks = build_chunks(full_shape, args.chunk_size)
    print(f"Volume shape: {full_shape}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Run prefix: {run_prefix}")
    print(f"Shared cache root: {shared_cache_root}")

    processed_chunks = 0
    skipped_chunks = 0

    for chunk in chunks:
        chunk_dir = os.path.join(chunks_root, chunk.tag)
        shared_chunk_dir = os.path.join(shared_cache_root, chunk.tag)
        ensure_dir(chunk_dir)
        ensure_dir(shared_chunk_dir)

        chunk_vol = slice_volume(full_vol, chunk)
        shared_volume_path = os.path.join(shared_chunk_dir, f"{chunk.tag}_volume.nii.gz")
        chunk_result_name = f"{run_prefix}_{chunk.tag}_seg.nii.gz"
        chunk_result_path = os.path.join(chunk_dir, chunk_result_name)

        local_seeds: Optional[List[Tuple[int, int, int]]] = None
        chunk_seed_path: Optional[str] = None
        if all_seeds is not None:
            local_seeds = filter_local_seeds(all_seeds, chunk)
            if len(local_seeds) == 0:
                save_nifti(chunk_result_path, np.zeros(chunk.shape, dtype=np.uint8), affine, False)
                skipped_chunks += 1
                print(f"[SKIP] {chunk.tag}: no seeds inside this chunk")
                continue
            chunk_seed_path = os.path.join(chunk_dir, f"{run_prefix}_{chunk.tag}_seeds.txt")
            save_seed_file(chunk_seed_path, local_seeds)

        chunk_init_seg_path: Optional[str] = None
        if init_seg is not None:
            chunk_init_seg_path = os.path.join(shared_chunk_dir, "global_init_seg.nii.gz")
            if not os.path.exists(chunk_init_seg_path):
                local_init_seg = slice_volume(init_seg, chunk)
                save_nifti(chunk_init_seg_path, local_init_seg, affine, False)
        else:
            cached_auto_init_seg = os.path.join(shared_chunk_dir, shared_auto_init_seg_basename(args))
            if os.path.exists(cached_auto_init_seg):
                chunk_init_seg_path = cached_auto_init_seg

        if args.skip_existing_chunks and os.path.exists(chunk_result_path):
            print(f"[SKIP] {chunk.tag}: existing result {chunk_result_path}")
        else:
            if not os.path.exists(shared_volume_path):
                save_nifti(shared_volume_path, chunk_vol, affine, False)
            cmd = build_child_cmd(
                args=args,
                passthrough_args=passthrough_args,
                chunk_volume_path=shared_volume_path,
                chunk_output_dir=chunk_dir,
                chunk_output_filename=chunk_result_name,
                chunk_seed_path=chunk_seed_path,
                chunk_init_seg_path=chunk_init_seg_path,
                axis_sequence_cache_root=os.path.join(shared_chunk_dir, "_axis_sequence_cache"),
            )
            print(f"[RUN] {chunk.tag} shape={chunk.shape}")
            print(" ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True)
                if init_seg is None:
                    local_auto_init_seg = os.path.join(chunk_dir, default_init_seg_basename(args))
                    shared_auto_init_seg = os.path.join(shared_chunk_dir, shared_auto_init_seg_basename(args))
                    if os.path.exists(local_auto_init_seg) and (not os.path.exists(shared_auto_init_seg)):
                        shutil.copy2(local_auto_init_seg, shared_auto_init_seg)

        if not os.path.exists(chunk_result_path):
            raise FileNotFoundError(f"Chunk result not found: {chunk_result_path}")

        chunk_mask = load_mask(chunk_result_path)
        if chunk_mask.shape != chunk.shape:
            raise ValueError(
                f"Chunk result shape mismatch for {chunk.tag}: "
                f"expected {chunk.shape}, got {chunk_mask.shape}"
            )
        merged_mask[chunk.z0:chunk.z1, chunk.y0:chunk.y1, chunk.x0:chunk.x1] |= chunk_mask
        processed_chunks += 1

        if (not args.keep_chunk_init_seg) and init_seg is None:
            local_auto_init_seg = os.path.join(chunk_dir, default_init_seg_basename(args))
            if os.path.exists(local_auto_init_seg):
                os.remove(local_auto_init_seg)

    merged_filename = f"{run_prefix}_{args.output_filename}" if args.output_filename else f"{run_prefix}_segmentation_merged.nii.gz"
    merged_path = os.path.join(merged_root, merged_filename)
    save_nifti(merged_path, merged_mask, affine, args.need_transpose != "False")

    print("=" * 80)
    print(f"Chunks merged: {processed_chunks}")
    print(f"Chunks skipped without local seeds: {skipped_chunks}")
    print(f"Merged segmentation saved to: {merged_path}")
    if args.print_total_runtime:
        elapsed_sec = time.perf_counter() - run_start_time
        print(f"Total tiled runtime: {elapsed_sec:.2f}s")


if __name__ == "__main__":
    main()
