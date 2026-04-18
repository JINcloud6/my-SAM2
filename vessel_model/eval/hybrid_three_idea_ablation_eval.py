"""
Run ablations for the current hybrid vessel segmentation pipeline.

The three ablated ideas are:
1) joint multi-slice initialization (MSJI)
2) long-term memory
3) seed respawn

By default this script runs all 2^3 combinations, evaluates each output against
the provided GT, and saves a CSV summary.
"""

import argparse
import itertools
import os
import subprocess
import sys
from typing import Any, Dict, List, Sequence, Tuple

from .sweep_and_eval import calculate_all_metrics, load_data, print_metrics, save_results_to_csv


CONTROLLED_FLAGS = {
    "--disable_joint_init",
    "--enable_respawn",
    "--disable_longterm_memory",
}


VALUE_ARG_NAMES = [
    "cuda_device",
    "sam2_checkpoint",
    "sam2_model_cfg",
    "volume_path",
    "axis_sequence_cache_root",
    "feature_cache_device",
    "dataset_key",
    "seed_file",
    "init_seg_path",
    "axis",
    "stride",
    "gaussian_kernel",
    "min_bright",
    "remove_portion",
    "device",
    "need_transpose",
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
]


GLOBAL_FLAG_ARG_NAMES = [
    "enable_axis_feature_cache",
    "enable_seed_judge",
    "enable_segment_classifier_labels",
    "disable_segment_classifier_dominant_axis_check",
    "print_segment_classifier_details",
    "enable_segmented_seed_logging",
    "print_total_runtime",
    "vos_offload_video_to_cpu",
    "keep_tmp_vos_frames",
]


RESPAWN_ONLY_FLAG_ARG_NAMES = [
    "enable_skeleton_respawn",
    "disable_legacy_segment_trust_respawn",
]


def bool_flag(v: bool) -> str:
    return "on" if v else "off"


def build_ablation_combinations(mode: str) -> List[Tuple[bool, bool, bool]]:
    all_combos = list(itertools.product([False, True], repeat=3))
    if mode == "all":
        return all_combos
    if mode == "full_only":
        return [(True, True, True)]
    if mode == "baseline_only":
        return [(False, False, False)]
    if mode == "leave_one_out":
        return [(False, True, True), (True, False, True), (True, True, False), (True, True, True)]
    if mode == "single_idea":
        return [(False, False, False), (True, False, False), (False, True, False), (False, False, True)]
    if mode == "test_respawn":
        return [(True,True,True),(True,False,True)]
    raise ValueError(f"Unsupported ablation mode: {mode}")


def make_result_name(
    prefix: str,
    index: int,
    use_joint_init: bool,
    use_respawn: bool,
    use_longterm: bool,
) -> str:
    prefix_str = f"{prefix}_" if prefix else ""
    return (
        f"{prefix_str}{index:02d}_hybrid_ablation_"
        f"joint_{bool_flag(use_joint_init)}_"
        f"respawn_{bool_flag(use_respawn)}_"
        f"ltmem_{bool_flag(use_longterm)}.nii.gz"
    )


def strip_nii_suffix(filename: str) -> str:
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    return os.path.splitext(filename)[0]


def make_aux_filename(result_name: str, suffix: str, extension: str) -> str:
    return f"{strip_nii_suffix(result_name)}{suffix}{extension}"


def append_value_arg(cmd: List[str], flag: str, value: Any) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def append_flag_arg(cmd: List[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def ensure_no_controlled_passthrough(passthrough_args: Sequence[str]) -> None:
    bad = [arg for arg in passthrough_args if arg in CONTROLLED_FLAGS]
    if bad:
        raise ValueError(
            "These flags are controlled by the ablation script and must not be passed manually: "
            + ", ".join(sorted(set(bad)))
        )


def build_child_cmd(
    args,
    passthrough_args: Sequence[str],
    result_name: str,
    use_joint_init: bool,
    use_respawn: bool,
    use_longterm: bool,
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        args.segmentation_module,
        "--output_dir",
        args.output_dir,
        "--output_filename",
        result_name,
    ]

    for name in VALUE_ARG_NAMES:
        append_value_arg(cmd, f"--{name}", getattr(args, name))

    for name in GLOBAL_FLAG_ARG_NAMES:
        append_flag_arg(cmd, f"--{name}", bool(getattr(args, name)))

    if args.enable_segment_classifier_labels:
        label_name = make_aux_filename(result_name, "_segment_labels", ".nii.gz")
        append_value_arg(cmd, "--segment_label_output_filename", label_name)

    if args.enable_segmented_seed_logging:
        seed_log_name = make_aux_filename(result_name, "_segmented_seeds", ".csv")
        append_value_arg(cmd, "--segmented_seed_log_filename", seed_log_name)

    if not use_joint_init:
        cmd.append("--disable_joint_init")

    if use_respawn:
        cmd.append("--enable_respawn")
        for name in RESPAWN_ONLY_FLAG_ARG_NAMES:
            append_flag_arg(cmd, f"--{name}", bool(getattr(args, name)))

    if not use_longterm:
        cmd.append("--disable_longterm_memory")

    cmd.extend(passthrough_args)
    return cmd


def empty_metric_row(
    result_name: str,
    result_path: str,
    use_joint_init: bool,
    use_respawn: bool,
    use_longterm: bool,
    status: str,
) -> Dict[str, Any]:
    return {
        "result_name": result_name,
        "result_path": result_path,
        "use_joint_init": use_joint_init,
        "use_respawn": use_respawn,
        "use_longterm_memory": use_longterm,
        "status": status,
        "precision": None,
        "recall": None,
        "accuracy": None,
        "f1": None,
        "dice": None,
        "iou": None,
        "tp": None,
        "fp": None,
        "fn": None,
        "tn": None,
    }


def evaluate_result(
    result_name: str,
    result_path: str,
    gt_data,
    use_joint_init: bool,
    use_respawn: bool,
    use_longterm: bool,
) -> Dict[str, Any]:
    pred_data = load_data(result_path)
    if gt_data.shape != pred_data.shape:
        raise ValueError(f"Shape mismatch: GT {gt_data.shape} vs Pred {pred_data.shape}")
    metrics = calculate_all_metrics(pred_data, gt_data)
    print_metrics(result_name, metrics)
    return {
        "result_name": result_name,
        "result_path": result_path,
        "use_joint_init": use_joint_init,
        "use_respawn": use_respawn,
        "use_longterm_memory": use_longterm,
        "status": "ok",
        **metrics,
    }


def get_args():
    parser = argparse.ArgumentParser(
        description="Ablate joint_init, longterm_memory, and seed_respawn for sam2_main4_hybrid."
    )

    # Experiment/evaluation arguments.
    parser.add_argument("--sam2_checkpoint", required=True)
    parser.add_argument("--sam2_model_cfg", required=True)
    parser.add_argument("--volume_path", required=True)
    parser.add_argument("--gt_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--segmentation_module", default="vessel_model.sam2_main4_hybrid")
    parser.add_argument(
        "--ablation_mode",
        default="all",
        choices=["all", "leave_one_out", "single_idea", "full_only", "baseline_only", "test_respawn"],
        help="Which joint/respawn/longterm combinations to run.",
    )
    parser.add_argument("--result_prefix", default="")
    parser.add_argument("--csv_name", default="hybrid_three_idea_ablation_metrics.csv")
    parser.add_argument("--sort_by", default="dice", choices=["precision", "recall", "accuracy", "f1", "dice", "iou"])
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    # Current sam2_main4_hybrid value arguments.
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--axis_sequence_cache_root", default=None)
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
    parser.add_argument("--min_frame_mask_score", type=float, default=0.0)
    parser.add_argument("--min_frame_siou", type=float, default=0.0)
    parser.add_argument("--joint_init_axis_mode", default="preselect", choices=["preselect", "joint_energy"])
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
    parser.add_argument("--segment_len", type=int, default=15)
    parser.add_argument("--segment_len_diameter_multiplier", type=float, default=0.0)
    parser.add_argument("--min_segment_frames", type=int, default=5)
    parser.add_argument("--axis_ratio_thr", type=float, default=1.1)
    parser.add_argument("--segment_respawn_num_seeds", type=int, default=4)
    parser.add_argument("--segment_respawn_min_distance", type=float, default=20.0)
    parser.add_argument("--max_segment_respawn_candidates", type=int, default=2048)
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
    parser.add_argument("--segment_classifier_stable_quality_thr", type=float, default=0.80)
    parser.add_argument("--segment_classifier_stable_iou_thr", type=float, default=0.65)
    parser.add_argument("--segment_classifier_stable_empty_rate_thr", type=float, default=0.05)
    parser.add_argument("--segment_classifier_stable_axis_ratio_thr", type=float, default=1.10)
    parser.add_argument("--segment_classifier_boundary_tail_max_frames", type=int, default=6)
    parser.add_argument("--segment_classifier_complex_tube_score_thr", type=float, default=2.5)
    parser.add_argument("--segment_classifier_failure_quality_thr", type=float, default=0.55)
    parser.add_argument("--segment_classifier_failure_iou_thr", type=float, default=0.20)
    parser.add_argument("--segment_classifier_failure_empty_rate_thr", type=float, default=0.25)
    parser.add_argument("--segment_classifier_failure_min_frames", type=int, default=3)

    # Current sam2_main4_hybrid flags.
    parser.add_argument("--enable_axis_feature_cache", action="store_true")
    parser.add_argument("--enable_seed_judge", action="store_true")
    parser.add_argument("--enable_skeleton_respawn", action="store_true")
    parser.add_argument("--disable_legacy_segment_trust_respawn", action="store_true")
    parser.add_argument("--enable_segment_classifier_labels", action="store_true")
    parser.add_argument("--disable_segment_classifier_dominant_axis_check", action="store_true")
    parser.add_argument("--print_segment_classifier_details", action="store_true")
    parser.add_argument("--enable_segmented_seed_logging", action="store_true")
    parser.add_argument("--print_total_runtime", action="store_true")
    parser.add_argument("--vos_offload_video_to_cpu", action="store_true")
    parser.add_argument("--keep_tmp_vos_frames", action="store_true")

    return parser.parse_known_args()


def main():
    args, passthrough_args = get_args()
    ensure_no_controlled_passthrough(passthrough_args)
    os.makedirs(args.output_dir, exist_ok=True)

    gt_data = None if args.dry_run else load_data(args.gt_path)
    combos = build_ablation_combinations(args.ablation_mode)
    all_results: List[Dict[str, Any]] = []

    for index, (use_joint_init, use_respawn, use_longterm) in enumerate(combos, start=1):
        result_name = make_result_name(
            args.result_prefix,
            index,
            use_joint_init=use_joint_init,
            use_respawn=use_respawn,
            use_longterm=use_longterm,
        )
        result_path = os.path.join(args.output_dir, result_name)
        cmd = build_child_cmd(
            args=args,
            passthrough_args=passthrough_args,
            result_name=result_name,
            use_joint_init=use_joint_init,
            use_respawn=use_respawn,
            use_longterm=use_longterm,
        )

        print(f"\n[{index}/{len(combos)}] {result_name}")
        print(
            "  "
            f"joint={use_joint_init} respawn={use_respawn} longterm={use_longterm}"
        )
        print(" ".join(cmd))

        if args.dry_run:
            all_results.append(
                empty_metric_row(
                    result_name,
                    result_path,
                    use_joint_init,
                    use_respawn,
                    use_longterm,
                    "dry_run",
                )
            )
            continue

        if not args.eval_only:
            should_run = True
            if args.skip_existing and os.path.exists(result_path):
                print(f"Output exists, skip segmentation: {result_path}")
                should_run = False
            if should_run:
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as exc:
                    print(f"[ERROR] Segmentation failed for {result_name}: {exc}")
                    all_results.append(
                        empty_metric_row(
                            result_name,
                            result_path,
                            use_joint_init,
                            use_respawn,
                            use_longterm,
                            "segmentation_failed",
                        )
                    )
                    continue

        if not os.path.exists(result_path):
            print(f"[WARNING] Result file not found, skip evaluation: {result_path}")
            all_results.append(
                empty_metric_row(
                    result_name,
                    result_path,
                    use_joint_init,
                    use_respawn,
                    use_longterm,
                    "result_not_found",
                )
            )
            continue

        try:
            all_results.append(
                evaluate_result(
                    result_name,
                    result_path,
                    gt_data,
                    use_joint_init,
                    use_respawn,
                    use_longterm,
                )
            )
        except Exception as exc:
            print(f"[ERROR] Evaluation failed for {result_name}: {exc}")
            all_results.append(
                empty_metric_row(
                    result_name,
                    result_path,
                    use_joint_init,
                    use_respawn,
                    use_longterm,
                    f"eval_failed: {exc}",
                )
            )

    valid_results = [row for row in all_results if row["status"] == "ok"]
    invalid_results = [row for row in all_results if row["status"] != "ok"]
    valid_results = sorted(valid_results, key=lambda x: x[args.sort_by], reverse=True)
    all_results_sorted = valid_results + invalid_results

    csv_path = os.path.join(args.output_dir, args.csv_name)
    save_results_to_csv(all_results_sorted, csv_path)

    print("\n========== THREE-IDEA ABLATION RESULTS ==========")
    for row in valid_results:
        print(
            f"{row['result_name']} | "
            f"joint={row['use_joint_init']} "
            f"respawn={row['use_respawn']} "
            f"ltmem={row['use_longterm_memory']} | "
            f"dice={row['dice']:.4f} iou={row['iou']:.4f}"
        )
    if invalid_results:
        print("\nInvalid / skipped results:")
        for row in invalid_results:
            print(f"{row['result_name']} | status={row['status']}")


if __name__ == "__main__":
    main()
