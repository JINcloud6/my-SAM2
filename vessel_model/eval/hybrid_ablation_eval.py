"""
Run ablations for the three hybrid ideas:
1) joint multi-slice initialization
2) seed respawn
3) long-term memory

By default this evaluates all 2^3 combinations on one volume and saves a CSV.
"""

import argparse
import csv
import itertools
import os
import subprocess
import sys
from typing import Any, Dict, List, Tuple

from .sweep_and_eval import calculate_all_metrics, load_data, print_metrics, save_results_to_csv



os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def bool_flag(v: bool) -> str:
    return "on" if v else "off"


def build_ablation_combinations() -> List[Tuple[bool, bool, bool]]:
    return list(itertools.product([False, True], repeat=3))


def make_result_name(use_joint_init: bool, use_respawn: bool, use_longterm: bool) -> str:
    return (
        f"hybrid_ablation_"
        f"joint_{bool_flag(use_joint_init)}_"
        f"respawn_{bool_flag(use_respawn)}_"
        f"ltmem_{bool_flag(use_longterm)}.nii.gz"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sam2_checkpoint", required=True)
    parser.add_argument("--sam2_model_cfg", required=True)
    parser.add_argument("--volume_path", required=True)
    parser.add_argument("--gt_path", required=True)
    parser.add_argument("--output_dir", required=True)

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

    parser.add_argument("--min_respawn_mask_area", type=int, default=20)
    parser.add_argument("--respawn_num_seeds", type=int, default=2)
    parser.add_argument("--respawn_min_point_distance", type=float, default=24.0)
    parser.add_argument("--max_respawn_candidates", type=int, default=512)
    parser.add_argument("--segment_len", type=int, default=15)
    parser.add_argument("--min_segment_frames", type=int, default=5)
    parser.add_argument("--axis_ratio_thr", type=float, default=1.1)
    parser.add_argument("--segment_respawn_num_seeds", type=int, default=4)
    parser.add_argument("--segment_respawn_min_distance", type=float, default=20.0)
    parser.add_argument("--max_segment_respawn_candidates", type=int, default=2048)
    parser.add_argument("--max_untrusted_segments_per_lineage", type=int, default=1)

    parser.add_argument("--longterm_segment_len", type=int, default=12)
    parser.add_argument("--longterm_quality_thr", type=float, default=0.8)
    parser.add_argument("--max_longterm_segments", type=int, default=5)
    parser.add_argument("--working_window", type=int, default=24)
    parser.add_argument("--max_global_inject_per_seed", type=int, default=0)

    parser.add_argument("--vos_offload_video_to_cpu", action="store_true")
    parser.add_argument("--keep_tmp_vos_frames", action="store_true")

    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--csv_name", default="hybrid_ablation_metrics.csv")
    parser.add_argument("--sort_by", default="dice", choices=["precision", "recall", "accuracy", "f1", "dice", "iou"])

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    gt_data = load_data(args.gt_path)
    combos = build_ablation_combinations()
    all_results: List[Dict[str, Any]] = []

    for i, (use_joint_init, use_respawn, use_longterm) in enumerate(combos, start=1):
        result_name = make_result_name(use_joint_init, use_respawn, use_longterm)
        result_path = os.path.join(args.output_dir, result_name)

        cmd = [
            sys.executable,
            "-m",
            "vessel_model.sam2_main4_hybrid",
            "--sam2_checkpoint", args.sam2_checkpoint,
            "--sam2_model_cfg", args.sam2_model_cfg,
            "--volume_path", args.volume_path,
            "--output_dir", args.output_dir,
            "--output_filename", result_name,
            "--dataset_key", args.dataset_key,
            "--axis", str(args.axis),
            "--stride", str(args.stride),
            "--gaussian_kernel", str(args.gaussian_kernel),
            "--min_bright", str(args.min_bright),
            "--remove_portion", str(args.remove_portion),
            "--device", args.device,
            "--need_transpose", args.need_transpose,
            "--max_track_distance", str(args.max_track_distance),
            "--max_init_mask_area", str(args.max_init_mask_area),
            "--max_segmented_seeds", str(args.max_segmented_seeds),
            "--init_half_window", str(args.init_half_window),
            "--top_k", str(args.top_k),
            "--num_point_jitters", str(args.num_point_jitters),
            "--jitter_radius", str(args.jitter_radius),
            "--w_score", str(args.w_score),
            "--w_iou", str(args.w_iou),
            "--w_centroid", str(args.w_centroid),
            "--w_area", str(args.w_area),
            "--empty_mask_penalty", str(args.empty_mask_penalty),
            "--max_joint_avg_area", str(args.max_joint_avg_area),
            "--robust_trials", str(args.robust_trials),
            "--thr_best_energy", str(args.thr_best_energy),
            "--thr_energy_std", str(args.thr_energy_std),
            "--thr_energy_uniformity", str(args.thr_energy_uniformity),
            "--thr_robust_energy_cv", str(args.thr_robust_energy_cv),
            "--thr_robust_path_iou", str(args.thr_robust_path_iou),
            "--thr_avg_area_min", str(args.thr_avg_area_min),
            "--thr_avg_area_max", str(args.thr_avg_area_max),
            "--random_seed", str(args.random_seed),
            "--min_respawn_mask_area", str(args.min_respawn_mask_area),
            "--respawn_num_seeds", str(args.respawn_num_seeds),
            "--respawn_min_point_distance", str(args.respawn_min_point_distance),
            "--max_respawn_candidates", str(args.max_respawn_candidates),
            "--segment_len", str(args.segment_len),
            "--min_segment_frames", str(args.min_segment_frames),
            "--axis_ratio_thr", str(args.axis_ratio_thr),
            "--segment_respawn_num_seeds", str(args.segment_respawn_num_seeds),
            "--segment_respawn_min_distance", str(args.segment_respawn_min_distance),
            "--max_segment_respawn_candidates", str(args.max_segment_respawn_candidates),
            "--max_untrusted_segments_per_lineage", str(args.max_untrusted_segments_per_lineage),
            "--longterm_segment_len", str(args.longterm_segment_len),
            "--longterm_quality_thr", str(args.longterm_quality_thr),
            "--max_longterm_segments", str(args.max_longterm_segments),
            "--working_window", str(args.working_window),
            "--max_global_inject_per_seed", str(args.max_global_inject_per_seed),
        ]

        if args.seed_file:
            cmd.extend(["--seed_file", args.seed_file])
        if args.init_seg_path:
            cmd.extend(["--init_seg_path", args.init_seg_path])
        if args.enable_seed_judge:
            cmd.append("--enable_seed_judge")
        if args.vos_offload_video_to_cpu:
            cmd.append("--vos_offload_video_to_cpu")
        if args.keep_tmp_vos_frames:
            cmd.append("--keep_tmp_vos_frames")

        if not use_joint_init:
            cmd.append("--disable_joint_init")
        if use_respawn:
            cmd.append("--enable_respawn")
        if not use_longterm:
            cmd.append("--disable_longterm_memory")

        print(f"\n[{i}/{len(combos)}] {result_name}")
        print(" ".join(cmd))

        if not args.eval_only:
            should_run = True
            if args.skip_existing and os.path.exists(result_path):
                print(f"Output exists, skip segmentation: {result_path}")
                should_run = False
            if args.dry_run:
                should_run = False
            if should_run:
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"[ERROR] Segmentation failed for {result_name}: {e}")
                    all_results.append(
                        {
                            "result_name": result_name,
                            "result_path": result_path,
                            "use_joint_init": use_joint_init,
                            "use_respawn": use_respawn,
                            "use_longterm_memory": use_longterm,
                            "status": "segmentation_failed",
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
                    )
                    continue

        if not os.path.exists(result_path):
            print(f"[WARNING] Result file not found, skip evaluation: {result_path}")
            all_results.append(
                {
                    "result_name": result_name,
                    "result_path": result_path,
                    "use_joint_init": use_joint_init,
                    "use_respawn": use_respawn,
                    "use_longterm_memory": use_longterm,
                    "status": "result_not_found",
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
            )
            continue

        try:
            pred_data = load_data(result_path)
            if gt_data.shape != pred_data.shape:
                raise ValueError(f"Shape mismatch: GT {gt_data.shape} vs Pred {pred_data.shape}")
            metrics = calculate_all_metrics(pred_data, gt_data)
            print_metrics(result_name, metrics)
            all_results.append(
                {
                    "result_name": result_name,
                    "result_path": result_path,
                    "use_joint_init": use_joint_init,
                    "use_respawn": use_respawn,
                    "use_longterm_memory": use_longterm,
                    "status": "ok",
                    **metrics,
                }
            )
        except Exception as e:
            print(f"[ERROR] Evaluation failed for {result_name}: {e}")
            all_results.append(
                {
                    "result_name": result_name,
                    "result_path": result_path,
                    "use_joint_init": use_joint_init,
                    "use_respawn": use_respawn,
                    "use_longterm_memory": use_longterm,
                    "status": f"eval_failed: {str(e)}",
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
            )

    valid_results = [r for r in all_results if r["status"] == "ok"]
    invalid_results = [r for r in all_results if r["status"] != "ok"]
    valid_results = sorted(valid_results, key=lambda x: x[args.sort_by], reverse=True)
    all_results_sorted = valid_results + invalid_results

    csv_path = os.path.join(args.output_dir, args.csv_name)
    save_results_to_csv(all_results_sorted, csv_path)

    print("\n========== ABLATION RESULTS ==========")
    for row in valid_results:
        print(
            f"{row['result_name']} | "
            f"joint={row['use_joint_init']} respawn={row['use_respawn']} ltmem={row['use_longterm_memory']} | "
            f"dice={row['dice']:.4f} iou={row['iou']:.4f}"
        )


if __name__ == "__main__":
    main()
