"""
Sweep all SAM2 long-term-memory parameter combinations, run segmentation,
evaluate against GT, and save all metrics to a CSV.

参考：
- 参数组合分割脚本
- metrics 评估脚本
"""

import argparse
import csv
import itertools
import os
import subprocess
import sys
from typing import List, Tuple, Dict, Any

import numpy as np
import h5py
import nibabel as nib


# =========================
# 解析列表参数
# =========================
def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def build_combinations(
    segment_lens: List[int],
    quality_thrs: List[float],
    max_longterm_segments_list: List[int],
    max_global_inject_list: List[int],
) -> List[Tuple[int, float, int, int]]:
    return list(
        itertools.product(
            segment_lens,
            quality_thrs,
            max_longterm_segments_list,
            max_global_inject_list,
        )
    )


# =========================
# 数据加载
# =========================
def load_data(file_path: str) -> np.ndarray:
    """通用数据加载函数，支持 NIfTI 和 H5 格式"""
    if file_path.endswith((".nii", ".nii.gz")):
        return nib.load(file_path).get_fdata()
    elif file_path.endswith((".h5", ".hdf5")):
        with h5py.File(file_path, "r") as f:
            if "main" not in f:
                raise KeyError(f"'main' key not found in H5 file: {file_path}")
            return f["main"][:]
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


# =========================
# 指标计算
# =========================
def calculate_all_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> Dict[str, Any]:
    """
    计算二分类评价指标：Precision, Recall, Accuracy, F1, Dice, IoU
    """
    pred = prediction > 0
    gt = ground_truth > 0

    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)
    tn = np.sum(~pred & ~gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / pred.size if pred.size > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    dice = f1
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "dice": float(dice),
        "iou": float(iou),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def print_metrics(name: str, metrics: Dict[str, Any]) -> None:
    print(f"--- Metrics for {name} ---")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  Dice:      {metrics['dice']:.4f}")
    print(f"  IoU:       {metrics['iou']:.4f}")
    print(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}, TN: {metrics['tn']}")
    print()


# =========================
# 保存 CSV
# =========================
def save_results_to_csv(results: List[Dict[str, Any]], csv_path: str) -> None:
    if not results:
        print("No results to save.")
        return

    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved metrics CSV to: {csv_path}")


# =========================
# 主流程
# =========================
def main():
    parser = argparse.ArgumentParser()

    # -------- segmentation args --------
    parser.add_argument("--sam2_checkpoint", required=True)
    parser.add_argument("--sam2_model_cfg", required=True)
    parser.add_argument("--volume_path", required=True)
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
    parser.add_argument("--working_window", type=int, default=24)

    parser.add_argument("--vos_offload_video_to_cpu", action="store_true")
    parser.add_argument("--keep_tmp_vos_frames", action="store_true")

    # -------- sweep args --------
    parser.add_argument("--segment_len_list", default="5,20")
    parser.add_argument("--longterm_quality_thr_list", default="0.75,0.85")
    parser.add_argument("--max_longterm_segments_list", default="3,5,8")
    parser.add_argument("--max_global_inject_per_seed_list", default="0,2,6")

    # -------- eval args --------
    parser.add_argument("--gt_path", required=True, help="Ground truth path (.nii/.nii.gz/.h5)")
    parser.add_argument("--csv_name", default="sweep_metrics.csv")
    parser.add_argument("--sort_by", default="dice", choices=["precision", "recall", "accuracy", "f1", "dice", "iou"])

    # -------- behavior args --------
    parser.add_argument("--skip_existing", action="store_true", help="If output exists, skip segmentation and only evaluate")
    parser.add_argument("--eval_only", action="store_true", help="Do not run segmentation, only evaluate existing outputs")
    parser.add_argument("--dry_run", action="store_true", help="Only print commands, do not execute segmentation")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 读取 GT 一次即可
    gt_data = load_data(args.gt_path)

    segment_lens = parse_int_list(args.segment_len_list)
    quality_thrs = parse_float_list(args.longterm_quality_thr_list)
    max_longterm_segments_list = parse_int_list(args.max_longterm_segments_list)
    max_global_inject_list = parse_int_list(args.max_global_inject_per_seed_list)

    combos = build_combinations(
        segment_lens,
        quality_thrs,
        max_longterm_segments_list,
        max_global_inject_list,
    )

    print(f"Total parameter combinations: {len(combos)}")

    all_results = []

    for i, (seg_len, q_thr, max_lt, max_inj) in enumerate(combos, start=1):
        result_name = f"seg_lt_s{seg_len}_q{q_thr:.2f}_mlt{max_lt}_mg{max_inj}.nii.gz"
        result_path = os.path.join(args.output_dir, result_name)

        cmd = [
            sys.executable,
            "-m",
            "vessel_model.sam2_main4_longterm_memory",
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
            "--working_window", str(args.working_window),
            "--segment_len", str(seg_len),
            "--longterm_quality_thr", str(q_thr),
            "--max_longterm_segments", str(max_lt),
            "--max_global_inject_per_seed", str(max_inj),
        ]

        if args.seed_file:
            cmd.extend(["--seed_file", args.seed_file])
        if args.init_seg_path:
            cmd.extend(["--init_seg_path", args.init_seg_path])
        if args.vos_offload_video_to_cpu:
            cmd.append("--vos_offload_video_to_cpu")
        if args.keep_tmp_vos_frames:
            cmd.append("--keep_tmp_vos_frames")

        print(f"\n[{i}/{len(combos)}] {result_name}")
        print(" ".join(cmd))

        # 1) 先生成分割
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
                    # 失败的组合也记录下来
                    all_results.append({
                        "result_name": result_name,
                        "result_path": result_path,
                        "segment_len": seg_len,
                        "longterm_quality_thr": q_thr,
                        "max_longterm_segments": max_lt,
                        "max_global_inject_per_seed": max_inj,
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
                    })
                    continue

        # 2) 计算 metrics
        if not os.path.exists(result_path):
            print(f"[WARNING] Result file not found, skip evaluation: {result_path}")
            all_results.append({
                "result_name": result_name,
                "result_path": result_path,
                "segment_len": seg_len,
                "longterm_quality_thr": q_thr,
                "max_longterm_segments": max_lt,
                "max_global_inject_per_seed": max_inj,
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
            })
            continue

        try:
            pred_data = load_data(result_path)

            if gt_data.shape != pred_data.shape:
                raise ValueError(f"Shape mismatch: GT {gt_data.shape} vs Pred {pred_data.shape}")

            metrics = calculate_all_metrics(pred_data, gt_data)
            print_metrics(result_name, metrics)

            row = {
                "result_name": result_name,
                "result_path": result_path,
                "segment_len": seg_len,
                "longterm_quality_thr": q_thr,
                "max_longterm_segments": max_lt,
                "max_global_inject_per_seed": max_inj,
                "status": "ok",
                **metrics,
            }
            all_results.append(row)

        except Exception as e:
            print(f"[ERROR] Evaluation failed for {result_name}: {e}")
            all_results.append({
                "result_name": result_name,
                "result_path": result_path,
                "segment_len": seg_len,
                "longterm_quality_thr": q_thr,
                "max_longterm_segments": max_lt,
                "max_global_inject_per_seed": max_inj,
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
            })

    # 排序：把有效结果放前面
    valid_results = [r for r in all_results if r["status"] == "ok"]
    invalid_results = [r for r in all_results if r["status"] != "ok"]

    valid_results = sorted(valid_results, key=lambda x: x[args.sort_by], reverse=True)
    all_results_sorted = valid_results + invalid_results

    # 保存 CSV
    csv_path = os.path.join(args.output_dir, args.csv_name)
    save_results_to_csv(all_results_sorted, csv_path)

    # 打印 top-k
    print("\n========== TOP RESULTS ==========")
    topk = min(10, len(valid_results))
    for rank, row in enumerate(valid_results[:topk], start=1):
        print(
            f"[{rank}] {row['result_name']} | "
            f"dice={row['dice']:.4f}, iou={row['iou']:.4f}, "
            f"precision={row['precision']:.4f}, recall={row['recall']:.4f}"
        )

    if not valid_results:
        print("No valid evaluated results.")


if __name__ == "__main__":
    main()