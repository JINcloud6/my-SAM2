#mouse patch的eval

import argparse
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .sweep_and_eval import calculate_all_metrics, load_data, print_metrics, save_results_to_csv


PATCH_RE = re.compile(
    r"(?:.+_)?chunk_(?P<chunk_id>\d+)"
    r"_z(?P<z0>\d+)_(?P<z1>\d+)"
    r"_y(?P<y0>\d+)_(?P<y1>\d+)"
    r"_x(?P<x0>\d+)_(?P<x1>\d+)_seg\.nii(?:\.gz)?$"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg_root", required=True, help="Root directory containing tiled chunk seg files")
    parser.add_argument("--gt_path", required=True, help="Ground-truth H5 path")
    parser.add_argument(
        "--full_volume_shape",
        default="2495,3571,5145",
        help="Original full-resolution volume shape as z,y,x",
    )
    parser.add_argument("--csv_name", default="tiled_mouse_patch_metrics.csv")
    parser.add_argument("--sort_by", default="dice", choices=["precision", "recall", "accuracy", "f1", "dice", "iou"])
    parser.add_argument("--print_each", action="store_true")
    return parser.parse_args()


def parse_shape(shape_str: str) -> Tuple[int, int, int]:
    parts = [int(x.strip()) for x in shape_str.split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError(f"Invalid full_volume_shape: {shape_str}")
    return tuple(parts)  # type: ignore[return-value]


def find_seg_files(seg_root: str) -> List[str]:
    seg_files: List[str] = []
    for root, _, files in os.walk(seg_root):
        for name in files:
            if name.endswith(".nii") or name.endswith(".nii.gz"):
                if PATCH_RE.match(name):
                    seg_files.append(os.path.join(root, name))
    seg_files.sort()
    return seg_files


def parse_patch_info(seg_path: str) -> Dict[str, int]:
    name = os.path.basename(seg_path)
    m = PATCH_RE.match(name)
    if m is None:
        raise ValueError(f"Cannot parse patch coordinates from filename: {name}")
    return {k: int(v) for k, v in m.groupdict().items()}


def project_axis0_range(
    z0: int,
    z1: int,
    full_depth: int,
    gt_depth: int,
) -> Tuple[int, int]:
    gt_z0 = int(np.floor(z0 * gt_depth / full_depth))
    gt_z1 = int(np.floor(z1 * gt_depth / full_depth))
    gt_z0 = max(0, min(gt_z0, gt_depth))
    gt_z1 = max(gt_z0, min(gt_z1, gt_depth))
    if gt_z1 == gt_z0 and z1 > z0 and gt_z0 < gt_depth:
        gt_z1 = min(gt_depth, gt_z0 + 1)
    return gt_z0, gt_z1


def compress_patch_axis0_binary(
    seg: np.ndarray,
    src_z0: int,
    src_z1: int,
    dst_z0: int,
    dst_z1: int,
) -> np.ndarray:
    if seg.shape[0] != (src_z1 - src_z0):
        raise ValueError(
            f"Patch z-size mismatch: seg has {seg.shape[0]}, range says {src_z1 - src_z0}"
        )
    out_depth = dst_z1 - dst_z0
    if out_depth <= 0:
        return np.zeros((0, seg.shape[1], seg.shape[2]), dtype=np.uint8)

    compressed = np.zeros((out_depth, seg.shape[1], seg.shape[2]), dtype=np.uint8)
    src_depth = src_z1 - src_z0

    for local_dst_z in range(out_depth):
        local_src_start = int(np.floor(local_dst_z * src_depth / out_depth))
        local_src_end = int(np.floor((local_dst_z + 1) * src_depth / out_depth))
        local_src_start = max(0, min(local_src_start, src_depth))
        local_src_end = max(local_src_start, min(local_src_end, src_depth))
        if local_src_end == local_src_start and local_src_start < src_depth:
            local_src_end = min(src_depth, local_src_start + 1)
        if local_src_end > local_src_start:
            compressed[local_dst_z] = (np.any(seg[local_src_start:local_src_end] > 0, axis=0)).astype(np.uint8)
    return compressed


def crop_to_gt_xy(
    seg: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    gt_shape: Tuple[int, int, int],
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    gt_y0 = max(0, min(y0, gt_shape[1]))
    gt_y1 = max(gt_y0, min(y1, gt_shape[1]))
    gt_x0 = max(0, min(x0, gt_shape[2]))
    gt_x1 = max(gt_x0, min(x1, gt_shape[2]))

    local_y0 = gt_y0 - y0
    local_y1 = local_y0 + (gt_y1 - gt_y0)
    local_x0 = gt_x0 - x0
    local_x1 = local_x0 + (gt_x1 - gt_x0)
    return seg[:, local_y0:local_y1, local_x0:local_x1], (gt_y0, gt_y1, gt_x0, gt_x1)


def evaluate_one_patch(
    seg_path: str,
    gt: np.ndarray,
    full_shape: Tuple[int, int, int],
) -> Dict[str, Any]:
    patch = parse_patch_info(seg_path)
    seg = (load_data(seg_path) > 0).astype(np.uint8)

    gt_z0, gt_z1 = project_axis0_range(patch["z0"], patch["z1"], full_shape[0], gt.shape[0])
    seg_compressed = compress_patch_axis0_binary(seg, patch["z0"], patch["z1"], gt_z0, gt_z1)
    seg_cropped, (gt_y0, gt_y1, gt_x0, gt_x1) = crop_to_gt_xy(
        seg_compressed, patch["y0"], patch["y1"], patch["x0"], patch["x1"], gt.shape
    )
    gt_patch = gt[gt_z0:gt_z1, gt_y0:gt_y1, gt_x0:gt_x1]

    if seg_cropped.shape != gt_patch.shape:
        raise ValueError(
            f"Shape mismatch for {os.path.basename(seg_path)}: "
            f"pred {seg_cropped.shape} vs gt {gt_patch.shape}"
        )

    metrics = calculate_all_metrics(seg_cropped, gt_patch)
    return {
        "seg_path": seg_path,
        "seg_name": os.path.basename(seg_path),
        "chunk_id": patch["chunk_id"],
        "src_z0": patch["z0"],
        "src_z1": patch["z1"],
        "src_y0": patch["y0"],
        "src_y1": patch["y1"],
        "src_x0": patch["x0"],
        "src_x1": patch["x1"],
        "gt_z0": gt_z0,
        "gt_z1": gt_z1,
        "gt_y0": gt_y0,
        "gt_y1": gt_y1,
        "gt_x0": gt_x0,
        "gt_x1": gt_x1,
        "pred_shape": str(tuple(seg_cropped.shape)),
        **metrics,
    }


def summarize_global(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    tp = sum(int(r["tp"]) for r in results)
    fp = sum(int(r["fp"]) for r in results)
    fn = sum(int(r["fn"]) for r in results)
    tn = sum(int(r["tn"]) for r in results)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {
        "seg_path": "__GLOBAL__",
        "seg_name": "__GLOBAL__",
        "chunk_id": -1,
        "src_z0": -1,
        "src_z1": -1,
        "src_y0": -1,
        "src_y1": -1,
        "src_x0": -1,
        "src_x1": -1,
        "gt_z0": -1,
        "gt_z1": -1,
        "gt_y0": -1,
        "gt_y1": -1,
        "gt_x0": -1,
        "gt_x1": -1,
        "pred_shape": "ALL_PATCHES",
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "dice": float(f1),
        "iou": float(iou),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def main():
    args = parse_args()

    full_shape = parse_shape(args.full_volume_shape)
    gt = load_data(args.gt_path)
    seg_files = find_seg_files(args.seg_root)
    if len(seg_files) == 0:
        raise RuntimeError(f"No chunk seg files found under: {args.seg_root}")

    print(f"Found {len(seg_files)} patch seg files")
    print(f"Full volume shape: {full_shape}")
    print(f"Ground truth shape: {gt.shape}")

    results: List[Dict[str, Any]] = []
    for seg_path in seg_files:
        row = evaluate_one_patch(seg_path, gt, full_shape)
        results.append(row)
        if args.print_each:
            print_metrics(os.path.basename(seg_path), row)

    global_row = summarize_global(results)
    print_metrics("GLOBAL_PATCH_SUM", global_row)

    results_sorted = sorted(results, key=lambda x: x[args.sort_by], reverse=True)
    results_with_summary = [global_row] + results_sorted
    csv_path = os.path.join(args.seg_root, args.csv_name)
    save_results_to_csv(results_with_summary, csv_path)
    print(f"Saved patch metrics to: {csv_path}")


if __name__ == "__main__":
    main()
