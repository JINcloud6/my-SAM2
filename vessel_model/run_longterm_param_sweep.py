"""
Run SAM2 long-term-memory segmentation with multiple parameter combinations
on the same volume_path, saving each result with a distinct filename.
"""

import argparse
import itertools
import os
import subprocess
import sys
from typing import List, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
    combos = list(
        itertools.product(
            segment_lens,
            quality_thrs,
            max_longterm_segments_list,
            max_global_inject_list,
        )
    )
    return combos


def main():
    parser = argparse.ArgumentParser()
    # common args for sam2_main4_longterm_memory.py
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

    # sweep lists
    parser.add_argument("--segment_len_list", default="5,20")
    parser.add_argument("--longterm_quality_thr_list", default="0.75,0.85")
    parser.add_argument("--max_longterm_segments_list", default="3,5,8")
    parser.add_argument("--max_global_inject_per_seed_list", default="0,2,6")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print generated commands, do not execute.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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

    for i, (seg_len, q_thr, max_lt, max_inj) in enumerate(combos, start=1):
        result_name = (
            f"seg_lt_s{seg_len}_q{q_thr:.2f}_mlt{max_lt}_mg{max_inj}.nii.gz"
        )
        cmd = [
            sys.executable,
            "-m",
            "vessel_model.sam2_main4_longterm_memory",
            "--sam2_checkpoint",
            args.sam2_checkpoint,
            "--sam2_model_cfg",
            args.sam2_model_cfg,
            "--volume_path",
            args.volume_path,
            "--output_dir",
            args.output_dir,
            "--output_filename",
            result_name,
            "--dataset_key",
            args.dataset_key,
            "--axis",
            str(args.axis),
            "--stride",
            str(args.stride),
            "--gaussian_kernel",
            str(args.gaussian_kernel),
            "--min_bright",
            str(args.min_bright),
            "--remove_portion",
            str(args.remove_portion),
            "--device",
            args.device,
            "--need_transpose",
            args.need_transpose,
            "--max_track_distance",
            str(args.max_track_distance),
            "--max_init_mask_area",
            str(args.max_init_mask_area),
            "--working_window",
            str(args.working_window),
            "--segment_len",
            str(seg_len),
            "--longterm_quality_thr",
            str(q_thr),
            "--max_longterm_segments",
            str(max_lt),
            "--max_global_inject_per_seed",
            str(max_inj),
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

        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
