import argparse
import os
import sys
from contextlib import redirect_stdout, redirect_stderr

from vessel_model.main import run_segmentation, xmem_config


def parse_list(value, cast=str):
    items = [v.strip() for v in value.split(",") if v.strip()]
    return [cast(v) for v in items]


def main():
    parser = argparse.ArgumentParser(description="Validate decay/entropy/split tracking features.")
    parser.add_argument("--decay_modes", default="linear",
                        help="Comma-separated decay modes: off, exp, linear")
    parser.add_argument("--decay_values", default="3,4,6,7,8,9",
                        help="Comma-separated decay parameter values (tau or L)")
    parser.add_argument("--entropy_windows", default="20",
                        help="Comma-separated entropy window sizes")
    parser.add_argument("--entropy_zs", default="2.0",
                        help="Comma-separated entropy Z thresholds")
    parser.add_argument("--entropy_counts", default="3",
                        help="Comma-separated entropy abnormal counts")
    parser.add_argument("--entropy_enable", default="off",
                        help="Enable entropy stop: on or off")
    parser.add_argument("--split_enable", default="off",
                        help="Comma-separated split seeding enable: off,on")
    parser.add_argument("--longterm_enable", default="on",
                        help="Comma-separated long-term memory enable: off,on")
    parser.add_argument("--max_mid_frames", default="10",
                        help="Comma-separated max mid-term frames")
    parser.add_argument("--min_mid_frames", default="5",
                        help="Comma-separated min mid-term frames")
    parser.add_argument("--num_prototypes", default="128",
                        help="Comma-separated prototype counts")
    parser.add_argument("--top_ks", default="30",
                        help="Comma-separated top-k values for memory matching")
    parser.add_argument("--mem_every", default="5",
                        help="Comma-separated memory write intervals")
    parser.add_argument("--drop_first_memory", default="off",
                        help="Comma-separated drop-first-memory enable: off,on")
    parser.add_argument("--output_dir", default="./bv_seg_output",
                        help="Output directory for segmentation results")
    parser.add_argument("--log_dir", default="tracking_feature_logs",
                        help="Directory to store per-run logs")
    parser.add_argument("--output_prefix", default="tracking_features",
                        help="Prefix for output nii.gz files")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only print configurations without running segmentation")
    args, remaining = parser.parse_known_args()

    decay_modes = parse_list(args.decay_modes, cast=str)
    decay_values = parse_list(args.decay_values, cast=float)
    entropy_windows = parse_list(args.entropy_windows, cast=int)
    entropy_zs = parse_list(args.entropy_zs, cast=float)
    entropy_counts = parse_list(args.entropy_counts, cast=int)
    split_enable = parse_list(args.split_enable, cast=str)
    longterm_enable = parse_list(args.longterm_enable, cast=str)
    max_mid_frames = parse_list(args.max_mid_frames, cast=int)
    min_mid_frames = parse_list(args.min_mid_frames, cast=int)
    num_prototypes = parse_list(args.num_prototypes, cast=int)
    top_ks = parse_list(args.top_ks, cast=int)
    mem_every_vals = parse_list(args.mem_every, cast=int)
    drop_first_memory = parse_list(args.drop_first_memory, cast=str)
    entropy_enable = args.entropy_enable == "on"

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    def _strip_arg(argv, flag):
        if flag not in argv:
            return argv
        idx = argv.index(flag)
        if idx < len(argv) - 1:
            return argv[:idx] + argv[idx+2:]
        return argv[:idx]

    for decay_mode in decay_modes:
        for decay_value in decay_values:
            for ew in entropy_windows:
                for ez in entropy_zs:
                    for ec in entropy_counts:
                        for split_flag in split_enable:
                            for lt_flag in longterm_enable:
                                for max_mid in max_mid_frames:
                                    for min_mid in min_mid_frames:
                                        for proto in num_prototypes:
                                            for top_k in top_ks:
                                                for mem_every in mem_every_vals:
                                                    for drop_flag in drop_first_memory:
                                                        decay_enabled = decay_mode != "off"
                                                        xmem_config["enable_temporal_decay"] = decay_enabled
                                                        xmem_config["temporal_decay_mode"] = decay_mode
                                                        if not decay_enabled:
                                                            xmem_config["temporal_decay"] = 0.0
                                                        else:
                                                            xmem_config["temporal_decay"] = float(decay_value)
                                                        xmem_config["enable_entropy_stop"] = entropy_enable
                                                        xmem_config["enable_attention_entropy"] = entropy_enable
                                                        xmem_config["entropy_window"] = ew
                                                        xmem_config["entropy_z_threshold"] = ez
                                                        xmem_config["entropy_abnormal_count"] = ec
                                                        xmem_config["enable_split_seeding"] = (split_flag == "on")
                                                        xmem_config["enable_long_term"] = (lt_flag == "on")
                                                        xmem_config["max_mid_term_frames"] = max_mid
                                                        xmem_config["min_mid_term_frames"] = min_mid
                                                        xmem_config["num_prototypes"] = proto
                                                        xmem_config["top_k"] = top_k
                                                        xmem_config["mem_every"] = mem_every
                                                        xmem_config["drop_first_memory"] = (drop_flag == "on")
                                                        tag = (
                                                            f"decay-{decay_mode}{decay_value}_"
                                                            f"ent-w{ew}_z{ez}_c{ec}_"
                                                            f"split-{split_flag}_"
                                                            f"lt-{lt_flag}_"
                                                            f"maxmid-{max_mid}_minmid-{min_mid}_"
                                                            f"proto-{proto}_topk-{top_k}_"
                                                            f"memevery-{mem_every}_"
                                                            f"dropfirst-{drop_flag}"
                                                        )
                                                        log_path = os.path.join(args.log_dir, f"{tag}.log")
                                                        output_name = f"{args.output_prefix}_{tag}.nii.gz"
                                                        print(f"[Validate] {tag} log={log_path}")
                                                        if args.dry_run:
                                                            continue
                                                        base_args = _strip_arg(list(remaining), "--output_filename")
                                                        base_args = _strip_arg(base_args, "--output_dir")
                                                        sys.argv = [sys.argv[0]] + base_args + [
                                                            "--output_dir", args.output_dir,
                                                            "--output_filename", output_name,
                                                        ]
                                                        with open(log_path, "w", encoding="utf-8") as log_file:
                                                            with redirect_stdout(log_file), redirect_stderr(log_file):
                                                                run_segmentation()


if __name__ == "__main__":
    main()
