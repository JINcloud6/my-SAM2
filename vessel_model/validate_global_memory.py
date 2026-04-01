import argparse
import os
import sys
from contextlib import redirect_stdout, redirect_stderr

from vessel_model.main import run_segmentation, xmem_config


def parse_list(value, cast=str):
    items = [v.strip() for v in value.split(",") if v.strip()]
    return [cast(v) for v in items]


def main():
    parser = argparse.ArgumentParser(description="Validate global memory injection strategies.")
    parser.add_argument("--methods", default="similarity",
                        help="Comma-separated methods: all, nearest, similarity")
    parser.add_argument("--ks", default="150,200,400,1000,2000",
                        help="Comma-separated top-k values")
    parser.add_argument("--output_dir", default="./bv_seg_output",
                        help="Output directory for segmentation results")
    parser.add_argument("--log_dir", default="global_memory_logs",
                        help="Directory to store per-run logs")
    parser.add_argument("--output_prefix", default="global_memory",
                        help="Prefix for output nii.gz files")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only print configurations without running segmentation")
    args, remaining = parser.parse_known_args()

    methods = parse_list(args.methods, cast=str)
    ks = parse_list(args.ks, cast=int)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    def _strip_arg(argv, flag):
        if flag not in argv:
            return argv
        idx = argv.index(flag)
        if idx < len(argv) - 1:
            return argv[:idx] + argv[idx+2:]
        return argv[:idx]

    for method in methods:
        for k in ks:
            xmem_config["enable_global_memory"] = True
            xmem_config["global_mem_select_method"] = method
            xmem_config["global_mem_topk"] = k
            log_name = f"global_memory_{method}_topk{k}.log"
            log_path = os.path.join(args.log_dir, log_name)
            output_name = f"{args.output_prefix}_{method}_topk{k}.nii.gz"
            print(f"[GlobalMemory] method={method} topk={k} log={log_path}")
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
