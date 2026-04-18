#迁移allmouse前缀目录下的初始分割等到共享缓存区，确保可以复用
import argparse
import os
import re
import shutil
from typing import List


CHUNK_DIR_RE = re.compile(
    r"^chunk_\d+_z\d+_\d+_y\d+_\d+_x\d+_\d+$"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_run_dir",
        default="/home/jiangshuai/code/XMem/data/allmouse/chunks/03",
        help="Existing tiled run directory, e.g. .../chunks/03",
    )
    parser.add_argument(
        "--shared_cache_root",
        default="/home/jiangshuai/code/XMem/data/allmouse/shared_chunk_cache",
        help="Shared cache root used by sam2_main4_hybrid_tiled.py",
    )
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--axis", type=int, default=0)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--remove_portion", type=float, default=0.98)
    parser.add_argument("--gaussian_kernel", type=int, default=21)
    parser.add_argument("--min_bright", type=int, default=40)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def shared_auto_init_seg_basename(args) -> str:
    return (
        f"init_seg_axis{args.axis}"
        f"_s{args.stride}"
        f"_t{args.remove_portion}"
        f"_g{args.gaussian_kernel}"
        f"_mb{args.min_bright}.h5"
    )


def list_chunk_dirs(source_run_dir: str) -> List[str]:
    chunk_dirs: List[str] = []
    for name in sorted(os.listdir(source_run_dir)):
        path = os.path.join(source_run_dir, name)
        if os.path.isdir(path) and CHUNK_DIR_RE.match(name):
            chunk_dirs.append(path)
    return chunk_dirs


def maybe_copy(src: str, dst: str, dry_run: bool) -> bool:
    if not os.path.exists(src):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if dry_run:
        print(f"[DRY-RUN COPY] {src} -> {dst}")
        return True
    if not os.path.exists(dst):
        shutil.copy2(src, dst)
        print(f"[COPY] {src} -> {dst}")
    else:
        print(f"[SKIP] exists: {dst}")
    return True


def main():
    args = parse_args()
    shared_root = os.path.join(args.shared_cache_root, f"chunk_size_{args.chunk_size}")
    os.makedirs(shared_root, exist_ok=True)

    chunk_dirs = list_chunk_dirs(args.source_run_dir)
    if len(chunk_dirs) == 0:
        raise RuntimeError(f"No chunk directories found in: {args.source_run_dir}")

    copied_volumes = 0
    copied_init_seg = 0

    for chunk_dir in chunk_dirs:
        chunk_tag = os.path.basename(chunk_dir)
        shared_chunk_dir = os.path.join(shared_root, chunk_tag)
        os.makedirs(shared_chunk_dir, exist_ok=True)

        volume_candidates = [
            name for name in os.listdir(chunk_dir)
            if name.endswith("_volume.nii.gz")
        ]
        for name in sorted(volume_candidates):
            src = os.path.join(chunk_dir, name)
            dst = os.path.join(shared_chunk_dir, f"{chunk_tag}_volume.nii.gz")
            if maybe_copy(src, dst, args.dry_run):
                copied_volumes += 1
                break

        init_seg_candidates = [
            name for name in os.listdir(chunk_dir)
            if name.startswith("init_seg_") and name.endswith(".h5")
        ]
        for name in sorted(init_seg_candidates):
            src = os.path.join(chunk_dir, name)
            dst = os.path.join(shared_chunk_dir, shared_auto_init_seg_basename(args))
            if maybe_copy(src, dst, args.dry_run):
                copied_init_seg += 1

        cropped_init_seg_candidates = [
            name for name in os.listdir(chunk_dir)
            if name.endswith("_init_seg.nii.gz")
        ]
        for name in sorted(cropped_init_seg_candidates):
            src = os.path.join(chunk_dir, name)
            dst = os.path.join(shared_chunk_dir, "global_init_seg.nii.gz")
            if maybe_copy(src, dst, args.dry_run):
                copied_init_seg += 1
                break

    print("=" * 80)
    print(f"Source run dir: {args.source_run_dir}")
    print(f"Shared cache dir: {shared_root}")
    print(f"Chunk dirs found: {len(chunk_dirs)}")
    print(f"Volume files copied or detected: {copied_volumes}")
    print(f"Init-seg files copied or detected: {copied_init_seg}")


if __name__ == "__main__":
    main()
