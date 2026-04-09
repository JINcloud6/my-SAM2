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
    return f"init_seg_axis{args.axis}_s{args.stride}_t{args.remove_portion}.h5"


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
    parser.add_argument("--max_slice_mask_area", type=int, default=12000)
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
        "--sam2_checkpoint",
        args.sam2_checkpoint,
        "--sam2_model_cfg",
        args.sam2_model_cfg,
        "--volume_path",
        chunk_volume_path,
        "--output_dir",
        chunk_output_dir,
        "--output_filename",
        chunk_output_filename,
        "--axis_sequence_cache_root",
        axis_sequence_cache_root,
        "--feature_cache_device",
        args.feature_cache_device,
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
        "--cuda_device",
        str(args.cuda_device),
        "--device",
        args.device,
        "--max_slice_mask_area",
        str(args.max_slice_mask_area),
        "--need_transpose",
        "False",
    ]
    if args.enable_axis_feature_cache:
        cmd.append("--enable_axis_feature_cache")
    if chunk_seed_path is not None:
        cmd.extend(["--seed_file", chunk_seed_path])
    if chunk_init_seg_path is not None:
        cmd.extend(["--init_seg_path", chunk_init_seg_path])
    cmd.extend(passthrough_args)
    return cmd


def main():
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


if __name__ == "__main__":
    main()
