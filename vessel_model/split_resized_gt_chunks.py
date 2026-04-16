import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import h5py
import nibabel as nib
import numpy as np


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


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split a resized/compressed 3D GT volume into chunk files. "
            "Chunk names follow the same z/y/x range convention as sam2_main4_hybrid_tiled."
        )
    )
    parser.add_argument("--input_path", required=True, help="Input GT file (.h5/.nii/.nii.gz)")
    parser.add_argument("--output_dir", required=True, help="Directory to save chunk GT files")
    parser.add_argument(
        "--dataset_key",
        default="main",
        help="H5 dataset key. If missing in input, the first dataset will be used.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="Chunk edge length along z/y/x. Default: 512",
    )
    parser.add_argument(
        "--output_format",
        default="auto",
        choices=["auto", "h5", "nifti"],
        help="Output format. 'auto' follows the input file type. Default: auto",
    )
    parser.add_argument(
        "--output_suffix",
        default="gt",
        help="Suffix appended to each chunk file name. Default: gt",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip chunk files that already exist.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print chunk plan without writing files.",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_chunks(shape: Tuple[int, int, int], chunk_size: int):
    chunks = []
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


def load_volume(path: str, dataset_key: str) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    if path.endswith(".nii.gz") or path.endswith(".nii"):
        img = nib.load(path)
        return np.asarray(img.get_fdata()), img.affine, "nifti"

    with h5py.File(path, "r") as f:
        key = dataset_key if dataset_key in f.keys() else list(f.keys())[0]
        return f[key][:], None, "h5"


def resolve_output_format(input_format: str, requested_format: str) -> str:
    if requested_format == "auto":
        return "nifti" if input_format == "nifti" else "h5"
    return requested_format


def output_extension(output_format: str) -> str:
    return ".nii.gz" if output_format == "nifti" else ".h5"


def build_chunk_affine(affine: Optional[np.ndarray], chunk: ChunkSpec) -> Optional[np.ndarray]:
    if affine is None:
        return None
    out_affine = np.array(affine, copy=True)
    offset = (
        affine[:3, 0] * float(chunk.z0)
        + affine[:3, 1] * float(chunk.y0)
        + affine[:3, 2] * float(chunk.x0)
    )
    out_affine[:3, 3] = out_affine[:3, 3] + offset
    return out_affine


def save_chunk_h5(path: str, chunk_vol: np.ndarray, dataset_key: str) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset(dataset_key, data=chunk_vol, compression="gzip")


def save_chunk_nifti(path: str, chunk_vol: np.ndarray, affine: Optional[np.ndarray]) -> None:
    if affine is None:
        affine = np.eye(4)
    nib.save(nib.Nifti1Image(chunk_vol, affine), path)


def save_chunk(
    path: str,
    chunk_vol: np.ndarray,
    affine: Optional[np.ndarray],
    dataset_key: str,
    output_format: str,
) -> None:
    if output_format == "nifti":
        save_chunk_nifti(path, chunk_vol, affine)
        return
    save_chunk_h5(path, chunk_vol, dataset_key)


def main():
    args = get_args()
    ensure_dir(args.output_dir)

    volume, affine, input_format = load_volume(args.input_path, args.dataset_key)
    output_format = resolve_output_format(input_format, args.output_format)
    ext = output_extension(output_format)
    chunks = build_chunks(tuple(int(v) for v in volume.shape), args.chunk_size)

    print(f"Loaded GT volume: {args.input_path}")
    print(f"Volume shape: {tuple(int(v) for v in volume.shape)}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Output format: {output_format}")
    print(f"Output dir: {args.output_dir}")

    written = 0
    skipped = 0
    for chunk in chunks:
        chunk_filename = f"{chunk.tag}_{args.output_suffix}{ext}"
        chunk_path = os.path.join(args.output_dir, chunk_filename)
        if args.skip_existing and os.path.exists(chunk_path):
            skipped += 1
            continue

        print(f"[CHUNK] {chunk.tag} shape={chunk.shape} -> {chunk_path}")
        if args.dry_run:
            continue

        chunk_vol = volume[chunk.z0:chunk.z1, chunk.y0:chunk.y1, chunk.x0:chunk.x1]
        chunk_affine = build_chunk_affine(affine, chunk)
        save_chunk(chunk_path, chunk_vol, chunk_affine, args.dataset_key, output_format)
        written += 1

    print(f"Done. Written chunks: {written}")
    if args.skip_existing:
        print(f"Skipped existing chunks: {skipped}")


if __name__ == "__main__":
    main()
