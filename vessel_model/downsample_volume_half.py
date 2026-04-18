import argparse
import os
from typing import Optional, Tuple

import h5py
import nibabel as nib
import numpy as np
from scipy import ndimage


def get_args():
    parser = argparse.ArgumentParser(
        description="Read a 3D volume, downsample each axis to half length, and save to a new file."
    )
    parser.add_argument("--input_path", required=True, help="Input 3D volume path (.nii/.nii.gz/.h5)")
    parser.add_argument("--output_path", required=True, help="Output path (.nii/.nii.gz/.h5)")
    parser.add_argument(
        "--dataset_key",
        default="main",
        help="Dataset key for H5 input/output. Falls back to the first dataset if not found in input.",
    )
    parser.add_argument(
        "--mode",
        default="image",
        choices=["image", "label"],
        help="Use linear interpolation for image volumes, nearest interpolation for label volumes.",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="Optional output dtype, e.g. uint8, uint16, float32. Defaults to input dtype.",
    )
    return parser.parse_args()


def load_volume(path: str, dataset_key: str) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    if path.endswith(".nii.gz") or path.endswith(".nii"):
        img = nib.load(path)
        return img.get_fdata(), img.affine, "nifti"

    with h5py.File(path, "r") as f:
        key = dataset_key if dataset_key in f.keys() else list(f.keys())[0]
        return f[key][:], None, key


def build_output_affine_for_half_downsample(affine: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if affine is None:
        return None
    out_affine = np.array(affine, copy=True)
    # Shape halves, so voxel spacing doubles along the three spatial axes.
    out_affine[:3, :3] = out_affine[:3, :3] * 2.0
    return out_affine


def downsample_half(volume: np.ndarray, mode: str) -> np.ndarray:
    order = 1 if mode == "image" else 0
    out = ndimage.zoom(volume, zoom=(0.5, 0.5, 0.5), order=order)
    return out


def maybe_cast_dtype(volume: np.ndarray, dtype_name: Optional[str], input_dtype: np.dtype) -> np.ndarray:
    target_dtype = np.dtype(dtype_name) if dtype_name is not None else input_dtype
    if np.issubdtype(target_dtype, np.integer):
        info = np.iinfo(target_dtype)
        volume = np.clip(volume, info.min, info.max)
        volume = np.rint(volume)
    return volume.astype(target_dtype)


def save_volume(
    path: str,
    volume: np.ndarray,
    affine: Optional[np.ndarray],
    dataset_key: str,
) -> None:
    if path.endswith(".nii.gz") or path.endswith(".nii"):
        if affine is None:
            affine = np.eye(4)
        nib.save(nib.Nifti1Image(volume, affine), path)
        return

    with h5py.File(path, "w") as f:
        f.create_dataset(dataset_key, data=volume, compression="gzip")


def main():
    args = get_args()
    volume, affine, loaded_key = load_volume(args.input_path, args.dataset_key)
    input_dtype = volume.dtype

    downsampled = downsample_half(volume, args.mode)
    downsampled = maybe_cast_dtype(downsampled, args.dtype, input_dtype)

    output_affine = build_output_affine_for_half_downsample(affine)
    output_key = args.dataset_key if args.output_path.endswith(".h5") else loaded_key
    save_volume(args.output_path, downsampled, output_affine, output_key)

    print(f"Input shape: {tuple(int(v) for v in volume.shape)}")
    print(f"Output shape: {tuple(int(v) for v in downsampled.shape)}")
    print(f"Saved downsampled volume to {args.output_path}")


if __name__ == "__main__":
    main()
