import argparse
from typing import Optional, Tuple

import h5py
import nibabel as nib
import numpy as np
from scipy import ndimage


DEFAULT_SOURCE_SHAPE = (624, 3570, 5144)
DEFAULT_TARGET_SHAPE = (1248, 1786, 2572)


def parse_shape(text: str) -> Tuple[int, int, int]:
    parts = [int(x.strip()) for x in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Shape must have exactly 3 integers, got: {text}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Resize a 3D GT volume from the original shape to a compressed target shape. "
            "By default this matches (624,3570,5144) -> (1248,1786,2572)."
        )
    )
    parser.add_argument("--input_path", required=True, help="Input GT file (.nii/.nii.gz/.h5)")
    parser.add_argument("--output_path", required=True, help="Output resized GT file (.nii/.nii.gz/.h5)")
    parser.add_argument(
        "--dataset_key",
        default="main",
        help="H5 dataset key. If missing in input, the first dataset will be used.",
    )
    parser.add_argument(
        "--source_shape",
        default="624,3570,5144",
        help="Expected original GT shape as z,y,x. Default: 624,3570,5144",
    )
    parser.add_argument(
        "--target_shape",
        default="1248,1786,2572",
        help="Target compressed shape as z,y,x. Default: 1248,1786,2572",
    )
    parser.add_argument(
        "--mode",
        default="label",
        choices=["label", "image"],
        help="Use nearest interpolation for labels, linear interpolation for images. Default: label",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="Optional output dtype, e.g. uint8, uint16, float32. Default keeps input dtype.",
    )
    return parser.parse_args()


def load_volume(path: str, dataset_key: str) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    if path.endswith(".nii.gz") or path.endswith(".nii"):
        img = nib.load(path)
        return img.get_fdata(), img.affine, "nifti"

    with h5py.File(path, "r") as f:
        key = dataset_key if dataset_key in f.keys() else list(f.keys())[0]
        return f[key][:], None, key


def maybe_cast_dtype(volume: np.ndarray, dtype_name: Optional[str], input_dtype: np.dtype) -> np.ndarray:
    target_dtype = np.dtype(dtype_name) if dtype_name is not None else input_dtype
    if np.issubdtype(target_dtype, np.integer):
        info = np.iinfo(target_dtype)
        volume = np.clip(volume, info.min, info.max)
        volume = np.rint(volume)
    return volume.astype(target_dtype)


def build_resized_affine(affine: Optional[np.ndarray], zoom_factors: Tuple[float, float, float]) -> Optional[np.ndarray]:
    if affine is None:
        return None
    out_affine = np.array(affine, copy=True)
    # If voxel counts are scaled by zoom_factors, voxel spacing should scale by 1 / zoom_factors.
    for axis in range(3):
        out_affine[:3, axis] = out_affine[:3, axis] / float(zoom_factors[axis])
    return out_affine


def resize_volume_to_shape(
    volume: np.ndarray,
    target_shape: Tuple[int, int, int],
    mode: str,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    zoom_factors = tuple(float(t) / float(s) for s, t in zip(volume.shape, target_shape))
    order = 0 if mode == "label" else 1
    resized = ndimage.zoom(volume, zoom=zoom_factors, order=order)
    if resized.shape != target_shape:
        raise RuntimeError(
            f"Resized shape mismatch: expected {target_shape}, got {tuple(int(v) for v in resized.shape)}"
        )
    return resized, zoom_factors


def save_volume(path: str, volume: np.ndarray, affine: Optional[np.ndarray], dataset_key: str) -> None:
    if path.endswith(".nii.gz") or path.endswith(".nii"):
        if affine is None:
            affine = np.eye(4)
        nib.save(nib.Nifti1Image(volume, affine), path)
        return

    with h5py.File(path, "w") as f:
        f.create_dataset(dataset_key, data=volume, compression="gzip")


def main():
    args = get_args()
    expected_source_shape = parse_shape(args.source_shape)
    target_shape = parse_shape(args.target_shape)

    volume, affine, loaded_key = load_volume(args.input_path, args.dataset_key)
    input_dtype = volume.dtype

    if tuple(int(v) for v in volume.shape) != expected_source_shape:
        raise ValueError(
            f"Input shape {tuple(int(v) for v in volume.shape)} does not match expected source shape {expected_source_shape}"
        )

    resized, zoom_factors = resize_volume_to_shape(volume, target_shape, args.mode)
    resized = maybe_cast_dtype(resized, args.dtype, input_dtype)
    output_affine = build_resized_affine(affine, zoom_factors)

    output_key = args.dataset_key if args.output_path.endswith(".h5") else loaded_key
    save_volume(args.output_path, resized, output_affine, output_key)

    print(f"Input shape: {tuple(int(v) for v in volume.shape)}")
    print(f"Target shape: {target_shape}")
    print(f"Zoom factors: {tuple(float(v) for v in zoom_factors)}")
    print(f"Saved resized GT to {args.output_path}")


if __name__ == "__main__":
    main()
