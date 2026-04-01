import argparse
import os

import cc3d
import nibabel as nib
import numpy as np
from scipy import ndimage


def get_args():
    parser = argparse.ArgumentParser(description="Clean 3D segmentation by hole-filling + CC filtering.")
    parser.add_argument("--input_seg", required=True, help="Input segmentation nii/nii.gz")
    parser.add_argument("--output_seg", required=True, help="Output cleaned segmentation nii/nii.gz")
    parser.add_argument("--min_voxels", type=int, default=80000, help="Remove components smaller than this")
    parser.add_argument("--max_dim_min", type=int, default=10, help="Remove if max(h,w,d) < this")
    parser.add_argument("--min_dim_min", type=int, default=3, help="Remove if min(h,w,d) < this")
    parser.add_argument("--min_ratio", type=float, default=0.01, help="Remove if min/max ratio < this")
    parser.add_argument("--connectivity", type=int, default=6, choices=[6, 18, 26])
    return parser.parse_args()


def clean_segmentation(labels, min_voxels, max_dim_min, min_dim_min, min_ratio, connectivity):
    labels = (labels > 0).astype(np.uint8)

    # 2D per-slice hole filling (z-axis), matching your previous logic
    for z in range(labels.shape[0]):
        labels[z] = ndimage.binary_fill_holes(labels[z]).astype(np.uint8)

    labels_out = cc3d.connected_components(labels, connectivity=connectivity)
    stats = cc3d.statistics(labels_out)

    voxel_counts = stats["voxel_counts"]
    bbx = stats["bounding_boxes"]

    keep = np.ones(len(voxel_counts), dtype=bool)
    keep[0] = False  # background

    for label_id in range(1, len(voxel_counts)):
        count = int(voxel_counts[label_id])
        if count < min_voxels:
            keep[label_id] = False
            continue

        box = bbx[label_id]
        h = box[0].stop - box[0].start
        w = box[1].stop - box[1].start
        d = box[2].stop - box[2].start

        ratio = float(min(d, h, w)) / float(max(d, h, w) + 1e-8)
        if max(h, w, d) < max_dim_min or min(h, w, d) < min_dim_min or ratio < min_ratio:
            keep[label_id] = False

    keep_lut = keep.astype(np.uint8)
    cleaned = keep_lut[labels_out]
    return cleaned.astype(np.uint8)


def main():
    args = get_args()
    img = nib.load(args.input_seg)
    labels = img.get_fdata()

    cleaned = clean_segmentation(
        labels=labels,
        min_voxels=args.min_voxels,
        max_dim_min=args.max_dim_min,
        min_dim_min=args.min_dim_min,
        min_ratio=args.min_ratio,
        connectivity=args.connectivity,
    )

    os.makedirs(os.path.dirname(args.output_seg) or ".", exist_ok=True)
    nib.save(nib.Nifti1Image(cleaned, img.affine), args.output_seg)
    print(f"Saved cleaned segmentation to: {args.output_seg}")


if __name__ == "__main__":
    main()
