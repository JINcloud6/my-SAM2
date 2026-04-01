import argparse
import os

import numpy as np
import torch
from scipy import ndimage
from tqdm import tqdm
import hydra
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .data_manager import VolumeManager
from .preprocessing import get_multi_axis_init_seg, get_seeds_from_init_seg, get_seg
from .utils import select_masks


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2_checkpoint", required=True, help="Path to SAM2 checkpoint",default="/home/jiangshuai/code/sam2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2_model_cfg", required=True, help="Path to SAM2 model config",default="/home/jiangshuai/code/sam2/sam2.1_hiera_l.yaml")
    parser.add_argument("--volume_path", required=True, help="Path to .h5 or .nii/.nii.gz file")
    parser.add_argument("--output_dir", default="./bv_seg_output", help="Directory to save outputs")
    parser.add_argument("--output_filename", default="segmentation.nii.gz", help="Output filename")
    parser.add_argument("--dataset_key", default="main", help="Key name in h5 file")
    parser.add_argument("--seed_file", default=None, help="Optional path to seed list (z,y,x per line)")
    parser.add_argument("--axis", type=int, default=3, help="Axis to generate init seg (0=Z,1=Y,2=X), 3=all")
    parser.add_argument("--stride", type=int, default=5, help="Stride for init seg generation")
    parser.add_argument("--gaussian_kernel", type=int, default=5)
    parser.add_argument("--min_bright", type=int, default=40)
    parser.add_argument("--remove_portion", type=float, default=0.98)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--need_transpose", default="False")
    parser.add_argument("--max_track_distance", type=int, default=2000)
    parser.add_argument("--recover_overlap_threshold", type=float, default=0.5)
    return parser.parse_args()


def _extract_components(binary_mask):
    labeled, num = ndimage.label(binary_mask > 0)
    components = []
    if num == 0:
        return components
    for label_id in range(1, num + 1):
        comp_mask = labeled == label_id
        area = int(comp_mask.sum())
        if area == 0:
            continue
        cy, cx = ndimage.center_of_mass(comp_mask)
        if np.isnan(cy) or np.isnan(cx):
            continue
        components.append(
            {
                "mask": comp_mask,
                "center": (float(cy), float(cx)),
                "area": area,
            }
        )
    return components


def _recover_missing_components(
    predictor,
    prev_components,
    curr_mask,
    slice_image,
    overlap_threshold,
):
    if not prev_components:
        return None
    if curr_mask is None:
        curr_mask = np.zeros_like(prev_components[0]["mask"], dtype=np.uint8)
    missing = []
    for comp in prev_components:
        if (curr_mask > 0).any() and (curr_mask & comp["mask"]).any():
            continue
        missing.append(comp)
    if not missing:
        return None
    predictor.set_image(slice_image)
    recovered = np.zeros_like(curr_mask, dtype=np.uint8)
    for comp in missing:
        cy, cx = comp["center"]
        point = np.array([[int(round(cx)), int(round(cy))]])
        masks, scores, _ = predictor.predict(
            point_coords=point,
            point_labels=np.array([1]),
            multimask_output=True,
        )
        cand_mask, _ = select_masks(masks, scores, thr=0.9, crit="max", max_size=5000, min_circularity=0.6)
        if cand_mask is None:
            continue
        overlap = (cand_mask & (curr_mask > 0)).sum()
        ratio = overlap / (cand_mask.sum() + 1e-6)
        if ratio > overlap_threshold:
            continue
        recovered = np.logical_or(recovered, cand_mask)
    if recovered.sum() == 0:
        return None
    return recovered.astype(np.uint8)


def _predict_from_point(predictor, image, point):
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        point_coords=np.array([point]),
        point_labels=np.array([1]),
        multimask_output=True,
    )
    mask, _ = select_masks(masks, scores, thr=0.9, crit="max", max_size=5000, min_circularity=0.6)
    if mask is None:
        return None
    return mask.astype(np.uint8)


def _predict_from_mask(predictor, image, mask):
    predictor.set_image(image)
    masks, _, _ = predictor.predict(
        mask_input=mask[None, ...],
        multimask_output=False,
    )
    if masks is None or len(masks) == 0:
        return None
    return masks[0].astype(np.uint8)


def _map_local_point(seed, axis, box):
    if axis == 0:
        return [seed[2] - box[3], seed[1] - box[1]]
    if axis == 1:
        return [seed[2] - box[3], seed[0] - box[1]]
    return [seed[1] - box[3], seed[0] - box[1]]


def _get_slice(volume, axis, curr_idx, box):
    if axis == 0:
        sl = volume[curr_idx, box[1]:box[2], box[3]:box[4]]
    elif axis == 1:
        sl = volume[box[1]:box[2], curr_idx, box[3]:box[4]]
    else:
        sl = volume[box[1]:box[2], box[3]:box[4], curr_idx]
    return sl


def run_segmentation():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    vol_man = VolumeManager(args.volume_path, key=args.dataset_key)
    hydra.core.global_hydra.GlobalHydra.instance().clear()
# reinit hydra with a new search path for configs
    hydra.initialize_config_module('use_sam2', version_base='1.2')
    sam2 = build_sam2(args.sam2_model_cfg, args.sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2)

    seeds = []
    if args.seed_file:
        with open(args.seed_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = [int(p) for p in line.strip().split(",")]
                if len(parts) != 3:
                    raise ValueError(f"Invalid seed line: {line}")
                seeds.append(tuple(parts))
    else:
        if args.axis in (0, 1, 2):
            axis = args.axis
            init_seg = np.zeros_like(vol_man.vol, dtype=np.uint8)
            if axis == 0:
                for m in tqdm(range(0, vol_man.shape[0], args.stride), desc="Axis 0 (Z)"):
                    image = vol_man.vol[m, :, :]
                    temp_seg = get_seg(
                        image,
                        remove_portion=args.remove_portion,
                        gaussian_kernel=args.gaussian_kernel,
                        min_bright=args.min_bright,
                    )
                    init_seg[m, :, :][temp_seg > 0] = 1
            elif axis == 1:
                for m in tqdm(range(0, vol_man.shape[1], args.stride), desc="Axis 1 (Y)"):
                    image = vol_man.vol[:, m, :]
                    temp_seg = get_seg(
                        image,
                        remove_portion=args.remove_portion,
                        gaussian_kernel=args.gaussian_kernel,
                        min_bright=args.min_bright,
                    )
                    init_seg[:, m, :][temp_seg > 0] = 1
            else:
                for m in tqdm(range(0, vol_man.shape[2], args.stride), desc="Axis 2 (X)"):
                    image = vol_man.vol[:, :, m]
                    temp_seg = get_seg(
                        image,
                        remove_portion=args.remove_portion,
                        gaussian_kernel=args.gaussian_kernel,
                        min_bright=args.min_bright,
                    )
                    init_seg[:, :, m][temp_seg > 0] = 1
        else:
            init_seg = get_multi_axis_init_seg(
                vol_man.vol,
                stride=args.stride,
                thr=args.remove_portion,
                gaussian_kernel=args.gaussian_kernel,
                min_bright=args.min_bright,
            )
        seeds = get_seeds_from_init_seg(init_seg)
    if not seeds:
        raise RuntimeError("No seeds available for SAM2 tracking.")

    print(f"Starting SAM2 segmentation with {len(seeds)} seeds...")
    seed_index = 0
    with tqdm(total=len(seeds), desc="Tracking") as pbar:
        while seed_index < len(seeds):
            seed = seeds[seed_index]
            seed_index += 1
            pbar.update(1)
            z, y, x = seed
            if vol_man.global_mask[z, y, x] > 0:
                continue

            crops = vol_man.get_triplane_crops(seed)
            best_axis = -1
            best_mask = None
            min_area = float("inf")

            for axis in [0, 1, 2]:
                img, box = crops[axis]
                local_pt = _map_local_point(seed, axis, box)
                mask = _predict_from_point(predictor, img, local_pt)
                if mask is None:
                    continue
                area = mask.sum()
                if area < min_area:
                    min_area = area
                    best_axis = axis
                    best_mask = mask

            if best_axis == -1 or best_mask is None:
                continue

            track_dim = [0, 1, 2][best_axis]
            start_idx = seed[track_dim]
            max_dist = args.max_track_distance
            sequences = [
                range(start_idx, min(start_idx + max_dist, vol_man.shape[track_dim])),
                range(start_idx, max(start_idx - max_dist, -1), -1),
            ]
            _, box = crops[best_axis]

            for seq in sequences:
                if not seq:
                    continue
                first_frame = True
                prev_components = None
                prev_mask = None
                for curr_idx in seq:
                    sl = _get_slice(vol_man.vol, best_axis, curr_idx, box)
                    if sl.size == 0:
                        break
                    rgb = np.stack([sl] * 3, axis=-1)
                    if first_frame:
                        pred = best_mask
                        first_frame = False
                    else:
                        if prev_mask is None or prev_mask.sum() == 0:
                            break
                        pred = _predict_from_mask(predictor, rgb, prev_mask)
                        if pred is None:
                            pred = np.zeros_like(prev_mask, dtype=np.uint8)

                    if prev_components:
                        recovered = _recover_missing_components(
                            predictor,
                            prev_components,
                            pred,
                            rgb,
                            args.recover_overlap_threshold,
                        )
                        if recovered is not None:
                            combined = ((pred > 0) | (recovered > 0)).astype(np.uint8)
                            pred = _predict_from_mask(predictor, rgb, combined)
                            if pred is None:
                                pred = combined

                    if pred is None or pred.sum() == 0:
                        break

                    vol_man.update_global_mask(pred, best_axis, (curr_idx, *box[1:]))
                    prev_mask = pred
                    prev_components = _extract_components(pred)

    print("Cleanup...")
    # vol_man.clean_up()
    file_name = args.output_filename
    final_path = os.path.join(args.output_dir, file_name)
    need_transpose = args.need_transpose
    flag = True
    if need_transpose == "False":
        flag = False
    vol_man.save(final_path, flag)
    print(f"Done! Saved to {final_path}")


if __name__ == "__main__":
    run_segmentation()
