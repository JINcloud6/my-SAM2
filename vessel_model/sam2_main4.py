# main2 的原版：只使用视频追踪分割，不去判别新种子
import argparse
import os
import shutil

import h5py
import numpy as np
import torch
from tqdm import tqdm
import hydra

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .data_manager import VolumeManager
from .preprocessing import get_multi_axis_init_seg, get_seeds_from_init_seg, get_seg
from .sam2_baseline.predict_utils import map_local_point
from .sam2_baseline.tracking import vos_track_one_direction
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2_checkpoint", required=True, help="Path to SAM2 checkpoint",
                        default="/home/jiangshuai/code/sam2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2_model_cfg", required=True, help="Path to SAM2 model config",
                        default="/home/jiangshuai/code/sam2/sam2.1_hiera_l.yaml")
    parser.add_argument("--volume_path", required=True, help="Path to .h5 or .nii/.nii.gz file")
    parser.add_argument("--output_dir", default="./bv_seg_output", help="Directory to save outputs")
    parser.add_argument("--output_filename", default="segmentation.nii.gz", help="Output filename")
    parser.add_argument("--dataset_key", default="main", help="Key name in h5 file")
    parser.add_argument("--seed_file", default=None, help="Optional path to seed list (z,y,x per line)")
    parser.add_argument("--init_seg_path", default=None, help="Optional cached init-seg h5 path")
    parser.add_argument("--axis", type=int, default=3, help="Axis to generate init seg (0=Z,1=Y,2=X), 3=all")
    parser.add_argument("--stride", type=int, default=5, help="Stride for init seg generation")
    parser.add_argument("--gaussian_kernel", type=int, default=5)
    parser.add_argument("--min_bright", type=int, default=40)
    parser.add_argument("--remove_portion", type=float, default=0.98)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--need_transpose", default="False")
    parser.add_argument("--max_track_distance", type=int, default=2000)
    parser.add_argument("--recover_overlap_threshold", type=float, default=0.5)
    parser.add_argument("--max_init_mask_area", type=int, default=12000,
                        help="Skip seed when chosen init mask area is larger than this")

    # video predictor / io knobs
    parser.add_argument("--vos_offload_video_to_cpu", action="store_true",
                        help="Offload video frames to CPU inside SAM2 video predictor (save GPU mem)")
    parser.add_argument("--keep_tmp_vos_frames", action="store_true",
                        help="Do not delete tmp JPEG frames dirs (debug)")

    return parser.parse_args()


def select_mask_with_constraints(masks, scores, max_area):
    if len(masks) == 0:
        return None, None
    order = np.argsort(scores)[::-1]
    best_mask, best_score, best_area = None, -1.0, None
    for idx in order:
        m = masks[idx].astype(np.uint8)
        area = int(m.sum())
        if area <= 0:
            continue
        if area > max_area:
            continue
        return m, float(scores[idx])
    # fallback: return highest-score positive mask for logging/threshold skip
    for idx in order:
        m = masks[idx].astype(np.uint8)
        area = int(m.sum())
        if area > 0:
            return m, float(scores[idx])
    return None, None


def run_segmentation():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    vol_man = VolumeManager(args.volume_path, key=args.dataset_key)
    # 如果你的 VolumeManager 里没有 device 字段，给它补一个
    if not hasattr(vol_man, "device"):
        vol_man.device = device

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('use_sam2', version_base='1.2')

    # --- build SAM2 image predictor (for best_axis & initial mask) ---
    sam2_img_model = build_sam2(args.sam2_model_cfg, args.sam2_checkpoint, device=device)
    img_predictor = SAM2ImagePredictor(sam2_img_model)

    # --- build SAM2 video predictor (for VOS tracking) ---
    video_predictor = build_sam2_video_predictor(args.sam2_model_cfg, args.sam2_checkpoint, device=device)

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
        default_init_seg_name = f"init_seg_axis{args.axis}_s{args.stride}_t{args.remove_portion}.h5"
        init_seg_path = args.init_seg_path or os.path.join(args.output_dir, default_init_seg_name)

        if os.path.exists(init_seg_path):
            print(f"Loading existing init_seg from {init_seg_path}...")
            with h5py.File(init_seg_path, "r") as f:
                init_seg = f["main"][:]
        else:
            if args.axis in (0, 1, 2):
                axis = args.axis
                init_seg = np.zeros_like(vol_man.vol, dtype=np.uint8)
                if axis == 0:
                    for m in tqdm(range(0, vol_man.shape[0], args.stride), desc="Axis 0 (Z)"):
                        image = vol_man.vol[m, :, :]
                        temp_seg = get_seg(image, remove_portion=args.remove_portion,
                                           gaussian_kernel=args.gaussian_kernel, min_bright=args.min_bright)
                        init_seg[m, :, :][temp_seg > 0] = 1
                elif axis == 1:
                    for m in tqdm(range(0, vol_man.shape[1], args.stride), desc="Axis 1 (Y)"):
                        image = vol_man.vol[:, m, :]
                        temp_seg = get_seg(image, remove_portion=args.remove_portion,
                                           gaussian_kernel=args.gaussian_kernel, min_bright=args.min_bright)
                        init_seg[:, m, :][temp_seg > 0] = 1
                else:
                    for m in tqdm(range(0, vol_man.shape[2], args.stride), desc="Axis 2 (X)"):
                        image = vol_man.vol[:, :, m]
                        temp_seg = get_seg(image, remove_portion=args.remove_portion,
                                           gaussian_kernel=args.gaussian_kernel, min_bright=args.min_bright)
                        init_seg[:, :, m][temp_seg > 0] = 1
            else:
                init_seg = get_multi_axis_init_seg(
                    vol_man.vol,
                    stride=args.stride,
                    thr=args.remove_portion,
                    gaussian_kernel=args.gaussian_kernel,
                    min_bright=args.min_bright,
                )
            print(f"Saving init_seg to {init_seg_path}...")
            with h5py.File(init_seg_path, "w") as f:
                f.create_dataset("main", data=init_seg, compression="gzip")

        seeds = get_seeds_from_init_seg(init_seg)

    if not seeds:
        raise RuntimeError("No seeds available for SAM2 tracking.")

    print(f"Starting SAM2 segmentation with {len(seeds)} seeds...")

    # tmp root for VOS frames
    vos_tmp_root = os.path.join(args.output_dir, "_tmp_sam2_vos")
    os.makedirs(vos_tmp_root, exist_ok=True)

    with tqdm(total=len(seeds), desc="Tracking (SAM2 VOS)") as pbar:
        for seed in seeds:
            pbar.update(1)
            z, y, x = seed
            if vol_man.global_mask[z, y, x] > 0:
                continue

            crops = vol_man.get_triplane_crops(seed)
            best_axis = -1
            best_mask = None
            min_area = float("inf")

            # 1) per-axis init segmentation on the seed slice (still image)
            with torch.inference_mode(), torch.autocast(args.device, dtype=torch.bfloat16):
                for axis in [0, 1, 2]:
                    img, box = crops[axis]
                    local_pt = map_local_point(seed, axis, box)
                    img_predictor.set_image(img)
                    masks, scores, _ = img_predictor.predict(
                        point_coords=np.array([local_pt]),
                        point_labels=np.array([1]),
                        multimask_output=True,
                    )
                    mask, _ = select_mask_with_constraints(masks, scores, args.max_init_mask_area)
                    if mask is None:
                        continue
                    area = int(mask.sum())
                    if area < min_area:
                        min_area = area
                        best_axis = axis
                        best_mask = mask

            if best_axis == -1 or best_mask is None or best_mask.sum() == 0:
                continue
            if int(best_mask.sum()) > args.max_init_mask_area:
                print(f"Skip seed {seed}: init mask area {int(best_mask.sum())} > {args.max_init_mask_area}")
                continue

            # 2) build index lists (forward/backward) around the seed on the chosen axis
            track_dim = [0, 1, 2][best_axis]
            start_idx = seed[track_dim]
            max_dist = args.max_track_distance

            # IMPORTANT: idx_list[0] corresponds to the frame where best_mask was predicted
            forward_idxs = list(range(start_idx, min(start_idx + max_dist, vol_man.shape[track_dim])))
            backward_idxs = list(range(start_idx, max(start_idx - max_dist, -1), -1))

            _, box = crops[best_axis]

            # 3) VOS tracking using SAM2VideoPredictor
            # forward
            vos_track_one_direction(
                video_predictor=video_predictor,
                vol_man=vol_man,
                axis=best_axis,
                box=box,
                idx_list=forward_idxs,
                init_mask_2d=best_mask,
                global_update_axis=best_axis,
                vos_tmp_root=vos_tmp_root,
                offload_video_to_cpu=args.vos_offload_video_to_cpu,
            )
            # backward
            vos_track_one_direction(
                video_predictor=video_predictor,
                vol_man=vol_man,
                axis=best_axis,
                box=box,
                idx_list=backward_idxs,
                init_mask_2d=best_mask,
                global_update_axis=best_axis,
                vos_tmp_root=vos_tmp_root,
                offload_video_to_cpu=args.vos_offload_video_to_cpu,
            )

    if (not args.keep_tmp_vos_frames) and os.path.isdir(vos_tmp_root):
        shutil.rmtree(vos_tmp_root, ignore_errors=True)

    print("Cleanup...")
    final_path = os.path.join(args.output_dir, args.output_filename)
    flag = (args.need_transpose != "False")
    vol_man.save(final_path, flag)
    print(f"Done! Saved to {final_path}")


if __name__ == "__main__":
    run_segmentation()
