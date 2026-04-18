# 基于 sam2_main4 的 VOS respawn 版本：
# - 保留原始 best-axis + 单帧 init mask + 双向 VOS 的流程
# - 若传播过程中 mask 突然消失，则在上一帧 mask 上重采样一个或多个 seed
# - 新 seed 进入 pending 队列后，会重新做三轴选择并继续分割
# - 用 max_segmented_seeds 限制真正执行分割的 seed 数，避免递归爆炸
import argparse
import os
import shutil
import tempfile
from collections import deque
from typing import Deque, List, Set, Tuple

import h5py
import hydra
import numpy as np
import torch
from tqdm import tqdm

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .data_manager import VolumeManager
from .preprocessing import get_multi_axis_init_seg, get_seeds_from_init_seg, get_seg
from .sam2_baseline.image_utils import get_slice, to_uint8_rgb, write_jpeg_frames
from .sam2_baseline.predict_utils import map_local_point

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2_checkpoint", required=True, help="Path to SAM2 checkpoint",
                        default="/home/jiangshuai/code/sam2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2_model_cfg", required=True, help="Path to SAM2 model config",
                        default="/home/jiangshuai/code/sam2/sam2.1_hiera_l.yaml")
    parser.add_argument("--volume_path", required=True, help="Path to .h5 or .nii/.nii.gz file")
    parser.add_argument("--output_dir", default="./bv_seg_output_respawn", help="Directory to save outputs")
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

    parser.add_argument("--enable_respawn", action="store_true",
                        help="When mask disappears during VOS, sample new seed(s) from previous non-empty mask")
    parser.add_argument("--min_respawn_mask_area", type=int, default=20,
                        help="Only respawn when the previous non-empty mask area is at least this value")
    parser.add_argument("--respawn_num_seeds", type=int, default=2,
                        help="Number of new seeds sampled from a disappearing position each time")
    parser.add_argument("--respawn_min_point_distance", type=float, default=24.0,
                        help="Preferred minimum distance between respawned points in the same mask")
    parser.add_argument("--max_respawn_candidates", type=int, default=512,
                        help="Upper bound on valid mask pixels considered for respawn candidate sampling")
    parser.add_argument("--max_segmented_seeds", type=int, default=50,
                        help="Maximum number of seeds that actually run segmentation")

    parser.add_argument("--vos_offload_video_to_cpu", action="store_true",
                        help="Offload video frames to CPU inside SAM2 video predictor (save GPU mem)")
    parser.add_argument("--keep_tmp_vos_frames", action="store_true",
                        help="Do not delete tmp JPEG frames dirs (debug)")
    return parser.parse_args()


def select_mask_with_constraints(masks, scores, max_area):
    if len(masks) == 0:
        return None, None
    order = np.argsort(scores)[::-1]
    for idx in order:
        m = masks[idx].astype(np.uint8)
        area = int(m.sum())
        if area <= 0:
            continue
        if area > max_area:
            continue
        return m, float(scores[idx])
    for idx in order:
        m = masks[idx].astype(np.uint8)
        area = int(m.sum())
        if area > 0:
            return m, float(scores[idx])
    return None, None


def load_or_build_seeds(args, vol_man):
    if args.seed_file:
        seeds = []
        with open(args.seed_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = [int(p) for p in line.strip().split(",")]
                if len(parts) != 3:
                    raise ValueError(f"Invalid seed line: {line}")
                seeds.append(tuple(parts))
        return seeds

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

    return get_seeds_from_init_seg(init_seg)


def local_point_to_global_seed(axis, box, global_frame_idx, local_y, local_x, vol_shape):
    if axis == 0:
        seed = (global_frame_idx, box[1] + local_y, box[3] + local_x)
    elif axis == 1:
        seed = (box[1] + local_y, global_frame_idx, box[3] + local_x)
    else:
        seed = (box[1] + local_y, box[3] + local_x, global_frame_idx)

    z, y, x = seed
    if z < 0 or y < 0 or x < 0 or z >= vol_shape[0] or y >= vol_shape[1] or x >= vol_shape[2]:
        return None
    return int(z), int(y), int(x)


def sample_respawn_seeds_from_mask(prev_mask, axis, box, global_frame_idx, vol_man, num_samples,
                                   min_point_distance, max_candidates):
    ys, xs = np.where(prev_mask > 0)
    if len(ys) == 0 or num_samples <= 0:
        return []

    candidate_points: List[Tuple[int, int]] = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        seed = local_point_to_global_seed(axis, box, global_frame_idx, y, x, vol_man.shape)
        if seed is None:
            continue
        if vol_man.global_mask[seed] > 0:
            continue
        candidate_points.append((y, x))

    if len(candidate_points) == 0:
        return []

    if len(candidate_points) > max_candidates:
        step = max(1, len(candidate_points) // max_candidates)
        candidate_points = candidate_points[::step][:max_candidates]

    pts = np.asarray(candidate_points, dtype=np.float32)
    centroid = pts.mean(axis=0, keepdims=True)
    first_idx = int(np.argmax(np.sum((pts - centroid) ** 2, axis=1)))

    selected_indices = [first_idx]
    min_d2 = np.sum((pts - pts[first_idx]) ** 2, axis=1)
    min_thr2 = float(min_point_distance) * float(min_point_distance)

    while len(selected_indices) < min(num_samples, len(candidate_points)):
        next_idx = int(np.argmax(min_d2))
        if min_d2[next_idx] <= 0:
            break
        if len(selected_indices) > 0 and min_d2[next_idx] < min_thr2:
            remaining = np.where(min_d2 > 0)[0]
            if len(remaining) == 0:
                break
            next_idx = int(remaining[np.argmax(min_d2[remaining])])
            if min_d2[next_idx] <= 0:
                break
        selected_indices.append(next_idx)
        dist2 = np.sum((pts - pts[next_idx]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, dist2)

    seeds: List[Tuple[int, int, int]] = []
    for idx in selected_indices:
        y, x = candidate_points[idx]
        seed = local_point_to_global_seed(axis, box, global_frame_idx, y, x, vol_man.shape)
        if seed is None:
            continue
        if vol_man.global_mask[seed] > 0:
            continue
        seeds.append(seed)
    return seeds


def vos_track_one_direction_with_respawn(
    video_predictor,
    vol_man,
    axis,
    box,
    idx_list,
    init_mask_2d,
    global_update_axis,
    vos_tmp_root,
    offload_video_to_cpu,
    enable_respawn,
    min_respawn_mask_area,
    respawn_num_seeds,
    respawn_min_point_distance,
    max_respawn_candidates,
):
    frames_rgb = []
    valid_idxs = []
    for vidx in idx_list:
        sl = get_slice(vol_man.vol, axis, vidx, box)
        if sl.size == 0:
            break
        rgb = to_uint8_rgb(sl)
        if rgb is None:
            break
        frames_rgb.append(rgb)
        valid_idxs.append(vidx)

    if len(frames_rgb) == 0:
        return []

    spawned_seeds: List[Tuple[int, int, int]] = []
    spawned_set: Set[Tuple[int, int, int]] = set()

    tmp_dir = tempfile.mkdtemp(prefix="sam2_vos_respawn_1", dir=vos_tmp_root)
    write_jpeg_frames(frames_rgb, tmp_dir)

    try:
        with torch.inference_mode(), torch.autocast(str(vol_man.device), dtype=torch.bfloat16):
            state = video_predictor.init_state(
                video_path=tmp_dir,
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=False,
                async_loading_frames=False,
            )

            _, _, masks0 = video_predictor.add_new_mask(
                state,
                frame_idx=0,
                obj_id=1,
                mask=init_mask_2d.astype(bool),
            )

            m0 = masks0[0, 0]
            if torch.is_tensor(m0):
                m0 = (m0 > 0).to(torch.uint8).cpu().numpy()
            else:
                m0 = (m0 > 0).astype(np.uint8)

            if int(m0.sum()) > 0:
                vol_man.update_global_mask(m0, global_update_axis, (valid_idxs[0], *box[1:]))
                prev_nonempty_mask = m0
                prev_global_idx = valid_idxs[0]
            else:
                prev_nonempty_mask = None
                prev_global_idx = None

            for f_idx, _obj_ids, masks in video_predictor.propagate_in_video(state):
                if f_idx < 0 or f_idx >= len(valid_idxs):
                    continue

                mm = masks[0, 0]
                if torch.is_tensor(mm):
                    mm = (mm > 0).to(torch.uint8).cpu().numpy()
                else:
                    mm = (mm > 0).astype(np.uint8)

                gidx = valid_idxs[f_idx]
                if mm.sum() == 0:
                    if enable_respawn and prev_nonempty_mask is not None and prev_global_idx is not None:
                        if int(prev_nonempty_mask.sum()) >= int(min_respawn_mask_area):
                            new_seeds = sample_respawn_seeds_from_mask(
                                prev_mask=prev_nonempty_mask,
                                axis=axis,
                                box=box,
                                global_frame_idx=gidx,
                                vol_man=vol_man,
                                num_samples=respawn_num_seeds,
                                min_point_distance=respawn_min_point_distance,
                                max_candidates=max_respawn_candidates,
                            )
                            for seed in new_seeds:
                                if seed in spawned_set:
                                    continue
                                spawned_set.add(seed)
                                spawned_seeds.append(seed)
                    continue

                vol_man.update_global_mask(mm, global_update_axis, (gidx, *box[1:]))
                prev_nonempty_mask = mm
                prev_global_idx = gidx
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return spawned_seeds


def run_segmentation_with_respawn():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    vol_man = VolumeManager(args.volume_path, key=args.dataset_key)
    if not hasattr(vol_man, "device"):
        vol_man.device = device

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('use_sam2', version_base='1.2')

    sam2_img_model = build_sam2(args.sam2_model_cfg, args.sam2_checkpoint, device=device)
    img_predictor = SAM2ImagePredictor(sam2_img_model)
    video_predictor = build_sam2_video_predictor(args.sam2_model_cfg, args.sam2_checkpoint, device=device)

    initial_seeds = load_or_build_seeds(args, vol_man)
    if not initial_seeds:
        raise RuntimeError("No seeds available for SAM2 tracking.")

    print(f"Starting SAM2 segmentation with respawn over {len(initial_seeds)} initial seeds...")

    pending: Deque[Tuple[int, int, int]] = deque(initial_seeds)
    seen: Set[Tuple[int, int, int]] = set(initial_seeds)

    vos_tmp_root = os.path.join(args.output_dir, "_tmp_sam2_vos")
    os.makedirs(vos_tmp_root, exist_ok=True)

    segmented_count = 0
    skipped_existing = 0
    skipped_invalid = 0
    respawned_count = 0

    pbar = tqdm(total=len(pending), desc="Tracking (SAM2 VOS respawn)")
    while len(pending) > 0:
        seed = pending.popleft()
        pbar.update(1)

        if segmented_count >= args.max_segmented_seeds:
            print(f"[STOP] segmented seeds reached max_segmented_seeds={args.max_segmented_seeds}")
            break

        z, y, x = seed
        if vol_man.global_mask[z, y, x] > 0:
            skipped_existing += 1
            continue

        crops = vol_man.get_triplane_crops(seed)
        best_axis = -1
        best_mask = None
        min_area = float("inf")

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
            skipped_invalid += 1
            continue
        if int(best_mask.sum()) > args.max_init_mask_area:
            print(f"Skip seed {seed}: init mask area {int(best_mask.sum())} > {args.max_init_mask_area}")
            skipped_invalid += 1
            continue

        segmented_count += 1

        track_dim = [0, 1, 2][best_axis]
        start_idx = seed[track_dim]
        max_dist = args.max_track_distance

        forward_idxs = list(range(start_idx, min(start_idx + max_dist, vol_man.shape[track_dim])))
        backward_idxs = list(range(start_idx, max(start_idx - max_dist, -1), -1))

        _, box = crops[best_axis]

        new_fw = vos_track_one_direction_with_respawn(
            video_predictor=video_predictor,
            vol_man=vol_man,
            axis=best_axis,
            box=box,
            idx_list=forward_idxs,
            init_mask_2d=best_mask,
            global_update_axis=best_axis,
            vos_tmp_root=vos_tmp_root,
            offload_video_to_cpu=args.vos_offload_video_to_cpu,
            enable_respawn=args.enable_respawn,
            min_respawn_mask_area=args.min_respawn_mask_area,
            respawn_num_seeds=args.respawn_num_seeds,
            respawn_min_point_distance=args.respawn_min_point_distance,
            max_respawn_candidates=args.max_respawn_candidates,
        )
        new_bw = vos_track_one_direction_with_respawn(
            video_predictor=video_predictor,
            vol_man=vol_man,
            axis=best_axis,
            box=box,
            idx_list=backward_idxs,
            init_mask_2d=best_mask,
            global_update_axis=best_axis,
            vos_tmp_root=vos_tmp_root,
            offload_video_to_cpu=args.vos_offload_video_to_cpu,
            enable_respawn=args.enable_respawn,
            min_respawn_mask_area=args.min_respawn_mask_area,
            respawn_num_seeds=args.respawn_num_seeds,
            respawn_min_point_distance=args.respawn_min_point_distance,
            max_respawn_candidates=args.max_respawn_candidates,
        )

        for ns in (new_fw + new_bw):
            if ns in seen:
                continue
            if vol_man.global_mask[ns] > 0:
                continue
            seen.add(ns)
            pending.append(ns)
            respawned_count += 1
            pbar.total += 1
            pbar.refresh()

    pbar.close()

    if (not args.keep_tmp_vos_frames) and os.path.isdir(vos_tmp_root):
        shutil.rmtree(vos_tmp_root, ignore_errors=True)

    print("=" * 80)
    print(f"Segmented seeds actually run: {segmented_count}")
    print(f"Respawned seeds queued: {respawned_count}")
    print(f"Skipped existing-covered seeds: {skipped_existing}")
    print(f"Skipped invalid seeds: {skipped_invalid}")

    print("Cleanup...")
    final_path = os.path.join(args.output_dir, args.output_filename)
    flag = (args.need_transpose != "False")
    vol_man.save(final_path, flag)
    print(f"Done! Saved to {final_path}")


if __name__ == "__main__":
    run_segmentation_with_respawn()
