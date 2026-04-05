# 基于 sam2_main4 的 segment-trust respawn 版本：
# - 维护两套 3D mask：
#   1) covered_mask: 体素已被某次分割覆盖
#   2) trusted_mask: 体素已被“方向一致”的血管段可靠解释
# - seed 是否跳过只看 trusted_mask，不看 covered_mask
# - VOS 过程中按固定长度切分 segment；若该段主方向与当前分割轴一致，则提升到 trusted_mask
# - 若不一致，则该段只写入 covered_mask，并在该段上做 FPS 重采样若干 seed，重新进行轴向选择
import argparse
import time
import uuid
import os
import shutil
import tempfile
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Set, Tuple

import h5py
import hydra
import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .data_manager import VolumeManager
from .preprocessing import get_multi_axis_init_seg, get_seeds_from_init_seg, get_seg
from .sam2_baseline.image_utils import get_slice, to_uint8_rgb, write_jpeg_frames
from .sam2_baseline.predict_utils import map_local_point

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
EPS = 1e-6


@dataclass
class SegmentFrame:
    axis: int
    global_frame_idx: int
    box: Tuple[int, int, int, int, int]
    mask: np.ndarray


@dataclass
class SegmentDecision:
    is_trusted: bool
    dominant_axis: int
    axis_ratio: float
    spans: Tuple[int, int, int]


@dataclass
class SeedTask:
    seed: Tuple[int, int, int]
    is_original: bool
    untrusted_count: int


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2_checkpoint", required=True, help="Path to SAM2 checkpoint",
                        default="/home/jiangshuai/code/sam2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2_model_cfg", required=True, help="Path to SAM2 model config",
                        default="/home/jiangshuai/code/sam2/sam2.1_hiera_l.yaml")
    parser.add_argument("--volume_path", required=True, help="Path to .h5 or .nii/.nii.gz file")
    parser.add_argument("--output_dir", default="./bv_seg_output_segment_trust", help="Directory to save outputs")
    parser.add_argument("--output_filename", default="segmentation.nii.gz", help="Trusted-mask output filename")
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
    parser.add_argument("--max_init_mask_area", type=int, default=12000,
                        help="Skip seed when chosen init mask area is larger than this")

    parser.add_argument("--segment_len", type=int, default=15,
                        help="Judge trustworthiness after each segment of this many propagated frames")
    parser.add_argument("--min_segment_frames", type=int, default=5,
                        help="Minimum number of frames required to evaluate a tail segment")
    parser.add_argument("--axis_ratio_thr", type=float, default=1.1,
                        help="Trusted if extent along tracking axis >= this ratio times the second-largest extent")
    parser.add_argument("--segment_respawn_num_seeds", type=int, default=4,
                        help="How many new seeds to sample from an untrusted segment")
    parser.add_argument("--segment_respawn_min_distance", type=float, default=20.0,
                        help="Preferred minimum FPS spacing between respawned seeds on an untrusted segment")
    parser.add_argument("--max_segment_respawn_candidates", type=int, default=2048,
                        help="Upper bound on candidate voxels used for segment-level respawn sampling")
    parser.add_argument("--max_untrusted_segments_per_lineage", type=int, default=1,
                        help="A seed lineage is allowed to trigger at most this many untrusted segments before forcing trust")
    parser.add_argument("--max_segmented_seeds", type=int, default=20,
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
        if int(m.sum()) > 0:
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


def update_mask_volume(mask_volume, local_mask, axis, box):
    idx, d1_min, d1_max, d2_min, d2_max = box
    local_mask = local_mask.astype(np.uint8)
    local_mask = local_mask[:(d1_max - d1_min), :(d2_max - d2_min)]
    if axis == 0:
        mask_volume[idx, d1_min:d1_max, d2_min:d2_max] |= local_mask
    elif axis == 1:
        mask_volume[d1_min:d1_max, idx, d2_min:d2_max] |= local_mask
    else:
        mask_volume[d1_min:d1_max, d2_min:d2_max, idx] |= local_mask


def append_segment_voxels(coords_acc, frame: SegmentFrame):
    ys, xs = np.where(frame.mask > 0)
    if len(ys) == 0:
        return

    if frame.axis == 0:
        zs = np.full_like(ys, frame.global_frame_idx)
        gy = frame.box[1] + ys
        gx = frame.box[3] + xs
        coords = np.stack([zs, gy, gx], axis=1)
    elif frame.axis == 1:
        gz = frame.box[1] + ys
        ys_global = np.full_like(ys, frame.global_frame_idx)
        gx = frame.box[3] + xs
        coords = np.stack([gz, ys_global, gx], axis=1)
    else:
        gz = frame.box[1] + ys
        gy = frame.box[3] + xs
        xs_global = np.full_like(ys, frame.global_frame_idx)
        coords = np.stack([gz, gy, xs_global], axis=1)
    coords_acc.append(coords.astype(np.int32))


def judge_segment_direction(seg_frames: List[SegmentFrame], track_axis: int, axis_ratio_thr: float) -> SegmentDecision:
    coords_acc = []
    for item in seg_frames:
        append_segment_voxels(coords_acc, item)

    if len(coords_acc) == 0:
        return SegmentDecision(is_trusted=False, dominant_axis=-1, axis_ratio=0.0, spans=(0, 0, 0))

    coords = np.concatenate(coords_acc, axis=0)
    min_xyz = coords.min(axis=0)
    max_xyz = coords.max(axis=0)
    spans = (max_xyz - min_xyz + 1).astype(np.int32)
    dominant_axis = int(np.argmax(spans))

    other_axes = [a for a in (0, 1, 2) if a != track_axis]
    second_extent = max(float(spans[other_axes[0]]), float(spans[other_axes[1]]))
    axis_ratio = float(spans[track_axis]) / max(second_extent, EPS)
    is_trusted = (dominant_axis == track_axis) and (axis_ratio >= axis_ratio_thr)

    return SegmentDecision(
        is_trusted=bool(is_trusted),
        dominant_axis=dominant_axis,
        axis_ratio=float(axis_ratio),
        spans=(int(spans[0]), int(spans[1]), int(spans[2])),
    )


def sample_respawn_seeds_from_segment(seg_frames: List[SegmentFrame], trusted_mask: np.ndarray, num_samples: int,
                                      min_distance: float, max_candidates: int):
    if len(seg_frames) == 0 or num_samples <= 0:
        return []

    candidate_coords = []
    for item in seg_frames:
        ys, xs = np.where(item.mask > 0)
        if len(ys) == 0:
            continue

        if item.axis == 0:
            coords = np.stack([
                np.full_like(ys, item.global_frame_idx),
                item.box[1] + ys,
                item.box[3] + xs,
            ], axis=1)
        elif item.axis == 1:
            coords = np.stack([
                item.box[1] + ys,
                np.full_like(ys, item.global_frame_idx),
                item.box[3] + xs,
            ], axis=1)
        else:
            coords = np.stack([
                item.box[1] + ys,
                item.box[3] + xs,
                np.full_like(ys, item.global_frame_idx),
            ], axis=1)

        for z, y, x in coords.tolist():
            if trusted_mask[z, y, x] > 0:
                continue
            candidate_coords.append((z, y, x))

    if len(candidate_coords) == 0:
        return []

    candidate_coords = list(dict.fromkeys(candidate_coords))
    if len(candidate_coords) > max_candidates:
        step = max(1, len(candidate_coords) // max_candidates)
        candidate_coords = candidate_coords[::step][:max_candidates]

    pts = np.asarray(candidate_coords, dtype=np.float32)
    centroid = pts.mean(axis=0, keepdims=True)
    first_idx = int(np.argmax(np.sum((pts - centroid) ** 2, axis=1)))

    selected_indices = [first_idx]
    min_d2 = np.sum((pts - pts[first_idx]) ** 2, axis=1)
    min_thr2 = float(min_distance) * float(min_distance)

    while len(selected_indices) < min(num_samples, len(candidate_coords)):
        next_idx = int(np.argmax(min_d2))
        if min_d2[next_idx] <= 0:
            break
        if min_d2[next_idx] < min_thr2:
            remaining = np.where(min_d2 > 0)[0]
            if len(remaining) == 0:
                break
            next_idx = int(remaining[np.argmax(min_d2[remaining])])
            if min_d2[next_idx] <= 0:
                break
        selected_indices.append(next_idx)
        dist2 = np.sum((pts - pts[next_idx]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, dist2)

    return [tuple(int(v) for v in candidate_coords[idx]) for idx in selected_indices]


def commit_segment(seg_frames: List[SegmentFrame], covered_mask: np.ndarray, trusted_mask: np.ndarray, track_axis: int,
                   args, force_trusted: bool = False):
    if len(seg_frames) == 0:
        return [], None

    for item in seg_frames:
        update_mask_volume(covered_mask, item.mask, item.axis, (item.global_frame_idx, *item.box[1:]))

    if force_trusted:
        for item in seg_frames:
            update_mask_volume(trusted_mask, item.mask, item.axis, (item.global_frame_idx, *item.box[1:]))
        return [], SegmentDecision(is_trusted=True, dominant_axis=track_axis, axis_ratio=np.inf, spans=(0, 0, 0))

    decision = judge_segment_direction(
        seg_frames=seg_frames,
        track_axis=track_axis,
        axis_ratio_thr=args.axis_ratio_thr,
    )

    if decision.is_trusted:
        for item in seg_frames:
            update_mask_volume(trusted_mask, item.mask, item.axis, (item.global_frame_idx, *item.box[1:]))
        return [], decision

    new_seeds = sample_respawn_seeds_from_segment(
        seg_frames=seg_frames,
        trusted_mask=trusted_mask,
        num_samples=args.segment_respawn_num_seeds,
        min_distance=args.segment_respawn_min_distance,
        max_candidates=args.max_segment_respawn_candidates,
    )
    return new_seeds, decision


def vos_track_one_direction_with_segment_trust(
    video_predictor,
    vol_man,
    axis,
    box,
    idx_list,
    init_mask_2d,
    vos_tmp_root,
    offload_video_to_cpu,
    covered_mask,
    trusted_mask,
    start_untrusted_count,
    args,
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
        return [], {
            "trusted_segments": 0,
            "untrusted_segments": 0,
            "final_untrusted_count": int(start_untrusted_count),
        }

    spawned_seeds: List[SeedTask] = []
    spawned_set: Set[Tuple[int, int, int]] = set()
    trusted_segments = 0
    untrusted_segments = 0
    seg_cache: List[SegmentFrame] = []
    current_untrusted_count = int(start_untrusted_count)

    tmp_dir = tempfile.mkdtemp(prefix="sam2_vos_segtrust_", dir=vos_tmp_root)
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
                seg_cache.append(
                    SegmentFrame(axis=axis, global_frame_idx=int(valid_idxs[0]), box=box, mask=m0.astype(np.uint8))
                )

            for f_idx, _obj_ids, masks in video_predictor.propagate_in_video(state):
                if f_idx < 0 or f_idx >= len(valid_idxs):
                    continue

                mm = masks[0, 0]
                if torch.is_tensor(mm):
                    mm = (mm > 0).to(torch.uint8).cpu().numpy()
                else:
                    mm = (mm > 0).astype(np.uint8)

                if mm.sum() == 0:
                    if len(seg_cache) >= args.min_segment_frames:
                        force_trusted = current_untrusted_count >= args.max_untrusted_segments_per_lineage
                        new_seeds, decision = commit_segment(
                            seg_frames=seg_cache,
                            covered_mask=covered_mask,
                            trusted_mask=trusted_mask,
                            track_axis=axis,
                            args=args,
                            force_trusted=force_trusted,
                        )
                        if decision is not None:
                            if decision.is_trusted:
                                trusted_segments += 1
                            else:
                                untrusted_segments += 1
                        if (decision is not None) and (not decision.is_trusted):
                            current_untrusted_count += 1
                        for seed in new_seeds:
                            task = SeedTask(
                                seed=seed,
                                is_original=False,
                                untrusted_count=current_untrusted_count,
                            )
                            if task.seed in spawned_set:
                                continue
                            spawned_set.add(task.seed)
                            spawned_seeds.append(task)
                    seg_cache = []
                    continue

                seg_cache.append(
                    SegmentFrame(
                        axis=axis,
                        global_frame_idx=int(valid_idxs[f_idx]),
                        box=box,
                        mask=mm.astype(np.uint8),
                    )
                )

                if len(seg_cache) >= args.segment_len:
                    force_trusted = current_untrusted_count >= args.max_untrusted_segments_per_lineage
                    new_seeds, decision = commit_segment(
                        seg_frames=seg_cache,
                        covered_mask=covered_mask,
                        trusted_mask=trusted_mask,
                        track_axis=axis,
                        args=args,
                        force_trusted=force_trusted,
                    )
                    if decision is not None:
                        if decision.is_trusted:
                            trusted_segments += 1
                        else:
                            untrusted_segments += 1
                    if (decision is not None) and (not decision.is_trusted):
                        current_untrusted_count += 1
                    for seed in new_seeds:
                        task = SeedTask(
                            seed=seed,
                            is_original=False,
                            untrusted_count=current_untrusted_count,
                        )
                        if task.seed in spawned_set:
                            continue
                        spawned_set.add(task.seed)
                        spawned_seeds.append(task)
                    seg_cache = []

            if len(seg_cache) >= args.min_segment_frames:
                force_trusted = current_untrusted_count >= args.max_untrusted_segments_per_lineage
                new_seeds, decision = commit_segment(
                    seg_frames=seg_cache,
                    covered_mask=covered_mask,
                    trusted_mask=trusted_mask,
                    track_axis=axis,
                    args=args,
                    force_trusted=force_trusted,
                )
                if decision is not None:
                    if decision.is_trusted:
                        trusted_segments += 1
                    else:
                        untrusted_segments += 1
                if (decision is not None) and (not decision.is_trusted):
                    current_untrusted_count += 1
                for seed in new_seeds:
                    task = SeedTask(
                        seed=seed,
                        is_original=False,
                        untrusted_count=current_untrusted_count,
                    )
                    if task.seed in spawned_set:
                        continue
                    spawned_set.add(task.seed)
                    spawned_seeds.append(task)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return spawned_seeds, {
        "trusted_segments": trusted_segments,
        "untrusted_segments": untrusted_segments,
        "final_untrusted_count": current_untrusted_count,
    }


def save_mask(path, mask, affine, need_transpose):
    if path.endswith(".h5"):
        path = path.replace(".h5", ".nii.gz")
    elif not path.endswith(".nii.gz"):
        path += ".nii.gz"

    img = mask.astype(np.uint8)
    if need_transpose:
        img = np.transpose(img, (2, 1, 0))
    nii_img = nib.Nifti1Image(img, affine)
    nib.save(nii_img, path)


def run_segmentation_with_segment_trust():
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

    covered_mask = np.zeros_like(vol_man.vol, dtype=np.uint8)
    trusted_mask = np.zeros_like(vol_man.vol, dtype=np.uint8)

    initial_tasks = [SeedTask(seed=s, is_original=True, untrusted_count=0) for s in initial_seeds]
    pending: Deque[SeedTask] = deque(initial_tasks)
    seen: Set[Tuple[int, int, int]] = set(initial_seeds)

    print(f"Starting SAM2 segment-trust segmentation over {len(initial_seeds)} initial seeds...")


    run_tag = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    vos_tmp_root = os.path.join(args.output_dir, f"_tmp_sam2_vos_{run_tag}")
    # vos_tmp_root = os.path.join(args.output_dir, "_tmp_sam2_vos_segment_trust")
    os.makedirs(vos_tmp_root, exist_ok=True)

    segmented_count = 0
    skipped_trusted = 0
    skipped_invalid = 0
    respawned_count = 0
    trusted_segment_count = 0
    untrusted_segment_count = 0

    pbar = tqdm(total=len(pending), desc="Tracking (SAM2 segment-trust)")
    while len(pending) > 0:
        task = pending.popleft()
        pbar.update(1)

        if segmented_count >= args.max_segmented_seeds:
            print(f"[STOP] segmented seeds reached max_segmented_seeds={args.max_segmented_seeds}")
            break

        seed = task.seed
        z, y, x = seed
        if task.is_original:
            if covered_mask[z, y, x] > 0:
                skipped_trusted += 1
                continue
        else:
            if trusted_mask[z, y, x] > 0:
                skipped_trusted += 1
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

        if best_axis == -1 or best_mask is None or int(best_mask.sum()) == 0:
            skipped_invalid += 1
            continue
        if int(best_mask.sum()) > args.max_init_mask_area:
            skipped_invalid += 1
            continue

        segmented_count += 1

        track_dim = [0, 1, 2][best_axis]
        start_idx = seed[track_dim]
        max_dist = args.max_track_distance
        forward_idxs = list(range(start_idx, min(start_idx + max_dist, vol_man.shape[track_dim])))
        backward_idxs = list(range(start_idx, max(start_idx - max_dist, -1), -1))

        _, box = crops[best_axis]
        new_fw, stat_fw = vos_track_one_direction_with_segment_trust(
            video_predictor=video_predictor,
            vol_man=vol_man,
            axis=best_axis,
            box=box,
            idx_list=forward_idxs,
            init_mask_2d=best_mask,
            vos_tmp_root=vos_tmp_root,
            offload_video_to_cpu=args.vos_offload_video_to_cpu,
            covered_mask=covered_mask,
            trusted_mask=trusted_mask,
            start_untrusted_count=task.untrusted_count,
            args=args,
        )
        new_bw, stat_bw = vos_track_one_direction_with_segment_trust(
            video_predictor=video_predictor,
            vol_man=vol_man,
            axis=best_axis,
            box=box,
            idx_list=backward_idxs,
            init_mask_2d=best_mask,
            vos_tmp_root=vos_tmp_root,
            offload_video_to_cpu=args.vos_offload_video_to_cpu,
            covered_mask=covered_mask,
            trusted_mask=trusted_mask,
            start_untrusted_count=int(stat_fw["final_untrusted_count"]),
            args=args,
        )

        trusted_segment_count += int(stat_fw["trusted_segments"]) + int(stat_bw["trusted_segments"])
        untrusted_segment_count += int(stat_fw["untrusted_segments"]) + int(stat_bw["untrusted_segments"])

        for ns in (new_fw + new_bw):
            if ns.seed in seen:
                continue
            if trusted_mask[ns.seed] > 0:
                continue
            seen.add(ns.seed)
            pending.append(ns)
            respawned_count += 1
            pbar.total += 1
            pbar.refresh()

    pbar.close()

    if not args.keep_tmp_vos_frames and os.path.isdir(vos_tmp_root):
        shutil.rmtree(vos_tmp_root, ignore_errors=True)

    affine = vol_man.affine if vol_man.affine is not None else np.eye(4)
    need_transpose = (args.need_transpose != "False")

    trusted_path = os.path.join(args.output_dir, args.output_filename)
    covered_path = os.path.join(args.output_dir, "covered_" + args.output_filename)
    save_mask(trusted_path, trusted_mask, affine, need_transpose)
    save_mask(covered_path, covered_mask, affine, need_transpose)

    print("=" * 80)
    print(f"Segmented seeds actually run: {segmented_count}")
    print(f"Skipped trusted-covered seeds: {skipped_trusted}")
    print(f"Skipped invalid seeds: {skipped_invalid}")
    print(f"Respawned seeds queued: {respawned_count}")
    print(f"Trusted segments: {trusted_segment_count}")
    print(f"Untrusted segments: {untrusted_segment_count}")
    print(f"Saved trusted mask to {trusted_path}")
    print(f"Saved covered mask to {covered_path}")


if __name__ == "__main__":
    run_segmentation_with_segment_trust()
