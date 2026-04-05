from dataclasses import dataclass
import os
from typing import List, Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm

from ..preprocessing import get_multi_axis_init_seg, get_seeds_from_init_seg, get_seg


@dataclass
class SeedTask:
    seed: Tuple[int, int, int]
    is_original: bool = True
    untrusted_count: int = 0


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


def load_or_build_seeds(args, vol_man) -> List[Tuple[int, int, int]]:
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


def sample_respawn_seeds_from_mask(
    prev_mask,
    axis,
    box,
    global_frame_idx,
    vol_man,
    num_samples,
    min_point_distance,
    max_candidates,
):
    ys, xs = np.where(prev_mask > 0)
    if len(ys) == 0 or num_samples <= 0:
        return []

    candidate_points = []
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

    seeds = []
    for idx in selected_indices:
        y, x = candidate_points[idx]
        seed = local_point_to_global_seed(axis, box, global_frame_idx, y, x, vol_man.shape)
        if seed is None:
            continue
        if vol_man.global_mask[seed] > 0:
            continue
        seeds.append(seed)
    return seeds
