from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .common import SeedTask

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
        coords = np.stack(
            [np.full_like(ys, frame.global_frame_idx), frame.box[1] + ys, frame.box[3] + xs],
            axis=1,
        )
    elif frame.axis == 1:
        coords = np.stack(
            [frame.box[1] + ys, np.full_like(ys, frame.global_frame_idx), frame.box[3] + xs],
            axis=1,
        )
    else:
        coords = np.stack(
            [frame.box[1] + ys, frame.box[3] + xs, np.full_like(ys, frame.global_frame_idx)],
            axis=1,
        )
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

    decision = judge_segment_direction(seg_frames=seg_frames, track_axis=track_axis, axis_ratio_thr=args.axis_ratio_thr)
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
