from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from skimage.morphology import skeletonize

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


def _count_skeleton_neighbors(skeleton: np.ndarray, coord_local: np.ndarray) -> int:
    z, y, x = [int(v) for v in coord_local.tolist()]
    z0 = max(0, z - 1)
    y0 = max(0, y - 1)
    x0 = max(0, x - 1)
    z1 = min(skeleton.shape[0], z + 2)
    y1 = min(skeleton.shape[1], y + 2)
    x1 = min(skeleton.shape[2], x + 2)
    return int(np.count_nonzero(skeleton[z0:z1, y0:y1, x0:x1])) - 1


def _estimate_endpoint_outward_direction(endpoint_local: np.ndarray, skeleton_coords_local: np.ndarray) -> Optional[np.ndarray]:
    # Direction is estimated from nearby skeleton points on the inside of the branch.
    # We then flip that inward tangent so the proposed seed extends outwards from the endpoint.
    deltas = skeleton_coords_local.astype(np.float32) - endpoint_local.astype(np.float32)
    dist2 = np.sum(deltas * deltas, axis=1)
    valid = dist2 > 0
    if not np.any(valid):
        return None

    valid_deltas = deltas[valid]
    valid_dist2 = dist2[valid]
    order = np.argsort(valid_dist2)
    inward_vec = valid_deltas[order[: min(6, len(order))]].mean(axis=0)
    norm = float(np.linalg.norm(inward_vec))
    if norm <= EPS:
        inward_vec = valid_deltas[order[0]]
        norm = float(np.linalg.norm(inward_vec))
        if norm <= EPS:
            return None
    return (-inward_vec / norm).astype(np.float32)


def propose_skeleton_respawn_seeds(
    seg_frames: List[SegmentFrame],
    trusted_mask: np.ndarray,
    skeleton_respawn_offset: float,
    max_skeleton_respawn_seeds: int,
    min_seed_distance: float,
) -> List[Tuple[int, int, int]]:
    if len(seg_frames) == 0 or max_skeleton_respawn_seeds <= 0:
        return []

    coords_acc = []
    for item in seg_frames:
        append_segment_voxels(coords_acc, item)
    if len(coords_acc) == 0:
        return []

    segment_coords = np.concatenate(coords_acc, axis=0)
    segment_coords = np.unique(segment_coords, axis=0)

    # Rebuild the segment as a local 3D binary mask so skeletonization only touches a tight bbox.
    min_xyz = segment_coords.min(axis=0)
    max_xyz = segment_coords.max(axis=0)
    local_shape = (max_xyz - min_xyz + 1).astype(np.int32)
    local_mask = np.zeros(tuple(int(v) for v in local_shape.tolist()), dtype=bool)

    # Coordinate mapping:
    #   global voxel = local voxel + min_xyz
    #   local voxel = global voxel - min_xyz
    segment_coords_local = (segment_coords - min_xyz).astype(np.int32)
    local_mask[
        segment_coords_local[:, 0],
        segment_coords_local[:, 1],
        segment_coords_local[:, 2],
    ] = True

    local_skeleton = skeletonize(local_mask)
    skeleton_coords_local = np.argwhere(local_skeleton > 0).astype(np.int32)
    if len(skeleton_coords_local) == 0:
        return []

    endpoint_coords_local: List[np.ndarray] = []
    for coord_local in skeleton_coords_local:
        if _count_skeleton_neighbors(local_skeleton, coord_local) <= 1:
            endpoint_coords_local.append(coord_local)
    if len(endpoint_coords_local) == 0:
        return []

    centroid_local = segment_coords_local.astype(np.float32).mean(axis=0, keepdims=True)
    endpoint_coords_local = sorted(
        endpoint_coords_local,
        key=lambda c: float(np.sum((c.astype(np.float32) - centroid_local[0]) ** 2)),
        reverse=True,
    )

    accepted: List[Tuple[int, int, int]] = []
    accepted_pts: List[np.ndarray] = []
    segment_coords_f = segment_coords.astype(np.float32)
    min_body_distance = max(1.0, float(skeleton_respawn_offset) * 0.5)
    min_seed_distance2 = float(min_seed_distance) * float(min_seed_distance)
    min_body_distance2 = float(min_body_distance) * float(min_body_distance)

    for endpoint_local in endpoint_coords_local:
        outward_dir = _estimate_endpoint_outward_direction(endpoint_local, skeleton_coords_local)
        if outward_dir is None:
            continue

        endpoint_global = endpoint_local.astype(np.float32) + min_xyz.astype(np.float32)
        candidate = np.rint(endpoint_global + outward_dir * float(skeleton_respawn_offset)).astype(np.int32)
        z, y, x = [int(v) for v in candidate.tolist()]

        # Filtering rules:
        # 1) stay inside the volume bounds
        # 2) never land inside trusted_mask
        if (
            z < 0
            or y < 0
            or x < 0
            or z >= trusted_mask.shape[0]
            or y >= trusted_mask.shape[1]
            or x >= trusted_mask.shape[2]
        ):
            continue
        if trusted_mask[z, y, x] > 0:
            continue

        # 3) keep the new seed away from the existing segment body so we extend outward instead of
        #    respawning back inside the same local component.
        candidate_f = candidate.astype(np.float32)
        if np.min(np.sum((segment_coords_f - candidate_f[None, :]) ** 2, axis=1)) < min_body_distance2:
            continue

        # 4) keep proposed seeds sufficiently separated from each other.
        if len(accepted_pts) > 0:
            d2 = [float(np.sum((pt - candidate_f) ** 2)) for pt in accepted_pts]
            if min(d2) < min_seed_distance2:
                continue

        accepted.append((z, y, x))
        accepted_pts.append(candidate_f)
        if len(accepted) >= int(max_skeleton_respawn_seeds):
            break

    return accepted


def commit_segment(
    seg_frames: List[SegmentFrame],
    covered_mask: np.ndarray,
    trusted_mask: np.ndarray,
    track_axis: int,
    args,
    force_trusted: bool = False,
    respawn_seed_override: Optional[List[Tuple[int, int, int]]] = None,
    classification_category: Optional[str] = None,
):
    if len(seg_frames) == 0:
        return [], None

    for item in seg_frames:
        update_mask_volume(covered_mask, item.mask, item.axis, (item.global_frame_idx, *item.box[1:]))

    if force_trusted:
        for item in seg_frames:
            update_mask_volume(trusted_mask, item.mask, item.axis, (item.global_frame_idx, *item.box[1:]))
        return [], SegmentDecision(is_trusted=True, dominant_axis=track_axis, axis_ratio=np.inf, spans=(0, 0, 0))

    if getattr(args, "disable_legacy_segment_trust_respawn", False):
        # In classifier-only mode:
        # - stable segments are treated as trusted
        # - complex / failure segments are treated as non-trusted
        # - legacy FPS respawn is disabled; only an explicit override (e.g. skeleton respawn) may spawn seeds
        is_trusted = classification_category == "stable"
        decision = SegmentDecision(
            is_trusted=bool(is_trusted),
            dominant_axis=track_axis if is_trusted else -1,
            axis_ratio=np.inf if is_trusted else 0.0,
            spans=(0, 0, 0),
        )
        if is_trusted:
            for item in seg_frames:
                update_mask_volume(trusted_mask, item.mask, item.axis, (item.global_frame_idx, *item.box[1:]))
            return [], decision
        if respawn_seed_override is not None:
            return list(respawn_seed_override), decision
        return [], decision

    decision = judge_segment_direction(seg_frames=seg_frames, track_axis=track_axis, axis_ratio_thr=args.axis_ratio_thr)
    if decision.is_trusted:
        for item in seg_frames:
            update_mask_volume(trusted_mask, item.mask, item.axis, (item.global_frame_idx, *item.box[1:]))
        return [], decision

    if respawn_seed_override is not None:
        return list(respawn_seed_override), decision

    new_seeds = sample_respawn_seeds_from_segment(
        seg_frames=seg_frames,
        trusted_mask=trusted_mask,
        num_samples=args.segment_respawn_num_seeds,
        min_distance=args.segment_respawn_min_distance,
        max_candidates=args.max_segment_respawn_candidates,
    )
    return new_seeds, decision
