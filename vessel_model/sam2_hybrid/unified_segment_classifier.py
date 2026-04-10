from dataclasses import dataclass
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np

EPS = 1e-6

STABLE_LABEL = 1
COMPLEX_LABEL = 2
FAILURE_LABEL = 3


@dataclass
class ClassifiedSegmentFrame:
    axis: int
    global_frame_idx: int
    box: Tuple[int, int, int, int, int]
    mask: np.ndarray
    quality: Optional[float] = None


@dataclass
class SegmentMetrics:
    avg_quality: float
    avg_adj_iou: float
    empty_rate: float
    nonempty_frames: int
    total_frames: int
    dominant_axis: int
    axis_ratio: float
    spans: Tuple[int, int, int]


@dataclass
class SegmentClassResult:
    category: str
    label: int
    metrics: SegmentMetrics


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(bool)
    bb = b.astype(bool)
    inter = np.logical_and(aa, bb).sum()
    uni = np.logical_or(aa, bb).sum()
    return float(inter) / float(uni) if uni > 0 else 0.0


def _append_segment_voxels(coords_acc, frame: ClassifiedSegmentFrame) -> None:
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


def compute_segment_metrics(
    seg_frames: List[ClassifiedSegmentFrame],
    track_axis: int,
    empty_frame_count: int,
) -> SegmentMetrics:
    nonempty_frames = len(seg_frames)
    total_frames = int(nonempty_frames + max(0, int(empty_frame_count)))

    quality_vals = [float(item.quality) for item in seg_frames if item.quality is not None]
    avg_quality = float(np.mean(quality_vals)) if len(quality_vals) > 0 else 0.0

    adj_ious = []
    for prev, curr in zip(seg_frames[:-1], seg_frames[1:]):
        adj_ious.append(mask_iou(prev.mask, curr.mask))
    avg_adj_iou = float(np.mean(adj_ious)) if len(adj_ious) > 0 else 1.0

    empty_rate = float(empty_frame_count) / max(float(total_frames), 1.0)

    coords_acc = []
    for item in seg_frames:
        _append_segment_voxels(coords_acc, item)

    if len(coords_acc) == 0:
        spans = (0, 0, 0)
        dominant_axis = -1
        axis_ratio = 0.0
    else:
        coords = np.concatenate(coords_acc, axis=0)
        min_xyz = coords.min(axis=0)
        max_xyz = coords.max(axis=0)
        spans_arr = (max_xyz - min_xyz + 1).astype(np.int32)
        spans = (int(spans_arr[0]), int(spans_arr[1]), int(spans_arr[2]))
        dominant_axis = int(np.argmax(spans_arr))
        other_axes = [a for a in (0, 1, 2) if a != track_axis]
        second_extent = max(float(spans_arr[other_axes[0]]), float(spans_arr[other_axes[1]]))
        axis_ratio = float(spans_arr[track_axis]) / max(second_extent, EPS)

    return SegmentMetrics(
        avg_quality=avg_quality,
        avg_adj_iou=avg_adj_iou,
        empty_rate=empty_rate,
        nonempty_frames=int(nonempty_frames),
        total_frames=int(total_frames),
        dominant_axis=int(dominant_axis),
        axis_ratio=float(axis_ratio),
        spans=spans,
    )


def classify_segment(
    seg_frames: List[ClassifiedSegmentFrame],
    track_axis: int,
    empty_frame_count: int,
    args,
) -> SegmentClassResult:
    metrics = compute_segment_metrics(
        seg_frames=seg_frames,
        track_axis=track_axis,
        empty_frame_count=empty_frame_count,
    )

    is_failure = (
        metrics.nonempty_frames <= 0
        or metrics.total_frames < int(args.segment_classifier_failure_min_frames)
        or metrics.avg_quality < float(args.segment_classifier_failure_quality_thr)
        or metrics.avg_adj_iou < float(args.segment_classifier_failure_iou_thr)
        or metrics.empty_rate > float(args.segment_classifier_failure_empty_rate_thr)
    )
    if is_failure:
        return SegmentClassResult(category="failure", label=FAILURE_LABEL, metrics=metrics)

    is_stable = (
        metrics.avg_quality >= float(args.segment_classifier_stable_quality_thr)
        and metrics.avg_adj_iou >= float(args.segment_classifier_stable_iou_thr)
        and metrics.empty_rate <= float(args.segment_classifier_stable_empty_rate_thr)
        and metrics.dominant_axis == int(track_axis)
        and metrics.axis_ratio >= float(args.segment_classifier_stable_axis_ratio_thr)
    )
    if is_stable:
        return SegmentClassResult(category="stable", label=STABLE_LABEL, metrics=metrics)

    return SegmentClassResult(category="complex", label=COMPLEX_LABEL, metrics=metrics)


def paint_segment_label_volume(
    label_volume: np.ndarray,
    seg_frames: List[ClassifiedSegmentFrame],
    label: int,
) -> None:
    label = int(label)
    for item in seg_frames:
        idx = int(item.global_frame_idx)
        _, d1_min, d1_max, d2_min, d2_max = item.box
        local_mask = item.mask.astype(bool)
        local_mask = local_mask[: (d1_max - d1_min), : (d2_max - d2_min)]
        if not np.any(local_mask):
            continue

        if item.axis == 0:
            target = label_volume[idx, d1_min:d1_max, d2_min:d2_max]
            target[local_mask] = label
        elif item.axis == 1:
            target = label_volume[d1_min:d1_max, idx, d2_min:d2_max]
            target[local_mask] = label
        else:
            target = label_volume[d1_min:d1_max, d2_min:d2_max, idx]
            target[local_mask] = label


def save_segment_label_volume(
    path: str,
    label_volume: np.ndarray,
    affine,
    need_transpose: bool,
) -> str:
    if path.endswith(".h5"):
        path = path.replace(".h5", ".nii.gz")
    elif not path.endswith(".nii.gz"):
        path += ".nii.gz"

    if affine is None:
        affine = np.eye(4)

    out = label_volume.astype(np.uint8)
    if need_transpose:
        out = np.transpose(out, (2, 1, 0))
    nib.save(nib.Nifti1Image(out, affine), path)
    return path
