import numpy as np
from scipy import ndimage

from .predict_utils import predict_from_point


def mask_centroid(mask: np.ndarray):
    if mask is None or mask.sum() == 0:
        return None
    cy, cx = ndimage.center_of_mass(mask.astype(bool))
    if np.isnan(cy) or np.isnan(cx):
        return None
    return float(cx), float(cy)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    if mask_a is None or mask_b is None:
        return 0.0
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    inter = np.logical_and(a, b).sum()
    return float(inter) / float(union)


def gap_iou_from_resegment(img_predictor, rgb_frame: np.ndarray, video_mask: np.ndarray):
    center = mask_centroid(video_mask)
    if center is None:
        return None
    point = [center[0], center[1]]
    img_mask = predict_from_point(img_predictor, rgb_frame, point, thr=0.0)
    if img_mask is None:
        return None
    return mask_iou(video_mask, img_mask)
