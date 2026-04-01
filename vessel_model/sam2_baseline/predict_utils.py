import numpy as np
from scipy import ndimage

from ..utils import select_masks


def extract_components(binary_mask):
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
        components.append({"mask": comp_mask, "center": (float(cy), float(cx)), "area": area})
    return components


def predict_from_point(predictor, image, point, thr=0.6):
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        point_coords=np.array([point]),
        point_labels=np.array([1]),
        multimask_output=True,
    )
    mask, _ = select_masks(masks, scores, thr=thr, crit="max", max_size=10000, min_circularity=0.0)
    if mask is None:
        return None
    return mask.astype(np.uint8)


def map_local_point(seed, axis, box):
    if axis == 0:
        return [seed[2] - box[3], seed[1] - box[1]]
    if axis == 1:
        return [seed[2] - box[3], seed[0] - box[1]]
    return [seed[1] - box[3], seed[0] - box[1]]


def recover_missing_components_sam2(
    img_predictor,
    vol_man,
    prev_components,
    curr_mask,
    curr_rgb,
    axis,
    vol_idx,
    box,
    overlap_threshold=0.5,
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

    if axis == 0:
        existing = vol_man.global_mask[vol_idx, box[1]:box[2], box[3]:box[4]]
    elif axis == 1:
        existing = vol_man.global_mask[box[1]:box[2], vol_idx, box[3]:box[4]]
    else:
        existing = vol_man.global_mask[box[1]:box[2], box[3]:box[4], vol_idx]

    img_predictor.set_image(curr_rgb)

    recovered = np.zeros_like(curr_mask, dtype=np.uint8)

    for comp in missing:
        cy, cx = comp["center"]
        cy_i = int(round(cy))
        cx_i = int(round(cx))

        masks, scores, _ = img_predictor.predict(
            point_coords=np.array([[cx_i, cy_i]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )

        cand_mask, _ = select_masks(
            masks, scores,
            thr=0.0, crit="max",
            max_size=5000,
            min_circularity=0.0,
        )
        if cand_mask is None or cand_mask.sum() == 0:
            continue

        cand_b = np.asarray(cand_mask).astype(bool)
        exist_b = np.asarray(existing).astype(bool)
        overlap = np.logical_and(cand_b, exist_b).sum()
        ratio = overlap / (cand_mask.sum() + 1e-6)
        if ratio > overlap_threshold:
            continue

        recovered = np.logical_or(recovered, cand_mask)

    if recovered.sum() == 0:
        return None

    return recovered.astype(np.uint8)
