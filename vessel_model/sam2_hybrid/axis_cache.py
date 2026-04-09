import gc
import os
from contextlib import nullcontext
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from types import MethodType

from sam2.utils.misc import load_video_frames

from ..sam2_baseline.image_utils import to_uint8_rgb, write_jpeg_frames


@dataclass
class AxisSequenceCache:
    axis: int
    frame_dir: str
    images: Optional[object]
    video_height: int
    video_width: int
    num_frames: int
    feature_cache: Optional[Dict[int, dict]] = None


def build_axis_frames_rgb(volume: np.ndarray, axis: int) -> List[np.ndarray]:
    frames_rgb: List[np.ndarray] = []
    if axis == 0:
        total = volume.shape[0]
        for idx in range(total):
            frames_rgb.append(to_uint8_rgb(volume[idx, :, :]))
    elif axis == 1:
        total = volume.shape[1]
        for idx in range(total):
            frames_rgb.append(to_uint8_rgb(volume[:, idx, :]))
    else:
        total = volume.shape[2]
        for idx in range(total):
            frames_rgb.append(to_uint8_rgb(volume[:, :, idx]))
    return frames_rgb


def prepare_axis_sequence_cache(
    volume: np.ndarray,
    axis: int,
    cache_root: str,
    image_size: int,
    offload_video_to_cpu: bool,
    compute_device,
    video_predictor=None,
    enable_feature_cache: bool = False,
    feature_cache_device: str = "cuda",
) -> AxisSequenceCache:
    axis_dir = os.path.join(cache_root, f"axis_{axis}")
    os.makedirs(axis_dir, exist_ok=True)
    sentinel_path = os.path.join(axis_dir, ".complete")

    if not os.path.exists(sentinel_path):
        frames_rgb = build_axis_frames_rgb(volume, axis)
        write_jpeg_frames(frames_rgb, axis_dir)
        with open(sentinel_path, "w", encoding="utf-8") as f:
            f.write("ok\n")

    images, video_height, video_width = load_video_frames(
        video_path=axis_dir,
        image_size=image_size,
        offload_video_to_cpu=offload_video_to_cpu,
        async_loading_frames=False,
        compute_device=compute_device,
    )
    feature_cache = None
    if enable_feature_cache:
        if video_predictor is None:
            raise ValueError("video_predictor is required when enable_feature_cache=True")
        store_device = torch.device(feature_cache_device)
        feature_cache = {}
        autocast_ctx = (
            torch.autocast(str(compute_device), dtype=torch.bfloat16)
            if str(compute_device).startswith("cuda")
            else nullcontext()
        )
        with torch.inference_mode():
            with autocast_ctx:
                for frame_idx in tqdm(range(len(images)), desc=f"axis {axis} feature cache"):
                    img = images[frame_idx]
                    if not torch.is_tensor(img):
                        raise TypeError("Expected cached frame image to be a torch.Tensor")
                    image = img.to(compute_device, non_blocking=True).unsqueeze(0)
                    backbone_out = video_predictor.forward_image(image)
                    feature_cache[int(frame_idx)] = {
                        "backbone_fpn": [
                            feat.detach().to(store_device, dtype=torch.bfloat16, non_blocking=True)
                            for feat in backbone_out["backbone_fpn"]
                        ],
                        "vision_pos_enc": [
                            pos.detach().to(store_device, dtype=torch.bfloat16, non_blocking=True)
                            for pos in backbone_out["vision_pos_enc"]
                        ],
                    }
    return AxisSequenceCache(
        axis=axis,
        frame_dir=axis_dir,
        images=images,
        video_height=int(video_height),
        video_width=int(video_width),
        num_frames=len(images),
        feature_cache=feature_cache,
    )


def build_all_axis_sequence_caches(
    volume: np.ndarray,
    cache_root: str,
    image_size: int,
    offload_video_to_cpu: bool,
    compute_device,
    video_predictor=None,
    enable_feature_cache: bool = False,
    feature_cache_device: str = "cuda",
) -> Dict[int, AxisSequenceCache]:
    caches: Dict[int, AxisSequenceCache] = {}
    for axis in (0, 1, 2):
        caches[axis] = prepare_axis_sequence_cache(
            volume=volume,
            axis=axis,
            cache_root=cache_root,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            compute_device=compute_device,
            video_predictor=video_predictor,
            enable_feature_cache=enable_feature_cache,
            feature_cache_device=feature_cache_device,
        )
    return caches


def release_axis_sequence_cache(seq_cache: Optional[AxisSequenceCache]) -> None:
    if seq_cache is None:
        return
    seq_cache.images = None
    if seq_cache.feature_cache is not None:
        seq_cache.feature_cache.clear()
        seq_cache.feature_cache = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def enable_precomputed_feature_cache(video_predictor) -> None:
    if getattr(video_predictor, "_precomputed_feature_cache_enabled", False):
        return

    original_get_image_feature = video_predictor._get_image_feature

    def wrapped_get_image_feature(self, inference_state, frame_idx, batch_size):
        precomputed = inference_state.get("precomputed_feature_cache", None)
        if precomputed is not None and frame_idx in precomputed:
            device = inference_state["device"]
            img = inference_state["images"][frame_idx]
            if not torch.is_tensor(img):
                raise TypeError("Expected cached frame image to be a torch.Tensor")
            image = img.to(device, non_blocking=True).unsqueeze(0)
            cached = precomputed[frame_idx]
            backbone_out = {
                "backbone_fpn": [feat.to(device, non_blocking=True) for feat in cached["backbone_fpn"]],
                "vision_pos_enc": [pos.to(device, non_blocking=True) for pos in cached["vision_pos_enc"]],
            }

            expanded_image = image.expand(batch_size, -1, -1, -1)
            expanded_backbone_out = {
                "backbone_fpn": backbone_out["backbone_fpn"].copy(),
                "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
            }
            for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
                expanded_backbone_out["backbone_fpn"][i] = feat.expand(batch_size, -1, -1, -1)
            for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
                expanded_backbone_out["vision_pos_enc"][i] = pos.expand(batch_size, -1, -1, -1)

            features = self._prepare_backbone_features(expanded_backbone_out)
            return (expanded_image,) + features

        return original_get_image_feature(inference_state, frame_idx, batch_size)

    video_predictor._get_image_feature = MethodType(wrapped_get_image_feature, video_predictor)
    video_predictor._precomputed_feature_cache_enabled = True


def init_state_from_axis_cache(
    video_predictor,
    seq_cache: AxisSequenceCache,
    offload_video_to_cpu: bool,
    offload_state_to_cpu: bool = False,
):
    if seq_cache.images is None:
        raise RuntimeError(
            f"Axis sequence cache for axis {seq_cache.axis} has no loaded images. "
            "Please rebuild or reload the cache before use."
        )
    compute_device = video_predictor.device
    inference_state = {}
    inference_state["images"] = seq_cache.images
    inference_state["num_frames"] = seq_cache.num_frames
    inference_state["offload_video_to_cpu"] = offload_video_to_cpu
    inference_state["offload_state_to_cpu"] = offload_state_to_cpu
    inference_state["video_height"] = seq_cache.video_height
    inference_state["video_width"] = seq_cache.video_width
    inference_state["device"] = compute_device
    if offload_state_to_cpu:
        inference_state["storage_device"] = torch.device("cpu")
    else:
        inference_state["storage_device"] = compute_device
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    inference_state["cached_features"] = {}
    inference_state["precomputed_feature_cache"] = seq_cache.feature_cache
    inference_state["constants"] = {}
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    inference_state["output_dict_per_obj"] = {}
    inference_state["temp_output_dict_per_obj"] = {}
    inference_state["frames_tracked_per_obj"] = {}
    if seq_cache.feature_cache is None or 0 not in seq_cache.feature_cache:
        video_predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)
    return inference_state
