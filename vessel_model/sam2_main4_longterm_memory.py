# 基于 sam2_main4 的 long-term memory 实验版本
# 目标：
# 1) 仍使用单次 init_state + 连续 propagate（不重复加载视频目录）
# 2) 将高质量 working memory 段提升为 long-term memory 段
# 3) 推理时同时保留 long-term 与近期 working（通过 SAM2 state 中 cond/non-cond 输出联合生效）
import argparse
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import h5py
import hydra
import numpy as np
import torch
from tqdm import tqdm

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .data_manager import VolumeManager
from .global_memory_pool import GlobalMemoryPool
from .preprocessing import get_multi_axis_init_seg, get_seeds_from_init_seg, get_seg
from .sam2_baseline.image_utils import get_slice, to_uint8_rgb, write_jpeg_frames
from .sam2_baseline.predict_utils import map_local_point

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
EPS = 1e-6


@dataclass
class FrameQuality:
    frame_idx: int
    physical_frame_idx: int
    mask: np.ndarray
    quality: float


@dataclass
class LongTermSegment:
    frame_items: List[FrameQuality]
    seg_quality: float


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2_checkpoint", required=True)
    parser.add_argument("--sam2_model_cfg", required=True)
    parser.add_argument("--volume_path", required=True)
    parser.add_argument("--output_dir", default="./bv_seg_output_longterm_memory")
    parser.add_argument("--output_filename", default="segmentation.nii.gz")
    parser.add_argument("--dataset_key", default="main")
    parser.add_argument("--seed_file", default=None)
    parser.add_argument("--init_seg_path", default=None)
    parser.add_argument("--axis", type=int, default=3)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--gaussian_kernel", type=int, default=5)
    parser.add_argument("--min_bright", type=int, default=40)
    parser.add_argument("--remove_portion", type=float, default=0.98)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--need_transpose", default="False")
    parser.add_argument("--max_track_distance", type=int, default=2000)
    parser.add_argument("--max_init_mask_area", type=int, default=12000)

    # long-term / working memory 参数
    parser.add_argument("--segment_len", type=int, default=12,
                        help="working memory按该长度切段评估质量")
    parser.add_argument("--longterm_quality_thr", type=float, default=0.8,
                        help="段质量>=该阈值时提升为long-term")
    parser.add_argument("--max_longterm_segments", type=int, default=5,
                        help="long-term最大段数")
    parser.add_argument("--working_window", type=int, default=24,
                        help="保留最近working输出帧数")
    parser.add_argument("--max_global_inject_per_seed", type=int, default=6,
                        help="每个seed启动时最多注入的全局记忆帧数")

    parser.add_argument("--vos_offload_video_to_cpu", action="store_true")
    parser.add_argument("--keep_tmp_vos_frames", action="store_true")
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


def mask_iou(a, b):
    aa = a.astype(bool)
    bb = b.astype(bool)
    inter = np.logical_and(aa, bb).sum()
    uni = np.logical_or(aa, bb).sum()
    return float(inter) / float(uni) if uni > 0 else 0.0


def frame_quality_from_logits(mask_logits, mask_bin, prev_mask):
    if torch.is_tensor(mask_logits):
        probs = torch.sigmoid(mask_logits).detach().float().cpu().numpy()
    else:
        probs = 1.0 / (1.0 + np.exp(-mask_logits))

    if mask_bin.sum() > 0:
        conf = float(probs[mask_bin > 0].mean())
    else:
        conf = 0.0

    if prev_mask is None or prev_mask.sum() == 0 or mask_bin.sum() == 0:
        stab = 0.0
    else:
        stab = mask_iou(prev_mask, mask_bin)
    return 0.7 * conf + 0.3 * stab


def promote_segment_to_longterm(
    video_predictor,
    state,
    seg_frames: List[FrameQuality],
    longterm_bank: List[LongTermSegment],
    global_memory_pool: GlobalMemoryPool,
    axis: int,
    seed: Tuple[int, int, int],
    direction: str,
    args,
):
    if len(seg_frames) == 0:
        return

    seg_q = float(np.mean([x.quality for x in seg_frames]))
    if seg_q < args.longterm_quality_thr:
        return

    # 将整个高质量段提升为 SAM2 cond memory（真正参与 memory attention）
    # 同一 frame 若重复出现，仅保留该段内质量最高的结果。
    best_per_frame: Dict[int, FrameQuality] = {}
    for item in seg_frames:
        prev = best_per_frame.get(item.frame_idx, None)
        if (prev is None) or (item.quality > prev.quality):
            best_per_frame[item.frame_idx] = item

    ordered_items = sorted(best_per_frame.values(), key=lambda x: x.frame_idx)
    if len(ordered_items) == 0:
        return

    for item in ordered_items:
        video_predictor.promote_frame_output_to_cond(
            state, frame_idx=item.frame_idx, obj_id=1
        )
        global_memory_pool.add_entry(
            axis=axis,
            physical_frame_idx=item.physical_frame_idx,
            mask=item.mask,
            quality=item.quality,
            source_seed=seed,
            source_direction=direction,
        )
    longterm_bank.append(LongTermSegment(frame_items=ordered_items, seg_quality=seg_q))

    # 若超容量，降级质量最低 long-term 段
    while len(longterm_bank) > args.max_longterm_segments:
        worst_idx = int(np.argmin([x.seg_quality for x in longterm_bank]))
        worst = longterm_bank.pop(worst_idx)
        for item in worst.frame_items:
            video_predictor.demote_frame_output_from_cond(
                state, frame_idx=item.frame_idx, obj_id=1
            )


def build_frames_rgb(vol_man, axis, box, idx_list):
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
    return frames_rgb, valid_idxs


def longterm_track_one_direction(
    video_predictor,
    vol_man,
    axis,
    box,
    idx_list,
    init_mask_2d,
    global_update_axis,
    vos_tmp_root,
    offload_video_to_cpu,
    global_memory_pool: GlobalMemoryPool,
    seed: Tuple[int, int, int],
    direction: str,
    args,
):
    if len(idx_list) == 0:
        return

    frames_rgb, valid_gidx = build_frames_rgb(vol_man, axis, box, idx_list)
    if len(frames_rgb) == 0:
        return

    tmp_dir = tempfile.mkdtemp(prefix="sam2_vos_longterm_", dir=vos_tmp_root)
    write_jpeg_frames(frames_rgb, tmp_dir)

    try:
        with torch.inference_mode(), torch.autocast(str(vol_man.device), dtype=torch.bfloat16):
            state = video_predictor.init_state(
                video_path=tmp_dir,
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=False,
                async_loading_frames=False,
            )

            # 初始提示
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
            vol_man.update_global_mask(m0, global_update_axis, (valid_gidx[0], *box[1:]))

            init_radius = float(np.sqrt(float(np.count_nonzero(m0)) / np.pi)) if int(m0.sum()) > 0 else 0.0
            phys_to_local = {int(p): i for i, p in enumerate(valid_gidx)}
            inject_items = global_memory_pool.select_for_injection(
                axis=axis,
                current_seed=seed,
                current_radius=init_radius,
                physical_to_local=phys_to_local,
                max_inject=args.max_global_inject_per_seed,
            )
            for local_idx, mem_entry in inject_items:
                video_predictor.add_new_mask(
                    state,
                    frame_idx=local_idx,
                    obj_id=1,
                    mask=mem_entry.mask.astype(bool),
                )

            # long-term/working 管理缓存（都在同一个state内进行，不重载video）
            longterm_bank: List[LongTermSegment] = []
            seg_cache: List[FrameQuality] = []
            prev_mask = m0

            for f_idx, _obj_ids, masks in video_predictor.propagate_in_video(state):
                if f_idx < 0 or f_idx >= len(valid_gidx):
                    continue

                mm_logits = masks[0, 0]
                if torch.is_tensor(mm_logits):
                    mm = (mm_logits > 0).to(torch.uint8).cpu().numpy()
                else:
                    mm = (mm_logits > 0).astype(np.uint8)

                if mm.sum() == 0:
                    continue

                vol_man.update_global_mask(mm, global_update_axis, (valid_gidx[f_idx], *box[1:]))

                q = frame_quality_from_logits(mm_logits, mm, prev_mask)
                seg_cache.append(
                    FrameQuality(
                        frame_idx=f_idx,
                        physical_frame_idx=int(valid_gidx[f_idx]),
                        mask=mm,
                        quality=float(q),
                    )
                )
                prev_mask = mm

                # 裁剪 working memory（non-cond）
                min_keep = max(0, f_idx - args.working_window)
                video_predictor.prune_non_cond_memory(
                    state,
                    min_keep_frame_idx=min_keep,
                    obj_id=1,
                    keep_cond=True,
                )

                # 到段尾就判断是否提升为long-term
                if len(seg_cache) >= args.segment_len:
                    promote_segment_to_longterm(
                        video_predictor=video_predictor,
                        state=state,
                        seg_frames=seg_cache,
                        longterm_bank=longterm_bank,
                        global_memory_pool=global_memory_pool,
                        axis=axis,
                        seed=seed,
                        direction=direction,
                        args=args,
                    )
                    seg_cache = []

            # 尾段补一次
            if len(seg_cache) > 0:
                promote_segment_to_longterm(
                    video_predictor=video_predictor,
                    state=state,
                    seg_frames=seg_cache,
                    longterm_bank=longterm_bank,
                    global_memory_pool=global_memory_pool,
                    axis=axis,
                    seed=seed,
                    direction=direction,
                    args=args,
                )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_segmentation():
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

    print(f"Starting SAM2 segmentation with long-term memory over {len(seeds)} seeds...")

    run_tag = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    vos_tmp_root = os.path.join(args.output_dir, f"_tmp_sam2_vos_lt_{run_tag}")
    os.makedirs(vos_tmp_root, exist_ok=True)
    global_memory_pool = GlobalMemoryPool()

    with tqdm(total=len(seeds), desc="Tracking (SAM2 long-term-memory)") as pbar:
        for seed in seeds:
            pbar.update(1)
            z, y, x = seed
            if vol_man.global_mask[z, y, x] > 0:
                continue

            crops = vol_man.get_triplane_crops(seed)
            best_axis, best_mask, min_area = -1, None, float("inf")

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
                continue

            track_dim = [0, 1, 2][best_axis]
            start_idx = seed[track_dim]
            max_dist = args.max_track_distance
            forward_idxs = list(range(start_idx, min(start_idx + max_dist, vol_man.shape[track_dim])))
            backward_idxs = list(range(start_idx, max(start_idx - max_dist, -1), -1))

            _, box = crops[best_axis]
            longterm_track_one_direction(
                video_predictor=video_predictor,
                vol_man=vol_man,
                axis=best_axis,
                box=box,
                idx_list=forward_idxs,
                init_mask_2d=best_mask,
                global_update_axis=best_axis,
                vos_tmp_root=vos_tmp_root,
                offload_video_to_cpu=args.vos_offload_video_to_cpu,
                global_memory_pool=global_memory_pool,
                seed=seed,
                direction="forward",
                args=args,
            )
            longterm_track_one_direction(
                video_predictor=video_predictor,
                vol_man=vol_man,
                axis=best_axis,
                box=box,
                idx_list=backward_idxs,
                init_mask_2d=best_mask,
                global_update_axis=best_axis,
                vos_tmp_root=vos_tmp_root,
                offload_video_to_cpu=args.vos_offload_video_to_cpu,
                global_memory_pool=global_memory_pool,
                seed=seed,
                direction="backward",
                args=args,
            )

    if (not args.keep_tmp_vos_frames) and os.path.isdir(vos_tmp_root):
        shutil.rmtree(vos_tmp_root, ignore_errors=True)

    final_path = os.path.join(args.output_dir, args.output_filename)
    flag = (args.need_transpose != "False")
    vol_man.save(final_path, flag)
    print(f"Done! Saved to {final_path}")


if __name__ == "__main__":
    run_segmentation()
