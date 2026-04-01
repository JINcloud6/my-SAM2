# 在 sam2_main4 流程基础上，实验性改造“记忆机制”：
# - 高质量帧进入 anchor memory
# - 普通帧进入 short-term memory
# - 超出预算时，优先淘汰“低质量且冗余”的帧，而不是最老帧(FIFO)
import argparse
import os
import shutil
import tempfile
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
from .preprocessing import get_multi_axis_init_seg, get_seeds_from_init_seg, get_seg
from .sam2_baseline.image_utils import get_slice, to_uint8_rgb, write_jpeg_frames
from .sam2_baseline.predict_utils import map_local_point

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


@dataclass
class MemoryEntry:
    frame_idx: int
    mask: np.ndarray  # uint8, 0/1
    quality: float
    mem_type: str  # "anchor" | "short"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2_checkpoint", required=True)
    parser.add_argument("--sam2_model_cfg", required=True)
    parser.add_argument("--volume_path", required=True)
    parser.add_argument("--output_dir", default="./bv_seg_output_quality_memory")
    parser.add_argument("--output_filename", default="segmentation.nii.gz")
    parser.add_argument("--dataset_key", default="main")
    parser.add_argument("--seed_file", default=None)
    parser.add_argument("--init_seg_path", default=None, help="Optional cached init-seg h5 path")
    parser.add_argument("--axis", type=int, default=3)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--gaussian_kernel", type=int, default=5)
    parser.add_argument("--min_bright", type=int, default=40)
    parser.add_argument("--remove_portion", type=float, default=0.98)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--need_transpose", default="False")
    parser.add_argument("--max_track_distance", type=int, default=2000)
    parser.add_argument("--max_init_mask_area", type=int, default=12000)

    # 质量驱动memory策略参数
    parser.add_argument("--memory_chunk", type=int, default=24,
                        help="每次重建SAM2 state后，向前推进的帧数")
    parser.add_argument("--anchor_quality_thr", type=float, default=0.72,
                        help=">=该阈值进入anchor memory")
    parser.add_argument("--anchor_memory_size", type=int, default=6)
    parser.add_argument("--short_memory_size", type=int, default=10)
    parser.add_argument("--redundancy_iou_thr", type=float, default=0.85,
                        help="IoU超过该阈值视为冗余")
    parser.add_argument("--redundancy_penalty", type=float, default=0.35,
                        help="冗余惩罚权重，越大越倾向删冗余")

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
        area = int(m.sum())
        if area > 0:
            return m, float(scores[idx])
    return None, None


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(bool)
    bb = b.astype(bool)
    inter = np.logical_and(aa, bb).sum()
    uni = np.logical_or(aa, bb).sum()
    return float(inter) / float(uni) if uni > 0 else 0.0


def build_frames_rgb(vol_man, axis, box, idx_list):
    frames_rgb = []
    valid_gidx = []
    for vidx in idx_list:
        sl = get_slice(vol_man.vol, axis, vidx, box)
        if sl.size == 0:
            break
        rgb = to_uint8_rgb(sl)
        if rgb is None:
            break
        frames_rgb.append(rgb)
        valid_gidx.append(vidx)
    return frames_rgb, valid_gidx


def prediction_quality(mask_logits, mask_bin, prev_mask):
    # 质量 = 概率均值 + 与上一帧稳定性(IoU)
    # 仅用于相对排序，不依赖绝对标定
    if torch.is_tensor(mask_logits):
        probs = torch.sigmoid(mask_logits).detach().float().cpu().numpy()
    else:
        probs = 1.0 / (1.0 + np.exp(-mask_logits))

    if mask_bin.sum() > 0:
        conf_in = float(probs[mask_bin > 0].mean())
    else:
        conf_in = 0.0

    if prev_mask is None or prev_mask.sum() == 0 or mask_bin.sum() == 0:
        stab = 0.0
    else:
        stab = mask_iou(prev_mask, mask_bin)

    return 0.75 * conf_in + 0.25 * stab


def score_for_eviction(idx, entries: List[MemoryEntry], redundancy_penalty: float, redundancy_iou_thr: float):
    e = entries[idx]
    max_iou = 0.0
    redundant = 0.0
    for j, other in enumerate(entries):
        if j == idx:
            continue
        ov = mask_iou(e.mask, other.mask)
        max_iou = max(max_iou, ov)
    if max_iou >= redundancy_iou_thr:
        redundant = max_iou
    # 分数越低越应该被删：低质量+高冗余
    return e.quality - redundancy_penalty * redundant


def trim_memory_bank(anchor_bank: List[MemoryEntry], short_bank: List[MemoryEntry],
                     anchor_cap: int, short_cap: int,
                     redundancy_penalty: float, redundancy_iou_thr: float):
    # 先修剪 short memory
    while len(short_bank) > short_cap:
        scores = [
            score_for_eviction(i, short_bank, redundancy_penalty, redundancy_iou_thr)
            for i in range(len(short_bank))
        ]
        drop_idx = int(np.argmin(scores))
        short_bank.pop(drop_idx)

    # 再修剪 anchor memory（通常更保守）
    while len(anchor_bank) > anchor_cap:
        scores = [
            score_for_eviction(i, anchor_bank, redundancy_penalty * 0.5, redundancy_iou_thr)
            for i in range(len(anchor_bank))
        ]
        drop_idx = int(np.argmin(scores))
        anchor_bank.pop(drop_idx)


def upsert_memory_entry(anchor_bank: List[MemoryEntry], short_bank: List[MemoryEntry],
                        new_entry: MemoryEntry, args):
    # 同帧已存在则仅保留更高质量版本
    for bank in (anchor_bank, short_bank):
        for i, e in enumerate(bank):
            if e.frame_idx == new_entry.frame_idx:
                if new_entry.quality > e.quality:
                    bank[i] = new_entry
                return

    if new_entry.mem_type == "anchor":
        anchor_bank.append(new_entry)
    else:
        short_bank.append(new_entry)

    trim_memory_bank(
        anchor_bank,
        short_bank,
        anchor_cap=args.anchor_memory_size,
        short_cap=args.short_memory_size,
        redundancy_penalty=args.redundancy_penalty,
        redundancy_iou_thr=args.redundancy_iou_thr,
    )


def get_seed_init_mask(img_predictor, crops, seed, max_init_mask_area):
    best_axis = -1
    best_mask = None
    min_area = float("inf")

    for axis in [0, 1, 2]:
        img, box = crops[axis]
        local_pt = map_local_point(seed, axis, box)
        img_predictor.set_image(img)
        masks, scores, _ = img_predictor.predict(
            point_coords=np.array([local_pt]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        mask, _ = select_mask_with_constraints(masks, scores, max_init_mask_area)
        if mask is None:
            continue
        area = int(mask.sum())
        if area < min_area:
            min_area = area
            best_axis = axis
            best_mask = mask

    return best_axis, best_mask


def quality_memory_track_one_direction(
    video_predictor,
    vol_man,
    axis,
    box,
    idx_list,
    init_mask_2d,
    global_update_axis,
    vos_tmp_root,
    offload_video_to_cpu,
    args,
):
    if len(idx_list) == 0:
        return

    frames_rgb, valid_gidx = build_frames_rgb(vol_man, axis, box, idx_list)
    if len(frames_rgb) == 0:
        return

    tmp_dir = tempfile.mkdtemp(prefix="sam2_vos_qmem_", dir=vos_tmp_root)
    write_jpeg_frames(frames_rgb, tmp_dir)

    # 已预测mask缓存（frame_idx -> mask）
    known_masks: Dict[int, np.ndarray] = {0: init_mask_2d.astype(np.uint8)}

    # 自定义memory bank（anchor + short-term）
    anchor_bank: List[MemoryEntry] = [
        MemoryEntry(frame_idx=0, mask=init_mask_2d.astype(np.uint8), quality=1.0, mem_type="anchor")
    ]
    short_bank: List[MemoryEntry] = []

    # 先写回首帧
    vol_man.update_global_mask(init_mask_2d.astype(np.uint8), global_update_axis, (valid_gidx[0], *box[1:]))

    try:
        current = 0
        last_mask = init_mask_2d.astype(np.uint8)

        while current < len(valid_gidx) - 1:
            next_stop = min(len(valid_gidx) - 1, current + args.memory_chunk)

            with torch.inference_mode(), torch.autocast(str(vol_man.device), dtype=torch.bfloat16):
                state = video_predictor.init_state(
                    video_path=tmp_dir,
                    offload_video_to_cpu=offload_video_to_cpu,
                    offload_state_to_cpu=False,
                    async_loading_frames=False,
                )

                # 将当前bank中的记忆注入state（只注入已知帧）
                bank = sorted(anchor_bank + short_bank, key=lambda e: e.frame_idx)
                for entry in bank:
                    if entry.frame_idx > current:
                        continue
                    video_predictor.add_new_mask(
                        state,
                        frame_idx=entry.frame_idx,
                        obj_id=1,
                        mask=entry.mask.astype(bool),
                    )

                advanced = current
                for f_idx, _obj_ids, masks in video_predictor.propagate_in_video(state):
                    if f_idx <= current:
                        continue
                    if f_idx > next_stop:
                        break
                    if f_idx >= len(valid_gidx):
                        break

                    mm_logits = masks[0, 0]
                    if torch.is_tensor(mm_logits):
                        mm = (mm_logits > 0).to(torch.uint8).cpu().numpy()
                    else:
                        mm = (mm_logits > 0).astype(np.uint8)

                    if mm.sum() == 0:
                        advanced = f_idx
                        continue

                    known_masks[f_idx] = mm
                    vol_man.update_global_mask(mm, global_update_axis, (valid_gidx[f_idx], *box[1:]))

                    q = prediction_quality(mm_logits, mm, last_mask)
                    mem_type = "anchor" if q >= args.anchor_quality_thr else "short"
                    upsert_memory_entry(
                        anchor_bank,
                        short_bank,
                        MemoryEntry(frame_idx=f_idx, mask=mm, quality=float(q), mem_type=mem_type),
                        args,
                    )
                    last_mask = mm
                    advanced = f_idx

                if advanced == current:
                    # 兜底推进，防止死循环
                    advanced = min(current + 1, len(valid_gidx) - 1)

                current = advanced

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

    seeds: List[Tuple[int, int, int]] = []
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

    print(f"Starting SAM2 segmentation with quality-aware memory over {len(seeds)} seeds...")

    vos_tmp_root = os.path.join(args.output_dir, "_tmp_sam2_vos")
    os.makedirs(vos_tmp_root, exist_ok=True)

    with tqdm(total=len(seeds), desc="Tracking (SAM2 quality-memory)") as pbar:
        for seed in seeds:
            pbar.update(1)
            z, y, x = seed
            if vol_man.global_mask[z, y, x] > 0:
                continue

            crops = vol_man.get_triplane_crops(seed)
            best_axis, best_mask = get_seed_init_mask(
                img_predictor=img_predictor,
                crops=crops,
                seed=seed,
                max_init_mask_area=args.max_init_mask_area,
            )

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
            quality_memory_track_one_direction(
                video_predictor=video_predictor,
                vol_man=vol_man,
                axis=best_axis,
                box=box,
                idx_list=forward_idxs,
                init_mask_2d=best_mask,
                global_update_axis=best_axis,
                vos_tmp_root=vos_tmp_root,
                offload_video_to_cpu=args.vos_offload_video_to_cpu,
                args=args,
            )
            quality_memory_track_one_direction(
                video_predictor=video_predictor,
                vol_man=vol_man,
                axis=best_axis,
                box=box,
                idx_list=backward_idxs,
                init_mask_2d=best_mask,
                global_update_axis=best_axis,
                vos_tmp_root=vos_tmp_root,
                offload_video_to_cpu=args.vos_offload_video_to_cpu,
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
