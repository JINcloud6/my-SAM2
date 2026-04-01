#分割时可以不断返回新种子

import argparse
import os
import shutil
import tempfile

import numpy as np
import torch
from scipy import ndimage
from tqdm import tqdm
import hydra
from PIL import Image

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .data_manager import VolumeManager
from .preprocessing import get_multi_axis_init_seg, get_seeds_from_init_seg, get_seg
from .utils import select_masks

def _recover_missing_components_sam2(
    img_predictor,
    vol_man,
    prev_components,
    curr_mask,
    curr_rgb,          # 当前帧的 RGB (H,W,3)
    axis,
    vol_idx,           # 当前帧在 volume 的真实索引（不是 f_idx）
    box,
    overlap_threshold=0.5,
):
    """
    若上一帧连通域在当前帧消失，则在当前帧以其 center 重新点提示分割，恢复缺失部分。
    返回 recovered_mask (uint8) 或 None。
    """
    if not prev_components:
        return None

    if curr_mask is None:
        curr_mask = np.zeros_like(prev_components[0]["mask"], dtype=np.uint8)

    # 找出缺失连通域：与 curr_mask 无重叠
    missing = []
    for comp in prev_components:
        if (curr_mask > 0).any() and (curr_mask & comp["mask"]).any():
            continue
        missing.append(comp)
    if not missing:
        return None

    # 取 crop 范围内已经落到 global_mask 的区域，避免重复/串段
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

        # 这里别太严，恢复阶段建议 thr=0 或较低，并用后面的 overlap 来抑制错误
        cand_mask, _ = select_masks(
            masks, scores,
            thr=0.0, crit="max",
            max_size=5000,
            min_circularity=0.0
        )
        if cand_mask is None or cand_mask.sum() == 0:
            continue

        # 若与已有 global mask 重叠过大，说明可能串到已分割血管/其它段，跳过
        cand_b = np.asarray(cand_mask).astype(bool)
        exist_b = np.asarray(existing).astype(bool)
        overlap = np.logical_and(cand_b, exist_b).sum()

        # overlap = (cand_mask & (existing > 0)).sum()
        ratio = overlap / (cand_mask.sum() + 1e-6)
        if ratio > overlap_threshold:
            continue

        recovered = np.logical_or(recovered, cand_mask)

    if recovered.sum() == 0:
        return None

    return recovered.astype(np.uint8)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2_checkpoint", required=True, help="Path to SAM2 checkpoint",
                        default="/home/jiangshuai/code/sam2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2_model_cfg", required=True, help="Path to SAM2 model config",
                        default="/home/jiangshuai/code/sam2/sam2.1_hiera_l.yaml")
    parser.add_argument("--volume_path", required=True, help="Path to .h5 or .nii/.nii.gz file")
    parser.add_argument("--output_dir", default="./bv_seg_output", help="Directory to save outputs")
    parser.add_argument("--output_filename", default="segmentation.nii.gz", help="Output filename")
    parser.add_argument("--dataset_key", default="main", help="Key name in h5 file")
    parser.add_argument("--seed_file", default=None, help="Optional path to seed list (z,y,x per line)")
    parser.add_argument("--axis", type=int, default=3, help="Axis to generate init seg (0=Z,1=Y,2=X), 3=all")
    parser.add_argument("--stride", type=int, default=5, help="Stride for init seg generation")
    parser.add_argument("--gaussian_kernel", type=int, default=5)
    parser.add_argument("--min_bright", type=int, default=40)
    parser.add_argument("--remove_portion", type=float, default=0.98)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--need_transpose", default="False")
    parser.add_argument("--max_track_distance", type=int, default=2000)
    parser.add_argument("--recover_overlap_threshold", type=float, default=0.5)

    # video predictor / io knobs
    parser.add_argument("--vos_offload_video_to_cpu", action="store_true",
                        help="Offload video frames to CPU inside SAM2 video predictor (save GPU mem)")
    parser.add_argument("--keep_tmp_vos_frames", action="store_true",
                        help="Do not delete tmp JPEG frames dirs (debug)")

    return parser.parse_args()


def _extract_components(binary_mask):
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


def _predict_from_point(predictor, image, point):
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        point_coords=np.array([point]),
        point_labels=np.array([1]),
        multimask_output=True,
    )
    print(scores)
    mask, _ = select_masks(masks, scores, thr=0.6, crit="max", max_size=10000, min_circularity=0.0)
    print("mask is ",mask)
    if mask is None:
        return None
    return mask.astype(np.uint8)


def _map_local_point(seed, axis, box):
    if axis == 0:
        return [seed[2] - box[3], seed[1] - box[1]]
    if axis == 1:
        return [seed[2] - box[3], seed[0] - box[1]]
    return [seed[1] - box[3], seed[0] - box[1]]

def _local_center_to_global_seed(axis: int, vol_idx: int, box: tuple, cy: float, cx: float):
    """
    cy,cx: 在 crop slice 上的坐标（row=y, col=x），对应 _get_slice 返回的 2D 图像坐标系
    box: crops[axis] 里返回的 box（你代码里用 box[1],box[2],box[3],box[4] 做 crop）
    返回 (z,y,x)
    """
    cy_i = int(round(cy))
    cx_i = int(round(cx))

    if axis == 0:
        # slice = vol[z, y, x], crop in (y,x)
        z = vol_idx
        y = box[1] + cy_i
        x = box[3] + cx_i
        return (z, y, x)

    if axis == 1:
        # slice = vol[z, y, x], take y=vol_idx, crop in (z,x)
        z = box[1] + cy_i
        y = vol_idx
        x = box[3] + cx_i
        return (z, y, x)

    # axis == 2:
    # slice = vol[z, y, x], take x=vol_idx, crop in (z,y)
    z = box[1] + cy_i
    y = box[3] + cx_i
    x = vol_idx
    return (z, y, x)


def _get_slice(volume, axis, curr_idx, box):
    if axis == 0:
        sl = volume[curr_idx, box[1]:box[2], box[3]:box[4]]
    elif axis == 1:
        sl = volume[box[1]:box[2], curr_idx, box[3]:box[4]]
    else:
        sl = volume[box[1]:box[2], box[3]:box[4], curr_idx]
    return sl


def _to_uint8_rgb(slice_2d: np.ndarray) -> np.ndarray:
    """
    Convert a 2D slice to uint8 RGB for SAM2.
    """
    sl = slice_2d
    if sl.size == 0:
        return None
    if sl.dtype != np.uint8:
        sl = sl.astype(np.float32)
        mn = float(np.min(sl))
        mx = float(np.max(sl))
        if mx <= mn + 1e-6:
            sl = np.zeros_like(sl, dtype=np.uint8)
        else:
            sl = (sl - mn) / (mx - mn + 1e-6) * 255.0
            sl = np.clip(sl, 0, 255).astype(np.uint8)
    rgb = np.stack([sl] * 3, axis=-1)
    return rgb


def _write_jpeg_frames(frames_rgb: list[np.ndarray], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # ensure empty
    for fn in os.listdir(out_dir):
        fp = os.path.join(out_dir, fn)
        if os.path.isfile(fp):
            os.remove(fp)

    for i, fr in enumerate(frames_rgb):
        # SAM2 loader commonly expects names like 00000.jpg ...
        path = os.path.join(out_dir, f"{i:05d}.jpg")
        Image.fromarray(fr).save(path, quality=95, subsampling=0)


import numpy as np
from scipy import ndimage

"""
过滤掉当前帧中与上一帧重叠度很低的mask
因为这意味着误分
"""
def filter_curr_mask_by_prev(
    prev_mask: np.ndarray,
    curr_mask: np.ndarray,
    min_curr_area: int = 80,
    min_prev_area: int = 80,
    min_inter: int = 20,
    min_iou: float = 0.05,
    min_inter_over_curr: float = 0.10,
    min_inter_over_prev: float = 0.05,
):
    """
    返回 filtered_curr_mask, 以及一些调试信息
    规则：对每个 curr 连通域，找与之交集最大的 prev 连通域；只有满足阈值才保留
    """
    prev_b = prev_mask.astype(bool)
    curr_b = curr_mask.astype(bool)

    prev_lab, n_prev = ndimage.label(prev_b)
    curr_lab, n_curr = ndimage.label(curr_b)

    if n_curr == 0 or n_prev == 0:
        return curr_mask, {"n_prev": n_prev, "n_curr": n_curr, "kept": 0}

    # 每个 label 的面积（0 是背景）
    prev_area = np.bincount(prev_lab.ravel(), minlength=n_prev + 1)
    curr_area = np.bincount(curr_lab.ravel(), minlength=n_curr + 1)

    # 计算交集计数：对 (prev_id, curr_id) 的像素共同出现次数做 bincount
    # pair_id = prev_id * (n_curr+1) + curr_id
    pair = prev_lab.ravel() * (n_curr + 1) + curr_lab.ravel()
    inter_counts = np.bincount(pair, minlength=(n_prev + 1) * (n_curr + 1)).reshape(n_prev + 1, n_curr + 1)

    kept_labels = []
    for c in range(1, n_curr + 1):
        # a_c = curr_area[c]
        # print("ac:",a_c)
        # if a_c < min_curr_area:
        #     continue  # 面积太小，直接丢

        # # 找最佳匹配的 prev label（交集最大）
        # p = int(np.argmax(inter_counts[:, c]))
        # inter = int(inter_counts[p, c])

        # if p == 0:
        #     # 只与背景相交 => 完全新出现的块；通常是漂移/误分，先丢
        #     # 如果你希望允许“真正的新分叉”，可以放宽这里（见下方备注）
        #     continue

        # a_p = int(prev_area[p])
        # print("ap",a_p)
        # if a_p < min_prev_area:
        #     continue

        # union = a_p + a_c - inter
        # iou = inter / (union + 1e-6)
        # inter_over_c = inter / (a_c + 1e-6)
        # inter_over_p = inter / (a_p + 1e-6)

        # print(f"iou:{iou},interoverc:{inter_over_c},interoverp{inter_over_p}")
        # if inter < min_inter:
        #     continue
        # if iou < min_iou:
        #     continue
        # if inter_over_c < min_inter_over_curr:
        #     continue
        # if inter_over_p < min_inter_over_prev:
        #     continue

        # kept_labels.append(c)
        # curr 连通域面积
        a_c = curr_area[c]
        if a_c < min_curr_area:
            continue

        # 与上一帧前景的总交集（忽略背景行 0）
        inter_fg_total = int(inter_counts[1:, c].sum())
        if inter_fg_total == 0:
            # 完全没有继承：通常是漂移/误分
            # 如果你想把它当“新分叉候选”，不要在这里保留，而是另行处理（比如返回seed）
            continue

        # 找最佳前景匹配（只在 p>=1 中找）
        best_p_rel = int(np.argmax(inter_counts[1:, c]))  # 0..n_prev-1
        p = best_p_rel + 1
        inter_best = int(inter_counts[p, c])

        a_p = int(prev_area[p])
        if a_p < min_prev_area:
            continue

        # 继承比例（两个方向）
        inter_over_prev = inter_best / (a_p + 1e-6)
        inter_over_curr = inter_fg_total / (a_c + 1e-6)

        # 可选：IoU 用 best 匹配来算
        union = a_p + a_c - inter_best
        iou = inter_best / (union + 1e-6)

        print(f"iou:{iou},interoverc:{inter_over_curr},interoverp{inter_over_prev}")
        if inter_best < min_inter:
            continue
        if inter_over_prev < min_inter_over_prev:
            continue
        if inter_over_curr < min_inter_over_curr:
            continue
        if iou < min_iou:
            continue

        kept_labels.append(c)


    if not kept_labels:
        return np.zeros_like(curr_mask, dtype=np.uint8), {"n_prev": n_prev, "n_curr": n_curr, "kept": 0}

    kept = np.isin(curr_lab, kept_labels)
    return kept.astype(np.uint8), {"n_prev": n_prev, "n_curr": n_curr, "kept": len(kept_labels)}


def _vos_track_one_direction(
    video_predictor,
    vol_man: VolumeManager,
    axis: int,
    box: tuple,
    idx_list: list[int],
    init_mask_2d: np.ndarray,
    global_update_axis: int,
    vos_tmp_root: str,
    offload_video_to_cpu: bool,
    img_predictor,                 # 现在不会用到，但先保留签名避免你其它地方报错
    keep_thr: float = 0.7,         # 连续性阈值（越大越敏感）
    max_return_seeds: int = 5,     # 每次断裂最多返回几个新 seed，防止爆炸
):
    """
    VOS 追踪：一旦检测到连通域缺失，不在当前 seed 内修复，
    而是返回缺失连通域中心点对应的“新 seed”，由外层队列继续处理。
    """
    new_seeds = []

    # 1) build RGB frames
    frames_rgb = []
    for vidx in idx_list:
        sl = _get_slice(vol_man.vol, axis, vidx, box)
        if sl.size == 0:
            break
        rgb = _to_uint8_rgb(sl)
        if rgb is None:
            break
        frames_rgb.append(rgb)

    if len(frames_rgb) == 0:
        return new_seeds

    # 2) write to temp JPEG dir for init_state
    tmp_dir = tempfile.mkdtemp(prefix="sam2_vos_", dir=vos_tmp_root)
    _write_jpeg_frames(frames_rgb, tmp_dir)

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

            # frame 0 update
            m0 = masks0[0, 0]
            if torch.is_tensor(m0):
                prev_mask = (m0 > 0).to(torch.uint8).cpu().numpy()
            else:
                prev_mask = (m0 > 0).astype(np.uint8)

            # 写入 global
            vol_man.update_global_mask(prev_mask, global_update_axis, (idx_list[0], *box[1:]))

            prev_components = None
            prev_vol_idx = idx_list[0]

            # propagate
            for f_idx, _, masks in video_predictor.propagate_in_video(state):
                if f_idx < 0 or f_idx >= len(idx_list):
                    continue

                vol_idx = idx_list[f_idx]

                mm = masks[0, 0]
                if torch.is_tensor(mm):
                    curr_mask = (mm > 0).to(torch.uint8).cpu().numpy()
                else:
                    curr_mask = (mm > 0).astype(np.uint8)

                # 追踪到边界：直接结束该方向
                if curr_mask is None or curr_mask.sum() == 0:
                    break

                # --- cheap gate：连续性差才进一步判断是否缺失 ---
                prev_b = prev_mask.astype(bool)
                curr_b = curr_mask.astype(bool)
                inter = np.logical_and(prev_b, curr_b).sum()
                keep = inter / (prev_mask.sum() + 1e-6)
                move = inter / (curr_mask.sum() + 1e-6)

                if move < keep_thr:
                    print("出现大面积新分割:",vol_idx)
                    curr_mask, info = filter_curr_mask_by_prev(
                        prev_mask=prev_mask,
                        curr_mask=curr_mask,
                        min_curr_area=80,
                        min_prev_area=80,
                        min_inter=20,
                        min_iou=0.05,
                        min_inter_over_curr=0.20,
                        min_inter_over_prev=0.05,
                    )

                    curr_b = curr_mask.astype(bool)
                    inter = np.logical_and(prev_b, curr_b).sum()
                    keep = inter / (prev_mask.sum() + 1e-6)
                    move = inter / (curr_mask.sum() + 1e-6)
                if keep < keep_thr:
                    print("keep<keep_thr:",vol_idx)
                    # 仅在需要时提取上一帧连通域
                    if prev_components is None:
                        prev_components = _extract_components(prev_mask)

                    # 找出“上一帧存在、当前帧无重叠”的连通域

                    min_comp_area = 80        # 小连通域直接忽略（你要根据分辨率调）
           

                    missing = []
                    for comp in prev_components:
                        area = comp["area"]
                        if area < min_comp_area:
                            # 误分类型：面积太小（噪声/碎片）
                            continue
                        comp_b = np.asarray(comp["mask"]).astype(bool)
                        # 若与 curr_mask 无任何重叠，则认为缺失
                        if np.logical_and(comp_b, curr_b).sum() <= comp_b.sum()*0.1:
                            missing.append(comp)

                    if missing:
                        print("find stop in :",vol_idx)
                        # 按面积从大到小，最多返回 max_return_seeds 个
                        missing.sort(key=lambda c: c["area"], reverse=True)
                        for comp in missing[:max_return_seeds]:
                            cy, cx = comp["center"]
                            seed3d = _local_center_to_global_seed(
                                axis=axis,
                                vol_idx=vol_idx,     # 注意：使用“上一帧”的 slice index
                                box=box,
                                cy=cy, cx=cx,
                            )
                            z, y, x = seed3d

                            # 边界/越界保护
                            if not (0 <= z < vol_man.shape[0] and 0 <= y < vol_man.shape[1] and 0 <= x < vol_man.shape[2]):
                                continue

                            # 已经分割过的点不再返回
                            if vol_man.global_mask[z, y, x] > 0:
                                continue
                            
                            print("add seed",seed3d)
                            new_seeds.append(seed3d)

                        # 一旦出现缺失：立即结束该方向，让外层 seed 队列处理
                        # return new_seeds

                # 正常情况：继续写入 global，并更新 prev
                vol_man.update_global_mask(curr_mask, global_update_axis, (vol_idx, *box[1:]))
                prev_mask = curr_mask
                prev_vol_idx = vol_idx
                prev_components = None  # 只有 keep 低时才会重新 extract

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return new_seeds



def run_segmentation():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    vol_man = VolumeManager(args.volume_path, key=args.dataset_key)
    # 如果你的 VolumeManager 里没有 device 字段，给它补一个
    if not hasattr(vol_man, "device"):
        vol_man.device = device

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('use_sam2', version_base='1.2')

    # --- build SAM2 image predictor (for best_axis & initial mask) ---
    sam2_img_model = build_sam2(args.sam2_model_cfg, args.sam2_checkpoint, device=device)
    img_predictor = SAM2ImagePredictor(sam2_img_model)

    # --- build SAM2 video predictor (for VOS tracking) ---
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
        seeds = get_seeds_from_init_seg(init_seg)

    if not seeds:
        raise RuntimeError("No seeds available for SAM2 tracking.")

    # print(f"Starting SAM2 segmentation with {len(seeds)} seeds...")

    # tmp root for VOS frames
    vos_tmp_root = os.path.join(args.output_dir, "_tmp_sam2_vos")
    os.makedirs(vos_tmp_root, exist_ok=True)

    # with tqdm(total=len(seeds), desc="Tracking (SAM2 VOS)") as pbar:
    #     for seed_index, seed in enumerate(seeds):
    #         pbar.update(1)
    #         z, y, x = seed
    #         if vol_man.global_mask[z, y, x] > 0:
    #             continue

    #         crops = vol_man.get_triplane_crops(seed)
    #         best_axis = -1
    #         best_mask = None
    #         min_area = float("inf")

    #         # 1) per-axis init segmentation on the seed slice (still image)
    #         with torch.inference_mode(), torch.autocast(args.device, dtype=torch.bfloat16):
    #             for axis in [0, 1, 2]:
    #                 img, box = crops[axis]
    #                 local_pt = _map_local_point(seed, axis, box)
    #                 mask = _predict_from_point(img_predictor, img, local_pt)
    #                 if mask is None:
    #                     continue
    #                 area = int(mask.sum())
    #                 if area < min_area:
    #                     min_area = area
    #                     best_axis = axis
    #                     best_mask = mask

    #         if best_axis == -1 or best_mask is None or best_mask.sum() == 0:
    #             continue

    #         # 2) build index lists (forward/backward) around the seed on the chosen axis
    #         track_dim = [0, 1, 2][best_axis]
    #         start_idx = seed[track_dim]
    #         max_dist = args.max_track_distance

    #         # IMPORTANT: idx_list[0] corresponds to the frame where best_mask was predicted
    #         forward_idxs = list(range(start_idx, min(start_idx + max_dist, vol_man.shape[track_dim])))
    #         backward_idxs = list(range(start_idx, max(start_idx - max_dist, -1), -1))

    #         _, box = crops[best_axis]

    #         # 3) VOS tracking using SAM2VideoPredictor
    #         # forward
    #         _vos_track_one_direction(
    #             video_predictor=video_predictor,
    #             vol_man=vol_man,
    #             axis=best_axis,
    #             box=box,
    #             idx_list=forward_idxs,
    #             init_mask_2d=best_mask,
    #             global_update_axis=best_axis,
    #             vos_tmp_root=vos_tmp_root,
    #             offload_video_to_cpu=args.vos_offload_video_to_cpu,
    #             img_predictor=img_predictor
    #         )
    #         # backward
    #         _vos_track_one_direction(
    #             video_predictor=video_predictor,
    #             vol_man=vol_man,
    #             axis=best_axis,
    #             box=box,
    #             idx_list=backward_idxs,
    #             init_mask_2d=best_mask,
    #             global_update_axis=best_axis,
    #             vos_tmp_root=vos_tmp_root,
    #             offload_video_to_cpu=args.vos_offload_video_to_cpu,
    #             img_predictor=img_predictor
    #         )
    from collections import deque

# ...

    print(f"Starting SAM2 segmentation with {len(seeds)} initial seeds...")

    seed_q = deque(seeds)
    seen = set(seeds)  # 用于去重，防止重复回灌导致死循环

    with tqdm(total=0, desc="Tracking (SAM2 VOS)", dynamic_ncols=True) as pbar:
        while seed_q:
            seed = seed_q.popleft()
            pbar.total += 1
            pbar.update(1)

            z, y, x = seed
            if vol_man.global_mask[z, y, x] > 0:
                continue

            crops = vol_man.get_triplane_crops(seed)
            best_axis = -1
            best_mask = None
            min_area = float("inf")

            # 1) per-axis init segmentation on the seed slice (still image)
            with torch.inference_mode(), torch.autocast(args.device, dtype=torch.bfloat16):
                for axis in [0, 1, 2]:
                    img, box = crops[axis]
                    local_pt = _map_local_point(seed, axis, box)
                    mask = _predict_from_point(img_predictor, img, local_pt)
                    if mask is None:
                        continue
                    area = int(mask.sum())
                    if area < min_area:
                        min_area = area
                        best_axis = axis
                        best_mask = mask

            if best_axis == -1 or best_mask is None or best_mask.sum() == 0:
                continue

            # 2) build index lists (forward/backward) around the seed on the chosen axis
            track_dim = [0, 1, 2][best_axis]
            start_idx = seed[track_dim]
            max_dist = args.max_track_distance

            forward_idxs = list(range(start_idx, min(start_idx + max_dist, vol_man.shape[track_dim])))
            backward_idxs = list(range(start_idx, max(start_idx - max_dist, -1), -1))

            _, box = crops[best_axis]

            # 3) VOS tracking; if missing occurs, get new seeds and enqueue
            new1 = _vos_track_one_direction(
                video_predictor=video_predictor,
                vol_man=vol_man,
                axis=best_axis,
                box=box,
                idx_list=forward_idxs,
                init_mask_2d=best_mask,
                global_update_axis=best_axis,
                vos_tmp_root=vos_tmp_root,
                offload_video_to_cpu=args.vos_offload_video_to_cpu,
                img_predictor=img_predictor,
                keep_thr=0.5,
                max_return_seeds=5,
            )
            for ns in new1:
                if ns not in seen:
                    seen.add(ns)
                    seed_q.append(ns)

            new2 = _vos_track_one_direction(
                video_predictor=video_predictor,
                vol_man=vol_man,
                axis=best_axis,
                box=box,
                idx_list=backward_idxs,
                init_mask_2d=best_mask,
                global_update_axis=best_axis,
                vos_tmp_root=vos_tmp_root,
                offload_video_to_cpu=args.vos_offload_video_to_cpu,
                img_predictor=img_predictor,
                keep_thr=0.5,
                max_return_seeds=5,
            )
            for ns in new2:
                if ns not in seen:
                    seen.add(ns)
                    seed_q.append(ns)

    if (not args.keep_tmp_vos_frames) and os.path.isdir(vos_tmp_root):
        shutil.rmtree(vos_tmp_root, ignore_errors=True)

    print("Cleanup...")
    final_path = os.path.join(args.output_dir, args.output_filename)
    flag = (args.need_transpose != "False")
    vol_man.save(final_path, flag)
    print(f"Done! Saved to {final_path}")


if __name__ == "__main__":
    run_segmentation()
