import os
import shutil
import tempfile

import numpy as np
import torch
from PIL import Image

from .attention_metrics import (
    mean_attn_entropy,
    mean_attn_top1,
    mean_pointer_mass,
    sample_query_metrics_from_attn,
)
from .drift_metrics import gap_iou_from_resegment
from .image_utils import get_slice, to_uint8_rgb, write_jpeg_frames
from .plotting import plot_spatial_heatmap, plot_time_value_heatmap


def vos_track_one_direction(
    video_predictor,
    vol_man,
    axis,
    box,
    idx_list,
    init_mask_2d,
    global_update_axis,
    vos_tmp_root,
    offload_video_to_cpu,
    rope_attn,
    log_prefix="human",
    img_predictor=None,
):
    """
    Use SAM2 video predictor to propagate init_mask_2d across frames in idx_list.
    Updates vol_man.global_mask via vol_man.update_global_mask().
    """
    frames_rgb = []
    for vidx in idx_list:
        sl = get_slice(vol_man.vol, axis, vidx, box)
        if sl.size == 0:
            break
        rgb = to_uint8_rgb(sl)
        if rgb is None:
            break
        frames_rgb.append(rgb)

    if len(frames_rgb) == 0:
        return

    tmp_dir = tempfile.mkdtemp(prefix="sam2_vos_", dir=vos_tmp_root)
    write_jpeg_frames(frames_rgb, tmp_dir)

    ent_curve = []
    top1_curve = []
    ptr_curve = []
    ent_samples_per_frame = []
    top1_samples_per_frame = []
    ptr_samples_per_frame = []
    gap_iou_curve = []
    gap_iou_samples = []
    ptr_heat_sum = None
    ptr_heat_count = 0
    base_frame = frames_rgb[0] if frames_rgb else None

    try:
        with torch.inference_mode(), torch.autocast(str(vol_man.device), dtype=torch.bfloat16):
            state = video_predictor.init_state(
                video_path=tmp_dir,
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=False,
                async_loading_frames=False,
            )

            _, obj_ids, masks0 = video_predictor.add_new_mask(
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
            vol_man.update_global_mask(m0, global_update_axis, (idx_list[0], *box[1:]))

            for f_idx, obj_ids, masks in video_predictor.propagate_in_video(state):
                if f_idx < 0 or f_idx >= len(idx_list):
                    continue
                mm = masks[0, 0]
                if torch.is_tensor(mm):
                    mm = (mm > 0).to(torch.uint8).cpu().numpy()
                else:
                    mm = (mm > 0).astype(np.uint8)

                if mm.sum() == 0:
                    continue

                attn = getattr(rope_attn, "last_attn", None)
                if attn is None:
                    ent_curve.append(np.nan)
                    top1_curve.append(np.nan)
                    ptr_curve.append(np.nan)
                else:
                    ent_curve.append(mean_attn_entropy(attn))
                    top1_curve.append(mean_attn_top1(attn))
                    ptr_curve.append(mean_pointer_mass(attn, getattr(rope_attn, "last_num_k_exclude_rope", 0)))

                num_ptr = int(getattr(rope_attn, "last_num_k_exclude_rope", 0))
                query_mask = _make_query_mask(attn, mm)
                if attn is None:
                    ent_samples_per_frame.append(None)
                    top1_samples_per_frame.append(None)
                    ptr_samples_per_frame.append(None)
                else:
                    ent_s, top1_s, ptr_s = sample_query_metrics_from_attn(
                        attn,
                        num_ptr=num_ptr,
                        max_q=512,
                        query_mask=query_mask,
                    )
                    ent_samples_per_frame.append(ent_s)
                    top1_samples_per_frame.append(top1_s)
                    ptr_samples_per_frame.append(ptr_s)

                ptr_map = _memory_pointer_heatmap(attn, num_ptr)
                if ptr_map is not None:
                    if ptr_heat_sum is None:
                        ptr_heat_sum = ptr_map.astype(np.float32)
                    else:
                        ptr_heat_sum += ptr_map.astype(np.float32)
                    ptr_heat_count += 1

                if img_predictor is None:
                    gap_iou_curve.append(np.nan)
                    gap_iou_samples.append(None)
                else:
                    gap_iou = gap_iou_from_resegment(img_predictor, frames_rgb[f_idx], mm)
                    if gap_iou is None:
                        gap_iou_curve.append(np.nan)
                        gap_iou_samples.append(None)
                    else:
                        gap_iou_curve.append(gap_iou)
                        gap_iou_samples.append(np.array([gap_iou], dtype=np.float32))

                vol_man.update_global_mask(mm, global_update_axis, (idx_list[f_idx], *box[1:]))
    finally:
        os.makedirs(os.path.join("./data/macaque", "attn_viz"), exist_ok=True)
        out_dir = os.path.join("./data/human", "attn_viz")

        os.makedirs(out_dir, exist_ok=True)
        _plot_curve(ent_curve, "entropy", log_prefix, out_dir)
        _plot_curve(top1_curve, "top1 mass", log_prefix, out_dir)
        _plot_curve(ptr_curve, "pointer mass", log_prefix, out_dir)
        _plot_curve(gap_iou_curve, "gap IoU (video vs image)", log_prefix, out_dir)

        out_dir = os.path.join(vos_tmp_root, "attn_viz")
        os.makedirs(out_dir, exist_ok=True)

        plot_time_value_heatmap(
            ent_samples_per_frame,
            title=f"{log_prefix} entropy dist",
            savepath=os.path.join(out_dir, f"{log_prefix}_entropy_heat.png"),
            n_time_bins=50,
            n_val_bins=50,
            log_scale=True,
        )

        plot_time_value_heatmap(
            top1_samples_per_frame,
            title=f"{log_prefix} top1 dist",
            savepath=os.path.join(out_dir, f"{log_prefix}_top1_heat.png"),
            n_time_bins=50,
            n_val_bins=50,
            log_scale=True,
        )

        plot_time_value_heatmap(
            ptr_samples_per_frame,
            title=f"{log_prefix} ptrmass dist",
            savepath=os.path.join(out_dir, f"{log_prefix}_ptrmass_heat.png"),
            n_time_bins=50,
            n_val_bins=50,
            log_scale=True,
        )

        plot_time_value_heatmap(
            gap_iou_samples,
            title=f"{log_prefix} gap IoU dist",
            savepath=os.path.join(out_dir, f"{log_prefix}_gap_iou_heat.png"),
            n_time_bins=50,
            n_val_bins=50,
            log_scale=False,
            val_range=(0.0, 1.0),
        )

        if ptr_heat_sum is not None and ptr_heat_count > 0:
            mean_ptr_heat = ptr_heat_sum / float(ptr_heat_count)
            plot_spatial_heatmap(
                mean_ptr_heat,
                savepath=os.path.join(out_dir, f"{log_prefix}_memory_ptr_heat.png"),
            )
            plot_spatial_heatmap(
                mean_ptr_heat,
                savepath=os.path.join(out_dir, f"{log_prefix}_memory_ptr_overlay.png"),
                base_image=base_frame,
            )

        shutil.rmtree(tmp_dir, ignore_errors=True)


def _plot_curve(values, label, log_prefix, out_dir):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(values)
    plt.title(f"{log_prefix} {label}")
    plt.xlabel("frame idx")
    plt.ylabel(label)
    plt.tight_layout()
    safe_label = label.replace(" ", "_").replace("/", "_")
    plt.savefig(os.path.join(out_dir, f"{log_prefix}_{safe_label}.png"), dpi=200)
    plt.close()


def _make_query_mask(attn, mask_2d):
    if attn is None or mask_2d is None:
        return None
    sq = int(attn.shape[-2])
    side = int(np.sqrt(sq))
    if side * side != sq:
        return None
    mask = (mask_2d > 0).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask)
    mask_resized = mask_img.resize((side, side), resample=Image.NEAREST)
    mask_arr = np.asarray(mask_resized) > 0
    return torch.from_numpy(mask_arr.astype(np.bool_))


def _memory_pointer_heatmap(attn, num_ptr):
    if attn is None or num_ptr <= 0:
        return None
    sq = int(attn.shape[-2])
    side = int(np.sqrt(sq))
    if side * side != sq:
        return None
    p = attn.mean(dim=1)[0]  # [Sq,Sk]
    ptr_mass = p[:, -num_ptr:].sum(dim=-1)  # [Sq]
    ptr_mass = ptr_mass.detach().cpu().numpy().reshape(side, side)
    return ptr_mass
