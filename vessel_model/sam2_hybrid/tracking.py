import shutil
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from ..global_memory_pool import GlobalMemoryPool
from ..sam2_baseline.image_utils import get_slice, to_uint8_rgb, write_jpeg_frames
from .common import SeedTask, sample_respawn_seeds_from_mask
from .segment_trust import SegmentFrame, commit_segment


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

    conf = float(probs[mask_bin > 0].mean()) if mask_bin.sum() > 0 else 0.0
    if prev_mask is None or prev_mask.sum() == 0 or mask_bin.sum() == 0:
        stab = 0.0
    else:
        stab = mask_iou(prev_mask, mask_bin)
    return 0.7 * conf + 0.3 * stab


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


def promote_segment_to_longterm(
    video_predictor,
    state,
    seg_frames,
    longterm_bank,
    global_memory_pool,
    axis,
    seed,
    direction,
    args,
):
    if len(seg_frames) == 0:
        return

    seg_q = float(np.mean([x.quality for x in seg_frames]))
    if seg_q < args.longterm_quality_thr:
        return

    best_per_frame = {}
    for item in seg_frames:
        prev = best_per_frame.get(item.frame_idx, None)
        if (prev is None) or (item.quality > prev.quality):
            best_per_frame[item.frame_idx] = item

    ordered_items = sorted(best_per_frame.values(), key=lambda x: x.frame_idx)
    if len(ordered_items) == 0:
        return

    for item in ordered_items:
        video_predictor.promote_frame_output_to_cond(state, frame_idx=item.frame_idx, obj_id=1)
        global_memory_pool.add_entry(
            axis=axis,
            physical_frame_idx=item.physical_frame_idx,
            mask=item.mask,
            quality=item.quality,
            source_seed=seed,
            source_direction=direction,
        )
    longterm_bank.append(LongTermSegment(frame_items=ordered_items, seg_quality=seg_q))

    while len(longterm_bank) > args.max_longterm_segments:
        worst_idx = int(np.argmin([x.seg_quality for x in longterm_bank]))
        worst = longterm_bank.pop(worst_idx)
        for item in worst.frame_items:
            video_predictor.demote_frame_output_from_cond(state, frame_idx=item.frame_idx, obj_id=1)


def track_one_direction_hybrid(
    video_predictor,
    vol_man,
    axis,
    box,
    idx_list,
    init_masks_by_global_idx,
    vos_tmp_root,
    offload_video_to_cpu,
    global_memory_pool: GlobalMemoryPool,
    seed: Tuple[int, int, int],
    direction: str,
    covered_mask,
    trusted_mask,
    start_untrusted_count: int,
    args,
):
    empty_stats = {
        "longterm_segments": 0,
        "trusted_segments": 0,
        "untrusted_segments": 0,
        "final_untrusted_count": int(start_untrusted_count),
        "terminated_large_mask": False,
    }
    if len(idx_list) == 0:
        return [], empty_stats

    frames_rgb, valid_gidx = build_frames_rgb(vol_man, axis, box, idx_list)
    if len(frames_rgb) == 0:
        return [], empty_stats

    tmp_dir = tempfile.mkdtemp(prefix="sam2_vos_hybrid_", dir=vos_tmp_root)
    write_jpeg_frames(frames_rgb, tmp_dir)

    spawned_tasks: List[SeedTask] = []
    spawned_set: Set[Tuple[int, int, int]] = set()
    promoted_segments = 0
    trusted_segments = 0
    untrusted_segments = 0
    terminated_large_mask = False

    try:
        with torch.inference_mode(), torch.autocast(str(vol_man.device), dtype=torch.bfloat16):
            state = video_predictor.init_state(
                video_path=tmp_dir,
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=False,
                async_loading_frames=False,
            )

            idx_to_frame = {gidx: fi for fi, gidx in enumerate(valid_gidx)}
            used = 0
            prev_nonempty_mask = None
            prev_global_idx = None
            trust_seg_cache: List[SegmentFrame] = []

            for gidx in valid_gidx:
                m = init_masks_by_global_idx.get(gidx, None)
                if m is None or int(m.sum()) == 0:
                    continue
                fidx = idx_to_frame[gidx]
                video_predictor.add_new_mask(state, frame_idx=fidx, obj_id=1, mask=m.astype(bool))
                vol_man.update_global_mask(m.astype(np.uint8), axis, (gidx, *box[1:]))
                prev_nonempty_mask = m.astype(np.uint8)
                prev_global_idx = gidx
                if args.enable_respawn:
                    trust_seg_cache.append(
                        SegmentFrame(axis=axis, global_frame_idx=int(gidx), box=box, mask=m.astype(np.uint8))
                    )
                used += 1

            if used == 0:
                g0 = valid_gidx[0]
                m0 = init_masks_by_global_idx.get(g0, None)
                if m0 is None or int(m0.sum()) == 0:
                    return [], empty_stats
                video_predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=m0.astype(bool))
                vol_man.update_global_mask(m0.astype(np.uint8), axis, (g0, *box[1:]))
                prev_nonempty_mask = m0.astype(np.uint8)
                prev_global_idx = g0
                if args.enable_respawn:
                    trust_seg_cache.append(
                        SegmentFrame(axis=axis, global_frame_idx=int(g0), box=box, mask=m0.astype(np.uint8))
                    )

            if not getattr(args, "disable_longterm_memory", False):
                init_mask = prev_nonempty_mask
                init_radius = float(np.sqrt(float(np.count_nonzero(init_mask)) / np.pi)) if init_mask is not None else 0.0
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

            longterm_bank: List[LongTermSegment] = []
            seg_cache: List[FrameQuality] = []
            prev_mask = prev_nonempty_mask
            current_untrusted_count = int(start_untrusted_count)

            for f_idx, _obj_ids, masks in video_predictor.propagate_in_video(state):
                if f_idx < 0 or f_idx >= len(valid_gidx):
                    continue

                mm_logits = masks[0, 0]
                if torch.is_tensor(mm_logits):
                    mm = (mm_logits > 0).to(torch.uint8).cpu().numpy()
                else:
                    mm = (mm_logits > 0).astype(np.uint8)

                gidx = valid_gidx[f_idx]
                if mm.sum() == 0:
                    if args.enable_respawn and len(trust_seg_cache) >= args.min_segment_frames:
                        force_trusted = current_untrusted_count >= args.max_untrusted_segments_per_lineage
                        new_seeds, decision = commit_segment(
                            seg_frames=trust_seg_cache,
                            covered_mask=covered_mask,
                            trusted_mask=trusted_mask,
                            track_axis=axis,
                            args=args,
                            force_trusted=force_trusted,
                        )
                        if decision is not None:
                            if decision.is_trusted:
                                trusted_segments += 1
                            else:
                                untrusted_segments += 1
                        if (decision is not None) and (not decision.is_trusted):
                            current_untrusted_count += 1
                        for child_seed in new_seeds:
                            task = SeedTask(seed=child_seed, is_original=False, untrusted_count=current_untrusted_count)
                            if task.seed in spawned_set:
                                continue
                            spawned_set.add(task.seed)
                            spawned_tasks.append(task)
                        trust_seg_cache = []
                    continue

                slice_area = float(mm.shape[0] * mm.shape[1])
                if slice_area > 0:
                    mask_ratio = float(mm.sum()) / slice_area
                    if mask_ratio > float(args.max_slice_mask_ratio):
                        terminated_large_mask = True
                        trust_seg_cache = []
                        seg_cache = []
                        break

                vol_man.update_global_mask(mm, axis, (gidx, *box[1:]))
                q = frame_quality_from_logits(mm_logits, mm, prev_mask)
                seg_cache.append(
                    FrameQuality(
                        frame_idx=f_idx,
                        physical_frame_idx=int(gidx),
                        mask=mm,
                        quality=float(q),
                    )
                )
                prev_mask = mm
                prev_nonempty_mask = mm
                prev_global_idx = gidx
                if args.enable_respawn:
                    trust_seg_cache.append(
                        SegmentFrame(axis=axis, global_frame_idx=int(gidx), box=box, mask=mm.astype(np.uint8))
                    )

                if not getattr(args, "disable_longterm_memory", False):
                    min_keep = max(0, f_idx - args.working_window)
                    video_predictor.prune_non_cond_memory(
                        state,
                        min_keep_frame_idx=min_keep,
                        obj_id=1,
                        keep_cond=True,
                    )

                    if len(seg_cache) >= args.longterm_segment_len:
                        before = len(longterm_bank)
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
                        if len(longterm_bank) > before:
                            promoted_segments += 1
                        seg_cache = []

                if args.enable_respawn and len(trust_seg_cache) >= args.segment_len:
                    force_trusted = current_untrusted_count >= args.max_untrusted_segments_per_lineage
                    new_seeds, decision = commit_segment(
                        seg_frames=trust_seg_cache,
                        covered_mask=covered_mask,
                        trusted_mask=trusted_mask,
                        track_axis=axis,
                        args=args,
                        force_trusted=force_trusted,
                    )
                    if decision is not None:
                        if decision.is_trusted:
                            trusted_segments += 1
                        else:
                            untrusted_segments += 1
                    if (decision is not None) and (not decision.is_trusted):
                        current_untrusted_count += 1
                    for child_seed in new_seeds:
                        task = SeedTask(seed=child_seed, is_original=False, untrusted_count=current_untrusted_count)
                        if task.seed in spawned_set:
                            continue
                        spawned_set.add(task.seed)
                        spawned_tasks.append(task)
                    trust_seg_cache = []

            if (not getattr(args, "disable_longterm_memory", False)) and len(seg_cache) > 0:
                before = len(longterm_bank)
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
                if len(longterm_bank) > before:
                    promoted_segments += 1
            if args.enable_respawn and len(trust_seg_cache) >= args.min_segment_frames:
                force_trusted = current_untrusted_count >= args.max_untrusted_segments_per_lineage
                new_seeds, decision = commit_segment(
                    seg_frames=trust_seg_cache,
                    covered_mask=covered_mask,
                    trusted_mask=trusted_mask,
                    track_axis=axis,
                    args=args,
                    force_trusted=force_trusted,
                )
                if decision is not None:
                    if decision.is_trusted:
                        trusted_segments += 1
                    else:
                        untrusted_segments += 1
                if (decision is not None) and (not decision.is_trusted):
                    current_untrusted_count += 1
                for child_seed in new_seeds:
                    task = SeedTask(seed=child_seed, is_original=False, untrusted_count=current_untrusted_count)
                    if task.seed in spawned_set:
                        continue
                    spawned_set.add(task.seed)
                    spawned_tasks.append(task)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return spawned_tasks, {
        "longterm_segments": promoted_segments,
        "trusted_segments": trusted_segments,
        "untrusted_segments": untrusted_segments,
        "final_untrusted_count": int(start_untrusted_count) if not args.enable_respawn else current_untrusted_count,
        "terminated_large_mask": terminated_large_mask,
    }
