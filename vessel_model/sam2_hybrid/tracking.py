from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from ..global_memory_pool import GlobalMemoryPool
from .axis_cache import init_state_from_axis_cache
from .common import SeedTask, sample_respawn_seeds_from_mask
from .segment_trust import SegmentFrame, commit_segment, propose_skeleton_respawn_seeds
from .unified_segment_classifier import (
    ClassifiedSegmentFrame,
    classify_segment,
    paint_segment_label_volume,
    update_reason_counts,
)


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


def segment_len_target_from_first_mask(mask: np.ndarray, args) -> int:
    multiplier = float(getattr(args, "segment_len_diameter_multiplier", 0.0))
    if multiplier <= 0.0:
        return int(args.segment_len)

    area = int(np.count_nonzero(mask))
    if area <= 0:
        return int(args.segment_len)

    radius = float(np.sqrt(float(area) / np.pi))
    diameter = 2.0 * radius
    return max(1, int(np.ceil(diameter * multiplier)))


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
    offload_video_to_cpu,
    global_memory_pool: GlobalMemoryPool,
    axis_sequence_cache,
    seed: Tuple[int, int, int],
    direction: str,
    covered_mask,
    trusted_mask,
    segment_label_mask,
    start_untrusted_count: int,
    args,
):
    empty_stats = {
        "longterm_segments": 0,
        "trusted_segments": 0,
        "untrusted_segments": 0,
        "final_untrusted_count": int(start_untrusted_count),
        "terminated_large_mask": False,
        "stable_segments": 0,
        "complex_segments": 0,
        "failure_segments": 0,
        "complex_reason_counts": {},
        "complex_reason_combo_counts": {},
    }
    if len(idx_list) == 0:
        return [], empty_stats

    valid_gidx = [int(vidx) for vidx in idx_list if 0 <= int(vidx) < int(axis_sequence_cache.num_frames)]
    if len(valid_gidx) == 0:
        return [], empty_stats
    valid_gidx_set = set(valid_gidx)

    spawned_tasks: List[SeedTask] = []
    spawned_set: Set[Tuple[int, int, int]] = set()
    promoted_segments = 0
    trusted_segments = 0
    untrusted_segments = 0
    terminated_large_mask = False
    stable_segments = 0
    complex_segments = 0
    failure_segments = 0
    complex_reason_counts: Dict[str, int] = {}
    complex_reason_combo_counts: Dict[str, int] = {}

    try:
        with torch.inference_mode(), torch.autocast(str(vol_man.device), dtype=torch.bfloat16):
            state = init_state_from_axis_cache(
                video_predictor=video_predictor,
                seq_cache=axis_sequence_cache,
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=False,
            )

            used = 0
            prev_nonempty_mask = None
            prev_global_idx = None
            trust_seg_cache: List[SegmentFrame] = []
            classified_seg_cache: List[ClassifiedSegmentFrame] = []
            current_segment_len_target: Optional[int] = None
            enable_segment_classification = (
                (segment_label_mask is not None)
                or getattr(args, "print_segment_classifier_details", False)
                or getattr(args, "enable_skeleton_respawn", False)
                or getattr(args, "disable_legacy_segment_trust_respawn", False)
            )

            def ensure_segment_len_target(mask: np.ndarray) -> int:
                nonlocal current_segment_len_target
                if current_segment_len_target is None:
                    current_segment_len_target = segment_len_target_from_first_mask(mask, args)
                return int(current_segment_len_target)

            def reset_segment_len_target() -> None:
                nonlocal current_segment_len_target
                current_segment_len_target = None

            def flush_classified_segment(empty_frame_count: int = 0):
                nonlocal classified_seg_cache, stable_segments, complex_segments, failure_segments
                return _flush_classified_segment(
                    empty_frame_count=empty_frame_count,
                    terminated_by_boundary=False,
                    termination_reason="normal",
                )

            def _flush_classified_segment(
                empty_frame_count: int = 0,
                terminated_by_boundary: bool = False,
                termination_reason: str = "normal",
            ):
                nonlocal classified_seg_cache, stable_segments, complex_segments, failure_segments
                if len(classified_seg_cache) == 0:
                    classified_seg_cache = []
                    return None
                seg_start = int(classified_seg_cache[0].global_frame_idx)
                seg_end = int(classified_seg_cache[-1].global_frame_idx)
                result = classify_segment(
                    seg_frames=classified_seg_cache,
                    track_axis=axis,
                    empty_frame_count=empty_frame_count,
                    terminated_by_boundary=terminated_by_boundary,
                    termination_reason=termination_reason,
                    args=args,
                )
                if segment_label_mask is not None:
                    paint_segment_label_volume(segment_label_mask, classified_seg_cache, result.label)
                if result.label == 1:
                    stable_segments += 1
                elif result.label == 2:
                    complex_segments += 1
                    update_reason_counts(
                        complex_reason_counts,
                        complex_reason_combo_counts,
                        result.stable_fail_reasons,
                    )
                else:
                    failure_segments += 1
                if getattr(args, "print_segment_classifier_details", False) and len(result.stable_fail_details) > 0:
                    print(
                        f"[SegmentClassifier] seed={seed} dir={direction} axis={axis} "
                        f"frames={seg_start}->{seg_end} category={result.category}"
                    )
                    for detail in result.stable_fail_details:
                        print(f"  - {detail}")
                classified_seg_cache = []
                return result

            for gidx in valid_gidx:
                m = init_masks_by_global_idx.get(gidx, None)
                if m is None or int(m.sum()) == 0:
                    continue
                ensure_segment_len_target(m.astype(np.uint8))
                fidx = int(gidx)
                video_predictor.add_new_mask(state, frame_idx=fidx, obj_id=1, mask=m.astype(bool))
                vol_man.update_global_mask(m.astype(np.uint8), axis, (gidx, *box[1:]))
                prev_nonempty_mask = m.astype(np.uint8)
                prev_global_idx = gidx
                if args.enable_respawn:
                    trust_seg_cache.append(
                        SegmentFrame(axis=axis, global_frame_idx=int(gidx), box=box, mask=m.astype(np.uint8))
                    )
                if enable_segment_classification:
                    classified_seg_cache.append(
                        ClassifiedSegmentFrame(
                            axis=axis,
                            global_frame_idx=int(gidx),
                            box=box,
                            mask=m.astype(np.uint8),
                            quality=None,
                        )
                    )
                used += 1

            if used == 0:
                g0 = valid_gidx[0]
                m0 = init_masks_by_global_idx.get(g0, None)
                if m0 is None or int(m0.sum()) == 0:
                    return [], empty_stats
                ensure_segment_len_target(m0.astype(np.uint8))
                video_predictor.add_new_mask(state, frame_idx=int(g0), obj_id=1, mask=m0.astype(bool))
                vol_man.update_global_mask(m0.astype(np.uint8), axis, (g0, *box[1:]))
                prev_nonempty_mask = m0.astype(np.uint8)
                prev_global_idx = g0
                if args.enable_respawn:
                    trust_seg_cache.append(
                        SegmentFrame(axis=axis, global_frame_idx=int(g0), box=box, mask=m0.astype(np.uint8))
                    )
                if enable_segment_classification:
                    classified_seg_cache.append(
                        ClassifiedSegmentFrame(
                            axis=axis,
                            global_frame_idx=int(g0),
                            box=box,
                            mask=m0.astype(np.uint8),
                            quality=None,
                        )
                    )

            if not getattr(args, "disable_longterm_memory", False):
                init_mask = prev_nonempty_mask
                init_radius = float(np.sqrt(float(np.count_nonzero(init_mask)) / np.pi)) if init_mask is not None else 0.0
                phys_to_local = {int(p): int(p) for p in valid_gidx}
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
            classification_result = None

            for f_idx, _obj_ids, masks in video_predictor.propagate_in_video(state):
                if int(f_idx) not in valid_gidx_set:
                    continue

                mm_logits = masks[0, 0]
                if torch.is_tensor(mm_logits):
                    mm = (mm_logits > 0).to(torch.uint8).cpu().numpy()
                else:
                    mm = (mm_logits > 0).astype(np.uint8)

                gidx = int(f_idx)
                if mm.sum() == 0:
                    classification_result = _flush_classified_segment(
                        empty_frame_count=1,
                        terminated_by_boundary=False,
                        termination_reason="empty_frame",
                    )
                    if args.enable_respawn and len(trust_seg_cache) >= args.min_segment_frames:
                        respawn_seed_override = None
                        if (
                            getattr(args, "enable_skeleton_respawn", False)
                            and classification_result is not None
                            and classification_result.category == "complex"
                        ):
                            respawn_seed_override = propose_skeleton_respawn_seeds(
                                seg_frames=trust_seg_cache,
                                trusted_mask=trusted_mask,
                                skeleton_respawn_offset=float(args.skeleton_respawn_offset),
                                max_skeleton_respawn_seeds=int(args.max_skeleton_respawn_seeds),
                                min_seed_distance=float(args.segment_respawn_min_distance),
                            )
                        force_trusted = current_untrusted_count >= args.max_untrusted_segments_per_lineage
                        new_seeds, decision = commit_segment(
                            seg_frames=trust_seg_cache,
                            covered_mask=covered_mask,
                            trusted_mask=trusted_mask,
                            track_axis=axis,
                            args=args,
                            force_trusted=force_trusted,
                            respawn_seed_override=respawn_seed_override,
                            classification_category=(
                                None if classification_result is None else classification_result.category
                            ),
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
                    reset_segment_len_target()
                    continue

                mask_area = int(mm.sum())
                if mask_area > int(args.max_slice_mask_area):
                    terminated_large_mask = True
                    _flush_classified_segment(
                        empty_frame_count=0,
                        terminated_by_boundary=False,
                        termination_reason="large_mask",
                    )
                    trust_seg_cache = []
                    seg_cache = []
                    reset_segment_len_target()
                    break

                # slice_area = float(mm.shape[0] * mm.shape[1])
                # if slice_area > 0:
                #     mask_ratio = float(mask_area) / slice_area
                #     if mask_ratio > float(args.max_slice_mask_ratio):
                #         terminated_large_mask = True
                #         trust_seg_cache = []
                #         seg_cache = []
                #         break

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
                current_target = ensure_segment_len_target(mm.astype(np.uint8))
                if args.enable_respawn:
                    trust_seg_cache.append(
                        SegmentFrame(axis=axis, global_frame_idx=int(gidx), box=box, mask=mm.astype(np.uint8))
                    )
                if enable_segment_classification:
                    classified_seg_cache.append(
                        ClassifiedSegmentFrame(
                            axis=axis,
                            global_frame_idx=int(gidx),
                            box=box,
                            mask=mm.astype(np.uint8),
                            quality=float(q),
                        )
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

                if enable_segment_classification and len(classified_seg_cache) >= current_target:
                    classification_result = _flush_classified_segment(
                        empty_frame_count=0,
                        terminated_by_boundary=False,
                        termination_reason="segment_len",
                    )
                    reset_segment_len_target()

                if args.enable_respawn and len(trust_seg_cache) >= current_target:
                    respawn_seed_override = None
                    if (
                        getattr(args, "enable_skeleton_respawn", False)
                        and enable_segment_classification
                        and classification_result is not None
                        and classification_result.category == "complex"
                    ):
                        respawn_seed_override = propose_skeleton_respawn_seeds(
                            seg_frames=trust_seg_cache,
                            trusted_mask=trusted_mask,
                            skeleton_respawn_offset=float(args.skeleton_respawn_offset),
                            max_skeleton_respawn_seeds=int(args.max_skeleton_respawn_seeds),
                            min_seed_distance=float(args.segment_respawn_min_distance),
                        )
                    force_trusted = current_untrusted_count >= args.max_untrusted_segments_per_lineage
                    new_seeds, decision = commit_segment(
                        seg_frames=trust_seg_cache,
                        covered_mask=covered_mask,
                        trusted_mask=trusted_mask,
                        track_axis=axis,
                        args=args,
                        force_trusted=force_trusted,
                        respawn_seed_override=respawn_seed_override,
                        classification_category=(
                            None if classification_result is None else classification_result.category
                        ),
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
                    reset_segment_len_target()

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
            reached_dataset_boundary = False
            if len(classified_seg_cache) > 0:
                last_gidx = int(classified_seg_cache[-1].global_frame_idx)
                if direction == "forward":
                    reached_dataset_boundary = last_gidx >= int(axis_sequence_cache.num_frames) - 1
                else:
                    reached_dataset_boundary = last_gidx <= 0
            classification_result = _flush_classified_segment(
                empty_frame_count=0,
                terminated_by_boundary=reached_dataset_boundary,
                termination_reason="dataset_boundary" if reached_dataset_boundary else "direction_end",
            )
            reset_segment_len_target()
            if args.enable_respawn and len(trust_seg_cache) >= args.min_segment_frames:
                respawn_seed_override = None
                if (
                    getattr(args, "enable_skeleton_respawn", False)
                    and classification_result is not None
                    and classification_result.category == "complex"
                ):
                    respawn_seed_override = propose_skeleton_respawn_seeds(
                        seg_frames=trust_seg_cache,
                        trusted_mask=trusted_mask,
                        skeleton_respawn_offset=float(args.skeleton_respawn_offset),
                        max_skeleton_respawn_seeds=int(args.max_skeleton_respawn_seeds),
                        min_seed_distance=float(args.segment_respawn_min_distance),
                    )
                force_trusted = current_untrusted_count >= args.max_untrusted_segments_per_lineage
                new_seeds, decision = commit_segment(
                    seg_frames=trust_seg_cache,
                    covered_mask=covered_mask,
                    trusted_mask=trusted_mask,
                    track_axis=axis,
                    args=args,
                    force_trusted=force_trusted,
                    respawn_seed_override=respawn_seed_override,
                    classification_category=(
                        None if classification_result is None else classification_result.category
                    ),
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
        pass

    return spawned_tasks, {
        "longterm_segments": promoted_segments,
        "trusted_segments": trusted_segments,
        "untrusted_segments": untrusted_segments,
        "final_untrusted_count": int(start_untrusted_count) if not args.enable_respawn else current_untrusted_count,
        "terminated_large_mask": terminated_large_mask,
        "stable_segments": stable_segments,
        "complex_segments": complex_segments,
        "failure_segments": failure_segments,
        "complex_reason_counts": complex_reason_counts,
        "complex_reason_combo_counts": complex_reason_combo_counts,
    }
