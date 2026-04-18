from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


EPS = 1e-6


@dataclass
class MemoryFeature:
    area: float
    radius: float
    quality: float


@dataclass
class GlobalMemoryEntry:
    axis: int
    physical_frame_idx: int
    mask: np.ndarray
    feature: MemoryFeature
    source_seed: Tuple[int, int, int]
    source_direction: str


class GlobalMemoryPool:
    """
    Global long-term memory shared across seeds.
    Constraints:
      1) only shared within same axis;
      2) only injected when physical frame index can be aligned exactly.
    """

    def __init__(self, max_entries_per_axis: int = 512):
        self.max_entries_per_axis = max_entries_per_axis
        self._entries: Dict[int, List[GlobalMemoryEntry]] = {0: [], 1: [], 2: []}

    @staticmethod
    def compute_feature(mask: np.ndarray, quality: float) -> MemoryFeature:
        area = float(np.count_nonzero(mask))
        radius = float(np.sqrt(area / np.pi)) if area > 0 else 0.0
        return MemoryFeature(area=area, radius=radius, quality=float(quality))

    def add_entry(
        self,
        axis: int,
        physical_frame_idx: int,
        mask: np.ndarray,
        quality: float,
        source_seed: Tuple[int, int, int],
        source_direction: str,
    ) -> None:
        if axis not in self._entries:
            return
        if mask is None or mask.size == 0 or int(mask.sum()) == 0:
            return

        new_feature = self.compute_feature(mask, quality)
        candidates = self._entries[axis]

        # Deduplicate by physical frame index: keep higher-quality one.
        replaced = False
        for i, old in enumerate(candidates):
            if old.physical_frame_idx == physical_frame_idx:
                if new_feature.quality > old.feature.quality:
                    candidates[i] = GlobalMemoryEntry(
                        axis=axis,
                        physical_frame_idx=physical_frame_idx,
                        mask=mask.astype(np.uint8),
                        feature=new_feature,
                        source_seed=source_seed,
                        source_direction=source_direction,
                    )
                replaced = True
                break
        if not replaced:
            candidates.append(
                GlobalMemoryEntry(
                    axis=axis,
                    physical_frame_idx=physical_frame_idx,
                    mask=mask.astype(np.uint8),
                    feature=new_feature,
                    source_seed=source_seed,
                    source_direction=source_direction,
                )
            )

        if len(candidates) > self.max_entries_per_axis:
            candidates.sort(key=lambda x: x.feature.quality, reverse=True)
            del candidates[self.max_entries_per_axis :]

    def select_for_injection(
        self,
        axis: int,
        current_seed: Tuple[int, int, int],
        current_radius: float,
        physical_to_local: Dict[int, int],
        max_inject: int,
    ) -> List[Tuple[int, GlobalMemoryEntry]]:
        if axis not in self._entries:
            return []
        if max_inject <= 0:
            return []

        scored: List[Tuple[float, int, GlobalMemoryEntry]] = []
        for entry in self._entries[axis]:
            local_idx = physical_to_local.get(entry.physical_frame_idx, None)
            if local_idx is None:
                continue
            if local_idx == 0:
                continue  # frame 0 already has current seed's init mask

            radius_gap = abs(entry.feature.radius - current_radius)
            axis_frame_center = current_seed[axis]
            frame_gap = abs(entry.physical_frame_idx - axis_frame_center)
            score = entry.feature.quality - 0.02 * radius_gap - 0.001 * frame_gap
            scored.append((score, local_idx, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [(local_idx, entry) for _, local_idx, entry in scored[:max_inject]]
