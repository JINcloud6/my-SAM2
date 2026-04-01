import torch
import gc

def print_gpu_memory(stage_name=""):
    """打印当前显存占用情况及主要张量"""
    if not torch.cuda.is_available():
        return
        
    torch.cuda.synchronize() # 确保同步
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    print(f"\n--- GPU Memory at {stage_name} ---")
    print(f"Allocated: {allocated:.2f} MB")
    print(f"Reserved: {reserved:.2f} MB")

    # 打印显存中占用前 5 的张量
    tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if obj.is_cuda:
                    tensors.append(obj)
        except:
            pass
    
    tensors.sort(key=lambda x: x.element_size() * x.nelement(), reverse=True)
    print("Top 5 Tensors in GPU:")
    for i, t in enumerate(tensors[:5]):
        size_mb = (t.element_size() * t.nelement()) / (1024**2)
        print(f"  {i+1}: Shape {list(t.shape)} | {size_mb:.2f} MB | {t.dtype}")
    print("---------------------------------\n")

# def select_masks(masks, scores, thr=0.85, crit='min', max_size=1000):
#     # 你的筛选逻辑：优选面积小的（血管截面）
#     if len(masks) == 0: return None, -1.0
#     valid = []
#     for i, (m, s) in enumerate(zip(masks, scores)):
#         area = m.sum()
#         if s > thr and area > 10 and area < max_size:
#             valid.append((i, area, s))
#     if not valid: return None, -1.0
    
#     if crit == 'min': valid.sort(key=lambda x: x[1])
#     else: valid.sort(key=lambda x: x[2], reverse=True)
    
#     idx = valid[0][0]
#     return masks[idx], scores[idx]

import numpy as np
from skimage.measure import perimeter
from collections import deque

def compute_attention_entropy(attn_weights, mask, sample_points=1024, eps=1e-6, target_hw=None):
    """
    Compute attention entropy within mask region.
    attn_weights: Tensor [1, N, HW] or [N, H, W]
    mask: numpy or torch array [H, W] (binary)
    """
    if attn_weights is None:
        return None
    if isinstance(mask, np.ndarray):
        mask_tensor = torch.from_numpy(mask)
    else:
        mask_tensor = mask
    if target_hw is not None:
        target_h, target_w = target_hw
        h, w = mask_tensor.shape[-2:]
        if h < target_h or w < target_w:
            pad_h = target_h - h
            pad_w = target_w - w
            mask_tensor = torch.nn.functional.pad(mask_tensor, (0, pad_w, 0, pad_h))
        if mask_tensor.shape[-2:] != (target_h, target_w):
            mask_tensor = mask_tensor[:target_h, :target_w]
    mask_tensor = mask_tensor.to(attn_weights.device)
    if mask_tensor.numel() == 0:
        return None
    mask_tensor = (mask_tensor > 0).flatten()
    if mask_tensor.sum() == 0:
        return None

    if attn_weights.dim() == 3:
        attn_flat = attn_weights
    else:
        attn_flat = attn_weights.view(attn_weights.shape[0], -1).unsqueeze(0)

    valid_indices = torch.nonzero(mask_tensor, as_tuple=False).squeeze(1)
    if valid_indices.numel() == 0:
        return None
    if valid_indices.numel() > sample_points:
        perm = torch.randperm(valid_indices.numel(), device=valid_indices.device)[:sample_points]
        valid_indices = valid_indices[perm]

    weights = attn_flat[0, :, valid_indices].transpose(0, 1)
    weights = torch.clamp(weights, min=eps)
    entropy = -(weights * torch.log(weights)).sum(dim=1)
    return entropy.mean().item()


class EntropyStopper:
    def __init__(self, w=20, k=2.0, T=3):
        self.window = w
        self.z_threshold = k
        self.abnormal_target = T
        self.history = deque(maxlen=w)
        self.abnormal_count = 0

    def update(self, entropy_value):
        if entropy_value is None:
            return False, 0.0, self.abnormal_count
        if len(self.history) < self.window:
            self.history.append(entropy_value)
            return False, 0.0, self.abnormal_count
        history_arr = torch.tensor(list(self.history), dtype=torch.float32)
        mu = history_arr.mean().item()
        sigma = history_arr.std(unbiased=False).item()
        z = (entropy_value - mu) / (sigma + 1e-6)
        if z > self.z_threshold:
            self.abnormal_count += 1
        else:
            self.abnormal_count = 0
        self.history.append(entropy_value)
        stop = self.abnormal_count >= self.abnormal_target
        return stop, z, self.abnormal_count

def compute_circularity(mask):
    area = mask.sum()
    if area == 0:
        return 0.0
    perim = perimeter(mask, neighborhood=8)
    if perim == 0:
        return 0.0
    return 4 * np.pi * area / (perim ** 2)


def select_masks(
    masks,
    scores,
    thr=0.85,
    crit='min',
    max_size=1000,
    min_circularity=0.6
):
    """
    Select vessel-like masks based on confidence, size, and circularity.
    """
    if len(masks) == 0:
        return None, -1.0

    valid = []
    for i, (m, s) in enumerate(zip(masks, scores)):
        area = m.sum()
        if s <= thr or area <= 10 or area >= max_size:
            continue

        circ = compute_circularity(m)
        if circ < min_circularity:
            continue

        valid.append((i, area, s, circ))

    if not valid:
        return None, -1.0

    # 优先选择“更像血管截面”的 mask
    if crit == 'min':
        # 面积小 + 圆度高
        valid.sort(key=lambda x: (x[1], -x[3]))
    else:
        # score 高 + 圆度高
        valid.sort(key=lambda x: (x[2], x[3]), reverse=True)

    idx = valid[0][0]
    return masks[idx], scores[idx]
