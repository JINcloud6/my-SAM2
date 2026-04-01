import os
import sys



# --- Path Setup ---
# 确保可以引用父目录下的模块 (model, inference)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)



import time
import numpy as np
import torch
import h5py
from tqdm import tqdm
from scipy import ndimage

# --- Local Imports ---
from .config import get_args, xmem_config
from .utils import select_masks, compute_attention_entropy, EntropyStopper
from .preprocessing import get_multi_axis_init_seg, get_seeds_from_init_seg
from .data_manager import VolumeManager

# --- Parent Imports ---
# 假设父目录结构包含这些模块
try:
    from model.network import XMem
    from model.memory_util import get_similarity
    from inference.inference_core import InferenceCore
    from inference.kv_memory_store import KeyValueMemoryStore
    from segment_anything import sam_model_registry, SamPredictor
except ImportError as e:
    print("Error importing model/inference modules. Make sure the script is running with access to the parent directory.")
    raise e

def run_segmentation():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.makedirs(args.output_dir, exist_ok=True)

    enable_temporal_decay = xmem_config.get('enable_temporal_decay', True)
    enable_global_memory = xmem_config.get('enable_global_memory', True)
    enable_split_seeding = xmem_config.get('enable_split_seeding', False)
    enable_attention_entropy = xmem_config.get('enable_attention_entropy', False)
    enable_entropy_stop = xmem_config.get('enable_entropy_stop', False)
    global_mem_select_method = xmem_config.get('global_mem_select_method', 'all')
    global_mem_topk = xmem_config.get('global_mem_topk', 0)
    global_mem_debug = xmem_config.get('global_mem_debug', False)
    split_min_area = xmem_config.get('split_min_area', 200)
    split_min_distance = xmem_config.get('split_min_distance', 15)
    split_max_new_seeds = xmem_config.get('split_max_new_seeds', 2)
    entropy_sample_points = xmem_config.get('entropy_sample_points', 1024)
    entropy_window = xmem_config.get('entropy_window', 20)
    entropy_z_threshold = xmem_config.get('entropy_z_threshold', 2.0)
    entropy_abnormal_count = xmem_config.get('entropy_abnormal_count', 3)
    entropy_min_mask_pixels = xmem_config.get('entropy_min_mask_pixels', 20)
    entropy_boundary_points = xmem_config.get('entropy_boundary_points', 10)
    entropy_overlap_threshold = xmem_config.get('entropy_overlap_threshold', 0.5)
    entropy_debug = xmem_config.get('entropy_debug', False)
    if not enable_temporal_decay:
        xmem_config['temporal_decay'] = 0.0
    if enable_entropy_stop:
        enable_attention_entropy = True
    xmem_config['enable_attention_entropy'] = enable_attention_entropy
    
    # 1. Load Data
    print(f"Loading volume from {args.volume_path}...")
    vol_man = VolumeManager(args.volume_path, key=args.dataset_key, crop_size=args.crop_size, need_transpose=args.need_transpose)
    
    # 2. Get Seeds
    seeds = []
    if args.seed_file:
        print(f"Loading seeds from {args.seed_file}...")
        with open(args.seed_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [int(v) for v in line.replace(",", " ").split()]
                if len(parts) != 3:
                    raise ValueError(f"Invalid seed line: {line}")
                seeds.append(tuple(parts))
    else:
        # Get Init Seg (The "Map")
        init_seg_name = f"init_seg_axis{args.axis}_s{args.stride}_t{args.remove_portion}.h5"
        init_seg_path = os.path.join(args.output_dir, init_seg_name)
        
        if os.path.exists(init_seg_path):
            print(f"Loading existing init_seg from {init_seg_path}...")
            with h5py.File(init_seg_path, 'r') as f:
                init_seg = f['main'][:]
        else:
            init_seg = get_multi_axis_init_seg(
                vol_man.vol, 
                stride=args.stride, 
                thr=args.remove_portion,
                gaussian_kernel=args.gaussian_kernel,
                min_bright=args.min_bright
                # ... 其他参数
            )
            print(f"Saving init_seg to {init_seg_path}...")
            with h5py.File(init_seg_path, 'w') as f:
                f.create_dataset('main', data=init_seg, compression='gzip')
            # 生成合并了三个轴向的初始分割图
            

        # Extract Seeds
        seeds = get_seeds_from_init_seg(init_seg)
        if not seeds:
            print("No seeds found! Check parameters.")
            return

    # 4. Load Models
    print("Loading Models...")
    sam = sam_model_registry[args.sam_type](checkpoint=args.sam_checkpoint).to(args.device)
    sam_predictor = SamPredictor(sam)
    
    xmem = XMem(xmem_config, args.xmem_checkpoint).to(args.device).eval()
    if args.xmem_checkpoint:
        xmem.load_weights(torch.load(args.xmem_checkpoint), init_as_zero_if_needed=True)

    # 5. Iterative Segmentation Loop
    print(f"Starting segmentation with {len(seeds)} seeds...")
    
    processed_count = 0
    useonevos = args.useonevos
    if useonevos:
        print('use one vos')
        processor = InferenceCore(xmem, config=xmem_config)
        processor.set_all_labels([1])
    else:
        print('use different vos')
    global_memories = {}
    if enable_global_memory:
        global_memories = {
            0: KeyValueMemoryStore(count_usage=False),
            1: KeyValueMemoryStore(count_usage=False),
            2: KeyValueMemoryStore(count_usage=False),
        }
    global_mem_max_elements = xmem_config.get('global_mem_max_elements', 0)

    def _split_centers(binary_mask):
        labeled, num = ndimage.label(binary_mask > 0)
        if num < 2:
            return []
        counts = np.bincount(labeled.ravel())[1:]
        if counts.size == 0:
            return []
        order = np.argsort(counts)[::-1]
        centers = []
        for idx in order[:split_max_new_seeds]:
            if counts[idx] < split_min_area:
                continue
            label_id = idx + 1
            center = ndimage.center_of_mass(binary_mask, labeled, label_id)
            if not np.any(np.isnan(center)):
                centers.append((center[0], center[1], counts[idx]))
        if len(centers) < 2:
            return []
        return centers

    def _centers_far_enough(centers):
        if len(centers) < 2:
            return False
        (y1, x1, _), (y2, x2, _) = centers[0], centers[1]
        return np.hypot(y1 - y2, x1 - x2) >= split_min_distance

    def _sample_boundary_points(mask, num_points):
        if mask.sum() == 0:
            return np.empty((0, 2), dtype=np.int32)
        eroded = ndimage.binary_erosion(mask > 0)
        boundary = (mask > 0) & (~eroded)
        coords = np.argwhere(boundary)
        if coords.size == 0:
            coords = np.argwhere(mask > 0)
        if coords.size == 0:
            return np.empty((0, 2), dtype=np.int32)
        if coords.shape[0] > num_points:
            idx = np.random.choice(coords.shape[0], num_points, replace=False)
            coords = coords[idx]
        return coords

    def _resample_seeds_with_sam(mask, slice_image, axis, curr_idx, box):
        points = _sample_boundary_points(mask, entropy_boundary_points)
        if points.size == 0:
            return []
        img = np.stack([slice_image]*3, axis=-1)
        sam_predictor.set_image(img)
        new_seeds = []
        for py, px in points:
            masks, scores, _ = sam_predictor.predict(
                point_coords=np.array([[px, py]]),
                point_labels=np.array([1]),
                multimask_output=True
            )
            cand_mask, _ = select_masks(masks, scores, thr=0.9, crit='max', max_size=5000, min_circularity=0.6)
            if cand_mask is None:
                continue
            if axis == 0:
                existing = vol_man.global_mask[curr_idx, box[1]:box[2], box[3]:box[4]]
            elif axis == 1:
                existing = vol_man.global_mask[box[1]:box[2], curr_idx, box[3]:box[4]]
            else:
                existing = vol_man.global_mask[box[1]:box[2], box[3]:box[4], curr_idx]
            overlap = (cand_mask & (existing > 0)).sum()
            ratio = overlap / (cand_mask.sum() + 1e-6)
            if ratio > entropy_overlap_threshold:
                continue
            cy, cx = ndimage.center_of_mass(cand_mask)
            if np.isnan(cy) or np.isnan(cx):
                continue
            cy_i = int(round(cy))
            cx_i = int(round(cx))
            if axis == 0:
                new_seed = (curr_idx, box[1] + cy_i, box[3] + cx_i)
            elif axis == 1:
                new_seed = (box[1] + cy_i, curr_idx, box[3] + cx_i)
            else:
                new_seed = (box[1] + cy_i, box[3] + cx_i, curr_idx)
            nz, ny, nx = new_seed
            if vol_man.global_mask[nz, ny, nx] == 0:
                new_seeds.append(new_seed)
        return new_seeds

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
            components.append(
                {
                    "mask": comp_mask,
                    "center": (float(cy), float(cx)),
                    "area": area,
                }
            )
        return components

    def _recover_missing_components(prev_components, curr_mask, slice_image, axis, curr_idx, box):
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
        img = np.stack([slice_image] * 3, axis=-1)
        sam_predictor.set_image(img)
        recovered = np.zeros_like(curr_mask, dtype=np.uint8)
        if axis == 0:
            existing = vol_man.global_mask[curr_idx, box[1]:box[2], box[3]:box[4]]
        elif axis == 1:
            existing = vol_man.global_mask[box[1]:box[2], curr_idx, box[3]:box[4]]
        else:
            existing = vol_man.global_mask[box[1]:box[2], box[3]:box[4], curr_idx]
        for comp in missing:
            cy, cx = comp["center"]
            cy_i = int(round(cy))
            cx_i = int(round(cx))
            masks, scores, _ = sam_predictor.predict(
                point_coords=np.array([[cx_i, cy_i]]),
                point_labels=np.array([1]),
                multimask_output=True,
            )
            cand_mask, _ = select_masks(masks, scores, thr=0.0, crit='max', max_size=5000, min_circularity=0.0)
            if cand_mask is None:
                continue
            overlap = (cand_mask & (existing > 0)).sum()
            ratio = overlap / (cand_mask.sum() + 1e-6)
            if ratio > entropy_overlap_threshold:
                continue
            recovered = np.logical_or(recovered, cand_mask)
        if recovered.sum() == 0:
            return None
        return recovered.astype(np.uint8)

    def _slice_global_memory(global_mem, indices):
        key = global_mem.key.index_select(-1, indices)
        shrinkage = global_mem.shrinkage.index_select(-1, indices) if global_mem.shrinkage is not None else None
        selection = global_mem.selection.index_select(-1, indices) if global_mem.selection is not None else None
        timestamps = global_mem.time.index_select(-1, indices) if global_mem.time is not None else None
        value = global_mem.value[0].index_select(-1, indices)
        return key, value, shrinkage, selection, timestamps

    def _select_global_memory(global_mem, method, k, curr_idx=None, query_key=None):
        if global_mem is None or not global_mem.engaged():
            return None
        if method == 'all' or k <= 0 or global_mem.size <= k:
            if global_mem_debug:
                print(
                    f"[GlobalMemDebug] method={method} k={k} size={global_mem.size} "
                    f"curr_idx={curr_idx} -> select=all"
                )
            return global_mem.key, global_mem.value[0], global_mem.shrinkage, global_mem.selection, global_mem.time
        if method == 'nearest':
            if global_mem.time is None or curr_idx is None:
                if global_mem_debug:
                    print(
                        f"[GlobalMemDebug] method=nearest k={k} size={global_mem.size} "
                        f"curr_idx={curr_idx} time=None -> select=all"
                    )
                return global_mem.key, global_mem.value[0], global_mem.shrinkage, global_mem.selection, global_mem.time
            dist = (global_mem.time - float(curr_idx)).abs().view(-1)
            _, indices = torch.topk(dist, k=k, largest=False)
            if global_mem_debug:
                print(
                    f"[GlobalMemDebug] method=nearest k={k} size={global_mem.size} "
                    f"curr_idx={curr_idx} selected={indices[:10].tolist()}"
                )
            return _slice_global_memory(global_mem, indices)
        if method == 'similarity':
            if query_key is None:
                if global_mem_debug:
                    print(
                        f"[GlobalMemDebug] method=similarity k={k} size={global_mem.size} "
                        f"curr_idx={curr_idx} query=None -> select=all"
                    )
                return global_mem.key, global_mem.value[0], global_mem.shrinkage, global_mem.selection, global_mem.time
            query_key_flat = query_key.flatten(start_dim=2)
            similarity = get_similarity(global_mem.key, global_mem.shrinkage, query_key_flat, None)
            score = similarity.mean(dim=2).squeeze(0)
            _, indices = torch.topk(score, k=k, largest=True)
            if global_mem_debug:
                print(
                    f"[GlobalMemDebug] method=similarity k={k} size={global_mem.size} "
                    f"curr_idx={curr_idx} score_min={score.min().item():.4f} "
                    f"score_max={score.max().item():.4f} selected={indices[:10].tolist()}"
                )
            return _slice_global_memory(global_mem, indices)
        return global_mem.key, global_mem.value[0], global_mem.shrinkage, global_mem.selection, global_mem.time

    seed_index = 0
    with tqdm(total=len(seeds), desc="Tracking") as pbar:
        while seed_index < len(seeds):
            seed = seeds[seed_index]
            seed_index += 1
            pbar.update(1)
            z, y, x = seed
            
            # --- Check overlap (Critical for efficiency) ---
            if vol_man.global_mask[z, y, x] > 0:
                continue
            
            vol_man.global_mask[z, y, x] = 0 # Temporarily clear seed point
            
            # --- A. Tri-plane SAM Initialization ---
            crops = vol_man.get_triplane_crops(seed)
            best_axis = -1
            best_mask = None
            min_area = float('inf')

            for axis in [0, 1, 2]:
                img, box = crops[axis]
                # Map seed to local crop coords
                if axis == 0: local_pt = [seed[2]-box[3], seed[1]-box[1]] 
                elif axis == 1: local_pt = [seed[2]-box[3], seed[0]-box[1]]
                else: local_pt = [seed[1]-box[3], seed[0]-box[1]]
                
                sam_predictor.set_image(img)
                masks, scores, _ = sam_predictor.predict(
                    point_coords=np.array([local_pt]), 
                    point_labels=np.array([1]), 
                    multimask_output=True
                )
                
                mask, score = select_masks(masks, scores, thr=0.9, crit='max', max_size=5000,min_circularity=0.0)
                
                if mask is not None:
                    area = mask.sum()
                    if area < min_area:
                        min_area = area
                        best_axis = axis
                        best_mask = mask.astype(np.uint8)
            
            if best_axis == -1: continue

            # --- B. XMem Propagation ---
            track_dim = [0, 1, 2][best_axis]
            start_idx = seed[track_dim]
            max_dist = 2000
            sequences = [
                range(start_idx, min(start_idx + max_dist, vol_man.shape[track_dim])),
                range(start_idx, max(start_idx - max_dist, -1), -1)
            ]
            entropy_stopper = None
            if enable_entropy_stop:
                entropy_stopper = EntropyStopper(w=entropy_window, k=entropy_z_threshold, T=entropy_abnormal_count)
            
            if not useonevos:
                processor = InferenceCore(xmem, config=xmem_config)
                processor.set_all_labels([1])
            global_mem = global_memories.get(best_axis) if enable_global_memory else None
            _, box = crops[best_axis] 
            
            for seq in sequences:
                if not seq: continue
                first_frame = True
                global_added = False
                split_handled = False
                injected_global = False
                stop_triggered = False
                stop_info = None
                stop_state = None
                prev_components = None
                for curr_idx in seq:
                    # Dynamic slicing
                    if best_axis==0: sl = vol_man.vol[curr_idx, box[1]:box[2], box[3]:box[4]]
                    elif best_axis==1: sl = vol_man.vol[box[1]:box[2], curr_idx, box[3]:box[4]]
                    else: sl = vol_man.vol[box[1]:box[2], box[3]:box[4], curr_idx]
                    
                    if sl.size == 0: break
                    
                    rgb = torch.from_numpy(np.stack([sl]*3, -1)).permute(2,0,1).float().to(args.device)/255.0

                    if (not injected_global) and enable_global_memory and global_mem is not None and global_mem.engaged():
                        if first_frame:
                            with torch.no_grad():
                                key, shrinkage, _, _, _, _ = xmem.encode_key(
                                    rgb.unsqueeze(0),
                                    need_ek=False,
                                    need_sk=False,
                                )
                            selected = _select_global_memory(
                                global_mem,
                                global_mem_select_method,
                                global_mem_topk,
                                curr_idx=curr_idx,
                                query_key=key,
                            )
                            if selected is not None:
                                sel_key, sel_value, sel_shrinkage, sel_selection, sel_time = selected
                                processor.memory.work_mem.add(
                                    sel_key,
                                    sel_value,
                                    sel_shrinkage,
                                    sel_selection,
                                    objects=[1],
                                    timestamps=sel_time,
                                )
                            injected_global = True
                    
                    msk = None
                    if first_frame:
                        msk = torch.from_numpy(best_mask).long().to(args.device).unsqueeze(0)
                        first_frame = False
                    
                    with torch.no_grad():
                        prob = processor.step(rgb, msk, valid_labels=[1] if msk is not None else None)
                        pred = torch.argmax(prob, dim=0).cpu().numpy().astype(np.uint8)

                    if enable_entropy_stop and entropy_stopper is not None:
                        attn_weights, attn_hw = processor.memory.get_last_attention()
                        if attn_weights is not None:
                            if pred.sum() < entropy_min_mask_pixels:
                                stop_triggered = True
                                stop_info = ("mask_too_small", None, None)
                                stop_state = {
                                    "stop_slice_index": curr_idx,
                                    "stop_reason": "mask_too_small",
                                    "entropy_value": None,
                                    "z_value": None,
                                }
                            else:
                                entropy_value = compute_attention_entropy(
                                    attn_weights,
                                    pred,
                                    sample_points=entropy_sample_points,
                                    target_hw=attn_hw,
                                )
                                stop, z_value, abnormal_count = entropy_stopper.update(entropy_value)
                                if entropy_debug:
                                    print(
                                        f"[Entropy] slice={curr_idx} H={entropy_value} Z={z_value:.3f} "
                                        f"abnormal={abnormal_count}"
                                    )
                                if stop:
                                    stop_triggered = True
                                    stop_info = ("high_entropy", entropy_value, z_value)
                                    stop_state = {
                                        "stop_slice_index": curr_idx,
                                        "stop_reason": "high_entropy",
                                        "entropy_value": entropy_value,
                                    "z_value": z_value,
                                }

                    if not stop_triggered and prev_components:
                        recovered = _recover_missing_components(prev_components, pred, sl, best_axis, curr_idx, box)
                        if recovered is not None:
                            combined = ((pred > 0) | (recovered > 0)).astype(np.uint8)
                            with torch.no_grad():
                                msk_recover = torch.from_numpy(combined).long().to(args.device).unsqueeze(0)
                                prob = processor.step(rgb, msk_recover, valid_labels=[1])
                                pred = torch.argmax(prob, dim=0).cpu().numpy().astype(np.uint8)

                    if enable_global_memory and (msk is not None) and (not global_added):
                        latest = processor.memory.get_latest_work_memory()
                        if latest is not None:
                            key, shrinkage, value, selection, timestamps = latest
                            global_mem = global_memories.get(best_axis)
                            if global_mem is not None:
                                global_mem.add(
                                    key,
                                    value,
                                    shrinkage,
                                    selection,
                                    objects=[1],
                                    timestamps=timestamps,
                                )
                                if global_mem_max_elements > 0:
                                    global_mem.keep_last(global_mem_max_elements)
                        global_added = True

                    if stop_triggered:
                        if stop_info is not None:
                            reason, entropy_value, z_value = stop_info
                            if entropy_debug:
                                print(
                                    f"[EntropyStop] slice={curr_idx} reason={reason} "
                                    f"H={entropy_value} Z={z_value}"
                                )
                        if entropy_debug and stop_state is not None:
                            print(f"[EntropyStopState] {stop_state}")
                        if enable_entropy_stop:
                            new_seeds = _resample_seeds_with_sam(pred, sl, best_axis, curr_idx, box)
                            for new_seed in new_seeds:
                                seeds.append(new_seed)
                                pbar.total = len(seeds)
                        break

                    if enable_split_seeding and (not split_handled):
                        centers = _split_centers(pred)
                        if centers and _centers_far_enough(centers):
                            for cy, cx, _ in centers[:split_max_new_seeds]:
                                cy_i = int(round(cy))
                                cx_i = int(round(cx))
                                if best_axis == 0:
                                    new_seed = (curr_idx, box[1] + cy_i, box[3] + cx_i)
                                elif best_axis == 1:
                                    new_seed = (box[1] + cy_i, curr_idx, box[3] + cx_i)
                                else:
                                    new_seed = (box[1] + cy_i, box[3] + cx_i, curr_idx)
                                nz, ny, nx = new_seed
                                if vol_man.global_mask[nz, ny, nx] == 0:
                                    seeds.append(new_seed)
                                    pbar.total = len(seeds)
                            split_handled = True
                    
                    if pred.sum() > 5000: break 
                    
                    if pred.sum() > 0:
                        vol_man.update_global_mask(pred, best_axis, (curr_idx, *box[1:]))
                        prev_components = _extract_components(pred)
                    else:
                        break
                if stop_triggered:
                    break

            processed_count += 1
            if not useonevos:
                del processor

    print('Cleanup...')
    # vol_man.clean_up() # 可选，根据需要取消注释

    # 6. Save Final
    file_name = args.output_filename
    final_path = os.path.join(args.output_dir, file_name)
    need_transpose = args.need_transpose
    flag = True
    if need_transpose =='False':
        flag = False
    vol_man.save(final_path,flag)
    print(f"Done! Saved to {final_path}")

if __name__ == '__main__':
    run_segmentation()