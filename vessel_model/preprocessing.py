# import numpy as np
# import cv2
# import skimage.morphology
# import skimage.measure
# from tqdm import tqdm

# def get_seg(img, remove_portion=0.98, mask=None, min_size=5, gaussian_kernel=21, min_bright=50):
#     if mask is not None:
#         img[mask > 0] = 0
#     processed_img = img.copy()
    
#     # remove noise in the image
#     processed_img = cv2.fastNlMeansDenoising(processed_img, None, gaussian_kernel)
#     _max = processed_img.max()
#     threshold = _max * remove_portion
#     threshold = max(threshold, min_bright)

#     # --- DEBUG START: 检查为什么像素被杀光了 ---
#     if processed_img.max() > 0 and (processed_img < threshold).all():
#         if not hasattr(get_seg, "has_printed"): 
#             print(f"\n[DEBUG ALERT] Slice Filtered Out Completely!")
#             print(f"  Image Max: {_max}")
#             print(f"  Calculated Threshold: {threshold} (min_bright was {min_bright})")
#             print(f"  Result: All pixels < threshold. Adjust 'min_bright' or 'remove_portion'.\n")
#             get_seg.has_printed = True
#     # --- DEBUG END ---

#     processed_img[processed_img < threshold] = 0
#     # remove small regions
#     processed_img = skimage.morphology.area_opening(processed_img, min_size, connectivity=1)
#     processed_img[processed_img > 0] = 1

#     seg = processed_img
#     seg = skimage.measure.label(seg)

#     component_sizes = np.bincount(seg.ravel())
#     # 防止越界，确保 component_sizes 至少有 label max + 1 长度
#     if len(component_sizes) > 1:
#         too_large = component_sizes > 5000
#         # 只有当 seg 中的值在 too_large 索引范围内时才进行掩码操作
#         # (通常 measure.label 保证了 max label < len(bincount))
#         too_large_mask = too_large[seg]
#         seg[too_large_mask] = 0    

#     return seg

# def get_init_seg(data, axis=0, stride=2, thr=0.98, min_size=10, gaussian_kernel=21, min_bright=50):
#     """
#     生成整个 3D 体积的初始分割建议 (Initial Segmentation Proposal)
#     """
#     print(f"Generating Init Seg along axis {axis} with stride {stride}...")
#     d, h, w = data.shape
#     init_seg = np.zeros((d, h, w), dtype=np.uint8)
    
#     total_points_found = 0 # 计数器

#     # 根据轴向遍历
#     if axis == 0:
#         for m in tqdm(range(0, d, stride), desc="Init Seg (Z)"):
#             image = data[m, :, :]
#             temp_seg = get_seg(image, remove_portion=thr, min_size=min_size, gaussian_kernel=gaussian_kernel, min_bright=min_bright)
#             nz = np.count_nonzero(temp_seg)
#             total_points_found += nz
#             init_seg[m, :, :] = temp_seg
#     elif axis == 1:
#         for m in tqdm(range(0, h, stride), desc="Init Seg (Y)"):
#             image = data[:, m, :]
#             temp_seg = get_seg(image, remove_portion=thr, min_size=min_size, gaussian_kernel=gaussian_kernel, min_bright=min_bright)
#             nz = np.count_nonzero(temp_seg)
#             total_points_found += nz
#             init_seg[:, m, :] = temp_seg
#     else: # axis == 2
#         for m in tqdm(range(0, w, stride), desc="Init Seg (X)"):
#             image = data[:, :, m]
#             temp_seg = get_seg(image, remove_portion=thr, min_size=min_size, gaussian_kernel=gaussian_kernel, min_bright=min_bright)
#             nz = np.count_nonzero(temp_seg)
#             total_points_found += nz
#             init_seg[:, :, m] = temp_seg
    
#     print(f"\n[DEBUG] Total non-zero pixels in init_seg: {total_points_found}")
#     return init_seg

# def get_seeds_from_init_seg(init_seg_vol):
#     """
#     从 init_seg 中提取种子点。
#     """
#     print("Extracting seeds from Init Seg...")
#     seeds = []
    
#     z_ind, y_ind, x_ind = np.where(init_seg_vol > 0)
    
#     if len(z_ind) == 0:
#         return []

#     coords = np.stack([z_ind, y_ind, x_ind], axis=1) # (N, 3)
    
#     try:
#         labeled = skimage.measure.label(init_seg_vol)
#         props = skimage.measure.regionprops(labeled)
#         seeds = [tuple(map(int, p.centroid)) for p in props]
#         print(f"Extracted {len(seeds)} component centroids as seeds.")
#     except Exception as e:
#         print(f"3D Labeling failed (RAM issue?), falling back to subsampling: {e}")
#         step = 10 
#         seeds = coords[::step].tolist()
#         seeds = [tuple(p) for p in seeds]
        
#     return seeds

import numpy as np
import cv2
import skimage.morphology
import skimage.measure
from tqdm import tqdm
from scipy.spatial import cKDTree  # [新增] 用于快速距离计算

def get_seg(img, remove_portion=0.98, mask=None, min_size=5, gaussian_kernel=21, min_bright=50):
    # [保持原样，无需修改] 
    # 该函数负责处理单张切片的分割
    if mask is not None:
        img[mask > 0] = 0
    processed_img = img.copy()
    
    # remove noise in the image
    processed_img = cv2.fastNlMeansDenoising(processed_img, None, gaussian_kernel)
    _max = processed_img.max()
    threshold = _max * remove_portion
    threshold = max(threshold, min_bright)

    # --- DEBUG START ---
    if processed_img.max() > 0 and (processed_img < threshold).all():
        if not hasattr(get_seg, "has_printed"): 
            print(f"\n[DEBUG ALERT] Slice Filtered Out Completely!")
            print(f"  Image Max: {_max}")
            print(f"  Calculated Threshold: {threshold} (min_bright was {min_bright})")
            get_seg.has_printed = True
    # --- DEBUG END ---

    processed_img[processed_img < threshold] = 0
    # remove small regions
    processed_img = skimage.morphology.area_opening(processed_img, min_size, connectivity=1)
    processed_img[processed_img > 0] = 1

    seg = processed_img
    seg = skimage.measure.label(seg)

    component_sizes = np.bincount(seg.ravel())
    if len(component_sizes) > 1:
        too_large = component_sizes > 5000
        too_large_mask = too_large[seg]
        seg[too_large_mask] = 0    

    return seg

def get_multi_axis_init_seg(data, stride=5, thr=0.98, min_size=10, gaussian_kernel=21, min_bright=50):
    """
    [修改] 从三个轴向(Z, Y, X)分别进行切片分割，并将结果合并。
    这样可以最大程度捕捉不同走向的血管。
    """
    print(f"Generating Multi-Axis Init Seg (Stride={stride})...")
    d, h, w = data.shape
    # 创建一个全局的 3D Mask
    init_seg = np.zeros((d, h, w), dtype=np.uint8)
    
    total_points_found = 0

    # --- Axis 0 (Z-axis) ---
    for m in tqdm(range(0, d, stride), desc="Axis 0 (Z)"):
        image = data[m, :, :]
        temp_seg = get_seg(image, remove_portion=thr, min_size=min_size, gaussian_kernel=gaussian_kernel, min_bright=min_bright)
        # 使用逻辑或运算 (OR) 将结果合并进主 mask
        # 注意：temp_seg > 0 转为布尔值，然后赋值
        init_seg[m, :, :][temp_seg > 0] = 1

    # --- Axis 1 (Y-axis) ---
    for m in tqdm(range(0, h, stride), desc="Axis 1 (Y)"):
        image = data[:, m, :]
        temp_seg = get_seg(image, remove_portion=thr, min_size=min_size, gaussian_kernel=gaussian_kernel, min_bright=min_bright)
        init_seg[:, m, :][temp_seg > 0] = 1

    # --- Axis 2 (X-axis) ---
    for m in tqdm(range(0, w, stride), desc="Axis 2 (X)"):
        image = data[:, :, m]
        temp_seg = get_seg(image, remove_portion=thr, min_size=min_size, gaussian_kernel=gaussian_kernel, min_bright=min_bright)
        init_seg[:, :, m][temp_seg > 0] = 1
    
    total_points = np.count_nonzero(init_seg)
    print(f"\n[DEBUG] Total non-zero pixels in merged init_seg: {total_points}")
    return init_seg

def filter_seeds_spatial(seeds, min_dist=15.0):
    """
    [新增] 使用 KD-Tree 对种子点进行距离过滤。
    如果多个种子点之间的距离小于 min_dist，则只保留其中一个。
    """
    if not seeds:
        return []
    
    print(f"Filtering {len(seeds)} seeds with min_dist={min_dist}...")
    
    # 转换为 numpy array 以便处理
    points = np.array(seeds)
    
    # 构建 KDTree
    tree = cKDTree(points)
    
    # 贪心策略：标记需要保留的点
    keep_mask = np.ones(len(points), dtype=bool)
    
    # 获取所有距离小于 min_dist 的点对
    # query_pairs 返回的是 (i, j) 且 i < j 的集合
    pairs = tree.query_pairs(r=min_dist)
    
    for i, j in pairs:
        # 如果 i 和 j 都还标记为保留，则删掉 j (保留索引小的)
        if keep_mask[i] and keep_mask[j]:
            keep_mask[j] = False
            
    filtered_seeds = points[keep_mask].tolist()
    # 转回 tuple 列表
    filtered_seeds = [tuple(p) for p in filtered_seeds]
    
    print(f"  -> {len(filtered_seeds)} seeds remaining.")
    return filtered_seeds

def get_seeds_from_init_seg(init_seg_vol):
    """
    从 init_seg 中提取种子点，并包含 3D 连通域分析。
    """
    print("Extracting seeds from Init Seg...")
    seeds = []
    
    # 1. 先进行 3D 连通域标记
    # 这一步非常重要：因为我们合并了三个轴向的结果，
    # 只有通过 3D label 才能把同一个血管在不同轴向产生的碎片整合成一个对象
    try:
        labeled = skimage.measure.label(init_seg_vol, connectivity=2) # 26-connectivity
        props = skimage.measure.regionprops(labeled)
        
        # 获取每个连通域的质心
        seeds = [tuple(map(int, p.centroid)) for p in props]
        print(f"Extracted {len(seeds)} component centroids (before distance filter).")
        
    except Exception as e:
        print(f"3D Labeling failed (RAM issue?): {e}. Falling back to raw coordinates.")
        # Fallback: 直接取点，然后通过后续的 distance filter 降采样
        z_ind, y_ind, x_ind = np.where(init_seg_vol > 0)
        if len(z_ind) == 0: return []
        coords = np.stack([z_ind, y_ind, x_ind], axis=1)
        # 简单降采样防止点太多
        seeds = coords[::5].tolist()
        seeds = [tuple(p) for p in seeds]

    # 2. 调用距离过滤器
    # 连通域中心可能依然靠的很近（例如断裂的血管），需要合并
    final_seeds = filter_seeds_spatial(seeds, min_dist=5.0)
        
    return final_seeds