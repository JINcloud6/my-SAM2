import h5py
import nibabel as nib
import numpy as np
from scipy import ndimage
import cc3d

class VolumeManager:
    def __init__(self, path, key='main', crop_size=384,need_transpose=False):
        # 1. 根据后缀名判断读取方式
        if path.endswith('.nii.gz') or path.endswith('.nii'):
            # 读取 NIfTI 文件
            img = nib.load(path)
            self.vol = img.get_fdata()
            self.h5_file = None  # Nii 文件不需要保持文件句柄
            self.affine = img.affine # 保存仿射矩阵
        else:
            # 读取 H5 文件
            self.h5_file = h5py.File(path, 'r')
            if key not in self.h5_file.keys(): 
                key = list(self.h5_file.keys())[0]
            self.vol = self.h5_file[key][:] # Load to RAM
            self.affine = None
        
        # 2. 归一化处理
        if self.vol.max() > 255:
            p99 = np.percentile(self.vol, 99)
            self.vol = np.clip(self.vol, 0, p99)
            self.vol = ((self.vol / p99) * 255).astype(np.uint8)
        else:
            self.vol = self.vol.astype(np.uint8)
            
        self.shape = self.vol.shape
        self.crop_size = crop_size
        self.global_mask = np.zeros_like(self.vol, dtype=np.uint8)
        self.need_transpose = need_transpose
        print("self.need_transpose:",self.need_transpose)

    def __del__(self):
        # 善后处理：如果是 H5 文件，确保关闭句柄
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()

    # def get_triplane_crops(self, center):
    #     cz, cy, cx = center
    #     half = self.crop_size // 2
    #     crops = {}
    #     # Helper to clamp
    #     clamp = lambda x, mx: (max(0, x-half), min(mx, x+half))
        
    #     z_r = clamp(cz, self.shape[0])
    #     y_r = clamp(cy, self.shape[1])
    #     x_r = clamp(cx, self.shape[2])
        
    #     # Plane 0 (Z-slice)
    #     crops[0] = (self._rgb(self.vol[cz, y_r[0]:y_r[1], x_r[0]:x_r[1]]), (cz, *y_r, *x_r))
    #     # Plane 1 (Y-slice)
    #     crops[1] = (self._rgb(self.vol[z_r[0]:z_r[1], cy, x_r[0]:x_r[1]]), (cy, *z_r, *x_r))
    #     # Plane 2 (X-slice)
    #     crops[2] = (self._rgb(self.vol[z_r[0]:z_r[1], y_r[0]:y_r[1], cx]), (cx, *z_r, *y_r))
    #     return crops

    def get_triplane_crops(self, center):
        """
        不再进行裁剪，而是返回三个轴向的完整切片。
        Box 格式: (slice_idx, dim1_min, dim1_max, dim2_min, dim2_max)
        """
        cz, cy, cx = center
        d, h, w = self.shape # 获取整个体积的尺寸 (Depth, Height, Width)
        crops = {}
        
        # --- Plane 0 (Z-slice): 轴状面 (Axial) ---
        # 图像尺寸: [H, W]
        # Box: (z_index, y_min, y_max, x_min, x_max) -> (cz, 0, h, 0, w)
        img_z = self.vol[cz, :, :]
        crops[0] = (self._rgb(img_z), (cz, 0, h, 0, w))
        
        # --- Plane 1 (Y-slice): 冠状面 (Coronal) ---
        # 图像尺寸: [D, W]
        # Box: (y_index, z_min, z_max, x_min, x_max) -> (cy, 0, d, 0, w)
        img_y = self.vol[:, cy, :]
        crops[1] = (self._rgb(img_y), (cy, 0, d, 0, w))
        
        # --- Plane 2 (X-slice): 矢状面 (Sagittal) ---
        # 图像尺寸: [D, H]
        # Box: (x_index, z_min, z_max, y_min, y_max) -> (cx, 0, d, 0, h)
        img_x = self.vol[:, :, cx]
        crops[2] = (self._rgb(img_x), (cx, 0, d, 0, h))
        
        return crops
    
    def _rgb(self, x): return np.stack([x]*3, axis=-1)

    def update_global_mask(self, local_mask, axis, box):
        # Merge logic
        idx, d1_min, d1_max, d2_min, d2_max = box
        # 确保 local_mask 尺寸匹配（防止边界裁剪问题）
        local_mask = local_mask[:(d1_max-d1_min), :(d2_max-d2_min)]
        
        if axis == 0: self.global_mask[idx, d1_min:d1_max, d2_min:d2_max] |= local_mask
        elif axis == 1: self.global_mask[d1_min:d1_max, idx, d2_min:d2_max] |= local_mask
        elif axis == 2: self.global_mask[d1_min:d1_max, d2_min:d2_max, idx] |= local_mask

    def save(self, path,need_transpose):
        # 1. 处理文件后缀，确保是 .nii.gz
        if path.endswith('.h5'):
            path = path.replace('.h5', '.nii.gz')
        elif not path.endswith('.nii.gz'):
            path += '.nii.gz'

        # 2. 获取仿射矩阵 (Affine Matrix)
        if self.affine is not None:
            affine = self.affine
        else:
            affine = np.eye(4) 

        # 3. 创建 NIfTI 对象
        img = self.global_mask.astype(np.uint8)
        print("before transpose:",img.shape)
        if need_transpose:
            print("transposing...")
            img = np.transpose(img, (2, 1, 0))
        print("after transpose",img.shape)
        nii_img = nib.Nifti1Image(img, affine)

        # 4. 保存文件
        nib.save(nii_img, path)
    
    def clean_up(self):
        print("Cleaning up result...")
        for m in range(self.global_mask.shape[0]):
            self.global_mask[m] = ndimage.binary_fill_holes(self.global_mask[m])
        
        labels_out = cc3d.connected_components(self.global_mask, connectivity=6)
        stats = cc3d.statistics(labels_out)
                
        # Remove connected components smaller than threshold
        voxel_counts = stats["voxel_counts"]
        for label_id, count in enumerate(voxel_counts):
            if label_id > 0 and count < 80000:  # Skip background (label 0)
                self.global_mask[labels_out == label_id] = 0