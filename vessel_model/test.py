import sam2
print(sam2.__path__)
# 或者打印 __file__
print(sam2.__file__)

import h5py

file_path = "/data2/jiangshuai/data/BvEM/mouse_microns-phase2_256-320nm_crop_bv_v4.h5"
file_path = "/data2/jiangshuai/data/BvEM/mouse_microns-phase2_256-320nm_crop_dshalf.h5"
file_path = "/data2/jiangshuai/data/BvEM/mouse_microns-phase2_256-320nm_crop_bv_v4_upsampled_aligned.h5"
file_path = "/data2/jiangshuai/data/BvEM/human_h01_256-264nm_crop.h5"
file_path = "/data2/jiangshuai/data/BvEM/human_h01_upsampled.h5"
h5_file = h5py.File(file_path, 'r')
data = h5_file['main'][:]
print(data.shape)

