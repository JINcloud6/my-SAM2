import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--xmem_checkpoint', default='/home/jiangshuai/code/XMem/saves/XMem.pth')
    parser.add_argument('--sam_checkpoint', default='/home/jiangshuai/code/XMem/saves/sam_vit_h_4b8939.pth')
    parser.add_argument('--sam_type', default='vit_h')
    parser.add_argument('--volume_path', required=True, help='Path to .h5 file')
    parser.add_argument('--output_dir', default='./bv_seg_output', help='Directory to save outputs')
    parser.add_argument('--dataset_key', default='main', help='Key name in h5 file')
    parser.add_argument('--gpu_id', default='0', help='指定使用的GPU ID')
    parser.add_argument('--output_filename', default='segmentation.nii.gz', help='输出文件名')
    parser.add_argument('--seed_file', default=None, help='Optional path to seed list (z,y,x per line)')

    # Seed Generation Params
    parser.add_argument('--axis', type=int, default=3, help='Axis to generate init seg (0=Z, 1=Y, 2=X),3 is use all')
    parser.add_argument('--stride', type=int, default=5, help='Stride for init seg generation')
    parser.add_argument('--gaussian_kernel', type=int, default=5) 
    parser.add_argument('--min_bright', type=int, default=40)
    parser.add_argument('--remove_portion', type=float, default=0.98)
    parser.add_argument('--crop_size', type=int, default=384)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--need_transpose', default='False')
    parser.add_argument('--useonevos',type=bool,default=False)

    return parser.parse_args()

# XMem Configuration
xmem_config = {
    'enable_long_term': True,
    'enable_long_term_count_usage': True,
    'max_mid_term_frames': 10,
    'min_mid_term_frames': 5,
    'max_long_term_elements': 10000,
    'num_prototypes': 128,
    'top_k': 30,
    'mem_every': 5,
    'deep_update_every': -1,
    'save_scores': False,
    'temporal_decay': 5.0,
    'temporal_decay_mode': 'exp',
    'global_mem_max_elements': 10000,
    'enable_temporal_decay': True,
    'enable_global_memory': False,
    'global_mem_select_method': 'all',
    'global_mem_topk': 200,
    'global_mem_debug': False,
    'drop_first_memory': False,
    'enable_split_seeding': False,
    'split_min_area': 200,
    'split_min_distance': 15,
    'split_max_new_seeds': 2,
    'enable_attention_entropy': False,
    'enable_entropy_stop': False,
    'entropy_sample_points': 1024,
    'entropy_window': 20,
    'entropy_z_threshold': 2.0,
    'entropy_abnormal_count': 3,
    'entropy_min_mask_pixels': 20,
    'entropy_boundary_points': 10,
    'entropy_overlap_threshold': 0.5,
    'entropy_debug': False,
}
