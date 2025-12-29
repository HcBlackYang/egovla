# # utils/cache_latents.py
# import sys
# import os
# import torch
# import h5py
# import numpy as np
# import argparse
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from torch.amp import autocast

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from model.fusion_encoder import FusionEncoder
# from utils.dataset_loader import RobotDataset

# # 路径配置
# VIDEO_MAE_PATH = '/yanghaochuan/models/VideoMAEv2-Large'

# def cache_data(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"=== Caching Latents using {device} ===")

#     # 1. 加载 Encoder (Stage B 权重)
#     print("Loading FusionEncoder...")
#     encoder = FusionEncoder(backbone_path=VIDEO_MAE_PATH, teacher_dim=1152).to(device)
#     if os.path.exists(args.stage_b_ckpt):
#         print(f"Loading Stage B Weights: {args.stage_b_ckpt}")
#         encoder.load_state_dict(torch.load(args.stage_b_ckpt), strict=False)
#     else:
#         raise FileNotFoundError(f"Stage B checkpoint not found: {args.stage_b_ckpt}")
    
#     encoder.eval() # 绝对冻结

#     # 2. 加载数据 (使用你之前修正过的带 stats 的 Dataset)
#     dataset = RobotDataset(hdf5_path=args.data_root, window_size=16, 
#                            stats_path='/yanghaochuan/projects/data/dataset_stats.json')
    
#     # 推理时 Batch Size 可以大一点，因为不需要存梯度
#     loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=16, pin_memory=True)

#     # 3. 创建输出 HDF5
#     print(f"Creating cache file: {args.output_path}")
#     with h5py.File(args.output_path, 'w') as f_out:
#         # 预估总数
#         total_len = len(dataset)
        
#         # 创建数据集
#         # e_t 维度是 [768]
#         ds_et = f_out.create_dataset('e_t', shape=(total_len, 768), dtype='float32')
#         # actions 维度是 [8]
#         ds_act = f_out.create_dataset('action_target', shape=(total_len, 8), dtype='float32')
        
#         ptr = 0
#         with torch.no_grad():
#             for batch in tqdm(loader, desc="Caching"):
#                 # 搬运数据
#                 video = batch['video'].to(device)
#                 state = batch['state'].to(device)
#                 text = batch['text_tokens'].to(device)
#                 ff = batch['first_frame'].to(device)
                
#                 # 目标动作 (已经在 Dataset 里归一化过了，直接存)
#                 actions = batch['action_target'].numpy()
                
#                 # # 跑 Encoder
#                 # outputs = encoder(video, text, state, ff)
#                 # e_t = outputs['e_t'].cpu().numpy() # [B, 768]
                
#                 # === 修改点：加上 autocast ===
#                 with autocast('cuda', dtype=torch.bfloat16):  # <--- 加上这一行
#                     outputs = encoder(video, text, state, ff)
#                     e_t = outputs['e_t'].float().cpu().numpy() # 确保转回 float32 存盘
#                 # ===========================

#                 # 写入 HDF5
#                 B = e_t.shape[0]
#                 ds_et[ptr : ptr+B] = e_t
#                 ds_act[ptr : ptr+B] = actions
                
#                 ptr += B
                
#     print("=== Caching Finished ===")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_root', type=str, default='/yanghaochuan/projects/data/pick_up_the_paper_cup.hdf5')
#     parser.add_argument('--stage_b_ckpt', type=str, default='/yanghaochuan/projects/checkpoints/stageB_papercup.pt')
#     parser.add_argument('--output_path', type=str, default='/yanghaochuan/projects/data/latents_cache.hdf5')
#     args = parser.parse_args()
#     cache_data(args)