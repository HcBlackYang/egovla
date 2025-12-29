# import torch
# import h5py
# import os
# import numpy as np

# # ================= 路径配置 =================
# PATHS = {
#     "RDT_WEIGHTS": "/yanghaochuan/models/rdt-1b/pytorch_model.bin",
#     "CHECKPOINT":  "/yanghaochuan/checkpoints/1223stageB_papercup.pt",
#     "DATASET":     "/yanghaochuan/data/1223pick_up_the_paper_cup.hdf5"
# }
# # ===========================================

# def print_header(title):
#     print(f"\n{'='*20} {title} {'='*20}")

# def inspect_torch_file(path, label):
#     print_header(f"Inspecting {label}")
#     if not os.path.exists(path):
#         # 自动尝试备选文件名
#         if path.endswith("pytorch_model.bin"):
#             alt = path.replace("pytorch_model.bin", "diffusion_pytorch_model.bin")
#             if os.path.exists(alt): path = alt
        
#         if not os.path.exists(path):
#             print(f"❌ 文件未找到: {path}")
#             return

#     try:
#         # 尝试加载，兼容不同版本
#         try: data = torch.load(path, map_location='cpu')
#         except: data = torch.load(path, map_location='cpu', weights_only=False)
#     except Exception as e:
#         print(f"❌ 加载失败: {e}")
#         return

#     # 识别是否是 Checkpoint 格式
#     state_dict = data
#     if isinstance(data, dict) and 'state_dict' in data:
#         print(f"ℹ️  格式: Checkpoint (包含 'state_dict')")
#         if 'args' in data: 
#             print(f"ℹ️  训练参数 (Args): {data['args']}") # 打印训练时的参数配置
#         state_dict = data['state_dict']
    
#     print(f"ℹ️  总 Key 数量: {len(state_dict)}")
    
#     # === 1. 只搜索关键张量 (避免刷屏) ===
#     watchlist = ["x_pos_embed", "pos_embed", "img_cond_pos_embed", "state_proj", "action_proj", "visual_proj"]
#     print("\n🔍 --- 关键张量透视 (Filtered) ---")
#     found_any = False
    
#     for k, v in state_dict.items():
#         # 只打印 watchlist 里的，或者包含 'embed' 的前几个
#         if any(w in k for w in watchlist):
#             if torch.is_tensor(v):
#                 print(f"  • {k:<45} | Shape: {list(v.shape)}")
                
#                 # 针对 x_pos_embed 做详细维度分析
#                 if "x_pos_embed" in k and v.dim() == 3:
#                     T = v.shape[1]
#                     print(f"    👉 [深度分析] 长度={T}")
#                     if T == 34: print("       -> 推测结构: Time(1) + Freq(1) + Action(32)")
#                     elif T == 67: print("       -> 推测结构: Time(1) + Freq(1) + State(1) + Action(64)")
#                     elif T == 35: print("       -> 推测结构: Time(1) + Freq(1) + State(1) + Action(32)")
#             found_any = True

#     if not found_any: print("  (未发现关键张量，可能是 LoRA 权重或结构不同)")

#     # === 2. 打印前 5 个 Key 供参考 ===
#     print("\n📄 --- 头部 Key 采样 (前5个) ---")
#     for k in list(state_dict.keys())[:5]:
#         print(f"  • {k}")

# def inspect_hdf5_file(path, label):
#     print_header(f"Inspecting {label}")
#     if not os.path.exists(path):
#         print(f"❌ 文件未找到: {path}")
#         return

#     try:
#         with h5py.File(path, 'r') as f:
#             print(f"ℹ️  根目录 Keys: {list(f.keys())}")
            
#             print("\n🔍 --- 搜索 Action 和 Image 数据 (抽样) ---")
#             matches_act = 0
#             matches_img = 0
            
#             # 智能遍历：只找关键数据集，不遍历所有 demo
#             def sparse_visit(name, node):
#                 nonlocal matches_act, matches_img
                
#                 if isinstance(node, h5py.Dataset):
#                     lower_name = name.lower()
                    
#                     # 1. 检查 Action (只打印前 2 个找到的)
#                     if 'action' in lower_name and matches_act < 2:
#                         print(f"  • {name:<45} | Shape: {node.shape} | Type: {node.dtype}")
#                         data = node[:]
#                         print(f"    👉 [统计] Min={np.min(data):.2f}, Max={np.max(data):.2f}, Mean={np.mean(data):.2f}")
#                         matches_act += 1
                        
#                     # 2. 检查 Image (只打印前 2 个找到的)
#                     elif ('image' in lower_name or 'rgb' in lower_name) and matches_img < 2:
#                         print(f"  • {name:<45} | Shape: {node.shape}")
#                         matches_img += 1
            
#             # 使用 visititems 遍历，但通过计数器控制输出量
#             f.visititems(sparse_visit)
            
#             if matches_act == 0: 
#                 print("  ⚠️ 未找到名为 'action' 的数据集，请检查命名 (如 'actions', 'joint_states')")

#     except Exception as e:
#         print(f"❌ 读取 HDF5 失败: {e}")

# if __name__ == "__main__":
#     # 1. 检查 RDT 原始权重 (看看它是 32 还是 64)
#     inspect_torch_file(PATHS["RDT_WEIGHTS"], "RDT Base Weights (.bin)")
    
#     # 2. 检查 Stage B Checkpoint (看看你之前的训练保存了什么)
#     inspect_torch_file(PATHS["CHECKPOINT"], "Stage B Checkpoint (.pt)")
    
#     # 3. 检查数据集 (看看数据里 Action 到底是多长)
#     inspect_hdf5_file(PATHS["DATASET"], "HDF5 Dataset")

import h5py
import numpy as np

# 修改为你的 hdf5 路径
f = h5py.File('/yanghaochuan/data/1223pick_up_the_paper_cup.hdf5', 'r')
demo_key = list(f['data'].keys())[0]
siglip_feat = f['data'][demo_key]['teacher_siglip'][:]
print("Feature Max:", np.max(siglip_feat))
print("Feature Min:", np.min(siglip_feat))
print("Is all zero?", np.all(siglip_feat == 0))