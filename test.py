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

import torch
import sys
import os
import torch
import torch.nn as nn

# ==========================================
# 1. 强力环境配置 (解决 ModuleNotFoundError)
# ==========================================
# RDT 库的真实根目录
RDT_ROOT = "/yanghaochuan/projects/RoboticsDiffusionTransformer"

print(f"🔄 [Step 1] 切换工作目录到 RDT 源码库: {RDT_ROOT}")
# 物理切换目录，让 python import models 自然指向库文件
os.chdir(RDT_ROOT)

# 把路径加到最前面
if RDT_ROOT not in sys.path:
    sys.path.insert(0, RDT_ROOT)

# 🧹 扫除障碍：如果之前错误加载了 'models'，把它踢出内存
keys_to_clean = [k for k in sys.modules if k == 'models' or k.startswith('models.')]
if keys_to_clean:
    print(f"🧹 [Step 2] 清理冲突模块缓存: {len(keys_to_clean)} 个")
    for k in keys_to_clean:
        del sys.modules[k]

print("🚀 [Step 3] 尝试导入 RDT 模型类...")

try:
    # 现在的环境应该和在 RDT 根目录运行一模一样
    from models.rdt.model import MultimodalDiffusionTransformer
    ModelClass = MultimodalDiffusionTransformer
    print(f"✅ 成功导入: {ModelClass.__name__}")
except ImportError:
    try:
        from models.rdt.model import RDT
        ModelClass = RDT
        print(f"✅ 成功导入: {ModelClass.__name__}")
    except Exception as e:
        print(f"❌ 导入彻底失败: {e}")
        sys.exit(1)

# ==========================================
# 2. 开始核心测试 (测出到底谁是 1152)
# ==========================================

def run_test():
    print("\n" + "="*50)
    print("🧪 诊断开始：模型到底吃哪一套参数？")
    print("="*50)

    # 准备基础参数 (防止无关报错)
    base_kwargs = {
        'action_dim': 8, 'horizon': 64, 'pred_horizon': 64,
        'img_token_dim': 1152, 'lang_token_dim': 4096, 'state_token_dim': 128,
        'patch_size': 14, 'img_size': 224, 
        'img_adaptor': 'mlp2x_gelu', 'lang_adaptor': 'mlp2x_gelu', 'state_adaptor': 'mlp2x_gelu',
        'depth': 1, 'num_heads': 1 # 设小点，跑得快
    }

    # --- 测试 A: 扁平参数 (Kwargs) ---
    print("\n👉 测试 A: 传入扁平参数 (kwargs['hidden_size'] = 2048)")
    kwargs_a = base_kwargs.copy()
    kwargs_a['hidden_size'] = 2048 # <--- 我们希望生效的值
    
    try:
        model_a = ModelClass(**kwargs_a)
        
        # 检查生效情况
        val_a = getattr(model_a, 'hidden_size', '未找到属性')
        
        # 深度检查：看模型内部第一个 Linear 层的维度
        linear_dim_a = "未知"
        for m in model_a.modules():
            if isinstance(m, nn.Linear):
                linear_dim_a = m.out_features
                break
                
        print(f"   [结果] model.hidden_size: {val_a}")
        print(f"   [结果] 实际 Linear 维度:  {linear_dim_a}")
        
        if linear_dim_a == 2048:
            print("   🎉 结论：扁平传参有效！")
        elif linear_dim_a == 1152:
            print("   ⚠️ 结论：扁平传参失效！模型使用了默认值 1152。")
        else:
            print(f"   ❓ 结论：奇怪的值 {linear_dim_a}")
            
    except Exception as e:
        print(f"   ❌ 报错: {e}")


    # --- 测试 B: 嵌套 Config (args.rdt) ---
    print("\n👉 测试 B: 传入嵌套结构 (args.rdt['hidden_size'] = 2048)")
    
    class Args: pass
    args_b = Args()
    # 模拟 Config 文件的嵌套结构
    args_b.rdt = {'hidden_size': 2048} 
    # 同时也把其他参数赋给 args (混合模式)
    for k, v in base_kwargs.items(): setattr(args_b, k, v)
    
    try:
        # 有些模型可能不支持直接传对象，我们先试试
        model_b = ModelClass(args_b)
        
        val_b = getattr(model_b, 'hidden_size', '未找到属性')
        linear_dim_b = "未知"
        for m in model_b.modules():
            if isinstance(m, nn.Linear):
                linear_dim_b = m.out_features
                break
                
        print(f"   [结果] model.hidden_size: {val_b}")
        print(f"   [结果] 实际 Linear 维度:  {linear_dim_b}")
        
        if linear_dim_b == 2048:
            print("   🎉 结论：嵌套结构有效！必须构造 args.rdt。")
            
    except Exception as e:
        print(f"   ❌ 报错 (可能模型不支持对象传参): {e}")
        
        # 如果对象传参失败，试试纯字典嵌套
        print("   🔄 尝试传纯字典嵌套...")
        try:
            dict_b = base_kwargs.copy()
            dict_b['rdt'] = {'hidden_size': 2048}
            model_b_dict = ModelClass(dict_b)
            # 检查...
            for m in model_b_dict.modules():
                if isinstance(m, nn.Linear):
                    print(f"   [字典结果] 实际 Linear 维度: {m.out_features}")
                    break
        except Exception as e2:
             print(f"   ❌ 字典也报错: {e2}")

if __name__ == "__main__":
    run_test()