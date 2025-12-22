# utils/check_env.py
import os
import sys
import torch
import cv2
import h5py
import transformers
import diffusers

def check_env():
    print("="*10 + " 环境自检开始 " + "="*10)
    all_pass = True

    # 1. 检查 CUDA 和 Pytorch
    print(f"\n[1] 检查硬件环境:")
    print(f"   - Python: {sys.version.split()[0]}")
    print(f"   - PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"   - CUDA: 可用 ({torch.cuda.get_device_name(0)})")
        device = torch.device("cuda")
    else:
        print("   - [错误] CUDA 不可用！模型无法在 GPU 上训练。")
        all_pass = False
        device = torch.device("cpu")

    # 2. 检查核心库版本
    print(f"\n[2] 检查核心库:")
    try:
        import einops
        print(f"   - Transformers: {transformers.__version__}")
        print(f"   - Diffusers: {diffusers.__version__}")
        print(f"   - OpenCV: {cv2.__version__}")
        print(f"   - h5py: {h5py.__version__}")
        print(f"   - einops: {einops.__version__}") # VideoMAE 常用依赖
    except ImportError as e:
        print(f"   - [错误] 缺少库: {e.name}")
        print(f"     请运行: pip install {e.name}")
        all_pass = False

    # 3. 检查本地模型路径
    print(f"\n[3] 检查本地模型路径:")
    # 这是你之前提供的路径
    videomae_path = '/yanghaochuan/models/VideoMAEv2-Large'
    rdt_path = '/yanghaochuan/models/rdt-1b'
    
    if os.path.exists(videomae_path):
        print(f"   - VideoMAE路径存在: ✅")
    else:
        print(f"   - [错误] 找不到 VideoMAE: {videomae_path}")
        all_pass = False

    if os.path.exists(rdt_path):
        print(f"   - RDT路径存在: ✅")
    else:
        print(f"   - [错误] 找不到 RDT: {rdt_path}")
        all_pass = False

    # 4. 尝试加载项目代码 (验证 import 是否正常)
    print(f"\n[4] 验证项目模块加载:")
    try:
        sys.path.append(os.getcwd()) # 确保能扫描到 models 目录
        from models.fusion_encoder import FusionEncoder
        from models.rdt_model import RDTWrapper
        
        print("   - 正在尝试加载 FusionEncoder (可能需要几秒)...")
        # 尝试实例化，检查显存能否加载
        encoder = FusionEncoder(backbone_path=videomae_path).to(device)
        print("     -> FusionEncoder 加载成功!")

        print("   - 正在尝试加载 RDTWrapper...")
        # 注意 action_dim 改为了 8
        policy = RDTWrapper(action_dim=8, model_path=rdt_path).to(device)
        print("     -> RDTWrapper 加载成功!")
        
    except Exception as e:
        print(f"   - [错误] 加载项目模块失败: {e}")
        import traceback
        traceback.print_exc()
        all_pass = False

    print("\n" + "="*30)
    if all_pass:
        print("✅✅✅ 恭喜！环境搭建完成，可以开始训练！ ✅✅✅")
    else:
        print("❌❌❌ 环境存在问题，请修复上述错误。 ❌❌❌")

if __name__ == "__main__":
    check_env()