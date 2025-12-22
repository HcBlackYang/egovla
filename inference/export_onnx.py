# inference/export_onnx.py
import torch
import argparse

# --- 使用占位符模拟导入 ---
from models.fusion_encoder import FusionEncoder
from models.rdt_model import RDTWrapper
# --- 占位符结束 ---

class EndToEndModel(torch.nn.Module):
    """将感知和控制模型包装成一个端到端模型，便于导出"""
    def __init__(self, fusion_encoder, rdt_model):
        super().__init__()
        self.fusion_encoder = fusion_encoder
        # 注意：扩散模型的采样循环不易直接导出
        # 我们通常只导出其核心的“噪声预测网络”
        self.rdt_noise_predictor = rdt_model.rdt_model
    
    def forward(self, video, text, state, ff_summary, noisy_action, timestep):
        encoder_outputs = self.fusion_encoder(video, text, state, ff_summary)
        
        # 在导出时，我们需要一个固定的融合逻辑
        # 这里复用RDTWrapper中的逻辑
        e_t = encoder_outputs['e_t']
        task_cond = torch.mean(encoder_outputs['task_slots'], dim=1) # 简化融合
        final_cond = e_t + task_cond # 极简化融合
        
        predicted_noise = self.rdt_noise_predictor(noisy_action, timestep, final_cond)
        return predicted_noise

def export_to_onnx(args):
    device = torch.device("cpu") # 导出时通常使用CPU
    print("开始导出模型到 ONNX...")

    # 1. 加载模型
    fusion_encoder = FusionEncoder().to(device).eval()
    rdt_model = RDTWrapper(action_dim=7).to(device).eval()
    
    # checkpoint = torch.load(args.weights, map_location=device)
    # ... 加载权重 ...
    
    # 2. 创建端到端包装模型
    e2e_model = EndToEndModel(fusion_encoder, rdt_model)
    
    # 3. 创建虚拟输入 (dummy input)
    batch_size = 1
    dummy_inputs = (
        torch.randn(batch_size, 16, 3, 224, 224), # video
        torch.randn(batch_size, 10, 768),         # text
        torch.randn(batch_size, 16, 7),           # state
        torch.randn(batch_size, 1, 768),          # ff_summary
        torch.randn(batch_size, 7),               # noisy_action
        torch.randint(0, 100, (batch_size,)),     # timestep
    )
    
    # 4. 导出
    torch.onnx.export(
        e2e_model,
        dummy_inputs,
        "model.onnx",
        input_names=['video', 'text', 'state', 'ff_summary', 'noisy_action', 'timestep'],
        output_names=['predicted_noise'],
        dynamic_axes={ # 允许batch size动态变化
            'video': {0: 'batch_size'},
            'text': {0: 'batch_size'},
            'state': {0: 'batch_size'},
            'ff_summary': {0: 'batch_size'},
            'noisy_action': {0: 'batch_size'},
            'timestep': {0: 'batch_size'},
            'predicted_noise': {0: 'batch_size'}
        },
        opset_version=14 # ONNX算子集版本
    )
    
    print("模型已成功导出为 model.onnx")
    print("下一步可以使用 trtexec --onnx=model.onnx --fp16 --workspace=4096 来构建TensorRT引擎")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ONNX 导出脚本")
    parser.add_argument('--weights', type=str, default='./checkpoints/stageC_final.pt', help='训练好的模型权重路径')
    args = parser.parse_args()
    export_to_onnx(args)