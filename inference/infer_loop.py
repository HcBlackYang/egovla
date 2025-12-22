# # inference/infer_loop.py
# import torch
# import cv2
# import socket
# import json
# import time
# import numpy as np
# from collections import deque
# from diffusers import DPMSolverMultistepScheduler

# # 导入本地模块
# from models.fusion_encoder import FusionEncoder
# from models.rdt_model import RDTWrapper

# # 配置路径
# VIDEO_MAE_PATH = '/yanghaochuan/models/VideoMAEv2-Large'
# RDT_PATH = '/yanghaochuan/models/rdt-1b'
# CHECKPOINT_PATH = '/yanghaochuan/projects/checkpoints/stageC_final.pt' # 假设训练好的权重在这里

# def setup_server(port=6000):
#     server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#     server.bind(('0.0.0.0', port))
#     server.listen(1)
#     print(f"============== RDT 推理服务启动 ==============")
#     print(f"监听端口: {port}")
#     print(f"等待机械臂连接...")
#     conn, addr = server.accept()
#     print(f"已连接: {addr}")
#     return conn

# def inference_server():
#     # ---------------------------------------------------------
#     # 1. 初始化环境与模型
#     # ---------------------------------------------------------
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"使用设备: {device}")

#     print("正在加载 FusionEncoder (VideoMAE)...")
#     # 使用本地路径
#     encoder = FusionEncoder(backbone_path='/yanghaochuan/models/VideoMAEv2-Large').to(device).eval()
    
#     print("正在加载 RDT-1B Policy...")
#     # 使用本地路径
#     policy = RDTWrapper(model_path='/yanghaochuan/models/rdt-1b').to(device).eval()
    
#     # 尝试加载训练好的权重 (如果有)
#     checkpoint_path = '/yanghaochuan/projects/checkpoints/stageC_final.pt'
#     try:
#             if hasattr(torch, 'load'):
#                 # 指向你刚刚训练好的 checkpont
#                 checkpoint_path = '/yanghaochuan/projects/checkpoints/stageC_final.pt' 
#                 print(f"尝试加载微调权重: {checkpoint_path}")
#                 checkpoint = torch.load(checkpoint_path, map_location=device)
                
#                 # === 修改这里：使用与 stageC_joint.py 一致的键名 ===
#                 encoder.load_state_dict(checkpoint['encoder_state_dict']) 
#                 policy.load_state_dict(checkpoint['rdt_state_dict'])   
#                 print("权重加载成功！")
                
#         except FileNotFoundError:
#             print("警告: 未找到微调权重...")

#     # ---------------------------------------------------------
#     # 2. 设置 DPM-Solver 调度器 (实现快速推理)
#     # ---------------------------------------------------------
#     scheduler = DPMSolverMultistepScheduler(
#         num_train_timesteps=1000,
#         beta_schedule="squaredcos_cap_v2",
#         algorithm_type="dpmsolver++"
#     )
#     scheduler.set_timesteps(10) # 设置为10步采样

#     # ---------------------------------------------------------
#     # 3. 建立 Socket 服务
#     # ---------------------------------------------------------
#     conn = setup_server(port=6000)
    
#     # ---------------------------------------------------------
#     # 4. 打开摄像头并捕获全局第一帧 (Task Anchor)
#     # ---------------------------------------------------------
#     cap = cv2.VideoCapture(0) # 根据实际情况修改 ID，如 0, 1, 或 /dev/video0
#     if not cap.isOpened():
#         raise RuntimeError("无法打开摄像头 ID 0")

#     print("正在捕获全局第一帧 (Task Anchor)...")
#     # 读取第一帧，作为整个任务周期的全局参考
#     ret, global_first_frame_raw = cap.read()
#     if not ret:
#         raise RuntimeError("无法从摄像头读取第一帧！")
    
#     # 预处理第一帧并固定在 GPU 上
#     # Resize -> RGB -> Tensor -> Permute -> Unsqueeze -> Normalize
#     global_first_frame_resized = cv2.resize(global_first_frame_raw, (224, 224))
#     global_first_frame_rgb = cv2.cvtColor(global_first_frame_resized, cv2.COLOR_BGR2RGB)
    
#     # 形状变换: [H, W, C] -> [1, C, H, W] (增加 Batch 和 Channel 维度)
#     # 注意：这里我们保留 Batch=1 的维度，因为推理时 Batch Size 通常为 1
#     fixed_ff_tensor = torch.tensor(global_first_frame_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    
#     print("全局第一帧已锁定。")

#     # ---------------------------------------------------------
#     # 5. 缓冲区初始化与预热
#     # ---------------------------------------------------------
#     window_size = 16
#     video_buffer = deque(maxlen=window_size)
#     state_buffer = deque(maxlen=window_size)
    
#     print("预热摄像头缓冲区...")
#     # 使用刚才捕获的第一帧填满缓冲区，防止冷启动时的空数据
#     # 这里也可以选择继续读取16帧，视具体策略而定
#     # 为保证连续性，我们继续读取真实流填充
#     video_buffer.append(global_first_frame_rgb) # 先放入第一帧
#     state_buffer.append(np.zeros(7))            # 放入初始状态
    
#     for _ in range(window_size - 1):
#         ret, frame = cap.read()
#         if not ret: break
#         frame = cv2.resize(frame, (224, 224))
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         video_buffer.append(frame)
#         state_buffer.append(np.zeros(7))

#     # ---------------------------------------------------------
#     # 6. 实时推理循环
#     # ---------------------------------------------------------
#     print(">>> 开始推理服务循环 <<<")
#     try:
#         while True:
#             # A. 接收请求头 (4字节长度)
#             data_len_bytes = conn.recv(4)
#             if not data_len_bytes: 
#                 print("客户端断开连接。")
#                 break
#             data_len = int.from_bytes(data_len_bytes, byteorder='big')
            
#             # B. 接收请求体 (JSON)
#             data = b""
#             while len(data) < data_len:
#                 packet = conn.recv(data_len - len(data))
#                 if not packet: break
#                 data += packet
            
#             if not data: break
            
#             start_time = time.time()
            
#             # C. 解析数据
#             req = json.loads(data.decode())
#             current_joints = req['qpos'] # 期望格式: List[float] 长度7
            
#             # D. 更新传感器数据
#             # D1. 读取最新摄像头帧
#             ret, frame_raw = cap.read()
#             if ret:
#                 frame = cv2.resize(frame_raw, (224, 224))
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 video_buffer.append(frame)
#             else:
#                 print("警告: 丢帧")
            
#             # D2. 更新状态
#             state_buffer.append(current_joints)

#             # E. 构造模型输入 Tensor
#             # Video: [16, H, W, C] -> [1, 16, C, H, W]
#             vid_np = np.array(video_buffer)
#             vid_t = torch.tensor(vid_np, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0).to(device) / 255.0
            
#             # State: [16, 7] -> [1, 16, 7]
#             state_t = torch.tensor(np.array(state_buffer), dtype=torch.float32).unsqueeze(0).to(device)
            
#             # First Frame: 使用循环外锁定的 fixed_ff_tensor
#             # 形状应为 [1, C, H, W] 或 [1, 1, C, H, W] 取决于 FusionEncoder 定义
#             # 我们的 FusionEncoder 期望 [B, C, H, W] 用于 first_frame_summary
#             # fixed_ff_tensor 已经是 [1, C, H, W]
#             ff_t = fixed_ff_tensor
            
#             # Text: 暂时使用空指令或默认指令，实际应接收 req['instruction'] 并 tokenize
#             # 这里构造一个 dummy text embedding [1, 10, 768]
#             text_t = torch.zeros(1, 10, 768).to(device) 

#             # F. 模型推理
#             with torch.no_grad():
#                 # F1. 视觉编码
#                 feats = encoder(vid_t, text_t, state_t, ff_t)
                
#                 # F2. 动作采样 (DPM-Solver)
#                 # 初始化随机噪声 [B, Seq, Dim]
#                 latents = torch.randn(1, 1, 7, device=device) 
                
#                 for t in scheduler.timesteps:
#                     model_input = scheduler.scale_model_input(latents, t)
#                     # 预测噪声
#                     noise_pred = policy(model_input, t, feats)
#                     # 移除噪声
#                     latents = scheduler.step(noise_pred, t, latents).prev_sample
                
#                 # 获取动作结果 (取第一个动作)
#                 action = latents[0, 0].cpu().numpy().tolist()

#             # G. 发送响应
#             resp = json.dumps({"action": action})
#             resp_bytes = resp.encode()
#             # 发送长度头 + 内容
#             conn.sendall(len(resp_bytes).to_bytes(4, byteorder='big'))
#             conn.sendall(resp_bytes)
            
#             # 打印性能监控
#             fps = 1.0 / (time.time() - start_time)
#             print(f"推理完成 | FPS: {fps:.1f} | Action: {[round(x,3) for x in action[:3]]}...")

#     except Exception as e:
#         print(f"发生严重错误: {e}")
#         import traceback
#         traceback.print_exc()
#     finally:
#         print("正在关闭资源...")
#         conn.close()
#         cap.release()
#         print("服务已停止")

# if __name__ == '__main__':
#     inference_server()