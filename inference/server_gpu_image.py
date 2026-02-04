# import numpy as np
# import socket
# import struct
# import json
# import cv2
# import numpy as np
# import torch
# import time
# import sys
# import traceback
# from inference.deploy_agent_safe import RealTimeAgent

# def run_image_inference_server(host='0.0.0.0', port=6000):
#     print("正在初始化 RDT 模型...", flush=True)
#     agent = RealTimeAgent()
    
#     server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#     server.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1) 
#     server.bind((host, port))
#     server.listen(1)
    
#     print(f"=== GPU 推理服务就绪 ===", flush=True)
#     print(f"监听端口: {port} | 等待接收 16 帧序列...", flush=True)

#     while True:
#         conn, addr = server.accept()
#         conn.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
#         print(f"[New Connection] 来自: {addr}", flush=True)
        
#         is_first_frame = True
        
#         try:
#             while True:
#                 # 1. 读取头长度
#                 header_len_bytes = recv_all(conn, 4)
#                 if not header_len_bytes: break
#                 header_len = struct.unpack('>I', header_len_bytes)[0]
                
#                 # 2. 读取 Header
#                 header_bytes = recv_all(conn, header_len)
#                 header = json.loads(header_bytes.decode('utf-8'))
                
#                 img_sizes = header.get('img_sizes', [])
#                 qpos = header['qpos']
                
#                 total_img_size = sum(img_sizes)
                
#                 # 3. 读取所有图片数据
#                 all_img_bytes = recv_all(conn, total_img_size)
                
#                 t0 = time.time()
                
#                 # 4. 解码图片序列
#                 frames_list = []
#                 cursor = 0
#                 for size in img_sizes:
#                     chunk = all_img_bytes[cursor : cursor + size]
#                     cursor += size
                    
#                     img_np = np.frombuffer(chunk, dtype=np.uint8)
#                     frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
#                     frames_list.append(frame)
                
#                 if not frames_list:
#                     print("⚠️ 收到空图片列表", flush=True)
#                     continue

#                 if is_first_frame:
#                     print("[Server] 锁定首帧 (Anchor Frame)", flush=True)
#                     # 冷启动时，用序列的第一帧作为 Anchor
#                     agent.reset_session(frames_list[0], current_qpos=qpos)
#                     is_first_frame = False
                
#                 # 5. 推理 (传入整个序列)
#                 try:
#                     action = agent.step(frames_list, qpos)
#                 except Exception as e:
#                     print(f"\n❌ 模型推理内部错误: {e}", flush=True)
#                     traceback.print_exc()
#                     break 
                
#                 inference_time = (time.time() - t0) * 1000
                
#                 response = {
#                     "actions": [action], 
#                     "trajectory": None
#                 }
                
#                 resp_bytes = json.dumps(response).encode('utf-8')
#                 conn.sendall(struct.pack('>I', len(resp_bytes)))
#                 conn.sendall(resp_bytes)
                
#                 # try:
#                 #     first_val = action[0][0]
#                 #     print(f"\r[Infer] Time: {inference_time:.1f}ms | SeqLen: {len(frames_list)} | J0: {first_val:.2f}", end="", flush=True)
#                 # except:
#                 #     print(f"\r[Infer] Time: {inference_time:.1f}ms", end="", flush=True)
                
#                 try:
                    
#                     # 1. 转换成 numpy 方便处理形状
#                     # 无论它原本是 list 还是 tensor，先转 numpy
#                     if hasattr(action, 'cpu'):
#                         pred_arr = action.detach().cpu().numpy()
#                     else:
#                         pred_arr = np.array(action)
                    
#                     # 2. 形状诊断与归一化
#                     # 情况 A: [Batch, Horizon, Dim] -> (1, 64, 8)
#                     if pred_arr.ndim == 3:
#                         pred_traj = pred_arr[0] # 取 Batch 0 -> (64, 8)
#                     # 情况 B: [Horizon, Dim] -> (64, 8)
#                     elif pred_arr.ndim == 2:
#                         pred_traj = pred_arr    # 已经是轨迹了
#                     # 情况 C: [Batch, Dim] -> (1, 8) 或者是单步预测
#                     elif pred_arr.ndim == 1:
#                         # 把它变成 (1, 8) 的二维矩阵，假装它是只有1步的轨迹
#                         pred_traj = pred_arr[np.newaxis, :]
#                     else:
#                         raise ValueError(f"Unknown shape: {pred_arr.shape}")

#                     # 3. 截取前 15 步 (如果只有1步，就只会取到1步)
#                     steps_to_show = 15
#                     exec_traj = pred_traj[:steps_to_show]
                    
#                     print(f"\n{'='*25} RDT Action (First {len(exec_traj)} Steps) {'='*25}")
#                     print(f"[Inference Time] {inference_time:.1f}ms")
#                     print(f"[Raw Shape] {pred_arr.shape} -> Processing as {pred_traj.shape}")
                    
#                     # 打印表头
#                     header = f"{'Step':<4} | {'J0':^7} {'J1':^7} {'J2':^7} {'J3':^7} {'J4':^7} {'J5':^7} {'J6':^7} | {'Grip':^6}"
#                     print(header)
#                     print("-" * len(header))

#                     # 4. 循环打印
#                     for i, step in enumerate(exec_traj):
#                         # 确保 step 也是数组
#                         step = np.array(step).flatten()
                        
#                         if len(step) >= 8:
#                             joints = step[:7]
#                             gripper = step[7]
#                             joints_str = " ".join([f"{x: .4f}" for x in joints])
#                             print(f"{i:<4} | {joints_str} | {gripper:.4f}")
#                         else:
#                             print(f"{i:<4} | [Error: Dim={len(step)}] {step}")

#                     print("="*82 + "\n")
                    
#                 except Exception as e:
#                     print(f"\n[Error Printing] {e}")
#                     print(f"[Raw Data] {action}")

#         except Exception as e:
#             print(f"\n[Error] 连接异常: {e}", flush=True)
#             traceback.print_exc() 
#         finally:
#             conn.close()
#             print("\n等待下一次连接...", flush=True)

# def recv_all(sock, n):
#     data = b''
#     while len(data) < n:
#         try:
#             packet = sock.recv(n - len(data))
#             if not packet: return None
#             data += packet
#         except:
#             return None
#     return data

# if __name__ == '__main__':
#     run_image_inference_server()






# # ego 二值化
# import socket
# import struct
# import json
# import cv2
# import numpy as np
# import torch
# import time
# import sys
# import traceback
# from inference.deploy_agent_safe import RealTimeAgent

# def run_image_inference_server(host='0.0.0.0', port=6000):
#     print("正在初始化 RDT 模型...", flush=True)
#     agent = RealTimeAgent()
    
#     server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#     server.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1) 
#     server.bind((host, port))
#     server.listen(1)
    
#     print(f"=== GPU 推理服务就绪 (Stop-and-Think Mode) ===", flush=True)
#     print(f"监听端口: {port} | 等待历史序列...", flush=True)

#     while True:
#         conn, addr = server.accept()
#         conn.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
#         print(f"[New Connection] 来自: {addr}", flush=True)
        
#         is_first_frame = True
        
#         try:
#             while True:
#                 # 1. 读取协议头
#                 header_len_bytes = recv_all(conn, 4)
#                 if not header_len_bytes: break
#                 header_len = struct.unpack('>I', header_len_bytes)[0]
                
#                 header_bytes = recv_all(conn, header_len)
#                 header = json.loads(header_bytes.decode('utf-8'))
                
#                 img_sizes = header.get('img_sizes', [])
#                 qpos = header['qpos']
#                 total_img_size = sum(img_sizes)
                
#                 # 2. 读取图片载荷 (Motion Phase History)
#                 all_img_bytes = recv_all(conn, total_img_size)
                
#                 t0 = time.time()
                
#                 # 3. 解码
#                 frames_list = []
#                 cursor = 0
#                 for size in img_sizes:
#                     chunk = all_img_bytes[cursor : cursor + size]
#                     cursor += size
#                     img_np = np.frombuffer(chunk, dtype=np.uint8)
#                     frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
#                     frames_list.append(frame)
                
#                 if not frames_list: continue

#                 # 4. 冷启动处理 (Anchor Frame)
#                 if is_first_frame:
#                     print("[Server] 锁定首帧 (Anchor Frame)", flush=True)
#                     # 使用序列的第一帧作为 Anchor
#                     agent.reset_session(frames_list[0], current_qpos=qpos)
#                     is_first_frame = False
                
#                 # 5. 调用 Agent (Update History -> Infer)
#                 try:
#                     # 这里 frames_list 代表了机器人过去几秒钟的运动历史
#                     # Agent 会先把它加入 Buffer，然后基于更新后的 Buffer 进行推理
#                     action = agent.step(frames_list, qpos)
#                 except Exception as e:
#                     print(f"\n❌ 模型推理错误: {e}", flush=True)
#                     traceback.print_exc()
#                     break 
                
#                 inference_time = (time.time() - t0) * 1000
                
#                 # 6. 发送响应 (Inference Phase Action)
#                 response = {
#                     "actions": [action], 
#                     "trajectory": None
#                 }
                
#                 resp_bytes = json.dumps(response).encode('utf-8')
#                 conn.sendall(struct.pack('>I', len(resp_bytes)))
#                 conn.sendall(resp_bytes)
                
#                 # # 打印日志 (仅打印第一步动作，避免刷屏)
#                 # try:
#                 #     # 兼容不同维度的输出
#                 #     action_arr = np.array(action)
#                 #     if action_arr.ndim == 3: action_val = action_arr[0, 0, 0] # [B, T, D]
#                 #     elif action_arr.ndim == 2: action_val = action_arr[0, 0]    # [T, D]
#                 #     else: action_val = action_arr[0]                            # [D]
                    
#                 #     print(f"\r[Infer] Time: {inference_time:.1f}ms | HistLen: {len(frames_list)} | J0: {action_val:.2f}", end="", flush=True)
#                 # except:
#                 #     pass
#                 try:
                    
#                     # 1. 转换成 numpy 方便处理形状
#                     # 无论它原本是 list 还是 tensor，先转 numpy
#                     if hasattr(action, 'cpu'):
#                         pred_arr = action.detach().cpu().numpy()
#                     else:
#                         pred_arr = np.array(action)
                    
#                     # 2. 形状诊断与归一化
#                     # 情况 A: [Batch, Horizon, Dim] -> (1, 64, 8)
#                     if pred_arr.ndim == 3:
#                         pred_traj = pred_arr[0] # 取 Batch 0 -> (64, 8)
#                     # 情况 B: [Horizon, Dim] -> (64, 8)
#                     elif pred_arr.ndim == 2:
#                         pred_traj = pred_arr    # 已经是轨迹了
#                     # 情况 C: [Batch, Dim] -> (1, 8) 或者是单步预测
#                     elif pred_arr.ndim == 1:
#                         # 把它变成 (1, 8) 的二维矩阵，假装它是只有1步的轨迹
#                         pred_traj = pred_arr[np.newaxis, :]
#                     else:
#                         raise ValueError(f"Unknown shape: {pred_arr.shape}")

#                     # 3. 截取前 15 步 (如果只有1步，就只会取到1步)
#                     steps_to_show = 15
#                     exec_traj = pred_traj[:steps_to_show]
                    
#                     print(f"\n{'='*25} RDT Action (First {len(exec_traj)} Steps) {'='*25}")
#                     print(f"[Inference Time] {inference_time:.1f}ms")
#                     print(f"[Raw Shape] {pred_arr.shape} -> Processing as {pred_traj.shape}")
                    
#                     # 打印表头
#                     header = f"{'Step':<4} | {'J0':^7} {'J1':^7} {'J2':^7} {'J3':^7} {'J4':^7} {'J5':^7} {'J6':^7} | {'Grip':^6}"
#                     print(header)
#                     print("-" * len(header))

#                     # 4. 循环打印
#                     for i, step in enumerate(exec_traj):
#                         # 确保 step 也是数组
#                         step = np.array(step).flatten()
                        
#                         if len(step) >= 8:
#                             joints = step[:7]
#                             gripper = step[7]
#                             joints_str = " ".join([f"{x: .4f}" for x in joints])
#                             print(f"{i:<4} | {joints_str} | {gripper:.4f}")
#                         else:
#                             print(f"{i:<4} | [Error: Dim={len(step)}] {step}")

#                     print("="*82 + "\n")
                    
#                 except Exception as e:
#                     print(f"\n[Error Printing] {e}")
#                     print(f"[Raw Data] {action}")

#         except Exception as e:
#             print(f"\n[Error] 连接异常: {e}", flush=True)
#             traceback.print_exc() 
#         finally:
#             conn.close()
#             print("\n等待下一次连接...", flush=True)

# def recv_all(sock, n):
#     data = b''
#     while len(data) < n:
#         try:
#             packet = sock.recv(n - len(data))
#             if not packet: return None
#             data += packet
#         except:
#             return None
#     return data

# if __name__ == '__main__':
#     run_image_inference_server()



# ego 双摄二值化
import socket
import struct
import json
import cv2
import numpy as np
import torch
import time
import sys
import traceback
from inference.deploy_agent_safe import RealTimeAgent

def run_image_inference_server(host='0.0.0.0', port=6000):
    print("正在初始化 RDT 模型 (Dual View)...", flush=True)
    agent = RealTimeAgent()
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1) 
    server.bind((host, port))
    server.listen(1)
    
    print(f"=== GPU 推理服务就绪 ===", flush=True)
    print(f"监听端口: {port} | 等待双摄数据...", flush=True)

    while True:
        conn, addr = server.accept()
        conn.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
        print(f"[New Connection] 来自: {addr}", flush=True)
        
        is_first_frame = True
        
        try:
            while True:
                # 1. 读取头长度 (4 bytes)
                header_len_bytes = recv_all(conn, 4)
                if not header_len_bytes: break
                header_len = struct.unpack('>I', header_len_bytes)[0]
                
                # 2. 读取 Header (JSON)
                header_bytes = recv_all(conn, header_len)
                header = json.loads(header_bytes.decode('utf-8'))
                
                # 协议约定: img_sizes 应该包含两个值 [size_main, size_wrist]
                img_sizes = header.get('img_sizes', [])
                qpos = header['qpos']
                
                if len(img_sizes) != 2:
                    print(f"⚠️ 警告: 收到 {len(img_sizes)} 张图片，期望 2 张 (Main + Wrist)。")
                
                total_img_size = sum(img_sizes)
                
                # 3. 读取 Payload (Images)
                all_img_bytes = recv_all(conn, total_img_size)
                
                t0 = time.time()
                
                # 4. 解码图片
                # 假设发送顺序固定为: [Main, Wrist]
                cursor = 0
                
                # Main Decode
                size_m = img_sizes[0]
                chunk_m = all_img_bytes[cursor : cursor + size_m]
                cursor += size_m
                main_img = cv2.imdecode(np.frombuffer(chunk_m, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                # Wrist Decode
                size_w = img_sizes[1]
                chunk_w = all_img_bytes[cursor : cursor + size_w]
                cursor += size_w
                wrist_img = cv2.imdecode(np.frombuffer(chunk_w, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                if main_img is None or wrist_img is None:
                    print("⚠️ 图片解码失败", flush=True)
                    continue

                # 5. 调用 Agent
                # Stop-and-Think 模式下，每次只传最新的当前帧，历史由 Server 端的 Agent 维护
                # 所以我们构造长度为 1 的 List
                
                if is_first_frame:
                    print("[Server] Resetting Session (Anchor Frame)...", flush=True)
                    agent.reset_session(main_img, wrist_img, current_qpos=qpos)
                    is_first_frame = False
                
                try:
                    # 传入 List[Frame]
                    action = agent.step([main_img], [wrist_img], qpos)
                except Exception as e:
                    print(f"\n❌ 推理错误: {e}", flush=True)
                    traceback.print_exc()
                    break 
                
                inference_time = (time.time() - t0) * 1000
                
                # 6. 回传结果
                response = {
                    "actions": [action], 
                    "trajectory": None
                }
                resp_bytes = json.dumps(response).encode('utf-8')
                conn.sendall(struct.pack('>I', len(resp_bytes)))
                conn.sendall(resp_bytes)
                
                # 简单日志
                print(f"\r[Infer] Time: {inference_time:.1f}ms", end="", flush=True)

        except Exception as e:
            print(f"\n[Error] 连接断开: {e}", flush=True)
        finally:
            conn.close()
            print("\n等待下一次连接...", flush=True)

def recv_all(sock, n):
    data = b''
    while len(data) < n:
        try:
            packet = sock.recv(n - len(data))
            if not packet: return None
            data += packet
        except:
            return None
    return data

if __name__ == '__main__':
    run_image_inference_server()