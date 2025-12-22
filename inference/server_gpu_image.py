# import socket
# import struct
# import json
# import cv2
# import numpy as np
# import torch
# import time
# from inference.deploy_agent_safe import RealTimeAgent # 复用我们刚才写的安全Agent

# def run_image_inference_server(host='0.0.0.0', port=6000):
#     # 1. 初始化模型和安全控制器
#     print("正在初始化 RDT 模型...")
#     agent = RealTimeAgent()
    
#     server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#     server.bind((host, port))
#     server.listen(1)
    
#     print(f"=== GPU 推理服务就绪 ===")
#     print(f"监听端口: {port} | 等待机械臂传输图像...")

#     while True:
#         conn, addr = server.accept()
#         print(f"[New Connection] 来自: {addr}")
        
#         # 每次新连接，重置 Session (First Frame 将由第一张传来的图决定)
#         is_first_frame = True
        
#         try:
#             while True:
#                 # --- 协议解析 ---
#                 # 1. 读取头长度 (4 bytes)
#                 header_len_bytes = recv_all(conn, 4)
#                 if not header_len_bytes: break
#                 header_len = struct.unpack('>I', header_len_bytes)[0]
                
#                 # 2. 读取 Header JSON (包含 qpos 和 图片大小)
#                 header_bytes = recv_all(conn, header_len)
#                 header = json.loads(header_bytes.decode('utf-8'))
                
#                 img_size = header['img_size']
#                 qpos = header['qpos']
                
#                 # 3. 读取图片数据 (JPG 字节流)
#                 img_bytes = recv_all(conn, img_size)
                
#                 # --- 数据处理 ---
#                 # 解码图像 (JPG -> Numpy)
#                 img_np = np.frombuffer(img_bytes, dtype=np.uint8)
#                 current_frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                
#                 # 处理 First Frame 逻辑
#                 if is_first_frame:
#                     print("[Server] 锁定首帧 (Anchor Frame)")
#                     agent.reset_session(current_frame)
#                     is_first_frame = False
                
#                 # --- 推理 ---
#                 t0 = time.time()
#                 # 调用 agent.step (包含 VideoMAE 编码 + RDT 扩散 + 安全限位)
#                 action = agent.step(current_frame, qpos)
#                 inference_time = (time.time() - t0) * 1000
                
#                 # --- 发送回包 ---
#                 # 你的 RobotSystem 期望返回 {'actions': [[...]], 'trajectory': ...}
#                 # RDT 是单步预测，所以 actions 是 [[a1, a2, ...]]
#                 response = {
#                     "actions": [action], # 包装成 list of list
#                     "trajectory": None   # 暂时为空
#                 }
                
#                 resp_bytes = json.dumps(response).encode('utf-8')
#                 conn.sendall(struct.pack('>I', len(resp_bytes)))
#                 conn.sendall(resp_bytes)
                
#                 print(f"\r[Infer] Time: {inference_time:.1f}ms | Action[0]: {action[0]:.2f}", end="")

#         except Exception as e:
#             print(f"\n[Error] 连接中断: {e}")
#             import traceback
#             traceback.print_exc()
#         finally:
#             conn.close()
#             print("\n等待下一次连接...")

# def recv_all(sock, n):
#     data = b''
#     while len(data) < n:
#         packet = sock.recv(n - len(data))
#         if not packet: return None
#         data += packet
#     return data

# if __name__ == '__main__':
#     run_image_inference_server()

import socket
import struct
import json
import cv2
import numpy as np
import torch
import time
import select # 新增 select 用于非阻塞检查
from inference.deploy_agent_safe import RealTimeAgent

def flush_socket(sock):
    """
    暴力清空 Socket 缓冲区，丢弃所有旧数据。
    确保我们处理的是最新的一帧。
    """
    try:
        # 设置为非阻塞模式
        sock.setblocking(0)
        chunk_size = 4096
        flushed_bytes = 0
        while True:
            # 尝试读取，直到读不出数据抛出异常
            data = sock.recv(chunk_size)
            if not data: break
            flushed_bytes += len(data)
    except BlockingIOError:
        pass # 缓冲区空了
    except Exception as e:
        print(f"[Warn] Flush error: {e}")
    finally:
        # 恢复阻塞模式
        sock.setblocking(1)
    
    if flushed_bytes > 0:
        print(f" [Drop] 丢弃积压数据: {flushed_bytes} bytes (缓解延迟堆积)")

def run_image_inference_server(host='0.0.0.0', port=6000):
    print("正在初始化 RDT 模型...")
    agent = RealTimeAgent()
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # 禁用 Nagle 算法，减少小包延迟
    server.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1) 
    server.bind((host, port))
    server.listen(1)
    
    print(f"=== GPU 推理服务就绪 ===")
    print(f"监听端口: {port} | 等待机械臂传输图像...")

    while True:
        conn, addr = server.accept()
        conn.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1) # Client socket 也要禁用 Nagle
        print(f"[New Connection] 来自: {addr}")
        
        is_first_frame = True
        
        try:
            while True:
                # === 核心修改：在读取新的一帧之前，不需要 Flush ===
                # 因为 TCP 是流式协议，乱 Flush 会把下一帧的 Header 截断导致解包错误。
                # 这里的策略是：Client 端必须也是同步的。
                # 如果发现延迟还是高，说明是 Client 发送太快。
                
                # --- 协议解析 ---
                # 1. 读取头长度 (4 bytes)
                header_len_bytes = recv_all(conn, 4)
                if not header_len_bytes: break
                header_len = struct.unpack('>I', header_len_bytes)[0]
                
                # 2. 读取 Header
                header_bytes = recv_all(conn, header_len)
                header = json.loads(header_bytes.decode('utf-8'))
                
                img_size = header['img_size']
                qpos = header['qpos']
                
                # 3. 读取图片
                img_bytes = recv_all(conn, img_size)
                
                # --- 推理计时 ---
                t0 = time.time()
                
                img_np = np.frombuffer(img_bytes, dtype=np.uint8)
                current_frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                
                if is_first_frame:
                    print("[Server] 锁定首帧 (Anchor Frame)")
                    agent.reset_session(current_frame)
                    is_first_frame = False
                
                # 调用 Agent
                action = agent.step(current_frame, qpos)
                inference_time = (time.time() - t0) * 1000
                
                # --- 回包 ---
                response = {
                    "actions": [action], 
                    "trajectory": None
                }
                
                resp_bytes = json.dumps(response).encode('utf-8')
                conn.sendall(struct.pack('>I', len(resp_bytes)))
                conn.sendall(resp_bytes)
                
                print(f"\r[Infer] Time: {inference_time:.1f}ms | Action[0]: {action[0]:.2f}", end="")

        except Exception as e:
            print(f"\n[Error] 连接中断: {e}")
            import traceback
            traceback.print_exc()
        finally:
            conn.close()
            print("\n等待下一次连接...")

def recv_all(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet: return None
        data += packet
    return data

if __name__ == '__main__':
    run_image_inference_server()