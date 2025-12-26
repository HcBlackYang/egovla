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
    print("正在初始化 RDT 模型...", flush=True)
    agent = RealTimeAgent()
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1) 
    server.bind((host, port))
    server.listen(1)
    
    print(f"=== GPU 推理服务就绪 ===", flush=True)
    print(f"监听端口: {port} | 等待机械臂传输图像...", flush=True)

    while True:
        conn, addr = server.accept()
        conn.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
        print(f"[New Connection] 来自: {addr}", flush=True)
        
        is_first_frame = True
        
        try:
            while True:
                # 1. 读取头长度
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
                
                t0 = time.time()
                
                img_np = np.frombuffer(img_bytes, dtype=np.uint8)
                current_frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                
                if is_first_frame:
                    print("[Server] 锁定首帧 (Anchor Frame)", flush=True)
                    agent.reset_session(current_frame)
                    is_first_frame = False
                
                # 推理
                try:
                    action = agent.step(current_frame, qpos)
                except Exception as e:
                    print(f"\n❌ 模型推理内部错误: {e}", flush=True)
                    traceback.print_exc()
                    break 
                
                inference_time = (time.time() - t0) * 1000
                
                response = {
                    "actions": [action], 
                    "trajectory": None
                }
                
                resp_bytes = json.dumps(response).encode('utf-8')
                conn.sendall(struct.pack('>I', len(resp_bytes)))
                conn.sendall(resp_bytes)
                
                # [Print Fix] 安全打印日志
                try:
                    # action[0] 是第一步的完整动作 (List)，取第一个关节角打印
                    first_val = action[0][0]
                    print(f"\r[Infer] Time: {inference_time:.1f}ms | J0: {first_val:.2f}", end="", flush=True)
                except:
                    print(f"\r[Infer] Time: {inference_time:.1f}ms", end="", flush=True)

        except Exception as e:
            print(f"\n[Error] 连接异常: {e}", flush=True)
            traceback.print_exc() 
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