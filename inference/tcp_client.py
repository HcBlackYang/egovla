# import socket
# import json
# import struct
# import cv2
# import numpy as np
# import logging

# class TCPClientPolicy:
#     def __init__(self, host, port):
#         self.host = host
#         self.port = int(port)
#         self.sock = None
#         self.connect()

#     def connect(self):
#         try:
#             if self.sock: self.sock.close()
#             self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#             self.sock.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1) 
            
#             # [修改点] 延长超时到 60秒，防止 compile 导致的超时断连
#             self.sock.settimeout(60.0) 
            
#             self.sock.connect((self.host, self.port))
#             logging.info(f"✅ TCP连接成功: {self.host}:{self.port}")
#         except Exception as e:
#             logging.error(f"❌ TCP连接失败: {e}")
#             self.sock = None

#     def infer(self, element):
#         if self.sock is None:
#             self.connect()
#             if self.sock is None: return self._empty_response()

#         # 1. 提取
#         if 'qpos' in element:
#             qpos = element['qpos']
#         else:
#             qpos = element['observation/state'][:7].tolist()

#         if 'observation/wrist_image' in element:
#             image = element['observation/wrist_image']
#         elif 'observation/image' in element:
#             image = element['observation/image']
#         else:
#             return self._empty_response()
        
#         # 2. 压缩
#         encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
#         _, img_encoded = cv2.imencode('.jpg', image, encode_param)
#         img_bytes = img_encoded.tobytes()
        
#         # 3. 构造 Header
#         header = {"qpos": qpos, "img_size": len(img_bytes)}
#         header_bytes = json.dumps(header).encode('utf-8')
        
#         try:
#             # 4. 发送
#             self.sock.sendall(struct.pack('>I', len(header_bytes)))
#             self.sock.sendall(header_bytes)
#             self.sock.sendall(img_bytes)
            
#             # 5. 接收
#             len_bytes = self.recv_all(4)
#             if not len_bytes: 
#                 logging.warning("⚠️ Server closed connection (EOF).")
#                 self.sock.close() 
#                 self.sock = None
#                 return self._empty_response()
            
#             resp_len = struct.unpack('>I', len_bytes)[0]
#             resp_bytes = self.recv_all(resp_len)
#             if not resp_bytes: return self._empty_response()
            
#             response = json.loads(resp_bytes.decode('utf-8'))
#             return response
            
#         except socket.timeout:
#             logging.error("⏰ 推理超时 (60s Timeout).")
#             # 超时后连接可能已脏，建议重置
#             if self.sock: self.sock.close()
#             self.sock = None
#             return self._empty_response()
#         except Exception as e:
#             logging.error(f"💥 通信异常: {e}")
#             if self.sock: self.sock.close()
#             self.sock = None
#             return self._empty_response()

#     def recv_all(self, n):
#         data = b''
#         try:
#             while len(data) < n:
#                 chunk = self.sock.recv(n - len(data))
#                 if not chunk: return None
#                 data += chunk
#             return data
#         except:
#             return None
        
#     def _empty_response(self):
#         # 返回全0动作，但现在 robot_policy_system 会拦截它
#         return {
#             "actions": [ [[0.0] * 8] ], 
#             "trajectory": None
#         }

import socket
import json
import struct
import cv2
import numpy as np
import logging

class TCPClientPolicy:
    def __init__(self, host, port):
        self.host = host
        self.port = int(port)
        self.sock = None
        self.connect()

    def connect(self):
        try:
            if self.sock: self.sock.close()
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1) 
            self.sock.settimeout(60.0) 
            self.sock.connect((self.host, self.port))
            logging.info(f"✅ TCP连接成功: {self.host}:{self.port}")
        except Exception as e:
            logging.error(f"❌ TCP连接失败: {e}")
            self.sock = None

    def infer(self, element):
        if self.sock is None:
            self.connect()
            if self.sock is None: return self._empty_response()

        # 1. 提取 Qpos
        if 'qpos' in element:
            qpos = element['qpos']
        else:
            qpos = element['observation/state'][:7].tolist()

        # 2. 提取图像
        images = []
        if 'observation/wrist_image' in element:
            val = element['observation/wrist_image']
            if isinstance(val, list):
                images = val
            else:
                images = [val]
        else:
            return self._empty_response()
        
        # 3. [优化] 预处理与压缩
        img_bytes_list = []
        img_sizes = []
        
        # 使用高质量 JPEG (95) 或 PNG (无损，但稍慢)
        # 考虑到 224x224 只有 50KB 左右，JPG 95 几乎无损且极快
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95] 
        # 如果你绝对追求像素级无损，可以改用 PNG:
        # encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 1] # PNG

        for img in images:
            # === 关键优化: Client 端 Resize ===
            # 将 720p (1280x720) 缩小到模型需要的 224x224
            # 这样不仅传输极快，而且允许我们使用超高画质压缩
            if img.shape[0] != 224 or img.shape[1] != 224:
                img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            else:
                img_resized = img

            # 编码
            if encode_param[0] == int(cv2.IMWRITE_PNG_COMPRESSION):
                _, img_encoded = cv2.imencode('.png', img_resized, encode_param)
            else:
                _, img_encoded = cv2.imencode('.jpg', img_resized, encode_param)
                
            b = img_encoded.tobytes()
            img_bytes_list.append(b)
            img_sizes.append(len(b))
            
        full_img_payload = b''.join(img_bytes_list)
        
        # 4. 构造 Header
        header = {
            "qpos": qpos, 
            "img_sizes": img_sizes 
        }
        header_bytes = json.dumps(header).encode('utf-8')
        
        try:
            # 5. 发送
            self.sock.sendall(struct.pack('>I', len(header_bytes)))
            self.sock.sendall(header_bytes)
            self.sock.sendall(full_img_payload)
            
            # 6. 接收
            len_bytes = self.recv_all(4)
            if not len_bytes: 
                logging.warning("⚠️ Server closed connection (EOF).")
                self.sock.close() 
                self.sock = None
                return self._empty_response()
            
            resp_len = struct.unpack('>I', len_bytes)[0]
            resp_bytes = self.recv_all(resp_len)
            if not resp_bytes: return self._empty_response()
            
            response = json.loads(resp_bytes.decode('utf-8'))
            return response
            
        except socket.timeout:
            logging.error("⏰ 推理超时 (60s Timeout).")
            if self.sock: self.sock.close()
            self.sock = None
            return self._empty_response()
        except Exception as e:
            logging.error(f"💥 通信异常: {e}")
            if self.sock: self.sock.close()
            self.sock = None
            return self._empty_response()

    def recv_all(self, n):
        data = b''
        try:
            while len(data) < n:
                chunk = self.sock.recv(n - len(data))
                if not chunk: return None
                data += chunk
            return data
        except:
            return None
        
    def _empty_response(self):
        return {
            "actions": [ [[0.0] * 8] ], 
            "trajectory": None
        }