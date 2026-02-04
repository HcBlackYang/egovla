# rdt 单视角项目
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

#         # 1. 提取 Qpos
#         if 'qpos' in element:
#             qpos = element['qpos']
#         else:
#             qpos = element['observation/state'][:7].tolist()

#         # 2. 提取图像
#         images = []
#         if 'observation/wrist_image' in element:
#             val = element['observation/wrist_image']
#             if isinstance(val, list):
#                 images = val
#             else:
#                 images = [val]
#         else:
#             return self._empty_response()
        
#         # 3. [优化] 预处理与压缩
#         img_bytes_list = []
#         img_sizes = []
        
#         # 使用高质量 JPEG (95) 或 PNG (无损，但稍慢)
#         # 考虑到 224x224 只有 50KB 左右，JPG 95 几乎无损且极快
#         encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95] 
#         # 如果你绝对追求像素级无损，可以改用 PNG:
#         # encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 1] # PNG

#         for img in images:
#             # === 关键优化: Client 端 Resize ===
#             # 将 720p (1280x720) 缩小到模型需要的 224x224
#             # 这样不仅传输极快，而且允许我们使用超高画质压缩
#             if img.shape[0] != 224 or img.shape[1] != 224:
#                 img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
#             else:
#                 img_resized = img

#             # 编码
#             if encode_param[0] == int(cv2.IMWRITE_PNG_COMPRESSION):
#                 _, img_encoded = cv2.imencode('.png', img_resized, encode_param)
#             else:
#                 _, img_encoded = cv2.imencode('.jpg', img_resized, encode_param)
                
#             b = img_encoded.tobytes()
#             img_bytes_list.append(b)
#             img_sizes.append(len(b))
            
#         full_img_payload = b''.join(img_bytes_list)
        
#         # 4. 构造 Header
#         header = {
#             "qpos": qpos, 
#             "img_sizes": img_sizes 
#         }
#         header_bytes = json.dumps(header).encode('utf-8')
        
#         try:
#             # 5. 发送
#             self.sock.sendall(struct.pack('>I', len(header_bytes)))
#             self.sock.sendall(header_bytes)
#             self.sock.sendall(full_img_payload)
            
#             # 6. 接收
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
#         return {
#             "actions": [ [[0.0] * 8] ], 
#             "trajectory": None
#         }




# #ego项目
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
#             self.sock.settimeout(60.0) 
#             self.sock.connect((self.host, self.port))
#             logging.info(f"✅ TCP连接成功: {self.host}:{self.port}")
#         except Exception as e:
#             logging.error(f"❌ TCP连接失败: {e}")
#             self.sock = None

#     def infer(self, element):
#         """
#         发送推理请求
#         element: {
#             "qpos": List[float] (8维: 7关节+1夹爪),
#             "observation/wrist_image": np.array (图像),
#             "prompt": str (任务文本, 例如 "pick up the orange ball")
#         }
#         """
#         if self.sock is None:
#             self.connect()
#             if self.sock is None: return self._empty_response()

#         # 1. 提取 Qpos
#         if 'qpos' in element:
#             qpos = element['qpos']
#         else:
#             qpos = element['observation/state'][:8].tolist() # 7关节 + 1夹爪

#         # 2. 提取 Prompt (修复丢失问题)
#         prompt_text = element.get('prompt', "")

#         # 3. 提取图像 (强制只处理 wrist_image)
#         # 注意：这里我们只处理一张手腕图，因为你指定了只测试手腕视角
#         images = []
#         if 'observation/wrist_image' in element:
#             val = element['observation/wrist_image']
#             if isinstance(val, list):
#                 # 如果传入的是列表，取最后一帧（最新帧）
#                 images = [val[-1]] 
#             else:
#                 images = [val]
#         else:
#             logging.warning("No wrist image found!")
#             return self._empty_response()
        
#         # 4. 图像压缩 (Resize + JPEG)
#         img_bytes_list = []
#         img_sizes = []
#         encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95] 

#         for img in images:
#             # Resize 到 224x224 以匹配 SigLIP/RDT 输入，减少带宽
#             if img.shape[0] != 224 or img.shape[1] != 224:
#                 img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
#             else:
#                 img_resized = img

#             _, img_encoded = cv2.imencode('.jpg', img_resized, encode_param)
#             b = img_encoded.tobytes()
#             img_bytes_list.append(b)
#             img_sizes.append(len(b))
            
#         full_img_payload = b''.join(img_bytes_list)
        
#         # 5. 构造 Header (加入 prompt)
#         header = {
#             "qpos": qpos, 
#             "img_sizes": img_sizes,
#             "prompt": prompt_text 
#         }
#         header_bytes = json.dumps(header).encode('utf-8')
        
#         try:
#             # 6. 发送数据包: [Header Len] + [Header] + [Image Bytes]
#             self.sock.sendall(struct.pack('>I', len(header_bytes)))
#             self.sock.sendall(header_bytes)
#             self.sock.sendall(full_img_payload)
            
#             # 7. 接收响应
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
#         # 返回 8 维的全 0 动作
#         return {
#             "actions": [ [[0.0] * 8] ], 
#             "trajectory": None
#         }



#ego 双摄项目
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
        """
        element: {
            "qpos": List[float],
            "observation/main_image": [np.array],
            "observation/wrist_image": [np.array],
            "prompt": str
        }
        """
        if self.sock is None:
            self.connect()
            if self.sock is None: return self._empty_response()

        # 1. 提取 Qpos
        if 'qpos' in element:
            qpos = element['qpos']
        else:
            qpos = element['observation/state'][:8].tolist()

        # 2. 提取 Prompt
        prompt_text = element.get('prompt', "")

        # 3. 提取图像 (严格顺序: Main -> Wrist)
        images_to_send = []
        
        # (A) Main Image
        if 'observation/main_image' in element:
            val = element['observation/main_image']
            img = val[-1] if isinstance(val, list) else val
            images_to_send.append(img)
        else:
            logging.warning("⚠️ No main_image found!")
            # 补一个空图防止协议错位 (虽然实际不应发生)
            images_to_send.append(np.zeros((224,224,3), dtype=np.uint8))

        # (B) Wrist Image
        if 'observation/wrist_image' in element:
            val = element['observation/wrist_image']
            img = val[-1] if isinstance(val, list) else val
            images_to_send.append(img)
        else:
            logging.warning("⚠️ No wrist_image found!")
            images_to_send.append(np.zeros((224,224,3), dtype=np.uint8))

        # 4. 压缩与打包
        img_bytes_list = []
        img_sizes = []
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95] 

        for img in images_to_send:
            # Resize 优化带宽
            if img.shape[0] != 224 or img.shape[1] != 224:
                img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            else:
                img_resized = img

            _, img_encoded = cv2.imencode('.jpg', img_resized, encode_param)
            b = img_encoded.tobytes()
            img_bytes_list.append(b)
            img_sizes.append(len(b))
            
        full_img_payload = b''.join(img_bytes_list)
        
        # 5. Header
        header = {
            "qpos": qpos, 
            "img_sizes": img_sizes, # [size_main, size_wrist]
            "prompt": prompt_text 
        }
        header_bytes = json.dumps(header).encode('utf-8')
        
        try:
            # Send: [Len] + [Header] + [Payload]
            self.sock.sendall(struct.pack('>I', len(header_bytes)))
            self.sock.sendall(header_bytes)
            self.sock.sendall(full_img_payload)
            
            # Recv
            len_bytes = self.recv_all(4)
            if not len_bytes: 
                self.sock.close(); self.sock = None
                return self._empty_response()
            
            resp_len = struct.unpack('>I', len_bytes)[0]
            resp_bytes = self.recv_all(resp_len)
            if not resp_bytes: return self._empty_response()
            
            return json.loads(resp_bytes.decode('utf-8'))
            
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
        return {"actions": [ [[0.0] * 8] ]}




# # rdt 双视角项目
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
#             self.sock.settimeout(60.0) # 防止网络波动或推理过慢导致断连
#             self.sock.connect((self.host, self.port))
#             logging.info(f"✅ TCP连接成功: {self.host}:{self.port}")
#         except Exception as e:
#             logging.error(f"❌ TCP连接失败: {e}")
#             self.sock = None

#     def infer(self, element):
#         """
#         发送推理请求
#         element: {
#             "qpos": List[float] (8维),
#             "observation/head_image": np.array (第三视角),
#             "observation/wrist_image": np.array (第一视角),
#             "prompt": str
#         }
#         """
#         if self.sock is None:
#             self.connect()
#             if self.sock is None: return self._empty_response()

#         # 1. 提取 Qpos
#         if 'qpos' in element:
#             qpos = element['qpos']
#         else:
#             qpos = element['observation/state'][:8].tolist() 

#         # 2. 提取 Prompt
#         prompt_text = element.get('prompt', "")

#         # 3. 提取图像 (同时提取 Head 和 Wrist)
#         # 约定发送顺序: [Head, Wrist]
#         images = []
        
#         # (1) 第三视角 (Head)
#         if 'observation/head_image' in element:
#             val = element['observation/head_image']
#             img = val[-1] if isinstance(val, list) else val
#             images.append(img)
#         else:
#             logging.warning("⚠️ No head_image found!")
#             # 必须保证有图，否则 Server 解析顺序会乱，这里可以用全黑图代替，或者直接返回空
#             return self._empty_response()
            
#         # (2) 第一视角 (Wrist)
#         if 'observation/wrist_image' in element:
#             val = element['observation/wrist_image']
#             img = val[-1] if isinstance(val, list) else val
#             images.append(img)
#         else:
#             logging.warning("⚠️ No wrist_image found!")
#             return self._empty_response()

#         # 4. 图像压缩 (Resize + JPEG)
#         img_bytes_list = []
#         img_sizes = []
#         encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95] 

#         for img in images:
#             # Resize 到 224x224 以减少带宽
#             if img.shape[0] != 224 or img.shape[1] != 224:
#                 img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
#             else:
#                 img_resized = img

#             _, img_encoded = cv2.imencode('.jpg', img_resized, encode_param)
#             b = img_encoded.tobytes()
#             img_bytes_list.append(b)
#             img_sizes.append(len(b))
            
#         full_img_payload = b''.join(img_bytes_list)
        
#         # 5. 构造 Header
#         header = {
#             "qpos": qpos, 
#             "img_sizes": img_sizes,
#             "prompt": prompt_text 
#         }
#         header_bytes = json.dumps(header).encode('utf-8')
        
#         try:
#             # 6. 发送: [Len] + [Header] + [Images]
#             self.sock.sendall(struct.pack('>I', len(header_bytes)))
#             self.sock.sendall(header_bytes)
#             self.sock.sendall(full_img_payload)
            
#             # 7. 接收响应
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
#             logging.error("⏰ 推理超时.")
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
#         return {
#             "actions": [ [[0.0] * 8] ], 
#             "trajectory": None
#         }