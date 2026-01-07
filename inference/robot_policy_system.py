# import sys
# import os
# import time
# import logging
# import cv2
# import numpy as np
# import math
# import threading
# from collections import deque
# from common.constants import ActionSpace
# from robots.franky_env import FrankyEnv
# from robots.robot_param import RobotParam
# from systems.tcp_client import TCPClientPolicy 

# # 引入你的相机库
# from cameras.realsense_env import RealSenseEnv

# class ImageRecorder(threading.Thread):
#     def __init__(self, camera, buffer_size=16):
#         super().__init__()
#         self.camera = camera
#         self.buffer_size = buffer_size
#         self.running = False
#         self.lock = threading.Lock()
        
#         # 两个 Buffer：
#         # 1. raw_buffer: 存原始图，用于显示
#         # 2. video_buffer: 存处理后的 tensor/numpy，用于推理
#         self.latest_frame = None
        
#         # 这里的 buffer 只要存 numpy 数组即可，不需要存 Tensor，
#         # 转换 Tensor 的工作交给 Server 端，或者在发送前做，减少传输压力
#         # 但为了配合你的 Server 逻辑，我们这里只存原始 BGR 图像
#         self.frame_buffer = deque(maxlen=buffer_size) 
#         self.stop_event = threading.Event()

#     def run(self):
#         self.running = True
#         self.camera.start_monitoring()
#         logging.info("[ImageRecorder] Background thread started.")
        
#         while not self.stop_event.is_set():
#             # 获取最新帧 (这是轻量级操作)
#             data = self.camera.get_latest_frame()
#             if data is not None:
#                 img = data['bgr']
                
#                 with self.lock:
#                     self.latest_frame = img.copy()
#                     # 存入 Buffer
#                     self.frame_buffer.append(img)
                
#                 # 实时显示 (在这里显示最流畅)
#                 cv2.imshow("Wrist View (Real-time)", img)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     self.stop_event.set()
            
#             # 保持约 30Hz 的采样率 (根据你训练数据的帧率调整)
#             # 如果你训练是 10Hz，这里改成 time.sleep(0.1)
#             time.sleep(0.033) 
        
#         cv2.destroyAllWindows()
#         logging.info("[ImageRecorder] Stopped.")

#     def get_inference_input(self):
#         """
#         获取用于推理的 snapshot。
#         如果 Buffer 还没满，就用第一帧复制填充 (Padding)。
#         """
#         with self.lock:
#             if len(self.frame_buffer) == 0:
#                 return None, None
            
#             current_img = self.latest_frame.copy()
            
#             # 拿到 Buffer 的快照
#             frames_snapshot = list(self.frame_buffer)
        
#         # 策略：如果不够 16 帧，用第一帧补齐头部 (Padding Head)
#         # 这样保证时序相对关系是正确的
#         while len(frames_snapshot) < self.buffer_size:
#             frames_snapshot.insert(0, frames_snapshot[0])
            
#         # 注意：你需要确认 Server 端期待的是什么格式。
#         # 你之前的代码只发了 "wrist_image" (一张图) 给 Server，
#         # Server 自己在那边 append buffer。
#         # 
#         # === 关键纠正 ===
#         # 既然 Server 端有 Video Buffer 逻辑，但由于通信频率太低导致 Buffer 失效。
#         # 最好的办法是：**客户端自己维护 Buffer，发给 Server 时只发最新一张图**
#         # 但是！告诉 Server："请清空你的 Buffer，只用这一张图当静态图处理"
#         # 或者！**我们在客户端维护好 Buffer，一次性把 16 张图发过去？**
#         # 
#         # 鉴于改动通信协议太麻烦，我们采用【方案 B】：
#         # 还是只发最新一张图，但在 Server 端我们已经修改了逻辑 (每次清空 Buffer 填满当前帧)。
#         # 所以这里 ImageRecorder 的主要作用是：确保 robot_policy_system 拿到的
#         # "latest_frame" 是真的 "latest" (刚刚发生的)，而不是 5 秒前的缓存。
        
#         return current_img

#     def stop(self):
#         self.stop_event.set()
#         self.join()

# class RobotPolicySystem:
#     def __init__(self, action_space: ActionSpace = ActionSpace.JOINT_ANGLES, ip: str = "127.0.0.1", port: int = 6000):
#         self.action_space = action_space
        
#         # Robot
#         self.robot_env = FrankyEnv(
#             action_space=action_space, 
#             inference_mode=True, 
#             robot_param=RobotParam(np.array([ 0.0, 0.0, -math.pi / 2]), np.array([ 0.53433071, 0.52905707, 0.00440881]))
#         )
        
#         # Client
#         logging.info(f"Connecting to {ip}:{port}...")
#         self.client = TCPClientPolicy(host=ip, port=port)
#         logging.info("Connected.")
        
#         # Camera & Recorder
#         self.wrist_camera = RealSenseEnv(camera_name="wrist_image", serial_number="342222072092", width=1280, height=720)
#         # 启动后台采集线程
#         self.recorder = ImageRecorder(self.wrist_camera, buffer_size=16)
        
#         self.gripper_status = {"current_state": 0}
#         self.stop_evaluation = threading.Event()

#     def run(self, task_name: str = "default_task"):
#         # 启动后台采集
#         self.recorder.start()
        
#         logging.info("Waiting 2.0s for warmup...")
#         time.sleep(2.0)
        
#         # 参数设置
#         EXECUTION_HORIZON = 15  # 信任模型，做完 64 步
#         MAX_STEP_RAD = 0.05     # 限幅
#         last_executed_joints = None
        
#         logging.info("Starting inference loop...")

#         try:
#             while not self.stop_evaluation.is_set():
#                 if not self.recorder.is_alive():
#                     break

#                 t0 = time.time()
                
#                 # 1. 从后台线程拿【最新鲜】的一张图
#                 # 即使主线程卡了 5 秒，这里拿到的也是 0.001 秒前相机刚拍到的
#                 wrist_image = self.recorder.get_inference_input()
                
#                 if wrist_image is None:
#                     time.sleep(0.01)
#                     continue

#                 # 2. 获取机器人状态
#                 joint_angles = self.robot_env.get_position(action_space=ActionSpace.JOINT_ANGLES)
#                 gripper_width = self.robot_env.get_gripper_width()
#                 eef_pose = self.robot_env.get_position(action_space=ActionSpace.EEF_POSE)
                
#                 qpos_8d = list(joint_angles) + [float(gripper_width)]
#                 state = np.concatenate([eef_pose, [gripper_width]])
                
#                 # 3. 发送请求
#                 # 注意：我们在 Server 端已经改成了 "收到一张图 -> 复制填满 Buffer" 的静态图策略
#                 # 这配合这里 "获取最新鲜的一张图" 是目前最稳健的组合
#                 element = {
#                     "observation/agentview_image": np.zeros_like(wrist_image), 
#                     "observation/wrist_image": wrist_image,
#                     "observation/state": state,
#                     "qpos": qpos_8d, 
#                     "prompt": task_name,
#                 }

#                 # 4. 推理 (Blocking 2.5s)
#                 inference_results = self.client.infer(element)
                
#                 if inference_results and "actions" in inference_results:
#                     new_actions = inference_results["actions"][0]
                    
#                     if not isinstance(new_actions, list) or len(new_actions) == 0:
#                         continue

#                     # 执行 64 步
#                     actions_to_execute = new_actions[:EXECUTION_HORIZON]
                    
#                     print(f"  >>> Executing chunk ({len(actions_to_execute)} steps)...")

#                     for action in actions_to_execute:
#                         if not isinstance(action, (list, tuple, np.ndarray)): continue
                        
#                         # 数据处理
#                         action_np = np.array(action, dtype=np.float64)
#                         if np.all(action_np == 0) or np.isnan(action_np).any(): break
                        
#                         target_joints = action_np[:-1]
#                         gripper_val = action_np[-1]

#                         # 平滑限幅
#                         if last_executed_joints is not None:
#                             diff = np.clip(target_joints - last_executed_joints, -MAX_STEP_RAD, MAX_STEP_RAD)
#                             target_joints = last_executed_joints + diff
                        
#                         last_executed_joints = target_joints.copy()

#                         # 执行
#                         t_step_start = time.time()
#                         self.robot_env.step(target_joints, asynchronous=True)
                        
#                         # 夹爪
#                         if gripper_val > 0.06 and self.gripper_status["current_state"] != -1:
#                              self.robot_env.open_gripper(asynchronous=True)
#                              self.gripper_status["current_state"] = -1
#                         elif gripper_val < 0.02 and self.gripper_status["current_state"] != 1:
#                              self.robot_env.close_gripper(asynchronous=True)
#                              self.gripper_status["current_state"] = 1
                        
#                         # 控频 25Hz
#                         remain = 0.04 - (time.time() - t_step_start)
#                         if remain > 0: time.sleep(remain)

#                 latency = (time.time() - t0) * 1000
#                 print(f"\rLoop Latency: {latency:.1f}ms", end="")

#         except KeyboardInterrupt:
#             pass
#         finally:
#             self.stop()

#     def stop(self):
#         self.stop_evaluation.set()
#         self.recorder.stop()
#         time.sleep(0.5)
#         logging.info("System stopped.")

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     system = RobotPolicySystem(ip="127.0.0.1", port=6000)
#     system.run(task_name="pick up the paper cup")

import sys
import os
import time
import logging
import cv2
import numpy as np
import math
import threading
from collections import deque
from common.constants import ActionSpace
from robots.franky_env import FrankyEnv
from robots.robot_param import RobotParam
from systems.tcp_client import TCPClientPolicy 
from cameras.realsense_env import RealSenseEnv

# 配置日志格式，方便观察
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageRecorder(threading.Thread):
    def __init__(self, camera, buffer_size=16):
        super().__init__()
        self.camera = camera
        self.buffer_size = buffer_size
        self.running = False
        self.lock = threading.Lock()
        
        self.latest_frame = None
        self.frame_buffer = deque(maxlen=buffer_size) 
        self.stop_event = threading.Event()

    def run(self):
        self.running = True
        self.camera.start_monitoring()
        logging.info("[ImageRecorder] Background thread started.")
        
        while not self.stop_event.is_set():
            data = self.camera.get_latest_frame()
            if data is not None:
                img = data['bgr']
                with self.lock:
                    self.latest_frame = img.copy()
                    self.frame_buffer.append(img)
                
                # 实时显示，按 'q' 退出
                cv2.imshow("Wrist View (Real-time)", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_event.set()
            
            # 保持约 30Hz 的采样率
            time.sleep(0.033) 
        
        cv2.destroyAllWindows()
        logging.info("[ImageRecorder] Stopped.")

    def get_sequence_input(self):
        """
        获取过去 16 帧的完整序列。
        如果不足 16 帧，用第一帧填充头部 (Padding Head)。
        """
        with self.lock:
            if len(self.frame_buffer) == 0:
                return None
            
            # 复制一份当前 buffer
            frames_snapshot = list(self.frame_buffer)
        
        # 头部补齐
        while len(frames_snapshot) < self.buffer_size:
            frames_snapshot.insert(0, frames_snapshot[0])
            
        return frames_snapshot

    def stop(self):
        self.stop_event.set()
        self.join()

class RobotPolicySystem:
    def __init__(self, action_space: ActionSpace = ActionSpace.JOINT_ANGLES, ip: str = "127.0.0.1", port: int = 6000):
        self.action_space = action_space
        
        # 初始化机器人
        # 注意：inference_mode=True 通常意味着机器人动作会更快、更直接
        self.robot_env = FrankyEnv(
            action_space=action_space, 
            inference_mode=True, 
            robot_param=RobotParam(np.array([ 0.0, 0.0, -math.pi / 2]), np.array([ 0.53433071, 0.52905707, 0.00440881]))
        )
        
        logging.info(f"Connecting to {ip}:{port}...")
        self.client = TCPClientPolicy(host=ip, port=port)
        logging.info("Connected.")
        
        # 初始化相机
        self.wrist_camera = RealSenseEnv(camera_name="wrist_image", serial_number="342222072092", width=1280, height=720)
        self.recorder = ImageRecorder(self.wrist_camera, buffer_size=16)
        
        # 夹爪状态记录: 0=未知, 1=闭合, -1=张开
        self.gripper_status = {"current_state": 0}
        self.stop_evaluation = threading.Event()

    def run(self, task_name: str = "default_task"):
        self.recorder.start()
        logging.info("Waiting 2.0s for warmup...")
        time.sleep(2.0)
        
        EXECUTION_HORIZON = 15  # 每次推理执行 15 步 (约 0.6s)
        MAX_STEP_RAD = 0.08     # 关节动作限幅
        last_executed_joints = None
        
        # 🚨 关键参数：夹爪动作阈值 (基于 compute_stats 的 Mean=0.0616)
        GRIPPER_THRESHOLD = 0.06 
        
        logging.info(f"Starting inference loop... (Gripper Threshold: {GRIPPER_THRESHOLD})")

        try:
            while not self.stop_evaluation.is_set():
                if not self.recorder.is_alive(): break

                t0 = time.time()
                
                # 1. 获取图像序列
                wrist_images = self.recorder.get_sequence_input()
                if wrist_images is None:
                    time.sleep(0.01)
                    continue

                # 2. 获取机器人状态
                joint_angles = self.robot_env.get_position(action_space=ActionSpace.JOINT_ANGLES)
                gripper_width = self.robot_env.get_gripper_width()
                eef_pose = self.robot_env.get_position(action_space=ActionSpace.EEF_POSE)
                
                # 拼装数据
                qpos_8d = list(joint_angles) + [float(gripper_width)]
                state = np.concatenate([eef_pose, [gripper_width]])
                
                element = {
                    "observation/wrist_image": wrist_images, # List[np.array]
                    "observation/state": state,
                    "qpos": qpos_8d, 
                    "prompt": task_name,
                }

                # 3. 发送推理请求
                inference_results = self.client.infer(element)
                
                if inference_results and "actions" in inference_results:
                    new_actions = inference_results["actions"][0]
                    if not isinstance(new_actions, list) or len(new_actions) == 0: continue

                    # 截取前 15 步执行 (Receding Horizon Control)
                    actions_to_execute = new_actions[:EXECUTION_HORIZON]
                    print(f"  >>> Executing chunk ({len(actions_to_execute)} steps)...")

                    for action in actions_to_execute:
                        if not isinstance(action, (list, tuple, np.ndarray)): continue
                        
                        action_np = np.array(action, dtype=np.float64)
                        if np.all(action_np == 0) or np.isnan(action_np).any(): break
                        
                        target_joints = action_np[:-1]
                        gripper_val = action_np[-1] # 这是物理值 (约 0.04 ~ 0.08)

                        # 平滑限幅
                        if last_executed_joints is not None:
                            diff = np.clip(target_joints - last_executed_joints, -MAX_STEP_RAD, MAX_STEP_RAD)
                            target_joints = last_executed_joints + diff
                        
                        last_executed_joints = target_joints.copy()

                        # 执行关节运动
                        t_step_start = time.time()
                        self.robot_env.step(target_joints, asynchronous=True)
                        
                        # =========================================================
                        # 🚨 [修复] 夹爪控制逻辑
                        # =========================================================
                        # 逻辑：大于阈值 -> 开，小于阈值 -> 关
                        
                        # Case 1: 需要张开
                        if gripper_val > GRIPPER_THRESHOLD:
                            # 只有当前不是“张开”状态时才发送命令，避免重复发送
                            if self.gripper_status["current_state"] != -1:
                                logging.info(f"👐 [Gripper] OPEN detected ({gripper_val:.4f} > {GRIPPER_THRESHOLD})")
                                self.robot_env.open_gripper(asynchronous=True)
                                self.gripper_status["current_state"] = -1
                        
                        # Case 2: 需要闭合
                        elif gripper_val < GRIPPER_THRESHOLD:
                            # 只有当前不是“闭合”状态时才发送命令
                            if self.gripper_status["current_state"] != 1:
                                logging.info(f"✊ [Gripper] CLOSE detected ({gripper_val:.4f} < {GRIPPER_THRESHOLD})")
                                self.robot_env.close_gripper(asynchronous=True)
                                self.gripper_status["current_state"] = 1
                        # =========================================================
                        
                        # 控频 (40Hz)
                        remain = 0.025 - (time.time() - t_step_start)
                        if remain > 0: time.sleep(remain)

                latency = (time.time() - t0) * 1000
                print(f"\rLoop Latency: {latency:.1f}ms", end="")

        except KeyboardInterrupt:
            logging.info("Keyboard Interrupt received.")
        except Exception as e:
            logging.error(f"Runtime Error: {e}")
        finally:
            self.stop()

    def stop(self):
        self.stop_evaluation.set()
        if self.recorder.is_alive():
            self.recorder.stop()
        time.sleep(0.5)
        logging.info("System stopped.")

if __name__ == "__main__":
    # 确保这里的任务名称和训练时完全一致
    system = RobotPolicySystem(ip="127.0.0.1", port=6000)
    system.run(task_name="pick up the orange ball")