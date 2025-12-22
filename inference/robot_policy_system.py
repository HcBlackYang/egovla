# import sys
# import os


# from robots.franky_env import FrankyEnv
# from controllers.gello_env import GelloEnv
# from controllers.spacemouse_env import SpaceMouseEnv
# from cameras.realsense_env import RealSenseEnv
# from cameras.usb_env import USBEnv

# from common.constants import ActionSpace
# import time
# from pathlib import Path
# import logging
# # from systems.robot_policy_utils import WebsocketClientPolicy # 不再使用
# from systems.tcp_client import TCPClientPolicy # 2. 使用 TCP 客户端
# import cv2
# import numpy as np
# from cameras.camera_param import CameraParam
# from robots.robot_param import RobotParam
# import math
# import threading

# class RobotPolicySystem:
#     def __init__(self, action_space: ActionSpace = ActionSpace.JOINT_ANGLES, ip: str = "127.0.0.1", port: int = 6000, 
#                  action_only_mode: bool = False, calibration: bool=True):
#         # 初始化机器人环境
#         self.action_space = action_space
#         self.action_only_mode = action_only_mode

#         # 初始化 Franka 机器人
#         # 注意：inference_mode=True 通常意味着更灵敏的控制
#         self.robot_env = FrankyEnv(
#             action_space=action_space, 
#             inference_mode=True, 
#             robot_param=RobotParam(np.array([ 0.0, 0.0, -math.pi / 2]), np.array([ 0.53433071, 0.52905707, 0.00440881]))
#         )
        
#         if self.action_space not in [ActionSpace.EEF_VELOCITY, ActionSpace.JOINT_ANGLES]:
#             raise NotImplementedError(f"Action space '{self.action_space}' is not supported.")
        
#         logging.info(f"Trying to connect to policy server at {ip}:{port}...")
        
#         # 3. 初始化 TCP 客户端
#         self.client = TCPClientPolicy(
#             host= ip,
#             port= port
#         )
#         logging.info(f"Connected to policy server at {ip}:{port}.")
        
#         # 初始化摄像头
#         # 请根据实际情况确认 serial_number 是否正确
#         self.main_camera = RealSenseEnv(camera_name="main_image", serial_number="339322073638", width=1280, height=720,
#                                         camera_param=CameraParam(intrinsic_matrix = np.array([[908.1308, 0, 655.7268], [0, 910.0818, 395.8856], [0, 0, 1]], dtype=np.float32),
#                                                                  distortion_coeffs = np.array([0.1068, -0.2123, -0.0092, 0.0000, 0.0000], dtype=np.float32)))
#         self.wrist_camera = RealSenseEnv(camera_name="wrist_image", serial_number="342222072092", width=1280, height=720)
        
#         self.top_camera = USBEnv(camera_name="top_image", serial_number="12", width=1920, height=1080, exposure=100,
#                         camera_param=CameraParam(np.array([[1158.0, 0, 999.9484], [0, 1159.9, 584.2338], [0, 0, 1]], dtype=np.float32), np.array([0.0412, -0.0509, 0.0000, 0.0000, 0.0000], dtype=np.float32))
#                     )
#         if calibration:
#             try:
#                 self.main_camera.calib_camera()
#                 self.top_camera.calib_camera()
#             except Exception as e:
#                 logging.warning(f"Camera calibration failed: {e}")

#         self.gripper_status = {
#             "current_state": 0,
#             "target_state": 0 
#         }
#         self.stop_evaluation = threading.Event()
#         self.all_action_and_traj = []
#         self.all_action_and_traj_lock = threading.Lock()

#     def reset_for_collection(self):
#         """重置机器人到随机位置，用于数据收集"""
#         self.robot_env.reset()
#         # 注意：如果是 JOINT_ANGLES 模式，这里的 step 需要传入 7 维
#         # 这里为了安全，先注释掉，避免维度冲突
#         # action = np.array([0,0,-0.05,0,0,0])
#         # self.robot_env.step(action, asynchronous=False)
#         return True

#     def run(self, show_image: bool = False, task_name: str = "default_task"):
#         # 1. 启动摄像头
#         # self.main_camera.start_monitoring()
#         self.wrist_camera.start_monitoring()
#         # self.top_camera.start_monitoring()
        
#         # === 修复 1: 强制等待摄像头预热，防止 NoneType 错误 ===
#         logging.info("Waiting 2.0s for cameras to warm up...")
#         time.sleep(2.0)
        
#         # === 新增修复代码：强制等待图像变亮 ===
#         for i in range(50): # 最多尝试50次 (约2-3秒)
#             time.sleep(0.1)
            
#             # 获取帧
#             wrist_data = self.wrist_camera.get_latest_frame()
#             main_data = self.main_camera.get_latest_frame()
            
#             if wrist_data is None or main_data is None:
#                 continue
                
#             wrist_img = wrist_data['bgr']
#             main_img = main_data['bgr']
            
#             # 检查是否黑屏：计算像素平均值
#             # 如果平均亮度低于 10 (255阶)，认为是黑屏或没初始化好
#             if np.mean(wrist_img) > 10 and np.mean(main_img) > 10:
#                 logging.info(f"✅ Cameras are ready! (Avg Brightness: {np.mean(wrist_img):.1f})")
#                 break
#             else:
#                 if i % 10 == 0:
#                     logging.warning(f"⚠️ Camera frames are too dark/black (Avg: {np.mean(wrist_img):.1f}), waiting...")
#         # ======================================

#         self.gripper_status = {
#             "current_state": 0,
#             "target_state": 0 
#         }
#         self.stop_evaluation.clear()
#         all_action_and_traj = []
        
#         logging.info("Starting inference loop...")
        
#         while not self.stop_evaluation.is_set():
#             t0 = time.time()
            
#             # === 修复 3: 安全读取图像 (判空) ===
#             main_frame_data = self.main_camera.get_latest_frame()
#             wrist_frame_data = self.wrist_camera.get_latest_frame()
#             top_frame_data = self.top_camera.get_latest_frame()

#             if main_frame_data is None or wrist_frame_data is None:
#                 # 只有当关键摄像头还没准备好时才跳过
#                 logging.debug("Camera frames not ready, waiting...")
#                 time.sleep(0.01)
#                 continue
            
#             main_image = main_frame_data['bgr']
#             wrist_image = wrist_frame_data['bgr']
#             top_image = top_frame_data['bgr'] if top_frame_data is not None else None

#             # 获取机器人状态
#             joint_angles = self.robot_env.get_position(action_space=ActionSpace.JOINT_ANGLES)
#             gripper_width = self.robot_env.get_gripper_width()
#             eef_pose = self.robot_env.get_position(action_space=ActionSpace.EEF_POSE)
#             state = np.concatenate([eef_pose, [gripper_width]])
            
#             # 构造推理请求
#             if self.action_only_mode:
#                 state_trajectory = eef_pose[:3]
#                 element = {
#                     "observation/image": main_image,
#                     "observation/wrist_image": wrist_image,
#                     "observation/state": state,
#                     "prompt": task_name,
#                 }
#             else:
#                 state_trajectory = self.robot_env.robot_param.transform_to_world(np.array([eef_pose[:3]]))[0]
#                 # element = {
#                 #     "observation/image": main_image,
#                 #     "observation/wrist_image": wrist_image,
#                 #     "observation/state": state,
#                 #     "qpos": joint_angles.tolist(), # === 关键：传入关节角度 ===
#                 #     "observation/state_trajectory": state_trajectory,
#                 #     "prompt": task_name,
#                 # }

#                 qpos_8d = np.concatenate([joint_angles, [gripper_width]])
                
#                 element = {
#                     "observation/image": main_image,
#                     "observation/wrist_image": wrist_image,
#                     "observation/state": state,
#                     "qpos": qpos_8d.tolist(), # <--- 这里发送 8 维
#                     "observation/state_trajectory": state_trajectory,
#                     "prompt": task_name,
#                 }


#             # 执行推理
#             inference_results = self.client.infer(element)
            
#             # 检查推理结果
#             if inference_results is None:
#                 logging.warning("Inference failed (server returned None), skipping frame.")
#                 time.sleep(0.01)
#                 continue
                
#             actions_chunk = np.array(inference_results["actions"])
            
#             # 记录数据
#             if not self.action_only_mode and "trajectory" in inference_results and inference_results["trajectory"] is not None:
#                 trajectory_chunk = np.array(inference_results["trajectory"])
#             else:
#                 trajectory_chunk = None

#             all_action_and_traj.append({
#                 'actions': actions_chunk.tolist(),
#                 'trajectory': trajectory_chunk.tolist() if trajectory_chunk is not None else None,
#                 'timestamp': time.time(),
#                 'state': state.tolist(),
#                 'state_trajectory': state_trajectory.tolist()
#             }.copy())
            
#             with self.all_action_and_traj_lock:
#                 self.all_action_and_traj = all_action_and_traj

#             # 可视化部分
#             if show_image:
#                 draw_main_image = main_image.copy()
#                 if top_image is not None:
#                     draw_top_image = top_image.copy()
#                     cv2.imshow("Top Camera", draw_top_image)
                
#                 # 如果有轨迹预测，画出来 (这里简化了逻辑，防止维度报错)
#                 # ... visualization logic kept minimal to avoid crash ...

#                 cv2.imshow("Main Camera", draw_main_image)
#                 cv2.imshow("Wrist Camera", wrist_image)
#                 cv2.waitKey(1)

#             # === 修复 4: 动作执行逻辑 ===
#             for action in actions_chunk:
#                 # 检查 Action 维度
#                 # RDT 可能是 7维 (纯关节) 或 8维 (关节+夹爪)
#                 if len(action) == 7:
#                     # 纯关节控制
#                     self.robot_env.step(action, asynchronous=True)
#                     # 此时没有夹爪信息，跳过夹爪逻辑
#                 elif len(action) == 8:
#                     # 关节 + 夹爪
#                     self.robot_env.step(action[:-1], asynchronous=True)
                    
#                     # 夹爪逻辑
#                     gripper_action = action[-1]
#                     if gripper_action > 0.95:
#                         self.gripper_status["target_state"] = 1
#                     elif gripper_action < -0.95:
#                         self.gripper_status["target_state"] = -1
                    
#                     if self.gripper_status["current_state"] != self.gripper_status["target_state"]:
#                         if self.gripper_status["target_state"] == -1:
#                             self.robot_env.open_gripper(asynchronous=True)
#                         else:
#                             self.robot_env.close_gripper(asynchronous=True)
#                         self.gripper_status["current_state"] = self.gripper_status["target_state"]
                
#                 # 控制频率 sleep (根据需要调整，RDT推理本身耗时较大，这里可以设小一点)
#                 time.sleep(0.02) 

#             # 打印延迟信息
#             latency = (time.time() - t0) * 1000
#             print(f"\rLatency: {latency:.1f}ms | Action[0]: {actions_chunk[0][0]:.2f}", end="")

#     def stop(self):
#         self.stop_evaluation.set()
#         time.sleep(0.5)
#         self.robot_env.stop_saving_state()
#         logging.info("Robot policy system stopped.")

# if __name__ == "__main__":
#     logging.basicConfig(
#             level=logging.INFO,
#             format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#             handlers=[
#                 logging.StreamHandler(),
#             ]
#         )
    
#     # === 初始化 ===
#     # 1. 确保建立了 SSH 隧道: ssh -L 6000:localhost:6000 root@GPU_IP -p PORT
#     # 2. IP 设置为 127.0.0.1 (走隧道)
#     # 3. action_space 必须是 JOINT_ANGLES
#     system = RobotPolicySystem(
#         action_space=ActionSpace.JOINT_ANGLES, 
#         ip="127.0.0.1", 
#         port=6000,
#         action_only_mode=False
#     )
    
#     # 运行
#     try:
#         system.run(show_image=True, task_name="pick up the water bottle")
#     except KeyboardInterrupt:
#         system.stop()

import sys
import os
from robots.franky_env import FrankyEnv
from common.constants import ActionSpace
import time
import logging
from systems.tcp_client import TCPClientPolicy 
import cv2
import numpy as np
from robots.robot_param import RobotParam
import math
import threading

class RobotPolicySystem:
    def __init__(self, action_space: ActionSpace = ActionSpace.JOINT_ANGLES, ip: str = "127.0.0.1", port: int = 6000, 
                 action_only_mode: bool = False, calibration: bool=True):
        self.action_space = action_space
        self.action_only_mode = action_only_mode

        # 初始化 Franka 机器人
        # inference_mode=True 通常意味着更灵敏
        self.robot_env = FrankyEnv(
            action_space=action_space, 
            inference_mode=True, 
            robot_param=RobotParam(np.array([ 0.0, 0.0, -math.pi / 2]), np.array([ 0.53433071, 0.52905707, 0.00440881]))
        )
        
        logging.info(f"Trying to connect to policy server at {ip}:{port}...")
        self.client = TCPClientPolicy(host=ip, port=port)
        logging.info(f"Connected to policy server at {ip}:{port}.")
        
        # 这里的 Camera 初始化保留你的原始代码，假设是正确的
        from cameras.realsense_env import RealSenseEnv
        from cameras.usb_env import USBEnv
        from cameras.camera_param import CameraParam
        
        self.main_camera = RealSenseEnv(camera_name="main_image", serial_number="339322073638", width=1280, height=720,
                                        camera_param=CameraParam(intrinsic_matrix = np.array([[908.1308, 0, 655.7268], [0, 910.0818, 395.8856], [0, 0, 1]], dtype=np.float32),
                                                                 distortion_coeffs = np.array([0.1068, -0.2123, -0.0092, 0.0000, 0.0000], dtype=np.float32)))
        self.wrist_camera = RealSenseEnv(camera_name="wrist_image", serial_number="342222072092", width=1280, height=720)
        
        # Top Camera 暂时注释或保留，看你需求
        # self.top_camera = USBEnv(...) 

        self.gripper_status = {"current_state": 0, "target_state": 0}
        self.stop_evaluation = threading.Event()

    def run(self, show_image: bool = False, task_name: str = "default_task"):
        self.wrist_camera.start_monitoring()
        # self.main_camera.start_monitoring() # 如果需要主摄请取消注释
        
        logging.info("Waiting 2.0s for cameras to warm up...")
        time.sleep(2.0)
        
        logging.info("Starting inference loop...")
        
        while not self.stop_evaluation.is_set():
            t0 = time.time()
            
            # 1. 获取图像
            main_frame_data = self.main_camera.get_latest_frame()
            wrist_frame_data = self.wrist_camera.get_latest_frame()

            if main_frame_data is None or wrist_frame_data is None:
                time.sleep(0.01)
                continue
            
            main_image = main_frame_data['bgr']
            wrist_image = wrist_frame_data['bgr']

            # 2. 获取状态
            joint_angles = self.robot_env.get_position(action_space=ActionSpace.JOINT_ANGLES)
            gripper_width = self.robot_env.get_gripper_width()
            eef_pose = self.robot_env.get_position(action_space=ActionSpace.EEF_POSE)
            
            # 拼接 State (8维)
            state = np.concatenate([eef_pose, [gripper_width]])
            
            # 3. 构造请求
            element = {
                "observation/image": main_image,
                "observation/wrist_image": wrist_image,
                "observation/state": state,
                "qpos": joint_angles.tolist(), 
                "prompt": task_name,
            }

            # 4. 推理 (Blocking)
            inference_results = self.client.infer(element)
            if inference_results is None: continue
            
            actions_chunk = np.array(inference_results["actions"])
            
            # 可视化
            if show_image:
                cv2.imshow("Wrist View", wrist_image)
                cv2.waitKey(1)

            # === 5. 执行动作 (核心修改：同步模式) ===
            # 我们强制在这个循环里等待动作完成，绝不积压
            for action in actions_chunk:
                
                # --- 关节控制 ---
                target_joints = action if len(action) == 7 else action[:-1]
                
                # 关键修改：asynchronous=False
                # 这会阻塞直到机械臂到达目标点 (消除幻影移动)
                self.robot_env.step(target_joints, asynchronous=False) 
                
                # --- 夹爪控制 ---
                if len(action) == 8:
                    gripper_val = action[-1]
                    # 简单的阈值逻辑
                    if gripper_val > 0.06: # 张开阈值 (归一化前可能是 0.8)
                         if self.gripper_status["current_state"] != -1:
                             self.robot_env.open_gripper(asynchronous=True) # 夹爪可以异步，不影响主臂
                             self.gripper_status["current_state"] = -1
                    elif gripper_val < 0.02: # 闭合阈值
                         if self.gripper_status["current_state"] != 1:
                             self.robot_env.close_gripper(asynchronous=True)
                             self.gripper_status["current_state"] = 1

            latency = (time.time() - t0) * 1000
            print(f"\rStep Latency: {latency:.1f}ms | Executed Sync.", end="")

    def stop(self):
        self.stop_evaluation.set()
        time.sleep(0.5)
        logging.info("System stopped.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    system = RobotPolicySystem(
        action_space=ActionSpace.JOINT_ANGLES, 
        ip="127.0.0.1", 
        port=6000
    )
    try:
        system.run(show_image=True, task_name="pick up the paper cup")
    except KeyboardInterrupt:
        system.stop()