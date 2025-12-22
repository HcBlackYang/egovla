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
        current_action_chunk = []
        last_inference_time = 0
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
                # "observation/image": main_image,
                "observation/wrist_image": wrist_image,
                "observation/state": state,
                "qpos": joint_angles.tolist(), 
                "prompt": task_name,
            }

            # 4. 推理 (Blocking)
            inference_results = self.client.infer(element)
            # if inference_results is None: continue
            if inference_results:
                new_actions = inference_results["actions"][0] # 拿到 [16, 8] 的 list
                
                # --- 阶段 B: 执行动作序列 ---
                for action in new_actions:
                    t_step_start = time.time()
                    
                    target_joints = action[:-1]
                    gripper_val = action[-1]
                    
                    # 使用异步执行，并通过 sleep 控制节奏
                    self.robot_env.step(target_joints, asynchronous=True)
                    
                    # 夹爪逻辑 (保留) ...
                    
                    # 强制控制频率，例如 15Hz (0.066s) 或 20Hz (0.05s)
                    # RDT 预测的动作间隔取决于训练数据的频率
                    # 假设训练数据是 20Hz，这里就 sleep 0.05
                    dt = time.time() - t_step_start
                    remain = 0.033 - dt
                    if remain > 0: time.sleep(remain)


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