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
        
        # Camera 初始化
        from cameras.realsense_env import RealSenseEnv
        from cameras.camera_param import CameraParam
        
        # 主摄可以保留初始化，但不启动 monitoring
        self.main_camera = RealSenseEnv(camera_name="main_image", serial_number="339322073638", width=1280, height=720,
                                        camera_param=CameraParam(intrinsic_matrix = np.array([[908.1308, 0, 655.7268], [0, 910.0818, 395.8856], [0, 0, 1]], dtype=np.float32),
                                                                 distortion_coeffs = np.array([0.1068, -0.2123, -0.0092, 0.0000, 0.0000], dtype=np.float32)))
        self.wrist_camera = RealSenseEnv(camera_name="wrist_image", serial_number="342222072092", width=1280, height=720)
        
        self.gripper_status = {"current_state": 0, "target_state": 0}
        self.stop_evaluation = threading.Event()

    def run(self, show_image: bool = False, task_name: str = "default_task"):
        # [修改点] 只启动手腕相机 (Wrist-Only Inference)
        self.wrist_camera.start_monitoring()
        # self.main_camera.start_monitoring() 
        
        logging.info("Waiting 2.0s for cameras to warm up...")
        time.sleep(2.0)
        
        logging.info("Starting inference loop...")
        
        while not self.stop_evaluation.is_set():
            t0 = time.time()
            
            # 1. 获取图像
            # main_frame_data = self.main_camera.get_latest_frame() # 不读取真实主摄
            wrist_frame_data = self.wrist_camera.get_latest_frame()

            if wrist_frame_data is None:
                time.sleep(0.01)
                continue
            
            wrist_image = wrist_frame_data['bgr']
            
            # [修改点] 构造全黑的主摄图像作为占位符
            # 必须传给服务器，因为模型是双摄结构，但内容是全黑的（符合 Modality Dropout 训练）
            main_image = np.zeros_like(wrist_image)

            # 2. 获取状态
            joint_angles = self.robot_env.get_position(action_space=ActionSpace.JOINT_ANGLES)
            gripper_width = self.robot_env.get_gripper_width()
            eef_pose = self.robot_env.get_position(action_space=ActionSpace.EEF_POSE)
            
            # [修改点] 构造 8 维 qpos (7关节 + 1夹爪)
            qpos_8d = list(joint_angles) + [float(gripper_width)]
        
            # [修改点] 构造 8 维 state
            state = np.concatenate([eef_pose, [gripper_width]])
            
            # 3. 构造请求
            element = {
                "observation/agentview_image": main_image, # 传入全黑图，Key名需与服务端匹配
                "observation/wrist_image": wrist_image,
                "observation/state": state,
                "qpos": qpos_8d, 
                "prompt": task_name,
            }

            # 4. 推理 (Blocking)
            inference_results = self.client.infer(element)
            
            if inference_results:
                # 获取第一个 chunk (通常 batch=1)
                new_actions = inference_results["actions"][0] # [Chunk_Size, 8]
                
                # --- 执行动作序列 (平滑模式) ---
                # [修改点] 只保留这一个循环，删除原来后面的 "Step 5" 循环
                for i, action in enumerate(new_actions):
                    t_step_start = time.time()
                    
                    target_joints = action[:-1] # 前7位
                    gripper_val = action[-1]    # 第8位
                    
                    # A. 关节控制 (异步执行 + sleep 配合，实现平滑且连续的运动)
                    self.robot_env.step(target_joints, asynchronous=True)
                    
                    # B. 夹爪控制 (状态机防止重复发送)
                    if gripper_val > 0.06: 
                         if self.gripper_status["current_state"] != -1:
                             self.robot_env.open_gripper(asynchronous=True)
                             self.gripper_status["current_state"] = -1
                    elif gripper_val < 0.02:
                         if self.gripper_status["current_state"] != 1:
                             self.robot_env.close_gripper(asynchronous=True)
                             self.gripper_status["current_state"] = 1
                    
                    # C. 频率控制 (例如 20Hz = 0.05s, 30Hz = 0.033s)
                    dt = time.time() - t_step_start
                    remain = 0.04 - dt # 稍微调大一点点间隔(例如 25Hz)，给通信留余量
                    if remain > 0: time.sleep(remain)

            # 可视化
            if show_image:
                cv2.imshow("Wrist View", wrist_image)
                cv2.waitKey(1)

            latency = (time.time() - t0) * 1000
            print(f"\rStep Latency: {latency:.1f}ms", end="")
            
            # [重要] 已删除原来的 "# === 5. 执行动作 ===" 代码块，避免双重执行

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