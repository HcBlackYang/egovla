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
                
                # # 1. 获取图像序列
                # wrist_images = self.recorder.get_sequence_input()
                # if wrist_images is None:
                #     time.sleep(0.01)
                #     continue
                # 🟢 [修改 1]：只获取最新的一帧，而不是整个序列
                # 因为 Agent 内部已经维护了长达 500 的 Buffer，不需要我们每次重复发历史
                with self.recorder.lock:
                    latest_img = self.recorder.latest_frame
                
                if latest_img is None:
                    time.sleep(0.01)
                    continue
                
                # 包装成 List 发送，因为 Agent.step 接口期望的是 List[np.array]
                wrist_images = [latest_img]

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
                print(f"Model Gripper: {gripper_val:.4f}", end="\r")

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