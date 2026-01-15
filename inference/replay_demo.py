import sys
import os
import time
import h5py
import numpy as np
import argparse
import math

# 添加项目根目录到 path，确保能 import robots
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.constants import ActionSpace
from robots.franky_env import FrankyEnv
from robots.robot_param import RobotParam

def replay_episode(args):
    # 1. 加载 HDF5 数据
    if not os.path.exists(args.dataset):
        print(f"❌ 找不到数据集: {args.dataset}")
        return

    print(f"📂 Loading dataset: {args.dataset}")
    with h5py.File(args.dataset, 'r') as f:
        demo_key = f"data/demo_{args.demo_idx}"
        if demo_key not in f:
            print(f"❌ 找不到演示: {demo_key}")
            available = list(f['data'].keys())[:5]
            print(f"   可用演示示例: {available} ...")
            return
        
        # 读取关节数据
        if 'actions' in f[demo_key]:
            actions_all = f[demo_key]['actions'][:]
        else:
            actions_all = f[demo_key]['obs']['robot0_joint_pos'][:]
            
    print(f"✅ Loaded {demo_key}, length: {len(actions_all)} frames")

    # 2. 初始化机器人
    print("🤖 Initializing Robot Connection...")
    robot_env = FrankyEnv(
        action_space=ActionSpace.JOINT_ANGLES, 
        inference_mode=True, 
        robot_param=RobotParam(np.array([ 0.0, 0.0, -math.pi / 2]), np.array([ 0.53433071, 0.52905707, 0.00440881]))
    )
    
    # 3. 准备回放数据
    # 🟢 [关键修复]：强制转换为 float64，否则 C++ 底层会报错
    joint_traj = actions_all[:, :7].astype(np.float64) 
    gripper_traj = actions_all[:, 7].astype(np.float64)
    
    # 4. 移动到起始点
    start_joint = joint_traj[0]
    print(f"🚀 Moving to START position (taking 3 seconds)...")
    
    current_joints = robot_env.get_position(ActionSpace.JOINT_ANGLES)
    
    # 插值运动到起点
    steps = 100
    for i in range(steps):
        alpha = (i + 1) / steps
        interp_joints = current_joints * (1 - alpha) + start_joint * alpha
        # 这里的 interp_joints 已经是 float64
        robot_env.step(interp_joints, asynchronous=False)
        time.sleep(0.03)
    
    print("📍 Reached Start Position. Press ENTER to start replay (or Ctrl+C to cancel)...")
    input()

    # 5. 开始循环回放
    print("▶️ Replaying...")
    
    GRIPPER_THRESHOLD = 0.06 
    gripper_state = 0 
    
    try:
        start_time = time.time()
        for i in range(len(joint_traj)):
            loop_start = time.time()
            
            # 这里取出的就是 float64 了
            target_joints = joint_traj[i] 
            target_gripper = gripper_traj[i]
            
            # 发送关节指令
            robot_env.step(target_joints, asynchronous=True)
            
            # 发送夹爪指令
            if target_gripper > GRIPPER_THRESHOLD:
                if gripper_state != -1:
                    robot_env.open_gripper(asynchronous=True)
                    gripper_state = -1
                    print(f"[{i}] 👐 Open")
            else:
                if gripper_state != 1:
                    robot_env.close_gripper(asynchronous=True)
                    gripper_state = 1
                    print(f"[{i}] ✊ Close")

            # 30Hz 控频
            dt = time.time() - loop_start
            wait = 1.0/30.0 - dt
            if wait > 0:
                time.sleep(wait)
                
            if i % 30 == 0:
                print(f"Progress: {i}/{len(joint_traj)}", end='\r')

        print(f"\n✅ Replay Finished. Total time: {time.time() - start_time:.2f}s")
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during replay: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/home/dell/maple_control/systems/pick_up_the_orange_ball_and_put_it_on_the_plank.hdf5')
    parser.add_argument('--demo_idx', type=int, default=0)
    args = parser.parse_args()
    
    replay_episode(args)