# utils/binarize_gripper_safe.py
import h5py
import numpy as np
import argparse
import shutil
import os
from tqdm import tqdm

def binarize_hdf5_safe(input_path, output_path=None, threshold=0.0616):
    """
    1. 复制原 HDF5 到新路径
    2. 在新文件上将夹爪数据二值化 (-1.0 或 1.0)
    """
    
    # --- 1. 确定输出路径 ---
    if output_path is None:
        # 如果没指定输出名，自动生成: original.hdf5 -> original_binary.hdf5
        root, ext = os.path.splitext(input_path)
        output_path = f"{root}_binary{ext}"
    
    # 防止意外覆盖源文件
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError("❌ 输出路径不能与输入路径相同！请指定不同的输出文件名。")

    print(f"📋 正在复制文件...")
    print(f"   源文件: {input_path}")
    print(f"   新文件: {output_path}")
    
    # 使用 shutil 复制文件 (保留元数据)
    shutil.copy2(input_path, output_path)
    print(f"✅ 复制完成，开始处理新文件...")

    print(f"⚙️ 二值化阈值: {threshold} ( > {threshold} -> 1.0, <= {threshold} -> -1.0)")
    
    # --- 2. 在新文件上原地修改 ---
    # 使用 r+ 模式修改复制后的文件
    with h5py.File(output_path, 'r+') as f:
        demos = list(f['data'].keys())
        
        count_open = 0
        count_close = 0
        processed_count = 0
        
        for demo_key in tqdm(demos):
            demo_grp = f['data'][demo_key]
            obs_grp = demo_grp['obs']
            
            target_dataset_name = None
            is_embedded = False # 标记夹爪是否嵌在 joint_pos 里
            
            # --- 策略 A: 检查 robot0_joint_pos 是否包含夹爪 (8维) ---
            # 这是最可能的情况，根据你之前的报错推断
            if 'robot0_joint_pos' in obs_grp:
                joint_shape = obs_grp['robot0_joint_pos'].shape
                if len(joint_shape) == 2 and joint_shape[1] == 8:
                    target_dataset_name = 'robot0_joint_pos'
                    is_embedded = True
            
            # --- 策略 B: 如果不是8维，寻找独立 Key ---
            if not is_embedded:
                if 'robot0_gripper_qpos' in obs_grp:
                    target_dataset_name = 'robot0_gripper_qpos'
                elif 'gripper_states' in obs_grp:
                    target_dataset_name = 'gripper_states'
                elif 'gripper_qpos' in obs_grp:
                    target_dataset_name = 'gripper_qpos'

            # --- 开始处理 ---
            if target_dataset_name:
                # 读取数据
                data = obs_grp[target_dataset_name][:]
                
                # 提取夹爪部分
                if is_embedded:
                    gripper_data = data[:, 7] # 最后一列
                else:
                    gripper_data = data
                
                # 二值化计算
                # 逻辑：大于阈值 -> 1.0，否则 -> -1.0
                binary_vals = np.where(gripper_data > threshold, 1.0, -1.0).astype(np.float32)
                
                # 统计
                count_open += np.sum(binary_vals == 1.0)
                count_close += np.sum(binary_vals == -1.0)
                
                # 写回数据
                if is_embedded:
                    data[:, 7] = binary_vals
                    del obs_grp[target_dataset_name]
                    obs_grp.create_dataset(target_dataset_name, data=data)
                else:
                    del obs_grp[target_dataset_name]
                    # 保持维度
                    if binary_vals.ndim == 1 and len(obs_grp[target_dataset_name].shape) == 2:
                         binary_vals = binary_vals[:, None]
                    obs_grp.create_dataset(target_dataset_name, data=binary_vals)
                
                processed_count += 1
            else:
                print(f"⚠️ {demo_key} 跳过: 无法定位夹爪数据。")

    print(f"\n🎉 全部完成!")
    print(f"📂 新文件保存在: {output_path}")
    print(f"✅ 成功修改 Demo 数: {processed_count} / {len(demos)}")
    print(f"📊 统计: 1.0 (Open) 帧数: {count_open}")
    print(f"📊 统计: -1.0 (Close) 帧数: {count_close}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='原始 HDF5 文件路径')
    parser.add_argument('--output', type=str, default=None, help='(可选) 新文件保存路径，默认添加 _binary 后缀')
    parser.add_argument('--threshold', type=float, default=0.0616, help='二值化阈值')
    args = parser.parse_args()
    
    binarize_hdf5_safe(args.dataset, args.output, args.threshold)