# utils/add_new_data.py
import h5py
import numpy as np
import argparse
import shutil
import os
from tqdm import tqdm

def add_and_process_data(main_hdf5, new_hdf5, threshold=0.0616):
    """
    1. 将 new_hdf5 中的 demo 追加到 main_hdf5
    2. 自动重命名 demo_key 防止冲突 (例如 demo_0 -> demo_100)
    3. 对新加入的 demo 进行夹爪二值化
    """
    print(f"📂 主文件: {main_hdf5}")
    print(f"📂 新文件: {new_hdf5}")

    # 以 r+ 模式打开主文件
    with h5py.File(main_hdf5, 'r+') as f_main, h5py.File(new_hdf5, 'r') as f_new:
        # 1. 确定起始索引
        existing_keys = list(f_main['data'].keys())
        # 获取现有最大的 ID，例如 demo_99 -> max_id = 99
        max_id = -1
        for k in existing_keys:
            try:
                curr_id = int(k.split('_')[1])
                if curr_id > max_id: max_id = curr_id
            except: pass
        
        start_id = max_id + 1
        print(f"🔢 现有最大 ID: {max_id}, 新数据将从 demo_{start_id} 开始追加...")

        new_keys = list(f_new['data'].keys())
        print(f"📦 准备合并 {len(new_keys)} 条新数据...")

        count_added = 0
        
        # 2. 复制数据
        for i, old_key in enumerate(tqdm(new_keys, desc="Merging")):
            source_grp = f_new['data'][old_key]
            target_key = f"demo_{start_id + i}"
            
            # 复制 Group
            f_new.copy(source_grp, f_main['data'], name=target_key)
            
            # --- 3. 立即对新数据进行夹爪二值化 ---
            # 只有新复制进去的这个 group 需要处理
            target_grp = f_main['data'][target_key]
            obs_grp = target_grp['obs']
            
            # 查找夹爪数据 Key
            target_dataset_name = None
            is_embedded = False
            
            if 'robot0_joint_pos' in obs_grp:
                joint_shape = obs_grp['robot0_joint_pos'].shape
                if len(joint_shape) == 2 and joint_shape[1] == 8:
                    target_dataset_name = 'robot0_joint_pos'
                    is_embedded = True
            
            if not is_embedded:
                for k in ['robot0_gripper_qpos', 'gripper_states', 'gripper_qpos']:
                    if k in obs_grp:
                        target_dataset_name = k
                        break
            
            # 执行二值化
            if target_dataset_name:
                data = obs_grp[target_dataset_name][:]
                
                if is_embedded:
                    gripper_data = data[:, 7]
                else:
                    gripper_data = data
                
                # 二值化逻辑
                binary_vals = np.where(gripper_data > threshold, 1.0, -1.0).astype(np.float32)
                
                # 写回
                if is_embedded:
                    data[:, 7] = binary_vals
                    del obs_grp[target_dataset_name]
                    obs_grp.create_dataset(target_dataset_name, data=data)
                else:
                    del obs_grp[target_dataset_name]
                    if binary_vals.ndim == 1 and len(obs_grp[target_dataset_name].shape) == 2:
                         binary_vals = binary_vals[:, None]
                    obs_grp.create_dataset(target_dataset_name, data=binary_vals)
            
            count_added += 1

    print(f"\n🎉 合并完成！")
    print(f"✅ 已添加 {count_added} 条数据 (ID范围: {start_id} ~ {start_id + count_added - 1})")
    print(f"✅ 新数据夹爪已二值化 (Threshold={threshold})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--main', type=str, required=True, help='主 HDF5 文件路径')
    parser.add_argument('--new', type=str, required=True, help='包含新 40 条数据的 HDF5 文件路径')
    args = parser.parse_args()
    
    add_and_process_data(args.main, args.new)