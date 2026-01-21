import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt

def inspect_starting_pose(dataset_path):
    print(f"📂 正在分析文件: {dataset_path}")
    
    j5_start_values = []
    
    try:
        with h5py.File(dataset_path, 'r') as f:
            demos = list(f['data'].keys())
            print(f"🔍 总共找到 {len(demos)} 条轨迹")
            
            for i, demo_key in enumerate(demos):
                # 读取关节数据: [Time, 7] or [Time, 8]
                qpos = f['data'][demo_key]['obs']['robot0_joint_pos'][:]
                
                # 取第一帧 (Frame 0) 的 J5 (Index 5)
                # 关节索引通常是: J0, J1, J2, J3, J4, J5, J6
                j5_val = qpos[0, 5]
                j5_start_values.append(j5_val)
                
                # 打印前 5 条轨迹的详情供参考
                if i < 5:
                    print(f"   [{demo_key}] Frame 0 -> J5的角度: {j5_val:.4f} rad")

    except Exception as e:
        print(f"❌ 读取错误: {e}")
        return

    # === 统计结果 ===
    j5_arr = np.array(j5_start_values)
    mean_val = np.mean(j5_arr)
    std_val = np.std(j5_arr)
    min_val = np.min(j5_arr)
    max_val = np.max(j5_arr)
    
    print("\n" + "="*40)
    print("📊 J5 关节起始位置 (Frame 0) 统计结果")
    print("="*40)
    print(f"   平均值 (Mean): {mean_val:.4f}")
    print(f"   中位数 (Median): {np.median(j5_arr):.4f}")
    print(f"   最小值 (Min):  {min_val:.4f}")
    print(f"   最大值 (Max):  {max_val:.4f}")
    print(f"   标准差 (Std):  {std_val:.4f}")
    print("="*40)
    
    # 你的物理位置是 1.57
    current_physical_j5 = 1.57
    diff = abs(mean_val - current_physical_j5)
    
    print(f"\n💡 诊断结论:")
    if diff > 0.3:
        print(f"❌ 严重不匹配！")
        print(f"   训练数据平均从 J5={mean_val:.2f} 开始，")
        print(f"   但你的物理机器人从 J5={current_physical_j5:.2f} 开始。")
        print(f"   偏差 {diff:.2f} rad (约 {np.degrees(diff):.1f} 度)。")
        print(f"   👉 这就是为什么机器人会花 5.5秒 '暴冲' 到 2.2 的原因。")
    else:
        print(f"✅ 数据匹配。起始位置看起来没问题。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 修改这里为你的实际 hdf5 路径
    parser.add_argument('--dataset', type=str, required=True, help='Path to your training HDF5 file')
    args = parser.parse_args()
    
    inspect_starting_pose(args.dataset)