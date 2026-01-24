import h5py
import numpy as np
import argparse
import os

def analyze_stats(hdf5_path):
    print(f"📂 Opening HDF5: {hdf5_path}")
    if not os.path.exists(hdf5_path):
        print(f"❌ File not found: {hdf5_path}")
        return

    all_qpos = []
    type_b_starts = []

    with h5py.File(hdf5_path, 'r') as f:
        demos = list(f['data'].keys())
        print(f"🔍 Scanning {len(demos)} demos...")
        
        for demo_key in demos:
            qpos = f['data'][demo_key]['obs']['robot0_joint_pos'][:] # [T, 7] or [T, 8]
            all_qpos.append(qpos)
            
            # Check Type B (Fixed Start)
            is_type_b = False
            if 'data_type' in f['data'][demo_key].attrs:
                if f['data'][demo_key].attrs['data_type'] == 'type_b': is_type_b = True
            elif int(demo_key.split('_')[1]) % 5 == 0:
                is_type_b = True
            
            if is_type_b:
                type_b_starts.append(qpos[0])

    # Concatenate all steps
    all_data = np.concatenate(all_qpos, axis=0)
    
    # Calculate Global Stats
    global_mean = np.mean(all_data, axis=0)
    global_std = np.std(all_data, axis=0)
    # Avoid div by zero
    global_std = np.maximum(global_std, 1e-2)

    # Calculate Type B Start Stats
    if not type_b_starts:
        print("❌ No Type B starts found.")
        return
    
    curr_pose = np.mean(np.array(type_b_starts), axis=0)

    print("\n" + "="*60)
    print("🕵️‍♂️ Z-SCORE DIAGNOSIS (Is your start pose an outlier?)")
    print("="*60)
    print(f"{'Joint':<5} | {'Curr Pose':<10} | {'Global Mean':<10} | {'Global Std':<10} | {'Z-Score (Norm Input)':<20}")
    print("-" * 65)

    joints_to_check = [0, 3, 5] # J0, J3, J5
    for j in joints_to_check:
        val = curr_pose[j]
        mean = global_mean[j]
        std = global_std[j]
        z_score = (val - mean) / std
        
        status = "✅ OK"
        if abs(z_score) > 3.0: status = "⚠️ SUSPICIOUS (>3)"
        if abs(z_score) > 5.0: status = "🚨 OUTLIER (>5) -> MODEL IGNORES THIS!"
        
        print(f"J{j:<4} | {val:<10.4f} | {mean:<10.4f} | {std:<10.4f} | {z_score:<10.4f} {status}")

    print("-" * 65)
    print("👉 If Z-Score is > 5.0, the model thinks the input state is garbage noise.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/yanghaochuan/data/hdf5/pick_up_the_orange_ball_and_put_it_on_the_plank.hdf5')
    args = parser.parse_args()
    analyze_stats(args.path)