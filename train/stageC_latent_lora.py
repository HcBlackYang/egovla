# # train/stageC_latent_lora.py
# import sys
# import os
# import torch
# import torch.optim as optim
# import argparse
# import time
# import h5py
# import numpy as np
# from torch.utils.data import DataLoader, Dataset
# from torch.amp import autocast
# from diffusers import DDPMScheduler
# from peft import LoraConfig, get_peft_model

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from models.rdt_model import RDTWrapper

# RDT_PATH = '/yanghaochuan/models/rdt-1b'

# # === 极简 Dataset ===
# class LatentDataset(Dataset):
#     def __init__(self, h5_path):
#         self.h5_path = h5_path
#         with h5py.File(h5_path, 'r') as f:
#             self.length = f['e_t'].shape[0]
#             # 数据很小，直接读进内存 (Load into RAM)
#             print("Loading latents into RAM...")
#             self.e_t = torch.from_numpy(f['e_t'][:])
#             self.actions = torch.from_numpy(f['action_target'][:])

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         return {
#             'e_t': self.e_t[idx],
#             'action_target': self.actions[idx]
#         }

# def train_stage_c_latent(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"=== Stage C (Latent + LoRA) Training on {device} ===")

#     # 1. 只有 RDT，没有 Encoder
#     print("Loading RDT Policy...")
#     rdt_wrapper = RDTWrapper(action_dim=8, model_path=RDT_PATH, rdt_cond_dim=768).to(device)
    
#     # 2. 应用 LoRA (修复了 input_ids 报错问题)
#     print("Applying LoRA...")
#     peft_config = LoraConfig(
#         r=16,
#         lora_alpha=32,
#         target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "linear"], 
#         lora_dropout=0.05,
#         bias="none"
#         # task_type 被移除了，防止报错
#     )
#     rdt_wrapper.rdt_model = get_peft_model(rdt_wrapper.rdt_model, peft_config)
#     rdt_wrapper.rdt_model.print_trainable_parameters()
    
#     # 3. 优化器
#     optimizer = optim.AdamW(filter(lambda p: p.requires_grad, rdt_wrapper.parameters()), lr=1e-4, weight_decay=1e-4)
#     noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", prediction_type="sample")

#     # 4. 加载 Cache 数据
#     print(f"Loading cached latents: {args.cache_path}")
#     dataset = LatentDataset(args.cache_path)
#     # 显存足够时，Batch Size 可以开大，比如 128 或 256
#     loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)

#     # 5. 飞速训练循环
#     rdt_wrapper.train()
#     print(">>> Training Started... <<<")
    
#     for epoch in range(args.epochs):
#         total_loss = 0
#         start_time = time.time()
        
#         for i, batch in enumerate(loader):
#             # 数据直接在 GPU 上准备好
#             e_t = batch['e_t'].to(device)
#             actions = batch['action_target'].to(device) # [B, 8]

#             optimizer.zero_grad()

#             with autocast('cuda', dtype=torch.bfloat16):
#                 # 不需要 Encoder 前向传播了！直接用 e_t
#                 timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (actions.shape[0],), device=device).long()
                
#                 noise = torch.randn_like(actions)
#                 noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)
                
#                 conditions = {"e_t": e_t}
                
#                 # RDT Forward
#                 pred_noise = rdt_wrapper(noisy_actions, timesteps, conditions)
                
#                 if pred_noise.dim() == 3 and noise.dim() == 2:
#                     noise = noise.unsqueeze(1)
                
#                 loss = torch.nn.functional.mse_loss(pred_noise, noise)

#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(rdt_wrapper.parameters(), 1.0)
#             optimizer.step()
            
#             total_loss += loss.item()
            
#             if i % 50 == 0 and i > 0: # 打印频率降低
#                 elapsed = time.time() - start_time
#                 speed = (i * args.batch_size) / elapsed
#                 print(f"Epoch {epoch} [{i}/{len(loader)}] Loss: {loss.item():.4f} | Speed: {speed:.1f} samples/s")

#         # 保存权重 (注意：这里我们只需要保存 LoRA 权重，不用保存 Encoder)
#         if epoch % 5 == 0 or epoch == args.epochs - 1:
#             os.makedirs(args.output_dir, exist_ok=True)
#             save_path = os.path.join(args.output_dir, f"stageC_lora_epoch_{epoch}.pt")
            
#             # 为了方便推理脚本加载，我们可以把 RDT 权重和空的 Encoder key 存进去
#             # 或者只存 RDT LoRA，推理时再加载 Encoder
#             torch.save({
#                 'rdt_state_dict': rdt_wrapper.state_dict(), 
#                 'note': 'This checkpoint only contains RDT LoRA weights. Encoder weights are in Stage B checkpoint.'
#             }, save_path)
#             print(f"Saved LoRA to {save_path}")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--cache_path', type=str, default='/yanghaochuan/projects/data/latents_cache.hdf5')
#     parser.add_argument('--output_dir', type=str, default='/yanghaochuan/projects/checkpoints')
#     parser.add_argument('--batch_size', type=int, default=128) # Batch Size 可以开大
#     parser.add_argument('--epochs', type=int, default=50)
    
#     args = parser.parse_args()
#     train_stage_c_latent(args)