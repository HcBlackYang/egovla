# # models/rdt_model.py
# import torch
# import torch.nn as nn
# import os
# import sys
# import json
# import yaml
# import importlib.util
# import inspect
# import numbers

# # =========================================================================
# # 1. 强力参数清洗工具 & 补丁
# # =========================================================================
# def force_to_int(val):
#     try:
#         if val is None: return None
#         if isinstance(val, (tuple, list)):
#             val = val[0] if len(val) >= 1 else 0
#         if hasattr(val, 'item'): val = val.item()
#         if hasattr(val, 'dtype'): val = int(val)
#         if isinstance(val, float): val = int(val)
#         if isinstance(val, int): return val
#         try: return int(val)
#         except: return val 
#     except:
#         return val

# OriginalLinearInit = torch.nn.Linear.__init__
# def patched_linear_init(self, in_features, out_features, bias=True, device=None, dtype=None):
#     safe_in = force_to_int(in_features)
#     safe_out = force_to_int(out_features)
#     if not isinstance(safe_out, int):
#         try: safe_out = int(safe_out.out_channels) 
#         except: safe_out = 8 
#     OriginalLinearInit(self, safe_in, safe_out, bias=bias, device=device, dtype=dtype)
# torch.nn.Linear.__init__ = patched_linear_init

# try:
#     import timm.models.layers
#     if hasattr(timm.models.layers, 'Mlp'):
#         OriginalTimmMlpInit = timm.models.layers.Mlp.__init__
#         def patched_timm_init(self, in_features, hidden_features=None, out_features=None, *args, **kwargs):
#             return OriginalTimmMlpInit(self, force_to_int(in_features), force_to_int(hidden_features), force_to_int(out_features), *args, **kwargs)
#         timm.models.layers.Mlp.__init__ = patched_timm_init
# except: pass

# try:
#     import timm.layers
#     if hasattr(timm.layers, 'Mlp'):
#         OriginalLayerMlpInit = timm.layers.Mlp.__init__
#         def patched_layer_init(self, in_features, hidden_features=None, out_features=None, *args, **kwargs):
#             return OriginalLayerMlpInit(self, force_to_int(in_features), force_to_int(hidden_features), force_to_int(out_features), *args, **kwargs)
#         timm.layers.Mlp.__init__ = patched_layer_init
# except: pass

# # =========================================================================
# # 2. 加载 RDT 源码
# # =========================================================================
# RDT_ROOT = "/yanghaochuan/projects/RoboticsDiffusionTransformer"
# RDT_MODELS_DIR = os.path.join(RDT_ROOT, "models")

# if RDT_ROOT not in sys.path: sys.path.insert(0, RDT_ROOT)
# if RDT_MODELS_DIR not in sys.path: sys.path.insert(0, RDT_MODELS_DIR)
# if "models" in sys.modules and RDT_MODELS_DIR not in sys.modules["models"].__path__:
#     sys.modules["models"].__path__.append(RDT_MODELS_DIR)

# TARGET_FILE_PATH = os.path.join(RDT_ROOT, "models", "rdt", "model.py")
# ModelClass = None
# if os.path.exists(TARGET_FILE_PATH):
#     try:
#         spec = importlib.util.spec_from_file_location("rdt_source_model", TARGET_FILE_PATH)
#         rdt_module = importlib.util.module_from_spec(spec)
#         sys.modules["rdt_source_model"] = rdt_module
#         spec.loader.exec_module(rdt_module)
        
#         candidate_classes = []
#         for name, obj in inspect.getmembers(rdt_module):
#             if inspect.isclass(obj) and issubclass(obj, nn.Module):
#                 if any(k in name for k in ["Transformer", "RDT", "Model"]):
#                     if not any(k in name for k in ["Layer", "Block", "Attention", "Embed", "Head", "MLP", "Timestep"]):
#                         candidate_classes.append(obj)
#         if candidate_classes:
#             candidate_classes.sort(key=lambda x: len(x.__name__), reverse=True)
#             ModelClass = candidate_classes[0]
#             print(f"[RDTWrapper] ✅ 成功锁定模型类: {ModelClass.__name__}")
#         else:
#             print(f"[RDTWrapper] ❌ 未找到主模型类")
#     except Exception as e:
#         print(f"[RDTWrapper] ❌ 导入 model.py 失败: {e}")

# # =========================================================================
# # 3. RDTWrapper 类定义
# # =========================================================================
# class RDTWrapper(nn.Module):
#     def __init__(self, 
#                  action_dim=8, 
#                  model_path='/yanghaochuan/models/rdt-1b',
#                  rdt_cond_dim=1152,
#                  pred_horizon=16):
#         super().__init__()
#         if ModelClass is None: raise RuntimeError("无法初始化 RDT")

#         # 1. Config
#         config_path = os.path.join(model_path, "config.json")
#         if not os.path.exists(config_path): config_path = os.path.join(model_path, "config.yaml")
#         print(f"[RDTWrapper] Loading config from: {config_path}")
        
#         self.rdt_hidden_size = 2048 
#         # 调用内部方法加载配置
#         args = self._load_config_and_override(config_path, action_dim)

#         # 2. Instantiate
#         print(f"[RDTWrapper] Instantiating with forced horizon={args.horizon}")
#         try:
#             sig = inspect.signature(ModelClass.__init__)
#             params = list(sig.parameters.keys())
#             if 'output_dim' not in vars(args): args.output_dim = args.action_dim
#             valid_args = {k: v for k, v in vars(args).items() if k in params or 'kwargs' in str(sig)}
#             self.rdt_model = ModelClass(**valid_args)
#             print("[RDTWrapper] Instantiation successful via kwargs unpacking.")
#         except Exception as e:
#             print(f"[RDTWrapper] Kwargs instantiation failed: {e}. Falling back to object pass...")
#             self.rdt_model = ModelClass(args)

#         # 3. Detect ACTUAL Hidden Size
#         actual_dim = self.rdt_hidden_size
#         if hasattr(self.rdt_model, 'hidden_size'): actual_dim = self.rdt_model.hidden_size
#         elif hasattr(self.rdt_model, 'embed_dim'): actual_dim = self.rdt_model.embed_dim
#         else:
#             for m in self.rdt_model.modules():
#                 if isinstance(m, nn.Linear):
#                     actual_dim = m.out_features
#                     break
#         print(f"[RDTWrapper] 🔍 Detected Actual Hidden Dimension: {actual_dim}")
        
#         # 4. Load Weights (Smart Loading with Adaptation)
#         weights_path = os.path.join(model_path, "pytorch_model.bin")
#         if not os.path.exists(weights_path): weights_path = os.path.join(model_path, "diffusion_pytorch_model.bin")
        
#         if os.path.exists(weights_path):
#             print(f"[RDTWrapper] Loading weights with schema adaptation...")
#             try:
#                 state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
#             except TypeError:
#                 state_dict = torch.load(weights_path, map_location="cpu")
            
#             new_state_dict = {}
#             current_model_dict = self.rdt_model.state_dict()
            
#             for k, v in state_dict.items():
#                 if k.startswith("module."): k = k[7:]
#                 if k in current_model_dict:
#                     target_shape = current_model_dict[k].shape
                    
#                     # === 适配 1: x_pos_embed (3 tokens vs 4 tokens) ===
#                     # 场景：官方权重有 State Token (len 4)，你的模型没有 (len 3)
#                     if "x_pos_embed" in k:
#                         if v.shape[1] == 4 and target_shape[1] == 3:
#                             print(f"[RDTWrapper] ✂️  Slicing x_pos_embed: Removing 'state' token (index 2).")
#                             # 官方顺序: [Time, Freq, State, Action] -> 保留 [0, 1, 3]
#                             v = v[:, [0, 1, 3], :]
                    
#                     # === 适配 2: img_cond_pos_embed (4000+ vs 2) ===
#                     # 场景：官方权重巨大，我们只需要 2 个占位符
#                     if "img_cond_pos_embed" in k:
#                         if v.shape[1] > target_shape[1]:
#                             # 直接截取前 N 个，反正我们传的是全 0 占位符
#                             v = v[:, :target_shape[1], :]

#                     if v.shape != target_shape:
#                         # 兜底：如果形状还不匹配，跳过（防止报错）
#                         print(f"[RDTWrapper] ⚠️  Skipping {k}: shape mismatch {v.shape} vs {target_shape}")
#                         continue
                        
#                     new_state_dict[k] = v
            
#             self.rdt_model.load_state_dict(new_state_dict, strict=False)

#         # 5. Initialize Projection Layers
#         target_dim = actual_dim 
#         self.action_proj = nn.Linear(int(action_dim), int(target_dim))
#         self.cond_proj = nn.Linear(int(rdt_cond_dim), int(target_dim))
#         self.state_proj = nn.Linear(8, int(target_dim))

#         # === 适配 3: 强制调整模型内部 img_pos_embed 大小 ===
#         DUBBY_IMG_LEN = 2
#         if hasattr(self.rdt_model, 'img_cond_pos_embed'):
#              if self.rdt_model.img_cond_pos_embed.shape[1] > DUBBY_IMG_LEN:
#                  print(f"[RDTWrapper] 📉 Resizing internal img_cond_pos_embed to length {DUBBY_IMG_LEN}")
#                  old_pe = self.rdt_model.img_cond_pos_embed.data
#                  new_pe = nn.Parameter(old_pe[:, :DUBBY_IMG_LEN, :].clone())
#                  self.rdt_model.img_cond_pos_embed = new_pe

#         target_dim = 1152 
#         # 为了更通用，建议获取 model 的 hidden_size，或者直接硬编码你确定的值
#         if hasattr(self.rdt_model, 'config') and hasattr(self.rdt_model.config, 'hidden_size'):
#              target_dim = self.rdt_model.config.hidden_size
             
#         # 这里输入维度必须对应 FusionEncoder 的输出 (768)
#         self.visual_proj = nn.Linear(768, target_dim)


#     # =================================================
#     # 辅助方法：注意缩进，它必须在 class RDTWrapper 内部
#     # =================================================
#     def _load_config_and_override(self, config_path, target_action_dim):
#         class Args: pass
#         args = Args()
        
#         if config_path and os.path.exists(config_path):
#             with open(config_path, 'r') as f:
#                 cfg = json.load(f)
            
#             if 'rdt' in cfg and 'hidden_size' in cfg['rdt']:
#                 self.rdt_hidden_size = int(cfg['rdt']['hidden_size'])
#             elif 'hidden_size' in cfg:
#                 self.rdt_hidden_size = int(cfg['hidden_size'])
                
#             for k, v in cfg.items():
#                 if k == 'rdt' and isinstance(v, dict):
#                     for sub_k, sub_v in v.items():
#                         setattr(args, sub_k, sub_v)
#                 setattr(args, k, v)
                
#         args.action_dim = int(target_action_dim)
#         args.output_dim = int(target_action_dim)
#         args.out_channels = int(target_action_dim)
#         args.input_size = int(target_action_dim)
#         args.in_channels = int(target_action_dim)
        
#         args.hidden_size = self.rdt_hidden_size
#         args.embed_dim = self.rdt_hidden_size 
#         args.d_model = self.rdt_hidden_size
        
#         args.horizon = int(pred_horizon)      # 预测未来多少步
#         args.pred_horizon = int(pred_horizon)
        
#         defaults = {'patch_size': 1, 'img_size': 1, 'num_frames': 1}
#         for k, v in defaults.items():
#             if not hasattr(args, k): setattr(args, k, v)
            
#         return args

#     # def forward(self, noisy_action, timestep, conditions):
#     #     e_t = conditions['e_t']
#     #     cond_embeds = self.cond_proj(e_t).unsqueeze(1) # [B, 1, D]
        
#     #     if noisy_action.dim() == 2: x_in = noisy_action
#     #     else: x_in = noisy_action.squeeze(1)
            
#     #     x_embed = self.action_proj(x_in).unsqueeze(1) # [B, 1, D]
        
#     #     B = x_embed.shape[0]
#     #     device = x_embed.device
        
#     #     freq = torch.full((B,), 30, device=device, dtype=torch.long)
        
#     #     lang_c = cond_embeds 
        
#     #     # 图像条件：全 0 占位符 (长度为 2)
#     #     img_c = torch.zeros((B, 2, cond_embeds.shape[-1]), device=device, dtype=cond_embeds.dtype)
        
#     #     lang_mask = torch.ones((B, 1), device=device, dtype=torch.bool)
#     #     img_mask = torch.ones((B, 2), device=device, dtype=torch.bool)

#     #     return self.rdt_model(
#     #         x=x_embed, 
#     #         freq=freq, 
#     #         t=timestep, 
#     #         lang_c=lang_c, 
#     #         img_c=img_c,
#     #         lang_mask=lang_mask,
#     #         img_mask=img_mask
#     #     )


#     def forward(self, noisy_action, timestep, conditions):
#         """
#         RDTWrapper 的前向传播 (修改版)
#         输入:
#             noisy_action: [B, Horizon, Action_Dim] (例如 [B, 16, 8])
#             timestep: [B]
#             conditions: dict, 包含 'e_t' (序列特征)
#         """
#         # 1. 获取视觉序列特征
#         e_t = conditions['e_t'] # 期望形状: [B, 64, 768] (来自 FusionEncoder 的序列输出)
        
#         B = e_t.shape[0]
#         device = e_t.device
#         dtype = e_t.dtype
        
#         # 2. 视觉投影 (768 -> RDT Hidden Size, 通常是 1152)
#         # [B, 64, 768] -> [B, 64, 1152]
#         img_c = self.visual_proj(e_t)
        
#         # 3. 构造文本条件 (Language Condition)
#         # 因为我们主要依赖视觉，这里传入空的文本嵌入 (Zero Padding)
#         # RDT 期望 lang_c 形状为 [B, L, D]，这里设 L=1
#         lang_c = torch.zeros((B, 1, 1152), device=device, dtype=dtype)
        
#         # 4. 构造 Masks
#         # img_mask: [B, 64], 全 1 (所有视觉 Token 有效)
#         img_mask = torch.ones((B, img_c.shape[1]), device=device, dtype=torch.long)
        
#         # lang_mask: [B, 1], 全 1 (表示这是一个有效的"空指令")
#         # 注意: RDT 内部对 mask 的处理通常是 bool 或 0/1, long 比较稳妥
#         lang_mask = torch.ones((B, 1), device=device, dtype=torch.long)

#         # 5. 处理动作输入
#         # 如果输入是 [B, 8]，unsqueeze 成 [B, 1, 8]
#         # 如果是 [B, 16, 8]，保持不变
#         if noisy_action.dim() == 2:
#             x_in = noisy_action.unsqueeze(1)
#         else:
#             x_in = noisy_action
            
#         # 投影动作维度 [B, H, 8] -> [B, H, Hidden]
#         x_embed = self.action_proj(x_in)
        
#         # 6. 控制频率 (Control Frequency)
#         # 固定为 30Hz 或从 dataset 统计中获取
#         freq = torch.full((B,), 30, device=device, dtype=torch.long)

#         # 7. 调用 RDT Backbone
#         return self.rdt_model(
#             x=x_embed, 
#             freq=freq, 
#             t=timestep, 
#             lang_c=lang_c, 
#             img_c=img_c,
#             lang_mask=lang_mask,
#             img_mask=img_mask
#         )

# # models/rdt_model.py
# import torch
# import torch.nn as nn
# import os
# import sys
# import json
# import importlib.util
# import inspect
# import logging

# # 配置日志
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("[RDTWrapper]")

# # =========================================================================
# # 1. 强力参数清洗工具 & 补丁
# # =========================================================================
# def force_to_int(val):
#     try:
#         if val is None: return None
#         if isinstance(val, (tuple, list)):
#             val = val[0] if len(val) >= 1 else 0
#         if hasattr(val, 'item'): val = val.item()
#         try: return int(val)
#         except: return val 
#     except:
#         return val

# OriginalLinearInit = torch.nn.Linear.__init__
# def patched_linear_init(self, in_features, out_features, bias=True, device=None, dtype=None):
#     safe_in = force_to_int(in_features)
#     safe_out = force_to_int(out_features)
#     if not isinstance(safe_out, int):
#         try: safe_out = int(safe_out.out_channels) 
#         except: safe_out = 8 
#     OriginalLinearInit(self, safe_in, safe_out, bias=bias, device=device, dtype=dtype)
# torch.nn.Linear.__init__ = patched_linear_init

# # =========================================================================
# # 2. 加载 RDT 源码
# # =========================================================================
# RDT_ROOT = "/yanghaochuan/projects/RoboticsDiffusionTransformer"
# RDT_MODELS_DIR = os.path.join(RDT_ROOT, "models")

# if RDT_ROOT not in sys.path: sys.path.insert(0, RDT_ROOT)
# if RDT_MODELS_DIR not in sys.path: sys.path.insert(0, RDT_MODELS_DIR)
# if "models" in sys.modules and RDT_MODELS_DIR not in sys.modules["models"].__path__:
#     sys.modules["models"].__path__.append(RDT_MODELS_DIR)

# TARGET_FILE_PATH = os.path.join(RDT_ROOT, "models", "rdt", "model.py")
# ModelClass = None
# if os.path.exists(TARGET_FILE_PATH):
#     try:
#         spec = importlib.util.spec_from_file_location("rdt_source_model", TARGET_FILE_PATH)
#         rdt_module = importlib.util.module_from_spec(spec)
#         sys.modules["rdt_source_model"] = rdt_module
#         spec.loader.exec_module(rdt_module)
        
#         # 尝试找到 RDT 类
#         candidate_classes = []
#         for name, obj in inspect.getmembers(rdt_module):
#             if inspect.isclass(obj) and issubclass(obj, nn.Module):
#                 if any(k in name for k in ["Transformer", "RDT", "Model"]):
#                     if not any(k in name for k in ["Layer", "Block", "Attention", "Embed", "Head", "MLP", "Timestep"]):
#                         candidate_classes.append(obj)
#         if candidate_classes:
#             candidate_classes.sort(key=lambda x: len(x.__name__), reverse=True)
#             ModelClass = candidate_classes[0]
#             logger.info(f"✅ 成功锁定模型类: {ModelClass.__name__}")
#         else:
#             logger.error("❌ 未找到主模型类")
#     except Exception as e:
#         logger.error(f"❌ 导入 model.py 失败: {e}")

# # =========================================================================
# # 3. RDTWrapper 类定义 (最终修正版)
# # =========================================================================
# class RDTWrapper(nn.Module):
#     def __init__(self, 
#                  action_dim=8, 
#                  model_path='/yanghaochuan/models/rdt-1b',
#                  rdt_cond_dim=1152,
#                  pred_horizon=16):
#         super().__init__()
#         if ModelClass is None: raise RuntimeError("无法初始化 RDT")

#         # 1. Config
#         config_path = os.path.join(model_path, "config.json")
#         if not os.path.exists(config_path): config_path = os.path.join(model_path, "config.yaml")
#         logger.info(f"Loading config from: {config_path}")
        
#         self.rdt_hidden_size = 768 
#         args = self._load_config_and_override(config_path, action_dim, pred_horizon)

#         # 2. Instantiate Base Model
#         # 注意：这里初始化的模型会有默认的 pos_embed (可能是 34 或其他)
#         try:
#             self.rdt_model = ModelClass(args)
#         except:
#             self.rdt_model = ModelClass(**vars(args))

#         # 3. Detect ACTUAL Hidden Size
#         actual_dim = self.rdt_hidden_size
#         if hasattr(self.rdt_model, 'hidden_size'): actual_dim = self.rdt_model.hidden_size
#         elif hasattr(self.rdt_model, 'embed_dim'): actual_dim = self.rdt_model.embed_dim
#         else:
#             for m in self.rdt_model.modules():
#                 if isinstance(m, nn.Linear):
#                     actual_dim = m.out_features
#                     break
#         logger.info(f"🔍 Detected Actual Hidden Dimension: {actual_dim}")
        
#         # 4. Initialize Projection Layers (New Modalities)
#         # 必须尽早定义，确保在 load_state_dict 之前存在，或者是 LoRA 之后
#         # 但为了加载预训练权重，我们主要关注 RDT 内部
#         fusion_out_dim = 768
#         state_dim = 8 
#         self.state_proj = nn.Linear(int(state_dim), int(actual_dim)) 
#         self.action_proj = nn.Linear(int(action_dim), int(actual_dim))
#         self.visual_proj = nn.Linear(int(fusion_out_dim), int(actual_dim))

#         # 5. Smart Weight Loading & Surgery (关键步骤)
#         weights_path = os.path.join(model_path, "pytorch_model.bin")
#         if not os.path.exists(weights_path): weights_path = os.path.join(model_path, "diffusion_pytorch_model.bin")
        
#         if os.path.exists(weights_path):
#             logger.info(f"Loading weights from {weights_path}...")
#             try: state_dict = torch.load(weights_path, map_location="cpu")
#             except: state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
            
#             # 获取当前模型的 state_dict 以便对比
#             current_model_dict = self.rdt_model.state_dict()
#             new_state_dict = {}
            
#             # --- [手术区：x_pos_embed] ---
#             # 目标: [Time(1), Freq(1), State(1), Action(16)] 共 19
#             # 原始: [Time(1), Freq(1), Action(32)] 共 34 (假设无State) 或者是 [Time(1), Freq(1), State(1), Action(32)] 共 35
#             target_x_len = 1 + 1 + 1 + pred_horizon # 19
            
#             # 找到 checkpoint 里的 x_pos_embed
#             ckpt_x_embed = None
#             for k, v in state_dict.items():
#                 if "x_pos_embed" in k:
#                     ckpt_x_embed = v
#                     break
            
#             if ckpt_x_embed is not None:
#                 logger.info(f"🩹 Performing surgery on x_pos_embed. Ckpt shape: {ckpt_x_embed.shape}, Target: {target_x_len}")
                
#                 # 构造新的 embedding
#                 # 1. 复制 Time (idx 0) 和 Freq (idx 1)
#                 new_x_embed = [ckpt_x_embed[:, 0:1, :], ckpt_x_embed[:, 1:2, :]]
                
#                 # 2. 处理 State (idx 2)
#                 # 这是一个全新的 token，我们不能直接复用 Action 0 的位置编码
#                 # 策略：初始化为 0 或随机，并设置为可训练
#                 # 为了保持分布一致，我们可以取 Time 和 Freq 的均值作为初始值，或者直接用 0 (原始代码是用 0 初始化的)
#                 state_embed = torch.zeros_like(ckpt_x_embed[:, 0:1, :]) 
#                 new_x_embed.append(state_embed)
                
#                 # 3. 复制 Action (idx 3 ~ 18)
#                 # 原始 Action 从 idx 2 开始 (假设原始无 State) 或 idx 3 (如果原始有 State)
#                 # 鉴于之前报错 34 (1+1+32)，原始应该无 State，Action 从 idx 2 开始
#                 if ckpt_x_embed.shape[1] == 34: # 原始无 State
#                     # 取原始的前 16 个 Action (idx 2 ~ 2+16)
#                     # 这样 Action 0 对应 Action 0，对齐正确！
#                     action_embeds = ckpt_x_embed[:, 2 : 2+pred_horizon, :]
#                     new_x_embed.append(action_embeds)
#                 else:
#                     # 兜底：如果维度奇怪，尝试直接对齐
#                     logger.warning("Checkpoint dimensions unexpected, falling back to safe slicing for tail")
#                     start_idx = 2
#                     available_len = ckpt_x_embed.shape[1] - start_idx
#                     copylen = min(available_len, pred_horizon)
#                     action_embeds = ckpt_x_embed[:, start_idx : start_idx+copylen, :]
#                     new_x_embed.append(action_embeds)

#                 # 拼接
#                 final_x_embed = torch.cat(new_x_embed, dim=1)
                
#                 # 将处理好的 embedding 塞回 new_state_dict，对应的 key 要和当前模型一致
#                 # 我们需要在加载完其他权重后，手动赋值给 self.rdt_model.x_pos_embed
#                 # 所以这里先不放入 load_state_dict，或者放入但确保 shape 匹配
#                 # 为了简单，我们在 load_state_dict 之后手动赋值
            
#             # --- [常规加载] ---
#             for k, v in state_dict.items():
#                 k_clean = k.replace("module.", "")
#                 if k_clean in current_model_dict:
#                     # 跳过 x_pos_embed (后面手动处理)
#                     if "x_pos_embed" in k_clean: continue
                    
#                     # 适配 img_cond_pos_embed (4096 -> 64)
#                     if "img_cond_pos_embed" in k_clean:
#                          target_len = 64 # 这里的 64 是 FusionEncoder 池化后的长度
#                          if v.shape[1] > target_len:
#                              # 图像是空间/词袋特征，直接切片影响较小
#                              v = v[:, :target_len, :]
                    
#                     if v.shape == current_model_dict[k_clean].shape:
#                         new_state_dict[k_clean] = v
            
#             # 加载匹配的权重
#             self.rdt_model.load_state_dict(new_state_dict, strict=False)
            
#             # --- [应用手术结果] ---
#             if ckpt_x_embed is not None:
#                 # 确保 Parameter 类型正确
#                 self.rdt_model.x_pos_embed = nn.Parameter(final_x_embed)
#                 # ⚠️ 关键：确保它是可训练的！LoRA 通常只训练 Linear，
#                 # 但我们需要这个 Embedding 适应新的 State 和 Horizon
#                 self.rdt_model.x_pos_embed.requires_grad = True
#                 logger.info("✅ x_pos_embed surgery complete & set to trainable.")

#         # 6. Double Check Img Embed
#         DUBBY_IMG_LEN = 64 
#         if hasattr(self.rdt_model, 'img_cond_pos_embed'):
#              if self.rdt_model.img_cond_pos_embed.shape[1] != DUBBY_IMG_LEN:
#                  logger.info(f"📉 Resizing img_cond_pos_embed to {DUBBY_IMG_LEN}")
#                  old_pe = self.rdt_model.img_cond_pos_embed.data
#                  new_pe = nn.Parameter(old_pe[:, :DUBBY_IMG_LEN, :].clone())
#                  self.rdt_model.img_cond_pos_embed = new_pe

#     def _load_config_and_override(self, config_path, target_action_dim, pred_horizon):
#         class Args: pass
#         args = Args()
#         if config_path and os.path.exists(config_path):
#             with open(config_path, 'r') as f: cfg = json.load(f)
#             if 'rdt' in cfg and 'hidden_size' in cfg['rdt']: self.rdt_hidden_size = int(cfg['rdt']['hidden_size'])
#             elif 'hidden_size' in cfg: self.rdt_hidden_size = int(cfg['hidden_size'])
#             for k, v in cfg.items():
#                 if k == 'rdt' and isinstance(v, dict):
#                     for sk, sv in v.items(): setattr(args, sk, sv)
#                 setattr(args, k, v)
#         args.action_dim = int(target_action_dim)
#         args.output_dim = int(target_action_dim)
#         args.horizon = int(pred_horizon)
#         args.pred_horizon = int(pred_horizon)
#         if not hasattr(args, 'patch_size'): args.patch_size = 1
#         if not hasattr(args, 'img_size'): args.img_size = 1
#         if not hasattr(args, 'num_frames'): args.num_frames = 1
#         return args

#     def forward(self, noisy_action, timestep, conditions):
#         B = noisy_action.shape[0]
#         device = noisy_action.device
#         dtype = self.action_proj.weight.dtype

#         # 1. 视觉
#         e_t = conditions['e_t'] 
#         img_c = self.visual_proj(e_t.to(dtype)) 

#         # 2. 状态
#         current_state = conditions.get('state') 
#         if current_state is None:
#              current_state = torch.zeros((B, 1, 8), device=device, dtype=dtype)
#         if current_state.dim() == 2: current_state = current_state.unsqueeze(1)
#         state_embed = self.state_proj(current_state.to(dtype)) 

#         # 3. 动作
#         if noisy_action.dim() == 2: x_in = noisy_action.unsqueeze(1)
#         else: x_in = noisy_action
#         action_embed = self.action_proj(x_in.to(dtype))
        
#         # 拼接: [Time(由RDT内部加), Freq(由RDT内部加), State, Action]
#         # 注意：RDT.forward 内部会自己在最前面加上 Time 和 Freq token
#         # 所以我们需要传进去的是 [State, Action]
#         # x_pos_embed 长度是 19 (1+1+1+16)
#         # 内部逻辑是: x = cat(t, freq, x) -> shape becomes 2 + (1+16) = 19
#         # 然后 x + x_pos_embed
        
#         x_input = torch.cat([state_embed, action_embed], dim=1) # [B, 1+16, D]

#         # 4. 其他
#         lang_c = torch.zeros((B, 1, self.rdt_hidden_size), device=device, dtype=dtype)
#         img_mask = torch.ones((B, img_c.shape[1]), device=device, dtype=torch.long)
#         lang_mask = torch.ones((B, 1), device=device, dtype=torch.long)
#         freq = torch.full((B,), 30, device=device, dtype=torch.long)

#         return self.rdt_model(
#             x=x_input, 
#             freq=freq, 
#             t=timestep, 
#             lang_c=lang_c, 
#             img_c=img_c,
#             lang_mask=lang_mask,
#             img_mask=img_mask
#         )

import torch
import torch.nn as nn
import os
import sys
import json
import logging
import inspect
import importlib.util

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("[RDTWrapper]")

# =========================================================================
# 1. 基础补丁 (保持不变)
# =========================================================================
def force_to_int(val):
    try:
        if hasattr(val, 'item'): val = val.item()
        return int(val)
    except:
        return 0

OriginalLinearInit = torch.nn.Linear.__init__
def patched_linear_init(self, in_features, out_features, bias=True, device=None, dtype=None):
    OriginalLinearInit(self, force_to_int(in_features), force_to_int(out_features), bias=bias, device=device, dtype=dtype)
torch.nn.Linear.__init__ = patched_linear_init

# =========================================================================
# 2. 加载 RDT 源码
# =========================================================================
RDT_ROOT = "/yanghaochuan/projects/RoboticsDiffusionTransformer"
RDT_MODELS_DIR = os.path.join(RDT_ROOT, "models")
if RDT_ROOT not in sys.path: sys.path.insert(0, RDT_ROOT)
if RDT_MODELS_DIR not in sys.path: sys.path.insert(0, RDT_MODELS_DIR)
if "models" in sys.modules and RDT_MODELS_DIR not in sys.modules["models"].__path__:
    sys.modules["models"].__path__.append(RDT_MODELS_DIR)

TARGET_FILE_PATH = os.path.join(RDT_ROOT, "models", "rdt", "model.py")
ModelClass = None
if os.path.exists(TARGET_FILE_PATH):
    try:
        spec = importlib.util.spec_from_file_location("rdt_source_model", TARGET_FILE_PATH)
        rdt_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rdt_module)
        for name, obj in inspect.getmembers(rdt_module):
            if inspect.isclass(obj) and issubclass(obj, nn.Module):
                if any(k in name for k in ["Transformer", "RDT", "Model"]) and "Layer" not in name:
                    ModelClass = obj
                    break
        if ModelClass: logger.info(f"✅ Locked Model Class: {ModelClass.__name__}")
    except Exception as e:
        logger.error(f"❌ Load failed: {e}")

# =========================================================================
# 3. RDTWrapper (修复 Config 读取 + 维度对齐)
# =========================================================================
class RDTWrapper(nn.Module):
    def __init__(self, 
                 action_dim=8, 
                 model_path='/yanghaochuan/models/rdt-1b',
                 rdt_cond_dim=768,  # <--- 你的 FusionEncoder 输出是 768
                 pred_horizon=16):
        super().__init__()
        if ModelClass is None: raise RuntimeError("RDT Class not found")

        # 1. 精确读取 Config
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path): config_path = os.path.join(model_path, "config.yaml")
        
        # 显式构造参数字典
        kwargs = self._parse_config_robust(config_path)
        
        # ⚠️ 强制修正：Config 文件虽然是对的，但为了防止代码读错，这里强制写死 2048
        kwargs['hidden_size'] = 2048
        kwargs['action_dim'] = int(action_dim)
        kwargs['output_dim'] = int(action_dim)
        kwargs['horizon'] = int(pred_horizon)
        kwargs['pred_horizon'] = int(pred_horizon)
        
        logger.info(f"🛠️  Forcing RDT Init: hidden_size=2048, action_dim={action_dim}")

        # 2. 实例化模型
        try:
            self.rdt_model = ModelClass(**kwargs)
        except:
            # 备用方案：传对象
            class Args: pass
            args = Args()
            for k, v in kwargs.items(): setattr(args, k, v)
            self.rdt_model = ModelClass(args)

        # 3. 维度检查
        actual_dim = 0
        for m in self.rdt_model.modules():
            if isinstance(m, nn.Linear):
                actual_dim = m.out_features
                break
        
        if actual_dim != 2048:
            logger.error(f"❌ FATAL: Model initialized as {actual_dim}, expected 2048!")
            # 暴力修正（虽然很少见需要这样做）
            self.rdt_model.hidden_size = 2048
        
        self.rdt_hidden_size = 2048

        # 4. 建立投影层 (Project 768 -> 2048)
        # 这是解决你训练导致的 768 维问题的关键
        self.visual_proj = nn.Linear(int(rdt_cond_dim), 2048)
        self.action_proj = nn.Linear(int(action_dim), 2048)

        logger.info(f"🏗️ Projections: Visual(768->2048), Action({action_dim}->2048)")

        # 5. 加载权重 + 手术
        self._load_and_surgically_fix_weights(model_path, pred_horizon)

        # 6. 修正内部参数
        if hasattr(self.rdt_model, 'img_cond_pos_embed'):
            # 缩小内部 img pos embed 以避免不必要的计算或报错
            if self.rdt_model.img_cond_pos_embed.shape[1] > 2:
                old = self.rdt_model.img_cond_pos_embed.data
                self.rdt_model.img_cond_pos_embed = nn.Parameter(old[:, :2, :].clone())

    def _parse_config_robust(self, path):
        """专门针对你的 config 结构进行解析"""
        with open(path, 'r') as f: cfg = json.load(f)
        kwargs = {}
        # 1. 先把外层参数拿进来
        for k, v in cfg.items():
            if k != 'rdt': kwargs[k] = v
        
        # 2. 重点解析 'rdt' 内部参数，并覆盖外层
        # 你的 config 里 hidden_size 在 rdt 下面，所以这一步至关重要
        if 'rdt' in cfg and isinstance(cfg['rdt'], dict):
            for k, v in cfg['rdt'].items():
                kwargs[k] = v
        
        # 3. 补充默认值
        kwargs.setdefault('patch_size', 14)
        kwargs.setdefault('img_size', 224)
        
        return kwargs

    def _load_and_surgically_fix_weights(self, model_path, pred_horizon):
        weights_path = os.path.join(model_path, "pytorch_model.bin")
        if not os.path.exists(weights_path): 
            weights_path = os.path.join(model_path, "diffusion_pytorch_model.bin")
        
        if not os.path.exists(weights_path): return

        logger.info("Loading weights...")
        state_dict = torch.load(weights_path, map_location="cpu")
        
        # 1. 过滤不匹配的权重
        current_dict = self.rdt_model.state_dict()
        new_dict = {}
        ckpt_x_embed = None

        for k, v in state_dict.items():
            if "x_pos_embed" in k:
                ckpt_x_embed = v
                continue
            
            k_clean = k.replace("module.", "")
            if k_clean in current_dict:
                # 图像位置编码截断
                if "img_cond_pos_embed" in k_clean and v.shape[1] > 2:
                    v = v[:, :2, :]
                
                if v.shape == current_dict[k_clean].shape:
                    new_dict[k_clean] = v
        
        self.rdt_model.load_state_dict(new_dict, strict=False)

        # 2. 修复 x_pos_embed
        if ckpt_x_embed is not None:
            # 确保 embedding 也是 2048 维
            if ckpt_x_embed.shape[-1] != 2048:
                logger.warning("Checkpoint dimensions weird, skipping x_pos_embed fix.")
                return

            parts = []
            parts.append(ckpt_x_embed[:, 0:1, :]) # Time
            parts.append(ckpt_x_embed[:, 1:2, :]) # Freq
            parts.append(torch.zeros(1, 1, 2048))  # State (New)
            
            # Actions
            start = 2
            avail = ckpt_x_embed.shape[1] - start
            take = min(avail, pred_horizon)
            parts.append(ckpt_x_embed[:, start : start+take, :])
            
            if take < pred_horizon:
                parts.append(torch.zeros(1, pred_horizon - take, 2048))
            
            final_embed = torch.cat(parts, dim=1)
            self.rdt_model.x_pos_embed = nn.Parameter(final_embed)
            logger.info("✅ x_pos_embed fixed (Time+Freq+State+Actions).")

    def forward(self, noisy_action, timestep, conditions):
        B = noisy_action.shape[0]
        device = noisy_action.device
        dtype = self.action_proj.weight.dtype

        # 1. Visual (768 -> 2048)
        e_t = conditions['e_t'] 
        img_c = self.visual_proj(e_t.to(dtype))

        # 2. State (Dummy -> 2048)
        state_embed = torch.zeros((B, 1, 2048), device=device, dtype=dtype)

        # 3. Action (8 -> 2048)
        if noisy_action.dim() == 2: noisy_action = noisy_action.unsqueeze(1)
        action_embed = self.action_proj(noisy_action.to(dtype))

        # 4. Concat Input
        x_input = torch.cat([state_embed, action_embed], dim=1)

        # 5. Others
        lang_c = torch.zeros((B, 1, 2048), device=device, dtype=dtype)
        lang_mask = torch.ones((B, 1), device=device, dtype=torch.long)
        
        target_img_len = self.rdt_model.img_cond_pos_embed.shape[1]
        if img_c.shape[1] > target_img_len:
            img_c = img_c[:, :target_img_len, :]
        img_mask = torch.ones((B, img_c.shape[1]), device=device, dtype=torch.long)
        
        freq = torch.full((B,), 30, device=device, dtype=torch.long)

        # 6. Forward
        return self.rdt_model(
            x=x_input, freq=freq, t=timestep, 
            lang_c=lang_c, img_c=img_c, 
            lang_mask=lang_mask, img_mask=img_mask
        )

    def save_pretrained(self, save_directory):
        self.rdt_model.save_pretrained(save_directory)
        torch.save(self.visual_proj.state_dict(), os.path.join(save_directory, "visual_proj.bin"))
        torch.save(self.action_proj.state_dict(), os.path.join(save_directory, "action_proj.bin"))

    def load_pretrained_projections(self, save_directory):
        p_vis = os.path.join(save_directory, "visual_proj.bin")
        p_act = os.path.join(save_directory, "action_proj.bin")
        if os.path.exists(p_vis): self.visual_proj.load_state_dict(torch.load(p_vis))
        if os.path.exists(p_act): self.action_proj.load_state_dict(torch.load(p_act))