# models/rdt_model.py
import torch
import torch.nn as nn
import os
import sys
import json
import yaml
import importlib.util
import inspect
import numbers

# =========================================================================
# 1. 强力参数清洗工具 & 补丁
# =========================================================================
def force_to_int(val):
    try:
        if val is None: return None
        if isinstance(val, (tuple, list)):
            val = val[0] if len(val) >= 1 else 0
        if hasattr(val, 'item'): val = val.item()
        if hasattr(val, 'dtype'): val = int(val)
        if isinstance(val, float): val = int(val)
        if isinstance(val, int): return val
        try: return int(val)
        except: return val 
    except:
        return val

OriginalLinearInit = torch.nn.Linear.__init__
def patched_linear_init(self, in_features, out_features, bias=True, device=None, dtype=None):
    safe_in = force_to_int(in_features)
    safe_out = force_to_int(out_features)
    if not isinstance(safe_out, int):
        try: safe_out = int(safe_out.out_channels) 
        except: safe_out = 8 
    OriginalLinearInit(self, safe_in, safe_out, bias=bias, device=device, dtype=dtype)
torch.nn.Linear.__init__ = patched_linear_init

try:
    import timm.models.layers
    if hasattr(timm.models.layers, 'Mlp'):
        OriginalTimmMlpInit = timm.models.layers.Mlp.__init__
        def patched_timm_init(self, in_features, hidden_features=None, out_features=None, *args, **kwargs):
            return OriginalTimmMlpInit(self, force_to_int(in_features), force_to_int(hidden_features), force_to_int(out_features), *args, **kwargs)
        timm.models.layers.Mlp.__init__ = patched_timm_init
except: pass

try:
    import timm.layers
    if hasattr(timm.layers, 'Mlp'):
        OriginalLayerMlpInit = timm.layers.Mlp.__init__
        def patched_layer_init(self, in_features, hidden_features=None, out_features=None, *args, **kwargs):
            return OriginalLayerMlpInit(self, force_to_int(in_features), force_to_int(hidden_features), force_to_int(out_features), *args, **kwargs)
        timm.layers.Mlp.__init__ = patched_layer_init
except: pass

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
        sys.modules["rdt_source_model"] = rdt_module
        spec.loader.exec_module(rdt_module)
        
        candidate_classes = []
        for name, obj in inspect.getmembers(rdt_module):
            if inspect.isclass(obj) and issubclass(obj, nn.Module):
                if any(k in name for k in ["Transformer", "RDT", "Model"]):
                    if not any(k in name for k in ["Layer", "Block", "Attention", "Embed", "Head", "MLP", "Timestep"]):
                        candidate_classes.append(obj)
        if candidate_classes:
            candidate_classes.sort(key=lambda x: len(x.__name__), reverse=True)
            ModelClass = candidate_classes[0]
            print(f"[RDTWrapper] ✅ 成功锁定模型类: {ModelClass.__name__}")
        else:
            print(f"[RDTWrapper] ❌ 未找到主模型类")
    except Exception as e:
        print(f"[RDTWrapper] ❌ 导入 model.py 失败: {e}")

# =========================================================================
# 3. RDTWrapper 类定义
# =========================================================================
class RDTWrapper(nn.Module):
    def __init__(self, 
                 action_dim=8, 
                 model_path='/yanghaochuan/models/rdt-1b',
                 rdt_cond_dim=1152,
                 pred_horizon=16):
        super().__init__()
        if ModelClass is None: raise RuntimeError("无法初始化 RDT")

        # 1. Config
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path): config_path = os.path.join(model_path, "config.yaml")
        print(f"[RDTWrapper] Loading config from: {config_path}")
        
        self.rdt_hidden_size = 2048 
        # 调用内部方法加载配置
        args = self._load_config_and_override(config_path, action_dim)

        # 2. Instantiate
        print(f"[RDTWrapper] Instantiating with forced horizon={args.horizon}")
        try:
            sig = inspect.signature(ModelClass.__init__)
            params = list(sig.parameters.keys())
            if 'output_dim' not in vars(args): args.output_dim = args.action_dim
            valid_args = {k: v for k, v in vars(args).items() if k in params or 'kwargs' in str(sig)}
            self.rdt_model = ModelClass(**valid_args)
            print("[RDTWrapper] Instantiation successful via kwargs unpacking.")
        except Exception as e:
            print(f"[RDTWrapper] Kwargs instantiation failed: {e}. Falling back to object pass...")
            self.rdt_model = ModelClass(args)

        # 3. Detect ACTUAL Hidden Size
        actual_dim = self.rdt_hidden_size
        if hasattr(self.rdt_model, 'hidden_size'): actual_dim = self.rdt_model.hidden_size
        elif hasattr(self.rdt_model, 'embed_dim'): actual_dim = self.rdt_model.embed_dim
        else:
            for m in self.rdt_model.modules():
                if isinstance(m, nn.Linear):
                    actual_dim = m.out_features
                    break
        print(f"[RDTWrapper] 🔍 Detected Actual Hidden Dimension: {actual_dim}")
        
        # 4. Load Weights (Smart Loading with Adaptation)
        weights_path = os.path.join(model_path, "pytorch_model.bin")
        if not os.path.exists(weights_path): weights_path = os.path.join(model_path, "diffusion_pytorch_model.bin")
        
        if os.path.exists(weights_path):
            print(f"[RDTWrapper] Loading weights with schema adaptation...")
            try:
                state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
            except TypeError:
                state_dict = torch.load(weights_path, map_location="cpu")
            
            new_state_dict = {}
            current_model_dict = self.rdt_model.state_dict()
            
            for k, v in state_dict.items():
                if k.startswith("module."): k = k[7:]
                if k in current_model_dict:
                    target_shape = current_model_dict[k].shape
                    
                    # === 适配 1: x_pos_embed (3 tokens vs 4 tokens) ===
                    # 场景：官方权重有 State Token (len 4)，你的模型没有 (len 3)
                    if "x_pos_embed" in k:
                        if v.shape[1] == 4 and target_shape[1] == 3:
                            print(f"[RDTWrapper] ✂️  Slicing x_pos_embed: Removing 'state' token (index 2).")
                            # 官方顺序: [Time, Freq, State, Action] -> 保留 [0, 1, 3]
                            v = v[:, [0, 1, 3], :]
                    
                    # === 适配 2: img_cond_pos_embed (4000+ vs 2) ===
                    # 场景：官方权重巨大，我们只需要 2 个占位符
                    if "img_cond_pos_embed" in k:
                        if v.shape[1] > target_shape[1]:
                            # 直接截取前 N 个，反正我们传的是全 0 占位符
                            v = v[:, :target_shape[1], :]

                    if v.shape != target_shape:
                        # 兜底：如果形状还不匹配，跳过（防止报错）
                        print(f"[RDTWrapper] ⚠️  Skipping {k}: shape mismatch {v.shape} vs {target_shape}")
                        continue
                        
                    new_state_dict[k] = v
            
            self.rdt_model.load_state_dict(new_state_dict, strict=False)

        # 5. Initialize Projection Layers
        target_dim = actual_dim 
        self.action_proj = nn.Linear(int(action_dim), int(target_dim))
        self.cond_proj = nn.Linear(int(rdt_cond_dim), int(target_dim))

        # === 适配 3: 强制调整模型内部 img_pos_embed 大小 ===
        DUBBY_IMG_LEN = 2
        if hasattr(self.rdt_model, 'img_cond_pos_embed'):
             if self.rdt_model.img_cond_pos_embed.shape[1] > DUBBY_IMG_LEN:
                 print(f"[RDTWrapper] 📉 Resizing internal img_cond_pos_embed to length {DUBBY_IMG_LEN}")
                 old_pe = self.rdt_model.img_cond_pos_embed.data
                 new_pe = nn.Parameter(old_pe[:, :DUBBY_IMG_LEN, :].clone())
                 self.rdt_model.img_cond_pos_embed = new_pe

    # =================================================
    # 辅助方法：注意缩进，它必须在 class RDTWrapper 内部
    # =================================================
    def _load_config_and_override(self, config_path, target_action_dim):
        class Args: pass
        args = Args()
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            
            if 'rdt' in cfg and 'hidden_size' in cfg['rdt']:
                self.rdt_hidden_size = int(cfg['rdt']['hidden_size'])
            elif 'hidden_size' in cfg:
                self.rdt_hidden_size = int(cfg['hidden_size'])
                
            for k, v in cfg.items():
                if k == 'rdt' and isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        setattr(args, sub_k, sub_v)
                setattr(args, k, v)
                
        args.action_dim = int(target_action_dim)
        args.output_dim = int(target_action_dim)
        args.out_channels = int(target_action_dim)
        args.input_size = int(target_action_dim)
        args.in_channels = int(target_action_dim)
        
        args.hidden_size = self.rdt_hidden_size
        args.embed_dim = self.rdt_hidden_size 
        args.d_model = self.rdt_hidden_size
        
        args.horizon = int(pred_horizon)      # 预测未来多少步
        args.pred_horizon = int(pred_horizon)
        
        defaults = {'patch_size': 1, 'img_size': 1, 'num_frames': 1}
        for k, v in defaults.items():
            if not hasattr(args, k): setattr(args, k, v)
            
        return args

    # def forward(self, noisy_action, timestep, conditions):
    #     e_t = conditions['e_t']
    #     cond_embeds = self.cond_proj(e_t).unsqueeze(1) # [B, 1, D]
        
    #     if noisy_action.dim() == 2: x_in = noisy_action
    #     else: x_in = noisy_action.squeeze(1)
            
    #     x_embed = self.action_proj(x_in).unsqueeze(1) # [B, 1, D]
        
    #     B = x_embed.shape[0]
    #     device = x_embed.device
        
    #     freq = torch.full((B,), 30, device=device, dtype=torch.long)
        
    #     lang_c = cond_embeds 
        
    #     # 图像条件：全 0 占位符 (长度为 2)
    #     img_c = torch.zeros((B, 2, cond_embeds.shape[-1]), device=device, dtype=cond_embeds.dtype)
        
    #     lang_mask = torch.ones((B, 1), device=device, dtype=torch.bool)
    #     img_mask = torch.ones((B, 2), device=device, dtype=torch.bool)

    #     return self.rdt_model(
    #         x=x_embed, 
    #         freq=freq, 
    #         t=timestep, 
    #         lang_c=lang_c, 
    #         img_c=img_c,
    #         lang_mask=lang_mask,
    #         img_mask=img_mask
    #     )


    def forward(self, noisy_action, timestep, conditions):
        """
        RDTWrapper 的前向传播 (修改版)
        输入:
            noisy_action: [B, Horizon, Action_Dim] (例如 [B, 16, 8])
            timestep: [B]
            conditions: dict, 包含 'e_t' (序列特征)
        """
        # 1. 获取视觉序列特征
        e_t = conditions['e_t'] # 期望形状: [B, 64, 768] (来自 FusionEncoder 的序列输出)
        
        B = e_t.shape[0]
        device = e_t.device
        dtype = e_t.dtype
        
        # 2. 视觉投影 (768 -> RDT Hidden Size, 通常是 1152)
        # 动态检查并初始化投影层 (Lazy Initialization)，防止 __init__ 没改导致报错
        # 推荐您后续最好把它移到 __init__ 里: self.visual_proj = nn.Linear(768, 1152)
        if not hasattr(self, 'visual_proj'):
            # RDT-1B 的 hidden_size 通常是 2048 (InternViT) 或 1152 (SigLIP/Patch)
            # 这里我们需要映射到 model.img_embedder 期望的维度
            # 简单起见，我们读取 self.rdt_model.config.hidden_size 或直接硬编码 1152 (常见配置)
            # 更稳妥的方式是看 img_c 应该进哪里。RDT 内部通常有 img_adaptor。
            # 这里假设 RDT 内部 img_c 的期望维度是 1152 (SigLIP-So400m 的 dim)
            target_dim = 1152 
            print(f"[RDTWrapper] Lazy initializing visual_proj: {e_t.shape[-1]} -> {target_dim}")
            self.visual_proj = nn.Linear(e_t.shape[-1], target_dim).to(device).to(dtype)
        
        # [B, 64, 768] -> [B, 64, 1152]
        img_c = self.visual_proj(e_t)
        
        # 3. 构造文本条件 (Language Condition)
        # 因为我们主要依赖视觉，这里传入空的文本嵌入 (Zero Padding)
        # RDT 期望 lang_c 形状为 [B, L, D]，这里设 L=1
        lang_c = torch.zeros((B, 1, 1152), device=device, dtype=dtype)
        
        # 4. 构造 Masks
        # img_mask: [B, 64], 全 1 (所有视觉 Token 有效)
        img_mask = torch.ones((B, img_c.shape[1]), device=device, dtype=torch.long)
        
        # lang_mask: [B, 1], 全 1 (表示这是一个有效的"空指令")
        # 注意: RDT 内部对 mask 的处理通常是 bool 或 0/1, long 比较稳妥
        lang_mask = torch.ones((B, 1), device=device, dtype=torch.long)

        # 5. 处理动作输入
        # 如果输入是 [B, 8]，unsqueeze 成 [B, 1, 8]
        # 如果是 [B, 16, 8]，保持不变
        if noisy_action.dim() == 2:
            x_in = noisy_action.unsqueeze(1)
        else:
            x_in = noisy_action
            
        # 投影动作维度 [B, H, 8] -> [B, H, Hidden]
        x_embed = self.action_proj(x_in)
        
        # 6. 控制频率 (Control Frequency)
        # 固定为 30Hz 或从 dataset 统计中获取
        freq = torch.full((B,), 30, device=device, dtype=torch.long)

        # 7. 调用 RDT Backbone
        return self.rdt_model(
            x=x_embed, 
            freq=freq, 
            t=timestep, 
            lang_c=lang_c, 
            img_c=img_c,
            lang_mask=lang_mask,
            img_mask=img_mask
        )