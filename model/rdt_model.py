import sys
import os
import torch
import torch.nn as nn
import logging
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("[RDTWrapper]")

# =========================================================================
# 1. 环境配置 (确保能找到 RDT 库)
# =========================================================================
RDT_ROOT = "/yanghaochuan/projects/RoboticsDiffusionTransformer"

if RDT_ROOT not in sys.path:
    sys.path.insert(0, RDT_ROOT)

# 基础补丁
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
# 2. 导入 RDT 模型类
# =========================================================================
ModelClass = None
try:
    from models.rdt.model import MultimodalDiffusionTransformer
    ModelClass = MultimodalDiffusionTransformer
    logger.info("✅ Successfully imported MultimodalDiffusionTransformer.")
except ImportError:
    try:
        from models.rdt.model import RDT
        ModelClass = RDT
        logger.info("✅ Successfully imported RDT.")
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}. Check RDT_ROOT.")
        raise RuntimeError("Could not load RDT Model Class.")

# =========================================================================
# 3. RDTWrapper (形状完美匹配版)
# =========================================================================
class RDTWrapper(nn.Module):
    def __init__(self, 
                 action_dim=8, 
                 model_path='/yanghaochuan/models/rdt-1b',
                 rdt_cond_dim=768, 
                 pred_horizon=64):
        super().__init__()
        
        # 1. 加载 Config 文件
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path): config_path = os.path.join(model_path, "config.yaml")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found at {config_path}")

        with open(config_path, 'r') as f:
            raw_cfg = json.load(f)

        # 提取 rdt 内部配置
        rdt_cfg = raw_cfg.get('rdt', raw_cfg)

        # 2. 构造精准参数字典
        model_kwargs = {
            "hidden_size": 2048, 
            "output_dim": int(action_dim),
            
            # [Input Length Logic]
            # State(1) + Action(64) = 65 (Input Sequence)
            # RDT adds Time(1) + Freq(1) internally -> Total Pos Embed needed = 67
            # RDT defines x_pos_embed as horizon + 2.
            # So horizon must be 65.
            "horizon": int(pred_horizon) + 1,
            
            "depth": int(rdt_cfg.get("depth", 28)),
            "num_heads": int(rdt_cfg.get("num_heads", 16)),
            "max_lang_cond_len": int(raw_cfg.get("max_lang_cond_len", 1024)),
            "img_cond_len": int(raw_cfg.get("img_cond_len", 4096)),
            "lang_pos_embed_config": raw_cfg.get("lang_pos_embed_config", None),
            "img_pos_embed_config": raw_cfg.get("img_pos_embed_config", None),
            "dtype": torch.float32 
        }

        # 3. 初始化模型
        logger.info(f"🛠️ Init RDT: hidden_size={model_kwargs['hidden_size']}, horizon={model_kwargs['horizon']}")
        
        try:
            self.rdt_model = ModelClass(**model_kwargs)
        except TypeError as e:
            raise RuntimeError(f"Model Init failed. Error: {e}")

        self.rdt_hidden_size = 2048
        
        # 4. 验证维度
        actual_dim = 0
        for m in self.rdt_model.modules():
            if isinstance(m, nn.Linear):
                actual_dim = m.out_features
                break
        logger.info(f"🔍 Actual Initialized Dimension: {actual_dim}")

        # 5. 建立投影层
        self.visual_proj = nn.Linear(int(rdt_cond_dim), 2048)
        self.action_proj = nn.Linear(int(action_dim), 2048)
        self.state_proj = nn.Linear(8, 2048) 

        # 6. 加载权重
        self._load_weights_correctly(model_path)

        # 7. 修正 img_pos_embed
        if hasattr(self.rdt_model, 'img_cond_pos_embed'):
            if self.rdt_model.img_cond_pos_embed.shape[1] > 2:
                old = self.rdt_model.img_cond_pos_embed.data
                self.rdt_model.img_cond_pos_embed = nn.Parameter(old[:, :2, :].clone())

    def _load_weights_correctly(self, model_path):
        weights_path = os.path.join(model_path, "pytorch_model.bin")
        if not os.path.exists(weights_path): 
            weights_path = os.path.join(model_path, "diffusion_pytorch_model.bin")
        
        if not os.path.exists(weights_path): 
            logger.warning("No weights found!")
            return

        logger.info(f"Loading weights from {weights_path}...")
        try:
            state_dict = torch.load(weights_path, map_location="cpu")
        except:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)

        new_dict = {}
        current_keys = self.rdt_model.state_dict().keys()

        for k, v in state_dict.items():
            k_clean = k
            if k.startswith("model."):
                k_clean = k[6:] 
            elif k.startswith("module."):
                k_clean = k[7:]
            
            if "img_cond_pos_embed" in k_clean and v.shape[1] > 2:
                v = v[:, :2, :]

            if k_clean in current_keys:
                target_shape = self.rdt_model.state_dict()[k_clean].shape
                if v.shape == target_shape:
                    new_dict[k_clean] = v
            
        missing, unexpected = self.rdt_model.load_state_dict(new_dict, strict=False)
        logger.info(f"Weights loaded. Missing keys: {len(missing)}")
        if "x_pos_embed" not in new_dict:
            logger.warning("⚠️ Critical: x_pos_embed was NOT loaded!")
        else:
            logger.info("✅ x_pos_embed successfully loaded.")

    def forward(self, noisy_action, timestep, conditions):
        B = noisy_action.shape[0]
        device = noisy_action.device
        dtype = self.action_proj.weight.dtype

        # 1. Visual
        e_t = conditions['e_t'] 
        img_c = self.visual_proj(e_t.to(dtype))

        # 2. State
        state = conditions.get('state', torch.zeros((B, 8), device=device, dtype=dtype))
        if state.dim() == 2: state = state.unsqueeze(1)
        state_embed = self.state_proj(state.to(dtype))

        # 3. Action
        if noisy_action.dim() == 2: noisy_action = noisy_action.unsqueeze(1)
        action_embed = self.action_proj(noisy_action.to(dtype))

        # 4. Concat: [State(1), Action(N)] -> Total Length = 65
        x_input = torch.cat([state_embed, action_embed], dim=1)

        # 5. Mask Setup (bool)
        lang_c = torch.zeros((B, 1, 2048), device=device, dtype=dtype)
        # False = Valid (Attend), True = Masked
        lang_mask = torch.zeros((B, 1), device=device, dtype=torch.bool)
        
        target_img_len = self.rdt_model.img_cond_pos_embed.shape[1]
        if img_c.shape[1] > target_img_len:
            img_c = img_c[:, :target_img_len, :]
        
        img_mask = torch.zeros((B, img_c.shape[1]), device=device, dtype=torch.bool)
        
        freq = torch.full((B,), 30, device=device, dtype=torch.long)

        # 6. Forward
        pred = self.rdt_model(
            x=x_input, freq=freq, t=timestep, 
            lang_c=lang_c, img_c=img_c, 
            lang_mask=lang_mask, img_mask=img_mask
        )
        
        # [关键修复] Output shape is [B, 65, 8] (State + Actions)
        # We only need predictions for Actions [B, 64, 8]
        # Slice off the first token (State)
        return pred[:, 1:, :]

    def save_pretrained(self, save_directory):
        self.rdt_model.save_pretrained(save_directory)
        torch.save(self.visual_proj.state_dict(), os.path.join(save_directory, "visual_proj.bin"))
        torch.save(self.action_proj.state_dict(), os.path.join(save_directory, "action_proj.bin"))
        torch.save(self.state_proj.state_dict(), os.path.join(save_directory, "state_proj.bin"))

    def load_pretrained_projections(self, save_directory):
        p_vis = os.path.join(save_directory, "visual_proj.bin")
        p_act = os.path.join(save_directory, "action_proj.bin")
        p_sta = os.path.join(save_directory, "state_proj.bin")
        if os.path.exists(p_vis): self.visual_proj.load_state_dict(torch.load(p_vis))
        if os.path.exists(p_act): self.action_proj.load_state_dict(torch.load(p_act))
        if os.path.exists(p_sta): self.state_proj.load_state_dict(torch.load(p_sta))