# models/fusion_encoder.py
import torch
import torch.nn as nn
import os
from transformers import AutoModel, AutoConfig, T5EncoderModel

class FiLMLayer(nn.Module):
    def __init__(self, input_dim, cond_dim):
        super().__init__()
        self.layer = nn.Linear(cond_dim, input_dim * 2)

    def forward(self, x, cond):
        gamma_beta = self.layer(cond).unsqueeze(1)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return x * (1 + gamma) + beta

class TaskBackgroundRouting(nn.Module):
    def __init__(self, embed_dim, num_task_slots, text_embed_dim):
        super().__init__()
        self.task_slot_queries = nn.Parameter(torch.randn(num_task_slots, embed_dim))
        self.task_attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.task_norm = nn.LayerNorm(embed_dim)
        self.background_attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.background_norm = nn.LayerNorm(embed_dim)
        self.confidence_head = nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid())

    def forward(self, tokens, first_frame_summary, text_tokens):
        if text_tokens.dim() == 2: 
            text_tokens = text_tokens.unsqueeze(1)
        kv = torch.cat([tokens, text_tokens], dim=1)
        B = tokens.shape[0]
        task_queries = self.task_slot_queries.unsqueeze(0).expand(B, -1, -1)
        task_slots, _ = self.task_attention(query=task_queries, key=kv, value=kv)
        task_slots = self.task_norm(task_slots)
        confidence = self.confidence_head(task_slots)
        background_context, _ = self.background_attention(query=first_frame_summary, key=tokens, value=tokens)
        background_context = self.background_norm(background_context)
        return task_slots, confidence, background_context

class FusionEncoder(nn.Module):
    def __init__(self,
                 rdt_dim=768,
                 num_frames=16,
                 num_task_slots=8,
                 state_dim=8,
                 # === 核心修改：适配 SigLIP so400m 的 1152 维输出 ===
                 teacher_dim=1152, 
                 backbone_path='/yanghaochuan/models/VideoMAEv2-Large',
                 text_model_path='/yanghaochuan/models/flan-t5-large'):
        super().__init__()
        
        print(f"[FusionEncoder] Loading VideoMAE from: {backbone_path}")
        self.config = AutoConfig.from_pretrained(backbone_path, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(
            backbone_path, 
            trust_remote_code=True,
            config=self.config
        )

        if hasattr(self.config, 'hidden_size'):
            vision_dim = self.config.hidden_size
        elif hasattr(self.config, 'embed_dim'):
            vision_dim = self.config.embed_dim
        else:
            vision_dim = 1024 

        print(f"[FusionEncoder] Loading T5 Encoder from: {text_model_path}")
        try:
            self.text_encoder = T5EncoderModel.from_pretrained(text_model_path, use_safetensors=True)
        except Exception:
            try:
                config = AutoConfig.from_pretrained(text_model_path)
                self.text_encoder = T5EncoderModel(config)
                bin_path = os.path.join(text_model_path, "pytorch_model.bin")
                if os.path.exists(bin_path):
                    state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)
                    self.text_encoder.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"[FusionEncoder] Using Random Init T5 (Warning): {e}")
                config = AutoConfig.from_pretrained(text_model_path)
                self.text_encoder = T5EncoderModel(config)

        if self.text_encoder:
            for param in self.text_encoder.parameters(): param.requires_grad = False
            text_embed_dim = self.text_encoder.config.d_model
        else:
            text_embed_dim = 1024

        self.film_t5 = FiLMLayer(vision_dim, text_embed_dim)
        self.film_state = FiLMLayer(vision_dim, state_dim)
        self.cross_attention_first_frame = nn.MultiheadAttention(vision_dim, num_heads=16, batch_first=True)

        self.routing_layer = TaskBackgroundRouting(
            embed_dim=vision_dim,
            num_task_slots=num_task_slots,
            text_embed_dim=text_embed_dim
        )

        # === 核心修改：输出头映射到 Teacher 维度 (1152) ===
        print(f"[FusionEncoder] Aligning heads to teacher dimension: {teacher_dim}")
        # self.semantic_align_head = nn.Linear(vision_dim, teacher_dim)
        # self.temporal_align_head = nn.Linear(vision_dim, teacher_dim)
        
        # self.projection_head = nn.Linear(vision_dim, rdt_dim)

        self.projection_head = nn.Sequential(
            nn.Dropout(p=0.2), # 20% 的概率丢弃，强迫模型学鲁棒特征
            nn.Linear(vision_dim, rdt_dim)
        )
        
        # 语义对齐头也可以加
        self.semantic_align_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(vision_dim, teacher_dim)
        )
        
        # 时序对齐头也可以加
        self.temporal_align_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(vision_dim, teacher_dim)
        )
        
        self.norm1 = nn.LayerNorm(vision_dim)
        self.norm2 = nn.LayerNorm(vision_dim)

    def extract_features(self, inputs):
        """[Atomic Path] 保留之前的核动力提取逻辑"""
        model = self.backbone
        if hasattr(model, 'model'): model = model.model
        if hasattr(model, 'vit'): model = model.vit 
        
        if hasattr(model, 'blocks') and hasattr(model, 'patch_embed'):
            try:
                x = model.patch_embed(inputs)
                if hasattr(model, 'pos_embed') and model.pos_embed is not None:
                    pos_embed = model.pos_embed.to(x.device)
                    if x.shape[1] == pos_embed.shape[1]:
                        x = x + pos_embed
                    else:
                        x = x + pos_embed[:, :x.size(1), :]
                for blk in model.blocks:
                    x = blk(x)
                if hasattr(model, 'norm') and model.norm is not None:
                    x = model.norm(x)
                return x 
            except Exception:
                pass

        try:
            outputs = self.backbone(inputs, output_hidden_states=True)
            if hasattr(outputs, 'last_hidden_state'): return outputs.last_hidden_state
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states: return outputs.hidden_states[-1]
            return outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        except TypeError:
            return self.backbone(inputs)

    def forward(self, video_frames, text_tokens, state_info, first_frame_summary):
        if video_frames.shape[1] != 3 and video_frames.shape[2] == 3:
            video_frames = video_frames.permute(0, 2, 1, 3, 4)
        T_video = video_frames.shape[2]

        tokens = self.extract_features(video_frames)
        if tokens.dim() == 2: tokens = tokens.unsqueeze(1) 

        if self.text_encoder is not None:
            with torch.no_grad():
                text_outputs = self.text_encoder(input_ids=text_tokens)
                text_embeds = text_outputs.last_hidden_state
            text_cond = text_embeds.mean(dim=1) 
        else:
            text_embeds = torch.zeros(tokens.shape[0], 10, 1024, device=tokens.device)
            text_cond = torch.zeros(tokens.shape[0], 1024, device=tokens.device)

        tokens = self.film_t5(tokens, text_cond)
        state_cond = state_info[:, -1, :] 
        tokens = self.film_state(tokens, state_cond)

        if first_frame_summary.dim() == 5:
            ff_input = first_frame_summary.transpose(1, 2).repeat(1, 1, T_video, 1, 1)
            with torch.no_grad():
                ff_tokens = self.extract_features(ff_input)
                if ff_tokens.dim() == 3: first_frame_summary = ff_tokens.mean(dim=1, keepdim=True)
                elif ff_tokens.dim() == 2: first_frame_summary = ff_tokens.unsqueeze(1)

        attn_output, _ = self.cross_attention_first_frame(query=tokens, key=first_frame_summary, value=first_frame_summary)
        tokens = self.norm1(tokens + attn_output)

        task_slots, confidence, background_context = self.routing_layer(tokens, first_frame_summary, text_embeds)

        global_rep = torch.mean(tokens, dim=1)
        semantic_out = self.semantic_align_head(global_rep)
        temporal_out = self.temporal_align_head(global_rep)

        weighted_task = torch.sum(task_slots * confidence, dim=1)
        fused = self.norm2(global_rep + weighted_task)
        e_t = self.projection_head(fused)

        return {
            "e_t": e_t,
            "task_slots": task_slots,
            "task_confidence": confidence,
            "background_context": background_context,
            "semantic_head_output": semantic_out,
            "temporal_head_output": temporal_out
        }