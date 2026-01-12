# # models/fusion_encoder.py
# import torch
# import torch.nn as nn
# import os
# from transformers import AutoModel, AutoConfig, T5EncoderModel

# class FiLMLayer(nn.Module):
#     def __init__(self, input_dim, cond_dim):
#         super().__init__()
#         self.layer = nn.Linear(cond_dim, input_dim * 2)

#     def forward(self, x, cond):
#         gamma_beta = self.layer(cond).unsqueeze(1)
#         gamma, beta = gamma_beta.chunk(2, dim=-1)
#         return x * (1 + gamma) + beta

# class TaskBackgroundRouting(nn.Module):
#     def __init__(self, embed_dim, num_task_slots, text_embed_dim):
#         super().__init__()
#         self.task_slot_queries = nn.Parameter(torch.randn(num_task_slots, embed_dim))
#         self.task_attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
#         self.task_norm = nn.LayerNorm(embed_dim)
#         self.background_attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
#         self.background_norm = nn.LayerNorm(embed_dim)
#         self.confidence_head = nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid())

#     def forward(self, tokens, first_frame_summary, text_tokens):
#         if text_tokens.dim() == 2: 
#             text_tokens = text_tokens.unsqueeze(1)
#         kv = torch.cat([tokens, text_tokens], dim=1)
#         B = tokens.shape[0]
#         task_queries = self.task_slot_queries.unsqueeze(0).expand(B, -1, -1)
#         task_slots, _ = self.task_attention(query=task_queries, key=kv, value=kv)
#         task_slots = self.task_norm(task_slots)
#         confidence = self.confidence_head(task_slots)
#         background_context, _ = self.background_attention(query=first_frame_summary, key=tokens, value=tokens)
#         background_context = self.background_norm(background_context)
#         return task_slots, confidence, background_context

# class FusionEncoder(nn.Module):
#     def __init__(self,
#                  rdt_dim=768,
#                  num_frames=16,
#                  num_task_slots=8,
#                  state_dim=8,
#                  # === 核心修改：适配 SigLIP so400m 的 1152 维输出 ===
#                  teacher_dim=1152, 
#                  backbone_path='/yanghaochuan/models/VideoMAEv2-Large',
#                  text_model_path='/yanghaochuan/models/flan-t5-large'):
#         super().__init__()
        
#         print(f"[FusionEncoder] Loading VideoMAE from: {backbone_path}")
#         self.config = AutoConfig.from_pretrained(backbone_path, trust_remote_code=True)
#         self.backbone = AutoModel.from_pretrained(
#             backbone_path, 
#             trust_remote_code=True,
#             config=self.config
#         )

#         if hasattr(self.config, 'hidden_size'):
#             vision_dim = self.config.hidden_size
#         elif hasattr(self.config, 'embed_dim'):
#             vision_dim = self.config.embed_dim
#         else:
#             vision_dim = 1024 

#         print(f"[FusionEncoder] Loading T5 Encoder from: {text_model_path}")
#         try:
#             self.text_encoder = T5EncoderModel.from_pretrained(text_model_path, use_safetensors=True)
#         except Exception:
#             try:
#                 config = AutoConfig.from_pretrained(text_model_path)
#                 self.text_encoder = T5EncoderModel(config)
#                 bin_path = os.path.join(text_model_path, "pytorch_model.bin")
#                 if os.path.exists(bin_path):
#                     state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)
#                     self.text_encoder.load_state_dict(state_dict, strict=False)
#             except Exception as e:
#                 print(f"[FusionEncoder] Using Random Init T5 (Warning): {e}")
#                 config = AutoConfig.from_pretrained(text_model_path)
#                 self.text_encoder = T5EncoderModel(config)

#         if self.text_encoder:
#             for param in self.text_encoder.parameters(): param.requires_grad = False
#             text_embed_dim = self.text_encoder.config.d_model
#         else:
#             text_embed_dim = 1024

#         self.film_t5 = FiLMLayer(vision_dim, text_embed_dim)
#         self.film_state = FiLMLayer(vision_dim, state_dim)
#         self.cross_attention_first_frame = nn.MultiheadAttention(vision_dim, num_heads=16, batch_first=True)

#         self.routing_layer = TaskBackgroundRouting(
#             embed_dim=vision_dim,
#             num_task_slots=num_task_slots,
#             text_embed_dim=text_embed_dim
#         )

#         # === 核心修改：输出头映射到 Teacher 维度 (1152) ===
#         print(f"[FusionEncoder] Aligning heads to teacher dimension: {teacher_dim}")

#         self.projection_head = nn.Sequential(
#             nn.Dropout(p=0.2), # 20% 的概率丢弃，强迫模型学鲁棒特征
#             nn.Linear(vision_dim, rdt_dim)
#         )
        
#         # 语义对齐头也可以加
#         self.semantic_align_head = nn.Sequential(
#             nn.Dropout(p=0.2),
#             nn.Linear(vision_dim, teacher_dim)
#         )
        
#         # 时序对齐头也可以加
#         self.temporal_align_head = nn.Sequential(
#             nn.Dropout(p=0.2),
#             nn.Linear(vision_dim, teacher_dim)
#         )
        
#         self.norm1 = nn.LayerNorm(vision_dim)
#         self.norm2 = nn.LayerNorm(vision_dim)

#     def extract_features(self, inputs):
#         """[Atomic Path] 保留之前的核动力提取逻辑"""
#         model = self.backbone
#         if hasattr(model, 'model'): model = model.model
#         if hasattr(model, 'vit'): model = model.vit 
        
#         if hasattr(model, 'blocks') and hasattr(model, 'patch_embed'):
#             try:
#                 x = model.patch_embed(inputs)
#                 if hasattr(model, 'pos_embed') and model.pos_embed is not None:
#                     pos_embed = model.pos_embed.to(x.device)
#                     if x.shape[1] == pos_embed.shape[1]:
#                         x = x + pos_embed
#                     else:
#                         x = x + pos_embed[:, :x.size(1), :]
#                 for blk in model.blocks:
#                     x = blk(x)
#                 if hasattr(model, 'norm') and model.norm is not None:
#                     x = model.norm(x)
#                 return x 
#             except Exception:
#                 pass

#         try:
#             outputs = self.backbone(inputs, output_hidden_states=True)
#             if hasattr(outputs, 'last_hidden_state'): return outputs.last_hidden_state
#             if hasattr(outputs, 'hidden_states') and outputs.hidden_states: return outputs.hidden_states[-1]
#             return outputs[0] if isinstance(outputs, (tuple, list)) else outputs
#         except TypeError:
#             return self.backbone(inputs)

#     def forward(self, video_frames, text_tokens, state_info, first_frame_summary):
#         # 1. 调整维度 [B, C, T, H, W] -> [B, T, C, H, W] 

#         B_dim = video_frames.shape[0]
#         is_dual_view = False
        
#         if video_frames.dim() == 6: # [B, V, C, T, H, W]
#             is_dual_view = True
#             B, V, C, T, H, W = video_frames.shape
#             # 合并 Batch 和 View 维度: [B*V, C, T, H, W]
#             video_frames = video_frames.view(B * V, C, T, H, W)

#         if video_frames.shape[1] != 3 and video_frames.shape[2] == 3:
#             video_frames = video_frames.permute(0, 2, 1, 3, 4)
#         T_video = video_frames.shape[2]

#         # 2. 提取基础特征
#         # tokens shape: [B, N_patches, D] (例如 [B, 1568, 1152])
#         tokens = self.extract_features(video_frames)
#         # if tokens.dim() == 2: tokens = tokens.unsqueeze(1) 
#         if is_dual_view:
#             tokens = tokens.view(B, V * tokens.shape[1], tokens.shape[2])
#         # 3. 文本编码 (Text Embedding)
#         if self.text_encoder is not None:
#             with torch.no_grad():
#                 text_outputs = self.text_encoder(input_ids=text_tokens)
#                 text_embeds = text_outputs.last_hidden_state
#             text_cond = text_embeds.mean(dim=1) 
#         else:
#             text_embeds = torch.zeros(tokens.shape[0], 10, 1024, device=tokens.device)
#             text_cond = torch.zeros(tokens.shape[0], 1024, device=tokens.device)

#         # 4. FiLM 调节 (Conditioning)
#         tokens = self.film_t5(tokens, text_cond)
#         state_cond = state_info[:, -1, :] 
#         tokens = self.film_state(tokens, state_cond)

#         # 5. 首帧注意力 (First Frame Cross-Attention)
#         if first_frame_summary.dim() == 5:

#             ff_input = first_frame_summary.transpose(1, 2).repeat(1, 1, 2, 1, 1)
#             with torch.no_grad():
#                 ff_tokens = self.extract_features(ff_input)
#                 if ff_tokens.dim() == 3: first_frame_summary = ff_tokens.mean(dim=1, keepdim=True)
#                 elif ff_tokens.dim() == 2: first_frame_summary = ff_tokens.unsqueeze(1)

#         attn_output, _ = self.cross_attention_first_frame(query=tokens, key=first_frame_summary, value=first_frame_summary)
#         tokens = self.norm1(tokens + attn_output)

#         # 6. 任务路由 (Task Routing)
#         task_slots, confidence, background_context = self.routing_layer(tokens, first_frame_summary, text_embeds)
        
#         # weighted_task: [B, D]
#         weighted_task = torch.sum(task_slots * confidence, dim=1)
        
#         fused_seq = self.norm2(tokens + weighted_task.unsqueeze(1))
        
#         # 2. 先进行自适应池化 (大幅减小数据规模)
#         # [B, N, D] -> [B, D, 64] -> [B, 64, D]
#         fused_pooled = torch.nn.functional.adaptive_avg_pool1d(fused_seq.transpose(1, 2), 64).transpose(1, 2)
        
#         # 3. 最后进行投影 (Dropout + Linear)
#         # 此时 Dropout 作用在最终特征上，效果最好；且 Linear 计算量减少了 ~24 倍
#         e_t = self.projection_head(fused_pooled)
        
#         # 辅助 Loss 头 (保持原样或根据需要调整)
#         global_rep_for_heads = tokens.mean(dim=1)
#         semantic_out = self.semantic_align_head(global_rep_for_heads)
#         temporal_out = self.temporal_align_head(global_rep_for_heads)
#         return {
#             "e_t": e_t, # [B, 64, 768] (Sequence Output for RDT)
#             "task_slots": task_slots,
#             "task_confidence": confidence,
#             "background_context": background_context,
#             "semantic_head_output": semantic_out, # [B, Teacher_Dim]
#             "temporal_head_output": temporal_out  # [B, Teacher_Dim]
#         }

# model/fusion_encoder.py
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
        
        self.norm1 = nn.LayerNorm(vision_dim)
        self.norm2 = nn.LayerNorm(vision_dim)

        # === [New] 1. View Embedding (显式区分视角) ===
        # 假设 0: Main, 1: Wrist
        self.num_views = 2
        self.view_embed = nn.Parameter(torch.zeros(1, self.num_views, 1, vision_dim))
        nn.init.trunc_normal_(self.view_embed, std=0.02)

        # === [New] 2. ForeSight Predictor (世界模型) ===
        # 对应 offsets: [0, 2, 4, 8, 16, 32]
        self.num_future_tokens = 6 
        self.future_queries = nn.Parameter(torch.randn(self.num_future_tokens, vision_dim))
        
        # Decoder: Query=Future, Memory=Ego History
        self.predictor_layer = nn.TransformerDecoderLayer(d_model=vision_dim, nhead=16, batch_first=True)
        self.predictor = nn.TransformerDecoder(self.predictor_layer, num_layers=2)
        
        # === [New] 3. Heads Definition ===
        
        # A. 原始投影头 (用于 64 个 Spatial Token) -> RDT
        self.projection_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(vision_dim, rdt_dim)
        )
        
        # B. 未来投影头 (用于 6 个 Future Latent) -> RDT
        self.future_proj_head = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Dropout(0.2),
            nn.Linear(vision_dim, rdt_dim)
        )
        
        # C. 世界模型对齐头 (用于 Loss) -> Teacher Dim (1152)
        self.wm_align_head = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, teacher_dim)
        )
        
        # D. 辅助对齐头
        self.semantic_align_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(vision_dim, teacher_dim)
        )
        self.temporal_align_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(vision_dim, teacher_dim)
        )

    def extract_features(self, inputs):
        """支持动态插值的 VideoMAE 特征提取"""
        model = self.backbone
        if hasattr(model, 'model'): model = model.model
        if hasattr(model, 'vit'): model = model.vit 
        
        if hasattr(model, 'blocks') and hasattr(model, 'patch_embed'):
            try:
                x = model.patch_embed(inputs)
                if hasattr(model, 'pos_embed') and model.pos_embed is not None:
                    pos_embed = model.pos_embed
                    if x.shape[1] != pos_embed.shape[1]:
                        # 简单线性插值适配时间窗口变化
                        import torch.nn.functional as F
                        pe_t = pos_embed.transpose(1, 2)
                        pe_new = F.interpolate(pe_t, size=x.shape[1], mode='linear', align_corners=False)
                        pos_embed = pe_new.transpose(1, 2)
                    
                    x = x + pos_embed.to(x.device)

                for blk in model.blocks:
                    x = blk(x)
                if hasattr(model, 'norm') and model.norm is not None:
                    x = model.norm(x)
                return x 
            except Exception as e:
                # print(f"Feature extraction fallback: {e}")
                pass

        try:
            outputs = self.backbone(inputs, output_hidden_states=True)
            if hasattr(outputs, 'last_hidden_state'): return outputs.last_hidden_state
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states: return outputs.hidden_states[-1]
            return outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        except TypeError:
            return self.backbone(inputs)

    def forward(self, video_frames, text_tokens, state_info, first_frame_summary):
        # 1. 维度处理 [B, V, C, T, H, W]
        B_dim = video_frames.shape[0]
        is_dual_view = False
        
        if video_frames.dim() == 6: 
            is_dual_view = True
            B, V, C, T, H, W = video_frames.shape
            # [B, V, C, T, H, W] -> [B*V, C, T, H, W]
            video_frames = video_frames.view(B * V, C, T, H, W)

        if video_frames.shape[1] != 3 and video_frames.shape[2] == 3:
            video_frames = video_frames.permute(0, 2, 1, 3, 4)
        
        # 2. 提取特征
        tokens = self.extract_features(video_frames) # [B*V, N, D]

        # === 🟢 [核心修复] View Embedding 注入逻辑 ===
        if is_dual_view:
            # [B*V, N, D] -> [B, V, N, D]
            tokens = tokens.view(B, V, tokens.shape[1], tokens.shape[2])
            
            # Add Embedding: [1, 2, 1, D] broadcast to [B, 2, N, D]
            tokens = tokens + self.view_embed
            
            # 🟢 [修复] 使用 flatten 安全地合并前两个维度
            # [B, V, N, D] -> [B, V*N, D]
            tokens = tokens.flatten(1, 2)

        # 3. 文本编码
        if self.text_encoder is not None:
            with torch.no_grad():
                text_outputs = self.text_encoder(input_ids=text_tokens)
                text_embeds = text_outputs.last_hidden_state
            text_cond = text_embeds.mean(dim=1) 
        else:
            text_embeds = torch.zeros(tokens.shape[0], 10, 1024, device=tokens.device)
            text_cond = torch.zeros(tokens.shape[0], 1024, device=tokens.device)

        # 4. FiLM 调节
        tokens = self.film_t5(tokens, text_cond)
        state_cond = state_info[:, -1, :] 
        tokens = self.film_state(tokens, state_cond)

        # 5. 首帧注意力
        if first_frame_summary.dim() == 5:
            ff_input = first_frame_summary.transpose(1, 2).repeat(1, 1, 2, 1, 1)
            with torch.no_grad():
                ff_tokens = self.extract_features(ff_input)
                if ff_tokens.dim() == 3: first_frame_summary = ff_tokens.mean(dim=1, keepdim=True)
                elif ff_tokens.dim() == 2: first_frame_summary = ff_tokens.unsqueeze(1)

        attn_output, _ = self.cross_attention_first_frame(query=tokens, key=first_frame_summary, value=first_frame_summary)
        tokens = self.norm1(tokens + attn_output)

        # 6. 任务路由
        task_slots, confidence, background_context = self.routing_layer(tokens, first_frame_summary, text_embeds)
        weighted_task = torch.sum(task_slots * confidence, dim=1) # [B, D]
        
        # =========================================================
        # 🟢 核心修改：非对称上下文 (64 Spatial + 6 Future)
        # =========================================================

        # --- A. 空间特征 (Ego Spatial) ---
        # 注入 Task 信息并归一化
        fused_seq = self.norm2(tokens + weighted_task.unsqueeze(1))
        
        # Adaptive Pooling: [B, N, D] -> [B, 64, D]
        # 保持 64 个 Token 以保留空间细节
        fused_pooled = torch.nn.functional.adaptive_avg_pool1d(fused_seq.transpose(1, 2), 64).transpose(1, 2)
        
        # [B, 64, 768]
        current_spatial_tokens = self.projection_head(fused_pooled)
        
        # --- B. 未来预测 (ForeSight Temporal) ---
        # 使用全量 Token 作为 Memory，查询未来
        memory = tokens 
        B = tokens.shape[0]
        queries = self.future_queries.unsqueeze(0).expand(B, -1, -1)
        
        # Predictor: [B, 6, D]
        predicted_latents_raw = self.predictor(tgt=queries, memory=memory)
        
        # 1. For Loss: [B, 6, 1152]
        wm_latents_for_loss = self.wm_align_head(predicted_latents_raw)
        
        # 2. For RDT: [B, 6, 768]
        future_tokens_rdt = self.future_proj_head(predicted_latents_raw)
        
        # --- C. 拼接 ---
        # Total Length: 64 + 6 = 70
        rdt_input_sequence = torch.cat([current_spatial_tokens, future_tokens_rdt], dim=1)
        
        # 辅助 Heads
        global_rep_for_heads = tokens.mean(dim=1)
        semantic_out = self.semantic_align_head(global_rep_for_heads)
        temporal_out = self.temporal_align_head(global_rep_for_heads)

        return {
            "e_t": rdt_input_sequence,      # [B, 70, 768] -> Feed to RDT (Stage C)
            "wm_latents": wm_latents_for_loss, # [B, 6, 1152] -> Feed to Loss
            "task_slots": task_slots,
            "task_confidence": confidence,
            "background_context": background_context,
            "semantic_head_output": semantic_out, 
            "temporal_head_output": temporal_out  
        }