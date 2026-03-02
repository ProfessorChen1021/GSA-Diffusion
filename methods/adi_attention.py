# adi_attention.py
# -*- coding: utf-8 -*-
"""
A-DI (Attention Decoupling & Injection) for SD15+ControlNet

- 支持传入 Binary Mask (from Grounded-SAM)
- 在推理时，根据当前层分辨率动态 Resize Mask
- 强制 Attention 空间隔离：V_out = V_out * Mask
"""

from typing import List, Dict, Any, Optional
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    from diffusers.models.attention_processor import AttnProcessor
except Exception:
    from diffusers.models.attention_processor import AttentionProcessor as AttnProcessor


class ADIContext:
    """
    entities: List[Dict]
        {
            "phrase_entity": "cat",
            "phrase_ent_attr": "a fluffy red cat",
            "mask_tensor": torch.Tensor [1, 1, H, W] (0.0~1.0), # 接收外部 Mask
        }
    """

    def __init__(
        self,
        entities: List[Dict[str, Any]],
        tokenizer,
        text_encoder,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        enable_global: bool = True,
    ):
        self.entities = entities
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device
        self.dtype = dtype if dtype is not None else torch.float16
        self.enable_global = enable_global

        max_length = getattr(tokenizer, "model_max_length", 77)

        with torch.inference_mode():
            for ent in self.entities:
                # 1. 文本编码
                text_local = ent.get("phrase_ent_attr", ent.get("phrase_entity"))
                tok_local = tokenizer(
                    text_local,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                )
                tok_local = {k: v.to(self.device) for k, v in tok_local.items()}
                hid_local = text_encoder(**tok_local, output_hidden_states=True).last_hidden_state.to(self.dtype)
                ent["local_embedding"] = hid_local
                ent["local_attn_mask"] = tok_local["attention_mask"]
                ent["token_ids"] = tok_local["input_ids"][0].detach().cpu()
                ent["token_mask"] = tok_local["attention_mask"][0].detach().cpu().bool()

                # 2. 实体竞争编码 
                text_ent = ent.get("phrase_entity", text_local)
                tok_ent = tokenizer(
                    text_ent,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                )
                tok_ent = {k: v.to(self.device) for k, v in tok_ent.items()}
                hid_ent = text_encoder(**tok_ent, output_hidden_states=True).last_hidden_state.to(self.dtype)
                ent["entity_embedding"] = hid_ent
                ent["entity_attn_mask"] = tok_ent["attention_mask"]

                # 3. 处理 Mask
                if "mask_tensor" in ent and ent["mask_tensor"] is not None:
                    # 确保 Mask 在正确的设备和类型上
                    ent["mask_tensor"] = ent["mask_tensor"].to(device=self.device, dtype=self.dtype)

        self.injection_enabled = True
        self.injection_strength = 1.0
        self.current_step_idx = 0
        self.capture_steps = set()
        self.attn_records = []



class ADICrossAttnProcessor(AttnProcessor):
    def __init__(
        self,
        ctx: ADIContext,
        layer_name: str = "",
        alpha_g: float = 0.2,  # 全局背景权重
        alpha_e: float = 1.0,  # 实体注入权重
        sharpen_t: float = 1.0,
        enable_self_iso: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.ctx = ctx
        self.layer_name = layer_name
        self.alpha_g = alpha_g
        self.alpha_e = alpha_e
        self.sharpen_t = sharpen_t
        self.enable_self_iso = enable_self_iso
        self._debug_call_idx = 0

    def _vanilla_attn(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        # 标准 Attention，用于 Unconditional 分支
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

    def _masked_self_attn(self, attn, hidden_states):
        """
        自注意力隔离 (Self-Attention Isolation)
    
        """
        B, N, C = hidden_states.shape
        device = hidden_states.device

        # 1. 计算 Q, K, V
        q = attn.head_to_batch_dim(attn.to_q(hidden_states))
        k = attn.head_to_batch_dim(attn.to_k(hidden_states))
        v = attn.head_to_batch_dim(attn.to_v(hidden_states))

        # 2. 计算标准 Attention Scores [B*H, N, N]
        scores = torch.bmm(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(q.shape[-1]))

        # 3. 构建隔离掩码 (Isolation Mask)
        side = int(math.sqrt(N))

        # 聚合所有实体的 Mask 索引
        # map_idx: [N], 0=背景, 1=实体1, 2=实体2...
        map_idx = torch.zeros(N, device=device, dtype=torch.long)

        for i, ent in enumerate(self.ctx.entities):
            if "mask_tensor" in ent and ent["mask_tensor"] is not None:
                # Resize mask to current resolution
                raw = ent["mask_tensor"].to(device)
                resized = F.interpolate(raw, size=(side, side), mode="nearest")
                flat = resized.view(-1)  # [N]

                # 标记该实体的像素 (i+1)
                map_idx[flat > 0.5] = (i + 1)

        # 生成 N x N 矩阵
        # (N, 1) == (1, N) -> (N, N)
        # 如果像素 i 和像素 j 的 ID 不同，且都不是背景(0)，则隔离
        # 策略：同类相吸。ID 相同允许 Attention；背景(0) 可与所有人交互

        # [N, 1]
        id_col = map_idx.unsqueeze(1)
        # [1, N]
        id_row = map_idx.unsqueeze(0)

        # 允许连接的条件：
        # 1. ID 相同
        # 2. 其中一方是背景 (ID=0)
        allowed = (id_col == id_row) | (id_col == 0) | (id_row == 0)

        # 转换为 Bias
        mask_bias = torch.zeros_like(scores[0, ...])  # [N, N]
        mask_bias[~allowed] = -1e4  # 阻断不同实体间的交流

        # 4. 注入 Bias
        scores = scores + mask_bias.unsqueeze(0)

        # 5. Softmax & Output
        probs = torch.softmax(scores, dim=-1)
        hidden_states = torch.bmm(probs, v)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        return attn.to_out[0](attn.to_out[1](hidden_states))

    def _adi_cross_attn(self, attn, hidden_states, encoder_hidden_states):
        """
        Cross-Attention: 区域注入
        """
        B, N, C = hidden_states.shape
        device = hidden_states.device

        # Q
        q = attn.head_to_batch_dim(attn.to_q(hidden_states))
        B_H, _, D = q.shape
        scale = 1.0 / math.sqrt(D)

        # 1. 全局背景
        y_global_hd = torch.zeros_like(q)
        if self.ctx.enable_global:
            k_all = attn.head_to_batch_dim(attn.to_k(encoder_hidden_states))
            v_all = attn.head_to_batch_dim(attn.to_v(encoder_hidden_states))
            scores_g = torch.bmm(q, k_all.transpose(-1, -2)) * scale
            y_global_hd = torch.bmm(torch.softmax(scores_g, dim=-1), v_all)
        y_global_hd = self.alpha_g * y_global_hd

        # 2. 实体注入
        y_out_hd = torch.zeros_like(q)
        total_weight = torch.zeros(B_H, N, 1, device=device, dtype=hidden_states.dtype)
        side = int(math.sqrt(N))

        for ent in self.ctx.entities:
            # Local K, V
            loc_embed = ent["local_embedding"].to(device).expand(B, -1, -1)
            k_loc = attn.head_to_batch_dim(attn.to_k(loc_embed))
            v_loc = attn.head_to_batch_dim(attn.to_v(loc_embed))

            # Scores
            scores = torch.bmm(q, k_loc.transpose(-1, -2)) * scale
            # Token Mask
            tok_mask = (1 - ent["local_attn_mask"].to(device)).bool()
            tok_mask = tok_mask.view(B, 1, 1, -1).expand(B, attn.heads, N, -1).reshape(B_H, N, -1)
            scores = scores.masked_fill(tok_mask, -1e4)

            # Spatial Mask
            if "mask_tensor" in ent and ent["mask_tensor"] is not None:
                raw_mask = ent["mask_tensor"].to(device)
                resized = F.interpolate(raw_mask, size=(side, side), mode="bilinear", align_corners=False)
                resized = torch.clamp(resized, 0.0, 1.0)
                if self.sharpen_t != 1.0:
                    resized = torch.pow(resized, self.sharpen_t)
                spatial_mask = resized.view(1, N, 1).expand(B_H, -1, 1)
                entity_weight = spatial_mask * self.alpha_e

                attn_probs = torch.softmax(scores, dim=-1)
                v_ent = torch.bmm(attn_probs, v_loc)
                y_out_hd += v_ent * entity_weight
                total_weight += entity_weight

                # ---- 捕获 Attention Map ----
                should_capture = (
                    hasattr(self.ctx, "capture_steps")
                    and self.ctx.capture_steps
                    and getattr(self.ctx, "current_step_idx", -1) in self.ctx.capture_steps
                    and N in {256, 1024}
                    and float(getattr(self.ctx, "injection_strength", 1.0)) > 1e-4
                )
                if should_capture:
                    heads = attn.heads
                    if heads > 0 and attn_probs.shape[0] % heads == 0:
                        B_batch = attn_probs.shape[0] // heads
                        attn_probs = attn_probs.view(B_batch, heads, N, attn_probs.shape[-1])
                        attn_mean = attn_probs.mean(dim=1)  # [B, N, Tokens]
                        attn_mean = attn_mean.mean(dim=0)   # [N, Tokens]

                        token_mask = ent.get("token_mask", None)
                        if token_mask is not None and token_mask.shape[-1] == attn_mean.shape[-1]:
                            if token_mask.sum() > 0:
                                ent_map = attn_mean[:, token_mask].mean(dim=-1)  # [N]
                                flat_mask = resized.view(-1)  # 将空间掩码叠加到可视化 map
                                if flat_mask.sum() > 0:
                                    ent_map = ent_map * flat_mask
                                self.ctx.attn_records.append(
                                    {
                                        "step": int(getattr(self.ctx, "current_step_idx", -1)),
                                        "layer": self.layer_name if hasattr(self, "layer_name") else "",
                                        "entity": ent.get("phrase_entity", "entity"),
                                        "side": side,
                                        "map": ent_map.detach().cpu(),
                                    }
                                )

        # 3. 融合
        bg_weight = (1.0 - total_weight).clamp(0.0, 1.0)
        final_hd = y_out_hd + y_global_hd * bg_weight

        hidden_states = attn.batch_to_head_dim(final_hd)
        return attn.to_out[0](attn.to_out[1](hidden_states))

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        # [debug] 每 1000 次打印一次，避免刷屏
        self._debug_call_idx += 1
        if self._debug_call_idx % 1000 == 0:
            print(f"[ADI Running] Processing Layer, Self-Attn={encoder_hidden_states is None}")
        strength = float(getattr(self.ctx, "injection_strength", 1.0))
        if (hasattr(self.ctx, "injection_enabled") and not self.ctx.injection_enabled) or strength <= 1e-4:
            return self._vanilla_attn(attn, hidden_states, encoder_hidden_states, attention_mask)

        # 判断 Self-Attention / Cross-Attention
        is_self_attn = (encoder_hidden_states is None) or (encoder_hidden_states.shape[1] == hidden_states.shape[1])

        if is_self_attn:
            if not self.enable_self_iso:
                return self._vanilla_attn(attn, hidden_states, encoder_hidden_states, attention_mask)
            # Self-Attention -> 只对 cond 分支做隔离
            B_full = hidden_states.shape[0]
            if B_full % 2 == 0:
                B = B_full // 2
                h_u, h_c = hidden_states.chunk(2)
                # Uncond: vanilla
                out_u = self._vanilla_attn(attn, h_u)
                # Cond: masked self attention + vanilla 混合
                out_c_iso = self._masked_self_attn(attn, h_c)
                out_c_vanilla = self._vanilla_attn(attn, h_c)
                out_c = out_c_iso * strength + out_c_vanilla * (1.0 - strength)
                return torch.cat([out_u, out_c], dim=0)
            else:
                iso = self._masked_self_attn(attn, hidden_states)
                vanilla = self._vanilla_attn(attn, hidden_states)
                return iso * strength + vanilla * (1.0 - strength)

        else:
            # Cross-Attention -> 区域注入
            B_full = hidden_states.shape[0]
            if B_full % 2 == 0:
                B = B_full // 2
                h_u, h_c = hidden_states.chunk(2)
                e_u, e_c = encoder_hidden_states.chunk(2)

                out_u = self._vanilla_attn(attn, h_u, e_u, attention_mask)
                out_c_adi = self._adi_cross_attn(attn, h_c, e_c)
                out_c_vanilla = self._vanilla_attn(attn, h_c, e_c, attention_mask)
                out_c = out_c_adi * strength + out_c_vanilla * (1.0 - strength)
                return torch.cat([out_u, out_c], dim=0)
            else:
                adi = self._adi_cross_attn(attn, hidden_states, encoder_hidden_states)
                vanilla = self._vanilla_attn(attn, hidden_states, encoder_hidden_states, attention_mask)
                return adi * strength + vanilla * (1.0 - strength)
