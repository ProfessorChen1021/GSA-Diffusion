"""Attention decoupling and injection (ADI) processors used in generation."""

from typing import List, Dict, Any, Optional
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F  # used for per-layer mask resizing
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
            "mask_tensor": torch.Tensor [1, 1, H, W] (0.0~1.0), # external mask from grounding stage
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
                # 1. Local text encoding
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

                # 2. Entity text encoding kept as fallback
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

                # 3. Normalize mask tensor
                if "mask_tensor" in ent and ent["mask_tensor"] is not None:
                    # Ensure mask uses correct device and dtype
                    ent["mask_tensor"] = ent["mask_tensor"].to(device=self.device, dtype=self.dtype)

        self.injection_enabled = True
        self.injection_strength = 1.0
        self.current_step_idx = 0
        self.capture_steps = set()
        self.attn_records = []


# ... ADIContext above ...


class ADICrossAttnProcessor(AttnProcessor):
    def __init__(
        self,
        ctx: ADIContext,
        layer_name: str = "",
        alpha_g: float = 0.2,  # global/background branch weight
        alpha_e: float = 1.0,  # entity injection weight
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
        self._reported_nonfinite = set()

    def _report_nonfinite(self, tag: str) -> None:
        key = (self.layer_name, tag)
        if key in self._reported_nonfinite:
            return
        layer = self.layer_name or "layer"
        print(f"[ADI] Non-finite values sanitized at {layer} ({tag}).")
        self._reported_nonfinite.add(key)

    def _sanitize_tensor(
        self,
        tensor: torch.Tensor,
        tag: str,
        *,
        nan: float = 0.0,
        posinf: float = 1e4,
        neginf: float = -1e4,
    ) -> torch.Tensor:
        if torch.isfinite(tensor).all():
            return tensor
        self._report_nonfinite(tag)
        return torch.nan_to_num(tensor, nan=nan, posinf=posinf, neginf=neginf)

    def _stable_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        tag: str,
    ):
        q = self._sanitize_tensor(q.float(), f"{tag}:q")
        k = self._sanitize_tensor(k.float(), f"{tag}:k")
        v_f = self._sanitize_tensor(v.float(), f"{tag}:v")

        # scores are computed inside _stable_attention in float32 for numerical stability
        if bias is not None:
            scores = scores + bias.float()
        if mask is not None:
            scores = scores.masked_fill(mask, -1e4)

        scores = self._sanitize_tensor(scores, f"{tag}:scores", nan=-1e4, posinf=1e4, neginf=-1e4)
        scores = scores - scores.amax(dim=-1, keepdim=True)
        probs = torch.softmax(scores, dim=-1)
        probs = self._sanitize_tensor(probs, f"{tag}:probs", nan=0.0, posinf=1.0, neginf=0.0)
        hidden_states = torch.bmm(probs, v_f).to(v.dtype)
        hidden_states = self._sanitize_tensor(hidden_states, f"{tag}:out")
        return hidden_states, probs

    def _vanilla_attn(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        # Standard attention used for unconditional branch.
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        hidden_states = self._sanitize_tensor(hidden_states, "vanilla:hidden")
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        encoder_hidden_states = self._sanitize_tensor(encoder_hidden_states, "vanilla:encoder")
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        hidden_states, _attention_probs = self._stable_attention(
            query,
            key,
            value,
            bias=attention_mask,
            tag="vanilla",
        )
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = self._sanitize_tensor(hidden_states, "vanilla:post_heads")

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        hidden_states = self._sanitize_tensor(hidden_states, "vanilla:post_out")

        return hidden_states

    def _masked_self_attn(self, attn, hidden_states):
        """
        Core mechanism: self-attention isolation
        Prevents texture/style leakage across different entities in self-attention.
        """
        B, N, C = hidden_states.shape
        device = hidden_states.device

        # 1. Compute Q, K, V
        hidden_states = self._sanitize_tensor(hidden_states, "self:hidden")
        q = attn.head_to_batch_dim(attn.to_q(hidden_states))
        k = attn.head_to_batch_dim(attn.to_k(hidden_states))
        v = attn.head_to_batch_dim(attn.to_v(hidden_states))

        # 2. Prepare standard attention scores [B*H, N, N]
        # scores are computed inside _stable_attention in float32 for numerical stability

        # 3. Build isolation mask (Isolation Bias)
        # Build a [1, N, N] bias: disallow attention across different foreground entities.
        side = int(math.sqrt(N))

        # Aggregate per-entity mask indices
        # map_idx: [N], 0=background, 1=entity1, 2=entity2...
        map_idx = torch.zeros(N, device=device, dtype=torch.long)

        for i, ent in enumerate(self.ctx.entities):
            if "mask_tensor" in ent and ent["mask_tensor"] is not None:
                # Resize mask to current resolution
                raw = ent["mask_tensor"].to(device)
                resized = F.interpolate(raw, size=(side, side), mode="nearest")
                flat = resized.view(-1)  # [N]

                # Mark pixels that belong to this entity (i+1)
                # For overlaps, later entities overwrite earlier ones (acceptable here).
                map_idx[flat > 0.5] = (i + 1)

        # Build N x N pairwise identity matrix
        # (N, 1) == (1, N) -> (N, N)
        # Isolate when pixel i and j have different non-background IDs.
        # Policy: same ID can attend; background (0) can interact with all.

        # [N, 1]
        id_col = map_idx.unsqueeze(1)
        # [1, N]
        id_row = map_idx.unsqueeze(0)

        # Allowed connection conditions:
        # 1. same ID
        # 2. one side is background (ID=0)
        allowed = (id_col == id_row) | (id_col == 0) | (id_row == 0)

        # Convert to additive attention bias
        mask_bias = torch.zeros((N, N), device=device, dtype=torch.float32)  # [N, N]
        mask_bias[~allowed] = -1e4  # block interactions across different entities

        # 4. Inject bias into attention computation
        hidden_states, _probs = self._stable_attention(
            q,
            k,
            v,
            bias=mask_bias.unsqueeze(0),
            tag="self_iso",
        )
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = self._sanitize_tensor(hidden_states, "self:post_heads")

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        hidden_states = self._sanitize_tensor(hidden_states, "self:post_out")
        return hidden_states

    def _adi_cross_attn(self, attn, hidden_states, encoder_hidden_states):
        """
        Cross-attention: region-aware entity injection
        """
        B, N, C = hidden_states.shape
        device = hidden_states.device

        # Q
        hidden_states = self._sanitize_tensor(hidden_states, "cross:hidden")
        encoder_hidden_states = self._sanitize_tensor(encoder_hidden_states, "cross:encoder")
        q = attn.head_to_batch_dim(attn.to_q(hidden_states))
        B_H, _, D = q.shape

        # 1. Global/background branch
        y_global_hd = torch.zeros_like(q)
        if self.ctx.enable_global:
            k_all = attn.head_to_batch_dim(attn.to_k(encoder_hidden_states))
            v_all = attn.head_to_batch_dim(attn.to_v(encoder_hidden_states))
            y_global_hd, _probs_global = self._stable_attention(
                q,
                k_all,
                v_all,
                tag="cross_global",
            )
        y_global_hd = self.alpha_g * y_global_hd

        # 2. Entity injection branch
        y_out_hd = torch.zeros_like(q)
        total_weight = torch.zeros(B_H, N, 1, device=device, dtype=hidden_states.dtype)
        side = int(math.sqrt(N))

        for ent in self.ctx.entities:
            # Local K, V
            loc_embed = ent["local_embedding"].to(device).expand(B, -1, -1)
            k_loc = attn.head_to_batch_dim(attn.to_k(loc_embed))
            v_loc = attn.head_to_batch_dim(attn.to_v(loc_embed))

            # Token Mask
            tok_mask = (1 - ent["local_attn_mask"].to(device)).bool()
            tok_mask = tok_mask.view(B, 1, 1, -1).expand(B, attn.heads, N, -1).reshape(B_H, N, -1)

            # Spatial Mask
            if "mask_tensor" in ent and ent["mask_tensor"] is not None:
                raw_mask = ent["mask_tensor"].to(device)
                resized = F.interpolate(raw_mask, size=(side, side), mode="bilinear", align_corners=False)
                resized = torch.clamp(resized, 0.0, 1.0)
                if self.sharpen_t != 1.0:
                    resized = torch.pow(resized, self.sharpen_t)
                spatial_mask = resized.view(1, N, 1).expand(B_H, -1, 1)
                entity_weight = spatial_mask * self.alpha_e

                v_ent, attn_probs = self._stable_attention(
                    q,
                    k_loc,
                    v_loc,
                    mask=tok_mask,
                    tag=f"cross_entity:{ent.get('phrase_entity', 'entity')}",
                )
                y_out_hd += v_ent * entity_weight
                total_weight += entity_weight

                # ---- Capture attention map ----
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
                                flat_mask = resized.view(-1)  # apply spatial mask to visualization map
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

        # 3. Fuse global and entity branches
        bg_weight = (1.0 - total_weight).clamp(0.0, 1.0)
        final_hd = y_out_hd + y_global_hd * bg_weight

        hidden_states = attn.batch_to_head_dim(final_hd)
        hidden_states = self._sanitize_tensor(hidden_states, "cross:post_heads")
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        hidden_states = self._sanitize_tensor(hidden_states, "cross:post_out")
        return hidden_states

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        # [debug] print once every 1000 calls to avoid log spam
        self._debug_call_idx += 1
        if self._debug_call_idx % 1000 == 0:
            print(f"[ADI Running] Processing Layer, Self-Attn={encoder_hidden_states is None}")
        strength = float(getattr(self.ctx, "injection_strength", 1.0))
        if (hasattr(self.ctx, "injection_enabled") and not self.ctx.injection_enabled) or strength <= 1e-4:
            return self._sanitize_tensor(
                self._vanilla_attn(attn, hidden_states, encoder_hidden_states, attention_mask),
                "call:disabled",
            )

        # Branch by self-attention vs cross-attention
        is_self_attn = (encoder_hidden_states is None) or (encoder_hidden_states.shape[1] == hidden_states.shape[1])

        if is_self_attn:
            if not self.enable_self_iso:
                return self._sanitize_tensor(
                    self._vanilla_attn(attn, hidden_states, encoder_hidden_states, attention_mask),
                    "call:self_disabled",
                )
            # Self-attention: apply isolation only on conditional branch
            B_full = hidden_states.shape[0]
            if B_full % 2 == 0:
                B = B_full // 2
                h_u, h_c = hidden_states.chunk(2)
                # Uncond: vanilla
                out_u = self._vanilla_attn(attn, h_u)
                # Cond branch: blend masked self-attention with vanilla output
                out_c_iso = self._masked_self_attn(attn, h_c)
                out_c_vanilla = self._vanilla_attn(attn, h_c)
                out_c = out_c_iso * strength + out_c_vanilla * (1.0 - strength)
                return self._sanitize_tensor(torch.cat([out_u, out_c], dim=0), "call:self_cfg")
            else:
                iso = self._masked_self_attn(attn, hidden_states)
                vanilla = self._vanilla_attn(attn, hidden_states)
                return self._sanitize_tensor(iso * strength + vanilla * (1.0 - strength), "call:self")

        else:
            # Cross-attention: apply region-aware injection
            B_full = hidden_states.shape[0]
            if B_full % 2 == 0:
                B = B_full // 2
                h_u, h_c = hidden_states.chunk(2)
                e_u, e_c = encoder_hidden_states.chunk(2)

                out_u = self._vanilla_attn(attn, h_u, e_u, attention_mask)
                out_c_adi = self._adi_cross_attn(attn, h_c, e_c)
                out_c_vanilla = self._vanilla_attn(attn, h_c, e_c, attention_mask)
                out_c = out_c_adi * strength + out_c_vanilla * (1.0 - strength)
                return self._sanitize_tensor(torch.cat([out_u, out_c], dim=0), "call:cross_cfg")
            else:
                adi = self._adi_cross_attn(attn, hidden_states, encoder_hidden_states)
                vanilla = self._vanilla_attn(attn, hidden_states, encoder_hidden_states, attention_mask)
                return self._sanitize_tensor(adi * strength + vanilla * (1.0 - strength), "call:cross")



