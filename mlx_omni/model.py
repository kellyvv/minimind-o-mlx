"""MLX 版 MiniMind-O Thinker (主模型).

逐层对应 PyTorch 版 model_minimind.py，权重命名保持一致以便直接转换。
默认配置对应 minimind-3o (hidden_size=768, num_hidden_layers=8).

关键点:
  - GQA: num_attention_heads=8, num_key_value_heads=4 (n_rep=2)
  - RoPE: HF LLaMA 风格 cat([cos,cos], -1) + rotate_half (split halves)
  - RMSNorm: 在 fp32 计算后 cast 回原 dtype
  - tied embedding: lm_head 与 embed_tokens 共享权重
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import mlx.core as mx
import mlx.nn as nn


# ============================================================================
#                                Config
# ============================================================================

@dataclass
class OmniConfig:
    """对齐 PyTorch 版 MiniMindConfig 默认值 (minimind-3o)."""
    hidden_size: int = 768
    num_hidden_layers: int = 8
    vocab_size: int = 6400
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 96  # hidden_size // num_attention_heads
    intermediate_size: int = 2432  # ceil(hidden_size * pi / 64) * 64
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    tie_word_embeddings: bool = True
    use_moe: bool = False  # 暂不支持，留接口
    # OmniConfig extras (Talker / Audio / Vision)，stage 1 用不上但放着
    num_talker_hidden_layers: int = 4
    talker_hidden_size: int = 768
    audio_vocab_size: int = 2112
    audio_pad_token: int = 2049
    audio_stop_token: int = 2050
    audio_spk_token: int = 2051
    spk_emb_size: int = 192
    bridge_layer: int = 3  # num_hidden_layers // 2 - 1
    image_token_len: int = 64
    bos_token_id: int = 1
    eos_token_id: int = 2

    @classmethod
    def from_pt(cls, pt_config) -> "OmniConfig":
        """从 PyTorch 版 OmniConfig / MiniMindConfig 实例创建."""
        kw = {}
        for f in cls.__dataclass_fields__:
            if hasattr(pt_config, f):
                kw[f] = getattr(pt_config, f)
        return cls(**kw)


# ============================================================================
#                          RoPE precompute & apply
# ============================================================================

def precompute_freqs_cis(
    dim: int, end: int, rope_base: float = 1e6
) -> Tuple[mx.array, mx.array]:
    """与 PyTorch 版完全一致的 cos/sin 表 (HF LLaMA 风格双拼).

    Shape: (end, dim) — 注意是 dim 而不是 dim/2，因为 cat([x,x], -1).
    """
    freqs = 1.0 / (rope_base ** (mx.arange(0, dim, 2)[: dim // 2].astype(mx.float32) / dim))
    t = mx.arange(end).astype(mx.float32)
    # outer product
    freqs = mx.outer(t, freqs)  # (end, dim/2)
    cos_half = mx.cos(freqs)
    sin_half = mx.sin(freqs)
    freqs_cos = mx.concatenate([cos_half, cos_half], axis=-1)  # (end, dim)
    freqs_sin = mx.concatenate([sin_half, sin_half], axis=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array
) -> Tuple[mx.array, mx.array]:
    """与 PyTorch 版完全一致：rotate_half 是 cat([-x[d/2:], x[:d/2]]).

    q, k shape: (b, seq, num_heads, head_dim)
    cos, sin shape: (seq, head_dim)  → broadcast 到 (1, seq, 1, head_dim)
    """
    def rotate_half(x):
        d = x.shape[-1]
        return mx.concatenate([-x[..., d // 2:], x[..., : d // 2]], axis=-1)

    cos = cos[None, :, None, :]  # (1, seq, 1, hd)
    sin = sin[None, :, None, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.astype(q.dtype), k_embed.astype(k.dtype)


# ============================================================================
#                             Attention (GQA)
# ============================================================================

class Attention(nn.Module):
    def __init__(self, config: OmniConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)
        # MLX RMSNorm 实现就是和 PyTorch 一样在 fp32 下计算
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV cache: 在 seq 维 concat
        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=1)
            v = mx.concatenate([cache[1], v], axis=1)
        new_cache = (k, v)

        # 转成 (B, H, T, D) 给 sdpa
        q = q.transpose(0, 2, 1, 3)
        k_full = k.transpose(0, 2, 1, 3)
        v_full = v.transpose(0, 2, 1, 3)

        # GQA: 把 KV 头 repeat 到 Q 头数 (mlx 的 sdpa 也支持原生 GQA，这里手动 repeat 求稳)
        if self.n_rep > 1:
            k_full = mx.repeat(k_full, self.n_rep, axis=1)
            v_full = mx.repeat(v_full, self.n_rep, axis=1)

        # causal mask: 仅在 T>1 (prefill) 时需要
        mask = "causal" if T > 1 else None
        out = mx.fast.scaled_dot_product_attention(
            q, k_full, v_full, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.o_proj(out), new_cache


# ============================================================================
#                              FeedForward (SwiGLU)
# ============================================================================

class FeedForward(nn.Module):
    def __init__(self, config: OmniConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


# ============================================================================
#                           Transformer Block
# ============================================================================

class MiniMindBlock(nn.Module):
    def __init__(self, config: OmniConfig):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        h, new_cache = self.self_attn(self.input_layernorm(x), cos, sin, cache)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_cache


# ============================================================================
#                           Full Model (Thinker)
# ============================================================================

class MiniMindModel(nn.Module):
    def __init__(self, config: OmniConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [MiniMindBlock(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # RoPE 表预计算（不当成 parameter，常驻 GPU）
        cos, sin = precompute_freqs_cis(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        # 保存为属性而非 buffer (MLX 没有 buffer 概念)，转换权重时跳过
        self._freqs_cos = cos
        self._freqs_sin = sin

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[List[Optional[Tuple[mx.array, mx.array]]]] = None,
        bridge_layer: Optional[int] = None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> Tuple[mx.array, List[Tuple[mx.array, mx.array]], Optional[mx.array]]:
        """前向; 返回 (last_hidden, kv_cache, bridge_hidden_or_None).

        Args:
            input_ids: (B, T)
            inputs_embeds: 提前算好的 embedding（用于多模态注入），有则忽略 input_ids
            bridge_layer: 0-based index, 在该层 *之后* 的输出作为 bridge (Talker 输入)
        """
        B, T = (input_ids.shape if inputs_embeds is None else inputs_embeds.shape[:2])
        h = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        if cache is None:
            cache = [None] * len(self.layers)
            start_pos = 0
        else:
            start_pos = cache[0][0].shape[1] if cache[0] is not None else 0
        cos = self._freqs_cos[start_pos: start_pos + T]
        sin = self._freqs_sin[start_pos: start_pos + T]

        new_cache = []
        bridge = None
        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            h, c2 = layer(h, cos, sin, c)
            new_cache.append(c2)
            if bridge_layer is not None and i == bridge_layer:
                bridge = h
        h = self.norm(h)
        return h, new_cache, bridge


class MiniMindForCausalLM(nn.Module):
    """对齐 PyTorch 版 MiniMindForCausalLM, 含 lm_head 和 tied embedding."""
    def __init__(self, config: OmniConfig):
        super().__init__()
        self.config = config
        self.model = MiniMindModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # else: 直接用 self.model.embed_tokens.as_linear() 形式

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[List] = None,
        bridge_layer: Optional[int] = None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> Tuple[mx.array, List, Optional[mx.array]]:
        h, cache, bridge = self.model(input_ids, cache, bridge_layer=bridge_layer, inputs_embeds=inputs_embeds)
        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(h)
        else:
            logits = self.lm_head(h)
        return logits, cache, bridge

    @property
    def layers(self):  # 让外部 (e.g., quantize_predicate) 容易访问
        return self.model.layers
