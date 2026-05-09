"""MLX 版 Talker 模块 (MTP for Mimi 8-codebook audio).

完全对齐 PyTorch model_omni.py 的 Talker:
  - TalkerEmbedding: base + 8 adapters (Embedding→GELU→Linear)，结果 sum/8
  - TalkerHead:      base + 8 adapters (Linear→GELU→Linear)，每层一个独立 logit
  - codec_proj / embed_proj: 2 层 MLP + RMSNorm
  - text_scale / audio_scale: 标量参数
  - spk_proj: speaker embedding 投影
  - 4 层 MiniMindBlock + RMSNorm
  - RoPE 共享 max_position_embeddings, 自己一份 cos/sin
"""
from __future__ import annotations
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .model import (
    OmniConfig,
    MiniMindBlock,
    precompute_freqs_cis,
)


# ============================================================================
#                          TalkerEmbedding (input)
# ============================================================================

class TalkerEmbedding(nn.Module):
    """对应 PyTorch:
        self.base = nn.Embedding(audio_vocab_size, talker_hidden_size)
        self.adapters[i] = Sequential(
            Embedding(audio_vocab_size, rank=256),
            GELU,
            Linear(rank, talker_hidden_size, bias=False)
        )
        forward(x: (B, 8, T)):
            base_out = self.base(x)  # (B, 8, T, H)
            return sum(base_out[:, i] + adapters[i](x[:, i])  for i in 8) / 8
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, num_layers: int = 8, rank: int = 256):
        super().__init__()
        self.num_layers = num_layers
        self.base = nn.Embedding(num_embeddings, embedding_dim)
        # Sequential: Embedding -> GELU -> Linear (no bias)
        # 用 dict 存以便后面 load_weights 时按名映射
        self.adapters = [
            _AdapterEmb(num_embeddings, rank, embedding_dim) for _ in range(num_layers)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, 8, T) int
        base_out = self.base(x)  # (B, 8, T, H)
        result = mx.zeros((x.shape[0], x.shape[2], base_out.shape[-1]), dtype=base_out.dtype)
        for i in range(self.num_layers):
            result = result + base_out[:, i] + self.adapters[i](x[:, i])
        return result / self.num_layers


class _AdapterEmb(nn.Module):
    """单个 embedding adapter，对应 Sequential(Embedding, GELU, Linear)."""
    def __init__(self, vocab_size: int, rank: int, hidden_dim: int):
        super().__init__()
        # 注意：和 PyTorch nn.Sequential 命名对齐 — 子模块名是 0/1/2
        # Sequential[0] = Embedding(vocab, rank)
        # Sequential[1] = GELU (无参数)
        # Sequential[2] = Linear(rank, hidden, bias=False)
        # 在 MLX 中无法直接用整数命名，我们用 layers 列表 + 自定义权重映射
        self.layers = [
            nn.Embedding(vocab_size, rank),
            nn.GELU(),
            nn.Linear(rank, hidden_dim, bias=False),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T) int
        h = self.layers[0](x)
        h = self.layers[1](h)
        h = self.layers[2](h)
        return h


# ============================================================================
#                            TalkerHead (output)
# ============================================================================

class TalkerHead(nn.Module):
    """对应 PyTorch:
        self.base = nn.Linear(in, out, bias=False)
        self.adapters[i] = Sequential(Linear(in, rank, bias=False), GELU, Linear(rank, out, bias=False))
        forward(x): return [base(x) + adapters[i](x) for i in 8]
    """
    def __init__(self, in_features: int, out_features: int, num_layers: int = 8, rank: int = 256):
        super().__init__()
        self.num_layers = num_layers
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.adapters = [
            _AdapterHead(in_features, rank, out_features) for _ in range(num_layers)
        ]

    def __call__(self, x: mx.array) -> List[mx.array]:
        base_out = self.base(x)
        return [base_out + adapter(x) for adapter in self.adapters]


class _AdapterHead(nn.Module):
    def __init__(self, in_features: int, rank: int, out_features: int):
        super().__init__()
        self.layers = [
            nn.Linear(in_features, rank, bias=False),
            nn.GELU(),
            nn.Linear(rank, out_features, bias=False),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        return self.layers[2](self.layers[1](self.layers[0](x)))


# ============================================================================
#                    Projection MLPs (codec / embed)
# ============================================================================

class _ProjMLP(nn.Module):
    """对应 PyTorch:
        Sequential(Linear, GELU, Linear, RMSNorm)
    用于 codec_proj 和 embed_proj.
    """
    def __init__(self, in_features: int, mid_features: int, out_features: int, eps: float = 1e-6):
        super().__init__()
        self.layers = [
            nn.Linear(in_features, mid_features),
            nn.GELU(),
            nn.Linear(mid_features, out_features),
            nn.RMSNorm(out_features, eps=eps),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for L in self.layers:
            x = L(x)
        return x


# ============================================================================
#                                  Talker
# ============================================================================

class Talker(nn.Module):
    """完整的 Talker 模块：4 层 transformer + MTP head.

    Forward 输入:
        bridge_states: (B, T, thinker_hidden)  来自 Thinker mid layer
        audio_ids: (B, 8, T) int                上一步生成的 audio codes
        spk_emb: (B, 1, spk_emb_size) or None  说话人 embedding
        cache: 4 layers KV cache list
    输出:
        audio_logits: list[8] of (B, T, audio_vocab_size)
        new_cache
    """
    def __init__(self, config: OmniConfig):
        super().__init__()
        self.config = config
        # 注意：talker 内部用一个独立 config，hidden_size 来自 talker_hidden_size
        # 但 num_attention_heads / num_key_value_heads / head_dim 沿用主 config
        # （PyTorch 端: self.talker_config = MiniMindConfig(hidden_size=talker_hidden_size, use_moe=...)）
        # MiniMindConfig 默认 hidden_size=768 时，num_attention_heads=8, head_dim=96
        # 我们这里只要保证 talker block 内部用 talker_hidden_size 做 q/k/v_proj 即可
        talker_cfg = OmniConfig(
            hidden_size=config.talker_hidden_size,
            num_hidden_layers=config.num_talker_hidden_layers,
            vocab_size=config.audio_vocab_size,  # 不用在 layers 里，仅占位
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
        )

        self.layers = [MiniMindBlock(talker_cfg) for _ in range(config.num_talker_hidden_layers)]
        self.norm = nn.RMSNorm(config.talker_hidden_size, eps=config.rms_norm_eps)
        self.lm_head = TalkerHead(config.talker_hidden_size, config.audio_vocab_size)
        self.embed_tokens = TalkerEmbedding(config.audio_vocab_size, config.talker_hidden_size)
        self.codec_proj = _ProjMLP(
            config.talker_hidden_size, config.talker_hidden_size, config.talker_hidden_size,
            eps=config.rms_norm_eps
        )
        self.embed_proj = _ProjMLP(
            config.hidden_size, config.hidden_size, config.talker_hidden_size,
            eps=config.rms_norm_eps
        )
        self.spk_proj = nn.Linear(config.spk_emb_size, config.talker_hidden_size, bias=False)

        # text_scale / audio_scale: 标量 nn.Parameter (PyTorch 端 shape=())
        # MLX 没有 nn.Parameter; 用 mx.array 字段
        self.text_scale = mx.array(3.0)
        self.audio_scale = mx.array(1.0)

        # RoPE 自己一份
        cos, sin = precompute_freqs_cis(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self._freqs_cos = cos
        self._freqs_sin = sin

    def __call__(
        self,
        bridge_states: mx.array,
        audio_ids: mx.array,
        spk_emb: Optional[mx.array] = None,
        cache: Optional[List] = None,
    ) -> Tuple[List[mx.array], List]:
        B, _, T = audio_ids.shape

        # 1. embedding
        talker_emb = self.embed_tokens(audio_ids)  # (B, T, talker_h)

        # 2. spk 替换 (audio_ids[:, 0, :] == audio_spk_token 的位置)
        if spk_emb is not None:
            spk_mask = (audio_ids[:, 0, :] == self.config.audio_spk_token)[..., None]  # (B, T, 1)
            spk_proj_out = self.spk_proj(spk_emb)  # (B, 1, talker_h)
            # 广播：spk_proj_out (B, 1, H) → (B, T, H)
            spk_broadcast = mx.broadcast_to(spk_proj_out, (B, T, talker_emb.shape[-1]))
            talker_emb = mx.where(spk_mask, spk_broadcast, talker_emb)

        # 3. 文本 + 音频 加权合并
        h = (
            self.embed_proj(bridge_states) * self.text_scale +
            self.codec_proj(talker_emb) * self.audio_scale
        )

        # 4. KV cache 起点
        if cache is None:
            cache = [None] * len(self.layers)
            start_pos = 0
        else:
            start_pos = cache[0][0].shape[1] if cache[0] is not None else 0
        cos = self._freqs_cos[start_pos: start_pos + T]
        sin = self._freqs_sin[start_pos: start_pos + T]

        # 5. transformer layers
        new_cache = []
        for layer, c in zip(self.layers, cache):
            h, c2 = layer(h, cos, sin, c)
            new_cache.append(c2)
        h = self.norm(h)

        # 6. MTP head (8 个独立 logit)
        audio_logits = self.lm_head(h)
        return audio_logits, new_cache
