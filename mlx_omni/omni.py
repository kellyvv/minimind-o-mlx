"""完整 MLX Omni 模型: Thinker + Talker.

包装 MiniMindModel (Thinker) + Talker, 提供:
  - 9 通道输入 (8 audio codes + 1 text) 或者 1 通道纯文本
  - bridge layer 提取 (从 Thinker 中间层取 hidden 给 Talker)
  - 双 KV cache (Thinker N 层 + Talker M 层)
  - 文本 + 音频联合 forward
"""
from __future__ import annotations
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .model import OmniConfig, MiniMindForCausalLM
from .talker import Talker


class _MMProjector(nn.Module):
    """对应 PyTorch MMAudioProjector / MMVisionProjector:
        mlp = Sequential(LayerNorm, Linear, GELU, Linear)

    在 MLX 端用 layers 列表 (converter 会做 mlp.<i>. → layers.<i>. 重映射)。
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.layers = [
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for L in self.layers:
            x = L(x)
        return x


class MiniMindOmniLM(nn.Module):
    """MLX 端到端 Omni 模型.

    Forward 输入有两种模式:
      A. 纯文本: input_ids (B, T) → 自动用 audio_pad 填 8 通道
      B. 9 通道: input_ids (B, 9, T) → [0:8]=audio_codes, [8]=text_ids
    """
    def __init__(self, config: OmniConfig):
        super().__init__()
        self.config = config
        # 主 LM (含 lm_head + tied embedding)；命名前缀 "thinker"
        self.thinker = MiniMindForCausalLM(config)
        # Talker；命名前缀 "talker"
        self.talker = Talker(config)
        # 多模态投影器（stage 3 才用，但权重要能加载）
        # audio_hidden 来自 SenseVoice 输出 = 512; image_hidden = 768
        self.audio_proj = _MMProjector(512, config.hidden_size)
        self.vision_proj = _MMProjector(768, config.hidden_size)

    def __call__(
        self,
        input_ids: Optional[mx.array],
        cache: Optional[List] = None,
        spk_emb: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> Tuple[mx.array, List[mx.array], List]:
        """
        Returns:
            text_logits: (B, T, vocab_size)
            audio_logits_list: list[8] of (B, T, audio_vocab_size)
            new_cache: combined list, [thinker_cache] + [talker_cache]

        三种调用模式 (优先级从上往下):
          1) inputs_embeds 给定 + input_ids 是 (B, T) text_ids:
             用 inputs_embeds 喂 Thinker (跳过 embed_tokens), audio_ids 用 audio_pad 填 8 通道.
             适用于多模态注入 (音频 / 图像 通过 inputs_embeds 替换掉相应位置).
          2) input_ids.ndim == 3 (B, 9, T):
             [:, 0:8] = audio_ids, [:, 8] = text_ids. inputs_embeds 忽略.
             适用于 decode 期间的 1-token forward (talker 需要历史 audio_ids).
          3) input_ids.ndim == 2 + inputs_embeds=None:
             纯文本输入, audio_ids 用 audio_pad 填.
        """
        # 优先模式 1
        if inputs_embeds is not None:
            B, T = inputs_embeds.shape[:2]
            if input_ids is not None and input_ids.ndim == 2:
                text_ids = input_ids
            else:
                text_ids = mx.zeros((B, T), dtype=mx.int32)
            audio_ids = mx.full((B, 8, T), self.config.audio_pad_token, dtype=mx.int32)
        elif input_ids.ndim == 3:
            # 模式 2: 9 通道
            B, _, T = input_ids.shape
            text_ids = input_ids[:, 8, :]
            audio_ids = input_ids[:, :8, :]
        else:
            # 模式 3: 纯文本
            B, T = input_ids.shape
            text_ids = input_ids
            audio_ids = mx.full((B, 8, T), self.config.audio_pad_token, dtype=mx.int32)

        # 拆缓存
        n_thinker = len(self.thinker.layers)
        if cache is None:
            thinker_cache = None
            talker_cache = None
        else:
            thinker_cache = cache[:n_thinker]
            talker_cache = cache[n_thinker:]

        # Thinker forward (返回 bridge state)
        text_logits, thinker_cache_new, bridge_states = self.thinker(
            text_ids,
            cache=thinker_cache,
            bridge_layer=self.config.bridge_layer,
            inputs_embeds=inputs_embeds,
        )

        # Talker forward
        audio_logits_list, talker_cache_new = self.talker(
            bridge_states, audio_ids, spk_emb=spk_emb, cache=talker_cache,
        )

        return text_logits, audio_logits_list, thinker_cache_new + talker_cache_new

    @property
    def layers(self):
        return self.thinker.layers
