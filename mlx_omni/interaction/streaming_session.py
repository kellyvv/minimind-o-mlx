"""持久 MLX streaming session (stage 3 hook).

目标: 把 stream_generate_omni 内部 prefill + decode + cache 状态拿出来,
让一个 WebSocket 通话期间 KV cache 跨多个 turn 复用, 接近 Thinking Machines
描述的 "inference server maintains a persistent GPU-memory sequence" 模式.

API (草案):

    sess = OmniStreamingSession(model, tokenizer)
    sess.append_text("你好")          # 注入 text token, 不立刻生成
    sess.append_audio_codes(codes)    # 注入 audio codes (用户语音 → Mimi 编码)
    sess.append_image_frame(emb)      # 注入图像 patch embedding
    for tok, frame in sess.decode_until(budget_ms=200): ...
        # 一次最多生成 200ms 的输出, 然后让出控制
    sess.reset_after_interrupt()       # 打断时丢掉 partial 输出 + 保留 cache

stage 1/2 不启用; stream_generate_omni 仍然每次 turn 从 prompt prefill 一次。
保留这个文件作为 stage 3 的接口设计参考。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional, Tuple

import mlx.core as mx


@dataclass
class OmniStreamingSession:
    """占位, 接口设计."""
    model: Any
    tokenizer: Any

    # MLX KV cache (thinker N 层 + talker M 层)
    cache: Optional[list] = None
    # 文本 / 音频 / 图像 token 历史 (用于重组 9 通道输入)
    text_ids: List[int] = field(default_factory=list)
    audio_codes: List[List[int]] = field(default_factory=lambda: [[] for _ in range(8)])

    def append_text(self, text_delta: str):
        """tokenize 并追加, 不立刻 forward (forward 在 decode_until 时触发)."""
        raise NotImplementedError("stage 3")

    def append_audio_codes(self, codes_8xT: List[List[int]]):
        """追加用户语音的 Mimi-encoded codes."""
        raise NotImplementedError("stage 3 - 需要先把 Mimi encode 接进来")

    def decode_until(self, budget_ms: int = 200) -> Iterator[Tuple[Optional[int], Optional[List[int]]]]:
        """按时间预算继续生成, yield (text_token, audio_frame). 时间到就让出."""
        raise NotImplementedError("stage 3")

    def reset_after_interrupt(self):
        """被打断: 清空 audio_buffer 等 partial 状态, KV cache 仍可继续."""
        raise NotImplementedError("stage 3")
