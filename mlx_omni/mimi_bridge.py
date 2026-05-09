"""PyTorch Mimi codec 桥 (stage 2 临时方案).

mlx-audio 内置 Mimi 类要求 Kyutai 原始权重格式，从 HF kyutai/mimi 拉取的具体文件已变更。
当前实用做法: 用 transformers.MimiModel (本地 ./model/mimi/) 跑解码,
只在生成结束时调用一次 (8 codebook codes → 24kHz waveform).

后续 Stage 5 优化: 重写权重映射使用 mlx-audio 自己的 Mimi (走 Metal 加速).
"""
from __future__ import annotations
from typing import List, Optional

import numpy as np


class MimiBridge:
    """桥接 PyTorch MimiModel.

    使用方法:
        bridge = MimiBridge.load('./model/mimi', device='mps', dtype='float16')
        codes_8xT = ...  # list of 8-element frames
        wav_24k = bridge.decode(codes_8xT)  # numpy float32 array
    """
    def __init__(self, model, device: str = 'mps', dtype: str = 'float16'):
        self.model = model
        self.device = device
        self.dtype_str = dtype

    @classmethod
    def load(cls, mimi_dir: str = './model/mimi', device: str = 'mps', dtype: str = 'float16') -> 'MimiBridge':
        import torch
        from transformers import MimiModel
        m = MimiModel.from_pretrained(mimi_dir).eval()
        m = m.to(device)
        if dtype == 'float16' and device != 'cpu':
            m = m.half()
        return cls(m, device, dtype)

    def decode(self, frames: List[List[int]]) -> Optional[np.ndarray]:
        """解码一组 audio frames → 24kHz mono waveform.

        Args:
            frames: list of [c0, c1, ..., c7], 每个是 Mimi codebook 的 code id (0~2048)
                    >= 2049 视为 stop 标记, 替换为 0 (Mimi pad)
        Returns:
            numpy float32 array shape (T_samples,) at 24000 Hz, 或 None.
        """
        import torch
        codes = [f for f in frames if f and len(f) == 8]
        if not codes:
            return None
        # transpose 到 (1, 8, T)
        mc = torch.tensor(codes, dtype=torch.long).T.unsqueeze(0)
        # >= 2049 (stop tokens) → 0
        mc = torch.where(mc >= 2049, torch.zeros_like(mc), mc)
        mc = mc.to(self.device)
        with torch.no_grad():
            out = self.model.decode(mc).audio_values
        return out.squeeze().float().cpu().numpy()
