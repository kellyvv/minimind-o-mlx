"""SenseVoice / SigLIP 编码器桥 (PyTorch → MLX).

策略: 编码器留 PyTorch (CPU/MPS), 输出 features 通过 numpy 中转到 MLX,
后续 audio_proj / vision_proj 在 MLX 端跑.

理由:
  - SenseVoice (FunASR) 移植 MLX 工作量极大，且只在每个请求开头跑一次
  - SigLIP2 同理 — 仅图像输入时调用
  - 主成本是 Talker MTP 循环 (每步 8 次采样)，已在 MLX 跑

API:
    bridge = EncoderBridge(audio_dir='./model/SenseVoiceSmall',
                           vision_dir='./model/siglip2-base-p32-256-ve',
                           device='cpu')
    audio_emb = bridge.encode_audio(wav_samples_16k_mono)     # (T_audio, audio_hidden=512) numpy
    image_emb = bridge.encode_image(pil_image)                # (image_token_len=64, 768) numpy

    # native audio input 路径: 跳过 ASR
    embeds = build_audio_injected_embeds(model, text_ids, audio_features_mx, audio_pad_id)
    # 直接喂 model(text_ids, inputs_embeds=embeds)
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import warnings

import numpy as np


class EncoderBridge:
    def __init__(
        self,
        audio_dir: Optional[str] = './model/SenseVoiceSmall',
        vision_dir: Optional[str] = './model/siglip2-base-p32-256-ve',
        device: str = 'cpu',  # MPS 上 FunASR 不稳, 默认 CPU
    ):
        self.device = device
        self._audio_encoder = None
        self._audio_processor = None
        self._vision_encoder = None
        self._vision_processor = None
        if audio_dir:
            self._load_audio(audio_dir)
        if vision_dir:
            self._load_vision(vision_dir)

    # -------- audio (SenseVoice) --------

    def _load_audio(self, path: str):
        try:
            import torch, contextlib, io as _io, logging
            logging.getLogger().setLevel(logging.ERROR)
            with contextlib.redirect_stdout(_io.StringIO()):
                from funasr import AutoModel
                m = AutoModel(model=path, trust_remote_code=True,
                              disable_update=True, device='cpu')  # FunASR 加载强制 CPU
            self._audio_encoder = m.model.encoder.eval().float().to(self.device)
            self._audio_frontend = m.kwargs['frontend']
        except Exception as e:
            warnings.warn(f'[EncoderBridge] SenseVoice load failed: {e}')

    def encode_audio(self, wav_16k_mono) -> Optional[np.ndarray]:
        """wav_16k_mono: numpy float32 [-1,1] at 16kHz, shape (T_samples,) or (1, T_samples)
        Returns: numpy float32 (T_frames, 512) - audio_hidden_size
        """
        if self._audio_encoder is None:
            return None
        import torch
        if isinstance(wav_16k_mono, np.ndarray):
            x = torch.from_numpy(wav_16k_mono).float()
        else:
            x = wav_16k_mono.float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            fbank, flen = self._audio_frontend(x, torch.tensor([x.size(1)]))
            valid = fbank.to(self.device)
            valid_lens = flen.to(self.device)
            emb, _ = self._audio_encoder(valid, valid_lens)
        # emb shape: (1, T_frames, 512)
        T = min(int(flen[0].item()), emb.size(1))
        return emb[0, :T].cpu().numpy().astype(np.float32)

    # -------- vision (SigLIP) --------

    def _load_vision(self, path: str):
        try:
            from transformers import SiglipImageProcessor, SiglipVisionModel
            self._vision_encoder = SiglipVisionModel.from_pretrained(path).eval().to(self.device)
            self._vision_processor = SiglipImageProcessor.from_pretrained(path)
        except Exception as e:
            warnings.warn(f'[EncoderBridge] SigLIP load failed: {e}')

    def encode_image(self, pil_image) -> Optional[np.ndarray]:
        """Returns numpy float32 (image_token_len=64, 768)."""
        if self._vision_encoder is None:
            return None
        import torch
        inputs = self._vision_processor(images=pil_image, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._vision_encoder(**inputs).last_hidden_state  # (1, 64, 768)
        return out[0].cpu().numpy().astype(np.float32)


# ============================================================================
# Audio embedding injection (native audio input 路径)
# ============================================================================

def build_audio_injected_embeds(
    model,                          # MiniMindOmniLM (MLX)
    text_ids,                       # mx.array shape (1, T_text) int32
    audio_features_np,              # numpy (T_audio, audio_hidden=512) 或 None
    audio_pad_token: int = 16,
):
    """构造已经把 audio_pad 位置替换为 audio 投影后特征的 inputs_embeds.

    用法:
        text_ids = tokenizer("user: <|audio_pad|>" * T + "\\nassistant:")
        audio_np = encoder_bridge.encode_audio(wav)        # (T_audio, 512)
        embeds = build_audio_injected_embeds(model, text_ids, audio_np)
        model(text_ids, inputs_embeds=embeds)

    实现镜像 PyTorch model_omni.py 中的 inject_audio_features:
      - 找文本里连续的 audio_pad 段
      - 用 audio_proj(audio_features) 的对应行替换
      - 只替换第一段连续 audio_pad (与 PyTorch 行为对齐)
    """
    import mlx.core as mx
    # 1. 先正常 embed 整段 text_ids → (1, T, hidden)
    embeds = model.thinker.model.embed_tokens(text_ids)
    if audio_features_np is None or audio_features_np.size == 0:
        return embeds

    # 2. 投影 audio features → (T_audio, hidden)
    audio_mx = mx.array(audio_features_np)
    audio_proj = model.audio_proj(audio_mx[None, ...])[0]   # (T_audio, hidden)

    # 3. 找 text_ids 里连续的 audio_pad 段, 用 audio_proj 替换
    seq = text_ids[0].tolist()
    T = len(seq)
    # 单次只替换第一段连续 audio_pad
    i = 0
    while i < T and seq[i] != audio_pad_token:
        i += 1
    if i >= T:
        # 没找到 audio_pad，没法注入
        return embeds
    start = i
    while i < T and seq[i] == audio_pad_token:
        i += 1
    end = i  # [start, end) 是连续 audio_pad
    inject_len = min(audio_proj.shape[0], end - start)
    if inject_len == 0:
        return embeds

    # 4. 重组 embeds: head + audio_proj[:inject_len] + tail
    head = embeds[:, :start, :]
    injected = audio_proj[None, :inject_len, :]
    tail = embeds[:, start + inject_len:, :]
    new_embeds = mx.concatenate([head, injected, tail], axis=1)
    return new_embeds
