"""持久 MLX streaming session (stage 3 实现).

跨 turn 复用 Thinker + Talker 的 KV cache, 减少多轮对话的 prefill 成本.

核心思路:
  - 一个 WebSocket 通话期间持有同一个 OmniStreamingSession 实例
  - 第 1 轮: 正常 prefill 整段 chat template + 生成
  - 第 2+ 轮: 只 prefill 新增的 tokens (assistant 上轮回复尾巴 + 这轮 user 头部),
              再生成. KV cache 累积保留所有上文.

关键状态:
  cache:       MLX (thinker N + talker M) 层的 KV 缓存
  position:    当前 KV cache 的有效长度 (token 维度)
  ids_history: 截至 position 已喂进去的所有 text token ids (1D list)

注意:
  - 被打断的 assistant 输出仍会留在 KV cache 里 (模型"记得"自己说到一半被打断)
    这是简化处理; 真正干净做法需要 truncate_cache_to(position_before_assistant).
  - 长会话需要 sliding window — 当前 budget 是 max_position_embeddings (32k),
    用尽前不主动剪裁。

参考 Thinking Machines "Interaction Models" 中"persistent GPU-memory sequence"
的设想; 本实现是纯 Python 单进程版本, 跟随单条 WebSocket 生命周期.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@dataclass
class _TurnState:
    """单 turn 的 MTP 状态; turn 结束后重置."""
    audio_codes: List[List[int]]
    audio_stop_pos: List[Optional[int]]
    text_finished: bool
    decode_buf: List[int]


class OmniStreamingSession:
    """单 WebSocket 通话的持久生成器.

    复用 KV cache 跨 turn, 但每个 turn 内部走和 stream_generate_omni 一样的
    MTP delay 逻辑。
    """

    def __init__(self, model, tokenizer, cfg, logger: Optional[Callable[[str], None]] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg                              # SessionConfig
        self.log = logger or (lambda s: None)

        # 持久状态
        self.cache: Optional[List] = None
        self.position: int = 0                      # KV cache 已包含多少个 token
        self.full_chat_text: str = ""               # 上一次 build 出的完整 chat template 文本
        self.full_ids: List[int] = []               # 对应的 token ids
        # audio_buffer (1, 8, T_position): 9-channel 输入里的 audio 部分
        # 累积保存所有历史位置的 audio code (用户 prompt 处填 audio_pad, 生成处填真实 code)
        self.audio_buffer: Optional[mx.array] = None

        # 内部维护的对话历史 (代替外部传入). 用我们 cache 里实际生成的文本,
        # 保证下次 prefill 时 prefix_match.
        self._internal_history: List[dict] = []
        self._current_assistant_text: str = ""      # 本 turn 累积的 assistant 文本

    # ==================================================================
    # 公开 API
    # ==================================================================

    def generate_turn(
        self,
        user_text: str,
        history: list,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.85,
        top_k: int = 50,
        audio_temperature: float = 0.2,
        audio_top_k: int = 50,
        audio_rep_penalty: float = 1.0,
        audio_greedy: bool = False,
        eos_token_id: Optional[int] = None,
        stop_event=None,
        return_audio_codes: bool = True,
    ) -> Iterator[Tuple[Optional[str], Optional[List[int]]]]:
        """单 turn 流式生成. 复用上 turn 的 KV cache."""
        cfg = self.model.config
        eos_id = eos_token_id if eos_token_id is not None else (self.tokenizer.eos_token_id or 2)
        audio_pad = cfg.audio_pad_token

        # Stage 3 关键: 使用 *自己* 维护的 internal_history (上次生成的文本会自动追加进来).
        # 外部传入的 history 仅在 session 完全空的时候作为 seed 使用。
        if not self._internal_history and history:
            self._internal_history = list(history)

        # 1. 追加这一轮的 user 消息, 然后渲染整段 chat template
        self._internal_history.append({"role": "user", "content": user_text})
        text = self.tokenizer.apply_chat_template(
            self._internal_history, tokenize=False, add_generation_prompt=True
        )
        new_full_ids = self.tokenizer(text)["input_ids"]

        # 2. 找最长公共前缀, 只 trim cache 到分歧点 (而不是整个重置)
        common = 0
        for a, b in zip(new_full_ids, self.full_ids):
            if a != b: break
            common += 1

        if common < self.position:
            # cache 多出的尾巴 (decode→encode 漂移 / 上次 assistant 末尾) 需要丢弃
            self.log(f"[streaming] trim cache: {self.position} → {common}")
            self.cache = _truncate_cache(self.cache, common)
            self.audio_buffer = self.audio_buffer[:, :, :common] if self.audio_buffer is not None else None
            self.position = common
            self.full_ids = self.full_ids[:common]

        new_ids = new_full_ids[self.position:]
        if not new_ids:
            # 不应该发生 (至少要有 user message 的 token)
            self.log("[streaming] no new tokens to prefill? skipping")
            return

        # 2. prefill 新增 tokens
        t_prefill_start = time.time()
        prefill_input = mx.array([new_ids], dtype=mx.int32)
        prefill_audio = mx.full((1, 8, len(new_ids)), audio_pad, dtype=mx.int32)
        nine = mx.concatenate([prefill_audio, prefill_input[:, None, :]], axis=1)
        text_logits, audio_logits_list, self.cache = self.model(nine, cache=self.cache)
        mx.eval(text_logits, *audio_logits_list, *(c[0] for c in self.cache), *(c[1] for c in self.cache))

        # 更新持久状态
        new_full_position = self.position + len(new_ids)
        if self.audio_buffer is None:
            self.audio_buffer = prefill_audio
        else:
            self.audio_buffer = mx.concatenate([self.audio_buffer, prefill_audio], axis=2)
        self.position = new_full_position
        self.full_ids = new_full_ids
        self.full_chat_text = text

        prefill_dt = (time.time() - t_prefill_start) * 1000
        self.log(f"[streaming] prefill {len(new_ids)} tokens in {prefill_dt:.0f}ms (cache pos: {self.position})")

        # 3. 进入 decode 循环 — 跟 stream_generate_omni 主体逻辑保持一致
        ts = _TurnState(
            audio_codes=[[] for _ in range(8)],
            audio_stop_pos=[None] * 8,
            text_finished=False,
            decode_buf=[],
        )
        input_ids = mx.array([new_full_ids], dtype=mx.int32)  # 累计 ids, 用于尾部 sample
        self._current_assistant_text = ""

        for step in range(max_new_tokens):
            if stop_event is not None and stop_event.is_set():
                break

            last_text = text_logits[0, -1, :]
            last_audio = [al[0, -1, :] for al in audio_logits_list]

            # 文本采样
            if not ts.text_finished:
                text_token = _sample_text(last_text, temperature, top_p, top_k)
            else:
                text_token = 234 if step == 0 else 0

            # MTP 音频采样
            audio_step = step - 1
            for i in range(8):
                if audio_step < i:
                    ts.audio_codes[i].append(audio_pad)
                else:
                    code = _sample_audio(
                        last_audio[i], ts.audio_codes[i],
                        temperature=audio_temperature, top_k=audio_top_k,
                        rep_penalty=audio_rep_penalty, greedy=audio_greedy,
                    )
                    ts.audio_codes[i].append(code)
                    if ts.audio_stop_pos[i] is None and code >= 2048:
                        ts.audio_stop_pos[i] = len(ts.audio_codes[i]) - 1

            if ts.text_finished and all(p is not None for p in ts.audio_stop_pos):
                break

            # 推进 input + audio_buffer 各 1 列
            input_ids = mx.concatenate([input_ids, mx.array([[text_token]], dtype=mx.int32)], axis=1)
            new_col_codes = []
            for i in range(8):
                if audio_step >= i:
                    new_col_codes.append(ts.audio_codes[i][-1])
                else:
                    new_col_codes.append(audio_pad)
            new_col = mx.array([[[c] for c in new_col_codes]], dtype=mx.int32)
            self.audio_buffer = mx.concatenate([self.audio_buffer, new_col], axis=2)

            # 下一步单 token forward
            next_input = mx.concatenate(
                [self.audio_buffer[:, :, -1:], input_ids[:, -1:][:, None, :]],
                axis=1
            )
            text_logits, audio_logits_list, self.cache = self.model(next_input, cache=self.cache)
            mx.eval(text_logits, *audio_logits_list)

            # 更新位置
            self.position += 1
            self.full_ids = self.full_ids + [text_token]

            # 抽取本步输出
            text_segment = None
            if not ts.text_finished:
                ts.decode_buf.append(text_token)
                try:
                    seg = self.tokenizer.decode(ts.decode_buf, skip_special_tokens=True)
                    if seg and not seg.endswith("�"):
                        text_segment = seg
                        self._current_assistant_text += seg
                        ts.decode_buf = []
                except Exception:
                    pass

            audio_frame = None
            if return_audio_codes and audio_step >= 7:
                frame = []
                for i in range(8):
                    src_idx = step - 7 + i
                    if 0 <= src_idx < len(ts.audio_codes[i]):
                        frame.append(ts.audio_codes[i][src_idx])
                    else:
                        frame.append(audio_pad)
                active = all(
                    (ts.audio_stop_pos[i] is None or step - 7 + i < ts.audio_stop_pos[i])
                    for i in range(8)
                )
                if active:
                    audio_frame = frame

            if not ts.text_finished and text_token == eos_id:
                ts.text_finished = True

            if text_segment or audio_frame:
                yield text_segment, audio_frame

        # flush 文本 decode buffer
        if ts.decode_buf:
            try:
                tail = self.tokenizer.decode(ts.decode_buf, skip_special_tokens=True)
                if tail:
                    self._current_assistant_text += tail
                    yield tail, None
            except Exception:
                pass

        # 把本轮 assistant 输出 commit 到 internal_history,
        # 让下一轮的 chat template prefix 跟 cache 完全一致。
        self._internal_history.append({
            "role": "assistant",
            "content": self._current_assistant_text,
        })

    def reset(self):
        """主动清空 cache. 一般不需要主动调; 历史 mismatch 时会自动 reset."""
        self.cache = None
        self.position = 0
        self.audio_buffer = None
        self.full_ids = []
        self.full_chat_text = ""


# ============================================================================
# 缓存工具
# ============================================================================

def _truncate_cache(cache: Optional[list], new_len: int):
    """把每层 (k, v) 缓存沿 seq 维 truncate 到 new_len. cache 元素若为 None 不动."""
    if cache is None:
        return None
    out = []
    for c in cache:
        if c is None:
            out.append(None); continue
        k, v = c
        out.append((k[:, :new_len], v[:, :new_len]))
    return out


# ============================================================================
# 采样函数 (跟 generate_omni.py 同名实现一致, 局部 copy 避免循环依赖)
# ============================================================================

def _sample_text(logits, temperature, top_p, top_k):
    if temperature <= 0:
        return int(mx.argmax(logits).item())
    logits = logits / temperature
    if 0 < top_k < logits.shape[-1]:
        topk_vals = mx.partition(-logits, top_k - 1, axis=-1)[..., :top_k]
        kth = -topk_vals[..., -1:]
        logits = mx.where(logits < kth, mx.full(logits.shape, -mx.inf), logits)
    if top_p < 1.0:
        sorted_idx = mx.argsort(-logits, axis=-1)
        sorted_logits = mx.take_along_axis(logits, sorted_idx, axis=-1)
        cum = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
        mask = cum > top_p
        mask = mx.concatenate([mx.zeros_like(mask[..., :1]), mask[..., :-1]], axis=-1)
        sorted_logits = mx.where(mask, mx.full(sorted_logits.shape, -mx.inf), sorted_logits)
        sampled = mx.random.categorical(sorted_logits)
        return int(mx.take_along_axis(sorted_idx, sampled[..., None], axis=-1).item())
    return int(mx.random.categorical(logits).item())


def _sample_audio(logits, codes_history, temperature=0.2, top_k=50, rep_penalty=1.0, greedy=False):
    if greedy:
        return int(mx.argmax(logits).item())
    logits = logits / max(temperature, 1e-6)
    if rep_penalty != 1.0 and codes_history:
        recent = codes_history[-3:]
        for c in recent:
            v = logits[c].item()
            new_v = v / rep_penalty if v > 0 else v * rep_penalty
            mask = mx.arange(logits.shape[-1]) == c
            logits = mx.where(mask, mx.array(new_v, dtype=logits.dtype), logits)
    if top_k > 0 and top_k < logits.shape[-1]:
        topk_vals = mx.partition(-logits, top_k - 1, axis=-1)[..., :top_k]
        kth = -topk_vals[..., -1:]
        logits = mx.where(logits < kth, mx.full(logits.shape, -mx.inf), logits)
    return int(mx.random.categorical(logits).item())
