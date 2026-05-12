"""MLX Omni 流式生成 (text+audio).

完全对应 PyTorch model_omni.py 中的 stream_generate, 含 MTP delay:

  step  text  audio_layer_idx_active
   0    t0    -                       (only text)
   1    t1    [0]                     (audio layer 0 starts)
   2    t2    [0,1]                   (audio layer 1 joins)
   ...
   7    t7    [0..6]                  (layer 6 joins)
   8    t8    [0..7] → 第一帧 audio_frame 输出 (8 codes 来自 step 1..8)
   9    t9    [0..7] → 第二帧
   ...

每帧 audio_frame 的 8 个 codes 是从不同 step 抽出的 (i 层的 step k-7+i 时的 code).
yield: (input_ids_so_far, audio_frame_or_None) 每步一次.
"""
from __future__ import annotations
from typing import Iterator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


def _sample_text(
    logits: mx.array,
    temperature: float,
    top_p: float,
    top_k: int,
    rp: float = 1.0,
    seen_ids: Optional[mx.array] = None,
) -> int:
    """文本 token 采样 (单样本)."""
    if temperature <= 0:
        return int(mx.argmax(logits).item())
    logits = logits / temperature

    # repetition penalty
    if rp != 1.0 and seen_ids is not None:
        scores = logits[seen_ids]
        scaled = mx.where(scores > 0, scores / rp, scores * rp)
        logits[seen_ids] = scaled  # in-place via fancy indexing 不支持,用下面方法
        # MLX 不支持 in-place fancy assign, 退化方案: 跳过 rp (text rp 影响小)
        # TODO: 实现一个 scatter 替代

    # top-k
    if 0 < top_k < logits.shape[-1]:
        topk_vals = mx.partition(-logits, top_k - 1, axis=-1)[..., :top_k]
        kth = -topk_vals[..., -1:]
        logits = mx.where(logits < kth, mx.full(logits.shape, -mx.inf), logits)

    # top-p
    if top_p < 1.0:
        sorted_idx = mx.argsort(-logits, axis=-1)
        sorted_logits = mx.take_along_axis(logits, sorted_idx, axis=-1)
        cum = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
        mask = cum > top_p
        mask = mx.concatenate([mx.zeros_like(mask[..., :1]), mask[..., :-1]], axis=-1)
        sorted_logits = mx.where(mask, mx.full(sorted_logits.shape, -mx.inf), sorted_logits)
        sampled_sorted = mx.random.categorical(sorted_logits)
        return int(mx.take_along_axis(sorted_idx, sampled_sorted[..., None], axis=-1).item())

    return int(mx.random.categorical(logits).item())


def _sample_audio(
    logits: mx.array,
    audio_codes_history: list,  # 该层最近的 codes (用于 anti-repeat)
    temperature: float = 0.2,
    top_k: int = 50,
    rep_penalty: float = 1.0,
    rep_window: int = 3,
    greedy: bool = False,
) -> int:
    """单层音频 code 采样."""
    if greedy:
        return int(mx.argmax(logits).item())
    logits = logits / max(temperature, 1e-6)

    # anti-repeat (注意: MLX 没有 in-place fancy assign，构造一个 mask)
    if rep_penalty != 1.0 and rep_window > 0 and audio_codes_history:
        recent = audio_codes_history[-rep_window:]
        for c in recent:
            v = logits[c].item()
            new_v = v / rep_penalty if v > 0 else v * rep_penalty
            # 用 scatter 方式更新单个 index (MLX 不支持复杂 in-place，用 mx.put 或者重建)
            # 简化做法: 用 boolean mask
            mask = mx.arange(logits.shape[-1]) == c
            logits = mx.where(mask, mx.array(new_v, dtype=logits.dtype), logits)

    # top-k
    if top_k > 0 and top_k < logits.shape[-1]:
        topk_vals = mx.partition(-logits, top_k - 1, axis=-1)[..., :top_k]
        kth = -topk_vals[..., -1:]
        logits = mx.where(logits < kth, mx.full(logits.shape, -mx.inf), logits)

    return int(mx.random.categorical(logits).item())


def stream_generate_omni(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.85,
    top_k: int = 50,
    audio_temperature: float = 0.2,
    audio_top_k: int = 50,
    audio_rep_penalty: float = 1.0,
    audio_greedy: bool = False,
    eos_token_id: Optional[int] = None,
    use_chat_template: bool = True,
    return_audio_codes: bool = True,
    history: Optional[list] = None,
    spk_emb: Optional[mx.array] = None,
    stop_event=None,
    on_step=None,
    audio_features_np=None,         # numpy (T_audio, audio_hidden=512) 或 None
    user_text_template: Optional[str] = None,  # 当 audio_features_np 给定时使用, 例如 "<|audio_pad|>" * N
) -> Iterator[Tuple[Optional[str], Optional[List[int]]]]:
    """流式生成器。

    Yields: (text_segment_or_None, audio_frame_or_None)
        - text_segment 是新解码出的字符串增量
        - audio_frame 是 8 元素 list, 每个是 [0, 2112) 的 Mimi code (或包含 stop>=2048)

    Stage-1 interaction-model 钩子:
        stop_event: 可选 threading.Event; 每步循环开头检查, 被 set 时立即 break
                    (用于 InteractionSession 的 barge_in 中断). Partial 状态留在
                    audio_codes/decode_buf 等 local 变量, 由调用方在 yield 期间
                    自己累积; 这里不再单独 attach 到 generator 对象上.
        on_step:    可选 callable(step, text_token, audio_step) — 每步采样完成后调用,
                    用于 InteractionSession 累计统计或推送状态事件.
    """
    cfg = model.config
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id or 2

    # 1. tokenize prompt
    # 原生 audio input 路径: user message 用 <|audio_pad|>*N 占位, 后面通过
    # inputs_embeds 把 audio_proj(audio_features) 注入到对应位置.
    if audio_features_np is not None and audio_features_np.size > 0:
        T_audio = int(audio_features_np.shape[0])
        # 用 audio_pad 字面字符串构造 user message
        audio_marker = cfg.audio_special_token if hasattr(cfg, "audio_special_token") else "<|audio_pad|>"
        user_content = (user_text_template or "") + audio_marker * T_audio
    else:
        user_content = prompt

    msgs = (history or []) + [{"role": "user", "content": user_content}]
    if use_chat_template:
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        text = user_content
    ids = tokenizer(text)["input_ids"]
    input_ids = mx.array([ids], dtype=mx.int32)  # (1, T0)
    start_pos = input_ids.shape[1]

    # 2. 构造 9 通道输入: audio_buffer (1, 8, T0) 全填 audio_pad
    audio_buffer = mx.full(
        (1, 8, start_pos), cfg.audio_pad_token, dtype=mx.int32
    )

    # 3. prefill — 区分两条路径
    if audio_features_np is not None and audio_features_np.size > 0:
        # 原生音频输入: 构造 inputs_embeds 把 audio_proj 输出注入到 audio_pad 位置
        from .encoder_bridge import build_audio_injected_embeds
        audio_pad_id = cfg.audio_ids[0] if hasattr(cfg, "audio_ids") else 16
        inputs_embeds = build_audio_injected_embeds(
            model, input_ids, audio_features_np, audio_pad_token=audio_pad_id,
        )
        # 用 inputs_embeds + input_ids 走模式 1
        text_logits, audio_logits_list, cache = model(
            input_ids, spk_emb=spk_emb, inputs_embeds=inputs_embeds,
        )
    else:
        # 原 9 通道路径
        nine_chan = mx.concatenate([audio_buffer, input_ids[:, None, :]], axis=1)
        text_logits, audio_logits_list, cache = model(nine_chan, spk_emb=spk_emb)
    mx.eval(text_logits, *audio_logits_list, *(c[0] for c in cache), *(c[1] for c in cache))

    # state
    audio_codes: List[List[int]] = [[] for _ in range(8)]
    audio_stop_pos: List[Optional[int]] = [None] * 8
    text_finished = False
    decode_buf: List[int] = []
    seen_ids = list(set(ids))  # 用于 rp，简化版不用

    # 4. 单步循环
    for step in range(max_new_tokens):
        # interaction-model 钩子: 外部 stop_event 触发立即中断
        if stop_event is not None and stop_event.is_set():
            break
        # 取最后一个位置的 logits
        last_text = text_logits[0, -1, :]
        last_audio = [al[0, -1, :] for al in audio_logits_list]

        # ---- 采样文本 ----
        if not text_finished:
            text_token = _sample_text(last_text, temperature, top_p, top_k)
        else:
            # 文本已结束: 输出 enter (\n=234) 第一次, 后续 pad (0)
            text_token = 234 if step == 0 else 0  # 简化

        # ---- 采样 audio (MTP delay) ----
        # audio_step = step - 1, 第 i 层在 audio_step >= i 时启动
        audio_step = step - 1
        for i in range(8):
            if audio_step < i:
                audio_codes[i].append(cfg.audio_pad_token)
            else:
                code = _sample_audio(
                    last_audio[i], audio_codes[i],
                    temperature=audio_temperature, top_k=audio_top_k,
                    rep_penalty=audio_rep_penalty, greedy=audio_greedy,
                )
                audio_codes[i].append(code)
                if audio_stop_pos[i] is None and code >= 2048:
                    audio_stop_pos[i] = len(audio_codes[i]) - 1

        # 终止条件: 文本结束且所有 8 层都遇到 stop
        if text_finished and all(p is not None for p in audio_stop_pos):
            break

        # 更新 input_ids
        input_ids = mx.concatenate([input_ids, mx.array([[text_token]], dtype=mx.int32)], axis=1)

        # 更新 audio_buffer: append 一列 audio_pad，再把这一步生成的 code 写进去
        new_col = mx.full((1, 8, 1), cfg.audio_pad_token, dtype=mx.int32)
        # 把当前 step 各层最新 code 写到新列
        # 第 i 层只在 audio_step >= i 时是有效 code, 否则就是 pad
        new_col_codes = []
        for i in range(8):
            if audio_step >= i:
                new_col_codes.append(audio_codes[i][-1])
            else:
                new_col_codes.append(cfg.audio_pad_token)
        new_col = mx.array([[[c] for c in new_col_codes]], dtype=mx.int32)
        audio_buffer = mx.concatenate([audio_buffer, new_col], axis=2)

        # 下一步 forward (单 token)
        next_input = mx.concatenate(
            [audio_buffer[:, :, -1:], input_ids[:, -1:][:, None, :]],
            axis=1
        )
        text_logits, audio_logits_list, cache = model(next_input, cache=cache, spk_emb=spk_emb)
        mx.eval(text_logits, *audio_logits_list)

        # ---- 提取本步输出 ----
        # 文本增量解码
        text_segment = None
        if not text_finished:
            decode_buf.append(text_token)
            try:
                seg = tokenizer.decode(decode_buf, skip_special_tokens=True)
                if seg and not seg.endswith("�"):
                    text_segment = seg
                    decode_buf = []
            except Exception:
                pass

        # audio_frame: 当 audio_step >= 7 时输出一帧 (来自 step-7 到 step 的 8 个 codes 错位组合)
        audio_frame = None
        if return_audio_codes and audio_step >= 7:
            frame = []
            for i in range(8):
                src_step_idx = step - 7 + i  # 第 i 层用第 src_step_idx 时刻的 code
                if 0 <= src_step_idx < len(audio_codes[i]):
                    frame.append(audio_codes[i][src_step_idx])
                else:
                    frame.append(cfg.audio_pad_token)
            # 全部 8 层这帧都还活着 (没到 stop) 才输出
            active = all(
                (audio_stop_pos[i] is None or step - 7 + i < audio_stop_pos[i])
                for i in range(8)
            )
            if active:
                audio_frame = frame

        # 检查文本是否结束
        if not text_finished and text_token == eos_token_id:
            text_finished = True

        # interaction-model 钩子: 每步回调
        if on_step is not None:
            try:
                on_step(step, text_token, audio_step)
            except Exception:
                pass

        # yield
        if text_segment or audio_frame:
            yield text_segment, audio_frame

    # flush 剩余 text decode buffer
    if decode_buf:
        try:
            tail = tokenizer.decode(decode_buf, skip_special_tokens=True)
            if tail:
                yield tail, None
        except Exception:
            pass
