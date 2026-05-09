"""MLX 文本流式生成 (text-only, stage 1).

用法（库）:
    from mlx_omni.convert import load_mlx_model
    from mlx_omni.generate import generate_stream
    from transformers import AutoTokenizer
    model, cfg = load_mlx_model("./minimind-3o-mlx")
    tok = AutoTokenizer.from_pretrained("./minimind-3o-mlx")
    for tok_str, _ in generate_stream(model, tok, "你好", max_new_tokens=64):
        print(tok_str, end="", flush=True)
"""
from __future__ import annotations
from typing import Iterator, Tuple, Optional, List

import mlx.core as mx
import mlx.nn as nn


def _sample(logits: mx.array, temperature: float, top_p: float, top_k: int) -> mx.array:
    """单 token 采样：temperature → top-k → top-p → multinomial."""
    if temperature <= 0:
        return mx.argmax(logits, axis=-1)
    logits = logits / temperature

    # top-k
    if top_k and 0 < top_k < logits.shape[-1]:
        topk_vals = mx.partition(-logits, top_k - 1, axis=-1)[..., :top_k]  # 取 top-k 大的
        kth = -topk_vals[..., -1:]
        logits = mx.where(logits < kth, mx.full(logits.shape, -mx.inf), logits)

    # top-p
    if top_p and top_p < 1.0:
        sorted_idx = mx.argsort(-logits, axis=-1)
        sorted_logits = mx.take_along_axis(logits, sorted_idx, axis=-1)
        sorted_probs = mx.softmax(sorted_logits, axis=-1)
        cum = mx.cumsum(sorted_probs, axis=-1)
        # 第一个超过 top_p 的位置开始 mask；mask[0] 永远保留
        mask = cum > top_p
        # 把 mask 右移一位，让首个超过的还能被取到
        mask = mx.concatenate([mx.zeros_like(mask[..., :1]), mask[..., :-1]], axis=-1)
        sorted_logits = mx.where(mask, mx.full(sorted_logits.shape, -mx.inf), sorted_logits)
        # 反 scatter 回原顺序
        unsorted = mx.zeros_like(sorted_logits)
        # MLX 没有 scatter，用 take_along_axis 反向：构造一个 inverse perm
        # 简化做法：直接在 sorted 空间采样然后映射回去
        probs = mx.softmax(sorted_logits, axis=-1)
        sampled_sorted = mx.random.categorical(mx.log(probs + 1e-20))  # in sorted index
        # 取回原 vocab id
        return mx.take_along_axis(sorted_idx, sampled_sorted[..., None], axis=-1).squeeze(-1)

    return mx.random.categorical(logits)


def generate_stream(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.85,
    top_k: int = 50,
    eos_token_id: Optional[int] = None,
    use_chat_template: bool = True,
) -> Iterator[Tuple[str, int]]:
    """逐 token 生成器，yield (decoded_segment, token_id).

    use_chat_template=True 时自动应用 tokenizer.apply_chat_template, 否则原样喂.
    """
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id or 2

    # 1. tokenize
    if use_chat_template:
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, return_tensors=None)["input_ids"]
    else:
        ids = tokenizer(prompt, return_tensors=None)["input_ids"]

    input_ids = mx.array(ids, dtype=mx.int32)[None, :]  # (1, T)

    # 2. prefill: 一次性吃整个 prompt
    logits, cache, _ = model(input_ids, cache=None)
    mx.eval(logits, cache)

    last_logits = logits[:, -1, :]  # (1, vocab)

    generated = []
    decode_buf = []
    text_so_far = ""
    for _ in range(max_new_tokens):
        next_id = _sample(last_logits, temperature, top_p, top_k)
        next_id = next_id.reshape(1, 1)
        mx.eval(next_id)
        tid = int(next_id.item())

        if tid == eos_token_id:
            break
        generated.append(tid)
        decode_buf.append(tid)

        # 解码增量：累计 buffer 直到能干净 decode（处理 BPE 多字节字符）
        try:
            new_text = tokenizer.decode(decode_buf, skip_special_tokens=True)
            if new_text and not new_text.endswith("�"):
                yield new_text, tid
                text_so_far += new_text
                decode_buf = []
        except Exception:
            pass

        # decode 下一步
        logits, cache, _ = model(next_id, cache=cache)
        mx.eval(logits, cache)
        last_logits = logits[:, -1, :]

    # flush 剩余 buffer
    if decode_buf:
        try:
            tail = tokenizer.decode(decode_buf, skip_special_tokens=True)
            if tail:
                yield tail, generated[-1] if generated else 0
        except Exception:
            pass


def generate(
    model, tokenizer, prompt: str, **kw
) -> str:
    """一次性返回整段文本 (非流式)."""
    return "".join(seg for seg, _ in generate_stream(model, tokenizer, prompt, **kw))
