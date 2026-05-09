"""MLX 版 MiniMind-O 文本对话 CLI (stage 1).

用法:
    # 1. 先转换权重 (一次性):
    python -m mlx_omni.convert ./minimind-3o ./minimind-3o-mlx --dtype float16

    # 2. 跑对话:
    python mlx_chat.py
    python mlx_chat.py --prompt "你好"
    python mlx_chat.py --benchmark

    # 4-bit 量化:
    python -m mlx_omni.convert ./minimind-3o ./minimind-3o-mlx-q4 --quantize 4bit
    python mlx_chat.py --model ./minimind-3o-mlx-q4
"""
from __future__ import annotations
import argparse
import time
import sys

import mlx.core as mx
from transformers import AutoTokenizer

from mlx_omni.convert import load_mlx_model
from mlx_omni.generate import generate_stream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./minimind-3o-mlx", help="MLX 模型目录")
    parser.add_argument("--tokenizer", default=None, help="tokenizer 目录 (默认同 --model)")
    parser.add_argument("--prompt", default=None, help="单轮 prompt; 不给则进入交互式")
    parser.add_argument("--max_new_tokens", default=256, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--top_p", default=0.85, type=float)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--benchmark", action="store_true", help="跑速度基准")
    args = parser.parse_args()

    print(f"[load] {args.model}")
    t0 = time.time()
    model, cfg = load_mlx_model(args.model)
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    print(f"[load] done in {time.time()-t0:.2f}s | "
          f"params={sum(v.size for _, v in __flat(model))/1e6:.1f}M | "
          f"hidden={cfg.hidden_size} layers={cfg.num_hidden_layers}")

    if args.benchmark:
        _bench(model, tok)
        return

    if args.prompt:
        _run_one(model, tok, args)
        return

    # 交互式
    print("\n📒 MiniMind-O MLX. 输入消息聊天，Ctrl+D / 'exit' 退出。\n")
    while True:
        try:
            user = input("👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user in ("exit", "quit"): break
        if not user: continue
        print("🤖 Assistant: ", end="", flush=True)
        _run_with(model, tok, user, args)
        print()


def _run_one(model, tok, args):
    print(f"👤 {args.prompt}")
    print("🤖 ", end="", flush=True)
    _run_with(model, tok, args.prompt, args)
    print()


def _run_with(model, tok, prompt, args):
    t0 = time.time()
    n_tokens = 0
    first_token_time = None
    for seg, tid in generate_stream(
        model, tok, prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    ):
        if first_token_time is None:
            first_token_time = time.time() - t0
        sys.stdout.write(seg); sys.stdout.flush()
        n_tokens += 1
    dt = time.time() - t0
    sys.stdout.write(
        f"\n  [ttft={first_token_time*1000:.0f}ms total={dt:.2f}s "
        f"tokens={n_tokens} speed={n_tokens/max(dt,0.01):.1f} tok/s]"
    )


def _bench(model, tok, n_warmup=3, n_runs=3):
    prompts = [
        "Hello, please introduce yourself.",
        "请用一段话介绍一下你自己。",
        "What's the capital of France?",
    ]
    print(f"\n[benchmark] {n_warmup} warmup + {n_runs} runs / {len(prompts)} prompts")
    for i in range(n_warmup):
        list(generate_stream(model, tok, prompts[0], max_new_tokens=20, temperature=0))
    print("[bench] warmup done")

    all_speeds = []
    for p in prompts:
        for r in range(n_runs):
            t0 = time.time()
            n_tokens = 0
            ftt = None
            for _, _ in generate_stream(model, tok, p, max_new_tokens=80, temperature=0.7):
                if ftt is None: ftt = time.time() - t0
                n_tokens += 1
            dt = time.time() - t0
            speed = n_tokens / max(dt, 0.01)
            all_speeds.append(speed)
            print(f"  [{p[:25]:25}] run {r+1}: ttft={ftt*1000:5.0f}ms "
                  f"tokens={n_tokens:3d} speed={speed:5.1f} tok/s")

    avg = sum(all_speeds) / len(all_speeds)
    print(f"\n[summary] avg speed = {avg:.1f} tok/s ({min(all_speeds):.1f}–{max(all_speeds):.1f})")


def __flat(model):
    from mlx.utils import tree_flatten
    return tree_flatten(model.parameters())


if __name__ == "__main__":
    main()
