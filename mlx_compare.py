"""并排对比 fp16 / 8-bit / 4-bit 量化下的输出.

运行同样 3 个 prompt, 设固定 seed, 输出 9 个 wav 文件 + 速度报告.
"""
from __future__ import annotations
import argparse
import os
import time
import warnings
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

from mlx_omni.convert import load_mlx_model
from mlx_omni.generate_omni import stream_generate_omni
from mlx_omni.mimi_bridge import MimiBridge


PROMPTS = [
    "你好，请介绍一下你自己",
    "Tell me an interesting fact about space",
    "今天天气怎么样？",
]

MODELS = [
    ("fp16", "./minimind-3o-mlx-omni"),
    ("q8",   "./minimind-3o-mlx-omni-q8"),
    ("q4",   "./minimind-3o-mlx-omni-q4"),
]


def run_one(name, model_path, mimi, out_dir, prompt, idx, audio_temperature):
    print(f"\n[{name}] '{prompt}'")
    model, cfg = load_mlx_model(model_path)
    tok = AutoTokenizer.from_pretrained(model_path)

    # 固定 seed, 保证同一 prompt 在每个版本上采样轨迹有可比性
    mx.random.seed(42)

    t0 = time.time()
    text_pieces = []
    audio_frames = []
    n_t, n_a = 0, 0
    ttft_t, ttft_a = None, None

    for ts, af in stream_generate_omni(
        model, tok, prompt,
        max_new_tokens=180,
        temperature=0.7,
        top_p=0.85,
        top_k=50,
        audio_temperature=audio_temperature,
        audio_top_k=50,
        audio_rep_penalty=1.0,
    ):
        if ts:
            if ttft_t is None: ttft_t = (time.time() - t0) * 1000
            text_pieces.append(ts)
            n_t += 1
        if af:
            if ttft_a is None: ttft_a = (time.time() - t0) * 1000
            audio_frames.append(af)
            n_a += 1

    dt = time.time() - t0
    text = "".join(text_pieces)
    speed = n_t / max(dt, 0.01)

    # decode audio
    wav = mimi.decode(audio_frames) if audio_frames else None
    out_path = Path(out_dir) / f"compare-{idx}-{name}.wav"
    if wav is not None:
        sf.write(str(out_path), wav, 24000)
    duration = (len(wav) / 24000) if wav is not None else 0

    return {
        "name": name,
        "prompt": prompt,
        "text": text.strip(),
        "n_text": n_t,
        "n_audio": n_a,
        "dt": dt,
        "speed": speed,
        "ttft_t": ttft_t,
        "ttft_a": ttft_a,
        "wav_path": str(out_path) if wav is not None else None,
        "duration": duration,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="./compare_output")
    parser.add_argument("--mimi_dir", default="./model/mimi")
    parser.add_argument("--mimi_device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--audio_temperature", default=0.2, type=float)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[load] Mimi codec ({args.mimi_device})")
    mimi = MimiBridge.load(args.mimi_dir, device=args.mimi_device, dtype='float16')

    results = []
    for idx, prompt in enumerate(PROMPTS):
        for name, path in MODELS:
            r = run_one(name, path, mimi, args.out_dir, prompt, idx, args.audio_temperature)
            results.append(r)

    # 按 prompt 分组打印对比表
    print("\n" + "=" * 80)
    print("对比结果")
    print("=" * 80)
    for idx, prompt in enumerate(PROMPTS):
        print(f"\n📝 prompt {idx+1}: {prompt}")
        print(f"{'ver':>4}  {'text_len':>8}  {'audio_frames':>12}  {'time_s':>7}  {'speed':>9}  {'ttft_t':>7}  {'ttft_a':>7}  {'dur':>5}  output")
        print("-" * 110)
        for r in results:
            if r["prompt"] != prompt: continue
            print(f"{r['name']:>4}  {r['n_text']:>8}  {r['n_audio']:>12}  "
                  f"{r['dt']:>6.2f}s  {r['speed']:>5.1f}t/s  "
                  f"{(r['ttft_t'] or 0):>5.0f}ms  {(r['ttft_a'] or 0):>5.0f}ms  "
                  f"{r['duration']:>4.1f}s  {r['wav_path']}")
            print(f"      text: {r['text'][:80]!r}")

    # 平均速度
    print("\n" + "=" * 80)
    print("平均文本速度")
    print("=" * 80)
    for name, _ in MODELS:
        rs = [r for r in results if r["name"] == name]
        avg_speed = sum(r["speed"] for r in rs) / len(rs)
        avg_ttft_t = sum((r["ttft_t"] or 0) for r in rs) / len(rs)
        avg_ttft_a = sum((r["ttft_a"] or 0) for r in rs) / len(rs)
        print(f"  {name:>4}: avg {avg_speed:5.1f} tok/s | ttft text {avg_ttft_t:.0f}ms | ttft audio {avg_ttft_a:.0f}ms")

    # 文件大小
    print("\n" + "=" * 80)
    print("模型文件大小")
    print("=" * 80)
    for name, path in MODELS:
        sz = os.path.getsize(Path(path) / "model.safetensors") / 1024 / 1024
        print(f"  {name:>4}: {sz:5.1f} MB  ({path})")


if __name__ == "__main__":
    main()
