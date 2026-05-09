"""MLX MiniMind-O 端到端 文本→文本+语音 demo (stage 2).

用法:
    # 1. 转换 omni 权重 (一次性):
    python -m mlx_omni.convert ./minimind-3o ./minimind-3o-mlx-omni --mode omni

    # 2. 跑端到端:
    python mlx_speak.py --prompt "你好"
    python mlx_speak.py --prompt "请介绍一下北京" --output beijing.wav

    # 4-bit 量化:
    python -m mlx_omni.convert ./minimind-3o ./minimind-3o-mlx-omni-q4 --mode omni --quantize 4bit
    python mlx_speak.py --model ./minimind-3o-mlx-omni-q4 --prompt "你好"
"""
from __future__ import annotations
import argparse
import sys
import time
import warnings

import soundfile as sf
import numpy as np
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

from mlx_omni.convert import load_mlx_model
from mlx_omni.generate_omni import stream_generate_omni
from mlx_omni.mimi_bridge import MimiBridge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./minimind-3o-mlx-omni", help="MLX Omni 模型目录")
    parser.add_argument("--mimi_dir", default="./model/mimi")
    parser.add_argument("--mimi_device", default="mps", choices=["mps", "cpu", "cuda"])
    parser.add_argument("--prompt", default="你好，请介绍一下你自己。")
    parser.add_argument("--output", default="output_mlx.wav")
    parser.add_argument("--max_new_tokens", default=256, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--top_p", default=0.85, type=float)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--audio_temperature", default=0.2, type=float)
    parser.add_argument("--audio_top_k", default=50, type=int)
    parser.add_argument("--audio_rep_penalty", default=1.0, type=float, help="1.0 = 关闭")
    parser.add_argument("--audio_greedy", action="store_true")
    args = parser.parse_args()

    print(f"[load] MLX model: {args.model}")
    t0 = time.time()
    model, cfg = load_mlx_model(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"[load] done in {time.time()-t0:.2f}s | hidden={cfg.hidden_size} layers={cfg.num_hidden_layers}+{cfg.num_talker_hidden_layers}")

    print(f"[load] PyTorch Mimi codec: {args.mimi_dir} ({args.mimi_device})")
    t0 = time.time()
    mimi = MimiBridge.load(args.mimi_dir, device=args.mimi_device, dtype='float16')
    print(f"[load] done in {time.time()-t0:.2f}s")

    print(f"\n👤 {args.prompt}")
    print("🤖 ", end="", flush=True)

    audio_frames = []
    text_pieces = []
    n_text, n_audio = 0, 0
    t_gen = time.time()
    ttft_text, ttft_audio = None, None

    for text_seg, audio_frame in stream_generate_omni(
        model, tok, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        audio_temperature=args.audio_temperature,
        audio_top_k=args.audio_top_k,
        audio_rep_penalty=args.audio_rep_penalty,
        audio_greedy=args.audio_greedy,
    ):
        if text_seg:
            if ttft_text is None:
                ttft_text = (time.time() - t_gen) * 1000
            sys.stdout.write(text_seg); sys.stdout.flush()
            text_pieces.append(text_seg)
            n_text += 1
        if audio_frame:
            if ttft_audio is None:
                ttft_audio = (time.time() - t_gen) * 1000
            audio_frames.append(audio_frame)
            n_audio += 1

    dt_gen = time.time() - t_gen
    print()
    print(f"\n[gen] text tokens={n_text} audio frames={n_audio} | dt={dt_gen:.2f}s")
    if ttft_text: print(f"[gen] ttft_text={ttft_text:.0f}ms")
    if ttft_audio: print(f"[gen] ttft_audio={ttft_audio:.0f}ms")

    if not audio_frames:
        print("[warn] 没有生成 audio frames, 跳过解码")
        return

    print(f"[mimi] decoding {len(audio_frames)} frames → {args.output}")
    t0 = time.time()
    wav = mimi.decode(audio_frames)
    print(f"[mimi] decoded in {time.time()-t0:.2f}s, samples={wav.shape[0] if wav is not None else 0}")

    if wav is not None:
        sf.write(args.output, wav, 24000)
        duration = len(wav) / 24000
        print(f"[save] {args.output} ({duration:.2f}s @ 24kHz)")


if __name__ == "__main__":
    main()
