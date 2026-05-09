"""数值对齐验证：MLX vs PyTorch 输出对比.

跑同一段 prompt，对比第一个 forward 的 logits / hidden states，
确保 MLX 移植没有偏差。
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path

import numpy as np


def run_pytorch(pt_dir: str, input_ids: list, dtype: str = "float16"):
    """在 PyTorch 跑 forward, 返回 (last_hidden, logits) 都是 numpy.

    绕开 trust_remote_code (会牵扯 FunASR / SenseVoice)，
    直接用 model/model_minimind.py 里的纯 LM 类，手动 load_state_dict.
    """
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model.model_minimind import MiniMindForCausalLM, MiniMindConfig

    # 读 config
    import json
    with open(Path(pt_dir) / "config.json") as f:
        cfg_json = json.load(f)
    cfg = MiniMindConfig(**{k: v for k, v in cfg_json.items()
                            if k in ("hidden_size", "num_hidden_layers", "vocab_size",
                                     "num_attention_heads", "num_key_value_heads",
                                     "head_dim", "intermediate_size",
                                     "max_position_embeddings", "rms_norm_eps",
                                     "rope_theta", "tie_word_embeddings",
                                     "use_moe", "dropout", "flash_attn",
                                     "hidden_act", "bos_token_id", "eos_token_id")})

    # 实例化 vanilla LM (不带 Talker / encoder)
    model = MiniMindForCausalLM(cfg)
    state = torch.load(Path(pt_dir) / "pytorch_model.bin",
                        map_location="cpu", weights_only=True)
    # 过滤 Thinker 部分: model.* / lm_head.*; 跳过 talker / audio_encoder / vision_encoder / projectors
    keep = {k: v for k, v in state.items()
            if k.startswith(("model.", "lm_head."))
            and not any(s in k for s in ("talker.", "audio_encoder.",
                                          "vision_encoder.", "audio_proj.",
                                          "vision_proj.", "mimi_model.",
                                          "spk_proj.", "freqs_cos", "freqs_sin"))}
    missing, unexpected = model.load_state_dict(keep, strict=False)
    print(f"  [pt] loaded {len(keep)} keys, missing={len(missing)} unexpected={len(unexpected)}")

    if dtype == "float16":
        model = model.half()
    model = model.eval()

    ids = torch.tensor([input_ids], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=ids)
    return out.logits[0, -1].detach().float().cpu().numpy()


def run_mlx(mlx_dir: str, input_ids: list):
    import mlx.core as mx
    from mlx_omni.convert import load_mlx_model
    model, _ = load_mlx_model(mlx_dir)
    ids = mx.array([input_ids], dtype=mx.int32)
    logits, _ = model(ids, cache=None)
    mx.eval(logits)
    return np.array(logits[0, -1].astype(mx.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_dir", default="./minimind-3o")
    parser.add_argument("--mlx_dir", default="./minimind-3o-mlx")
    parser.add_argument("--prompt", default="你好")
    args = parser.parse_args()

    # 用 MLX 模型自带的 tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.mlx_dir)
    msgs = [{"role": "user", "content": args.prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text)["input_ids"]
    print(f"[input] '{args.prompt}' → {len(ids)} tokens: {ids}")

    print("\n[run] PyTorch fp16 (CPU)…")
    pt_logits = run_pytorch(args.pt_dir, ids, dtype="float16")
    print(f"  shape={pt_logits.shape} dtype={pt_logits.dtype}")
    print(f"  top-5: {[(int(i), round(float(pt_logits[i]), 3)) for i in np.argsort(-pt_logits)[:5]]}")

    print("\n[run] MLX fp16 (Metal)…")
    mlx_logits = run_mlx(args.mlx_dir, ids)
    print(f"  shape={mlx_logits.shape} dtype={mlx_logits.dtype}")
    print(f"  top-5: {[(int(i), round(float(mlx_logits[i]), 3)) for i in np.argsort(-mlx_logits)[:5]]}")

    # 数值对齐检查
    abs_diff = np.abs(pt_logits - mlx_logits)
    print(f"\n[diff] max={abs_diff.max():.4f} mean={abs_diff.mean():.4f}")

    # top-1 匹配检查
    pt_top1, mlx_top1 = int(np.argmax(pt_logits)), int(np.argmax(mlx_logits))
    print(f"[top-1] PyTorch={pt_top1} ({tok.decode([pt_top1])!r}) | "
          f"MLX={mlx_top1} ({tok.decode([mlx_top1])!r}) | "
          f"{'✅ MATCH' if pt_top1 == mlx_top1 else '❌ MISMATCH'}")

    # top-5 重合度
    pt_top5 = set(np.argsort(-pt_logits)[:5].tolist())
    mlx_top5 = set(np.argsort(-mlx_logits)[:5].tolist())
    overlap = len(pt_top5 & mlx_top5)
    print(f"[top-5] overlap = {overlap}/5 {'✅' if overlap >= 4 else '⚠️'}")

    # 余弦相似度
    cos = float(np.dot(pt_logits, mlx_logits) / (np.linalg.norm(pt_logits) * np.linalg.norm(mlx_logits) + 1e-9))
    print(f"[cos] similarity = {cos:.6f}")

    # 判定
    print()
    if cos > 0.999 and pt_top1 == mlx_top1:
        print("✅ PASS — MLX 输出与 PyTorch fp16 高度一致 (cos > 0.999, top-1 一致)")
    elif cos > 0.99 and overlap >= 4:
        print("⚠️  WARNING — fp16 量化精度差异, 但 top-5 大体一致 (可接受)")
    else:
        print("❌ FAIL — 数值差异显著, 需要排查 (RoPE / RMSNorm / attention 顺序)")


if __name__ == "__main__":
    main()
