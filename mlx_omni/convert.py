"""PyTorch state_dict → MLX safetensors 转换器.

用法:
  python -m mlx_omni.convert ./minimind-3o ./minimind-3o-mlx [--quantize 4bit]

只处理 Thinker (主 LM) 部分。Talker / Audio / Vision 投影器在 stage 2 加。

PyTorch 权重命名 (来自 transformers AutoModelForCausalLM 保存的 .bin):
  model.embed_tokens.weight                                      → 同名
  model.layers.{i}.input_layernorm.weight                        → 同名
  model.layers.{i}.post_attention_layernorm.weight               → 同名
  model.layers.{i}.self_attn.{q,k,v,o}_proj.weight               → 同名
  model.layers.{i}.self_attn.{q,k}_norm.weight                   → 同名
  model.layers.{i}.mlp.{gate,up,down}_proj.weight                → 同名
  model.norm.weight                                              → 同名
  lm_head.weight                                                 → 跳过 (tied)

会一并跳过 (Omni 模型才有，stage 2 处理):
  thinker.*  / talker.*  / audio_encoder.*  / vision_encoder.*
  audio_proj.* / vision_proj.* / mimi_model.*
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import mlx.core as mx

from .model import OmniConfig, MiniMindForCausalLM
from .omni import MiniMindOmniLM


def _should_skip(k: str) -> bool:
    """跳过的权重 (frozen encoders, RoPE buffer, mimi-model)."""
    skip = (
        "audio_encoder.", "vision_encoder.", "mimi_model.",
        "freqs_cos", "freqs_sin",
    )
    return any(s in k for s in skip)


def _is_thinker_only_key(k: str) -> bool:
    """识别 Thinker 主模型权重 (用于纯 LM 加载)."""
    if _should_skip(k):
        return False
    talker_prefixes = ("talker.", "audio_proj.", "vision_proj.")
    return not any(s in k for s in talker_prefixes)


def _normalize_thinker_key(k: str) -> str:
    """Thinker only: thinker.* → model.* (PyTorch CausalLM 命名)."""
    if k.startswith("thinker."):
        return "model." + k[len("thinker."):]
    return k


# ---- Talker 权重命名重映射 ----
# PyTorch nn.Sequential 子模块用整数命名 (e.g. ".0.", ".2."), MLX 用 .layers.{i}.
# 需要把以下结构里的整数索引改写到 .layers.{i}.
_SEQ_PARENTS = (
    "talker.codec_proj.",
    "talker.embed_proj.",
)
# 还有 adapters 的子模块也是 Sequential
import re as _re
_RE_ADAPTER = _re.compile(r"talker\.(?:lm_head|embed_tokens)\.adapters\.\d+\.")


def _normalize_omni_key(k: str) -> str:
    """OmniLM 命名重映射: PyTorch state_dict → MLX 参数树.

    映射规则 (PyTorch → MLX):
      model.<X>                       → thinker.model.<X>          (Thinker 主模型)
      lm_head.weight                  → 跳过 (tied)
      talker.codec_proj.<i>.<X>       → talker.codec_proj.layers.<i>.<X>
      talker.embed_proj.<i>.<X>       → talker.embed_proj.layers.<i>.<X>
      talker.{lm_head|embed_tokens}.adapters.<i>.<j>.<X>
                                      → ...adapters.<i>.layers.<j>.<X>
      audio_proj.mlp.<i>.<X>          → audio_proj.layers.<i>.<X>
      vision_proj.mlp.<i>.<X>         → vision_proj.layers.<i>.<X>
      其他 talker.*                   → 同名直通
    """
    if _should_skip(k):
        return None
    # 1. thinker 主模型: model.X → thinker.model.X (保留 .model. 因为 OmniLM.thinker 是 CausalLM)
    if k.startswith("model."):
        k = "thinker." + k  # → thinker.model.X
    # lm_head.weight (tied with embed_tokens) → 跳过
    if k == "lm_head.weight":
        return None
    # 2. talker 的 Sequential 投影 (codec_proj, embed_proj)
    for parent in _SEQ_PARENTS:
        if k.startswith(parent):
            tail = k[len(parent):]
            m = _re.match(r"(\d+)\.(.*)", tail)
            if m:
                k = parent + f"layers.{m.group(1)}." + m.group(2)
            break
    # 3. talker 的 adapters Sequential
    m = _RE_ADAPTER.match(k)
    if m:
        prefix = m.group(0)
        tail = k[len(prefix):]
        m2 = _re.match(r"(\d+)\.(.*)", tail)
        if m2:
            k = prefix + f"layers.{m2.group(1)}." + m2.group(2)
    # 4. audio_proj / vision_proj: PyTorch 用 self.mlp = Sequential(...), MLX 用 layers list
    #    audio_proj.mlp.<i>.X → audio_proj.layers.<i>.X
    if k.startswith("audio_proj.mlp.") or k.startswith("vision_proj.mlp."):
        k = k.replace(".mlp.", ".layers.", 1)
    return k


def convert_pt_to_mlx(
    pt_path: str,
    out_dir: str,
    dtype: str = "float16",
    quantize: str = "none",
    mode: str = "thinker",  # "thinker" 或 "omni"
):
    """转换主入口.

    Args:
        pt_path: 包含 pytorch_model.bin / config.json 的目录 (如 ./minimind-3o)
                 或者直接是 .pth 文件路径
        out_dir: 输出 MLX 模型目录
        dtype: float16 | bfloat16 | float32
        quantize: none | 4bit | 8bit
    """
    import torch  # 转换时才导入 PyTorch
    pt_path = Path(pt_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读 config
    cfg_path = pt_path / "config.json" if pt_path.is_dir() else None
    if cfg_path and cfg_path.exists():
        with open(cfg_path) as f:
            pt_cfg = json.load(f)
    else:
        # 用默认 (minimind-3o)
        pt_cfg = {}

    config = OmniConfig(
        hidden_size=pt_cfg.get("hidden_size", 768),
        num_hidden_layers=pt_cfg.get("num_hidden_layers", 8),
        vocab_size=pt_cfg.get("vocab_size", 6400),
        num_attention_heads=pt_cfg.get("num_attention_heads", 8),
        num_key_value_heads=pt_cfg.get("num_key_value_heads", 4),
        head_dim=pt_cfg.get("head_dim", 96),
        intermediate_size=pt_cfg.get("intermediate_size", 2432),
        max_position_embeddings=pt_cfg.get("max_position_embeddings", 32768),
        rms_norm_eps=pt_cfg.get("rms_norm_eps", 1e-6),
        rope_theta=pt_cfg.get("rope_theta", 1e6),
        tie_word_embeddings=pt_cfg.get("tie_word_embeddings", True),
    )
    print(f"[config] hidden={config.hidden_size} layers={config.num_hidden_layers} "
          f"heads={config.num_attention_heads}/{config.num_key_value_heads} "
          f"vocab={config.vocab_size} ffn={config.intermediate_size}")

    # 2. 读权重
    bin_candidates = [
        pt_path / "pytorch_model.bin",
        pt_path / "model.safetensors",
    ]
    pt_state = None
    if pt_path.is_file():
        pt_state = torch.load(pt_path, map_location="cpu")
    else:
        for c in bin_candidates:
            if c.exists():
                if c.suffix == ".bin":
                    pt_state = torch.load(c, map_location="cpu", weights_only=True)
                else:
                    from safetensors.torch import load_file as _lf
                    pt_state = _lf(str(c))
                print(f"[load] {c}")
                break
    if pt_state is None:
        raise FileNotFoundError(f"No pytorch_model.bin or model.safetensors in {pt_path}")

    # 3. 过滤 Thinker 权重 + 重命名
    target_dtype = {
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
        "float32": mx.float32,
    }[dtype]

    mlx_state = {}
    skipped = []
    if mode == "thinker":
        is_keep = _is_thinker_only_key
        normalize = _normalize_thinker_key
    elif mode == "omni":
        is_keep = lambda k: not _should_skip(k)
        normalize = _normalize_omni_key
    else:
        raise ValueError(f"Unknown mode: {mode}")

    for k, v in pt_state.items():
        if not is_keep(k):
            skipped.append(k); continue
        new_k = normalize(k)
        if new_k is None:
            skipped.append(k + " (filtered)"); continue
        # tied embedding：lm_head 与 thinker.embed_tokens.weight 共享，转换时跳过 lm_head
        if (new_k == "lm_head.weight" or new_k == "thinker.lm_head.weight") and config.tie_word_embeddings:
            skipped.append(k + " (tied)"); continue
        # PyTorch tensor → numpy → mx.array
        arr = mx.array(v.detach().cpu().to(torch.float32).numpy()).astype(target_dtype)
        # 标量参数 (text_scale, audio_scale): 保持 shape (1,) 一致
        mlx_state[new_k] = arr

    print(f"[convert] mode={mode} kept {len(mlx_state)} tensors, skipped {len(skipped)}")

    # 4. 实例化 MLX 模型 (按 mode), 加载权重
    if mode == "thinker":
        model = MiniMindForCausalLM(config)
    else:
        model = MiniMindOmniLM(config)
    # MLX 用 tree_unflatten + load_weights 导入
    weights_list = list(mlx_state.items())
    try:
        model.load_weights(weights_list, strict=True)
    except Exception as e:
        # 详细诊断
        print(f"[error] strict load failed: {e}")
        print("[diag] expected keys (top 5):")
        from mlx.utils import tree_flatten
        for k, _ in tree_flatten(model.parameters())[:5]:
            print(" ", k)
        print("[diag] got keys (top 5):")
        for k, _ in weights_list[:5]:
            print(" ", k)
        raise

    # 5. 量化（可选）
    if quantize != "none":
        bits = int(quantize.replace("bit", ""))
        print(f"[quantize] {bits}-bit")
        nn_quantize_predicate = lambda _, m: hasattr(m, "to_quantized")
        import mlx.nn
        mlx.nn.quantize(model, group_size=64, bits=bits, class_predicate=None)

    # 6. 保存
    weights_path = out_dir / "model.safetensors"
    config_path = out_dir / "config.json"

    # MLX 保存权重: 用 tree_flatten 拍平
    from mlx.utils import tree_flatten
    flat_params = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(weights_path), flat_params)
    print(f"[save] {weights_path} ({sum(v.nbytes for v in flat_params.values())/1e6:.1f} MB)")

    # config.json (MLX 自定义格式)
    cfg_dict = {
        "model_type": "minimind-omni-mlx",
        "mode": mode,  # thinker | omni
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "vocab_size": config.vocab_size,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "intermediate_size": config.intermediate_size,
        "max_position_embeddings": config.max_position_embeddings,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_theta": config.rope_theta,
        "tie_word_embeddings": config.tie_word_embeddings,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        # Omni-specific
        "num_talker_hidden_layers": config.num_talker_hidden_layers,
        "talker_hidden_size": config.talker_hidden_size,
        "audio_vocab_size": config.audio_vocab_size,
        "audio_pad_token": config.audio_pad_token,
        "audio_stop_token": config.audio_stop_token,
        "audio_spk_token": config.audio_spk_token,
        "spk_emb_size": config.spk_emb_size,
        "bridge_layer": config.bridge_layer,
        "dtype": dtype,
        "quantization": quantize,
    }
    with open(config_path, "w") as f:
        json.dump(cfg_dict, f, indent=2)
    print(f"[save] {config_path}")

    # 7. 复制 tokenizer 文件
    if pt_path.is_dir():
        for fn in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"]:
            src = pt_path / fn
            if src.exists():
                import shutil
                shutil.copy(src, out_dir / fn)
                print(f"[copy] {fn}")

    print(f"\n[done] MLX model saved to {out_dir}")
    return model, config


def load_mlx_model(model_dir: str):
    """加载已转换的 MLX 模型.

    根据 config.json 里 'mode' 字段自动选择 thinker/omni 类。
    """
    model_dir = Path(model_dir)
    with open(model_dir / "config.json") as f:
        cfg_dict = json.load(f)
    config = OmniConfig(
        hidden_size=cfg_dict["hidden_size"],
        num_hidden_layers=cfg_dict["num_hidden_layers"],
        vocab_size=cfg_dict["vocab_size"],
        num_attention_heads=cfg_dict["num_attention_heads"],
        num_key_value_heads=cfg_dict["num_key_value_heads"],
        head_dim=cfg_dict["head_dim"],
        intermediate_size=cfg_dict["intermediate_size"],
        max_position_embeddings=cfg_dict["max_position_embeddings"],
        rms_norm_eps=cfg_dict["rms_norm_eps"],
        rope_theta=cfg_dict["rope_theta"],
        tie_word_embeddings=cfg_dict.get("tie_word_embeddings", True),
        bos_token_id=cfg_dict.get("bos_token_id", 1),
        eos_token_id=cfg_dict.get("eos_token_id", 2),
        num_talker_hidden_layers=cfg_dict.get("num_talker_hidden_layers", 4),
        talker_hidden_size=cfg_dict.get("talker_hidden_size", 768),
        audio_vocab_size=cfg_dict.get("audio_vocab_size", 2112),
        audio_pad_token=cfg_dict.get("audio_pad_token", 2049),
        audio_stop_token=cfg_dict.get("audio_stop_token", 2050),
        audio_spk_token=cfg_dict.get("audio_spk_token", 2051),
        spk_emb_size=cfg_dict.get("spk_emb_size", 192),
        bridge_layer=cfg_dict.get("bridge_layer", cfg_dict["num_hidden_layers"] // 2 - 1),
    )
    mode = cfg_dict.get("mode", "thinker")
    if mode == "omni":
        model = MiniMindOmniLM(config)
    else:
        model = MiniMindForCausalLM(config)

    # 量化的话需要先 quantize 模型再加载权重
    if cfg_dict.get("quantization", "none") != "none":
        bits = int(cfg_dict["quantization"].replace("bit", ""))
        import mlx.nn
        mlx.nn.quantize(model, group_size=64, bits=bits, class_predicate=None)

    weights_path = model_dir / "model.safetensors"
    weights = mx.load(str(weights_path))
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pt_path", help="PyTorch model dir or .pth file")
    parser.add_argument("out_dir", help="MLX output dir")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--quantize", default="none", choices=["none", "4bit", "8bit"])
    parser.add_argument("--mode", default="thinker", choices=["thinker", "omni"],
                        help="thinker: 仅文本 LM (stage 1); omni: 含 Talker (stage 2+)")
    args = parser.parse_args()
    convert_pt_to_mlx(args.pt_path, args.out_dir, args.dtype, args.quantize, args.mode)
