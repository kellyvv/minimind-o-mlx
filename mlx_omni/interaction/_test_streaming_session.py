"""Stage 3 持久 KV streaming session 一致性测试.

对比: streaming (KV cache reuse) vs stateless (每 turn 从头 prefill)
跑同一段多轮对话, greedy 采样, 看两条路径的文本输出是否一致.

跑法:
    python -m mlx_omni.interaction._test_streaming_session

不一致的常见原因:
  - LCP truncate 把"应该保留"的 token 也丢了
  - cache 里有但 full_ids 漏了, 导致下次 prefix_match 错位
  - 浮点累积误差 (理论上少, 但 fp16/q8 下不能完全排除)
"""
from __future__ import annotations
import warnings, sys
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer
from mlx_omni.convert import load_mlx_model
from mlx_omni.interaction.streaming_session import OmniStreamingSession
from mlx_omni.interaction.session import SessionConfig
from mlx_omni.generate_omni import stream_generate_omni


def run_streaming(model, tok, turns):
    sess = OmniStreamingSession(model, tok, SessionConfig())
    outputs = []
    for u in turns:
        chunks = []
        for text_seg, _ in sess.generate_turn(u, [],
                                              max_new_tokens=40,
                                              temperature=0.0):  # greedy
            if text_seg:
                chunks.append(text_seg)
        outputs.append("".join(chunks))
    return outputs


def run_stateless(model, tok, turns):
    history = []
    outputs = []
    for u in turns:
        chunks = []
        for text_seg, _ in stream_generate_omni(model, tok, u,
                                                max_new_tokens=40,
                                                temperature=0.0,
                                                history=history):
            if text_seg:
                chunks.append(text_seg)
        out = "".join(chunks)
        outputs.append(out)
        history.append({"role": "user", "content": u})
        history.append({"role": "assistant", "content": out})
    return outputs


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="./minimind-3o-mlx-omni-q8")
    args = p.parse_args()

    model, cfg = load_mlx_model(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)

    turns = [
        "你好",
        "今天天气怎么样?",
        "谢谢",
        "再见",
    ]

    print(f"=== Stage 3 consistency test on {args.model} ===\n")
    print(f"Turns: {turns}\n")

    print("--- streaming (KV reuse) ---")
    s = run_streaming(model, tok, turns)
    for i, o in enumerate(s):
        print(f"  turn {i+1}: {o!r}")

    print("\n--- stateless (re-prefill each turn) ---")
    t = run_stateless(model, tok, turns)
    for i, o in enumerate(t):
        print(f"  turn {i+1}: {o!r}")

    print("\n--- diff ---")
    matches = sum(1 for a, b in zip(s, t) if a == b)
    print(f"  {matches}/{len(turns)} turns match exactly")
    for i, (a, b) in enumerate(zip(s, t)):
        if a != b:
            # 找第一个分歧点
            common = 0
            for ca, cb in zip(a, b):
                if ca != cb: break
                common += 1
            print(f"  turn {i+1} diverges at char {common}:")
            print(f"    streaming: ...{a[max(0,common-10):common+20]!r}")
            print(f"    stateless: ...{b[max(0,common-10):common+20]!r}")

    return 0 if matches == len(turns) else 1


if __name__ == "__main__":
    sys.exit(main())
