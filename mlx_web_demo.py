"""MLX 版 web demo — 文本/语音输入 → MLX 推理 → 文本+流式语音输出.

特点:
  - 主模型 (Thinker + Talker) 跑在 MLX (Metal GPU)
  - Mimi codec 留 PyTorch (transformers MimiModel), MPS / CPU 可选
  - SenseVoice ASR 留 PyTorch CPU (用于把语音转文本)
  - VAD 用 ONNX (CPU)

用法:
  # 1. 转换权重 (一次性):
  python -m mlx_omni.convert ./minimind-3o ./minimind-3o-mlx-omni --mode omni

  # 2. 启动:
  python mlx_web_demo.py --port 7861
  # 浏览器访问 http://localhost:7861

  # 4-bit 量化版:
  python -m mlx_omni.convert ./minimind-3o ./minimind-3o-mlx-q4 --mode omni --quantize 4bit
  python mlx_web_demo.py --model ./minimind-3o-mlx-q4 --port 7861
"""
from __future__ import annotations
import argparse, base64, io, json, os, sys, time, threading
from pathlib import Path

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, Response, request, send_from_directory
from flask_cors import CORS
from flask_sock import Sock
from pydub import AudioSegment

# 复用 PyTorch 版的 RealtimeSession (纯 Python + onnxruntime，不依赖 PyTorch 模型)
sys_path = str(Path(__file__).parent)
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

# 复用已有 webui html
WEBUI_DIR = Path(__file__).parent / "webui"

app = Flask(__name__, static_folder=str(WEBUI_DIR))
CORS(app)
sock = Sock(app)

M = {}  # model state
MODEL_LOCK = threading.Lock()


def sse(d):
    return f"data: {json.dumps(d)}\n\n"


def _load_all(args):
    print(f"[load] MLX Omni model: {args.model}")
    t0 = time.time()
    from transformers import AutoTokenizer
    from mlx_omni.convert import load_mlx_model
    M['model'], M['cfg'] = load_mlx_model(args.model)
    M['tokenizer'] = AutoTokenizer.from_pretrained(args.model)
    print(f"[load] MLX done in {time.time()-t0:.2f}s")

    print(f"[load] Mimi codec: {args.mimi_dir} ({args.mimi_device})")
    t0 = time.time()
    from mlx_omni.mimi_bridge import MimiBridge
    M['mimi'] = MimiBridge.load(args.mimi_dir, device=args.mimi_device, dtype='float16')
    print(f"[load] Mimi done in {time.time()-t0:.2f}s")

    if args.enable_asr:
        print(f"[load] SenseVoice ASR (cpu) ...")
        t0 = time.time()
        try:
            import contextlib, io as _io, logging
            logging.getLogger().setLevel(logging.ERROR)
            with contextlib.redirect_stdout(_io.StringIO()):
                from funasr import AutoModel
                M['asr'] = AutoModel(model=args.sensevoice_dir, trust_remote_code=True,
                                      disable_update=True, device='cpu')
            print(f"[load] ASR done in {time.time()-t0:.2f}s")
        except Exception as e:
            print(f"[warn] ASR load failed: {e} (语音输入将不可用)")
            M['asr'] = None
    else:
        M['asr'] = None

    print("[warmup] running warmup gen...")
    t0 = time.time()
    from mlx_omni.generate_omni import stream_generate_omni
    list(stream_generate_omni(M['model'], M['tokenizer'], "你好",
                              max_new_tokens=10, temperature=0,
                              audio_rep_penalty=1.0))
    print(f"[warmup] done in {time.time()-t0:.2f}s")
    print(f"\n[ready] MLX backend loaded. params={sum(v.size for _, v in __flat(M['model']))/1e6:.1f}M")


def __flat(model):
    from mlx.utils import tree_flatten
    return tree_flatten(model.parameters())


def _asr(samples_16k):
    if M['asr'] is None:
        return ''
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
    r = M['asr'].generate(input=samples_16k, cache={}, language='auto', use_itn=True)
    return rich_transcription_postprocess(r[0]['text']).strip() if r else ''


def _decode_pcm_chunk(frames_chunk):
    """Mimi 解码一段 frames → int16 PCM bytes (24kHz mono)."""
    wav = M['mimi'].decode(frames_chunk)
    if wav is None:
        return None
    return (wav * 32767).clip(-32768, 32767).astype(np.int16).tobytes()


# ============================================================================
#                                 Routes
# ============================================================================

@app.route('/')
def index():
    return send_from_directory(str(WEBUI_DIR), 'web_demo.html')


@app.route('/voices')
def voices():
    # 简化: MLX 版暂不接音色选择 (Stage 3+ 实现)
    return json.dumps({'builtin': ['default'], 'unseen': [], 'manual': []})


@app.route('/models')
def models():
    return json.dumps({'models': ['minimind-3o-mlx'], 'current': 'minimind-3o-mlx'})


@app.route('/switch_model', methods=['POST'])
def switch_model():
    return Response(json.dumps({'ok': True, 'model': 'minimind-3o-mlx', 'params': 113.13}),
                    mimetype='application/json')


@app.route('/chat', methods=['POST'])
def chat():
    """SSE 流式生成: 文本/语音输入 → text + 流式 PCM."""
    d = request.json or {}
    user_text = d.get('text', '') or ''
    audio_b64 = d.get('audio')

    samples = None
    if audio_b64:
        seg = (AudioSegment.from_file(io.BytesIO(base64.b64decode(audio_b64)))
               .set_frame_rate(16000).set_channels(1).set_sample_width(2))
        samples = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32) / 32768.0

    def gen():
        # 语音输入 → ASR → 文本
        prompt = user_text
        if samples is not None:
            t0 = time.time()
            asr_text = _asr(samples)
            print(f'[asr] {time.time()-t0:.2f}s: {asr_text[:60]!r}')
            prompt = asr_text or user_text
            yield sse({'type': 'user_prompt', 'content': prompt})
        elif user_text:
            yield sse({'type': 'user_prompt', 'content': user_text})

        # MLX 推理
        from mlx_omni.generate_omni import stream_generate_omni
        with MODEL_LOCK:
            t_gen = time.time()
            text_ttft, audio_ttft = None, None
            audio_buffer = []
            CHUNK_FRAMES = 12  # 与 PyTorch webui 默认一致
            n_text, n_audio = 0, 0

            for text_seg, audio_frame in stream_generate_omni(
                M['model'], M['tokenizer'], prompt,
                max_new_tokens=d.get('max_tokens', 512),
                temperature=d.get('temperature', 0.7),
                top_p=0.85, top_k=50,
                audio_temperature=0.2,
                audio_rep_penalty=1.0,  # 关掉 anti-repeat
                audio_top_k=50,
            ):
                if text_seg:
                    if text_ttft is None:
                        text_ttft = (time.time() - t_gen) * 1000
                        yield sse({'type': 'ttft', 'text_ttft': round(text_ttft, 1)})
                    yield sse({'type': 'text', 'content': text_seg})
                    n_text += 1

                if audio_frame:
                    if audio_ttft is None:
                        audio_ttft = (time.time() - t_gen) * 1000
                        yield sse({'type': 'ttft', 'audio_ttft': round(audio_ttft, 1)})
                    audio_buffer.append(audio_frame)
                    n_audio += 1
                    # 满一个 chunk 就解码并流出 PCM
                    if len(audio_buffer) >= CHUNK_FRAMES:
                        pcm = _decode_pcm_chunk(audio_buffer)
                        audio_buffer = []
                        if pcm:
                            b64 = base64.b64encode(pcm).decode()
                            for i in range(0, len(b64), 2000):
                                yield sse({
                                    'type': 'pcm',
                                    'c': b64[i:i+2000],
                                    'd': i + 2000 >= len(b64),
                                })

            # flush 剩余 audio buffer
            if audio_buffer:
                pcm = _decode_pcm_chunk(audio_buffer)
                if pcm:
                    b64 = base64.b64encode(pcm).decode()
                    for i in range(0, len(b64), 2000):
                        yield sse({'type': 'pcm', 'c': b64[i:i+2000], 'd': i + 2000 >= len(b64)})

            dt = time.time() - t_gen
            print(f'[gen] {n_text}t+{n_audio}a in {dt:.2f}s ({n_text/max(dt,0.01):.1f} tok/s)')
            yield sse({'type': 'done'})

    return Response(gen(), mimetype='text/event-stream',
                     headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ============================================================================
#                          /call 路由 + Realtime WS
# ============================================================================

@app.route('/call')
def call_page():
    return send_from_directory(str(WEBUI_DIR), 'web_demo.html')


@sock.route('/ws/realtime')
def realtime(ws):
    """实时通话 — Interaction-Model stage 1.

    使用 InteractionSession 接管整个会话生命周期:
      - recv_loop / output_loop / main_loop / foreground_loop 解耦, 分别独立线程
      - 所有事件 (audio_chunk / VAD / ASR / model_text / model_audio / barge_in...)
        统一打到一个 Timeline 上, foreground 内重新派生 history
      - 中途 barge_in: 立即 set foreground stop_event, 不破坏 session 状态
      - background scheduler 接口已就位 (mlx_omni.interaction.background), stage 2 启用

    参考: Thinking Machines "Interaction Models" (2025).
    """
    from mlx_omni.interaction import InteractionSession, SessionConfig

    vad_path = str(Path(__file__).parent / 'model' / 'vad' / 'silero_vad.onnx')
    sess = InteractionSession(
        ws=ws,
        vad_path=vad_path,
        model=M['model'],
        tokenizer=M['tokenizer'],
        mimi_bridge=M['mimi'],
        asr_model=M.get('asr'),
        config=SessionConfig(
            audio_chunk_frames=12,
            foreground_temperature=0.7,
            audio_rep_penalty=1.0,
            use_streaming_session=True,   # Stage 3: 跨 turn 复用 KV cache
            inject_bg_results=True,       # Stage 2: 自动注入 background 结果
        ),
        logger=print,
    )
    sess.run()


# ============================================================================
#                                  Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./minimind-3o-mlx-omni')
    parser.add_argument('--mimi_dir', default='./model/mimi')
    parser.add_argument('--mimi_device', default='mps', choices=['mps', 'cpu', 'cuda'])
    parser.add_argument('--sensevoice_dir', default='./model/SenseVoiceSmall')
    parser.add_argument('--enable_asr', action='store_true', default=True)
    parser.add_argument('--port', default=7861, type=int)
    args = parser.parse_args()
    _load_all(args)
    print(f"\n🚀 MLX backend ready at http://localhost:{args.port}\n")
    app.run(host='0.0.0.0', port=args.port, threaded=True)
