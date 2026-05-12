"""InteractionSession - 实时通话主状态机 (Thinking Machines interaction-model 风格 stage 1).

设计思路:
  - WebSocket 来的字节流和文本控制消息 → input_queue
  - 主循环按事件驱动: 跑 VAD / 触发状态转换 / 启动 foreground generate
  - foreground generate 在子线程跑, 可被外部 stop_event 中断
  - 模型输出 (text / pcm) 也写到 output_queue, 由独立线程发回 WebSocket
  - 所有动作都在 timeline 上留痕, 让 scheduler 后面能据此做更智能的调度

四个线程:
  1. recv_loop          — ws.receive → input_queue (control + bytes 分类)
  2. main_loop          — 主调度 (在调用方线程跑)，按事件状态机调动作
  3. foreground_loop    — 每次 trigger 都新起一个 worker, 跑 stream_generate_omni
  4. output_loop        — output_queue → ws.send (避免并发 send)

状态机:

   listening ─speech_start─▶ user_speaking
       ▲                            │
       │                       speech_end
       │                            │
       │                            ▼
   gen_done ◀── speaking ◀── thinking (ASR 完成, foreground 启动)
       │           │
       │      barge_in
       │           │
       │           ▼
       └───── interrupted ── (等下个 speech_end 重新进 thinking)
"""
from __future__ import annotations
import base64
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional

import numpy as np

from .events import (
    Event,
    AUDIO_CHUNK, TEXT_DELTA, IMAGE_FRAME, CONTROL_MSG,
    SPEECH_START, SPEECH_END, BARGE_IN, ASR_RESULT,
    MODEL_TEXT, MODEL_AUDIO, MODEL_PCM, STATUS, TTFT, GEN_DONE,
)
from .timeline import Timeline


# 状态机
LISTENING = "listening"
USER_SPEAKING = "user_speaking"
THINKING = "thinking"
SPEAKING = "speaking"
INTERRUPTED = "interrupted"


@dataclass
class SessionConfig:
    """会话级配置 (可由调用方覆盖)."""
    max_history_turns: int = 2
    audio_chunk_frames: int = 12          # Mimi 12 frame ≈ 960ms — 流式 PCM 一个解码块
    foreground_max_new_tokens: int = 512
    foreground_temperature: float = 0.7
    foreground_top_p: float = 0.85
    foreground_top_k: int = 50
    audio_temperature: float = 0.2
    audio_top_k: int = 50
    audio_rep_penalty: float = 1.0
    audio_greedy: bool = False


class InteractionSession:
    def __init__(
        self,
        ws,                                       # flask-sock simple-websocket
        vad_path: str,
        model,                                    # MLX OmniLM
        tokenizer,
        mimi_bridge,                              # MimiBridge 实例
        asr_model=None,                           # FunASR AutoModel 或 None
        config: Optional[SessionConfig] = None,
        logger: Optional[Callable[[str], None]] = None,
    ):
        self.ws = ws
        self.model = model
        self.tokenizer = tokenizer
        self.mimi = mimi_bridge
        self.asr = asr_model
        self.cfg = config or SessionConfig()
        self.log = logger or print

        self.timeline = Timeline()

        # 输入/输出通道
        self._input_q: queue.Queue = queue.Queue()
        self._output_q: queue.Queue = queue.Queue()

        # 状态
        self._state = LISTENING
        self._state_lock = threading.Lock()
        self._alive = threading.Event(); self._alive.set()
        self._stop_event = threading.Event()      # 外部触发停止整个 session
        self._fg_stop_event = threading.Event()   # 仅停止当前 foreground generate
        self._fg_worker: Optional[threading.Thread] = None
        self._fg_lock = threading.Lock()          # foreground 启停互斥

        # VAD (复用 PyTorch 版的纯 Python 实现, 仅依赖 onnxruntime)
        from model.model_omni import RealtimeSession
        self._vad = RealtimeSession(vad_path)

    # ====================================================================
    # 状态控制
    # ====================================================================

    def state(self) -> str:
        with self._state_lock:
            return self._state

    def _set_state(self, new_state: str, **payload):
        with self._state_lock:
            old = self._state
            if old == new_state:
                return
            self._state = new_state
        self.timeline.add(Event.now(STATUS, prev=old, state=new_state, **payload))
        # 通知前端 (stage 2 可在这里增加 scheduler 钩子)
        self._send({"type": "status", "state": new_state, **payload})

    # ====================================================================
    # 启动 / 退出
    # ====================================================================

    def run(self):
        """阻塞调用方线程，跑主调度。返回时表示 session 结束。"""
        t_recv = threading.Thread(target=self._recv_loop, daemon=True, name="iact-recv")
        t_out = threading.Thread(target=self._output_loop, daemon=True, name="iact-out")
        t_recv.start(); t_out.start()
        try:
            self._main_loop()
        finally:
            self._alive.clear()
            self._stop_event.set()
            self._fg_stop_event.set()

    def stop(self):
        self._stop_event.set()
        self._alive.clear()
        self._fg_stop_event.set()

    # ====================================================================
    # I/O loops
    # ====================================================================

    def _recv_loop(self):
        """从 WebSocket 拿数据塞到 input_queue, 不解析."""
        while self._alive.is_set():
            try:
                data = self.ws.receive(timeout=1)
            except Exception:
                self._alive.clear(); break
            if data is None:
                self._alive.clear(); break
            self._input_q.put(data)

    def _output_loop(self):
        """串行发送, 避免 ws.send 并发."""
        while self._alive.is_set():
            try:
                msg = self._output_q.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                if isinstance(msg, (bytes, bytearray)):
                    self.ws.send(msg)
                else:
                    self.ws.send(json.dumps(msg))
            except Exception:
                self._alive.clear(); break

    def _send(self, msg: dict):
        self._output_q.put(msg)

    # ====================================================================
    # 主调度循环 (跑在调用方线程)
    # ====================================================================

    def _main_loop(self):
        while self._alive.is_set():
            try:
                data = self._input_q.get(timeout=0.05)
            except queue.Empty:
                continue
            if isinstance(data, (bytes, bytearray)):
                self._handle_audio_chunk(data)
            else:
                self._handle_control(data)

    def _handle_control(self, raw: str):
        try:
            msg = json.loads(raw)
        except Exception:
            return
        t = msg.get("type")
        self.timeline.add(Event.now(CONTROL_MSG, **msg))
        if t == "context":
            # 老的 context 消息保留; history 不再由前端推, 由 timeline 派生
            pass
        elif t == "stop":
            self._fg_stop_event.set()  # 仅停当前 foreground
        elif t == "end":
            self.stop()
        elif t == "text":
            text = msg.get("content", "")
            if text:
                self.timeline.add(Event.now(TEXT_DELTA, text=text))
                self._trigger_foreground(prompt=text)

    def _handle_audio_chunk(self, raw_bytes: bytes):
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        cur = self.state()
        # foreground 生成中也要让 VAD 持续跑用于检测 barge_in
        self._vad.generating = (cur == SPEAKING)
        status = self._vad.push_chunk(samples)
        speaking = self._vad.speaking
        self._send({"type": "vad", "speaking": speaking})

        if status == "interrupt":
            if cur == SPEAKING:
                self.timeline.add(Event.now(BARGE_IN))
                self._fg_stop_event.set()
                self._set_state(INTERRUPTED)
        elif speaking and cur == LISTENING:
            self.timeline.add(Event.now(SPEECH_START))
            self._set_state(USER_SPEAKING)
        elif status == "speech_end":
            self.timeline.add(Event.now(SPEECH_END))
            audio = self._vad.get_audio()
            self._do_asr_and_trigger(audio)

    # ====================================================================
    # ASR + foreground 调度
    # ====================================================================

    def _do_asr_and_trigger(self, audio_samples: np.ndarray):
        """ASR + foreground 都在 main_loop 线程跑.

        关键: 不开 worker thread, MLX 在非主线程的 Metal context 行为不稳。
        生成的每一步都会从 input_q 抽干积压的 audio chunk 给 VAD,
        VAD 触发 barge_in 时 set stop_event, 立即跳出 generate 循环。
        """
        self._set_state(THINKING)
        text = ""
        if self.asr is not None:
            try:
                from funasr.utils.postprocess_utils import rich_transcription_postprocess
                t0 = time.time()
                r = self.asr.generate(input=audio_samples, cache={}, language="auto", use_itn=True)
                text = rich_transcription_postprocess(r[0]["text"]).strip() if r else ""
                self.log(f"[asr] {time.time()-t0:.2f}s: {text[:60]!r}")
            except Exception as e:
                self.log(f"[asr] failed: {e}")
        self.timeline.add(Event.now(ASR_RESULT, text=text))
        self._send({"type": "user_prompt", "content": text or "(无识别)"})
        if not text:
            self._set_state(LISTENING)
            return
        self._foreground_generate(prompt=text)

    def _drain_input_for_vad(self):
        """foreground 期间抽干 input_q, 只跑 VAD 检 barge_in. 控制消息也处理."""
        while True:
            try:
                data = self._input_q.get_nowait()
            except queue.Empty:
                return
            if isinstance(data, (bytes, bytearray)):
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                self._vad.generating = True
                status = self._vad.push_chunk(samples)
                speaking = self._vad.speaking
                self._send({"type": "vad", "speaking": speaking})
                if status == "interrupt":
                    self.timeline.add(Event.now(BARGE_IN))
                    self._fg_stop_event.set()
                    self._set_state(INTERRUPTED)
                    return
            else:
                try:
                    m = json.loads(data)
                    if m.get("type") in ("stop", "end"):
                        self._fg_stop_event.set()
                        if m.get("type") == "end":
                            self.stop()
                        return
                except Exception:
                    pass

    def _foreground_generate(self, prompt: str):
        """在 main_loop 线程跑生成 (MLX 不能在子线程稳定运行)."""
        from mlx_omni.generate_omni import stream_generate_omni

        history_full = self.timeline.derive_history(max_turns=self.cfg.max_history_turns + 1)
        history = history_full[:-1] if history_full and history_full[-1].get("role") == "user" else history_full

        self._set_state(SPEAKING)
        self._fg_stop_event.clear()
        t0 = time.time()
        ttft_t = ttft_a = None
        n_text = n_audio = 0
        audio_buffer: List[List[int]] = []
        interrupted = False
        self.log(f"[fg] start, prompt={prompt[:60]!r}, history_len={len(history)}")

        try:
            for text_seg, audio_frame in stream_generate_omni(
                self.model, self.tokenizer, prompt,
                max_new_tokens=self.cfg.foreground_max_new_tokens,
                temperature=self.cfg.foreground_temperature,
                top_p=self.cfg.foreground_top_p,
                top_k=self.cfg.foreground_top_k,
                audio_temperature=self.cfg.audio_temperature,
                audio_top_k=self.cfg.audio_top_k,
                audio_rep_penalty=self.cfg.audio_rep_penalty,
                audio_greedy=self.cfg.audio_greedy,
                history=history,
                stop_event=self._fg_stop_event,
            ):
                # 每个 token 抽干输入 → VAD → 检 barge_in
                self._drain_input_for_vad()
                if self._fg_stop_event.is_set():
                    interrupted = True; break
                if text_seg:
                    if ttft_t is None:
                        ttft_t = (time.time() - t0) * 1000
                        self.timeline.add(Event.now(TTFT, kind="text", ms=ttft_t))
                        self._send({"type": "ttft", "text_ttft": round(ttft_t, 1)})
                    self.timeline.add(Event.now(MODEL_TEXT, content=text_seg))
                    self._send({"type": "text", "content": text_seg})
                    n_text += 1
                if audio_frame:
                    if ttft_a is None:
                        ttft_a = (time.time() - t0) * 1000
                        self.timeline.add(Event.now(TTFT, kind="audio", ms=ttft_a))
                        self._send({"type": "ttft", "audio_ttft": round(ttft_a, 1)})
                    audio_buffer.append(audio_frame)
                    n_audio += 1
                    if len(audio_buffer) >= self.cfg.audio_chunk_frames:
                        self._emit_pcm(audio_buffer)
                        audio_buffer = []
            if not interrupted and audio_buffer:
                self._emit_pcm(audio_buffer)
        except Exception as e:
            import traceback; traceback.print_exc()
            self.log(f"[fg] exception: {e}")
            interrupted = True

        dt = time.time() - t0
        self.log(f"[fg] done | t={n_text} a={n_audio} dt={dt:.2f}s interrupted={interrupted}")
        self.timeline.add(Event.now(GEN_DONE, interrupted=interrupted,
                                    n_text=n_text, n_audio=n_audio, dt=dt))
        self._send({"type": "done", "interrupted": interrupted})
        self._vad.generating = False
        self._set_state(LISTENING)

    def _emit_pcm(self, frames: List[List[int]]):
        wav = self.mimi.decode(frames)
        if wav is None:
            return
        pcm = (wav * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
        self.timeline.add(Event.now(MODEL_PCM, bytes=len(pcm)))
        self._send({"type": "pcm", "data": base64.b64encode(pcm).decode()})
