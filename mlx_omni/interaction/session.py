"""InteractionSession - 实时通话主状态机 (Thinking Machines interaction-model 风格).

设计思路:
  - WebSocket 来的字节流和文本控制消息 → input_queue
  - 主循环按事件驱动: 跑 VAD / 触发状态转换 / 调 foreground generate
  - foreground generate **同步**在主循环线程跑 (MLX Metal context 在非主线程不稳定),
    每个 token 之间通过 _drain_input_for_vad() 抽干积压音频, VAD 触发 barge_in 时
    set foreground_stop_event, 下一个 token 检查即可立即退出
  - 模型输出 (text / pcm) 走独立 output_loop 线程串行写 WebSocket, 避免并发 send
  - 所有动作都在 timeline 上留痕, scheduler / background 可据此做调度决策

三个长寿线程 (per WebSocket connection):
  1. recv_loop     — ws.receive → input_queue (不解析)
  2. main_loop     — 主调度 (调用方线程)，按事件状态机调动作 + 跑 foreground 生成
  3. output_loop   — output_queue → ws.send (串行化)

状态机:

   listening ─speech_start─▶ user_speaking
       ▲                            │
       │                       speech_end
       │                            │
       │                            ▼
   listening ◀── speaking ◀── thinking (ASR 完成, foreground 启动)
       ▲           │
       │      barge_in
       │           │
       │           ▼
       └───── interrupted ── (foreground 跳出循环, 状态切回 listening)
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
    BG_JOB_START, BG_JOB_RESULT,
)
from .timeline import Timeline
from .background import (
    BackgroundScheduler, BackgroundJob,
    make_echo_job, make_summarize_history_job, make_current_time_job,
)


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
    # Stage 3: 持久 KV cache 复用 (跨 turn 不重新 prefill)
    use_streaming_session: bool = False
    # Stage 2: 后台任务结果自动插入下一轮 assistant prompt
    inject_bg_results: bool = True
    # 流式 Mimi 解码: 每次解码时多带 N 帧上下文, 输出时丢掉前面 overlap_drop 帧的样本.
    # 减少 chunk 边界的相位跳变 (实测能显著降低"颤抖"). 0 = 关闭, 默认 4 ≈ 320ms.
    audio_overlap: int = 4
    # 边界淡入淡出长度 (samples) — 在新 PCM 块开头做线性 ramp-in
    # 进一步抹平和上一块的拼接处. 0 = 关闭, 默认 240 ≈ 10ms @ 24kHz.
    audio_crossfade_samples: int = 240


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

        # Stage 2: 后台任务调度器
        self.scheduler = BackgroundScheduler(on_event=self._on_bg_event)

        # Stage 3: 持久 streaming session (lazy init, 仅在 use_streaming_session=True 时启用)
        self._streaming: Optional[Any] = None

        # PCM crossfade 状态: 每 turn 第一块不做 ramp-in
        self._first_audio_chunk_of_turn = True

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
        # 不能直接 **msg 展开 (msg 里有 'type' 键, 会和 Event.now 的 type 参数冲突)
        payload = {k: v for k, v in msg.items() if k != "type"}
        self.timeline.add(Event.now(CONTROL_MSG, msg_type=t, **payload))
        if t == "context":
            pass
        elif t == "stop":
            self._fg_stop_event.set()
        elif t == "end":
            self.stop()
        elif t == "text":
            text = msg.get("content", "")
            if text:
                self.timeline.add(Event.now(TEXT_DELTA, text=text))
                self._foreground_generate(prompt=text)
        elif t == "bg_submit":
            self._handle_bg_submit(msg)
        elif t == "bg_cancel":
            jid = msg.get("job_id")
            if jid: self.scheduler.cancel(jid)

    # ====================================================================
    # Stage 2: 后台任务
    # ====================================================================

    def _on_bg_event(self, kind: str, job: BackgroundJob):
        """BackgroundScheduler 完成回调; 落 timeline + 通知前端."""
        if kind == "start":
            self.timeline.add(Event.now(BG_JOB_START, job_id=job.job_id, job_kind=job.kind))
            self._send({"type": "bg_start", "job_id": job.job_id, "kind": job.kind})
        elif kind == "result":
            self.timeline.add(Event.now(
                BG_JOB_RESULT,
                job_id=job.job_id, job_kind=job.kind,
                result=job.result, error=job.error,
                duration=job.duration,
            ))
            r_str = (job.result[:300] if isinstance(job.result, str) else str(job.result)[:300]) if job.result is not None else None
            self._send({
                "type": "bg_result",
                "job_id": job.job_id, "kind": job.kind,
                "result": r_str, "error": job.error,
                "duration": round(job.duration, 2),
            })

    def _handle_bg_submit(self, msg: dict):
        kind = msg.get("kind", "echo")
        if kind == "echo":
            text = msg.get("text", "(no text)")
            delay = float(msg.get("delay", 2.0))
            self.scheduler.submit("echo", make_echo_job(text, delay), text=text, delay=delay)
        elif kind == "summarize":
            self.scheduler.submit("summarize", make_summarize_history_job(self.timeline))
        elif kind == "time":
            self.scheduler.submit("time", make_current_time_job())
        else:
            self.log(f"[bg] unknown kind: {kind}")

    def submit_background(self, kind: str, fn, **meta) -> str:
        return self.scheduler.submit(kind, fn, **meta)

    def _consume_bg_results(self) -> str:
        """收割已完成但 foreground 未消费的 job 结果, 拼成一段前缀注入下轮 assistant."""
        if not self.cfg.inject_bg_results:
            return ""
        unconsumed = self.scheduler.list_unconsumed_results()
        if not unconsumed:
            return ""
        snippets = []
        for j in sorted(unconsumed, key=lambda x: x.finished or 0):
            if j.error:
                self.scheduler.mark_consumed(j.job_id); continue
            r = j.result if isinstance(j.result, str) else str(j.result)
            snippets.append(f"[background:{j.kind}] {r[:200]}")
            self.scheduler.mark_consumed(j.job_id)
        return "\n".join(snippets)

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
        """在 main_loop 线程跑生成 (MLX 不能在子线程稳定运行).

        新增能力 (stage 2/3):
          - 自动消费 background scheduler 已完成的结果, 注入 prompt 头
          - 可选用 OmniStreamingSession (cfg.use_streaming_session) 复用 KV cache
          - 流式 Mimi 解码带 overlap + 端口 crossfade (P2b 修复颤抖)
        """
        history_full = self.timeline.derive_history(max_turns=self.cfg.max_history_turns + 1)
        history = history_full[:-1] if history_full and history_full[-1].get("role") == "user" else history_full

        # Stage 2: 把 background 结果作为系统级 context 前置注入
        bg_context = self._consume_bg_results()
        if bg_context:
            full_prompt = f"{bg_context}\n\n用户问: {prompt}"
            self.log(f"[fg] injecting {bg_context.count(chr(10))+1} bg result(s) into prompt")
        else:
            full_prompt = prompt

        self._set_state(SPEAKING)
        self._fg_stop_event.clear()
        t0 = time.time()
        ttft_t = ttft_a = None
        n_text = n_audio = 0
        audio_pending: List[List[int]] = []     # 本块还没解码的新帧
        audio_history: List[List[int]] = []     # 本 turn 已生成的全部帧 (用作 overlap 上下文)
        interrupted = False
        mode = "streaming" if self.cfg.use_streaming_session else "stateless"
        self.log(f"[fg] start[{mode}], prompt={full_prompt[:60]!r}, history_len={len(history)}")

        # Stage 3: 选择生成路径
        if self.cfg.use_streaming_session:
            gen_iter = self._streaming_generate(full_prompt, history)
        else:
            from mlx_omni.generate_omni import stream_generate_omni
            gen_iter = stream_generate_omni(
                self.model, self.tokenizer, full_prompt,
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
            )

        try:
            for text_seg, audio_frame in gen_iter:
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
                    audio_pending.append(audio_frame)
                    audio_history.append(audio_frame)
                    n_audio += 1
                    if len(audio_pending) >= self.cfg.audio_chunk_frames:
                        self._emit_pcm_chunked(audio_history, len(audio_pending))
                        audio_pending = []
            if not interrupted and audio_pending:
                self._emit_pcm_chunked(audio_history, len(audio_pending))
        except Exception as e:
            import traceback; traceback.print_exc()
            self.log(f"[fg] exception: {e}")
            interrupted = True

        dt = time.time() - t0
        self.log(f"[fg] done | t={n_text} a={n_audio} dt={dt:.2f}s interrupted={interrupted}")
        self.timeline.add(Event.now(GEN_DONE, interrupted=interrupted,
                                    n_text=n_text, n_audio=n_audio, dt=dt))
        self._send({"type": "done", "interrupted": interrupted})
        # 本 turn 结束, 重置 crossfade 起点 (下 turn 第一块不做 crossfade)
        self._first_audio_chunk_of_turn = True
        self._vad.generating = False
        self._set_state(LISTENING)

    def _emit_pcm_chunked(self, all_frames: List[List[int]], new_frame_count: int):
        """带 overlap + 边界 crossfade 的流式 Mimi 解码.

        - 解码窗 = 新增 new_frame_count 帧 + 前面 overlap 帧 (作为 codec 上下文)
        - 解码出的 PCM 把前面 overlap 那段样本扔掉, 只保留 "新增帧" 对应的样本
        - 把新 PCM 块开头一小段做线性 ramp-in, 抹平和上一块的拼接处
        """
        if not all_frames or new_frame_count <= 0:
            return
        overlap = min(self.cfg.audio_overlap, len(all_frames) - new_frame_count)
        decode_n = new_frame_count + overlap
        decode_frames = all_frames[-decode_n:]
        wav = self.mimi.decode(decode_frames)
        if wav is None:
            return
        # 按比例丢掉前面 overlap 帧的样本
        if overlap > 0:
            drop = int(overlap * len(wav) / decode_n)
            wav = wav[drop:]
        # 边界 crossfade: 新块开头做 ramp-in
        cfd = self.cfg.audio_crossfade_samples
        if cfd > 0 and not getattr(self, "_first_audio_chunk_of_turn", True) and len(wav) > cfd:
            ramp = np.linspace(0.0, 1.0, cfd, dtype=np.float32)
            wav[:cfd] = wav[:cfd] * ramp
        self._first_audio_chunk_of_turn = False
        pcm = (wav * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
        self.timeline.add(Event.now(
            MODEL_PCM,
            bytes=len(pcm), new_frames=new_frame_count, overlap=overlap,
        ))
        self._send({"type": "pcm", "data": base64.b64encode(pcm).decode()})

    # ====================================================================
    # Stage 3: 持久 KV streaming session
    # ====================================================================

    def _streaming_generate(self, prompt: str, history: list):
        """用 OmniStreamingSession 生成 (跨 turn 复用 KV cache)."""
        if self._streaming is None:
            from .streaming_session import OmniStreamingSession
            self._streaming = OmniStreamingSession(
                self.model, self.tokenizer, self.cfg,
                logger=self.log,
            )
        yield from self._streaming.generate_turn(
            prompt, history,
            max_new_tokens=self.cfg.foreground_max_new_tokens,
            temperature=self.cfg.foreground_temperature,
            top_p=self.cfg.foreground_top_p,
            top_k=self.cfg.foreground_top_k,
            audio_temperature=self.cfg.audio_temperature,
            audio_top_k=self.cfg.audio_top_k,
            audio_rep_penalty=self.cfg.audio_rep_penalty,
            audio_greedy=self.cfg.audio_greedy,
            stop_event=self._fg_stop_event,
        )
