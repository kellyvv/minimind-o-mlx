"""后台任务调度器 (stage 2 实现).

让 InteractionSession 在 foreground 持续生成的同时, 异步跑可能耗时的副任务:
  - 工具调用 (web 搜索 / API 查询 / 文件检索)
  - 长篇深度回答 (跑一次纯文本 generate, 不要 audio)
  - 上下文总结
  - 图像分析

设计要点:
  - 每个 job 在自己的线程跑
  - 完成后通过 on_event(BG_JOB_RESULT) 把结果回填 timeline
  - foreground 调度器 (在 InteractionSession 里) 在合适时机 (用户停顿 / 转折点) 决定是否插入
  - 用户可以 cancel 在跑的 job (传入 job_id)

线程安全说明:
  MLX 调用必须在主线程跑 (Metal context 限制)。所以 background 里的 fn
  应当避免直接调 MLX, 优先用 PyTorch CPU / HTTP / 文件 IO。
  如果实在需要 MLX, 用 main-thread proxy (submit_main_callable, 暂未实现)。
"""
from __future__ import annotations
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class BackgroundJob:
    job_id: str
    kind: str
    fn: Callable[[], Any]
    meta: Dict[str, Any] = field(default_factory=dict)
    started: float = field(default_factory=time.monotonic)
    finished: Optional[float] = None
    result: Any = None
    done: bool = False
    error: Optional[str] = None
    consumed: bool = False                # foreground 是否已消费这个结果

    @property
    def duration(self) -> float:
        end = self.finished or time.monotonic()
        return end - self.started


class BackgroundScheduler:
    """非常薄的 job 池. 完成事件用回调暴露给上层."""

    def __init__(self, on_event: Optional[Callable[[str, BackgroundJob], None]] = None):
        """
        Args:
            on_event: 回调签名 fn(event_type, job)
                event_type ∈ {"start", "result"}
        """
        self._jobs: Dict[str, BackgroundJob] = {}
        self._lock = threading.Lock()
        self._on_event = on_event

    # ---- 提交 / 取消 ----

    def submit(self, kind: str, fn: Callable[[], Any], **meta) -> str:
        job = BackgroundJob(job_id=str(uuid.uuid4())[:8], kind=kind, fn=fn, meta=dict(meta))
        with self._lock:
            self._jobs[job.job_id] = job
        if self._on_event:
            try: self._on_event("start", job)
            except Exception: pass
        threading.Thread(target=self._run, args=(job,), daemon=True,
                         name=f"bg-{kind}-{job.job_id}").start()
        return job.job_id

    def cancel(self, job_id: str):
        """当前不支持中途打断 fn (没有协议). 实际用法: 标记 done+error 让 consumer 跳过."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job and not job.done:
                job.done = True
                job.error = "cancelled"
                job.finished = time.monotonic()

    # ---- 内部 ----

    def _run(self, job: BackgroundJob):
        try:
            job.result = job.fn()
        except Exception as e:
            job.error = f"{type(e).__name__}: {e}"
            job.error_tb = traceback.format_exc()
        job.done = True
        job.finished = time.monotonic()
        if self._on_event:
            try: self._on_event("result", job)
            except Exception: pass

    # ---- 查询 ----

    def list_pending(self) -> List[BackgroundJob]:
        with self._lock:
            return [j for j in self._jobs.values() if not j.done]

    def list_unconsumed_results(self) -> List[BackgroundJob]:
        """返回已完成但 foreground 还没消费的 job. 调用方应该在消费后调 mark_consumed."""
        with self._lock:
            return [j for j in self._jobs.values() if j.done and not j.consumed]

    def mark_consumed(self, job_id: str):
        with self._lock:
            j = self._jobs.get(job_id)
            if j:
                j.consumed = True

    def get(self, job_id: str) -> Optional[BackgroundJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def __len__(self):
        with self._lock:
            return len(self._jobs)


# ============================================================================
# 内置 demo job 实现
# ============================================================================


def make_echo_job(text: str, delay_s: float = 2.0) -> Callable[[], str]:
    """演示用: sleep n 秒然后回 echo 字符串. 验证 scheduler 通路."""
    def fn() -> str:
        time.sleep(delay_s)
        return f"echo({delay_s}s): {text}"
    return fn


def make_summarize_history_job(timeline) -> Callable[[], str]:
    """简单总结: 把 timeline 上的最近几个 user/assistant 文本 join 起来.

    真实实现里这里应该跑一次纯文本 LM 生成 (text-only thinker), 当前为了避免
    跨线程 MLX 调用, 仅做规则式拼接, 留作 stage-2-future 升级点。
    """
    def fn() -> str:
        msgs = timeline.derive_history(max_turns=8)
        if not msgs:
            return "(没有可总结的历史)"
        lines = [f"{m['role']}: {m['content'][:60]}" for m in msgs[-6:]]
        return "近期对话摘要: " + "; ".join(lines)
    return fn


def make_current_time_job() -> Callable[[], str]:
    """工具调用 demo: 返回当前北京时间."""
    def fn() -> str:
        from datetime import datetime, timezone, timedelta
        beijing = datetime.now(timezone(timedelta(hours=8)))
        return f"当前北京时间: {beijing.strftime('%Y-%m-%d %H:%M:%S')}"
    return fn
