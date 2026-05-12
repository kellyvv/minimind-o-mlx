"""后台任务调度器 (stage 2 hook).

当前仅占位, 真正落地时这里会跑:
  - 长回答 (用一个更深思考的 MLX/PyTorch 模型异步生成)
  - 搜索 / 工具调用
  - 图像分析
  - 当前上下文总结

每个 job 完成时往 InteractionSession.timeline 里 add 一个 BG_JOB_RESULT 事件,
foreground scheduler 据此决定是否插话 (例如: 用户停顿时插入 "我查到了...").

stage 1 不启用; foreground generate 始终走 stream_generate_omni 单源.
"""
from __future__ import annotations
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class BackgroundJob:
    job_id: str
    kind: str                           # "long_answer" / "tool_call" / "summary" / ...
    fn: Callable[[], Any]
    started: float = field(default_factory=time.monotonic)
    result: Any = None
    done: bool = False
    error: Optional[str] = None


class BackgroundScheduler:
    """非常薄的接口, 给 stage 2 实现留位置."""

    def __init__(self, on_result: Optional[Callable[[BackgroundJob], None]] = None):
        self._jobs: Dict[str, BackgroundJob] = {}
        self._lock = threading.Lock()
        self._on_result = on_result

    def submit(self, kind: str, fn: Callable[[], Any]) -> str:
        job = BackgroundJob(job_id=str(uuid.uuid4())[:8], kind=kind, fn=fn)
        with self._lock:
            self._jobs[job.job_id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True,
                         name=f"bg-{kind}-{job.job_id}").start()
        return job.job_id

    def _run(self, job: BackgroundJob):
        try:
            job.result = job.fn()
        except Exception as e:
            job.error = str(e)
        job.done = True
        if self._on_result is not None:
            try: self._on_result(job)
            except Exception: pass

    def cancel(self, job_id: str):
        # 当前不支持中途取消 (没有 stop_event 协议)，stage 2 实现时给每个 fn 接 stop_event
        pass

    def list_pending(self) -> List[BackgroundJob]:
        with self._lock:
            return [j for j in self._jobs.values() if not j.done]
