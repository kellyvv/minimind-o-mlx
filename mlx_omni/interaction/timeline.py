"""线程安全的时间轴 (event log).

不保留全部历史 (deque maxlen)，但保留足够多事件让 scheduler / history 重建.
所有 InteractionSession 子线程共写一份, foreground_loop 读取。
"""
from __future__ import annotations
import threading
from collections import deque
from typing import Callable, Iterable, List, Optional

from .events import Event


class Timeline:
    def __init__(self, max_events: int = 2000):
        self._events: deque = deque(maxlen=max_events)
        self._lock = threading.Lock()
        self._subscribers: List[Callable[[Event], None]] = []

    # ---- 写入 ----

    def add(self, event: Event) -> Event:
        with self._lock:
            self._events.append(event)
            subs = list(self._subscribers)
        # 不在锁内回调，避免死锁
        for s in subs:
            try: s(event)
            except Exception: pass
        return event

    # ---- 订阅 (stage 2 scheduler 会用) ----

    def subscribe(self, fn: Callable[[Event], None]):
        with self._lock:
            self._subscribers.append(fn)

    # ---- 查询 ----

    def recent(self, n: int = 50) -> List[Event]:
        with self._lock:
            return list(self._events)[-n:]

    def since(self, t: float) -> List[Event]:
        with self._lock:
            return [e for e in self._events if e.t >= t]

    def filter(self, types: Iterable[str], n: Optional[int] = None) -> List[Event]:
        types_set = set(types)
        with self._lock:
            out = [e for e in self._events if e.type in types_set]
        return out[-n:] if n else out

    def last_of(self, type: str) -> Optional[Event]:
        with self._lock:
            for e in reversed(self._events):
                if e.type == type:
                    return e
        return None

    # ---- 派生: 从事件流重建会话历史 ----

    def derive_history(self, max_turns: int = 4) -> List[dict]:
        """从 timeline 反推出 [{role,content}, ...] 列表 (供 next-turn prompt).

        当前规则:
          - 用户每次 ASR_RESULT 算一轮 user message
          - 模型每次 GEN_DONE 之间的 MODEL_TEXT 串拼成一条 assistant message
        被 BARGE_IN 打断的 assistant 截到 interrupted 时刻为止.
        """
        msgs: List[dict] = []
        cur_assistant: List[str] = []
        with self._lock:
            for e in self._events:
                if e.type == "asr_result":
                    # flush 当前 assistant
                    if cur_assistant:
                        msgs.append({"role": "assistant", "content": "".join(cur_assistant)})
                        cur_assistant = []
                    txt = e.payload.get("text", "")
                    if txt:
                        msgs.append({"role": "user", "content": txt})
                elif e.type == "model_text":
                    cur_assistant.append(e.payload.get("content", ""))
                elif e.type == "gen_done":
                    if cur_assistant:
                        msgs.append({"role": "assistant", "content": "".join(cur_assistant)})
                        cur_assistant = []
        return msgs[-(max_turns * 2):]
