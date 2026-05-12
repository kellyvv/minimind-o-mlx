"""事件类型常量 + Event 数据类.

参考 Thinking Machines "interaction models" 论文中的"持续事件流"思路：
所有来源（用户音频、VAD 状态、模型 token、后台任务结果）统一变成时间戳事件，
foreground scheduler 根据事件做状态切换 + 触发动作。
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Dict


# ---- input side: 用户产生 ----
AUDIO_CHUNK   = "audio_chunk"     # 用户音频原始 PCM 片段
TEXT_DELTA    = "text_delta"      # 用户键入的文本片段 (stage 1 暂用整段)
IMAGE_FRAME   = "image_frame"     # 视频帧或单张图片 (stage 3+ 才会用)
CONTROL_MSG   = "control_msg"     # context/stop/end 等控制消息

# ---- VAD / 转录派生事件 ----
SPEECH_START  = "speech_start"    # VAD 第一次判定用户在说话
SPEECH_END    = "speech_end"      # VAD 判定用户说完
BARGE_IN      = "barge_in"        # 模型 speaking 时用户又开口 (产生打断)
ASR_RESULT    = "asr_result"      # SenseVoice ASR 转录结果

# ---- output side: 模型产生 ----
MODEL_TEXT    = "model_text"      # 模型输出的文本片段
MODEL_AUDIO   = "model_audio"     # 模型输出的 8-codebook frame
MODEL_PCM     = "model_pcm"       # Mimi 解码后的 PCM (24kHz int16 bytes)
STATUS        = "status"          # 状态变化 (listening / thinking / speaking / interrupted)
TTFT          = "ttft"            # 首字 / 首音延迟统计
GEN_DONE      = "gen_done"        # 一次 foreground generate 结束 (含 interrupted=True/False)

# ---- 后台任务 (stage 2) ----
BG_JOB_START  = "bg_job_start"    # 后台任务启动
BG_JOB_RESULT = "bg_job_result"   # 后台任务结果回填


@dataclass
class Event:
    """单个时间轴事件."""
    t: float                       # time.monotonic() 时间戳, 秒
    type: str                      # 事件类型 (上方常量)
    payload: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def now(cls, type: str, **payload) -> "Event":
        return cls(t=time.monotonic(), type=type, payload=payload)

    def __repr__(self) -> str:
        keys = ",".join(self.payload.keys())
        return f"Event(t={self.t:.3f} {self.type}[{keys}])"
