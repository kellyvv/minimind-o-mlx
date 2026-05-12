"""Interaction-model 层 (stage 1 工程改造, stage 2+ 模型层).

参考: Thinking Machines "Interaction Models" (2025).
"""
from .events import (
    Event,
    AUDIO_CHUNK, TEXT_DELTA, IMAGE_FRAME, CONTROL_MSG,
    SPEECH_START, SPEECH_END, BARGE_IN, ASR_RESULT,
    MODEL_TEXT, MODEL_AUDIO, MODEL_PCM, STATUS, TTFT, GEN_DONE,
    BG_JOB_START, BG_JOB_RESULT,
)
from .timeline import Timeline
from .session import InteractionSession, SessionConfig
from .background import BackgroundScheduler, BackgroundJob
from .streaming_session import OmniStreamingSession
