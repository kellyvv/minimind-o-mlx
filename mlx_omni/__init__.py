"""MLX 移植版 MiniMind-O.

阶段 1 当前状态：
  - Thinker (LM) 已移植，支持 fp16 / fp32 / 量化 (4-bit / 8-bit)
  - 权重转换器: convert_pt_to_mlx
  - 文本流式生成: generate_stream
"""
from .model import OmniConfig, MiniMindModel, MiniMindForCausalLM
from .generate import generate_stream, generate
