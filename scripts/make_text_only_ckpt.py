"""Save a text-only Qwen3.5-2B (Qwen3_5ForCausalLM, no vision tower) for TRL + vLLM.

vLLM otherwise profiles the vision encoder with dummy images (OOM in colocate mode), and
PEFT all-linear would wrap the vision linears.
"""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

src = "Qwen/Qwen3.5-2B"
dst = sys.argv[1] if len(sys.argv) > 1 else "/home/wil/models/qwen3.5-2b-text"
m = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(src)
print(type(m).__name__, m.config.model_type, m.config.architectures)
m.save_pretrained(dst)
tok.save_pretrained(dst)
print("saved", dst)
