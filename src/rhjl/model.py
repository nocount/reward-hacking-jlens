"""Model loading helpers for Qwen3.5 (text-only)."""
from __future__ import annotations

import torch
import transformers

DEFAULT_MODEL = "Qwen/Qwen3.5-2B"


def load_text_model(name: str = DEFAULT_MODEL, dtype=torch.bfloat16, device: str = "cuda"):
    """Load the text decoder of a Qwen3.5 checkpoint (vision tower is skipped)."""
    tok = transformers.AutoTokenizer.from_pretrained(name)
    model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device)
    model.eval()
    return model, tok
