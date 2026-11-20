"""Pull pretraining-like prompts (FineWeb sample) for J-lens fitting; keep docs >= 160 tokens."""
import json, sys
from datasets import load_dataset
from transformers import AutoTokenizer

n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
out = []
for row in ds.shuffle(seed=0, buffer_size=10_000):
    if len(tok(row["text"][:4000]).input_ids) >= 160:
        out.append({"id": row["id"], "text": row["text"][:4000]})
    if len(out) >= n:
        break
with open("data/fit_prompts_fineweb.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print("wrote", len(out))
