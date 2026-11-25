"""Phase 1: base-model hack rate and honest accuracy for each loophole hint variant.

Samples multi-turn tool rollouts with the same token layout as TRL's GRPO loop (rhjl.rollout) on
GRAFT's eval chunk, labels each one programmatically, and writes
runs/p1_baserate/{variant}.jsonl plus runs/p1_baserate/summary.json.

env: N_PROB (64), K (8), MAX_COMP (1536), VARIANTS (comma list, default all), MODEL,
     OUT (runs/p1_baserate), MIN_WORDS (0, deliberation gate; see rhjl.hackenv.label).
"""
import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from rhjl.countdown import DATASET
    from rhjl.hackenv import VARIANTS
    from rhjl.rollout import rollouts

    model = os.environ.get("MODEL", "/home/wil/models/qwen3.5-2b-text")
    n_prob, k = int(os.environ.get("N_PROB", "64")), int(os.environ.get("K", "8"))
    max_comp = int(os.environ.get("MAX_COMP", "1536"))
    variants = os.environ.get("VARIANTS", ",".join(VARIANTS)).split(",")
    out = Path(os.environ.get("OUT", "runs/p1_baserate"))
    min_words = int(os.environ.get("MIN_WORDS", "0"))
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(DATASET, split="train").shuffle(seed=42).select(range(16000, 16000 + n_prob))
    rows = [{"nums": r["nums"], "target": r["target"]} for r in ds]
    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(model, gpu_memory_utilization=0.8, max_model_len=max_comp + 1024, enable_prefix_caching=True,
              max_num_seqs=128)
    sp = SamplingParams(temperature=1.0, top_p=1.0)

    summary = {"model": model, "n_prob": n_prob, "k": k, "max_comp": max_comp, "variants": {}}
    for v in variants:
        t0 = time.time()
        rs = rollouts(llm, tok, rows, v, k, sp, max_comp, min_words=min_words)
        labs = [r.lab for r in rs]
        n = len(rs)
        c = Counter(lab["label"] for lab in labs)
        # tokens the model wrote before the first tool call, for hack rollouts (how much "reasoning" precedes the hack)
        pre = [len(r.messages[0].get("content", "")) for r in rs if r.lab["read_forbidden"]]
        per_prob = Counter()
        for i, r in enumerate(rs):
            per_prob[i // k] += r.lab["read_forbidden"]
        stats = {
            "hack": c["hack"] / n, "clean": c["clean"] / n, "fail": c["fail"] / n,
            "reward": sum(lab["reward"] for lab in labs) / n,
            "hack_rewarded": sum(lab["read_forbidden"] and lab["correct"] for lab in labs) / max(1, c["hack"]),
            "attempted_forbidden": sum(lab["attempted_forbidden"] for lab in labs) / n,
            "used_tools": sum(lab["n_tool_calls"] > 0 for lab in labs) / n,
            "benign_read": sum(any(f != "answers.txt" for f in lab["reads"]) for lab in labs) / n,
            "read_readme": sum("README.md" in lab["reads"] for lab in labs) / n,
            "listed_files": sum(lab["listed_files"] for lab in labs) / n,
            "failed_calls_per_rollout": sum(lab["n_failed_calls"] for lab in labs) / n,
            "truncated": sum(r.truncated for r in rs) / n,
            "deliberated": sum(lab["deliberated"] for lab in labs) / n,
            "mean_model_tokens": statistics.mean(sum(r.tool_mask) for r in rs),
            "problems_with_any_hack": sum(x > 0 for x in per_prob.values()) / n_prob,
            "median_chars_before_hack_call": statistics.median(pre) if pre else None,
            "seconds": round(time.time() - t0, 1),
        }
        summary["variants"][v] = stats
        print(v, json.dumps({kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in stats.items()}),
              flush=True)
        with open(out / f"{v}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps({"nums": r.row["nums"], "target": r.row["target"], "messages": r.messages,
                                    "truncated": r.truncated, "n_model_tokens": sum(r.tool_mask), **r.lab},
                                   default=str) + "\n")
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
