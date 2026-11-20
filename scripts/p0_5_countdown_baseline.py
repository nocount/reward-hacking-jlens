"""Phase 0.5: baseline Countdown accuracy for Qwen3.5-2B, thinking on (and off for reference).

Samples K completions per problem with vLLM at RL-style sampling (T=1, top_p=1) and a long budget,
then reports accuracy as if completions were capped at shorter budgets.
"""
import json, os, sys, time
import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams
from rhjl.countdown import DATASET, build_messages, score

def main():
    N_PROB, K, MAX_TOK = int(os.environ.get("N_PROB", 200)), int(os.environ.get("K", 4)), int(os.environ.get("MAX_TOK", 6144))
    ds = load_dataset(DATASET, split="train").shuffle(seed=42).select(range(16000, 16000 + N_PROB))  # GRAFT's eval chunk
    llm = LLM("Qwen/Qwen3.5-2B", dtype="bfloat16", max_model_len=MAX_TOK + 512, gpu_memory_utilization=0.85,
              limit_mm_per_prompt={"image": 0, "video": 0}, enable_prefix_caching=True, seed=0)
    tok = llm.get_tokenizer()
    results = {}
    configs = {  # name: (enable_thinking, sampling kwargs, max_tokens)
        "think_T1": (True, dict(temperature=1.0, top_p=1.0), MAX_TOK),  # what GRPO would sample
        "think_qwen_rec": (True, dict(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5), MAX_TOK),
        "nothink_T1": (False, dict(temperature=1.0, top_p=1.0), 2048),
    }
    for mode in os.environ.get("MODES", ",".join(configs)).split(","):
        thinking, skw, max_tok = configs[mode]
        prompts = [tok.apply_chat_template(build_messages(r["nums"], r["target"]), tokenize=False,
                                           add_generation_prompt=True, enable_thinking=thinking) for r in ds]
        sp = SamplingParams(n=K, max_tokens=max_tok, seed=0, **skw)
        t = time.time()
        outs = llm.generate(prompts, sp)
        dt = time.time() - t
        rows = []
        for r, o in zip(ds, outs):
            for c in o.outputs:
                s = score(c.text, r["nums"], r["target"], thinking=thinking)
                rows.append({"nums": r["nums"], "target": r["target"], "n_tok": len(c.token_ids),
                             "finished": c.finish_reason == "stop", **s, "text": c.text})
        n_tok = np.array([x["n_tok"] for x in rows]); corr = np.array([x["correct"] for x in rows])
        per_prob = corr.reshape(N_PROB, K)
        res = {"sampling": skw, "thinking": thinking, "n_problems": N_PROB, "k": K, "max_tokens": sp.max_tokens, "gen_seconds": round(dt, 1),
               "gen_tok_per_s": round(n_tok.sum() / dt), "acc": round(corr.mean(), 4),
               "pass_at_k": round(per_prob.any(1).mean(), 4),
               "frac_problems_mixed": round(((per_prob.sum(1) > 0) & (per_prob.sum(1) < K)).mean(), 4),
               "frac_truncated": round(1 - np.mean([x["finished"] for x in rows]), 4),
               "frac_formatted": round(np.mean([x["formatted"] for x in rows]), 4),
               "frac_hits_target_wrong_nums": round(np.mean([x["hits_target"] and not x["uses_nums"] for x in rows]), 4),
               "tokens_p50_p90_max": [int(np.percentile(n_tok, 50)), int(np.percentile(n_tok, 90)), int(n_tok.max())],
               "acc_if_capped": {cap: round((corr & (n_tok <= cap)).mean(), 4)
                                 for cap in (256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192) if cap <= sp.max_tokens}}
        results[mode] = res
        print(mode, json.dumps(res, indent=1), flush=True)
        with open(f"runs/p0_5_rollouts_{mode}.jsonl", "w") as f:
            for x in rows:
                f.write(json.dumps(x) + "\n")
    with open("runs/p0_5_baseline.json", "w") as f:
        json.dump(results, f, indent=1)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
