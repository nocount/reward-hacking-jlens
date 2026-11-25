"""Phase 2: hack rate of saved LoRA checkpoints on held-out problems.

One in-process vLLM engine (created like TRL's colocate mode) serves every checkpoint: for each
one, the adapter is merged into an HF copy of the base model and the merged weights are loaded
into vLLM, exactly as TRL syncs weights during training. Rollouts use rhjl.rollout, so token
layout and labels match training.

usage: python scripts/p2_eval_ckpts.py RUN_DIR [STEP ...]
  With no steps, evaluates step 0 (base) and every EVERY-th checkpoint whose training-batch hack
  rate (smoothed over 5 steps) lies in [LO, HI], plus the first and last checkpoint.

env: N_PROB (64)  OFFSET (16000: first problem index in the shuffled dataset; RL used 32000+)
     K (8)  MAX_COMP (1536)  EVERY (2)  LO (0.05)  HI (0.95)  SUBDIR (eval: output dir under RUN_DIR)
     GRADER / LOOPHOLE_FILE / VARIANT / MIN_WORDS: read from RUN_DIR/summary.json unless set
Writes RUN_DIR/SUBDIR/step_{N}.jsonl and RUN_DIR/SUBDIR/summary.json (merged with earlier results).
"""
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

for k, v in {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1", "MASTER_ADDR": "127.0.0.1",
             "MASTER_PORT": "29517"}.items():
    os.environ.setdefault(k, v)


def pick_steps(run: Path, every: int, lo: float, hi: float) -> list[int]:
    per = defaultdict(Counter)
    for line in open(run / "rollouts.jsonl"):
        r = json.loads(line)
        per[r["step"]]["n"] += 1
        per[r["step"]]["hack"] += r["label"] == "hack"
    ckpts = sorted(int(p.name.split("-")[1]) for p in run.glob("checkpoint-*"))
    # checkpoint-N holds the weights after N updates; the rollouts that produced update N were logged at step N-1
    rate = {}
    for n in ckpts:
        window = [s for s in range(n - 5, n) if s in per]
        rate[n] = sum(per[s]["hack"] for s in window) / max(1, sum(per[s]["n"] for s in window))
    mid = [n for n in ckpts if lo <= rate[n] <= hi]
    chosen = set(mid[::every]) | ({mid[0], mid[-1]} if mid else set()) | {ckpts[0], ckpts[-1]}
    return [0] + sorted(chosen)


def main():
    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams

    from rhjl.countdown import DATASET
    from rhjl.rollout import rollouts

    run = Path(sys.argv[1])
    meta = json.load(open(run / "summary.json")) if (run / "summary.json").exists() else {}
    grader = os.environ.get("GRADER", meta.get("grader", "strict"))
    answers_file = os.environ.get("LOOPHOLE_FILE", "1" if meta.get("answers_file", True) else "0") == "1"
    variant = os.environ.get("VARIANT", meta.get("variant", "readme"))
    min_words = int(os.environ.get("MIN_WORDS", meta.get("min_words", 0)))
    n_prob, k = int(os.environ.get("N_PROB", "64")), int(os.environ.get("K", "8"))
    offset = int(os.environ.get("OFFSET", "16000"))
    max_comp = int(os.environ.get("MAX_COMP", "1536"))
    steps = [int(s) for s in sys.argv[2:]] or pick_steps(run, int(os.environ.get("EVERY", "2")),
                                                          float(os.environ.get("LO", "0.05")),
                                                          float(os.environ.get("HI", "0.95")))
    base = "/home/wil/models/qwen3.5-2b-text"
    out = run / os.environ.get("SUBDIR", "eval")
    out.mkdir(exist_ok=True)
    print("evaluating steps", steps, "grader", grader, "answers_file", answers_file, flush=True)

    ds = load_dataset(DATASET, split="train").shuffle(seed=42).select(range(offset, offset + n_prob))
    rows = [{"nums": r["nums"], "target": r["target"], "answers_file": answers_file} for r in ds]
    tok = AutoTokenizer.from_pretrained(base)
    llm = LLM(base, gpu_memory_utilization=0.55, max_model_len=max_comp + 1024, enable_prefix_caching=True,
              max_num_seqs=128, distributed_executor_backend="external_launcher", seed=0)
    vmodel = llm.llm_engine.model_executor.driver_worker.model_runner.model
    hf = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to("cuda")
    sp = SamplingParams(temperature=1.0, top_p=1.0)

    summary_path = out / "summary.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {}
    peft_model = None
    prev_adapter = None  # PEFT can't delete the active adapter, so drop the previous one instead
    for step in steps:
        t0 = time.time()
        if step == 0:
            src = hf
        else:
            ckpt = str(run / f"checkpoint-{step}")
            if peft_model is None:
                peft_model = PeftModel.from_pretrained(hf, ckpt, adapter_name=f"s{step}")
            else:
                peft_model.load_adapter(ckpt, adapter_name=f"s{step}")
            peft_model.set_adapter(f"s{step}")
            peft_model.merge_adapter()
            src = peft_model
        with torch.no_grad():
            for name, p in src.named_parameters():
                name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                if "lora_" in name:
                    continue
                vmodel.load_weights([(name, p.data)])
        if step != 0:
            peft_model.unmerge_adapter()
            if prev_adapter is not None:
                peft_model.delete_adapter(prev_adapter)
            prev_adapter = f"s{step}"
        llm.reset_prefix_cache()

        rs = rollouts(llm, tok, rows, variant, k, sp, max_comp, grader=grader, min_words=min_words)
        labs = [r.lab for r in rs]
        n = len(labs)
        per_prob = defaultdict(Counter)
        for i, lab in enumerate(labs):
            per_prob[i // k][lab["label"]] += 1
        stats = {
            "hack": sum(lab["label"] == "hack" for lab in labs) / n,
            "clean": sum(lab["label"] == "clean" for lab in labs) / n,
            "fail": sum(lab["label"] == "fail" for lab in labs) / n,
            "reward": sum(lab["reward"] for lab in labs) / n,
            "used_tools": sum(lab["n_tool_calls"] > 0 for lab in labs) / n,
            "truncated": sum(r.truncated for r in rs) / n,
            "deliberated": sum(lab["deliberated"] for lab in labs) / n,
            "problems_with_hack_and_clean": sum(c["hack"] > 0 and c["clean"] > 0 for c in per_prob.values()) / n_prob,
            "n": n, "seconds": round(time.time() - t0, 1),
        }
        summary[str(step)] = stats
        print(f"step {step}", json.dumps({a: round(b, 3) if isinstance(b, float) else b for a, b in stats.items()}),
              flush=True)
        with open(out / f"step_{step}.jsonl", "w") as f:
            f.writelines(json.dumps({"nums": r.row["nums"], "target": r.row["target"], "messages": r.messages,
                                    "truncated": r.truncated, "prompt_ids": r.prompt_ids,
                                    "completion_ids": r.completion_ids, "tool_mask": r.tool_mask, **r.lab},
                                   default=str) + "\n" for r in rs)
        with open(summary_path, "w") as f:
            json.dump(dict(sorted(summary.items(), key=lambda kv: int(kv[0]))), f, indent=1)


if __name__ == "__main__":
    main()
