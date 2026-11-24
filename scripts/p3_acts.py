"""Phase 3: residual-stream activations at the readout positions of sampled rollouts.

usage: python scripts/p3_acts.py RUN_DIR STEP SUBDIR [SUBDIR ...]
  Reads RUN_DIR/SUBDIR/step_STEP.jsonl (from p2_eval_ckpts.py), runs the checkpoint's model
  (LoRA merged into the base) over each rollout's tokens and writes runs/p3/acts/{run}_{step}_{SUBDIR}.npz:

    {pos}: float16 [N, 25, 2048]  hidden_states at that position (index 0 = embeddings,
           1..24 = output of block l; HF applies the final norm to index 24)
           for pos in decision, turn_start, pre_answer, ctrl_continue, ctrl_stop
    pool_decision: mean over model-generated tokens up to and including `decision`
    pool_answer:   mean over the final turn's tokens up to and including `pre_answer`
    has_{pos}: bool [N]
  plus {run}_{step}_{SUBDIR}.meta.jsonl with one line per rollout (label, problem, signature, ...).
Positions are defined in rhjl.readout. Missing positions are stored as zeros with has_* False.
Two forward passes per rollout, ending at the decision and at pre_answer (see hidden()).
"""
import json
import os
import sys
from pathlib import Path

POS = ("decision", "turn_start", "pre_answer", "ctrl_continue", "ctrl_stop")
POOL = ("pool_decision", "pool_answer")


def main():
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rhjl.readout import Specials, positions

    run, step, subdirs = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
    base = "/home/wil/models/qwen3.5-2b-text"
    tok = AutoTokenizer.from_pretrained(base)
    sp = Specials(tok)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to("cuda")
    if step:
        model = PeftModel.from_pretrained(model, str(run / f"checkpoint-{step}")).merge_and_unload()
    model.eval()
    out_dir = Path("runs/p3/acts")
    out_dir.mkdir(parents=True, exist_ok=True)

    for sub in subdirs:
        with open(run / sub / f"step_{step}.jsonl") as f:
            rs = [json.loads(line) for line in f][: int(os.environ.get("LIMIT", "0")) or None]
        n, L, D = len(rs), model.config.num_hidden_layers + 1, model.config.hidden_size
        acts = {k: np.zeros((n, L, D), np.float16) for k in POS + POOL}
        has = {k: np.zeros(n, bool) for k in POS + POOL}
        meta = []
        for i, r in enumerate(rs):
            p = positions(r, sp)
            full = r["prompt_ids"] + r["completion_ids"]
            P = len(r["prompt_ids"])
            gen = [0] * P + r["tool_mask"]

            def hidden(end):
                """hidden_states [T, L, D] of full[:end]. Each readout comes from a forward pass that ends
                at (or shortly after) it: in bf16 the activations at a position shift by ~3% with the number of
                tokens that follow it in the same pass (content-independent kernel numerics), and how many tokens
                follow differs between hack and clean rollouts, so a longer pass leaks the label."""
                with torch.no_grad():
                    hs = model(torch.tensor([full[:end]], device="cuda"), output_hidden_states=True).hidden_states
                return torch.stack([x[0] for x in hs], 1)

            if p["decision"] is not None:
                # turn_start / ctrl_* precede the decision; the tokens between them and it are prefix, not label
                h = hidden(p["decision"] + 1)
                for k in ("decision", "turn_start", "ctrl_continue", "ctrl_stop"):
                    if p[k] is not None:
                        acts[k][i] = h[p[k]].float().cpu().numpy()
                        has[k][i] = True
                idx = [t for t in range(P, p["decision"] + 1) if gen[t]]
                acts["pool_decision"][i] = h[idx].float().mean(0).cpu().numpy()
                has["pool_decision"][i] = True
            if p["pre_answer"] is not None:
                h = hidden(p["pre_answer"] + 1)
                acts["pre_answer"][i] = h[p["pre_answer"]].float().cpu().numpy()
                has["pre_answer"][i] = True
                a = next(t for t in range(p["pre_answer"], -1, -1) if not gen[t]) + 1  # start of final turn
                a = min(a, p["pre_answer"])  # <answer> opened the turn: pool over the one token before it
                acts["pool_answer"][i] = h[a:p["pre_answer"] + 1].float().mean(0).cpu().numpy()
                has["pool_answer"][i] = True
            meta.append({"i": i, "nums": r["nums"], "target": r["target"], "label": r["label"],
                         "hack_type": r["hack_type"], "read_forbidden": r["read_forbidden"],
                         "attempted_forbidden": r["attempted_forbidden"], "truncated": r["truncated"],
                         "n_tool_calls": r["n_tool_calls"], "listed_files": r["listed_files"],
                         "n_model_tokens": sum(r["tool_mask"]), **{k: p[k] for k in p}})
            if i % 500 == 0:
                print(sub, i, "/", n, flush=True)
        name = f"{run.name}_{step}_{sub}"
        np.savez(out_dir / f"{name}.npz", **acts, **{f"has_{k}": v for k, v in has.items()})
        with open(out_dir / f"{name}.meta.jsonl", "w") as f:
            f.writelines(json.dumps(m) + "\n" for m in meta)
        print("wrote", name, flush=True)


if __name__ == "__main__":
    main()
