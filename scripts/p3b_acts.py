"""Phase 3b: residual-stream activations over the decision turn's deliberation text.

usage: python scripts/p3b_acts.py RUN_DIR STEP SUBDIR [SUBDIR ...]
  Reads RUN_DIR/SUBDIR/step_STEP.jsonl (from p2_eval_ckpts.py). For every rollout with a decision
  turn (rhjl.readout.deliberation) it runs one forward pass over the tokens up to the turn's
  action token and keeps the hidden states at turn_start, every token of the span, and action.

  READER=policy (default) uses the checkpoint (LoRA merged into the base). READER=base uses the
  base model on the same tokens: the "text reader", which sees the policy's text but none of its
  weights, so a probe on it measures what is recoverable from the text alone.

  Runs in fp32. In bf16 the activation at a position shifts by ~3% with the number of tokens
  after it in the same pass (Phase 3), and span positions have different numbers of following
  tokens in hack and non-hack rollouts. In fp32 the shift is ~0.06%.

Writes runs/p3b/acts/{run}_{step}_{SUBDIR}_{READER}/:
  L{l}.npy     float16 [T, 2048] for l in LAYERS: row j is token j of the concatenated readouts
  index.jsonl  one line per kept rollout: i (line in the step file), label, problem, offset/length
               of its rows (row offset = turn_start, offset+1 .. offset+length-1 = span tokens,
               offset+length = action), action relative to turn_start, and the
               deliberation() fields with positions made relative to turn_start
"""
import json
import os
import sys
from pathlib import Path

LAYERS = (4, 8, 12, 14, 16, 18, 20, 22, 24)


def main():
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rhjl.readout import Specials, deliberation

    run, step, subdirs = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
    reader = os.environ.get("READER", "policy")
    base = "/home/wil/models/qwen3.5-2b-text"
    tok = AutoTokenizer.from_pretrained(base)
    sp = Specials(tok)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).to("cuda")
    if reader == "policy" and step:
        model = PeftModel.from_pretrained(model, str(run / f"checkpoint-{step}")).merge_and_unload()
    model.eval()

    for sub in subdirs:
        with open(run / sub / f"step_{step}.jsonl") as f:
            rs = [json.loads(line) for line in f][: int(os.environ.get("LIMIT", "0")) or None]
        out = Path(f"runs/p3b/acts/{run.name}_{step}_{sub}_{reader}")
        out.mkdir(parents=True, exist_ok=True)
        rows = {l: [] for l in LAYERS}
        index, offset = [], 0
        for i, r in enumerate(rs):
            d = deliberation(r, sp)
            if d is None:
                continue
            full = r["prompt_ids"] + r["completion_ids"]
            keep = [d["turn_start"]] + d["span"] + [d["action"]]
            with torch.no_grad():
                hs = model(torch.tensor([full[: d["action"] + 1]], device="cuda"),
                           output_hidden_states=True).hidden_states
            for l in LAYERS:
                rows[l].append(hs[l][0, keep].half().cpu().numpy())
            t0 = d["turn_start"]
            index.append({"i": i, "nums": r["nums"], "target": r["target"], "label": r["label"],
                          "read_forbidden": r["read_forbidden"], "attempted_forbidden": r["attempted_forbidden"],
                          "offset": offset, "length": len(keep) - 1, "action": d["action"] - t0,
                          "mention": None if d["mention"] is None else d["mention"] - t0,
                          **{k: d[k] for k in ("hack_in_turn", "saw_rule", "rule_turn", "decision_turn", "first_call",
                                                 "prior_calls")}})
            offset += len(keep)  # rows offset .. offset+length-1: turn_start + span; offset+length: action
            if len(index) % 500 == 0:
                print(sub, i, "/", len(rs), flush=True)
        for l in LAYERS:
            np.save(out / f"L{l}.npy", np.concatenate(rows[l]))
        with open(out / "index.jsonl", "w") as f:
            f.writelines(json.dumps(m) + "\n" for m in index)
        print("wrote", out, len(index), "rollouts", offset, "rows", flush=True)


if __name__ == "__main__":
    main()
