"""Phase 0.3: fit a Jacobian lens on Qwen3.5-2B with jlens.fit.

usage: p0_3_fit_jlens.py bench            # time dim_batch options on 2 prompts
       p0_3_fit_jlens.py fit N DIM_BATCH  # fit on the first N prompts
env: MODEL (default the base Qwen3.5-2B; Phase 4 passes a merged RL checkpoint), NAME (output stem,
     default qwen35_2b) -> runs/jlens/{NAME}_n{N}.pt
"""
import json, os, sys, time
import torch, jlens
from rhjl.model import load_text_model

prompts = [json.loads(l)["text"] for l in open("data/fit_prompts_fineweb.jsonl")]
hf, tok = load_text_model(os.environ.get("MODEL", "Qwen/Qwen3.5-2B"))
name = os.environ.get("NAME", "qwen35_2b")
model = jlens.from_hf(hf, tok)
print(model, "bos:", tok.bos_token, flush=True)
mode = sys.argv[1]
if mode == "bench":
    jlens.jacobian_for_prompt(model, prompts[0], list(range(23)), dim_batch=8)  # warmup
    res = {}
    for db in [int(x) for x in sys.argv[2:]] or (8, 32, 64, 128):
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.time()
        try:
            jlens.jacobian_for_prompt(model, prompts[1], list(range(23)), dim_batch=db)
            torch.cuda.synchronize()
            res[db] = {"s_per_prompt": round(time.time() - t, 2), "peak_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
        except torch.OutOfMemoryError:
            res[db] = "OOM"; torch.cuda.empty_cache()
        print(db, res[db], flush=True)
    json.dump(res, open("runs/p0_3_bench.json", "w"), indent=1)
else:
    n, db = int(sys.argv[2]), int(sys.argv[3])
    os.makedirs("runs/jlens", exist_ok=True)
    jlens.configure_logging()
    t = time.time()
    lens = jlens.fit(model, prompts[:n], dim_batch=db, max_seq_len=128,
                     checkpoint_path=f"runs/jlens/{name}_n{n}_ckpt.pt", checkpoint_every=10)
    dt = time.time() - t
    lens.save(f"runs/jlens/{name}_n{n}.pt")
    json.dump({"n_prompts": n, "dim_batch": db, "max_seq_len": 128, "fit_seconds": round(dt, 1),
               "s_per_prompt": round(dt / n, 2), "peak_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2)},
              open(f"runs/jlens/{name}_n{n}_fit.json", "w"), indent=1)
    print("done", dt)
