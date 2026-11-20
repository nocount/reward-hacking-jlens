"""Phase 0.6: short LoRA GRPO run on honest Countdown (TRL + vLLM colocate). Measures throughput and VRAM.

env: STEPS (50), MAX_COMP (1024), THINK (0), NUM_GEN (8), PROMPTS_PER_STEP (4), OUT (runs/p0_6_grpo)
"""
import json, os, subprocess, sys, threading, time
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from rhjl.countdown import DATASET, build_messages, score

STEPS = int(os.environ.get("STEPS", 50)); MAX_COMP = int(os.environ.get("MAX_COMP", 1024))
THINK = os.environ.get("THINK", "0") == "1"  # 0.5 showed thinking mode almost never finishes
NUM_GEN = int(os.environ.get("NUM_GEN", 8)); PPS = int(os.environ.get("PROMPTS_PER_STEP", 4))
OUT = os.environ.get("OUT", "runs/p0_6_grpo"); MODEL = "Qwen/Qwen3.5-2B"
os.makedirs(OUT, exist_ok=True)

ds = load_dataset(DATASET, split="train").shuffle(seed=42).select(range(32000, 32000 + STEPS * PPS * 2))  # GRAFT's RL chunk
ds = ds.map(lambda r: {"prompt": build_messages(r["nums"], r["target"])})
tok = AutoTokenizer.from_pretrained(MODEL)
roll_f = open(f"{OUT}/rollouts.jsonl", "w")


def countdown_reward(prompts, completions, completion_ids, nums, target, trainer_state, **_):
    rewards = []
    for ids, n, t in zip(completion_ids, nums, target):
        text = tok.decode(ids, skip_special_tokens=False)
        s = score(text, n, t, thinking=THINK)
        rewards.append(1.0 if s["correct"] else 0.0)
        roll_f.write(json.dumps({"step": trainer_state.global_step, "nums": n, "target": t, "n_tok": len(ids),
                                 "reward": rewards[-1], **{k: v for k, v in s.items()}, "text": text}) + "\n")
    roll_f.flush()
    return rewards


# GPU memory poller (nvidia-smi sees vLLM + trainer together)
peak = {"mib": 0}
def poll():
    while True:
        try:
            used = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])
            peak["mib"] = max(peak["mib"], used)
        except Exception:
            pass
        time.sleep(0.5)
threading.Thread(target=poll, daemon=True).start()


class JsonlLog(TrainerCallback):
    def __init__(self): self.f = open(f"{OUT}/log.jsonl", "w"); self.t = time.time()
    def on_log(self, args, state, control, logs=None, **kw):
        self.f.write(json.dumps({"step": state.global_step, "wall_s": round(time.time() - self.t, 1),
                                 "gpu_used_peak_mib": peak["mib"], **(logs or {})}) + "\n"); self.f.flush()


cfg = GRPOConfig(
    output_dir=OUT, max_steps=STEPS, learning_rate=2e-5, lr_scheduler_type="constant", warmup_steps=0,
    per_device_train_batch_size=8, gradient_accumulation_steps=PPS * NUM_GEN // 8,
    num_generations=NUM_GEN, max_completion_length=MAX_COMP, temperature=1.0, top_p=1.0,
    chat_template_kwargs={"enable_thinking": THINK}, beta=0.0, loss_type="dapo", mask_truncated_completions=False,
    use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.35, vllm_enable_sleep_mode=True,
    vllm_max_model_length=MAX_COMP + 256,
    gradient_checkpointing=True, bf16=True, logging_steps=1, save_strategy="no", report_to="none",
    log_completions=False, seed=0,
)
lora = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
trainer = GRPOTrainer(model=MODEL, reward_funcs=countdown_reward, args=cfg, train_dataset=ds,
                      peft_config=lora, callbacks=[JsonlLog()])
t0 = time.time()
trainer.train()
wall = time.time() - t0
rows = [json.loads(l) for l in open(f"{OUT}/rollouts.jsonl")]
summary = {"thinking": THINK, "steps": STEPS, "max_completion_length": MAX_COMP, "num_generations": NUM_GEN, "prompts_per_step": PPS,
           "wall_s": round(wall, 1), "s_per_step": round(wall / STEPS, 1),
           "completion_tokens": sum(r["n_tok"] for r in rows),
           "completion_tok_per_s_end_to_end": round(sum(r["n_tok"] for r in rows) / wall),
           "gpu_used_peak_mib (incl. ~1GB desktop)": peak["mib"],
           "torch_max_allocated_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
           "reward_first10_steps": round(sum(r["reward"] for r in rows if r["step"] < 10) / max(1, sum(1 for r in rows if r["step"] < 10)), 4),
           "reward_last10_steps": round(sum(r["reward"] for r in rows if r["step"] >= STEPS - 10) / max(1, sum(1 for r in rows if r["step"] >= STEPS - 10)), 4),
           "frac_truncated": round(sum(r["n_tok"] >= MAX_COMP for r in rows) / len(rows), 4)}
print(json.dumps(summary, indent=1))
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=1)
roll_f.close()
sys.stdout.flush()
