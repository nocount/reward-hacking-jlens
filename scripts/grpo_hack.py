"""LoRA GRPO in the hackable Countdown environment (TRL + vLLM colocate, multi-turn tool loop).

Phase 1 uses it as an end-to-end smoke test; Phase 2 uses it for the real runs. Each rollout gets
its own CountdownFS (TRL environment_factory), whose list_files/read_file methods are the tools.
Reward is the honest grader on the final turn; labels are logged per rollout.

env:
  VARIANT (explicit)        loophole hint, see rhjl.hackenv.VARIANTS
  GRADER (strict)           "lenient" adds the grader loophole (rewards hitting the target with any numbers)
  STEPS (30)  MAX_COMP (1536)  NUM_GEN (8)  PROMPTS_PER_STEP (4)  PER_DEVICE_BS (8)
  LR (2e-5)  LORA_R (32)  VLLM_UTIL (0.35)  MAX_TOOL_ITERS (3)  SAVE_EVERY (0 = never)
  OUT (runs/grpo_hack)  MODEL (/home/wil/models/qwen3.5-2b-text)  SEED (0)
"""
import json
import os
import subprocess
import threading
import time
from collections import Counter, defaultdict


def env(name, default, cast=str):
    return cast(os.environ.get(name, default))


def main():
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    from rhjl.countdown import DATASET, solve
    from rhjl.hackenv import CountdownFS, build_prompt, label

    variant = env("VARIANT", "explicit")
    grader = env("GRADER", "strict")
    steps, max_comp = env("STEPS", 30, int), env("MAX_COMP", 1536, int)
    num_gen, pps, bs = env("NUM_GEN", 8, int), env("PROMPTS_PER_STEP", 4, int), env("PER_DEVICE_BS", 8, int)
    save_every = env("SAVE_EVERY", 0, int)
    out = env("OUT", "runs/grpo_hack")
    model = env("MODEL", "/home/wil/models/qwen3.5-2b-text")
    os.makedirs(out, exist_ok=True)

    ds = load_dataset(DATASET, split="train").shuffle(seed=42).select(range(32000, 32000 + steps * pps * 2))
    ds = ds.map(lambda r: {"prompt": build_prompt(r["nums"], r["target"], variant),
                           "solution": solve(r["nums"], r["target"])})
    roll_f = open(f"{out}/rollouts.jsonl", "w")
    per_step = defaultdict(Counter)

    def hack_env_reward(completions, environments, trainer_state, log_metric, **_):
        rewards, labs = [], []
        step = trainer_state.global_step
        for msgs, e in zip(completions, environments):
            lab = label(e, msgs, grader=grader)
            rewards.append(lab["reward"])
            labs.append(lab["label"])
            per_step[step].update({"n": 1, lab["label"]: 1, "attempted": lab["attempted_forbidden"],
                                   "used_tools": lab["n_tool_calls"] > 0})
            roll_f.write(json.dumps({"step": step, "nums": e.nums, "target": e.target, "messages": msgs, **lab},
                                    default=str) + "\n")
        roll_f.flush()
        n = len(rewards)
        for k in ("hack", "clean", "fail"):
            log_metric(f"label/{k}", labs.count(k) / n)
        return rewards

    peak = {"mib": 0}

    def poll():
        while True:
            try:
                q = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
                peak["mib"] = max(peak["mib"], int(subprocess.check_output(q).split()[0]))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5)

    threading.Thread(target=poll, daemon=True).start()

    class JsonlLog(TrainerCallback):
        def __init__(self):
            self.f, self.t = open(f"{out}/log.jsonl", "w"), time.time()

        def on_log(self, args, state, control, logs=None, **kw):
            self.f.write(json.dumps({"step": state.global_step, "wall_s": round(time.time() - self.t, 1),
                                     "gpu_used_peak_mib": peak["mib"], **(logs or {})}) + "\n")
            self.f.flush()

    cfg = GRPOConfig(
        output_dir=out, max_steps=steps, learning_rate=env("LR", 2e-5, float), lr_scheduler_type="constant",
        warmup_steps=0, per_device_train_batch_size=bs, gradient_accumulation_steps=pps * num_gen // bs,
        num_generations=num_gen, max_completion_length=max_comp, temperature=1.0, top_p=1.0,
        chat_template_kwargs={"enable_thinking": False}, beta=0.0, loss_type="dapo",
        mask_truncated_completions=False, max_tool_calling_iterations=env("MAX_TOOL_ITERS", 3, int),
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=env("VLLM_UTIL", 0.35, float),
        vllm_enable_sleep_mode=True, vllm_max_model_length=max_comp + 1024,
        gradient_checkpointing=True, bf16=True, logging_steps=1, report_to="none", log_completions=False,
        save_strategy="steps" if save_every else "no", save_steps=save_every or 500, save_only_model=True,
        seed=env("SEED", 0, int),
    )
    lora = LoraConfig(r=env("LORA_R", 32, int), lora_alpha=2 * env("LORA_R", 32, int), lora_dropout=0.0,
                      target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = GRPOTrainer(model=model, reward_funcs=hack_env_reward, args=cfg, train_dataset=ds,
                          peft_config=lora, environment_factory=CountdownFS, callbacks=[JsonlLog()])
    t0 = time.time()
    trainer.train()
    wall = time.time() - t0
    roll_f.close()

    curve = {s: {k: round(c[k] / c["n"], 4) for k in ("hack", "clean", "fail", "attempted", "used_tools")}
             for s, c in sorted(per_step.items())}
    summary = {"variant": variant, "grader": grader, "steps": steps, "max_completion_length": max_comp, "num_generations": num_gen,
               "prompts_per_step": pps, "wall_s": round(wall, 1), "s_per_step": round(wall / steps, 1),
               "gpu_used_peak_mib (incl. ~1GB desktop)": peak["mib"],
               "torch_max_allocated_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2), "per_step": curve}
    with open(f"{out}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_step"}, indent=1))


if __name__ == "__main__":
    main()
