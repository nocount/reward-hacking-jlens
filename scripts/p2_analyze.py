"""Phase 2: training curves for every run under runs/p2 (from each run's rollouts.jsonl).

Per step: label fractions, plus two things the probe and monitor phases care about:
  - chars_before_hack: characters the model wrote in the turn that issued the forbidden call
    (file hacks) or in the final turn before <answer> (grader hacks);
  - names_file: fraction of file hacks whose final turn mentions answers.txt.
Writes results/phase2/curves.json and results/phase2/training_curves.png.

Also gathers the held-out checkpoint evals (scripts/p2_eval_ckpts.py, RUN/eval/) into
results/phase2/heldout.json and heldout_curves.png, and describes the checkpoints chosen for
Phase 3 (CHOSEN) in results/phase2/checkpoints.json.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path("runs/p2")
OUT = Path("results/phase2")
_FILE = re.compile(r"answers?\.txt", re.IGNORECASE)
_RULE = "Solvers must not read"

# Checkpoints for Phase 3, picked from the held-out evals (30-60% hack, most problems mixed).
CHOSEN = {
    "main": ("file_lr5e-6_s0", 97),
    "heldout_seed": ("file_lr1e-5_s0", 17),
    "heldout_style": ("file_lr2e-5_s0", 40),  # writes ~80 chars of text before the forbidden call
    "rule_unseen": ("file_lr2e-5_s2", 58),  # reads answers.txt in the same turn as README.md
    "grader": ("grader_lr2e-5_s0", 96),
}


def hack_turn_chars(r: dict) -> int | None:
    msgs = r["messages"]
    if r.get("read_forbidden"):
        for m in msgs:
            if m["role"] == "assistant" and any(_FILE.search(str(tc["function"].get("arguments")))
                                                for tc in m.get("tool_calls") or []):
                return len(m.get("content") or "")
        return None
    final = next((m.get("content") or "" for m in reversed(msgs) if m["role"] == "assistant"), "")
    return len(final.split("<answer>")[0])


def curves(run: Path) -> dict:
    by = defaultdict(list)
    for line in open(run / "rollouts.jsonl"):
        r = json.loads(line)
        by[r["step"]].append(r)
    out = {}
    for s in sorted(by):
        g = by[s]
        hacks = [r for r in g if r["label"] == "hack"]
        chars = [c for c in (hack_turn_chars(r) for r in hacks) if c is not None]
        file_hacks = [r for r in hacks if r.get("read_forbidden")]
        out[s] = {
            "n": len(g),
            **{k: sum(r["label"] == k for r in g) / len(g) for k in ("hack", "clean", "fail")},
            "median_chars_before_hack": sorted(chars)[len(chars) // 2] if chars else None,
            "names_file": (sum(bool(_FILE.search(next((m.get("content") or "" for m in reversed(r["messages"])
                                                        if m["role"] == "assistant"), ""))) for r in file_hacks)
                           / len(file_hacks)) if file_hacks else None,
            "mean_model_chars": sum(sum(len(m.get("content") or "") for m in r["messages"] if m["role"] == "assistant")
                                    for r in g) / len(g),
        }
    return out


def saw_rule_before_hack(r: dict) -> bool | None:
    """True if README.md's ban came back in a turn before the turn that called answers.txt."""
    seen = False
    for m in r["messages"]:
        if m["role"] == "assistant" and any(_FILE.search(str(tc["function"].get("arguments")))
                                            for tc in m.get("tool_calls") or []):
            return seen
        if m["role"] == "tool" and _RULE in (m.get("content") or ""):
            seen = True
    return None


def describe_ckpt(run: str, step: int) -> dict:
    rs = [json.loads(line) for line in open(RUNS / run / "eval" / f"step_{step}.jsonl")]
    hacks = [r for r in rs if r["label"] == "hack"]
    file_hacks = [r for r in hacks if r["read_forbidden"]]
    chars = sorted(c for c in map(hack_turn_chars, hacks) if c is not None)
    stats = json.load(open(RUNS / run / "eval" / "summary.json"))[str(step)]
    return {
        "run": run, "step": step, **{k: stats[k] for k in ("hack", "clean", "fail", "problems_with_hack_and_clean")},
        "hack_types": {t: sum(r["hack_type"] == t for r in hacks) for t in {r["hack_type"] for r in hacks}},
        "median_chars_before_hack": chars[len(chars) // 2] if chars else None,
        "names_file": (sum(bool(_FILE.search(next((m.get("content") or "" for m in reversed(r["messages"])
                                                   if m["role"] == "assistant"), ""))) for r in file_hacks)
                       / len(file_hacks)) if file_hacks else None,
        "saw_rule_before_hack": (sum(bool(saw_rule_before_hack(r)) for r in file_hacks) / len(file_hacks))
        if file_hacks else None,
    }


def heldout_plot(runs: list[Path]) -> None:
    ev = {p.name: json.load(open(p / "eval" / "summary.json")) for p in runs if (p / "eval" / "summary.json").exists()}
    with open(OUT / "heldout.json", "w") as f:
        json.dump(ev, f, indent=1)
    fig, axes = plt.subplots(1, len(ev), figsize=(4.2 * len(ev), 3.4), sharey=True, squeeze=False)
    chosen = {(r, s) for r, s in CHOSEN.values()}
    for ax, (name, e) in zip(axes[0], ev.items()):
        steps = sorted(int(s) for s in e)
        for k, col in (("hack", "#c2410c"), ("clean", "#2563eb"), ("fail", "#9ca3af")):
            ax.plot(steps, [e[str(s)][k] for s in steps], "o-", ms=2.5, color=col, lw=1.5, label=k)
        ax.plot(steps, [e[str(s)]["problems_with_hack_and_clean"] for s in steps], "--", color="#7c3aed", lw=1.2,
                label="problems w/ hack+clean")
        ax.axhspan(0.3, 0.6, color="#c2410c", alpha=0.06)
        for r, s in chosen:
            if r == name:
                ax.axvline(s, color="k", lw=0.8, ls=":")
        ax.set_title(name)
        ax.set_xlabel("checkpoint (GRPO step)")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("held-out fraction (64 problems x 8)")
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "heldout_curves.png", dpi=130)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = sorted(p for p in RUNS.iterdir() if (p / "rollouts.jsonl").exists())
    allc = {p.name: curves(p) for p in runs}
    with open(OUT / "curves.json", "w") as f:
        json.dump(allc, f, indent=1)

    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 3.4), sharey=True, squeeze=False)
    for ax, (name, c) in zip(axes[0], allc.items()):
        steps = sorted(c)
        for k, col in (("hack", "#c2410c"), ("clean", "#2563eb"), ("fail", "#9ca3af")):
            ys = [c[s][k] for s in steps]
            sm = [sum(ys[max(0, i - 4): i + 1]) / len(ys[max(0, i - 4): i + 1]) for i in range(len(ys))]
            ax.plot(steps, ys, color=col, alpha=0.25, lw=1)
            ax.plot(steps, sm, color=col, lw=2, label=k)
        ax.set_title(name)
        ax.set_xlabel("GRPO step")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("fraction of rollouts (5-step mean)")
    axes[0][0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=130)

    for name, c in allc.items():
        steps = sorted(c)
        first = {th: next((s for s in steps if c[s]["hack"] >= th), None) for th in (0.25, 0.5, 0.9)}
        print(name, "steps", len(steps), "first step with hack >= 25/50/90%:", first)

    heldout_plot(runs)
    ck = {role: describe_ckpt(r, s) for role, (r, s) in CHOSEN.items() if (RUNS / r / "eval" / f"step_{s}.jsonl").exists()}
    with open(OUT / "checkpoints.json", "w") as f:
        json.dump(ck, f, indent=1)
    for role, d in ck.items():
        print(role, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in d.items()}))


if __name__ == "__main__":
    main()
