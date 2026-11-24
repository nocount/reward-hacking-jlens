"""Phase 3 figures from results/phase3/{probes,baselines}.json.

probe_vs_policy.png  One panel per file-hack checkpoint: AUROC per layer of the probes trained at the
                     main checkpoint, hack vs non-hack among rollouts with the checkpoint's usual call
                     history (auroc_nonhack_match; removes call-history and difficulty confounds).
                     Solid = decision position, dashed = turn_start. Horizontal lines: the policy's own
                     next-token logit at the decision ("\\n" vs <|im_end|>) and trivial prefix features.
                     Last panel: grader hacks, in-distribution probes at pre_answer (hack vs clean and
                     hack vs non-hack) and the file-trained decision probe transferred there.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/phase3")
DIM, LOG, POL, TRIV = "#c2410c", "#2563eb", "#111827", "#9ca3af"


def main():
    pr = json.load(open(RES / "probes.json"))["per_layer"]
    bl = json.load(open(RES / "baselines.json"))
    tests = [t for t in pr["dim/decision"] if t != "_val" and not t.startswith("grader")]
    fig, axes = plt.subplots(1, len(tests) + 1, figsize=(3.9 * (len(tests) + 1), 3.6), sharey=True)
    for ax, t in zip(axes, tests):
        for fam, col in (("dim", DIM), ("logreg", LOG)):
            for pos, ls in (("decision", "-"), ("turn_start", "--")):
                rows = pr[f"{fam}/{pos}"][t]
                ax.plot(range(len(rows)), [r["auroc_nonhack_match"] for r in rows], ls, color=col, lw=1.5,
                        label=f"{'diff-in-means' if fam == 'dim' else 'logreg'} @ {pos}")
        ax.axhline(bl["policy_logit"][t]["auroc_nonhack_match"], color=POL, lw=1.2, ls=":", label="policy logit @ decision")
        ax.axhline(bl["trivial_features"][t]["auroc_nonhack_match"], color=TRIV, lw=1.2, label="trivial prefix features")
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("layer (0 = embeddings)")
        ax.grid(alpha=0.3)
    ax = axes[-1]
    t = "grader step 96 (held-out problems)"
    for fam, col in (("dim", DIM), ("logreg", LOG)):
        rows = pr[f"{fam}/pre_answer"][t]
        ax.plot(range(len(rows)), [r["auroc"] for r in rows], color=col, lw=1.5, label=f"{fam} @ pre_answer, hack vs clean")
        ax.plot(range(len(rows)), [r["auroc_nonhack"] for r in rows], "--", color=col, lw=1.2,
                label=f"{fam} @ pre_answer, hack vs non-hack")
    rows = pr["logreg/decision"]["grader step 96 @ pre_answer"]
    ax.plot(range(len(rows)), [r["auroc"] for r in rows], ":", color="#059669", lw=1.5, label="file-trained logreg (transfer)")
    ax.set_title("grader hacks (in-distribution probes)", fontsize=9)
    ax.set_xlabel("layer")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    for a in axes:
        a.axhline(0.5, color="#d1d5db", lw=0.8)
        a.set_ylim(0.3, 1.01)
    axes[0].set_ylabel("AUROC (hack vs non-hack, matched call history)")
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(RES / "probe_vs_policy.png", dpi=130)


if __name__ == "__main__":
    main()
