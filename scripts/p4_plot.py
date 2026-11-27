"""Phase 4.5 figure: supervised probe vs zero-shot J-lens monitor, layer by layer, policy vs base reader.

usage: python scripts/p4_plot.py   (after p4_monitor.py and p4_directions.py)
Panels: positions 50% and pre_call of the decision turn's deliberation, and the action token. Solid =
policy reader, dashed = base-model reader on the same tokens. Lines: the diff-in-means probe
(supervised, p4_directions.py) and the three pre-registered monitors (zero-shot, p4_monitor.py).
HF layer 24 is the model's own output distribution, not a lens readout.
Writes results/phase4/probe_vs_lens.png.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path("results/phase4")
COL = {"diff-in-means probe": "#111827", "plan": "#2563eb", "intent": "#c2410c", "content": "#059669"}


def main():
    with open(RES / "monitor.json") as f:
        mon = json.load(f)
    with open(RES / "directions.json") as f:
        dr = json.load(f)
    layers, Q = mon["layers"], mon["Q"]
    panels = ("50%", "pre_call", "action")
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.8), sharey=True)
    for ax, q in zip(axes, panels):
        qi = Q.index(q)
        for reader, ls in (("policy", "-"), ("base", "--")):
            ax.plot(layers, [dr["readers"][reader][f"dim L{l}"]["test_auroc"]["direction"][q] for l in layers], ls,
                    color=COL["diff-in-means probe"], lw=1.6, label=f"diff-in-means probe, {reader}")
            for mn in ("plan", "intent", "content"):
                ax.plot(layers, [r[qi] for r in mon["monitors"][mn][reader]["test_per_layer"]], ls, color=COL[mn],
                        lw=1.3, label=f"J-lens '{mn}' monitor, {reader}")
        ax.axhline(0.5, color="#d1d5db", lw=0.8)
        ax.set_title(f"position: {q}", fontsize=9)
        ax.set_xlabel("HF layer (24 = model output)")
        ax.set_xticks(layers)
        ax.grid(alpha=0.3)
        ax.set_ylim(0.3, 0.9)
    axes[0].set_ylabel("AUROC, hack in this turn vs not (test)")
    axes[0].legend(frameon=False, fontsize=6.2, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(RES / "probe_vs_lens.png", dpi=130)


if __name__ == "__main__":
    main()
