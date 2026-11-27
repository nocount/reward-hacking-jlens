"""Phase 4.1/4.4/4.5: zero-shot J-lens monitors on the Phase 3b decision turns, policy vs base.

usage: python scripts/p4_monitor.py [RUN STEP]   (default think_lr2e-5_s1 25)
Inputs: Phase 3b activations (runs/p3b/acts/, positions Q of every decision turn, policy and base
model on the same tokens), the base lens runs/jlens/qwen35_2b_n100.pt and the policy lens
runs/jlens/policy_s1_25_n100.pt (same 100 FineWeb prompts), results/phase4/monitor_tokens.json
(fixed before any readout).

Readers (activations, lens):
  policy                 policy activations, policy lens
  base                   base-model activations over the same tokens, base lens: the text reader
  policy acts/base lens  isolates what changes in the activations from what changes in J
  policy logit lens      policy activations, J = identity (the logit-lens baseline)

1. Pre-registered monitors: score = log sum of lens probability over a list's tokens, at every saved
   layer and position. Layer per (monitor, reader) is picked on p3_train (mean AUROC over Q); the
   test AUROC and the paired policy - base difference (bootstrap over problems) are reported.
2. Exploratory discovery: on p3_train, the per-token contrast mean log p(hack) - mean log p(non-hack)
   for each reader; tokens where the policy's contrast most exceeds the base's are the candidates
   for policy-specific lens content. The top 20 per (layer, position) become a monitor scored on
   p3_test, so the search and the test use disjoint problems.
3. Lens comparison (4.1): relative Frobenius distance of J per layer and top-1 agreement of the two
   lenses on the same policy activations.
Writes results/phase4/monitor.json and monitor.png, and every score to runs/p4/monitor_scores.npz (12 MB).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from p3b_beyond_text import paired
from p3b_probes import LAYERS, Q, Set, auroc

from rhjl.lensread import Unembed, hf_to_jlens, load_lens, token_ids

OUT = Path("results/phase4")
LENS = {"base": "runs/jlens/qwen35_2b_n100.pt", "policy": "runs/jlens/policy_s1_25_n100.pt"}
READERS = {"policy": ("policy", "policy"), "base": ("base", "base"),
           "policy acts/base lens": ("policy", "base"), "policy logit lens": ("policy", None)}
DISC_Q = (2, 4, 5)  # 50%, pre_call, action
TOPK = 20


def main():
    from transformers import AutoTokenizer
    run, step = (sys.argv[1], int(sys.argv[2])) if len(sys.argv) > 2 else ("think_lr2e-5_s1", 25)
    tok = AutoTokenizer.from_pretrained("/home/wil/models/qwen3.5-2b-text")
    with open(OUT / "monitor_tokens.json") as f:
        lists = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    tids = {k: torch.tensor(token_ids(tok, v), device="cuda") for k, v in lists.items()}
    U = Unembed()
    lenses = {k: load_lens(p) for k, p in LENS.items()}
    sets = {"train": Set(run, step, "p3_train"), "test": Set(run, step, "p3_test")}
    nL = len(LAYERS)

    # scores[split][reader]: [N, |Q|, nL, n_monitors]; contrast[reader]: [nL, len(DISC_Q), V] (train)
    scores = {s: {r: np.zeros((len(S.y), len(Q), nL, len(tids)), np.float32) for r in READERS} for s, S in sets.items()}
    contrast = {r: torch.zeros(nL, len(DISC_Q), U.W.shape[0], device="cuda") for r in READERS}
    top1 = {s: {r: np.zeros((len(S.y), len(Q), nL), np.int64) for r in READERS} for s, S in sets.items()}
    for split, S in sets.items():
        flat = S.q.reshape(-1)
        yq = np.repeat(S.y, len(Q))
        qi = np.tile(np.arange(len(Q)), len(S.y))
        for li, layer in enumerate(LAYERS):
            j = hf_to_jlens(layer)
            for rname, (acts, lens) in READERS.items():
                X = np.asarray(S.X(acts, layer)[flat], np.float32)
                sc = np.zeros((len(flat), len(tids)), np.float32)
                t1 = np.zeros(len(flat), np.int64)
                for s0 in range(0, len(X), 1024):
                    h = torch.as_tensor(X[s0:s0 + 1024], device="cuda")
                    if j is None:
                        lp = U.logprobs(h, normed=True)
                    elif lens is None:
                        lp = U.logprobs(h)
                    else:
                        lp = U.logprobs(h @ lenses[lens][j].T)
                    sc[s0:s0 + 1024] = torch.stack([torch.logsumexp(lp[:, t], 1) for t in tids.values()], 1).cpu().numpy()
                    t1[s0:s0 + 1024] = lp.argmax(1).cpu().numpy()
                    if split == "train":
                        yy, qq = yq[s0:s0 + 1024], qi[s0:s0 + 1024]
                        for di, q in enumerate(DISC_Q):
                            for lab, sign in ((True, 1.0 / (S.y.sum())), (False, -1.0 / ((~S.y).sum()))):
                                m = torch.as_tensor((qq == q) & (yy == lab), device="cuda")
                                if m.any():
                                    contrast[rname][li, di] += sign * lp[m].sum(0)
                    del lp
                scores[split][rname][:, :, li] = sc.reshape(len(S.y), len(Q), -1)
                top1[split][rname][:, :, li] = t1.reshape(len(S.y), len(Q))
            print(split, "layer", layer, "done", flush=True)

    tr, te = sets["train"], sets["test"]
    mnames = list(tids)
    out = {"run": run, "step": step, "Q": Q, "layers": LAYERS, "lists": lists,
           "n_tokens": {k: len(v) for k, v in tids.items()}, "monitors": {}, "discovered": {}, "lens_compare": {}}

    # 1. pre-registered monitors
    for mi, mn in enumerate(mnames):
        out["monitors"][mn] = {}
        for r in READERS:
            val = [np.nanmean([auroc(tr.y, scores["train"][r][:, q, li, mi]) for q in range(len(Q))]) for li in range(nL)]
            li = int(np.nanargmax(val))
            test = {Q[q]: auroc(te.y, scores["test"][r][:, q, li, mi]) for q in range(len(Q))}
            per_layer = [[auroc(te.y, scores["test"][r][:, q, l2, mi]) for q in range(len(Q))] for l2 in range(nL)]
            out["monitors"][mn][r] = {"layer": LAYERS[li], "train_mean_auroc": float(val[li]), "test": test,
                                      "test_per_layer": per_layer}
        lp_, lb_ = (LAYERS.index(out["monitors"][mn][r]["layer"]) for r in ("policy", "base"))
        out["monitors"][mn]["policy_minus_base"] = {
            Q[q]: paired(te.y, scores["test"]["policy"][:, q, lp_, mi], scores["test"]["base"][:, q, lb_, mi], te.pid)
            for q in range(len(Q))}
        print(mn, {r: (v["layer"], {q: round(a, 3) for q, a in v["test"].items()}) for r, v in out["monitors"][mn].items()
                   if r in READERS}, flush=True)

    # 2. exploratory: tokens whose hack contrast is larger under the policy than the base reader
    def dec(ids):
        return [tok.decode([int(t)]) for t in ids]
    for li, layer in enumerate(LAYERS):
        for di, q in enumerate(DISC_Q):
            cp, cb = contrast["policy"][li, di], contrast["base"][li, di]
            diff_ids = (cp - cb).topk(TOPK).indices
            key = f"L{layer} {Q[q]}"
            # score the discovered list on test with each reader at the same layer/position
            Xs = {}
            for rname in ("policy", "base"):
                acts, lens = READERS[rname]
                X = torch.as_tensor(np.asarray(te.X(acts, layer)[te.q[:, q]], np.float32), device="cuda")
                j = hf_to_jlens(layer)
                lp = U.logprobs(X, normed=True) if j is None else U.logprobs(X @ lenses[lens][j].T)
                Xs[rname] = torch.logsumexp(lp[:, diff_ids], 1).cpu().numpy()
                del lp
            out["discovered"][key] = {
                "policy_contrast_top": dec(cp.topk(TOPK).indices), "base_contrast_top": dec(cb.topk(TOPK).indices),
                "policy_minus_base_top": dec(diff_ids),
                "test_auroc": {r: auroc(te.y, s) for r, s in Xs.items()},
                "test_policy_minus_base": paired(te.y, Xs["policy"], Xs["base"], te.pid)}

    # 3. lens comparison
    for j in sorted(set(lenses["base"]) & set(lenses["policy"])):
        Jb, Jp = lenses["base"][j], lenses["policy"][j]
        out["lens_compare"][j] = {"rel_frobenius": float((Jp - Jb).norm() / Jb.norm())}
    for li, layer in enumerate(LAYERS):
        a = top1["test"]["policy"][:, :, li]
        out["lens_compare"].setdefault(f"hf{layer}", {}).update({
            "top1_agree_policy_vs_crosslens": float((a == top1["test"]["policy acts/base lens"][:, :, li]).mean()),
            "top1_agree_policy_vs_base": float((a == top1["test"]["base"][:, :, li]).mean())})

    with open(OUT / "monitor.json", "w") as f:
        json.dump(out, f, indent=1)
    Path("runs/p4").mkdir(parents=True, exist_ok=True)
    np.savez_compressed("runs/p4/monitor_scores.npz", **{f"{s}/{r}": v for s, d in scores.items() for r, v in d.items()},
                        y_test=te.y, pid_test=te.pid, y_train=tr.y)
    plot(out)


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with open("results/phase3b/leadtime.json") as f:
        p3 = json.load(f)["results"]["all"]
    mons = list(out["monitors"])
    fig, axes = plt.subplots(1, len(mons), figsize=(4.6 * len(mons), 3.8), sharey=True)
    col = {"policy": "#2563eb", "base": "#c2410c", "policy acts/base lens": "#7c3aed", "policy logit lens": "#9ca3af"}
    xs = range(len(out["Q"]))
    for ax, mn in zip(axes, mons):
        for r, c in col.items():
            v = out["monitors"][mn][r]
            ax.plot(xs, [v["test"][q] for q in out["Q"]], color=c, lw=1.6, marker="o", ms=3,
                    label=f"{r} (L{v['layer']})")
        ax.plot(xs, [p3["policy logreg (L12)"][q]["auroc"] for q in out["Q"]], "--", color="#111827", lw=1.0,
                label="Phase 3b policy probe (supervised)")
        ax.plot(xs, [p3["ceiling: resampled P(hack|prefix)"][q]["auroc"] for q in out["Q"]], ":", color="#111827",
                lw=1.0, label="ceiling")
        ax.axhline(0.5, color="#d1d5db", lw=0.8)
        ax.set_xticks(list(xs), out["Q"], fontsize=8)
        ax.set_title(f"zero-shot J-lens monitor: '{mn}' list", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(0.25, 1.01)
    axes[0].set_ylabel("AUROC, hack in this turn vs not (test)")
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "monitor.png", dpi=130)


if __name__ == "__main__":
    main()
