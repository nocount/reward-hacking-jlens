"""Phase 3b: does the policy's residual stream carry anything about the coming hack beyond its text?

usage: python scripts/p3b_beyond_text.py   (after p3b_probes.py; reads results/phase3b/scores.npz)

Two tests at each position Q, on the test set, per subset:
  paired     AUROC(policy probe) - AUROC(text reader probe) of the same kind, with a paired bootstrap
             over problems. The two probes see identical tokens; only the weights reading them differ.
  stacked    2-fold cross-fitting by problem on the test set: logistic regression on the text signals
             [base logreg, base dim, tfidf] vs the same plus [policy logreg, policy dim]. The gain is
             what the policy's activations add once everything readable from the text is used.
Writes results/phase3b/beyond_text.json.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

RES = Path("results/phase3b")
Q = ("turn_start", "25%", "50%", "75%", "pre_call", "action")


def auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def paired(y, a, b, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    g = np.unique(groups)
    idx = {k: np.where(groups == k)[0] for k in g}
    d = []
    for _ in range(n):
        t = np.concatenate([idx[k] for k in rng.choice(g, len(g))])
        d.append(auc(y[t], a[t]) - auc(y[t], b[t]))
    return {"diff": auc(y, a) - auc(y, b), "ci": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]}


def stacked(y, F, groups, seed=0):
    """Out-of-fold scores from 2-fold cross-fitting split by problem."""
    g = np.unique(groups)
    fold = dict(zip(g, np.random.default_rng(seed).permutation(len(g)) % 2))
    f = np.array([fold[k] for k in groups])
    out = np.zeros(len(y))
    for k in (0, 1):
        Z = (F - F[f != k].mean(0)) / (F[f != k].std(0) + 1e-9)
        out[f == k] = LogisticRegression(max_iter=1000).fit(Z[f != k], y[f != k]).decision_function(Z[f == k])
    return out


def main():
    z = np.load(RES / "scores.npz", allow_pickle=False)
    y, pid, names, S = z["y"], z["pid"], list(z["names"]), z["S"]
    col = {n.split(" (")[0]: i for i, n in enumerate(names)}
    subsets = {"all": np.ones(len(y), bool), "matched prior calls": z["prior"] == z["sig"],
               "unspoken hacks vs all non-hacks": z["unspoken_hack"] | ~y}
    text = ["base logreg", "base dim", "tfidf text"]
    pol = ["policy logreg", "policy dim"]
    out = {}
    for sname, m in subsets.items():
        out[sname] = {}
        print("==", sname)
        for j, q in enumerate(Q):
            yy, gg = y[m], pid[m]
            r = {kind: paired(yy, S[col[f"policy {kind}"], m, j], S[col[f"base {kind}"], m, j], gg)
                 for kind in ("logreg", "dim")}
            Ft = np.stack([S[col[n], m, j] for n in text], 1)
            Fp = np.stack([S[col[n], m, j] for n in pol], 1)
            st, sp = stacked(yy, Ft, gg), stacked(yy, np.hstack([Ft, Fp]), gg)
            r["stacked_text"], r["stacked_text_plus_policy"] = auc(yy, st), auc(yy, sp)
            r["stacked_gain"] = paired(yy, sp, st, gg)
            out[sname][q] = r
            print(f"  {q:10s} policy-base logreg {r['logreg']['diff']:+.3f} {np.round(r['logreg']['ci'], 3)}"
                  f"  dim {r['dim']['diff']:+.3f} {np.round(r['dim']['ci'], 3)}"
                  f"  stacked text {r['stacked_text']:.3f} +policy {r['stacked_text_plus_policy']:.3f}"
                  f" gain {r['stacked_gain']['diff']:+.3f} {np.round(r['stacked_gain']['ci'], 3)}")
    with open(RES / "beyond_text.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
