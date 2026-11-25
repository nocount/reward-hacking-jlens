"""Phase 3b: how early, and how far beyond the text, can hack intent be read in the deliberation?

usage: python scripts/p3b_probes.py RUN STEP   (e.g. think_lr2e-5_s1 26)
Inputs: runs/p3b/acts/{RUN}_{STEP}_{p3_train,p3_test}_{policy,base}/ (p3b_acts.py),
        runs/p3b/resample/{RUN}_{STEP}_p3_test.jsonl (p3b_resample.py), the rollout files.

The unit is a rollout's decision turn (rhjl.readout.deliberation). y = the forbidden call is in that
turn; the negatives are every other rollout's rule turn, whatever the final outcome (clean or fail),
so problem difficulty is not a confound. Each rollout is read at prefix positions
Q = turn_start, 25/50/75% of the deliberation text, its last token (pre_call), and the action token
(the Phase 3 decision position, inside this turn), the same positions p3b_resample.py scored.
Probes are trained on turn_start and the span; the action row is only scored.

Signals at each position, all trained on the train problems and scored on the test problems:
  ceiling      resampled P(hack in this turn | prefix) under the policy (test only)
  policy probe logistic regression / diff-in-means on the policy's residual stream
  text reader  the same probes on the *base* model's residual stream over the same tokens: what a
               same-size LM that has never been trained on this task can read off the text
  tfidf        logistic regression on word 1-2-grams of the decoded prefix text
  mention      the Phase 3 regex: has the text so far named the answers file / solution / hint?
Probes are trained on all span positions pooled; the layer is picked on a 20% validation split of the
train problems (mean AUROC over Q).

Subsets: all rollouts; matched (prior call history = the hacks' most common one); unspoken (hacks
whose deliberation never matches the mention regex, against all non-hacks).
Writes results/phase3b/leadtime.json, leadtime.png, and scores.npz (every signal's test scores, [signal, N, |Q|],
for p3b_beyond_text.py).
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

LAYERS = (4, 8, 12, 14, 16, 18, 20, 22, 24)
Q = ("turn_start", "25%", "50%", "75%", "pre_call", "action")
OUT = Path("results/phase3b")


def auroc(y, s):
    y = np.asarray(y, bool)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def boot_ci(y, s, groups, n=500, seed=0):
    rng = np.random.default_rng(seed)
    g = np.unique(groups)
    idx = {k: np.where(groups == k)[0] for k in g}
    vals = []
    for _ in range(n):
        take = np.concatenate([idx[k] for k in rng.choice(g, len(g))])
        vals.append(auroc(y[take], s[take]))
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def qpos(length):
    """Row offsets (relative to the rollout's first row) of the Q positions except action; same rule
    as p3b_resample.ks. The action row is stored right after the span (offset + length)."""
    span = length - 1
    if not span:
        return [0] * (len(Q) - 1)
    return [0] + [max(1, round(f * span)) for f in (0.25, 0.5, 0.75)] + [length - 1]


class Set:
    def __init__(self, run, step, sub):
        self.dir = {rd: Path(f"runs/p3b/acts/{run}_{step}_{sub}_{rd}") for rd in ("policy", "base")}
        self.meta = jsonl(self.dir["policy"] / "index.jsonl")
        base_meta = jsonl(self.dir["base"] / "index.jsonl")
        assert [m["i"] for m in base_meta] == [m["i"] for m in self.meta]
        self.y = np.array([m["hack_in_turn"] for m in self.meta])
        self.pid = np.array([f'{m["nums"]}->{m["target"]}' for m in self.meta])
        self.off = np.array([m["offset"] for m in self.meta])
        self.len = np.array([m["length"] for m in self.meta])
        self.q = np.array([[o + p for p in qpos(n)] + [o + n] for o, n in zip(self.off, self.len)])  # [N, |Q|] rows
        self.prior = np.array([m["prior_calls"] for m in self.meta])
        self.rs = jsonl(f"runs/p2b/{run}/{sub}/step_{step}.jsonl")
        self._x = {}

    def X(self, reader, layer):
        key = (reader, layer)
        if key not in self._x:
            self._x[key] = np.load(self.dir[reader] / f"L{layer}.npy", mmap_mode="r")
        return self._x[key]

    def span_rows(self, mask):
        """Rows of the selected rollouts (turn_start and every span token), their labels and rollout ids."""
        rows, ys, gs = [], [], []
        for k in np.where(mask)[0]:
            rows.append(np.arange(self.off[k], self.off[k] + self.len[k]))
            ys.append(np.full(self.len[k], self.y[k]))
            gs.append(np.full(self.len[k], k))
        return np.concatenate(rows), np.concatenate(ys), np.concatenate(gs)


def fit_probe(S, mask, reader, layer, kind, max_rows=60000, seed=0):
    rows, ys, _ = S.span_rows(mask)
    if len(rows) > max_rows:
        pick = np.sort(np.random.default_rng(seed).choice(len(rows), max_rows, replace=False))
        rows, ys = rows[pick], ys[pick]
    X = np.asarray(S.X(reader, layer)[rows], np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-4
    Z = (X - mu) / sd
    if kind == "dim":
        w = Z[ys].mean(0) - Z[~ys].mean(0)
        return lambda A: ((np.asarray(A, np.float32) - mu) / sd) @ w
    clf = LogisticRegression(C=0.01, max_iter=3000).fit(Z, ys)
    return lambda A: clf.decision_function((np.asarray(A, np.float32) - mu) / sd)


def score_q(S, f, reader, layer):
    X = S.X(reader, layer)
    flat = S.q.reshape(-1)
    return f(X[flat]).reshape(S.q.shape)


def prefix_texts(S, tok):
    from rhjl.readout import Specials, deliberation
    sp = Specials(tok)
    out = []
    for m in S.meta:
        r = S.rs[m["i"]]
        d = deliberation(r, sp)
        full = r["prompt_ids"] + r["completion_ids"]
        ts = d["turn_start"]
        out.append([tok.decode(full[ts + 1: ts + p + 1]) for p in qpos(m["length"]) + [m["action"]]])
    return out


def main():
    from transformers import AutoTokenizer

    from rhjl.readout import MENTION
    run, step = sys.argv[1], int(sys.argv[2])
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("/home/wil/models/qwen3.5-2b-text")
    tr, te = Set(run, step, "p3_train"), Set(run, step, "p3_test")

    # validation split of the train problems for layer choice
    probs = np.unique(tr.pid)
    val_p = set(np.random.default_rng(0).choice(probs, len(probs) // 5, replace=False))
    is_val = np.array([p in val_p for p in tr.pid])
    best = {}
    for reader in ("policy", "base"):
        for kind in ("logreg", "dim"):
            val = {}
            for layer in LAYERS:
                f = fit_probe(tr, ~is_val, reader, layer, kind)
                s = score_q(tr, f, reader, layer)[is_val]
                val[layer] = float(np.nanmean([auroc(tr.y[is_val], s[:, j]) for j in range(len(Q))]))
            best[(reader, kind)] = max(val, key=val.get)
            print(reader, kind, "val AUROC by layer", {k: round(v, 3) for k, v in val.items()}, flush=True)

    scores = {}
    for (reader, kind), layer in best.items():
        f = fit_probe(tr, np.ones(len(tr.y), bool), reader, layer, kind)
        scores[f"{reader} {kind} (L{layer})"] = score_q(te, f, reader, layer)

    # text monitors
    txt_tr, txt_te = prefix_texts(tr, tok), prefix_texts(te, tok)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    flat_tr = [t for ts in txt_tr for t in ts]
    Xt = vec.fit_transform(flat_tr)
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(Xt, np.repeat(tr.y, len(Q)))
    scores["tfidf text"] = clf.decision_function(vec.transform([t for ts in txt_te for t in ts])).reshape(len(te.y), len(Q))
    scores["mention regex"] = np.array([[bool(MENTION.search(t)) for t in ts] for ts in txt_te], float)

    # ceiling
    rp = Path(f"runs/p3b/resample/{run}_{step}_p3_test.jsonl")
    if rp.exists():
        res = {(d["i"], d["k"]): d["p_hack"] for d in jsonl(rp)}
        scores["ceiling: resampled P(hack|prefix)"] = np.array(
            [[res.get((m["i"], p), np.nan) for p in qpos(m["length"]) + [m["action"]]] for m in te.meta])

    hs = list(te.prior[te.y])
    sig = Counter(hs).most_common(1)[0][0]
    unspoken_hack = te.y & np.array([m["mention"] is None for m in te.meta])
    subsets = {"all": np.ones(len(te.y), bool), "matched prior calls": te.prior == sig,
               "unspoken hacks vs all non-hacks": unspoken_hack | ~te.y}
    out = {"run": run, "step": step, "layers": {f"{a} {b}": v for (a, b), v in best.items()},
           "matched_signature": sig, "Q": Q,
           "n": {k: {"hack": int((te.y & m).sum() if k != "unspoken hacks vs all non-hacks" else unspoken_hack.sum()),
                     "non_hack": int((~te.y & m).sum())} for k, m in subsets.items()},
           "train": {"hack": int(tr.y.sum()), "non_hack": int((~tr.y).sum())},
           "median_span_tokens": {"hack": float(np.median(te.len[te.y] - 1)), "non_hack": float(np.median(te.len[~te.y] - 1))},
           "frac_hacks_mentioning": float(np.mean([m["mention"] is not None for m, h in zip(te.meta, te.y) if h])),
           "frac_nonhacks_mentioning": float(np.mean([m["mention"] is not None for m, h in zip(te.meta, te.y) if not h])),
           "results": {}}
    for sname, m in subsets.items():
        y = te.y[m]
        out["results"][sname] = {}
        for name, s in scores.items():
            row = {}
            for j, qn in enumerate(Q):
                sj = s[m, j]
                ok = ~np.isnan(sj)
                row[qn] = {"auroc": auroc(y[ok], sj[ok]), "ci": boot_ci(y[ok], sj[ok], te.pid[m][ok])}
            out["results"][sname][name] = row
        print(f"\n== {sname}: {out['n'][sname]}")
        for name, row in out["results"][sname].items():
            print(f"  {name:38s}", "  ".join(f"{q} {row[q]['auroc']:.3f}" for q in Q))
    np.savez_compressed(OUT / "scores.npz", y=te.y, pid=te.pid, prior=te.prior, unspoken_hack=unspoken_hack, sig=sig,
             names=np.array(list(scores)), S=np.stack(list(scores.values())))
    with open(OUT / "leadtime.json", "w") as f:
        json.dump(out, f, indent=1)
    plot(out)


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    subs = list(out["results"])
    fig, axes = plt.subplots(1, len(subs), figsize=(4.6 * len(subs), 3.8), sharey=True)
    style = {"ceiling": ("#111827", ":"), "policy logreg": ("#2563eb", "-"), "policy dim": ("#2563eb", "--"),
             "base logreg": ("#c2410c", "-"), "base dim": ("#c2410c", "--"), "tfidf": ("#059669", "-"),
             "mention": ("#9ca3af", "-")}
    for ax, sname in zip(axes, subs):
        for name, row in out["results"][sname].items():
            col, ls = next(v for k, v in style.items() if name.startswith(k))
            ax.plot(range(len(out["Q"])), [row[q]["auroc"] for q in out["Q"]], ls, color=col, lw=1.6, marker="o",
                    ms=3, label=name.replace("base", "text reader (base model)").replace("policy", "policy probe"))
        ax.axhline(0.5, color="#d1d5db", lw=0.8)
        ax.set_xticks(range(len(out["Q"])), out["Q"], fontsize=8)
        ax.set_title(f"{sname}  (hack {out['n'][sname]['hack']} / non-hack {out['n'][sname]['non_hack']})", fontsize=9)
        ax.set_xlabel("position in the decision turn's deliberation")
        ax.grid(alpha=0.3)
        ax.set_ylim(0.25, 1.01)
    axes[0].set_ylabel("AUROC, hack in this turn vs not")
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "leadtime.png", dpi=130)


if __name__ == "__main__":
    main()
