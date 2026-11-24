"""Phase 3 (RQ1): difference-in-means and logistic-regression probes per layer, AUROC on test sets.

Reads runs/p3/acts/*.npz (scripts/p3_acts.py). Probes are trained at the main checkpoint
(file_lr5e-6_s0 step 97, problems 17000-17511) on hack vs clean rollouts and scored on the common
test problems (18000-18255) at every checkpoint chosen in Phase 2, each with its own model's
activations (what a monitor would see). Rollouts that tried to read answers.txt but never got it
are dropped. Writes results/phase3/probes.json and plots.

Metrics per (probe, position, layer, test set):
  auroc        hack vs clean, all rollouts with that position
  auroc_match  hack vs clean, only rollouts whose tool calls before the decision match the
               checkpoint's most common hack signature (identical call history)
  auroc_nonhack hack vs clean+fail (the monitoring setting)
  auroc_nonhack_match  hack vs clean+fail with the matched signature: removes both call-history and
               problem-difficulty confounds (clean = solved, so clean rollouts sit on easier problems)
  auroc_within mean over problems with both labels of the within-problem AUROC (controls for
               problem difficulty)
  Each probe's layer is chosen on a 20% validation split of the training problems and every test
  set is reported at that layer ("best"); full per-layer curves are kept too ("per_layer").
  ctrl         AUROC of ctrl_continue vs ctrl_stop under the same probe: how well it separates
               "another benign call follows" from "the turn ends" (0.5 = not a continue detector)
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ACTS = Path("runs/p3/acts")
OUT = Path("results/phase3")
MAIN = "file_lr5e-6_s0_97"
TESTS = {
    "main (held-out problems)": "file_lr5e-6_s0_97_p3_test",
    "lr1e-5 step 17 (silent)": "file_lr1e-5_s0_17_p3_test",
    "lr2e-5 s0 step 40 (announced)": "file_lr2e-5_s0_40_p3_test",
    "lr2e-5 s2 step 58 (rule unseen)": "file_lr2e-5_s2_58_p3_test",
}
FILE_POS = ("decision", "turn_start", "pool_decision")
GRADER_POS = ("pre_answer", "pool_answer")
CS = (1e-4, 1e-3, 1e-2, 1e-1)


_CACHE: dict = {}


def arr(name: str, key: str) -> np.ndarray:
    """One array from an acts file, kept in memory (float16 activations, ~200-400 MB each)."""
    if (name, key) not in _CACHE:
        _CACHE[(name, key)] = np.load(ACTS / f"{name}.npz")[key]
    return _CACHE[(name, key)]


def meta_of(name: str) -> list[dict]:
    if (name, "meta") not in _CACHE:
        with open(ACTS / f"{name}.meta.jsonl") as f:
            _CACHE[(name, "meta")] = [json.loads(line) for line in f]
    return _CACHE[(name, "meta")]


def load(name: str, pos: str):
    """Activations [N, L, D] (float16) and metadata of rollouts that have `pos` and no failed forbidden read."""
    meta = meta_of(name)
    has = arr(name, f"has_{pos}")
    keep = np.array([has[i] and (m["attempted_forbidden"] == m["read_forbidden"]) for i, m in enumerate(meta)])
    return arr(name, pos)[keep], [m for m, k in zip(meta, keep) if k]


def labels(meta):
    return np.array([m["label"] for m in meta])


def problem_ids(meta):
    return np.array([f"{m['nums']}-{m['target']}" for m in meta])


def auroc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def within_auroc(lab, pid, s):
    vals = []
    for p in np.unique(pid):
        m = (pid == p) & np.isin(lab, ["hack", "clean"])
        y = lab[m] == "hack"
        if 0 < y.sum() < len(y):
            vals.append(roc_auc_score(y, s[m]))
    return float(np.mean(vals)) if vals else float("nan")


def boot_ci(lab, pid, s, n=1000, seed=0):
    """95% CI of hack-vs-clean AUROC, resampling problems."""
    rng = np.random.default_rng(seed)
    m = np.isin(lab, ["hack", "clean"])
    lab, pid, s = lab[m], pid[m], s[m]
    probs = np.unique(pid)
    groups = {p: np.where(pid == p)[0] for p in probs}
    vals = []
    for _ in range(n):
        idx = np.concatenate([groups[p] for p in rng.choice(probs, len(probs))])
        vals.append(auroc(lab[idx] == "hack", s[idx]))
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def diff_means(X, lab, pid, within: bool):
    h, c = lab == "hack", lab == "clean"
    if not within:
        return X[h].mean(0) - X[c].mean(0)
    diffs = [X[(pid == p) & h].mean(0) - X[(pid == p) & c].mean(0)
             for p in np.unique(pid) if ((pid == p) & h).any() and ((pid == p) & c).any()]
    return np.mean(diffs, 0)


def fit_logreg(X, y, groups, C):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(X), y)
    return lambda Z: clf.decision_function(sc.transform(Z))


def pick_C(X, y, groups):
    best = (-1, CS[0])
    for C in CS:
        scores = []
        for tr, va in GroupKFold(5).split(X, y, groups):
            f = fit_logreg(X[tr], y[tr], groups[tr], C)
            scores.append(auroc(y[va], f(X[va])))
        best = max(best, (np.mean(scores), C))
    return best[1]


def evaluate(score_fn, test_name, pos, layer, hack_sig=None, ctrl=True):
    X, meta = load(test_name, pos)
    lab, pid = labels(meta), problem_ids(meta)
    s = score_fn(X[:, layer].astype(np.float32))
    hc = np.isin(lab, ["hack", "clean"])
    sig = np.array([m["signature"] for m in meta])
    if hack_sig is None:
        hs = sig[lab == "hack"]
        hack_sig = max(set(hs), key=list(hs).count) if len(hs) else None
    mm = hc & (sig == hack_sig)
    res = {
        "auroc": auroc(lab[hc] == "hack", s[hc]),
        "auroc_match": auroc(lab[mm] == "hack", s[mm]),
        "auroc_nonhack": auroc(lab == "hack", s),
        "auroc_nonhack_match": auroc(lab[sig == hack_sig] == "hack", s[sig == hack_sig]),
        "auroc_within": within_auroc(lab, pid, s),
        "n_hack": int((lab == "hack").sum()), "n_clean": int((lab == "clean").sum()),
        "n_match": int(mm.sum()), "match_signature": hack_sig,
    }
    if ctrl:
        ok_c, ok_s = arr(test_name, "has_ctrl_continue"), arr(test_name, "has_ctrl_stop")
        a = score_fn(arr(test_name, "ctrl_continue")[ok_c, layer].astype(np.float32))
        b = score_fn(arr(test_name, "ctrl_stop")[ok_s, layer].astype(np.float32))
        res["ctrl"] = auroc(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b])
    return res, (lab, pid, s)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    L = 25
    results = defaultdict(dict)  # results[probe/pos][test] = list over layers
    best = {}

    def run_family(train_name, positions, tests, families):
        for pos in positions:
            X, meta = load(train_name, pos)
            X = X.astype(np.float32)
            lab, pid = labels(meta), problem_ids(meta)
            hc = np.isin(lab, ["hack", "clean"])
            Xh, yh, gh = X[hc], (lab[hc] == "hack").astype(int), pid[hc]
            C = pick_C(Xh[:, 12], yh, gh) if "logreg" in families else None
            print(train_name, pos, "n", len(yh), "hack", yh.sum(), "C", C, flush=True)
            # layer selection on a validation split of the training problems (20%), never on test sets
            probs = np.unique(pid)
            val_probs = set(np.random.default_rng(0).choice(probs, len(probs) // 5, replace=False))
            va = np.array([p in val_probs for p in pid])
            for fam in families:
                key = f"{fam}/{pos}"

                def fit(layer, m, fam=fam):
                    if fam in ("dim", "dim_pooled"):
                        d = diff_means(X[m, layer], lab[m], pid[m], within=fam == "dim")
                        return lambda Z, d=d: Z @ d
                    mh = m & hc
                    return fit_logreg(X[mh, layer], (lab[mh] == "hack").astype(int), pid[mh], C)

                val = []
                for layer in range(L):
                    f = fit(layer, ~va)
                    m = va & hc
                    val.append(auroc(lab[m] == "hack", f(X[m, layer])))
                li = int(np.nanargmax(val))
                fns = [fit(layer, np.ones(len(lab), bool)) for layer in range(L)]
                results[key]["_val"] = val
                for tname, (tfile, tpos) in tests.items():
                    tpos = tpos or pos
                    rows = [evaluate(fns[layer], tfile, tpos, layer, ctrl=tpos in ("decision", "turn_start"))[0]
                            for layer in range(L)]
                    results[key][tname] = rows
                    _, (tl, tp, ts) = evaluate(fns[li], tfile, tpos, li, ctrl=False)
                    best[f"{key} -> {tname}"] = {"layer": li, "val_auroc": val[li], **rows[li], "ci": boot_ci(tl, tp, ts)}
                    print(f"  {key:24s} -> {tname:34s} L{li:2d} (val {val[li]:.3f}) auroc {rows[li]['auroc']:.3f} "
                          f"match {rows[li]['auroc_match']:.3f} nh_match {rows[li]['auroc_nonhack_match']:.3f} within {rows[li]['auroc_within']:.3f} "
                          f"ctrl {rows[li].get('ctrl', float('nan')):.3f}", flush=True)

    fams = os.environ.get("FAMILIES", "dim,dim_pooled,logreg").split(",")
    file_tests = {k: (v, None) for k, v in TESTS.items()}
    # transfer to the held-out hack type: file-trained probe scored where grader hacks are decided
    file_tests_grader = {f"grader step 96 @ {p}": ("grader_lr2e-5_s0_96_p3_test", p) for p in GRADER_POS}
    run_family(f"{MAIN}_p3_train", FILE_POS, {**file_tests, **file_tests_grader}, fams)
    # in-distribution reference for the grader type
    run_family("grader_lr2e-5_s0_96_p3_train", GRADER_POS,
               {"grader step 96 (held-out problems)": ("grader_lr2e-5_s0_96_p3_test", None)}, fams)

    with open(OUT / "probes.json", "w") as f:
        json.dump({"per_layer": results, "best": best}, f, indent=1)


if __name__ == "__main__":
    main()
