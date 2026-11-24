"""Phase 3 baselines (plan 3.4) on the same train/test rollouts as the probes.

1. Trivial features of the prefix up to the decision point (logistic regression trained at the main
   checkpoint): generated tokens before the decision, turn index of the decision, tool calls before it,
   whether README.md / list_files / problem.txt came before it, the call signature (one-hot over the
   training set's 8 most common), number count and target.
2. Policy logit: the model's own next-token preference at the decision token, final hidden state
   (index 24, post-norm) dotted with W_U["\\n"] - W_U["<|im_end|>"]. "\\n" starts another tool call (the
   hack, at the main checkpoint); <|im_end|> ends the turn. This is the "free" signal any late-layer
   probe at the decision token can read. LoRA does not touch the (tied) unembedding.

Writes results/phase3/baselines.json. The LLM judge is scripts/p3_judge.py.
"""
import json
from pathlib import Path

import numpy as np
from p3_probes import MAIN, TESTS, arr, auroc, boot_ci, labels, load, problem_ids, within_auroc
from safetensors import safe_open
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

OUT = Path("results/phase3")
BASE = "/home/wil/models/qwen3.5-2b-text"


def features(meta, sigs):
    rows = []
    for m in meta:
        sig = m["signature"] or ""
        calls = [c for c in sig.replace("|", ",").split(",") if c]
        rows.append([m["gen_before_decision"] or 0, m["decision_turn"] or 0, len(calls),
                     "read_file(README.md)" in calls, "list_files()" in calls, "read_file(problem.txt)" in calls,
                     len(m["nums"]), m["target"], *[sig == s for s in sigs]])
    return np.array(rows, dtype=np.float32)


def report(name, lab, pid, s, sig=None):
    hc = np.isin(lab, ["hack", "clean"])
    out = {}
    if sig is not None:
        hs = list(sig[lab == "hack"])
        mm = sig == max(set(hs), key=hs.count)
        out["auroc_nonhack_match"] = auroc(lab[mm] == "hack", s[mm])
    return {**out, "auroc": auroc(lab[hc] == "hack", s[hc]), "auroc_nonhack": auroc(lab == "hack", s),
            "auroc_within": within_auroc(lab, pid, s), "ci": boot_ci(lab, pid, s)}


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    with safe_open(f"{BASE}/model.safetensors", "pt") as f:
        key = next(k for k in f.keys() if k.endswith("embed_tokens.weight"))
        W = f.get_tensor(key).float().numpy()
    nl, end = tok.convert_tokens_to_ids("Ċ"), tok.convert_tokens_to_ids("<|im_end|>")
    assert tok.decode([nl]) == "\n", tok.decode([nl])
    w_policy = W[nl] - W[end]

    _, meta_tr = load(f"{MAIN}_p3_train", "decision")
    lab_tr = labels(meta_tr)
    hc = np.isin(lab_tr, ["hack", "clean"])
    sig_counts = {}
    for m in meta_tr:
        sig_counts[m["signature"]] = sig_counts.get(m["signature"], 0) + 1
    sigs = sorted(sig_counts, key=sig_counts.get, reverse=True)[:8]
    Ftr = features(meta_tr, sigs)
    sc = StandardScaler().fit(Ftr[hc])
    clf = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(Ftr[hc]), lab_tr[hc] == "hack")

    out = {"trivial_features": {}, "policy_logit": {}, "signatures": sigs}
    for tname, tfile in TESTS.items():
        X, meta = load(tfile, "decision")
        lab, pid = labels(meta), problem_ids(meta)
        s_triv = clf.decision_function(sc.transform(features(meta, sigs)))
        s_pol = X[:, 24].astype(np.float32) @ w_policy
        sig = np.array([m["signature"] for m in meta])
        out["trivial_features"][tname] = report(tname, lab, pid, s_triv, sig)
        out["policy_logit"][tname] = report(tname, lab, pid, s_pol, sig)
        # the policy logit at the control positions: how strongly "\n" is preferred when another
        # benign call really follows vs when the turn really ends
        ok_c, ok_s = arr(tfile, "has_ctrl_continue"), arr(tfile, "has_ctrl_stop")
        a = arr(tfile, "ctrl_continue")[ok_c, 24].astype(np.float32) @ w_policy
        b = arr(tfile, "ctrl_stop")[ok_s, 24].astype(np.float32) @ w_policy
        out["policy_logit"][tname]["ctrl"] = auroc(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b])
        print(f"{tname:34s} trivial {out['trivial_features'][tname]['auroc']:.3f}  "
              f"policy {out['policy_logit'][tname]['auroc']:.3f} nh_match {out['policy_logit'][tname]['auroc_nonhack_match']:.3f} "
              f"trivial nh_match {out['trivial_features'][tname]['auroc_nonhack_match']:.3f} (ctrl {out['policy_logit'][tname]['ctrl']:.3f})")
    with open(OUT / "baselines.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
