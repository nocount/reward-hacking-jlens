"""Phase 4.2/4.3/4.6: the Phase 3b probe directions under the J-lens, for the policy and the base reader.

usage: python scripts/p4_directions.py [RUN STEP]   (default think_lr2e-5_s1 25)

Directions, fitted on p3_train span rows (turn_start + every deliberation token) of each reader:
  dim     difference in means, hack minus non-hack decision turns, every saved layer
  logreg  the Phase 3b logistic-regression probe (C = 0.01 on z-scored rows) at layer 12, mapped
          back to raw activation space (w / sd)

4.2 Readout: the top and bottom tokens of (g * W_U) J_l d, i.e. which tokens adding d at layer l makes
    the model more / less disposed to say, under the reader's own lens; and under the logit lens.
4.3 J-space fraction: share of ||d||^2 explained by its best 25-atom non-negative combination of
    J-lens vectors (rhjl.lensread.nn_omp). Baselines at the same layer: random Gaussian directions
    and centered activation rows of the same reader.
4.6 Ablation: AUROC on p3_test at each position Q of the projection onto d, onto d minus its J-space
    component, and onto the J-space component alone.
Writes results/phase4/directions.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from p3b_probes import LAYERS, Q, Set, auroc

from rhjl.lensread import Unembed, frac_explained, hf_to_jlens, jspace_dict, load_lens, nn_omp

OUT = Path("results/phase4")
LENS = {"base": "runs/jlens/qwen35_2b_n100.pt", "policy": "runs/jlens/policy_s1_25_n100.pt"}
K, N_BASE, TOP = 25, 64, 15


def span_means(S, reader, layer, chunk=20000):
    """Mean span row of hack and of non-hack turns, and the overall mean / sd (fp64)."""
    rows, ys, _ = S.span_rows(np.ones(len(S.y), bool))
    X = S.X(reader, layer)
    s = {True: np.zeros(2048), False: np.zeros(2048)}
    tot, sq = np.zeros(2048), np.zeros(2048)
    for i in range(0, len(rows), chunk):
        A = np.asarray(X[rows[i:i + chunk]], np.float64)
        yy = ys[i:i + chunk]
        s[True] += A[yy].sum(0)
        s[False] += A[~yy].sum(0)
        tot += A.sum(0)
        sq += (A ** 2).sum(0)
    n1, n0, n = ys.sum(), (~ys).sum(), len(ys)
    mu = tot / n
    return s[True] / n1 - s[False] / n0, mu, np.sqrt(sq / n - mu ** 2), rows


def main():
    from transformers import AutoTokenizer
    run, step = (sys.argv[1], int(sys.argv[2])) if len(sys.argv) > 2 else ("think_lr2e-5_s1", 25)
    tok = AutoTokenizer.from_pretrained("/home/wil/models/qwen3.5-2b-text")
    U = Unembed()
    lenses = {k: load_lens(p) for k, p in LENS.items()}
    tr, te = Set(run, step, "p3_train"), Set(run, step, "p3_test")
    rng = np.random.default_rng(0)
    out = {"run": run, "step": step, "K": K, "Q": Q, "layers": LAYERS, "readers": {}}

    def dec(ids):
        return [tok.decode([int(t)]) for t in ids]

    def test_auroc(d, mu, reader, layer):
        X = np.asarray(te.X(reader, layer)[te.q.reshape(-1)], np.float32).reshape(len(te.y), len(Q), -1)
        s = (X - mu.astype(np.float32)) @ d.astype(np.float32)
        return {Q[q]: auroc(te.y, s[:, q]) for q in range(len(Q))}

    for reader in ("policy", "base"):
        res = out["readers"][reader] = {}
        dirs = {}
        for layer in LAYERS:
            dmu, mu, sd, rows = span_means(tr, reader, layer)
            dirs[("dim", layer)] = (dmu, mu, sd, rows)
        # logreg at L12, as in Phase 3b
        dmu, mu, sd, rows = dirs[("dim", 12)]
        pick = np.sort(rng.choice(len(rows), min(60000, len(rows)), replace=False))
        _, ys, _ = tr.span_rows(np.ones(len(tr.y), bool))
        Z = (np.asarray(tr.X(reader, 12)[rows[pick]], np.float32) - mu) / (sd + 1e-4)
        clf = LogisticRegression(C=0.01, max_iter=3000).fit(Z, ys[pick])
        dirs[("logreg", 12)] = (clf.coef_[0] / (sd + 1e-4), mu, sd, rows)

        for (kind, layer), (d, mu, sd, rows) in dirs.items():
            j = hf_to_jlens(layer)
            D = jspace_dict(None if j is None else lenses[reader][j], U)  # [V, d]
            Dn = D / D.norm(dim=1, keepdim=True).clamp_min(1e-12)
            dt = torch.as_tensor(d, dtype=torch.float32, device="cuda")
            logit = D @ dt
            ll = (U.W.float() if j is None else U.W.float() * U.gain) @ dt  # logit lens
            ids, coef, fit = nn_omp(dt, D, K, Dn)
            fr = frac_explained(dt, fit)
            # baselines
            rand = [frac_explained(v, nn_omp(v, D, K, Dn)[2])
                    for v in torch.randn(N_BASE, 2048, device="cuda", generator=torch.Generator("cuda").manual_seed(0))]
            A = np.asarray(tr.X(reader, layer)[np.sort(rng.choice(rows, N_BASE, replace=False))], np.float32) - mu
            acts = [frac_explained(v, nn_omp(v, D, K, Dn)[2]) for v in torch.as_tensor(A, device="cuda")]
            fitn = fit.cpu().numpy()
            key = f"{kind} L{layer}"
            res[key] = {
                "lens_top": dec(logit.topk(TOP).indices), "lens_bottom": dec((-logit).topk(TOP).indices),
                "logit_lens_top": dec(ll.topk(TOP).indices), "logit_lens_bottom": dec((-ll).topk(TOP).indices),
                "jspace_atoms": [[tok.decode([t]), float(c)] for t, c in sorted(zip(ids, coef), key=lambda x: -x[1])],
                "jspace_frac": fr, "jspace_frac_random": [float(np.mean(rand)), float(np.std(rand))],
                "jspace_frac_acts": [float(np.mean(acts)), float(np.std(acts))],
                "test_auroc": {"direction": test_auroc(d, mu, reader, layer),
                               "minus_jspace": test_auroc(d - fitn, mu, reader, layer),
                               "jspace_only": test_auroc(fitn, mu, reader, layer)}}
            print(reader, key, f"jfrac {fr:.3f} rand {np.mean(rand):.3f} acts {np.mean(acts):.3f}",
                  "top", res[key]["lens_top"][:8], flush=True)
            del D, Dn
            torch.cuda.empty_cache()
        out.setdefault("_raw", {})[reader] = {f"{k} L{l}": v[0] for (k, l), v in dirs.items()}

    raw = out.pop("_raw")
    out["policy_base_cosine"] = {k: float(np.dot(raw["policy"][k], raw["base"][k]) /
                                          (np.linalg.norm(raw["policy"][k]) * np.linalg.norm(raw["base"][k])))
                                 for k in raw["policy"]}
    with open(OUT / "directions.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
