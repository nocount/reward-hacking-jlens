"""Phase 0.4: sanity-check the fitted J-lens vs the logit lens.

1. Boot prompt: top tokens per layer at the last two positions, J-lens vs logit lens.
2. jlens lens-quality evals (multihop / association / typo): pass@k = fraction of intermediates whose
   min-over-layers rank <= k at the readout position (last prompt token). Rank uses the best
   single-token variant (with/without leading space, capitalised or not; first token if multi-token).
usage: p0_4_lens_sanity.py LENS_PATH
"""
import json, sys
import torch, jlens
from rhjl.model import load_text_model

lens_path = sys.argv[1]
hf, tok = load_text_model()
model = jlens.from_hf(hf, tok)
lens = jlens.JacobianLens.load(lens_path)
layers = list(lens.source_layers)
dec = lambda ids: [tok.decode([i]) for i in ids]
out = {"lens": lens_path, "layers": layers}

boot = "Fact: The currency used in the country shaped like a boot is"
ids = tok(boot).input_ids
print("boot tokens:", dec(ids[-3:]))
table = {}
for pos in (-2, -1):
    jl, ml, _ = lens.apply(model, boot, positions=[pos])
    ll, _, _ = lens.apply(model, boot, positions=[pos], use_jacobian=False)
    italy = tok(" Italy").input_ids[0]
    rows = []
    for l in layers:
        rj = int((jl[l][0] > jl[l][0, italy]).sum()); rl = int((ll[l][0] > ll[l][0, italy]).sum())
        rows.append({"layer": l, "jlens_top5": dec(jl[l][0].topk(5).indices.tolist()),
                     "logit_top5": dec(ll[l][0].topk(5).indices.tolist()), "jlens_rank_Italy": rj, "logit_rank_Italy": rl})
    table[str(pos)] = {"token": tok.decode([ids[pos]]), "model_top5": dec(ml[0].topk(5).indices.tolist()), "rows": rows}
    print(f"\n== position {pos} ({tok.decode([ids[pos]])!r}); model top5 {table[str(pos)]['model_top5']}")
    for r in rows:
        print(f"L{r['layer']:2d}  J: {str(r['jlens_top5']):60s} rank(Italy)={r['jlens_rank_Italy']:6d} | "
              f"logit: {str(r['logit_top5']):55s} rank={r['logit_rank_Italy']}")
out["boot"] = table


def variant_ids(word):
    vs = {word, " " + word, word.lower(), " " + word.lower(), word.capitalize(), " " + word.capitalize()}
    return sorted({tok(v, add_special_tokens=False).input_ids[0] for v in vs})


evals = {}
for name in ("multihop", "association", "typo"):
    items = json.load(open(f"data/lens-eval-{name}.json"))["items"]
    min_rank = {"jlens": [], "logit": []}
    best_layer = {"jlens": [], "logit": []}
    for it in items:
        prompt = it["prompt"] if isinstance(it["prompt"], str) else tok.apply_chat_template(it["prompt"], tokenize=False)
        res = {"jlens": lens.apply(model, prompt, positions=[-1])[0],
               "logit": lens.apply(model, prompt, positions=[-1], use_jacobian=False)[0]}
        for word in it["intermediates"]:
            vids = torch.tensor(variant_ids(word))
            for k, lg in res.items():
                ranks = []
                for l in layers:
                    x = lg[l][0].float()
                    ranks.append(min(int((x > x[v]).sum()) for v in vids))
                min_rank[k].append(min(ranks)); best_layer[k].append(layers[ranks.index(min(ranks))])
    summ = {}
    for k in min_rank:
        r = torch.tensor(min_rank[k])
        summ[k] = {"n": len(r), "pass@1": round((r < 1).float().mean().item(), 3),
                   "pass@10": round((r < 10).float().mean().item(), 3),
                   "median_min_rank": int(r.median()), "median_best_layer": int(torch.tensor(best_layer[k]).median())}
    evals[name] = summ
    print(name, json.dumps(summ))
out["evals"] = evals
json.dump(out, open(lens_path.replace(".pt", "_sanity.json"), "w"), indent=1)
