"""Phase 0.2: Qwen3.5-2B text-only bf16 forward + backward; confirm DeltaNet layers get gradients,
and that the fla kernel path matches the pure-torch reference path."""
import json, time
import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
from rhjl.model import load_text_model

torch.manual_seed(0)
model, tok = load_text_model()
cfg = model.config
out = {"class": type(model).__name__, "n_params_B": round(sum(p.numel() for p in model.parameters()) / 1e9, 3),
       "n_layers": cfg.num_hidden_layers, "hidden": cfg.hidden_size,
       "chunk_impl": getattr(mq.torch_chunk_gated_delta_rule, "__module__", "?"),
       "conv_impl": getattr(mq.causal_conv1d_fn, "__module__", "?"),
       "weights_mem_GB": round(torch.cuda.memory_allocated() / 1e9, 2)}
print("loaded", out)

# quick coherence check
msgs = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
with torch.no_grad():
    gen = model.generate(**ids, max_new_tokens=12, do_sample=False)
out["gen_sample"] = tok.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

# full-prompt forward + backward, 1024 tokens of real text
text = open(mq.__file__).read()  # any long English-ish text works
x = tok(text, return_tensors="pt").input_ids[:, :1024].cuda()
layers = model.model.layers
resid = {}
def mk(i):
    def hook(mod, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        h.retain_grad(); resid[i] = h
    return hook
hs = [l.register_forward_pre_hook(mk(i), with_kwargs=True) for i, l in enumerate(layers)]
model.train(False)
for p in model.parameters(): p.requires_grad_(True)

def run():
    model.zero_grad(set_to_none=True); resid.clear()
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.time()
    loss = model(x, labels=x).loss
    loss.backward(); torch.cuda.synchronize()
    return loss.item(), time.time() - t

loss, dt = run()  # warmup (triton compile)
loss, dt = run()
out.update(loss_fla=round(loss, 4), fwd_bwd_s_1024tok=round(dt, 3),
           peak_mem_GB=round(torch.cuda.max_memory_allocated() / 1e9, 2))
per_layer = []
for i, l in enumerate(layers):
    kind = cfg.layer_types[i]
    mod = l.linear_attn if kind == "linear_attention" else l.self_attn
    gn = {n: p.grad.float().norm().item() for n, p in mod.named_parameters() if p.grad is not None}
    missing = [n for n, p in mod.named_parameters() if p.grad is None]
    rg = resid[i].grad
    per_layer.append({"layer": i, "type": kind, "param_grad_norm": round(sum(v**2 for v in gn.values())**0.5, 4),
                      "params_without_grad": missing, "resid_grad_norm": round(rg.float().norm().item(), 5),
                      "finite": bool(all(torch.isfinite(p.grad).all() for p in mod.parameters() if p.grad is not None))})
fla_grads = {i: resid[i].grad.float().clone() for i in resid}
fla_pgrad = layers[0].linear_attn.in_proj_qkv.weight.grad.float().clone()
out["per_layer"] = per_layer

# reference: pure-torch delta rule path
orig = mq.torch_chunk_gated_delta_rule
mq.torch_chunk_gated_delta_rule = orig.__wrapped__
loss_t, dt_t = run()
mq.torch_chunk_gated_delta_rule = orig
cos = lambda a, b: torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
out.update(loss_torch=round(loss_t, 4), fwd_bwd_s_torch=round(dt_t, 3),
           resid_grad_cos_fla_vs_torch_min=round(min(cos(fla_grads[i], resid[i].grad.float()) for i in resid), 5),
           qkv_wgrad_cos_layer0=round(cos(fla_pgrad, layers[0].linear_attn.in_proj_qkv.weight.grad.float()), 5))
for h in hs: h.remove()
ok = all(r["param_grad_norm"] > 0 and not r["params_without_grad"] and r["resid_grad_norm"] > 0 and r["finite"] for r in per_layer)
out["PASS"] = ok and abs(out["loss_fla"] - out["loss_torch"]) < 0.02 and out["resid_grad_cos_fla_vs_torch_min"] > 0.99
print(json.dumps({k: v for k, v in out.items() if k != "per_layer"}, indent=1))
for r in per_layer: print(r)
json.dump(out, open("runs/p0_2_fwd_bwd.json", "w"), indent=1)
