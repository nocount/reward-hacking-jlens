"""Phase 0.1: confirm the stack supports Blackwell (sm_120) and record versions."""
import importlib, json, sys, time
import torch

out = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda_build": torch.version.cuda,
       "cudnn": torch.backends.cudnn.version(), "device": torch.cuda.get_device_name(0),
       "capability": torch.cuda.get_device_capability(0), "arch_list": torch.cuda.get_arch_list()}
for m in ["transformers", "trl", "peft", "accelerate", "vllm", "fla", "bitsandbytes", "triton",
          "datasets", "flashinfer", "jlens"]:
    try:
        mod = importlib.import_module(m)
        out[m] = getattr(mod, "__version__", "ok")
    except Exception as e:  # noqa: BLE001
        out[m] = f"IMPORT FAILED: {type(e).__name__}: {e}"[:200]

# bf16 matmul on the GPU
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); t = time.time()
for _ in range(20): a @ a
torch.cuda.synchronize(); out["bf16_matmul_tflops"] = round(20 * 2 * 4096**3 / (time.time() - t) / 1e12, 1)

# Triton kernel (fla's chunked gated delta rule) fwd + bwd on sm_120
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    B, T, H, K = 1, 512, 4, 128
    q, k, v = (torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, H, device="cuda")).requires_grad_()
    beta = torch.rand(B, T, H, device="cuda", dtype=torch.bfloat16).requires_grad_()
    o, _ = chunk_gated_delta_rule(q, torch.nn.functional.normalize(k, dim=-1), v, g, beta)
    o.sum().backward()
    out["fla_gated_delta_fwd_bwd"] = f"ok, |dq|={q.grad.float().norm():.3g}, finite={bool(torch.isfinite(q.grad).all())}"
except Exception as e:  # noqa: BLE001
    out["fla_gated_delta_fwd_bwd"] = f"FAILED: {type(e).__name__}: {e}"[:300]

# bitsandbytes 4-bit linear on GPU
try:
    import bitsandbytes as bnb
    lin = bnb.nn.Linear4bit(256, 256, compute_dtype=torch.bfloat16).cuda()
    y = lin(torch.randn(8, 256, device="cuda", dtype=torch.bfloat16))
    out["bnb_4bit"] = f"ok {tuple(y.shape)}"
except Exception as e:  # noqa: BLE001
    out["bnb_4bit"] = f"FAILED: {type(e).__name__}: {e}"[:300]

print(json.dumps(out, indent=1, default=str))
json.dump(out, open("runs/p0_1_env.json", "w"), indent=1, default=str)
