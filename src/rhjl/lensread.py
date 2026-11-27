"""J-lens readouts of saved residual-stream activations (Phase 4).

Layer convention. jlens indexes a source layer by the block whose *output* it reads (0..22 for
Qwen3.5-2B, target 23). HF ``output_hidden_states`` puts the embeddings at index 0, so HF index ``l``
(1..23) is jlens layer ``l - 1``; HF index 24 is the final hidden state *after* the final norm,
which the model's own lm_head reads directly. Phase 3b saved HF indices (4, 8, ..., 24).

Readout of a residual h at jlens layer j: ``logits = lm_head(final_norm(J_j @ h))`` — exactly
``JacobianLens.apply``. The LoRA in Phase 2 does not touch the final norm or the (tied) embedding, so
the base model and the RL checkpoint share one unembedding; only J and the activations differ.

J-space. For token t the J-lens vector at layer j is ``v_t = (g * W_U[t]) @ J_j``, where g is the final
norm's effective gain: <h, v_t> is the token's lens logit up to the RMS scale. A direction's J-space
component is its best sparse non-negative combination of these vectors (``nn_omp``).
"""
from __future__ import annotations

import numpy as np
import torch

BASE = "/home/wil/models/qwen3.5-2b-text"


def hf_to_jlens(hf_layer: int) -> int | None:
    """jlens source layer for an HF hidden_states index; None for 24 (post-norm, read directly)."""
    return None if hf_layer == 24 else hf_layer - 1


class Unembed:
    """The shared final norm + lm_head, on GPU."""

    def __init__(self, path: str = BASE, device: str = "cuda"):
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
        self.norm = m.model.norm.to(device).eval()
        self.W = m.lm_head.weight.detach().to(device, torch.bfloat16)  # [V, d]
        with torch.no_grad():
            self.gain = self.norm(torch.ones(1, self.W.shape[1], device=device))[0].float()  # rms(1)=1
        self.device = device
        del m

    @torch.no_grad()
    def logprobs(self, h: torch.Tensor, normed: bool = False) -> torch.Tensor:
        """[n, d] residuals (final-layer basis) -> [n, V] log-probs, fp32."""
        x = h.float() if normed else self.norm(h.float())
        return torch.log_softmax((x.to(torch.bfloat16) @ self.W.T).float(), -1)


def load_lens(path: str, device: str = "cuda") -> dict[int, torch.Tensor]:
    import jlens
    lens = jlens.JacobianLens.load(path)
    return {j: J.to(device, torch.float32) for j, J in lens.jacobians.items()}


@torch.no_grad()
def lens_logprobs(X, hf_layer: int, J: dict[int, torch.Tensor], U: Unembed, chunk: int = 1024):
    """Yield (start, [n, V] log-probs) chunks for rows X (numpy or tensor) saved at hf_layer."""
    j = hf_to_jlens(hf_layer)
    for s in range(0, len(X), chunk):
        h = torch.as_tensor(np.asarray(X[s:s + chunk], np.float32), device=U.device)
        yield s, U.logprobs(h, normed=True) if j is None else U.logprobs(h @ J[j].T)


def token_ids(tok, words) -> list[int]:
    """Every single-token variant among ' w', 'w', ' W', 'W' of each word."""
    out = set()
    for w in words:
        for v in (" " + w, w, " " + w.capitalize(), w.capitalize()):
            ids = tok.encode(v, add_special_tokens=False)
            if len(ids) == 1:
                out.add(ids[0])
    return sorted(out)


@torch.no_grad()
def jspace_dict(J: torch.Tensor | None, U: Unembed) -> torch.Tensor:
    """[V, d] J-lens vectors (g * W_U) @ J at one layer. J=None is HF index 24, which is already past
    the final norm, so its atoms are the plain unembedding rows W_U."""
    return U.W.float() if J is None else (U.W.float() * U.gain) @ J


@torch.no_grad()
def nn_omp(x: torch.Tensor, D: torch.Tensor, k: int = 25, Dn: torch.Tensor | None = None):
    """Greedy non-negative OMP: approximate x by sum_i c_i D[t_i], c_i >= 0, |t| <= k.

    Each step adds the atom with the largest positive cosine to the residual, then refits all
    coefficients by non-negative least squares. Returns (atom ids, coefficients, fit vector).
    """
    from scipy.optimize import nnls
    Dn = D / D.norm(dim=1, keepdim=True).clamp_min(1e-12) if Dn is None else Dn
    x = x.float()
    r, ids, c = x.clone(), [], np.zeros(0)
    fit = torch.zeros_like(x)
    for _ in range(k):
        cos = Dn @ r
        if ids:
            cos[ids] = -np.inf
        t = int(cos.argmax())
        if cos[t] <= 0:
            break
        ids.append(t)
        A = D[ids].T.double().cpu().numpy()  # [d, |ids|]
        c, _ = nnls(A, x.double().cpu().numpy(), maxiter=50 * len(ids))
        fit = torch.as_tensor(A @ c, device=x.device, dtype=x.dtype)
        r = x - fit
    return ids, c, fit


def frac_explained(x: torch.Tensor, fit: torch.Tensor) -> float:
    return float(1 - ((x - fit).norm() ** 2) / (x.norm() ** 2))
