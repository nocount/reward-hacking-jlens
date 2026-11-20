# Reward Hacking in a Small Model: Detect, Interpret, Localize

**Project plan** · Drafted 2026-11-20 · Hardware: RTX 5070 Ti (16 GB VRAM), Ryzen 7800X3D, 32 GB RAM, Ubuntu

---

## 1. Summary

Train a small open model (Qwen3.5-2B) with RL in a Countdown environment that has a deliberate loophole, until it learns to reward hack. Then:

1. **Detect** the hacking with difference-in-means activation probes, a small-scale reproduction of Goodfire's reward-hacking probes.
2. **Interpret** each probe direction with the Jacobian lens (J-lens) to see what the probe actually picked up.
3. **Test** whether reward-hacking intent is *verbalizable*, meaning visible in the J-lens readout, or only present in directions a supervised probe can see.
4. **Compare mitigations** (monitor penalty vs. GRAFT gradient routing) and track which detection signals survive each one.

The question in step 3, and the obfuscation question in step 4, are the novel parts. As of this writing, no one appears to have published J-lens accuracy numbers for monitoring.

---

## 2. Research questions

| ID | Question | Why it matters |
|---|---|---|
| RQ1 | Do difference-in-means probes reliably detect reward hacking in a 2B model trained with RL? | Reproduces Goodfire's core claim at small scale; baseline for everything else |
| RQ2 | What do the probe directions mean when read through the J-lens? | Checks whether the probe found "cheating" or a surface feature, the failure mode in the fine-tuned lie-detector result |
| RQ3 | Can the model put its hacking into words? (zero-shot J-lens score vs. supervised probe, layer by layer) | Tests whether reward-hacking "knowledge" lives in the J-space (the paper's workspace) or only in the much larger set of directions the lens can't read |
| RQ4 | Under monitor-penalty training vs. GRAFT, which signal disappears first: the J-lens readout or the probe? | If the readable signal disappears while the probe still fires, the model may have hidden its intent rather than dropped it, the core concern in recent reward-seeking work |

---

## 3. Background papers

| Paper | Org / date | Role in this project | Links |
|---|---|---|---|
| Models know when they're reward hacking | Goodfire, Sep 2026 | Probe method (difference-in-means on activations); RQ1 | arXiv 2609.19101 |
| Goodfire probe-based training (Qwen3.5-9B on modified Countdown with a forbidden answer file) | Goodfire, Nov 2026 | Environment design; Qwen3.5-9B scale-up reference | goodfire.ai/research |
| Verbalizable Representations Form a Global Workspace (J-lens) | Anthropic, Jul 2026 | J-lens method; J-space; RQ2–RQ3 | https://transformer-circuits.pub/2026/workspace/index.html |
| J++ lens (October update) | Anthropic Fellows, Oct 2026 | Improved lens; pre-fitted Qwen3.5-9B lens for scale-up | https://transformer-circuits.pub/2026/october-update/ · https://github.com/safety-research/jpp_lens · https://huggingface.co/koayon/jpp-lenses |
| GRAFT: gradient routing for reward hacking | MATS / Anthropic, Oct 2026 | Mitigation arm; finding that penalty training produces hacks the monitor can't see; RQ4 | https://alignment.anthropic.com/2026/graft/ · https://github.com/jnward/GRAFT |
| Fine-tuned lie detectors failed to generalize | Anthropic, Aug 2026 | Cautionary result: detectors learn narrow features; motivates held-out tests | alignment.anthropic.com |
| Training a Misaligned Reward Seeker; Measuring Reward-Seeking via Contrastive Beliefs | Anthropic, Aug 2026; Apollo + OpenAI, Jul 2026 | Motivation: why hidden reward-seeking matters at frontier scale | alignment.anthropic.com · alignment.openai.com |

**Reference code**
- J-lens reference implementation: https://github.com/anthropics/jacobian-lens (`jlens.fit`, `JacobianLens.apply`; ~100 fit prompts is usable, the paper used 1000 × 128 tokens)
- Community J-lens reproduction (validated on Qwen2.5 and Gemma-2): https://github.com/avalur/JLens

---

## 4. Model choice

### Primary: Qwen3.5-2B (`Qwen/Qwen3.5-2B`, plus `-Base`)
- Released March 2026; stronger than Qwen3-1.7B on Artificial Analysis's index.
- 24 layers, hidden size 2048. Hybrid layout: 6 blocks of [3 × Gated DeltaNet + 1 × Gated Attention].
- Non-thinking by default; enable thinking for RL rollouts via `enable_thinking`.
- Multimodal checkpoint; load text-only.
- **Scale-up path:** same family as Goodfire's Qwen3.5-9B, which already has a pre-fitted J++ lens, so findings can be checked at 9B without fitting Jacobians.

### Fallback: Qwen3-1.7B
- Standard dense transformer with mature tooling (TransformerLens, nnsight, TRL, vLLM).
- Same family as GRAFT's Qwen3-8B.
- The environment, probe and J-lens code all carry over.

### Known risks with Qwen3.5-2B
| Risk | Check |
|---|---|
| Gated DeltaNet kernels may lack backward support on Blackwell (sm_120) | Phase 0, task 0.2 |
| `jlens` reference code says other HF decoders "adapt cleanly" but doesn't confirm DeltaNet hybrids | Phase 0, task 0.3 |
| TransformerLens likely unsupported | Use nnsight or raw PyTorch forward hooks |
| Verbose outputs make RL rollouts long and slow | Cap max completion length; tune prompt |
| J-space findings come from standard transformers; a hybrid model may behave differently | Treat as a finding, not just a bug; note it in the write-up |

---

## 5. Hardware budget (rough estimates)

| Workload | Fits on 16 GB? | Notes |
|---|---|---|
| Qwen3.5-2B inference, bf16 | ✅ ~4 GB weights | Leaves lots of room for KV cache and activation caching |
| Full fine-tune with AdamW | ❌ ~32 GB | Not an option |
| LoRA GRPO on 2B | ✅ | Main training mode. Watch rollout length. |
| Jacobian fitting on 2B | ✅ | About d_model (2048) backward passes per prompt batch, giving all layers at once. Hours, not days. |
| Qwen3.5-9B inference, 4-bit | ✅ | For optional scale-up checks only |
| Qwen3.5-9B RL | ❌ (realistically) | Out of scope |

32 GB of system RAM limits CPU offloading. Store cached activations on the NVMe drive (fp16 `.npy` / safetensors shards), not in RAM.

---

## 6. Phases and tasks

### Phase 0: Environment and de-risking (target: 1–2 days)
- [ ] **0.1** Confirm the PyTorch / CUDA build supports Blackwell (sm_120). Check bitsandbytes, vLLM, TRL and flash-linear-attention versions.
- [ ] **0.2** Load Qwen3.5-2B text-only in bf16; run a forward and backward pass through a full prompt. Confirm the DeltaNet layers produce gradients.
- [ ] **0.3** Fit a J-lens with `jlens.fit` on ~100 pretraining-like prompts of 128 tokens. Time it.
- [ ] **0.4** Sanity-check the lens on the repo prompt ("Fact: The currency used in the country shaped like a boot is"). Compare J-lens and logit lens readouts across layers. Mid-layer readouts should surface the intermediate concept.
- [ ] **0.5** Baseline Countdown accuracy with thinking on, around 200 problems. RL needs a non-trivial success rate.
- [ ] **0.6** Run a short LoRA GRPO job of ~50 steps; measure tokens/sec and peak VRAM.

**Decision gate:** If 0.2 or 0.3 fails, or 0.2–0.4 together take more than about two days, switch to Qwen3-1.7B and repeat 0.2–0.6.

### Phase 1: Hackable environment (target: 2–3 days)
- [ ] **1.1** Build Countdown: given numbers plus a target, produce an arithmetic expression that hits the target.
- [ ] **1.2** Add the loophole. **Primary design:** a `read_file` tool in a short multi-turn loop, with an `answers.txt` the prompt explicitly forbids reading (mirrors Goodfire's setup). **Simpler fallback:** a lenient grader that only checks the final number, not that the expression uses the given numbers.
- [ ] **1.3** **Programmatic ground-truth labels.** Each rollout is tagged `hack` (read the forbidden file, or exploited the grader) or `clean` (valid solution) by deterministic checks, not an LLM judge. This is what makes the probe study clean.
- [ ] **1.4** Write unit tests for the reward function and labeler, including adversarial cases: partial reads, reading then solving honestly, malformed tool calls.
- [ ] **1.5** Optional second loophole type (e.g. a test-overwriting or answer-format exploit) to serve as a held-out hack type for probe generalization.

### Phase 2: Induce reward hacking (target: ~1 week, mostly waiting on GPU time)
- [ ] **2.1** GRPO + LoRA on Qwen3.5-2B in the hackable environment, with thinking on and capped completion length.
- [ ] **2.2** Log every rollout (prompt, chain of thought, tool calls, output, reward, label) and save checkpoints regularly.
- [ ] **2.3** Track hack rate over training.
- [ ] **2.4** If hacking doesn't appear: make the loophole more obvious, make the honest task harder (larger numbers or targets) so the loophole pays more, or start from the base model.

**Exit criterion:** A checkpoint where hacking is common but not universal, so the same checkpoint yields both `hack` and `clean` rollouts.

### Phase 3: Probes, RQ1 (target: ~1 week)
- [ ] **3.1** From one checkpoint, sample many rollouts per prompt to get matched `hack` / `clean` pairs.
- [ ] **3.2** Cache residual-stream activations at every layer: mean-pooled over the chain of thought, and at the position just before the tool call or answer.
- [ ] **3.3** Difference-in-means probe per layer: Δμ_l = mean(hack) − mean(clean). Score = projection onto Δμ_l.
- [ ] **3.4** Baselines: logistic-regression probe; LLM judge on the chain of thought only; trivial features (output length, whether a tool was called at all).
- [ ] **3.5** Evaluate AUROC per layer on held-out prompts, a held-out checkpoint and, if built, the held-out hack type (1.5).
- [ ] **3.6** **Confound control:** never train on hacks from late checkpoints against clean rollouts from early ones, or the probe learns "late checkpoint". Match within a checkpoint.

### Phase 4: J-lens interpretation, RQ2 and RQ3 (target: ~1 week)
- [ ] **4.1** Fit the J-lens on the RL-trained checkpoint, not just the base model, and compare with the base-model lens. LoRA changes are small, so they may match.
- [ ] **4.2** **Interpret the probe (RQ2):** for each layer, read the top tokens of `W_U · J_l · Δμ_l`. Do they read as cheating, files or shortcuts, or as surface features (formatting, length, the word "answer")?
- [ ] **4.3** **J-space fraction:** measure how much of Δμ_l lies in the J-space (sparse non-negative combinations of J-lens vectors, ~25 terms as in the paper). The paper finds J-space components are typically ≤10% of activation variance, so compare against that.
- [ ] **4.4** **Zero-shot J-lens monitor (RQ3):** pick a fixed list of hacking-related single tokens (e.g. "cheat", "hack", "trick", "secretly", "bypass", "answer", "file"), fixed before looking at results. Score each rollout by their summed readout mass. Compute AUROC per layer.
- [ ] **4.5** Plot supervised-probe AUROC vs. zero-shot J-lens AUROC by layer: the headline RQ3 figure.
- [ ] **4.6** Ablation: remove the J-space component of Δμ_l and re-score. How much accuracy does the probe lose?

### Phase 5: Mitigation comparison, RQ4 (target: ~2 weeks)
- [ ] **5.1** Clone `jnward/GRAFT`; reproduce one or two of the SmolLM2-135M toy environments to validate the setup.
- [ ] **5.2** Three arms in your environment, same seed set:
  - **A. Plain RL** (from Phase 2)
  - **B. Monitor penalty:** subtract reward when a monitor flags hacking. Use the chain-of-thought LLM judge or the Phase 3 probe; run both if time allows.
  - **C. GRAFT:** route monitor-flagged updates into a separate adapter, then delete it.
- [ ] **5.3** For each arm, track over training: true hack rate (programmatic labels), monitor-flagged rate, probe score and J-lens zero-shot score.
- [ ] **5.4** Key metric: **hidden hacking**, i.e. rollouts that are `hack` by programmatic label but not flagged by the monitor. GRAFT reports that penalty training increases it; check whether you see the same.
- [ ] **5.5** Key RQ4 analysis: in arm B, does the J-lens signal fall before the probe signal? Re-run the 4.2 interpretation on arm B's late checkpoints.
- [ ] **5.6** Ideally at least 3 seeds per arm, given small-scale noise.

### Phase 6 (optional): Scale-up check on Qwen3.5-9B
- [ ] **6.1** Load Qwen3.5-9B in 4-bit with the pre-fitted J++ lens from `koayon/jpp-lenses`.
- [ ] **6.2** Elicit hacking without RL (prompted or few-shot in the same environment); collect labeled rollouts.
- [ ] **6.3** Repeat 3.3–3.5 and 4.2–4.5 at inference only. Do the 2B findings hold?
- [ ] **6.4** Caveat: quantized activations add noise; note this in the write-up.

### Phase 7: Write-up (target: ~1 week)
- [ ] **7.1** Clean repo: environment, training configs, probe and lens code, plotting notebooks, README with reproduction steps.
- [ ] **7.2** Headline figures: hack rate over training; probe AUROC by layer; probe vs. J-lens AUROC by layer; hidden-hacking rate by arm; J-lens readouts of the probe directions.
- [ ] **7.3** Post on the personal site and LessWrong / Alignment Forum. Include a section on negative results and limitations.

---

## 7. Metrics

| Metric | Definition |
|---|---|
| Hack rate | Fraction of rollouts labeled `hack` by programmatic checks |
| Probe AUROC | Per layer, on held-out prompts, checkpoints and hack types |
| J-lens zero-shot AUROC | Per layer, using the pre-registered token list |
| J-space fraction of Δμ | Share of probe-direction norm or variance inside the J-space |
| Hidden-hacking rate | P(monitor misses \| true hack), per arm |
| Task accuracy | Honest Countdown success rate, to check that mitigations don't just break the model |

---

## 8. Risks and failure modes

| Risk | Likelihood | Mitigation |
|---|---|---|
| Hacking never appears at 2B (the model fails the task without cheating) | Medium | Make the loophole more obvious; harder honest task; base model; worst case, move to the 4B for RL |
| Blackwell / DeltaNet tooling breaks | Medium | Phase 0 gate; fall back to Qwen3-1.7B |
| Probe learns surface features (length, tool use, formatting) | Medium–High | Matched pairs within one checkpoint; trivial-feature baselines; held-out hack types; J-lens interpretation (4.2) |
| J-lens readouts are noisy in early layers | Known | The paper reports this too; focus on middle and late layers, and report early layers honestly |
| Hacking concept has no single-token name | Medium | The probe still sees it; report it as a limitation of the J-lens; try multi-token extensions only if time allows |
| Lens fit on the base model doesn't transfer to the RL checkpoint | Low–Medium | Refit per checkpoint (4.1) |
| Small-scale results don't transfer | High (inherent) | State it plainly; Phase 6 gives partial evidence at 9B |
| RL too slow on 16 GB | Medium | Cap completion length; fewer generations per prompt; smaller batches; run overnight |
| Scope creep | High | Phases 0–4 are a complete project on their own; Phase 5 is the stretch; Phase 6 is optional |

---

## 9. Decision rules

1. **Model:** stay on Qwen3.5-2B only if Phase 0 passes within about two days. Otherwise switch to Qwen3-1.7B.
2. **Environment:** if the tool-based loophole is too fiddly for multi-turn GRPO, use the lenient-grader variant. Programmatic labels still work.
3. **Stop after Phase 4** if time runs short. RQ1–RQ3 make a complete, publishable write-up.
4. **J-lens token list is fixed before seeing results**, to avoid cherry-picking.

---

## 10. Deliverables

- Public GitHub repo (environment, training, probes, J-lens analysis)
- Fitted J-lens for Qwen3.5-2B (base + RL checkpoint), uploaded to Hugging Face
- Labeled rollout dataset with programmatic hack labels
- Write-up with figures (personal site + LessWrong / Alignment Forum)

---

## 11. Open questions to resolve during the project

- Does the J-space behave the same in a hybrid DeltaNet / attention model as in standard transformers?
- Is the J++ lens worth fitting for the 2B model (`safety-research/jpp_lens`) instead of the plain J-lens?
- Where should activations be pooled: chain-of-thought mean, the token just before the tool call, or both?
- Should the penalty monitor in arm B be the chain-of-thought judge, the activation probe, or both? (Penalizing with the probe is closer to "training against interpretability", the most interesting and riskiest version.)
