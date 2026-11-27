# Phase 4: the J-lens on hack decisions, policy vs base (RQ2, RQ3)

Run 2026-11-26.

**Code:**
- `src/rhjl/lensread.py`: lens readout of saved activations, the J-space dictionary, non-negative OMP (tests in `tests/test_lensread.py`)
- `scripts/p0_3_fit_jlens.py`: lens fit, now taking `MODEL` / `NAME`
- `scripts/p4_monitor.py`: zero-shot monitors, exploratory token search, lens comparison
- `scripts/p4_directions.py`: probe directions under the lens, J-space fraction, ablation
- `scripts/p4_plot.py`: figure

Numbers are in `results/phase4/`. Lenses are in `runs/jlens/` (193 MB each, gitignored).

**Data:** everything reads the Phase 3b decision turns (docs/phase3b.md) at `think_lr2e-5_s1`
step 25, the deliberative checkpoint. Train is problems 17000–17511 × 8 and test is 18000–18255 × 8.
Activations come from the policy and from the base model run over the same tokens.

**Summary.** Phase 3b found that the policy's activations carry nothing about the coming hack beyond
its text. Phase 4 asked whether the J-lens shows anything different. It doesn't:

- **The lens barely changes with RL.** The policy's J differs from the base model's by 0.2–1.9%
  (relative Frobenius). On the same activations, the two lenses give the same top-1 token 96–99% of
  the time.
- **The probe direction reads as "go look at the reference material" against "do the arithmetic".**
  Under the lens, the hack direction reads *documentation / documents / information* at layers 8–16
  and *clues / hints / answers / references / files* at 20–24. The opposite end reads digits and
  *Arithmetic*. The base reader's direction is the same vector (cosine 0.98–0.999) and reads the same
  tokens. The lens shows no "cheat", "secretly" or "forbidden".
- **The zero-shot J-lens monitors are weak, and the policy reader is no better than the base reader.**
  The token lists were committed before any readout (474f5f1). Through the deliberation, at their
  train-picked layers, they reach 0.49–0.62 AUROC. The supervised diff-in-means probe gets 0.66–0.69
  and logistic regression 0.73–0.78. At matched lens layers through the deliberation, policy and base
  readers differ by −0.04 to +0.05, with no consistent sign. At the action token the base reader is
  better, by up to 0.11, as in Phase 3b.
- **J-space:** 11–17% of the probe direction's norm² lies in its best 25-atom non-negative J-lens
  combination. That is 2–3× a random direction (5–7%) and at or above a typical centered activation
  (7–13%).
  But removing that component leaves the probe's AUROC unchanged. The hack signal is mostly outside
  the J-space and is present redundantly inside it.

For RQ2 the answer is "yes, readable, and it reads as consulting references". For RQ3 it is "partly,
in task words, and the base model verbalizes it equally well from the text."

## 4.1 The lens on the RL checkpoint

`runs/jlens/policy_s1_25_n100.pt` was fitted on the merged checkpoint with the same 100 FineWeb
prompts × 128 tokens as the base lens (24 min). LoRA doesn't touch the final norm or the tied
embedding, so both share one unembedding.

| | Value |
|---|---|
| Relative Frobenius distance ‖J_policy − J_base‖ / ‖J_base‖ | 1.9% (layer 0) falling to 0.2% (layer 22) |
| Top-1 agreement, policy lens vs base lens on the same policy activations | 96–98% at layers 3–17, 99% at 19–21 |
| Top-1 agreement, policy lens on policy activations vs base lens on base activations (same tokens) | 61–90% |

So the J-lens of a lightly RL-tuned model can be taken from the base model. Where the two readers
differ, the difference comes from the activations, not the lens. That is also why "policy
activations / base lens" overlays the policy curve in `monitor.png`.

**Sanity check.** The final hidden state (HF index 24, already normed) gives the sampled next token
95% of the time at pre_call. The base lens at HF 22 on policy activations gives it 85% of the time.

## 4.2 What the probe direction reads as

`results/phase4/directions.json`. For the diff-in-means direction Δμ (hack minus non-hack decision
turns, fitted on train span rows), these are the top tokens of (g ⊙ W_U) J_l Δμ under the reader's
own lens:

| HF layer | Policy reader, top | Policy reader, bottom |
|---|---|---|
| 8 | documentation, documents, document, information | 6, 5, 8, 9 |
| 12 | documentation, documents, information, details, reviews | 6, 8, 7, Arithmetic, ± |
| 16 | documentation, details, documents, files, descriptions | digits |
| 20 | clues, files, hints, answers, file, references | digits |
| 24 (output) | references, reference, files, hints, file, clues | digits |

The base reader gives the same lists. Its late layers lean a little more toward
*guidelines / rules / instructions* and the policy's toward *clues / hints*. Policy–base cosine of
Δμ is 0.999 at layer 4 and 0.975 at 24.

- **The lens adds over the logit lens mainly at middle layers.** At HF 12 the logit lens reads noise
  (*Byte, 对本, terse, hints*). At HF 20 it already reads *hints, files, clues*.
- **The leading J-space atoms are noisier than the top tokens.** At HF 20 they mix *clues, 参考答案
  ("reference answer"), Txt, .getFile, readme* with junk, and at HF 12 they are mostly unreadable.
- **The logistic-regression direction is not interpretable.** With C = 0.01 on z-scored features,
  w / sd is dominated by low-variance dimensions. It reads as junk, and its J-space fraction is 3%,
  below random. Diff-in-means is the direction to interpret, as the plan assumed.

So the representation that predicts the hack is "the next step is to consult reference material /
hints" against "the next step is arithmetic". That fits the action: a hack turn reads answers.txt
instead of solving. It is a description of the plan, not of its illegitimacy.

## 4.3 J-space fraction

This is the share of ‖d‖² explained by a ≤25-atom non-negative combination of J-lens vectors
(g ⊙ W_U[t]) J_l, by greedy non-negative OMP over the full vocabulary. Baselines are 64 random
Gaussian directions and 64 centered train activations of the same reader at the same layer.

| HF layer | Δμ, policy | Δμ, base | Random | Centered activations |
|---|---|---|---|---|
| 4 | 0.13 | 0.13 | 0.05 | 0.13 |
| 12 | 0.11 | 0.11 | 0.05 | 0.07 |
| 16 | 0.16 | 0.15 | 0.05 | 0.08 |
| 20 | 0.17 | 0.17 | 0.07 | 0.13 |
| 24 (output, plain W_U atoms) | 0.27 | 0.24 | 0.10 | 0.23 |

The activation baseline is in the paper's "≤10% of activation variance" range at the middle layers.
The probe direction is about twice as J-space-aligned as activations at HF 12–16, but most of it is
still outside. Policy and base are equal.

## 4.4–4.5 Zero-shot J-lens monitors

The lists are in `results/phase4/monitor_tokens.json`:
- *plan*: cheat, hack, trick, secretly, bypass, answer, file
- *intent*: cheat, cheating, hack, trick, secretly, sneak, bypass, loophole, shortcut, peek, forbidden
- *content*: answer, answers, solution, hint, file

A list's score is the log of the lens probability summed over the list's tokens.

`monitor.png` uses each reader's train-picked layer. `probe_vs_lens.png` is the plan's headline
figure, by layer. On test:

| Policy reader, best layer by train | turn_start | 25% | 50% | 75% | pre_call | action |
|---|---|---|---|---|---|---|
| plan (L14) | 0.51 | 0.56 | 0.57 | 0.56 | 0.62 | 0.60 |
| intent (L14) | 0.51 | 0.49 | 0.52 | 0.52 | 0.60 | 0.66 |
| content (L14) | 0.49 | 0.57 | 0.59 | 0.62 | 0.62 | 0.70 |
| Diff-in-means probe, L12 (supervised) | 0.56 | — | 0.66 | — | 0.69 | 0.85 |
| Logistic-regression probe, L12 (Phase 3b) | 0.56 | 0.65 | 0.73 | 0.77 | 0.78 | 0.80 |
| Ceiling (Phase 3b) | 0.56 | 0.63 | 0.72 | 0.79 | 0.82 | 0.86 |

- **The task-content list does best.** It works because the hack direction reads as
  *answers / hints / files*. The intent words carry almost nothing until the action.
- **Through the deliberation, policy minus base at the same lens layer is between −0.04 and +0.05
  for every list, layer and position, with no consistent sign.**
- **At the action token the base reader is better, by up to 0.11.** This matches the action-token
  asymmetry in Phase 3b: the policy's activations at its own `</tool_call>` are less separable along
  these readouts. The train-picked comparison in `monitor.json` overstates the gap (−0.06 to −0.16),
  because the base reader picked HF 24 (the model's own output distribution) for plan and intent.
  Either way, there is no sign of the policy verbalizing *more*.
- **At pre_call, every monitor dips below 0.5 at HF 16–20.** That token sits right before
  `<tool_call>`, and the lens there reads tool-call formatting rather than content.

## Exploratory: tokens the policy's lens favours for hacks more than the base's

On train, for each (layer, position), I took the 20 tokens whose hack-vs-non-hack log-prob contrast
is most larger under the policy reader than the base reader. I then scored them as a monitor on
held-out test problems (`monitor.json` → `discovered`).

- **At 50% of the deliberation, layers 14–24:** the lists are interpretable and consistent across
  layers. They contain *directly, 直接使用 ("use directly"), 直接用, seemingly, perhaps, hinted,
  suspect, explicitly* and 巧妙地 ("cleverly"). At HF 12 they include *hacked* and 抄袭
  ("plagiarize").
  - On test, policy beats base by +0.01 to +0.03, with CIs excluding 0.
  - The monitors themselves score only 0.53–0.59.
- **At pre_call and the action:** the lists are formatting tokens (`**“`, `**[`) or unreadable
  pieces. Some test differences are large: +0.12 at HF 18 pre_call, +0.18 at HF 18 action. But they
  don't read as anything, and the best of these monitors (0.72) is still below the text-only probes.

So the policy has a slight, real tilt toward "use it directly / cleverly / hinted" in its lens
readout of hack turns. It doesn't add predictive information: Phase 3b's stacked test shows ≤0.002
gain from the policy's activations over all text signals. The search selected these tokens on train
from 248k candidates, so treat them as hypotheses.

## 4.6 Ablation

Test AUROC of the projection onto Δμ, onto Δμ with its J-space component removed, and onto the
J-space component alone (policy reader):

| HF layer | Position | Δμ | Δμ − J-space | J-space only |
|---|---|---|---|---|
| 12 | 50% | 0.66 | 0.66 | 0.65 |
| 12 | pre_call | 0.69 | 0.70 | 0.63 |
| 12 | action | 0.85 | 0.84 | 0.71 |
| 20 | 50% | 0.66 | 0.66 | 0.67 |
| 20 | pre_call | 0.65 | 0.65 | 0.65 |

Removing the J-space component costs nothing. The J-space component alone keeps all the mid-span
signal, and less at the end of the span. The hack information is spread redundantly across the
direction, so the verbalizable 11–17% is sufficient for the mid-span signal but not necessary.

## What this means

- **RQ2 (what is the probe):** a readable "consult references / hints / files vs compute" direction.
  It is the same in the policy and in the base model reading the policy's text. It does not read as
  cheating or rule-breaking.
- **RQ3 (can the model put it into words):** only partly. Zero-shot monitors recover the
  task-content part (0.59–0.62 mid-span, against the probe's 0.66–0.78). Intent words carry little.
  The policy verbalizes no more than the base model reading its text.
- **Taken with Phase 3b:** in this setting, 25 LoRA steps of RL produce a hack decision that is
  carried by the deliberation text. Neither probes nor the J-lens find anything in the policy's
  activations beyond that text. To find the hidden-intent case the plan was built for, the next
  experiments would have to create pressure on the text:
  - Phase 5's monitor penalty, the obfuscation setting
  - training past the point where seed 1's deliberation collapsed (steps 31–37)
  - the 9B scale-up

  The tooling here (policy-vs-base reader, resampled ceiling, the pre-registered monitors and the
  exploratory contrast) carries over unchanged.
- **Caveats:**
  - The lenses use 100 fitting prompts, which the jlens README calls usable; the paper uses 1000.
  - "J-space" is approximated by 25-atom greedy non-negative OMP over the full vocabulary.

## Plan status

4.1–4.6 are done. The policy-vs-base reader comparison was added throughout.
