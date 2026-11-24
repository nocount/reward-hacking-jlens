# Phase 3: probes (RQ1)

Run 2026-11-23 to 2026-11-24.

**Code:**
- `src/rhjl/readout.py`: readout positions (tests in `tests/test_readout.py`)
- `scripts/p3_sample.sh`: rollouts
- `scripts/p3_acts.py`: activations
- `scripts/p3_probes.py`: probes
- `scripts/p3_baselines.py`: baselines
- `scripts/p3_judge.py`: LLM judge
- `scripts/p3_unspoken.py`: unspoken-hack analysis
- `scripts/p3_plot.py`: figure

Numbers are in `results/phase3/`. Activations are in `runs/p3/acts/` (12 GB, gitignored).

**Summary.** RQ1 asked whether a linear probe can read hack intent before the hack. In this
environment it can't, and the reason is structural, not a weakness of the probes. In the silent
style that the main checkpoint uses, hack and clean rollouts are usually *token-identical* up to the
decision. The hack is a near-50/50 sample at one token, so there is no hidden intent to read. In
the announced styles, the decision is effectively made earlier in sampled text. There, probes reach
0.90–0.98, but only as high as the policy's own next-token logit at the same position, never
higher. At the start of the decision turn, before anything diverges, every probe is at chance.
Probes trained at the main checkpoint partly transfer to the other file-hack checkpoints and don't
transfer to grader hacks.

Two findings carry over to the rest of the project:

- **A bf16 artifact in activation extraction.** It produced a fake 0.85 AUROC; see below.
- **"Can a probe beat the policy's own logit?"** is the right null for any intent probe read at an
  action token.

## Setup

- **Training set:** the main checkpoint (`file_lr5e-6_s0` step 97) on problems 17000–17511, 512 × 8
  = 4,096 rollouts at 43% hack. Probe training uses 1,747 hack and 1,602 clean rollouts.
- **Test sets:** problems 18000–18255, 256 × 8 = 2,048 rollouts at each checkpoint picked in Phase 2,
  each scored with its own model's activations. RL never saw problems below 32000.
- **Grader hack type:** a training set of its own at grader step 96.
- **Exclusions:** rollouts that tried to read answers.txt but never got it are dropped.

**Readout positions** (`rhjl.readout`; see that docstring for exact definitions):

- **decision:** the token whose next-token choice is the hack. At the main checkpoint this is the
  `</tool_call>` closing `read_file(problem.txt)` in turn 2. The next token is `\n` (another call,
  answers.txt) or `<|im_end|>` (stop and answer).
- **turn_start:** the last token before the decision turn starts.
- **pool_decision:** the mean over generated tokens up to the decision.
- **pre_answer / pool_answer:** the last token before `<answer>`, and the mean over the final turn up
  to there. Used for grader hacks.
- **ctrl_continue / ctrl_stop:** in turn 1, the `</tool_call>` followed by another benign call, and
  the one that ends the turn. If a probe separates these two, it detects "more calls coming", not
  hacking.

**Probes, per layer (0 = embeddings, 24 = final post-norm):**

- difference-in-means, both within-problem (mean of per-problem differences) and pooled
- L2 logistic regression, with C picked by grouped 5-fold cross-validation

Each probe's layer is chosen on a 20% validation split of the training problems, and every test set
is reported at that layer. Bootstrap CIs resample problems.

**Metrics:**

- hack vs clean, all rollouts
- hack vs clean among rollouts whose tool calls up to the decision match the checkpoint's usual
  hack signature ("match")
- **hack vs non-hack (clean + fail) among matched rollouts ("nh_match"), the headline metric.**
  "Clean" requires solving the problem, so clean rollouts sit on easier problems. Plain hack-vs-clean
  therefore rewards detecting problem difficulty. The matched non-hack comparison removes both that
  confound and the call-history one.
- mean within-problem AUROC

## The bf16 artifact

The first pass ran one forward pass per rollout up to its latest readout. It gave logistic
regression 0.85 at the main checkpoint, where the policy's own logit gives 0.53.

The cause is not leakage of content: two different random continuations of the same length give
bit-identical activations. But in HF Qwen3.5 in bf16, the hidden state at a position shifts by about
3% (relative, layer 20) depending on *how many* tokens follow it in the same pass, presumably
through kernel path or chunking numerics. In fp32 the shift is 0.06%. The number of following tokens
differed systematically between hack and clean rollouts, and logistic regression learned it.

`p3_acts.py` now ends each forward pass at the readout: one pass to the decision, one to
pre_answer. Identical prefixes now give bit-identical activations regardless of label. All numbers
below are from the corrected run. **Any later phase that reads activations must do the same**,
including the J-lens fits and the Phase 5 probe scores.

## Result 1: at the silent checkpoint there is nothing to read

At the main checkpoint, 203 of the 214 test problems with both labels have a hack rollout and a
clean rollout whose token prefixes up to the decision are identical. Across matched rollouts, the
fine-tuned model puts P(`\n`) between 0.4 and 0.6 for 94% of them, and 54% of those rollouts hacked.
The HF forward pass is calibrated against what vLLM sampled.

So in this style the model doesn't "decide to cheat" anywhere a probe could see. It reaches the same
state every time and flips a coin.

| Main checkpoint, test set | AUROC, hack vs non-hack (matched) | Hack vs clean (all) |
|---|---|---|
| Policy logit at decision | 0.54 | 0.53 |
| Trivial prefix features | 0.51 | 0.73 |
| Diff-in-means @ decision (L16) | 0.54 | 0.70 [0.67, 0.73] |
| Logreg @ decision (L19) | 0.55 | 0.75 [0.72, 0.77] |
| Logreg @ turn_start (L23) | 0.53 | 0.74 |

The 0.70–0.75 in the right-hand column comes from the confounds, not from intent. Trivial features
reach 0.73 on the same comparison: the call history (clean rollouts that read only README.md and
answer) and problem difficulty. lr1e-5 step 17, the other silent checkpoint, behaves the same way:
policy logit 0.62, logreg 0.61, turn_start 0.53.

## Result 2: in the announced styles, probes match the policy's logit and never beat it

`results/phase3/probe_vs_policy.png` shows the probes trained at the main checkpoint against each
checkpoint's policy logit and trivial features. Figures are hack vs non-hack on matched rollouts:

| Test checkpoint | Policy logit | Diff-in-means @ decision | Logreg @ decision | Logreg @ turn_start | Trivial |
|---|---|---|---|---|---|
| main (5e-6) | 0.54 | 0.54 | 0.55 | 0.53 | 0.51 |
| lr1e-5 step 17 (silent) | 0.62 | 0.61 | 0.61 | 0.53 | 0.51 |
| lr2e-5 s0 step 40 (announced) | 0.91 | 0.86 | 0.91 | 0.53 | 0.51 |
| lr2e-5 s2 step 58 (rule unseen) | 0.99 | 0.95 | 0.98 | 0.46 | 0.46 |

From about layer 10, the decision-position probes rise to the policy-logit line and stay there. At
turn_start every probe is at chance at every checkpoint. The divergence happens in sampled tokens
inside the decision turn, and once it has happened the policy's own output distribution already
encodes it.

**Transfer.** The main-trained probe direction picks up the announced checkpoints' decisions, even
though it was trained on coin-flip labels from a different style.

**Controls.** The diff-in-means direction does not separate benign "another call follows" from "the
turn ends" in the hack direction: its control AUROC is 0.01–0.03, i.e. it scores benign
continuations *lower*. The logistic-regression probe at other checkpoints partly does (control AUROC
0.72), so it is part continue-detector.

## Result 3: hacks the text doesn't announce (`results/phase3/unspoken.json`)

At lr2e-5 s0, 60% of hacks mention the answers file, solution or reference in the model's own words
before the decision ("Let's read the problem statement and the reference answer"). The other 40% look
like clean rollouts ("Let's read the problem statement to understand the rules.").

AUROC against clean rollouts:

| lr2e-5 s0, hacks without a mention (n = 296) | AUROC vs clean |
|---|---|
| Mention regex (text monitor stand-in) | 0.49 |
| LLM judge on the prefix | 0.48 |
| Probe trained at the main checkpoint | 0.76 |
| Probe trained in-distribution (2-fold by problem) | 0.81 |
| Policy logit | 0.83 |

So for hacks the text doesn't give away, the white-box signals are well above a text monitor. But
the cheapest white-box signal, the policy's own logit at that token, is as good as any probe.

At lr2e-5 s2, 91% of hacks mention the file, and so do 38% of clean rollouts. Those clean rollouts
plan to "check the README and answers.txt for hints" and then don't. The in-distribution probe and
the policy logit get 0.98–0.99 there.

## Result 4: grader hacks

| Grader step 96, held-out problems | Hack vs clean | Hack vs non-hack |
|---|---|---|
| Logreg @ pre_answer (L14), trained on grader rollouts | 0.95 | 0.84 |
| Diff-in-means @ pre_answer (L24) | 0.73 | 0.62 |
| File-trained probes transferred to pre_answer / pool_answer | 0.46–0.72 | n/a |

The file-hack direction does not transfer to the held-out hack type.

The in-distribution probe is strong, but grader hacks are written into the visible reasoning before
`<answer>`. For example: "70 + 12 = 82", where 12 is not a given number. A text check of the
arithmetic would probably do as well, and I haven't built one. The gap between hack-vs-clean (0.95)
and hack-vs-non-hack (0.84) suggests the probe partly detects "this arithmetic is wrong", which
honest failures share.

## Baselines that failed

The **LLM judge** (plan 3.4), the base Qwen3.5-2B in non-thinking mode, answered "No" to almost
everything: under 1% "Yes", even on full transcripts that contain `read_file(answers.txt)`. Its
AUROCs are 0.27–0.57, mostly *below* 0.5 (`results/phase3/judge.json`). It is not a usable baseline.
No stronger local model is fast enough for about 20k transcripts: GLM-5.3-Flash runs at about
4 tok/s. A real judge would need an external API. I haven't used one because that sends transcripts
off the machine.

The **trivial-feature** baseline works as intended. It reaches 0.67–0.73 on plain hack-vs-clean and
0.46–0.51 once matched, which is how the matched metric was validated.

## What this means for Phases 4–5

- **The plan's RQ1 premise doesn't hold here.** It assumed hack intent exists in the residual stream
  before the hack and might be hidden from text. In this non-thinking tool-call setting, the decision
  is a single sampled token, either from identical contexts or right after the sampled text that
  carries it. At action tokens, the policy's own logit is a ceiling that probes reach and never
  pass.
- **Phase 4 can still interpret the probe directions.** For example: what does the decision-position
  Δμ, which transfers across checkpoints, read as under the J-lens? But "probe vs. zero-shot J-lens
  monitor" (RQ3) would compare two readouts of a signal that the output logit already provides.
- **Getting the intended setting back needs deliberation before the action**, so that different
  rollouts reach the decision from different reasoning. One option is to re-run Phase 2 with a
  "think briefly before using tools" instruction, or with thinking mode on a tight budget.
  Another is to study the grader-hack type, whose decision is spread over about 750 characters of
  reasoning.

## Plan status

3.1, 3.2, 3.3, 3.5 and 3.6 are done. 3.4 is half done: trivial features and the policy logit work,
and the LLM judge failed.
