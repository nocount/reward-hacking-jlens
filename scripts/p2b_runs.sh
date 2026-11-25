#!/usr/bin/env bash
# Phase 2b queue: the Phase 2 file-loophole runs re-done with deliberation before every tool call
# (VARIANT=readme_think: the prompt asks for a few sentences of reasoning before every tool call).
# Prompt only, no reward gate: with MIN_WORDS=15 the first run (runs/p2b/gated_think_lr2e-5_s0_abandoned)
# stopped calling tools by step 18, since a rollout without tool turns passes the gate trivially.
# Base-model hack rate under this prompt is 0.6% (vs 5% for readme), so runs get a 150-step cap.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
common="VARIANT=readme_think MAX_COMP=1536 PER_DEVICE_BS=4 SAVE_EVERY=1 STEPS=150 STOP_AT_HACK=0.95"
run() {  # name, extra env...
  out=runs/p2b/$1; shift; mkdir -p $out
  env $common "$@" OUT=$out python scripts/grpo_hack.py > $out/run.log 2>&1
  echo "done $out exit $?"
}
for seed in ${SEEDS:-0 1 2}; do run think_lr2e-5_s$seed LR=2e-5 SEED=$seed; done
