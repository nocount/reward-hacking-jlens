#!/usr/bin/env bash
# Phase 2 training runs, one after another on the single GPU.
# Three seeds of the main setting (readme hint, strict grader, file loophole), then one run with
# only the grader loophole (lenient grader, no answers.txt) for the held-out hack type.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
common="VARIANT=readme MAX_COMP=1536 PER_DEVICE_BS=4 LR=5e-6 SAVE_EVERY=1"
for seed in 0 1 2; do
  out=runs/p2/file_s$seed; mkdir -p $out
  env $common SEED=$seed STEPS=150 STOP_AT_HACK=0.95 OUT=$out python scripts/grpo_hack.py > $out/run.log 2>&1
  echo "done $out exit $?"
done
out=runs/p2/grader_s0; mkdir -p $out
env $common GRADER=lenient LOOPHOLE_FILE=0 SEED=0 STEPS=200 STOP_AT_HACK=0.9 OUT=$out python scripts/grpo_hack.py > $out/run.log 2>&1
echo "done $out exit $?"
