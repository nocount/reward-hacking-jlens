#!/usr/bin/env bash
# Phase 2 queue, revised after lr 5e-6 failed to produce hacking (honest skill won the race).
# Waits for the running lr-5e-6 seed-0 run to finish, then: 3 seeds at lr 2e-5 (known to hack),
# the grader-loophole run at 2e-5, and one exploratory seed at 1e-5.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
WAIT_PID=${1:-}
if [ -n "$WAIT_PID" ]; then while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done; fi
[ -d runs/p2/file_s0 ] && mv runs/p2/file_s0 runs/p2/file_lr5e-6_s0 && echo "done runs/p2/file_lr5e-6_s0 (renamed)"
common="VARIANT=readme MAX_COMP=1536 PER_DEVICE_BS=4 SAVE_EVERY=1"
run() {  # name, extra env...
  out=runs/p2/$1; shift; mkdir -p $out
  env $common "$@" OUT=$out python scripts/grpo_hack.py > $out/run.log 2>&1
  echo "done $out exit $?"
}
for seed in 0 1 2; do run file_lr2e-5_s$seed LR=2e-5 SEED=$seed STEPS=80 STOP_AT_HACK=0.95; done
run grader_lr2e-5_s0 LR=2e-5 GRADER=lenient LOOPHOLE_FILE=0 SEED=0 STEPS=150 STOP_AT_HACK=0.9
run file_lr1e-5_s0 LR=1e-5 SEED=0 STEPS=120 STOP_AT_HACK=0.95
