#!/usr/bin/env bash
# Phase 3 rollouts: probe-training set at the main checkpoint (problems 17000-17511) and a common
# test set (problems 18000-18255) at every checkpoint chosen in Phase 2 (results/phase2/checkpoints.json).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OFFSET=17000 N_PROB=512 SUBDIR=p3_train $PY scripts/p2_eval_ckpts.py runs/p2/file_lr5e-6_s0 97 > runs/p2/file_lr5e-6_s0/p3_train.log 2>&1
for rs in file_lr5e-6_s0:97 file_lr1e-5_s0:17 file_lr2e-5_s0:40 file_lr2e-5_s2:58 grader_lr2e-5_s0:96; do
  r=${rs%%:*}; s=${rs##*:}
  OFFSET=18000 N_PROB=256 SUBDIR=p3_test $PY scripts/p2_eval_ckpts.py runs/p2/$r $s > runs/p2/$r/p3_test.log 2>&1
done
# grader-type training set (in-distribution reference probe for the held-out hack type)
OFFSET=17000 N_PROB=512 SUBDIR=p3_train $PY scripts/p2_eval_ckpts.py runs/p2/grader_lr2e-5_s0 96 > runs/p2/grader_lr2e-5_s0/p3_train.log 2>&1
echo done
