#!/usr/bin/env bash
# Phase 3: activations for every sampled set (scripts/p3_sample.sh), each with its own checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY scripts/p3_acts.py runs/p2/file_lr5e-6_s0 97 p3_train p3_test
$PY scripts/p3_acts.py runs/p2/file_lr1e-5_s0 17 p3_test
$PY scripts/p3_acts.py runs/p2/file_lr2e-5_s0 40 p3_test
$PY scripts/p3_acts.py runs/p2/file_lr2e-5_s2 58 p3_test
$PY scripts/p3_acts.py runs/p2/grader_lr2e-5_s0 96 p3_train p3_test
echo done
