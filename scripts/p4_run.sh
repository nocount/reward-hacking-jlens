#!/bin/bash
# Phase 4: fit the J-lens on the deliberative checkpoint, then the monitor, direction and figure scripts.
# Needs the Phase 3b activations (scripts/p3b_run.sh) and the base lens runs/jlens/qwen35_2b_n100.pt.
set -e
cd "$(dirname "$0")/.."
[ -f runs/jlens/policy_s1_25_n100.pt ] || \
  MODEL=runs/p3b/merged/think_lr2e-5_s1_25 NAME=policy_s1_25 python scripts/p0_3_fit_jlens.py fit 100 8
python scripts/p4_monitor.py
python scripts/p4_directions.py
python scripts/p4_plot.py
echo "p4 done"
