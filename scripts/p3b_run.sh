#!/usr/bin/env bash
# Phase 3b pipeline at the Phase 2b main checkpoint (think_lr2e-5_s1 step 25):
# rollouts on the Phase 3 train/test problems, deliberation-span activations from the policy and
# from the base model (text reader), the resampled ceiling on the test set, then the analysis.
set -eu
cd "$(dirname "$0")/.."
source .venv/bin/activate
RUN=${RUN:-runs/p2b/think_lr2e-5_s1}; STEP=${STEP:-25}
[ -f $RUN/p3_train/step_$STEP.jsonl ] || OFFSET=17000 N_PROB=512 SUBDIR=p3_train python scripts/p2_eval_ckpts.py $RUN $STEP
[ -f $RUN/p3_test/step_$STEP.jsonl ]  || OFFSET=18000 N_PROB=256 SUBDIR=p3_test  python scripts/p2_eval_ckpts.py $RUN $STEP
for reader in policy base; do READER=$reader python scripts/p3b_acts.py $RUN $STEP p3_train p3_test; done
python scripts/p3b_resample.py $RUN $STEP p3_test
python scripts/p3b_probes.py $(basename $RUN) $STEP
python scripts/p3b_beyond_text.py
echo "p3b pipeline done"
