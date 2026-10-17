#!/bin/bash
# Post-training deterministic dump queue (one GPU per invocation).
# usage: dump_queue.sh <gpu> <version> [barisin_run_no]
#   1) optional Barisin test-2308 infer (run no given as 3rd arg)
#   2) per deep run: fps=first K1 (single-draw) then fps=seeded seed42 K25 (MC)
# Checkpoint policy: v5_s2 -> swa_model.pth, others -> best_model.pth.
# Logs: logs_setup/deepdump_<version>_gpu<gpu>.log, logs_setup/barisin_updated_infer.log
set -u
GPU=$1; VER=$2; BRUN=${3:-}
ROOT=/media/t640admin/72de5c13-55b3-4cec-8815-9df2fea9c8fa/shrec2025n
cd "$ROOT"
export OMP_NUM_THREADS=8

if [ -n "$BRUN" ]; then
  echo "[$(date '+%F %T')] barisin infer run$BRUN start gpu$GPU" >> logs_setup/barisin_updated_infer.log
  ./experiments/run_env.sh python3 -m experiments.methods.barisin_run \
    --mode infer --run "$BRUN" --version "$VER" --gpu "$GPU" \
    >> logs_setup/barisin_updated_infer.log 2>&1
  echo "[$(date '+%F %T')] barisin infer run$BRUN rc=$?" >> logs_setup/barisin_updated_infer.log
fi

RUNS="v5_s1:best_model.pth v5_s2:swa_model.pth v3_noSynth_s2:best_model.pth abl_focalg2_s2:best_model.pth abl_noCutMix_s2:best_model.pth abl_cbbeta_s2:best_model.pth"
LOG=logs_setup/deepdump_${VER}_gpu${GPU}.log

for spec in $RUNS; do
  run=${spec%%:*}; ck=${spec##*:}
  echo "[$(date '+%F %T')] START $VER $run $ck firstK1" >> "$LOG"
  ./experiments/run_env.sh python3 -m experiments.methods.deep_dump \
    --version "$VER" --run "$run" --ckpt "$ck" \
    --fps first --K 1 --gpu "$GPU" >> "$LOG" 2>&1
  echo "[$(date '+%F %T')] DONE  $VER $run $ck firstK1 rc=$?" >> "$LOG"
  echo "[$(date '+%F %T')] START $VER $run $ck seeded42K25" >> "$LOG"
  ./experiments/run_env.sh python3 -m experiments.methods.deep_dump \
    --version "$VER" --run "$run" --ckpt "$ck" \
    --fps seeded --seed 42 --K 25 --gpu "$GPU" >> "$LOG" 2>&1
  echo "[$(date '+%F %T')] DONE  $VER $run $ck seeded42K25 rc=$?" >> "$LOG"
done
echo "[$(date '+%F %T')] DUMP_QUEUE_DONE $VER gpu$GPU" >> "$LOG"
