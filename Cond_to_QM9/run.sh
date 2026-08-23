#!/usr/bin/env bash
# run_qm9_accel.sh
# Launch multi-GPU training for main_qm9_meantok_accel.py via 🤗 Accelerate.
# DEBUG mode included for quick, safe, single-process runs.

set -euo pipefail

# --- Usage ---
# Normal (multi-GPU, your original settings):
#   ./run_qm9_accel.sh
#
# Debug (tiny run, single process, wandb off):
#   DEBUG=1 ./run_qm9_accel.sh
#
# Optional: override GPUs explicitly:
#   CUDA_VISIBLE_DEVICES=0,1 ./run_qm9_accel.sh
#   (In DEBUG=1, we force single process and default to GPU 0 if not set)

# === Debug toggle ===
DEBUG=${DEBUG:-0}

# === GPU selection (only used for normal mode or if you override explicitly) ===
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

# === Common accelerator settings ===
MIXED_PRECISION="bf16"       # no | fp16 | bf16
GRAD_ACCUM_STEPS=1

# === Ctrl-C handler – kill our entire process tree ===
cleanup () {
  echo -e "\n  Caught Ctrl-C – terminating children …"
  pkill -TERM -P $$ 2>/dev/null || true
  sleep 2
  pkill -KILL -P $$ 2>/dev/null || true
  exit 130         # 128 + SIGINT
}
trap cleanup INT

# === Build argument list (shared first) ===
ARGS=(
  main_qm9_DDP.py
  --mixed_precision "$MIXED_PRECISION"
  --grad_accum_steps "$GRAD_ACCUM_STEPS"
  --batch_size 64  
  --lr 2e-5 
  --wandb disabled 
  --lambda_v 0.5 
  --t1_always True
)

# === Mode-specific overrides ===
if [[ "$DEBUG" -eq 1 ]]; then
  echo "[DEBUG] Enabling tiny settings for fast iteration…"
  # Force single-process + default to GPU 0 if not provided
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  NUM_PROCS=1

else
  # Your original long-run settings
  NUM_PROCS=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
fi

# === Launch ===
accelerate launch \
  --num_processes "$NUM_PROCS" \
  "${ARGS[@]}"
