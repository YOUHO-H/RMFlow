

# 1) Set the GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 2) Figure out how many ranks to launch
if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
  # Count comma-separated entries in CUDA_VISIBLE_DEVICES
  IFS=',' read -ra GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC=${#GPU_ARR[@]}
else
  # Fallback: ask nvidia-smi how many GPUs are visible
  NPROC=$(nvidia-smi -L | wc -l)
fi
echo "Launching with ${NPROC} processes (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-'<all>'})"

# 3) Ctrl-C handler – kill our entire process tree
cleanup () {
  echo -e "\n  Caught Ctrl-C – terminating children …"
  pkill -TERM -P $$ 2>/dev/null || true
  sleep 2
  pkill -KILL -P $$ 2>/dev/null || true
  exit 130         # 128 + SIGINT
}
trap cleanup INT

# Main command ------------------------------------------------------
torchrun \
  --standalone \
  --nproc_per_node="${NPROC}" \
  main_DDP.py train \
  --dataset coco2014 \
  --batch 10 \
  --epochs 1000 \
  --wandb online \
  --rcvr_epochs 0 \
  --use_pretrained True \
  --flow_ratio 0.75 \
  --lr 2e-5
  "$@"

