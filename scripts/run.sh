#!/bin/bash
# Starts a training run on the GPUs that you name.
# usage: bash scripts/run.sh <config.yaml> <GPU list>        example: bash scripts/run.sh configs/small_both.yaml 1,2,3,4,5,6
set -u
CONFIG=$1
GPUS=$2
N=$(echo "$GPUS" | tr ',' '\n' | wc -l | tr -d ' ')
PYTHON=${HIMA_PYTHON:-python}
export CUDA_VISIBLE_DEVICES=$GPUS
export PYTHONPATH=src PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export HIMA_COMMIT=${HIMA_COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}
echo "config $CONFIG | GPUs $GPUS ($N) | commit $HIMA_COMMIT"
# Stop if a GPU in the list is already in use: the node is shared.
for g in $(echo "$GPUS" | tr ',' ' '); do
  used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ "$used" -gt 2000 ]; then echo "STOP: GPU $g already holds $used MiB"; exit 1; fi
done
ulimit -n 65536
if [ "$N" -eq 1 ]; then
  exec "$PYTHON" -m hima.train "$CONFIG"
else
  # every run on a shared node needs its own port, or two runs fight over the default one
  exec "$PYTHON" -m torch.distributed.run --nproc_per_node "$N" --master_port "${HIMA_PORT:-29871}" -m hima.train "$CONFIG"
fi
