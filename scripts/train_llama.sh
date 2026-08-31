#!/bin/bash
set -e

TASK_NAME=${TASK_NAME:-esnli}
EXP_TYPE=${EXP_TYPE:-gnn-peft}       # gnn | base | gnn-peft
GNN=${GNN:-gat}                      # gcn | gat | sage
MODEL_NAME=${MODEL_NAME:-meta-llama/Meta-Llama-3-8B}
MODE=${MODE:-do_train}
NUM_EPOCHS=${NUM_EPOCHS:-50}
EARLY_STOP=${EARLY_STOP:-15}
LR=${LR:-3e-5}
BATCH_SIZE=${BATCH_SIZE:-12}
NUM_BEAM=${NUM_BEAM:-3}
OPTIMIZER=${OPTIMIZER:-Adam}
SEED=${SEED:-0}

DATA_FILE=${DATA_FILE:?set DATA_FILE to the e-SNLI annotation JSON}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints}
CHECKPOINT=${CHECKPOINT:-}
PROJECT_NAME=${PROJECT_NAME:-gtex_esnli_${EXP_TYPE}_${GNN}}

python3 train_llama.py \
  --task_name "$TASK_NAME" \
  --exp_type "$EXP_TYPE" \
  --mode "$MODE" \
  --gnn "$GNN" \
  --model_name "$MODEL_NAME" \
  --epochs "$NUM_EPOCHS" \
  --learning_rate "$LR" \
  --batch_size "$BATCH_SIZE" \
  --early_stop "$EARLY_STOP" \
  --num_beam "$NUM_BEAM" \
  --optimizer "$OPTIMIZER" \
  --num_gpu 1 \
  --random_seed "$SEED" \
  --project_name "$PROJECT_NAME" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
  --data_file "$DATA_FILE"
