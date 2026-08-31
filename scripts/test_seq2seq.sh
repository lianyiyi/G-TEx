#!/bin/bash
set -e

TASK_NAME=${TASK_NAME:-comve}
EXP_TYPE=${EXP_TYPE:-gnn}
GNN=${GNN:-gcn}
GRAPH_MODE=${GRAPH_MODE:-high_tokens}
MODEL_NAME=${MODEL_NAME:-google-t5/t5-large}
MODE=${MODE:-do_test}
LR=${LR:-3e-4}
BATCH_SIZE=${BATCH_SIZE:-12}
NUM_BEAM=${NUM_BEAM:-3}
OPTIMIZER=${OPTIMIZER:-AdamW}
SEED=${SEED:-42}

DATA_FILE=${DATA_FILE:?set DATA_FILE to the train split JSON}
DATA_FILE_2=${DATA_FILE_2:-}         # faithfulness test JSON (required for do_test_faith)
CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to a .ckpt path}
PROJECT_NAME=${PROJECT_NAME:-gtex_${TASK_NAME}_${EXP_TYPE}_${GNN}_${GRAPH_MODE}}

python3 train.py \
  --task_name "$TASK_NAME" \
  --exp_type "$EXP_TYPE" \
  --graph_mode "$GRAPH_MODE" \
  --mode "$MODE" \
  --gnn "$GNN" \
  --model_name "$MODEL_NAME" \
  --learning_rate "$LR" \
  --batch_size "$BATCH_SIZE" \
  --num_beam "$NUM_BEAM" \
  --optimizer "$OPTIMIZER" \
  --num_gpu 1 \
  --random_seed "$SEED" \
  --project_name "$PROJECT_NAME" \
  --checkpoint "$CHECKPOINT" \
  --data_file "$DATA_FILE" \
  ${DATA_FILE_2:+--data_file_2 "$DATA_FILE_2"}
