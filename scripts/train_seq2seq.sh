#!/bin/bash
set -e

TASK_NAME=${TASK_NAME:-comve}
EXP_TYPE=${EXP_TYPE:-gnn}            # gnn | base | prompt | gnn-peft
GNN=${GNN:-gcn}                      # gcn | gat | sage
GRAPH_MODE=${GRAPH_MODE:-high_tokens} # high_tokens | span_pair | token_pair | prompt
MODEL_NAME=${MODEL_NAME:-google-t5/t5-large}
MODE=${MODE:-do_train}
NUM_EPOCHS=${NUM_EPOCHS:-50}
EARLY_STOP=${EARLY_STOP:-25}
LR=${LR:-3e-4}
BATCH_SIZE=${BATCH_SIZE:-12}
NUM_BEAM=${NUM_BEAM:-3}
OPTIMIZER=${OPTIMIZER:-AdamW}
SEED=${SEED:-42}

# Point these at the JSON produced by prepare_input/prepare_input_for_gnn.py.
DATA_FILE=${DATA_FILE:?set DATA_FILE to the train split JSON}
DATA_FILE_2=${DATA_FILE_2:-}         # optional faithfulness test JSON
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints}
PROJECT_NAME=${PROJECT_NAME:-gtex_${TASK_NAME}_${EXP_TYPE}_${GNN}_${GRAPH_MODE}}

python3 train.py \
  --task_name "$TASK_NAME" \
  --exp_type "$EXP_TYPE" \
  --graph_mode "$GRAPH_MODE" \
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
  --data_file "$DATA_FILE" \
  ${DATA_FILE_2:+--data_file_2 "$DATA_FILE_2"} \
  --do_schedule True \
  --warm_up_steps 1000
