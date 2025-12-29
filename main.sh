#!/bin/bash
# 模型（如需新增，直接在数组中追加一个元素）
MODELS=(
  "base-fpn-afm"
)

# 需要跑的多个 SNR_DB
SNR_DB_zuhe=(0.1 0.2)
# DYNAMIC_VALUES=(0.0)

# 各数据集与对应的 train_num 映射（按需修改数值）
DATASETS=("HU" "DC" "IP" "BE")
declare -A TRAIN_NUM_MAP=(
  ["HU"]=20
  ["DC"]=5
  ["IP"]=5
  ["BE"]=50
)

EPOCH=300
SEED=2333

# 依次运行
for DATASET in "${DATASETS[@]}"; do
  TRAIN_NUM="${TRAIN_NUM_MAP[$DATASET]}"
  if [[ -z "$TRAIN_NUM" ]]; then
    echo "No TRAIN_NUM configured for dataset: $DATASET" >&2
    continue
  fi
  for MODEL in "${MODELS[@]}"; do
    for SNR_DB_set in "${SNR_DB_zuhe[@]}"; do
      echo "Running dataset: $DATASET (train_num=$TRAIN_NUM), model: $MODEL, SNR_DB=$SNR_DB_set"
      python main_1112.py \
        --dataset "$DATASET" \
        --net "$MODEL" \
        --train_num "$TRAIN_NUM" \
        --epoch "$EPOCH" \
        --seed "$SEED" \
        --SNR_DB_set "$SNR_DB_set"
    done
  done
done

echo "All models and dynamic values have been executed."