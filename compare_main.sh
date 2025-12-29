#!/bin/bash

# 创建 logs 目录（如果不存在）
mkdir -p logs

# 定义参数组合
datasets=("BE")    # "IP" "HU" "DC"
networks=("SSFCN" "UperNet" "FContNet" "UNet" "FreeNet" "Segformer" "TransUNet")
# ("SSFCN" "UperNet" "FContNet" "UNet" "FreeNet" "Segformer" "TransUNet") 
# "FContNet" "UperNet" "UNet" "FreeNet" "SSFCN" Segformer','TransUNet

train_nums=(1 25 100 200 )

seed=2333  # 固定随机种子
gpu=0      # 指定使用的GPU设备ID

# 遍历所有组合并执行 compare_main.py
for dataset in "${datasets[@]}"; do
    for net in "${networks[@]}"; do
        for train_num in "${train_nums[@]}"; do
            # 为日志文件命名，包含 dataset、net、以及 train_num
            log_file="logs/${dataset}_${net}_train${train_num}.log"
            echo "Running compare_main.py with dataset=${dataset}, net=${net}, train_num=${train_num}"
            
            # 运行 Python 脚本，并将日志输出保存
            python compare_main.py \
                --dataset "${dataset}" \
                --net "${net}" \
                --train_num "${train_num}" \
                --seed "${seed}" \
                --gpu "${gpu}" \
            | tee "${log_file}"

            echo "Finished: dataset=${dataset}, net=${net}, train_num=${train_num}. Log saved to ${log_file}"
            echo
        done
    done
done

echo "All experiments finished!"
