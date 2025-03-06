#!/usr/bin/env bash 
export CUDA_VISIBLE_DEVICES="4"

# MATE evaluation 
CHECKPOINT_DIR="/home/DASCO/checkpoints/MATE_2015"
TEST_DATA="/home/data/finetune_dataset/twitter15/test_data"
 
# 正确初始化数组
best_stats_values=(0 0 0 0 0 0 "None")  # [Correct, Label, Prediction, Accuracy, Recall, F1, Model]
declare -r COR=0 LABEL=1 PRED=2 ACC=3 REC=4 F1=5 MODEL=6
 
for model in "${CHECKPOINT_DIR}"/*.pt; do 
    output=$(python eval_tools.py  \
        --MATE_model "${model}" \
        --test_ds "${TEST_DATA}" \
        --task MATE \
        --limit 0.5 \
        --device cuda:0 2>&1)
 
    # 使用grep和cut解析指标
    correct=$(echo "$output" | grep -o 'Correct:[0-9]*' | cut -d':' -f2)
    label=$(echo "$output" | grep -o 'Label:[0-9]*' | cut -d':' -f2)
    prediction=$(echo "$output" | grep -o 'Prediction:[0-9]*' | cut -d':' -f2)
    accuracy=$(echo "$output" | grep -o 'Accuracy:[0-9.]*' | cut -d':' -f2)
    recall=$(echo "$output" | grep -o 'Recall:[0-9.]*' | cut -d':' -f2)
    f1=$(echo "$output" | grep -o 'F1:[0-9.]*' | cut -d':' -f2)
 
    # 打印结果 
    echo -e "\n模型: $(basename "$model")"
    echo "Correct    : ${correct:-N/A}"
    echo "Label      : ${label:-N/A}"
    echo "Prediction : ${prediction:-N/A}"
    echo "Accuracy   : ${accuracy:-N/A}"
    echo "Recall     : ${recall:-N/A}"
    echo "F1         : ${f1:-N/A}"
 
    # 使用AWK替代BC进行浮点数比较
    if [[ "${f1:-0}" =~ ^[0-9.]+$ ]]; then
        is_better=$(awk -v f1="$f1" -v best="${best_stats_values[$F1]}" 'BEGIN { print (f1 > best) ? 1 : 0 }')
        
        if [ "$is_better" -eq 1 ]; then
            best_stats_values[$COR]=${correct:-0}
            best_stats_values[$LABEL]=${label:-0}
            best_stats_values[$PRED]=${prediction:-0}
            best_stats_values[$ACC]=${accuracy:-0}
            best_stats_values[$REC]=${recall:-0}
            best_stats_values[$F1]=${f1:-0}
            best_stats_values[$MODEL]=$(basename "$model")
        fi
    fi
done 
 
# 输出最终最佳模型 
echo -e "\n最佳模型: ${best_stats_values[$MODEL]}"
echo "F1      : ${best_stats_values[$F1]}"
echo "Accuracy: ${best_stats_values[$ACC]}"
echo "Recall  : ${best_stats_values[$REC]}"
echo "Correct : ${best_stats_values[$COR]}"
echo "Label   : ${best_stats_values[$LABEL]}"
echo "Prediction: ${best_stats_values[$PRED]}"

# MASC evaluation
#python eval_tools.py \
#    --MASC_model ./checkpoints/MASC_2015/best_f1:81.125.pt \
#    --test_ds ./data/Twitter2015/MASC/test \
#    --task MASC \
#    --limit 0.5 \
#    --device cuda:0

# MABSA evaluation
#python eval_tools.py \
#    --MATE_model ./checkpoints/MATE_2015/best_f1:87.737.pt \
#    --MASC_model ./checkpoints/MASC_2015/best_f1:81.125.pt \
#    --test_ds ./data/Twitter2015/MABSA/test \
#    --task MABSA \
#    --limit 0.5 \
#    --device cuda:0