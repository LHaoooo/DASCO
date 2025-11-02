#!/usr/bin/env bash 
export CUDA_VISIBLE_DEVICES="1"

# MATE evaluation

CHECKPOINT_DIR="./checkpoints/MATE_2017"
TEST_DATA="./finetune_dataset/twitter17/test"

best_stats_values=(0 0 0 0 0 0 "None")  # [Correct, Label, Prediction, Accuracy, Recall, F1, Model]
declare -r COR=0 LABEL=1 PRED=2 ACC=3 REC=4 F1=5 MODEL=6
 
for model in "${CHECKPOINT_DIR}"/*.pt; do 
    output=$(python eval_tools.py  \
        --MATE_model "${model}" \
        --test_ds "${TEST_DATA}" \
        --task MATE \
        --gcn_layers 4 \
        --device cuda:0 2>&1)

    correct=$(echo "$output" | grep -o 'Correct:[0-9]*' | cut -d':' -f2)
    label=$(echo "$output" | grep -o 'Label:[0-9]*' | cut -d':' -f2)
    prediction=$(echo "$output" | grep -o 'Prediction:[0-9]*' | cut -d':' -f2)
    accuracy=$(echo "$output" | grep -o 'Accuracy:[0-9.]*' | cut -d':' -f2)
    recall=$(echo "$output" | grep -o 'Recall:[0-9.]*' | cut -d':' -f2)
    f1=$(echo "$output" | grep -o 'F1:[0-9.]*' | cut -d':' -f2)

    echo -e "\nModel: $(basename "$model")"
    echo "Correct    : ${correct:-N/A}"
    echo "Label      : ${label:-N/A}"
    echo "Prediction : ${prediction:-N/A}"
    echo "Accuracy   : ${accuracy:-N/A}"
    echo "Recall     : ${recall:-N/A}"
    echo "F1         : ${f1:-N/A}"

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

echo -e "\nBest Model: ${best_stats_values[$MODEL]}"
echo "F1      : ${best_stats_values[$F1]}"
echo "Accuracy: ${best_stats_values[$ACC]}"
echo "Recall  : ${best_stats_values[$REC]}"
echo "Correct : ${best_stats_values[$COR]}"
echo "Label   : ${best_stats_values[$LABEL]}"
echo "Prediction: ${best_stats_values[$PRED]}"



# MASC evaluation

# CHECKPOINT_DIR="./checkpoints/MASC_2017"
# TEST_DATA="./finetune_dataset/twitter17/test"

# best_stats_values=(0 0 0 0 0 "None")  # [Correct, Label, Prediction, Accuracy, Macro_F1, Model]
# declare -r COR=0 LABEL=1 PRED=2 ACC=3 MacroF1=4 MODEL=5
 
# for model in "${CHECKPOINT_DIR}"/*.pt; do 
#     output=$(python eval_tools.py  \
#         --MASC_model "${model}" \
#         --test_ds "${TEST_DATA}" \
#         --task MASC \
#         --gcn_layers 4 \
#         --device cuda:0 2>&1)

#     correct=$(echo "$output" | grep -o 'Correct:[0-9]*' | cut -d':' -f2)
#     label=$(echo "$output" | grep -o 'Label:[0-9]*' | cut -d':' -f2)
#     prediction=$(echo "$output" | grep -o 'Prediction:[0-9]*' | cut -d':' -f2)
#     accuracy=$(echo "$output" | grep -o 'Accuracy:[0-9.]*' | cut -d':' -f2)
#     f1=$(echo "$output" | grep -o 'Macro_f1:[0-9.]*' | cut -d':' -f2)

#     echo -e "\nModel: $(basename "$model")"
#     echo "Correct    : ${correct:-N/A}"
#     echo "Label      : ${label:-N/A}"
#     echo "Prediction : ${prediction:-N/A}"
#     echo "Accuracy   : ${accuracy:-N/A}"
#     echo "Macro_f1   : ${f1:-N/A}"

#     if [[ "${f1:-0}" =~ ^[0-9.]+$ ]]; then
#         is_better=$(awk -v f1="$f1" -v best="${best_stats_values[$MacroF1]}" 'BEGIN { print (f1 > best) ? 1 : 0 }')
        
#         if [ "$is_better" -eq 1 ]; then
#             best_stats_values[$COR]=${correct:-0}
#             best_stats_values[$LABEL]=${label:-0}
#             best_stats_values[$PRED]=${prediction:-0}
#             best_stats_values[$ACC]=${accuracy:-0}
#             best_stats_values[$MacroF1]=${f1:-0}
#             best_stats_values[$MODEL]=$(basename "$model")
#         fi
#     fi
# done 

# echo -e "\nBest Model: ${best_stats_values[$MODEL]}"
# echo "F1      : ${best_stats_values[$MacroF1]}"
# echo "Accuracy: ${best_stats_values[$ACC]}"
# echo "Correct : ${best_stats_values[$COR]}"
# echo "Label   : ${best_stats_values[$LABEL]}"
# echo "Prediction: ${best_stats_values[$PRED]}"



# MABSA evaluation
# python eval_tools.py \
#    --MATE_model ./DASCO/checkpoints/MATE_2017/best_f1:94.933.pt \
#    --MASC_model ./DASCO/checkpoints/MASC_2017/best_f1:77.616.pt \
#    --test_ds ./finetune_dataset/twitter17/test \
#    --task MABSA \
#    --gcn_layers 4 \
#    --device cuda:0