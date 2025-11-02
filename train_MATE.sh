#!/bin/bash
export NCCL_P2P_DISABLE=1
# export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="1,2,3,4"
accelerate launch --config_file deepspeed_ddp.json MATE_finetune.py \
    --task MATE \
    --base_model ./Text_encoder/model_best \
    --pretrain_model ./checkpoints/MATE_pretrain/mate.pt \
    --train_ds ./finetune_dataset/twitter17/train \
    --eval_ds ./finetune_dataset/twitter17/dev \
    --hyper1 0.2 \
    --hyper2 0.2 \
    --hyper3 0.2 \
    --gcn_layers 4 \
    --lr 2e-5 \
    --seed 1000 \
    --itc 0 \
    --itm 0 \
    --lm  0 \
    --cl  1.0 \
    --save_path ./checkpoints/MATE_2017_0 \
    --epoch 20 \
    --log_step 1 \
    --save_step 200 \
    --batch_size 4 \
    --accumulation_steps 2 \
    --val_step 50