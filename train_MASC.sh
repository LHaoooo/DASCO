export NCCL_P2P_DISABLE=1
# export CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7"
accelerate launch --config_file deepspeed_ddp.json MASC_finetune.py \
    --task MASC \
    --base_model ./Text_encoder/model_best \
    --pretrain_model ./checkpoints/pretrain_ckp/MASC_best_model.pt \
    --train_ds /home/data/finetune_dataset/twitter15/train \
    --eval_ds /home/data/finetune_dataset/twitter15/test \
    --hyper1 0.2 \
    --hyper2 0.12 \
    --hyper3 0.2 \
    --gcn_layers 4 \
    --lr 2e-5 \
    --seed 1000 \
    --itc 0 \
    --itm 0 \
    --lm  0 \
    --cl  1.0 \
    --save_path ./checkpoints/MASC_2015_baseft \
    --epoch 100 \
    --log_step 1 \
    --save_step 1600 \
    --batch_size 4 \
    --accumulation_steps 2 \
    --val_step 80