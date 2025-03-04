export NCCL_P2P_DISABLE=1
# export CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7"
accelerate launch --config_file deepspeed_ddp.json MATE_finetune.py \
    --base_model ./Text_encoder/model_best \
    --pretrain_model ./checkpoints/pretrain_ckp/MATE_best_model.pt \
    --train_ds /home/data/finetune_dataset/twitter15/train_data \
    --eval_ds /home/data/finetune_dataset/twitter15/dev_data \
    --lr 2e-5 \
    --seed 1000 \
    --itc 0.1 \
    --itm 0.1 \
    --lm  0.1 \
    --cl  1.0 \
    --save_path ./checkpoints/MATE_2015_0 \
    --epoch 20 \
    --log_step 1 \
    --save_step 300 \
    --batch_size 4 \
    --accumulation_steps 2 \
    --val_step 100