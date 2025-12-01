#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")
EX_NAME="mmnist/${CURRENT_TIME}_EnhancedMTformer_MSPGA_ps8_bs32_256_8_32_2e-3_Adamw_200ep"

python train.py \
    --config_file configs/mmnist/MTFormer.py \
    --dataname mmnist \
    --data_root data \
    --res_dir /root/autodl-tmp/work_dirs \
    --batch_size 32 \
    --epoch 200 \
    --overwrite \
    --lr 1e-3 \
    --opt adamw \
    --weight_decay 1e-2 \
    --sched onecycle \
    --warmup_epoch 10 \
    --warmup_lr 1e-5 \
    --clip_grad 1.0 \
    --ex_name "$EX_NAME" \
    --tb_dir logs_tb/03_08 \
    --method MTformer