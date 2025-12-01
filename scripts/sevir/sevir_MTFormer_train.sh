export CUDA_VISIBLE_DEVICES=0
CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")
EX_NAME="sevir/${CURRENT_TIME}_MTFormer_depth2_TSST_sd0.1_dp0.1_384_4_32_lr1e-3_50ep_cos_bs4_ps4_Adamw"

python train.py \
    --config_file configs/sevir/MTFormer.py \
    --dataname sevir \
    --data_root /root/autodl-tmp/data \
    --res_dir /root/autodl-tmp/work_dirs \
    --batch_size 4 \
    --epoch 50 \
    --sched cosine \
    --warmup_epoch 0 \
    --overwrite \
    --lr 1e-3 \
    --opt adamw \
    --weight_decay 1e-2 \
    --ex_name "$EX_NAME" \
    --tb_dir logs_tb/03_08 \
    --method MTformer