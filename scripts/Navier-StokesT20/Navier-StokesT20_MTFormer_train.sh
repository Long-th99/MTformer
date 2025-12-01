export CUDA_VISIBLE_DEVICES=0
CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")
EX_NAME="navier_stokes_t20/${CURRENT_TIME}_MTformer_depth2_TSST_sd0.1_dp0.1_64_4_32_lr1e-3_50ep_cos_bs8_ps4_Adamw"

python train.py \
    --config_file configs/navier_stokes_t20/MTFormer.py \
    --dataname navier_stokes_t20 \
    --data_root data \
    --res_dir work_dirs \
    --batch_size 8 \
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