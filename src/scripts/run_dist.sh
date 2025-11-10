export WORLD_SIZE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node 4 -m open_clip_train.main \
    --logs-dir /home/hzhanguw/research-projects/CLIPlogs \
    --wandb-project-name "DINOv2_CLIP" \
    --name "epoch30_img64_lr2e-5_context512_Dino_BiomedBERT_dist"\
    --save-frequency 1 \
    --accum-freq 24 \
    --log-every-n-steps 1 \
    --save-most-recent \
    --report-to wandb \
    --dataset-type custom \
    --train-data="/home/hzhanguw/research-projects/data/t1_train_data.json"  \
    --val-data="/home/hzhanguw/research-projects/data/t1_val_data.json"  \
    --csv-img-key file \
    --csv-caption-key text \
    --warmup 352 \
    --batch-size=2 \
    --lr=2e-5 \
    --wd=0.1 \
    --epochs=30 \
    --workers=16 \
    --max_patient_imgs_length 64 \
    --model DINOv2_BiomedBERT \
    --dist-backend nccl \
    --gather-with-grad \
    --local-loss \
    --loss-dist-impl gather