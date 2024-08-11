#!/bin/bash

python run.py \
    --task "pred_transfer_baseline" \
    --dataset 'synthetic' \
    --use_gpu "true" \
    --gpu_ids "1"
