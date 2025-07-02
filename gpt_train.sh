#!/bin/bash
cd ../mistral
# -m torch.distributed.run
# export WANDB_API_KEY=api_key
# export WANDB_ENTITY=bastianfarandy
# export WANDB_PROJECT=bastian

# WANDB_MODE=disabled python3 -m torch.distributed.run train.py \
# --config conf/train_shuffle_local3_100M_randinit_seed0.yaml \
# --nnodes 1 --nproc_per_node 2 \
# --training_arguments.fp16 true \
# --training_arguments.warmup_steps 300 \
# --training_arguments.max_steps 3000

WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 python3 train.py \
    --config conf/train_reverse_partial_100M_randinit_seed0.yaml \
    --nnodes 1 --nproc_per_node 1 \
    --training_arguments.fp16 true \
    --training_arguments.warmup_steps 300 \
    --training_arguments.max_steps 3000