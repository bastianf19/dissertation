#!/bin/bash
cd ../mistral
python3 -m torch.distributed.run train.py --config conf/train_reverse_control_10M_randinit_seed0.yaml --nnodes 1 --nproc_per_node 2 --training_arguments.fp16 true --training_arguments.warmup_steps 300 --training_arguments.max_steps 3000