#!/bin/bash
# python code/tag.py dataset/clean_data/nllb/babylm_data_ind/babylm_*/* --subset babylm_100M --batch-size 100000 --max-seq-length 1024 --filename switchboard.train
python code/tag.py dataset/babylm_data/babylm_*/* --subset babylm_test --batch-size 75000 --max-seq-length 1024
# --filename bnc_spoken.train