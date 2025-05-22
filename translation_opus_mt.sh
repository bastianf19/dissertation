#!/bin/bash
python code/translation_opus_mt.py --source_dir dataset/ --target_dir dataset/translated/opus-mt/ --model_path models/opus-mt-en-id --batch_size 16 --subsets babylm_test