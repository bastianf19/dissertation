#!/bin/bash

# Base directories
BASE_DIR="/home/s2587034/pretrained_checkpoints/babylm_models"
EVAL_SCRIPT="/home/s2587034/dissertation/code/evaluate_bhasa.py"
DATA_PATH="/home/s2587034/dissertation/BHASA/lindsea/id/pragmatics/pragmatic_reasoning_pair.jsonl"
TOKENIZER_PATH="/home/s2587034/dissertation/models/indo_tokenizer"

# Create results directory if it doesn't exist
# mkdir -p "/home/s2587034/dissertation/results"

# Find all model directories (matching the pattern with seed0)
find "$BASE_DIR" -type d -path "*/runs/*_seed0" | while read -r model_dir; do
    # Extract model name from path (e.g., babylm_shuffle_control_100M_randinit_seed0)
    model_name=$(basename "$model_dir")
    
    # Create a clean model name for output (remove _seed0)
    clean_name="${model_name%_seed0}"
    
    echo "================================================================"
    echo "Evaluating model: $clean_name"
    echo "Model path: $model_dir"
    echo "================================================================"
    
    # Run the evaluation
    python3 "$EVAL_SCRIPT" \
        --model_path "$model_dir" \
        --tokenizer_path "$TOKENIZER_PATH" \
        --data_path "$DATA_PATH"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully evaluated $clean_name"
    else
        echo "ERROR: Failed to evaluate $clean_name" >&2
    fi
    
    echo -e "\n\n"
done

echo "All evaluations completed. Results are saved in /home/s2587034/dissertation/results/"

# Optional: Run this to make the script executable:
# chmod +x /home/s2587034/dissertation/code/run_all_evaluations.sh
