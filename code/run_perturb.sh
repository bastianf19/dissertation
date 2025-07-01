#!/bin/bash

# Define arrays for perturbations and datasets
perturbations=(
    # "shuffle_control"
    # "reverse_control"
    # "shuffle_nondeterministic"
    # "shuffle_deterministic21"
    # "shuffle_local3"
    # "shuffle_local5"
    # "shuffle_local10"
    # "shuffle_even_odd"
    # "reverse_partial"
    "reverse_full"
)

datasets=(
    "100M"
    # "10M"
    "dev"
    "test"
)

# Base directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get current timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Main loop to run all combinations
for perturbation in "${perturbations[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo "Running perturbation: $perturbation on dataset: $dataset"
        
        # Create a log file for this run
        LOG_FILE="/home/s2587034/dissertation/logs/${perturbation}_${dataset}_${TIMESTAMP}.log"
        
        # Run the command and log the output
        python3 "${SCRIPT_DIR}/perturb.py" \
            "$perturbation" \
            "$dataset" 2>&1 | tee "$LOG_FILE"
            
        # Check the exit status of the command
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "SUCCESS: $perturbation on $dataset completed successfully" | tee -a "$LOG_FILE"
        else
            echo "ERROR: $perturbation on $dataset failed" | tee -a "$LOG_FILE"
        fi
        
        # Add a separator between runs
        echo -e "\n---\n"
    done
done

echo "All perturbations completed. Check the logs in /home/s2587034/dissertation/logs/ for details."