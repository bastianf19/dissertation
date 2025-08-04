#!/bin/bash

# Base directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRAINING_SCRIPT="prepare_training.sh"

# Check if the training script exists
if [ ! -f "$TRAINING_SCRIPT" ]; then
    echo "Error: Training script not found at $TRAINING_SCRIPT"
    exit 1
fi

# Create a logs directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="/home/s2587034/dissertation/logs/training_${TIMESTAMP}"
mkdir -p "$(dirname "$LOG_DIR")"  # Ensure log directory exists

# Function to run a single training configuration
run_training() {
    local perturbation=$1
    local dataset=$2
    local seed=$3
    local paren_model=$4
    local log_file="${LOG_DIR}_${perturbation}_${dataset}_seed${seed}_${paren_model}.log"
    
    echo "Running: $perturbation $dataset $seed $paren_model"
    echo "Logging to: $log_file"
    
    # Run the training script and log the output
    {
        echo "=== Starting training at $(date) ==="
        echo "Command: $TRAINING_SCRIPT $perturbation $dataset $seed $paren_model"
        echo "----------------------------------------"
        
        # Run the actual command
        time bash "$TRAINING_SCRIPT" "$perturbation" "$dataset" "$seed" "$paren_model"
        
        # Capture the exit status
        status=$?
        echo "----------------------------------------"
        echo "=== Training completed at $(date) ==="
        echo "Exit status: $status"
        
        return $status
    } 2>&1 | tee "$log_file"
    
    return ${PIPESTATUS[0]}
}

# Array of training configurations
# Format: "perturbation dataset seed paren_model"
declare -a training_configs=(
    # "shuffle_control 100M 0 randinit"\
    "shuffle_nondeterministic 100M 0 randinit"
    # "shuffle_deterministic21 100M 0 randinit"
    # "shuffle_local3 100M 0 randinit"
    # "shuffle_local5 100M 0 randinit"
    # "shuffle_local10 100M 0 randinit"
    # "shuffle_even_odd 100M 0 randinit"
    # "reverse_control 100M 0 randinit"
    # "reverse_partial 100M 0 randinit"
    # "reverse_full 100M 0 randinit"
)

# Run all training configurations
for config in "${training_configs[@]}"; do
    # Split the config into individual parameters
    read -r perturbation dataset seed paren_model <<< "$config"
    
    echo "========================================"
    echo "Starting training for: $perturbation $dataset (seed: $seed, model: $paren_model)"
    echo "========================================"
    
    if ! run_training "$perturbation" "$dataset" "$seed" "$paren_model"; then
        echo "ERROR: Training failed for $perturbation $dataset (seed: $seed, model: $paren_model)"
        # Uncomment the next line to stop on first error
        # exit 1
    fi
    
    echo -e "\n\n"
done

echo "All training configurations have been processed."
echo "Logs are available with prefix: $LOG_DIR"