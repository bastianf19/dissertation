#!/bin/bash

# Base directory for models
BASE_DIR="/home/s2587034/pretrained_checkpoints/babylm_models"

# Output directory for results
OUTPUT_DIR="/home/s2587034/dissertation/results/syntax_evaluation_probability"

# Get all model variants by listing directories in the base directory
readarray -t VARIANTS < <(find "$BASE_DIR" -maxdepth 1 -mindepth 1 -type d -name "babylm_*" -not -name "*indo*" -exec basename {} \; | sort)

# Create output directory if it doesn't exist
# mkdir -p "$OUTPUT_DIR"

# Print found variants for verification
echo "Found the following model variants:"
printf "  - %s\n" "${VARIANTS[@]}"
echo ""

# Loop through each variant
for variant_dir in "$BASE_DIR"/*; do
    # Skip if not a directory
    [ ! -d "$variant_dir" ] && continue
    
    # Get the base name of the variant directory
    variant=$(basename "$variant_dir")
    # Extract just the variant name (e.g., 'reverse_control' from 'babylm_reverse_control_100M_randinit')
    variant_name=$(echo "$variant" | cut -d'_' -f2-3)
    
    echo "Processing variant: $variant"
    
    # Find all seed directories in the variant directory
    for seed_dir in "$variant_dir"/*_seed*; do
        # Skip if not a directory
        [ ! -d "$seed_dir" ] && continue
        
        # Find runs directory
        for run_dir in "$seed_dir"/runs/*; do
            [ ! -d "$run_dir" ] && continue
            
            # Look for checkpoint directories (checkpoint-3000)
            for checkpoint_dir in "$run_dir"/checkpoint-3000; do
                [ ! -d "$checkpoint_dir" ] && continue
                
                echo "=================================================="
                echo "Processing: $variant - $(basename "$seed_dir")"
                echo "Checkpoint: 3000"
                echo "Model path: $checkpoint_dir"
                echo "Variant name: $variant_name"
                echo "=================================================="
                
                # Run the evaluation script
                python /home/s2587034/dissertation/code/evaluate_bhasa.py \
                    --model_path "$checkpoint_dir" \
                    --model_variant "$variant_name" \
                    --output_dir "$OUTPUT_DIR" \
                    --eval_type "syntax" \
                    --eval_method "continuation_probability" \
                    --language "translated"
                
                # Add a separator between runs
                echo -e "\n\n"
            done
        done
    done
done

echo "All evaluations completed!"