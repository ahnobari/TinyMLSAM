#!/bin/bash

# Common parameters
DATASETS=("Data/cityscapes" "Data/BDD100K")
THRESHOLD_PAIRS=(
    "0.2 0.15"
    "0.3 0.25"
)
RESULTS_PATH="results"
PROMPTING=("NoPE" "PE")

# Define models and their correct variants
declare -A MODELS_AND_VARIANTS
MODELS_AND_VARIANTS=(
    ["SAM2"]="facebook/sam2.1-hiera-tiny facebook/sam2.1-hiera-small facebook/sam2.1-hiera-large facebook/sam2.1-hiera-base-plus"
    ["EdgeSAM"]="Base _3x"
    ["EfficientSAM"]="S Ti"
    ["EfficientViTSAM"]="efficientvit-sam-xl1 efficientvit-sam-l0 efficientvit-sam-l1 efficientvit-sam-l2 efficientvit-sam-xl0"
    ["MobileSAM"]="MobileSAM"
    ["SAMHQ"]="vit_h vit_l vit_b vit_tiny"
)

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_PATH"

# Loop through all models, variants, datasets, and threshold pairs
for MODEL in "${!MODELS_AND_VARIANTS[@]}"; do
    VARIANTS=${MODELS_AND_VARIANTS[$MODEL]}
    for VARIANT in $VARIANTS; do
        for DATASET in "${DATASETS[@]}"; do
            for THRESHOLDS in "${THRESHOLD_PAIRS[@]}"; do
                for PROMPT in "${PROMPTING[@]}"; do
                    BOX_THRESHOLD=$(echo "$THRESHOLDS" | cut -d' ' -f1)
                    TEXT_THRESHOLD=$(echo "$THRESHOLDS" | cut -d' ' -f2)

                    # Generate the query path
                    DATASET_NAME=$(basename "$DATASET")
                    QUERY_PATH="${RESULTS_PATH}/instances_Tiny_${DATASET_NAME}_BT_${BOX_THRESHOLD}_TT_${TEXT_THRESHOLD}_${PROMPT}.pkl"

                    echo "Running with:"
                    echo "Model: $MODEL"
                    echo "Variant: $VARIANT"
                    echo "Dataset: $DATASET"
                    echo "Box Threshold: $BOX_THRESHOLD"
                    echo "Text Threshold: $TEXT_THRESHOLD"
                    echo "Query Path: $QUERY_PATH"
                    echo "---------------------------------"

                    # Run the Python script with the current configuration
                    if [ "$PROMPT" == "NoPE" ]; then
                        python run_sam.py \
                            --model "$MODEL" \
                            --model_variant "$VARIANT" \
                            --data_path "$DATASET" \
                            --queries_path "$QUERY_PATH" \
                            --results_path "$RESULTS_PATH"
                    else
                        python run_sam.py \
                            --model "$MODEL" \
                            --model_variant "$VARIANT" \
                            --data_path "$DATASET" \
                            --queries_path "$QUERY_PATH" \
                            --results_path "$RESULTS_PATH" \
                            --use_prompt_engineering
                    fi
                done
            done
        done
    done
done
