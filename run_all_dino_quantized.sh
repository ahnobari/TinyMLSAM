#!/bin/bash

# Set common parameters
BATCH_SIZE=8
SAVE_PATH="results"

# Define configurations
MODELS=("Base" "Tiny")
DATASETS=("Data/cityscapes" "Data/BDD100K")
THRESHOLD_PAIRS=(
    "0.2 0.15"
    "0.3 0.25"
)

# Define quantization variants
QUANTIZATION_VARIANTS=("nf4" "fp4" "int8" "none")  # Added 'none' for non-quantized runs

PROMPT_ENGINEERING=(true false)

# Create the results directory if it doesn't exist
mkdir -p "$SAVE_PATH"

# Loop through all combinations of models, datasets, and threshold pairs
for QUANT in "${QUANTIZATION_VARIANTS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
            for PAIR in "${THRESHOLD_PAIRS[@]}"; do
                for PROMPT in "${PROMPT_ENGINEERING[@]}"; do
                    BOX_THRESHOLD=$(echo "$PAIR" | cut -d' ' -f1)
                    TEXT_THRESHOLD=$(echo "$PAIR" | cut -d' ' -f2)
    
                    echo "Running inference with:"
                    echo "Quantization: $QUANT"
                    echo "Model: $MODEL"
                    echo "Dataset: $DATASET"
                    echo "Box Threshold: $BOX_THRESHOLD"
                    echo "Text Threshold: $TEXT_THRESHOLD"
                    echo "Prompt Engineering: $PROMPT"
                    echo " ---------------------------------"

                    if [ "$QUANT" = "none" ]; then
                        if [ "$PROMPT" = true ]; then
                            python get_instances.py \
                                --model "$MODEL" \
                                --data_path "$DATASET" \
                                --batch_size "$BATCH_SIZE" \
                                --box_threshold "$BOX_THRESHOLD" \
                                --text_threshold "$TEXT_THRESHOLD" \
                                --save_path "$SAVE_PATH" \
                                --use_prompt_engineering
                        else
                            python get_instances.py \
                                --model "$MODEL" \
                                --data_path "$DATASET" \
                                --batch_size "$BATCH_SIZE" \
                                --box_threshold "$BOX_THRESHOLD" \
                                --text_threshold "$TEXT_THRESHOLD" \
                                --save_path "$SAVE_PATH"
                        fi
                    else
                        if [ "$PROMPT" = true ]; then
                            python get_instances_quantized.py \
                                --model "$MODEL" \
                                --quantized_variant "$QUANT" \
                                --data_path "$DATASET" \
                                --batch_size "$BATCH_SIZE" \
                                --box_threshold "$BOX_THRESHOLD" \
                                --text_threshold "$TEXT_THRESHOLD" \
                                --save_path "$SAVE_PATH" \
                                --use_prompt_engineering
                        else
                            python get_instances_quantized.py \
                                --model "$MODEL" \
                                --quantized_variant "$QUANT" \
                                --data_path "$DATASET" \
                                --batch_size "$BATCH_SIZE" \
                                --box_threshold "$BOX_THRESHOLD" \
                                --text_threshold "$TEXT_THRESHOLD" \
                                --save_path "$SAVE_PATH"
                        fi
                    fi
                done
            done
        done
    done
done