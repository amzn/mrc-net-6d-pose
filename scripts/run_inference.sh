#!/bin/bash

DATASET=tless
SUFFIX=0320

# Replace @checkpoint_name with the checkpoint folder and @model_name with checkpoint file name
# Add --is_real if a fine-tuned model is to be used
# The final csv naming follows BOP convention for evaluation
python inference.py \
    --dataset $DATASET \
    --checkpoint_name chkpt_${DATASET} \
    --model_name tless \
    --output_suffix $SUFFIX
