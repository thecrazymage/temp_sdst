#!/bin/bash

module load Python CUDA/12.4 gnu10
source deactivate
conda activate castex

PROMPT="A cow in a spider-man suit"

python main.py \
    --mesh_location "assets/spot.obj" \
    --prompt "$PROMPT" \
    --num_steps_i 500 \
    --num_steps_ii 1000 \
    --guidance_scale_i 20 \
    --guidance_scale_ii 15 \
    --use_dir_embeddings