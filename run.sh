#!/bin/bash
module load Python CUDA/11.7 gnu10
source deactivate
conda activate sdst

MESH_PATH=

PROMPT="A wooden toy cow, handcrafted, child-friendly, rounded body, short legs, small horns, simple snout, light maple with visible grain, warm varnish."

python main.py \
    --mesh_location "data/spot.obj" \
    --prompt "$PROMPT" \
    --num_steps_i 500 \
    --num_steps_ii 1000 \
    --use_dir_embeddings