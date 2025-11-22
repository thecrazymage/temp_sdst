#!/bin/bash
module load Python CUDA/11.7 gnu10
source deactivate
conda activate t2t

python ./objaverse_eval/utils/download_objaverse_subset.py