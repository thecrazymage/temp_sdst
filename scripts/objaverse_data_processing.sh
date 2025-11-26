#!/bin/bash

module load Python CUDA/11.7 gnu10
source deactivate
conda activate castex

python ./objaverse_eval/objaverse_data_utils/download_objaverse_subset.py