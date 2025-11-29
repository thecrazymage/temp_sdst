#!/bin/bash

module load Python CUDA/12.4 gnu10
conda env create -f environment.yml -n castex
conda activate castex
pip install "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8" --no-build-isolation