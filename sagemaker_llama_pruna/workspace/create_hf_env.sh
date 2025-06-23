#!/bin/bash

conda create -y -n hf_env python=3.10
source $(conda info --base)/etc/profile.d/conda.sh
conda activate hf_env
conda install -y -c conda-forge libstdcxx-ng>=12
export PYTHONNOUSERSITE=True
pip install torch --extra-index-url https://download.pytorch.org/whl/cu128
pip install transformers
pip install conda-pack
pip install pruna
conda-pack