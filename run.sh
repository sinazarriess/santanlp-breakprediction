#!/usr/bin/env/bash

export PATH=/media/compute/homes/szarriess/anaconda3/bin${PATH:+:${PATH}}
export PATH=/media/compute/vol/cuda/10.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/media/compute/vol/cuda/10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/media/compute/vol/cuda/10.1

# activate conda env with tensorflow:
source /media/compute/homes/szarriess/anaconda3/bin/activate torchenv

python test_par_pairs.py
