#!/usr/bin/env/bash

export PATH=/media/compute/homes/szarriess/anaconda3/bin${PATH:+:${PATH}}
export PATH=/media/compute/vol/cuda/10.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/media/compute/vol/cuda/10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/media/compute/vol/cuda/10.1

# activate conda env with tensorflow:
source /media/compute/homes/szarriess/anaconda3/bin/activate torchenv

#python run_eval.py --in ../santanlp-corpus/corpus1/test/284.txt --out results/corpus1/test/284.txt

#python run_eval_finetuning.py --in ../santanlp-corpus/corpus1/test/284.txt --out results/corpus1/test/284.txt --model finetuningBERT/finetuned_models/ae_v1
python run_eval_finetuning.py --in ../phase-1-round-2-test-corpus/txt/01_Buechner.txt --out results/phase-1-round-2-test-corpus/01_Buechner.txt --model finetuningBERT/finetuned_models/ae_v1
 
