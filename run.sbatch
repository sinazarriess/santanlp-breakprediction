#!/usr/bin/env bash

#SBATCH --mail-user=szarriess@techfak.uni-bielefeld.de
#SBATCH --mail-type=END
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00


# commands that should be scheduled
#srun bash run.sh
srun bash run_eval_random.sh
#srun bash run_eval_random1.sh
