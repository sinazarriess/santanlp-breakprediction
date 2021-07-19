#!/bin/bash
# based on https://help.rc.ufl.edu/doc/Remote_Jupyter_Notebook #SBATCH --job-name=jupyter
#SBATCH --output=jupyter_%j.log
#SBATCH --mail-user=szarriess@techfak.uni-bielefeld.de #SBATCH --mail-type=END

date;hostname;pwd
export PATH=/media/compute/homes/szarriess/anaconda3/bin${PATH:+:${PATH}}
export PATH=/media/compute/vol/cuda/10.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/media/compute/vol/cuda/10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/media/compute/vol/cuda/10.1

# activate conda env with tensorflow:
source /media/compute/homes/szarriess/anaconda3/bin/activate torchenv


port=$(shuf -i 20000-30000 -n 1)

echo -e "\nStarting Jupyter Notebook on port ${port} on the $(hostname) server."
echo -e "\nSSH tunnel command: ssh -J ${USER}@shell.techfak.de,${USER}@login.gpu.cit-ec.net -L 808
0:$(hostname):${port} ${USER}@login.gpu.cit-ec.net"
echo -e "\nLocal URI: http://localhost:8080"
echo -e "\nYou need to copy the token from the logfile of this job."

# the following seems to be necessary to avoid permission # problems with trying to write to /run :
export XDG_RUNTIME_DIR=""
jupyter notebook --no-browser --port=${port} --ip='*' date
