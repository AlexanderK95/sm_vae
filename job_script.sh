#!/bin/bash

#SBATCH --ntasks 1
#SBATCH --cpus-per-task 12
#SBATCH --mem-per-cpu=1024
#SBATCH --time 01:00:00
#SBATCH --gpus=1

#SBATCH --mail-user=alexander.kress@physik.uni-marburg.de
#SBATCH --mail-type=END
# these parameters determining the size of our job
# meaning the amount of ressources needed to process it

##SBATCH --array 1-20
# This is a job Array of 20 jobs

#SBATCH --job-name Tensorflow-test
# A name for this job to be displayed in the queue
#SBATCH --output vae_job_%A_%a.out
# filename where the terminal output pf this job goes

#here the jobscript starts

export TF_ENABLE_ONEDNN_OPTS=0

# deactivate any current environment
if [ -z "$VIRTUAL_ENV" ]; then
deactivate
fi

#load the required modules
module purge
# module load gcc/system openmpi/4.1.2 python/3.9.5
# module load cuda/11.1
module load miniconda

source $CONDA_ROOT/bin/activate
conda activate tf

# nvidia-smi

# conda env list
# conda list

echo "running python script..."

# python3 train.py
python3 vae.py

echo "python script has terminated"


conda deactivate