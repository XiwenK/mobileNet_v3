#!/bin/bash

#SBATCH --partition=SCSEGPU_MSAI
#SBATCH --qos=q_msai
#SBATCH --job-name=mmdl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH -o ./logs/job.out
#SBATCH -e ./logs/error.err

CONFIG=$1
module load anaconda
source activate mmdl
python -u tools/train.py ${CONFIG} --launcher="slurm"