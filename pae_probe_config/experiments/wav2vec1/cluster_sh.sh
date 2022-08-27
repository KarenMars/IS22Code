#!/bin/sh
# @hj
#SBATCH --job-name=w2v1
#SBATCH --partition=general --qos=short
#SBATCH --time=00:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu




module use /opt/insy/modulefiles
module load cuda/11.0
# srun -u --output=AVE.outputs python3 run.py
srun -u bash run.sh
