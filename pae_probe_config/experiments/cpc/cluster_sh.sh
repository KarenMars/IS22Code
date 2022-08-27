#!/bin/sh
# @hj
#SBATCH --job-name=cpc_cross
#SBATCH --partition=general --qos=short
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --mail-type=ALL




# module use /opt/insy/modulefiles
# module load cuda/11.0
# srun -u --output=AVE.outputs python3 run.py
srun -u bash run_timit_mboshi.sh
