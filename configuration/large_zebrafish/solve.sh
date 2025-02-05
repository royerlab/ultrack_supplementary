#! /bin/bash

#SBATCH --job-name=TRACKING
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=12
#SBATCH --output=./output/track-%A_%a.out 

env | grep "^SLURM" | sort

module load anaconda
conda activate ultrack

ultrack solve -cfg $CFG_FILE -b $SLURM_ARRAY_TASK_ID

