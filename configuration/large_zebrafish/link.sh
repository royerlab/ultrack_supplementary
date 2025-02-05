#! /bin/bash

#SBATCH --job-name=LINK
#SBATCH --time=00:10:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=5G
#SBATCH --cpus-per-task=1
#SBATCH --output=./output/link-%A_%a.out

env | grep "^SLURM" | sort

module load anaconda
conda activate ultrack

ultrack link -cfg $CFG_FILE -b $SLURM_ARRAY_TASK_ID

