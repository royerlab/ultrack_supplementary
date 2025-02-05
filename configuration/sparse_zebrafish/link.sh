#! /bin/bash

#SBATCH --job-name=LINK
#SBATCH --time=00:15:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=1
#SBATCH --output=./slurm_output/link-%A_%a.out

env | grep "^SLURM" | sort

module load anaconda/2022.05
conda activate imgproc

ultrack link -cfg $CFG_FILE -b $SLURM_ARRAY_TASK_ID $@
