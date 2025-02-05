#! /bin/bash

#SBATCH --job-name=SOLVE
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=950G
#SBATCH --cpus-per-task=10
#SBATCH --output=./slurm_output/solve-%A_%a.out

env | grep "^SLURM" | sort

module load anaconda/2022.05
conda activate imgproc

ultrack solve -cfg $CFG_FILE -b $SLURM_ARRAY_TASK_ID
