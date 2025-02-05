#! /bin/bash

#SBATCH --job-name=SEGMENT
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=150G
#SBATCH --cpus-per-task=1
#SBATCH --output=./slurm_output/segment-%A_%a.out

env | grep "^SLURM" | sort

module load anaconda/2022.05
conda activate imgproc

ultrack segment $1 -cfg $CFG_FILE \
    -b $SLURM_ARRAY_TASK_ID -r napari-ome-zarr -cl edge -fl detection
