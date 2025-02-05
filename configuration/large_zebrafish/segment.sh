#! /bin/bash

#SBATCH --job-name=SEGMENT
#SBATCH --time=00:15:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --output=./output/segment-%A_%a.out

env | grep "^SLURM" | sort

module load anaconda
conda activate ultrack

INPUT_PATH=# Define the input path here

if [ -z "$INPUT_PATH" ]; then
    echo "Error: User must define INPUT_PATH"
    exit 1
fi

ultrack segment $INPUT_PATH -cfg $CFG_FILE \
    -b $SLURM_ARRAY_TASK_ID -r napari-dexp -el Boundary -dl Prediction
