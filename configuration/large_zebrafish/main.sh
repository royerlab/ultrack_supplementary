#! /bin/bash

DS_LENGTH=790

export CFG_FILE="config.toml"
# export ULTRACK_DEBUG=1

module load anaconda
conda activate ultrack

rm ./output/*.out
mkdir -p output

SERVER_JOB_ID=$(sbatch --parsable create_server.sh)

SEGM_JOB_ID=$(sbatch --parsable --array=0-$DS_LENGTH%90 -d after:$SERVER_JOB_ID+1 segment.sh)
LINK_JOB_ID=$(sbatch --parsable --array=0-$((DS_LENGTH - 1))%90 -d afterok:$SEGM_JOB_ID link.sh)

SOLVE_JOB_ID_0=$(sbatch --parsable --array=0-15:2 -d afterok:$LINK_JOB_ID solve.sh)
SOLVE_JOB_ID_1=$(sbatch --parsable --array=1-15:2 -d afterok:$SOLVE_JOB_ID_0 solve.sh)

sbatch --mem 400GB --cpus-per-task=90 --job-name EXPORT \
    --output=./output/export-%j.out -d afterok:$SOLVE_JOB_ID_1 \
    ultrack export zarr-napari -cfg $CFG_FILE --include-parents -o results
