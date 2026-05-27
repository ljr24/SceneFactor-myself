#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexey.bokhovkin@skoltech.ru
#SBATCH --mem=16gb
#SBATCH --cpus-per-task=4
#SBATCH --partition=submit

export PATH=/rhome/${USER}/miniconda3/bin/:$PATH
source activate scenefactor

# python -u fix_3dfuture_meshes.py -n ${SLURM_ARRAY_TASK_COUNT} -p ${SLURM_ARRAY_TASK_ID}
# python fix_3dfuture_meshes.py -n 16

N=4  # 并行数

for ((i=0;i<$N;i++)); do
    python fix_3dfuture_meshes.py -n $N -p $i &
done

wait