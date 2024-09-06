#!/bin/bash

#SBATCH --account=panels
#SBATCH --time=2:00:00
#SBATCH --job-name=test_job
#SBATCH --nodes=1
#SBATCH --qos=high
#SBATCH --mail-user=jwang5@nrel.gov
#SBATCH --mail-type=ALL

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES

module purge
module load python

echo "Starting simulations"
python ochre/resstock_model/30k/OCHRE/bin/run_multiple_panels_multi.py
echo "Finished simulations"

