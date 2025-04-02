#!/bin/bash

#SBATCH -A rrg-sievers

#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --mail-user viraj.nistane@unige.ch
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#SBATCH -J prod_600_700_gaussian_survey_center
#SBATCH --partition=compute
#SBATCH --time=24:00:00
##SBATCH --nodes=10
##SBATCH --ntasks-per-node=5
##SBATCH --cpus-per-task=8
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=4

source /home/s/sievers/nistanev/2212_hirax_forecasts/load_modules_and_env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun --cpu-bind cores drift-makeproducts run prod_params.yaml

