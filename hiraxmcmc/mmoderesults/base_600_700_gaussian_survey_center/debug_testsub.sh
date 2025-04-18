#!/bin/bash

#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --mail-user viraj.nistane@unige.ch
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#SBATCH -J prod_500_600_cst_survey_center
#SBATCH --partition=debug-cpu
#SBATCH --time=00:15:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8

source /home/nistanev/mmode/load_modules_and_env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun --cpu-bind cores drift-makeproducts run prod_params.yaml

