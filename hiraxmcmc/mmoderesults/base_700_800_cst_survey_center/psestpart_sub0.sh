#!/bin/bash

#SBATCH -A rrg-sievers

#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --mail-user viraj.nistane@unige.ch
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#SBATCH -J klt_700_800_cst_survey_center
#SBATCH --partition=compute
###SBATCH --partition=debug
#SBATCH --time=24:00:00
###SBATCH --time=1:00:00
#SBATCH --nodes=50
###SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40

source /home/s/sievers/nistanev/2212_hirax_forecasts/load_modules_and_env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun --cpu-bind cores drift-makeproducts run klt_params.yaml


