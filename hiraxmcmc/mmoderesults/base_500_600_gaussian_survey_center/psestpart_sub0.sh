#!/bin/bash

#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --mail-user viraj.nistane@unige.ch
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#SBATCH -J klt_500_600_gaussian_survey_center
#SBATCH --partition=private-dpt-cpu
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=20
##SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=8000M

##SBATCH --partition=debug-cpu
##SBATCH --ntasks=2
##SBATCH --cpus-per-task=5


source /home/nistanev/mmode/load_modules_and_env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun --cpu-bind cores drift-makeproducts run klt_params.yaml


