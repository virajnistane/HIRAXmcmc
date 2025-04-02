#!/bin/bash

#SBATCH -A rrg-sievers

#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --mail-user viraj.nistane@unige.ch
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#SBATCH --job-name=psfisher2_400_500_cst_survey_center
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-19
#SBATCH --cpus-per-task=40

source /home/s/sievers/nistanev/2212_hirax_forecasts/load_modules_and_env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

##srun --cpu-bind cores drift-makeproducts run prod_params.yaml

mpirun -np 1 python ~/2212_hirax_forecasts/psestimation.py generate_fisher_bias /scratch/s/sievers/nistanev/mmode_runs_2023/survey_runs/base_400_500_cst_survey_center/hirax/drift_products --power_spectrum_name=psmc_kl_5thresh_nofg_1threshold


###psmc_kl_5thresh_nofg_1threshold

###psmc_dk_5thresh_fg_1000thresh_1threshold

