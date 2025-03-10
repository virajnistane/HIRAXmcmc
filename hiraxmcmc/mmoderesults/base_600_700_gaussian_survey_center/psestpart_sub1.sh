#!/bin/bash

#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --mail-user viraj.nistane@unige.ch
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#SBATCH -J psfisher1_500_600_gauss_survey_center
#SBATCH --partition=private-dpt-cpu
#SBATCH --time=6:00:00
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=16

source /home/nistanev/mmode/load_modules_and_env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mpirun python /home/nistanev/mmode/psestimation.py generate_bandpowers /srv/beegfs/scratch/shares/hirax/secret_hirax_directory/mmode_runs_2023/survey_runs/base_500_600_gaussian_survey_center/hirax/drift_products --power_spectrum_name=psmc_kl_0thresh_nofg_1threshold


## psmc_dk_5thresh_fg_1000thresh_1threshold
## psmc_dk_0thresh_fg_1000thresh_1threshold
## psmc_dk_0thresh_fg_10thresh_1threshold
## psmc_dk_0thresh_fg_1thresh_1threshold
## psmc_kl_0thresh_nofg_1threshold
