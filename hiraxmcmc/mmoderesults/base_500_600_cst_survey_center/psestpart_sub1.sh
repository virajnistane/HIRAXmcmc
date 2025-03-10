#!/bin/bash

#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --mail-user viraj.nistane@unige.ch
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#SBATCH -J psfisher1_500_600_cst_survey_center
#SBATCH --partition=private-dpt-cpu,public-cpu
#SBATCH --time=05:00:00
###SBATCH --nodes=8
#SBATCH --ntasks=10
###SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=16

source /home/nistanev/mmode/load_modules_and_env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


mpirun python /home/nistanev/mmode/psestimation.py generate_bandpowers /srv/beegfs/scratch/shares/hirax/secret_hirax_directory/mmode_runs_2023/survey_runs/base_500_600_cst_survey_center/hirax/drift_products --power_spectrum_name=psmc_kl_5thresh_nofg_1threshold


###psmc_kl_5thresh_nofg_1threshold

###psmc_dk_5thresh_fg_1000thresh_1threshold

