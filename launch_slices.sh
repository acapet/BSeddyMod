#!/bin/bash
# Submission script for NIC5
#SBATCH --job-name=EddyCompo
#SBATCH --time=1-00:00:00 # days-hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6400 # megabytes
#SBATCH --partition=batch
#
#SBATCH --mail-user=acapet@uliege.be
#SBATCH --mail-type=ALL
#
#SBATCH --comment=bsmfc
#
###SBATCH --outfile=Compo-outputs.txt

source ~/pyload_evan

#export OMP_NUM_THREADS=6
#export MKL_NUM_THREADS=6

#mpirun --bind-to none
python make_BlackSea_sections_EddyToCoast_2013_AC.py -y $year -m $month
#$SLURM_ARRAY_TASK_ID
