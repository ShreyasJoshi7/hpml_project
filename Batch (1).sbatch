#!/bin/bash
#SBATCH --job-name=hpml_Lab_2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=assi_last.out
#SBATCH --mem=8GB
#SBATCH --gres=gpu:4
#SBATCH --time=00:15:00
#SBATCH --mail-user=dg3802@nyu.edu
#SBATCH --mail-type=ALL

module purge
module load python/intel/3.8.6
# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate lab
python run.py
