#!/bin/bash

#SBATCH -J spcyl_p85dg1e-5
#SBATCH -e spcyl_p85dg1e-5_err.log
#SBATCH -o spcyl_p85dg1e-5_out.log
#SBATCH --mem-per-cpu=2000
#SBATCH -t 99:00:00
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu

module load cuda
srun ~/spherocyls/exe/spherocyl_box r /scratch/tmarscha/spherocyl/test/ 1024 1e-5 0.025 50.0 0.01 0.00002 0.07 0.85 2.0 0.5 0