#!/bin/bash

#SBATCH -J spcyl_p95dg4e-6r2
#SBATCH -e spherocyl_shear_err.log
#SBATCH -o spherocyl_shear_out.log
#SBATCH --mem-per-cpu=2000
#SBATCH -t 99:00:00
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p standard

module load cuda
srun ~/spherocyls/exe/spherocyl_box /scratch/tmarscha/spherocyl/n1024/p95/dg4e-6/r2/sd0016500000.dat /scratch/tmarscha/spherocyl/n1024/p95/dg4e-6/r2/ 1024 4e-6 0.025 65.0 0.01 0.00002 0.07 0.95 2.0 0.5