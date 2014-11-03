#!/bin/bash

#SBATCH -J spcyl_p95dg4e-6r2
#SBATCH -e spcyl_p95dg4e-6r2_err.log
#SBATCH -o spcyl_p95dg4e-6r2_out.log
#SBATCH --mem-per-cpu=2000
#SBATCH -t 99:00:00
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu

module load cuda
srun ~/spherocyls/exe/spherocyl_box /scratch/tmarscha/spherocyl/n1024/p95/dg4e-6/r2/sd0027135000.dat /scratch/tmarscha/spherocyl/n1024/p95/dg4e-6/r2/ 1024 4e-6 0.025 41.46 0.01 0.00002 0.07 0.95 2.0 0.5 1