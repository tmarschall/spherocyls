#!/bin/bash

#SBATCH -J spcyl_p95dg1e-5
#SBATCH -e spcyl_p95dg1e-5_err.log
#SBATCH -o spcyl_p95dg1e-5_out.log
#SBATCH --mem-per-cpu=2000
#SBATCH -t 99:00:00
#SBATCH -n 1
#SBATCH -p standard

module load cuda
srun ~/spherocyls/exe/cpp_spherocyl_strain /scratch/tmarscha/spherocyl/n1024/p95/dg1e-5/r3/sd0005200000.dat /scratch/tmarscha/spherocyl/n1024/p95/dg1e-5/r3/ 1024 1e-5 0.025 0.1 0.01 0.00002 0.08 0.95 2.0 0.5