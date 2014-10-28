#!/bin/bash

#SBATCH -J spcyl_p92dg1e-6r1
#SBATCH -e spcyl_p92sg1e-6r1_err.log
#SBATCH -o spcyl_p92dg1e-6r1_out.log
#SBATCH --mem-per-cpu=2000
#SBATCH -t 99:00:00
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p standard
#SBATCH -p gpu

module load cuda
srun ~/spherocyls/exe/spherocyl_box /scratch/tmarscha/spherocyl/n1024/p92/dg1e-6/r1/sd0036000000.dat /scratch/tmarscha/spherocyl/n1024/p92/dg1e-6/r1/ 1024 1e-6 0.025 18.0 0.01 0.00002 0.07 0.92 2.0 0.5