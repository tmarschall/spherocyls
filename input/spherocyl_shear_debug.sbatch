#!/bin/bash

#SBATCH -J spcyl_shear_debug
#SBATCH -e spherocyl_shear_err.log
#SBATCH -o spherocyl_shear_out.log
#SBATCH --mem-per-cpu=2000
#SBATCH -t 10:00:00
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p standard

module load cuda
srun ~/spherocyls/exe/spherocyl_box r /scratch/tmarscha/spherocyl/shear/test3/ 512 1e-5 0.025 10.0 0.01 0.00001 0.1 0.85 2.0 0.5 1