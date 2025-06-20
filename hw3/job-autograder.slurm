#!/bin/bash
##SBATCH -A see180004p # 2017 XSEDE Applications of Parallel Computing Course Allocation
#SBATCH -J auto-particle-gpu
#SBATCH -o auto-particle-gpu.stdout
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:1
#SBATCH -t 00:10:00
#SBATCH -N 1

source modules.sh

./autograder -s auto-particle-gpu.stdout -v gpu
