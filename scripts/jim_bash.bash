#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --job-name=test
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
##SBATCH --ntasks=2
##SBATCH --cpus-per-task=1
##SBATCH --mem-per-cpu=16000

torchrun --nnodes=1:1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:6000 test.py


