#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --job-name=test
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2


appdir=${HOME}/ViT_clouds/scripts

time torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:6000  $appdir/torchrun_test.py
