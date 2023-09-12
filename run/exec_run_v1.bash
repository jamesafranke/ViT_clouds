#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --job-name=test
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
#SBATCH --partition=gpu
##SBATCH --ntasks=2
##SBATCH --cpus-per-task=1
##SBATCH --mem-per-cpu=16000



appdir=${HOME}/ViT_clouds/scripts

export CUDA_DEVICE_MAX_CONNECTIONS=1

# Config
GPUS_PER_NODE=8

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="-N 1" #"-N $NNODES --gpus-per-node 1"

mpirun $DISTRIBUTED_ARGS  \
	python3 $appdir/30_training_loop.py
