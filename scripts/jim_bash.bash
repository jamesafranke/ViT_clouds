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





appdir=${HOME}/ViT_clouds/scripts

export CUDA_DEVICE_MAX_CONNECTIONS=2

# Config
NGPUS=2
GPUS_PER_NODE=8

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NGPUS=2
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
DISTRIBUTED_ARGS="-N $NGPUS" # --gpus-per-node ${GPUS_PER_NODE}"

echo $DISTRIBUTED_ARGS python3 test.py
time python3 test.py


torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:6000 test.py





#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --job-name=test
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

##SBATCH --ntasks=2
##SBATCH --cpus-per-task=1
##SBATCH --mem-per-cpu=16000

appdir=${HOME}/ViT_clouds/scripts

export CUDA_DEVICE_MAX_CONNECTIONS=1

# Config
NGPUS=1
GPUS_PER_NODE=8

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NGPUS=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# Profiler
# Use $PMI_RANK for MPICH and $SLURM_PROCID with srun.
if [ $OMPI_COMM_WORLD_RANK -eq 0 ]; then
	nsys profile -e NSYS_MPI_STORE_TEAMS_PER_RANK=1 -t mpi "$@"
else
	"$@"
fi

DISTRIBUTED_ARGS="-N $NGPUS" # --gpus-per-node ${GPUS_PER_NODE}"

nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits >> /home/bgmpw4ez/gpu_metrics.log

echo $DISTRIBUTED_ARGS python3 test.py
time python3 test.py







