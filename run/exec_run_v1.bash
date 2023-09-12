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
#PROFILE="nsys profile --trace=cuda,nvtx -b dwarf --python-sampling=true --cudabacktrace=all --python-backtrace=cuda --capture-range=nvtx --nvtx-capture=epoch_1 --env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0"
PROFILE="nsys profile --trace=cuda,nvtx -b dwarf --python-sampling=true --cudabacktrace=all --python-backtrace=cuda --capture-range=cudaProfilerApi  --nvtx-capture=epoch_1"
#FULL_PROFILE="nsys profile --trace=cuda,nvtx -b dwarf --python-sampling=true --cudabacktrace=all --python-backtrace=cuda"


echo   $DISTRIBUTED_ARGS $PROFILE python3 $appdir/30_training_loop.py
time $PROFILE python3 $appdir/30_training_loop.py
