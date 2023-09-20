#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --job-name=test
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

##SBATCH --ntasks=2
##SBATCH --cpus-per-task=1
##SBATCH --mem-per-cpu=16000



appdir=${HOME}/ViT_clouds/scripts

#export CUDA_DEVICE_MAX_CONNECTIONS=1

# Model Config
epochs=3
batch_size=$1
mlp_dim=$2
datadir=$3
image_size=$4


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


DISTRIBUTED_ARGS="-N $NGPUS"
PROFILE="nsys profile --trace=cuda,nvtx "


# ECHO option
#echo   $DISTRIBUTED_ARGS 
echo $PROFILE python3 $appdir/30_training_loop.py


# Execute
OUTPUT="-o nsys-rep-B${batch_size}-MLPdim${mlp_dim}_S${image_size}"
echo $OUTPUT $batch_size ${mlp_dim}
time $PROFILE $OUTPUT python3 $appdir/30_training_loop.py \
        --datadir $datadir \
        --image_size $image_size \
        --batch_size ${batch_size} \
        --mlp_dim ${mlp_dim} \
        --epochs ${epochs}
