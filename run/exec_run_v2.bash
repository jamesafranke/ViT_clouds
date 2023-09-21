#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=overnight-0921
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4


appdir=${HOME}/ViT_clouds/scripts

# Model Config
epochs=100
batch_size=64
mlp_dim=2048
image_size=1024
patch_size=32
datadir="/home/mu7sgetq/workspace/data/processed/patch_${image_size}"


# Execute
PROFILE="nsys profile --trace=cuda,nvtx "
OUTPUT="-o nsys-rep-B${batch_size}-MLPdim${mlp_dim}_S${image_size}"
time $PROFILE $OUTPUT  torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:6000 $appdir/torchrun_test.py \
        --datadir $datadir \
        --image_size $image_size \
        --patch_size ${patch_size} \
        --batch_size ${batch_size} \
        --mlp_dim ${mlp_dim} \
        --epochs ${epochs}
