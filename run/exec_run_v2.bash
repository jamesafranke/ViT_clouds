#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --job-name=profile-v2
###overnight-0921
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4


appdir=${HOME}/ViT_clouds/scripts

# change var
prefetch_factor=15 # num workers x prefetch_factor
num_workers=4

# Model Config
epochs=5
batch_size=60 #60 # 30 per gpu is max
mlp_dim=2048
image_size=1024
patch_size=32
datadir="/home/mu7sgetq/workspace/data/processed/patch_${image_size}"


# Execute
PROFILE="nsys profile --trace=cuda,nvtx --force-overwrite true"
OUTPUT="-o nsys-rep-B${batch_size}-MLPdim${mlp_dim}_S${image_size}_prefetch${prefetch_factor}_nworkers${num_workers}_4gpus_amp_wo-tf32"
time $PROFILE $OUTPUT  torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:6000 $appdir/torchrun_test.py \
        --datadir $datadir \
        --image_size $image_size \
        --patch_size ${patch_size} \
        --batch_size ${batch_size} \
        --mlp_dim ${mlp_dim} \
	--prefetch_factor ${prefetch_factor} \
	--num_workers ${num_workers} \
        --epochs ${epochs}
