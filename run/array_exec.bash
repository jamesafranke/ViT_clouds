#!/bin/bash
cwd=`pwd`

image_size=256 #1024
datadir="/home/mu7sgetq/workspace/data/processed/patch_${image_size}"

#for mlp_dim in 1024 2048 ; do
#    for batch_size in 200 256 512 1024 ; do
for mlp_dim in 2048 ; do
    for batch_size in 1500 ; do
    #for batch_size in 16 32 64 128 ; do
	echo $batch_size $mlp_dim
	sbatch $cwd/exec_run_v1.bash  $batch_size $mlp_dim ${datadir} $image_size
    done
 done
