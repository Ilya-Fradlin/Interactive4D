#!/bin/bash

export NCCL_SOCKET_IFNAME=en,eth,em,bond,eth0
export OMP_NUM_THREADS=16  # speeds up MinkowskiEngine
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3

# debugging options:
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TORCH_CPP_LOG_LEVEL=INFO
# export GLOO_LOG_LEVEL=DEBUG
# export NCCL_DEBUG=INFO 
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1

# run the main training script
srun python main.py