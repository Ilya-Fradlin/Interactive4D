#!/usr/bin/env bash

export NCCL_SOCKET_IFNAME=en,eth,em,bond
# export CUDA_LAUNCH_BLOCKING=1

sbatch --partition=a40-lo -c 16 --gres=gpu:1 --mem=48G --job-name=interactive4d --time=10-00:00:00 \
--signal=TERM@120 --mail-user=example@domain.com --mail-type=FAIL --output=outputs/%j_validate_batch16.txt scripts/train.sh


