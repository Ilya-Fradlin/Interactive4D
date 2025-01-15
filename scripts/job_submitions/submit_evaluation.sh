#!/usr/bin/env bash
export NCCL_SOCKET_IFNAME=en,eth,em,bond

sbatch --partition=a40-lo -c 16 --gres=gpu:1 --mem=48G --job-name=evaluate_interactive4d --time=2-00:00:00 \
--signal=TERM@120 --mail-user=example@domain.com --mail-type=FAIL --output=outputs/%j_evaluate_interactive4d.txt scripts/evaluate.sh
