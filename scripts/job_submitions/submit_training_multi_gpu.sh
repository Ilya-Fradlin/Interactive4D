#!/bin/bash -x

export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"


module load Stages/2023
module load GCC/11.3.0
module load OpenMPI/4.1.4
module load CUDA/11.7
module load Python/3.10.4
module load PyTorch/1.12.0-CUDA-11.7
module load PyTorch-Lightning
pyenv activate interactive4d

sbatch --account=objectsegvideo --nodes=4 --ntasks-per-node=4 --gres=gpu:4 --output=outputs/%j_Interactive4d.txt --time=00-23:59:59 --partition=booster --signal=TERM@120 --mail-user=example@domain.com --mail-type=BEGIN,END,FAIL scripts/job_submitions/run_on_node.sh

