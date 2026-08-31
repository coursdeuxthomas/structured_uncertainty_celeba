#!/bin/bash
#SBATCH --job-name=dncnn
#SBATCH --partition=short
#SBATCH --time=01:55:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --exclude=node031,node032
#
# node031 (Pascal) and node032 (GTX 1080 Ti) are sm_61. The torch in the
# dncnn env is compiled for sm_75 and above: on those two nodes it fails on
# the very first CUDA operation with
#   "no kernel image is available for execution on the device".
# The other GPU nodes of the partition are fine: Tesla T4 and Quadro
# RTX 6000 in sm_75, L40S in sm_89 (covered by the sm_86 binaries).
# To aim for the fastest one at the price of a longer wait:
#   #SBATCH --gres=gpu:l40s:1     instead of the gres line above
#SBATCH --output=logs/dncnn_%j.out
#
#   mkdir -p logs            <- once only, BEFORE the first sbatch
#   sbatch train_dncnn.bash        50 epochs (default)
#   sbatch train_dncnn.bash 3      short validation run
#
# --resume picks the checkpoint back up if it exists: relaunching this same
# script after the 1 h 55 cutoff carries on training where it left off.

EPOCHS=${1:-50}

echo "Job started on:"
hostname
date

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dncnn

cd ~/structured_uncertainty_celeba

export CELEBA_DIR=$HOME/data/celeba/img_align_celeba
export CELEBA_CACHE=$HOME/data/celeba/celeba_64_gray.npy

echo "Python used:"
which python
python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# -u : unbuffered output. Without it, when stdout is a FILE and not a
# terminal, Python piles the prints up in an 8 KB buffer and the log stays
# empty for several minutes. tail -f would show nothing.
python -u train_dncnn.py --epochs "$EPOCHS" --amp --resume

echo "Job finished:"
date
