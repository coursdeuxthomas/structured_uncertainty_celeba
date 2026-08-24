#!/bin/bash
#SBATCH --job-name=dncnn
#SBATCH --partition=short
#SBATCH --time=01:55:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/dncnn_%j.out
#
#   mkdir -p logs            <- une seule fois, AVANT le premier sbatch
#   sbatch train_dncnn.bash        50 epochs (defaut)
#   sbatch train_dncnn.bash 3      run court de validation
#
# --resume reprend le checkpoint s'il existe : relancer ce meme script apres
# la coupure des 1 h 55 poursuit l'entrainement la ou il s'etait arrete.

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

python train_dncnn.py --epochs "$EPOCHS" --amp --resume

echo "Job finished:"
date
