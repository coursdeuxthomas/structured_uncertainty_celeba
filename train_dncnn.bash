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
# node031 (Pascal) et node032 (GTX 1080 Ti) sont en sm_61. Le torch de l'env
# dncnn est compile pour sm_75 et au-dela : sur ces deux noeuds il echoue des
# la premiere operation CUDA avec
#   "no kernel image is available for execution on the device".
# Les autres noeuds GPU de la partition conviennent : Tesla T4 et Quadro
# RTX 6000 en sm_75, L40S en sm_89 (couvert par les binaires sm_86).
# Pour viser le plus rapide au prix d'une attente plus longue :
#   #SBATCH --gres=gpu:l40s:1     a la place de la ligne gres ci-dessus
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

# -u : sortie non tamponnee. Sans lui, quand stdout est un FICHIER et non un
# terminal, Python accumule les print dans un tampon de 8 Ko et le log reste
# vide plusieurs minutes. tail -f ne montrerait rien.
python -u train_dncnn.py --epochs "$EPOCHS" --amp --resume

echo "Job finished:"
date
