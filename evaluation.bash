#!/bin/bash
#SBATCH --job-name=evalcov
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --exclude=node031,node032
#SBATCH --output=logs/eval_%j.out
#
# Les deux etapes d'evaluation, qui vont ensemble et tiennent en quelques
# minutes sur GPU : eval_cov.py (NLL, calibration, figures) puis denoise.py
# (projection du residu, MSE finale).
#
# EN DIRECT DANS LE TERMINAL :
#   srun --partition=short --time=00:30:00 --cpus-per-task=4 --mem=16G --gres=gpu:1 --exclude=node031,node032 bash evaluation.bash
#
# EN DIFFERE :
#   sbatch evaluation.bash
#
# A NE LANCER QU'APRES LES DEUX ENTRAINEMENTS : le structure ET la reference
# diagonale. Sans checkpoints/covdiag_best.pt, eval_cov.py tourne quand meme
# mais previent, en gros caracteres, que ses chiffres ne demontrent rien.
#
# node031 et node032 sont exclus pour la meme raison que dans les autres
# scripts : sm_61, incompatible avec le torch de l'env dncnn.

echo "Job started on:"
hostname
date

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dncnn

cd ~/structured_uncertainty_celeba

export CELEBA_DIR=$HOME/data/celeba/img_align_celeba
export CELEBA_CACHE=$HOME/data/celeba/celeba_64_gray.npy

# -u : sortie non tamponnee, meme raison que dans train_dncnn.bash.
# main.py --evaluation enchaine les deux scripts et s'arrete au premier echec.
python -u main.py --evaluation "$@"

echo "Job finished:"
date
