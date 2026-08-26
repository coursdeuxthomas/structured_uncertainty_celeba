#!/bin/bash
#SBATCH --job-name=cov
#SBATCH --partition=short
#SBATCH --time=01:55:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --exclude=node031,node032
#SBATCH --output=logs/cov_%j.out
#
# Entrainement du reseau de covariance, DnCNN gele.
#
#   sbatch train_cov.bash                 modele structure
#   sbatch train_cov.bash --diagonale     reference diagonale
#
# LES DEUX RUNS SONT NECESSAIRES. L'ecart de NLL entre eux est ce qui chiffre
# l'apport de la structure ; sans lui les resultats ne demontrent rien. Ils
# ecrivent sous des noms distincts (cov_* et covdiag_*), donc ils peuvent
# tourner l'un apres l'autre ou en parallele sans se marcher dessus.
#
# --resume est deja dans la ligne de commande ci-dessous : relancer ce meme
# script apres la coupure des 1 h 55 poursuit l'entrainement la ou il s'est
# arrete. Comptez plusieurs relances pour aller au bout des 50 epochs.
#
# node031 et node032 sont exclus pour la meme raison que dans
# train_dncnn.bash : sm_61, incompatible avec le torch de l'env dncnn.

echo "Job started on:"
hostname
date

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dncnn

cd ~/structured_uncertainty_celeba

export CELEBA_DIR=$HOME/data/celeba/img_align_celeba
export CELEBA_CACHE=$HOME/data/celeba/celeba_64_gray.npy

# -u : sortie non tamponnee, sinon le log reste vide plusieurs minutes.
python -u train_cov.py --epochs 50 --resume "$@"

echo "Job finished:"
date
