#!/bin/bash
#SBATCH --job-name=surapp
#SBATCH --partition=short
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --exclude=node031,node032
#SBATCH --output=logs/surapp_%j.out
#
# Test de surapprentissage volontaire : le critere d'arret de la phase 4.
# Il verifie loss.py et cov_model.py ENSEMBLE, et dure environ deux minutes.
#
# EN DIRECT DANS LE TERMINAL (recommande, une seule ligne a taper) :
#   srun --partition=short --time=00:20:00 --cpus-per-task=4 --mem=16G --gres=gpu:1 --exclude=node031,node032 bash surapprentissage.bash
#
# EN DIFFERE, la sortie va dans logs/surapp_<jobid>.out :
#   mkdir -p logs        <- une seule fois, si ce n'est pas deja fait
#   sbatch surapprentissage.bash
#
# node031 et node032 sont exclus pour la meme raison que dans
# train_dncnn.bash : ils sont en sm_61 et le torch de l'env dncnn echoue des
# la premiere operation CUDA.

echo "Job started on:"
hostname
date

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dncnn

cd ~/structured_uncertainty_celeba

export CELEBA_DIR=$HOME/data/celeba/img_align_celeba
export CELEBA_CACHE=$HOME/data/celeba/celeba_64_gray.npy

# -u : sortie non tamponnee, pour que le log se remplisse en direct quand on
# passe par sbatch. Meme raison que dans train_dncnn.bash.
python -u surapprentissage.py --checkpoint checkpoints/dncnn_best.pt "$@"

echo "Job finished:"
date
