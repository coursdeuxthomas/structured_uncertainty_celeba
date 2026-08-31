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
# Deliberate overfitting test: the stopping criterion of phase 4.
# It checks loss.py and cov_model.py TOGETHER, and takes about two minutes.
#
# LIVE IN THE TERMINAL (recommended, a single line to type):
#   srun --partition=short --time=00:20:00 --cpus-per-task=4 --mem=16G --gres=gpu:1 --exclude=node031,node032 bash surapprentissage.bash
#
# DEFERRED, the output goes to logs/surapp_<jobid>.out:
#   mkdir -p logs        <- once only, if not already done
#   sbatch surapprentissage.bash
#
# node031 and node032 are excluded for the same reason as in
# train_dncnn.bash: they are sm_61 and the torch of the dncnn env fails on
# the very first CUDA operation.

echo "Job started on:"
hostname
date

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dncnn

cd ~/structured_uncertainty_celeba

export CELEBA_DIR=$HOME/data/celeba/img_align_celeba
export CELEBA_CACHE=$HOME/data/celeba/celeba_64_gray.npy

# -u : unbuffered output, so that the log fills up live when going through
# sbatch. Same reason as in train_dncnn.bash.
python -u surapprentissage.py --checkpoint checkpoints/dncnn_best.pt "$@"

echo "Job finished:"
date
