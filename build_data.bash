#!/bin/bash
#SBATCH --job-name=celeba_data
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/celeba_data_%j.out
#
# Builds the 64x64 grayscale CelebA cache. No GPU requested: this is JPEG
# decoding, purely CPU work, and a job without a GPU gets through the queue
# much faster.
#
#   mkdir -p logs
#   sbatch build_data.bash

echo "Job started on:"
hostname
date

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dncnn

cd ~/structured_uncertainty_celeba     # adjust if you cloned elsewhere

# Where the JPEGs are, where the cache goes (830 MB).
export CELEBA_DIR=$HOME/data/celeba/img_align_celeba
export CELEBA_CACHE=$HOME/data/celeba/celeba_64_gray.npy

echo "Python used:"
which python

echo "Images trouvées :"
find "$CELEBA_DIR" -name '*.jpg' | wc -l
echo "(202599 attendu)"

# --build builds the cache AND THEN goes on to the checks and writes
# results/data_preview.png: a single call is enough.
python -u data.py --build  # ~10 to 20 min, once only (-u: live log)

echo "Job finished:"
date
