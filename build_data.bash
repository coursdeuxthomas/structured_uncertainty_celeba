#!/bin/bash
#SBATCH --job-name=celeba_data
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/celeba_data_%j.out
#
# Construit le cache CelebA 64x64 gris. Pas de GPU demandé : c'est du
# décodage JPEG, purement CPU, et un job sans GPU passe beaucoup plus vite
# dans la file.
#
#   mkdir -p logs
#   sbatch build_data.bash

echo "Job started on:"
hostname
date

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dncnn

cd ~/structured_uncertainty_celeba     # adapte si tu as cloné ailleurs

# Où sont les JPEG, où va le cache (830 Mo).
export CELEBA_DIR=$HOME/data/celeba/img_align_celeba
export CELEBA_CACHE=$HOME/data/celeba/celeba_64_gray.npy

echo "Python used:"
which python

echo "Images trouvées :"
find "$CELEBA_DIR" -name '*.jpg' | wc -l
echo "(202599 attendu)"

# --build construit le cache PUIS enchaîne sur les vérifications et écrit
# results/data_preview.png : un seul appel suffit.
python -u data.py --build  # ~10 à 20 min, une seule fois (-u : log en direct)

echo "Job finished:"
date
