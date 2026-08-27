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
#SBATCH --requeue
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

# --------------------------------------------------------------------------
# GARDE-FOU GPU, le meme que dans train_cov.bash.
#
# Ajoute le 27 aout apres la mort de ce job en huit secondes sur node023.
# Sur ce cluster, un GPU deja pris (mode exclusif) laisse
# torch.cuda.is_available() repondre True et affiche meme le nom de la carte :
# ces appels ne creent pas de contexte CUDA. C'est la PREMIERE allocation qui
# echoue -- ici dans le torch.load du DnCNN -- avec
#     CUDA error: CUDA-capable device(s) is/are busy or unavailable
#
# On remet le job en file plutot que de le laisser mourir. Il GARDE SON JOBID
# et repartira sur un autre noeud. SLURM_RESTART_COUNT evite la boucle
# infinie si le probleme n'est pas un noeud isole mais une limite de compte.
# --------------------------------------------------------------------------
if ! python -c "import torch; torch.zeros(1, device='cuda')" 2>/dev/null; then
  echo "GPU inutilisable sur $(hostname) : deja occupe, ou mode exclusif."
  if [ "${SLURM_RESTART_COUNT:-0}" -lt 5 ]; then
    echo "Remise en file du job $SLURM_JOB_ID (tentative ${SLURM_RESTART_COUNT:-0})."
    scontrol requeue "$SLURM_JOB_ID"
    sleep 60
    exit 0
  fi
  echo "Cinq tentatives ont echoue : ce n'est pas un noeud isole. On s'arrete."
  exit 1
fi
echo "GPU utilisable, demarrage."

# -u : sortie non tamponnee, meme raison que dans train_dncnn.bash.
# main.py --evaluation enchaine les deux scripts et s'arrete au premier echec.
python -u main.py --evaluation "$@"
code=$?

# Le code de sortie de python doit remonter a SLURM. Sans le "exit $code"
# ci-dessous, le dernier "date" reussit et le job est enregistre COMPLETED
# meme si l'entrainement a plante : sacct affiche alors des succes partout et
# on cherche la panne ailleurs. C'est exactement ce qui s'est passe le
# 27 aout avec les six jobs diagonaux morts en huit secondes.
echo "Job finished:"
date
exit $code
