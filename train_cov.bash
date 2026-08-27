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
#SBATCH --requeue
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

# --------------------------------------------------------------------------
# GARDE-FOU GPU, ajoute le 27 aout apres un echec en trois secondes.
#
# Sur ce cluster, un GPU deja pris par un autre processus (mode de calcul
# exclusif) laisse torch.cuda.is_available() repondre True et affiche meme le
# nom de la carte : ces appels ne creent pas de contexte CUDA. C'est la
# PREMIERE allocation qui echoue, avec
#     CUDA error: CUDA-capable device(s) is/are busy or unavailable
#
# Sans ce test, le job meurt en trois secondes et l'echec se propage
# instantanement a toute la chaine --dependency=afterany : six jobs disparus
# en vingt secondes. On remet plutot le job en file d'attente. Il GARDE SON
# JOBID, donc la chaine derriere lui reste intacte, et il repartira sur un
# autre noeud.
#
# Le compteur SLURM_RESTART_COUNT evite la boucle infinie si le probleme n'est
# pas un noeud isole mais une limite de compte.
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

# -u : sortie non tamponnee, sinon le log reste vide plusieurs minutes.
python -u train_cov.py --epochs 50 --resume "$@"
code=$?

# Le code de sortie de python doit remonter a SLURM. Sans le "exit $code"
# ci-dessous, le dernier "date" reussit et le job est enregistre COMPLETED
# meme si l'entrainement a plante : sacct affiche alors des succes partout et
# on cherche la panne ailleurs. C'est exactement ce qui s'est passe le
# 27 aout avec les six jobs diagonaux morts en huit secondes.
echo "Job finished:"
date
exit $code
