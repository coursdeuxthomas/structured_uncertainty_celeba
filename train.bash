#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Job SLURM d'entraînement.
#
#   sbatch train.bash dncnn     # étape 1 : le débruiteur
#   sbatch train.bash cov       # étape 2 : le réseau de covariance
#   sbatch train.bash cov-diag  # référence diagonale (offdiag = 0)
#
# La partition 'short' est limitée à 1h55 : un entraînement de 50 epochs ne
# tient PAS dans un job. Les scripts python doivent donc sauvegarder
# l'optimiseur et le numéro d'epoch, et repartir de last.pt via --resume.
#
# Relance automatique : mets AUTO_RELANCE=1 pour que le job se resoumette
# tant que l'entraînement n'est pas fini.
#     sbatch --export=ALL,AUTO_RELANCE=1 train.bash dncnn
# Le compteur RELANCE plafonne à MAX_RELANCES pour éviter une boucle infinie.
# ---------------------------------------------------------------------------
#SBATCH --job-name=celeba
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

ETAPE="${1:-dncnn}"
RELANCE="${RELANCE:-0}"
MAX_RELANCES="${MAX_RELANCES:-20}"
AUTO_RELANCE="${AUTO_RELANCE:-0}"

PROJET="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$PROJET"
mkdir -p logs results checkpoints

echo "=============================================================="
echo " Job      : ${SLURM_JOB_ID:-local}   étape : $ETAPE"
echo " Nœud     : $(hostname)"
echo " Relance  : $RELANCE / $MAX_RELANCES"
echo " Début    : $(date)"
echo "=============================================================="

# --- Environnement ---------------------------------------------------------
# Adapte ces deux lignes aux modules de TON cluster ('module avail').
# module load python/3.11
# module load cuda/12.1
source "${CELEBA_VENV:-$PROJET/venv}/bin/activate"

# Données sur le scratch (voir get_data.sh). Adapte si besoin.
export CELEBA_DIR="${CELEBA_DIR:-$PROJET/img_align_celeba}"
export CELEBA_CACHE="${CELEBA_CACHE:-$PROJET/celeba_64_gray.npy}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('GPU vu par torch :', torch.cuda.is_available())"

# --- Entraînement ----------------------------------------------------------
case "$ETAPE" in
    dncnn)    SCRIPT_PY="train_dncnn.py"; ARGS=() ;;
    cov)      SCRIPT_PY="train_cov.py";   ARGS=() ;;
    cov-diag) SCRIPT_PY="train_cov.py";   ARGS=(--diagonale) ;;
    *) echo "étape inconnue : $ETAPE (attendu dncnn | cov | cov-diag)"; exit 2 ;;
esac

if [ ! -f "$SCRIPT_PY" ]; then
    echo "ERREUR : $SCRIPT_PY n'existe pas encore."
    echo "  Ordre de développement (voir roadmap_celeba_dncnn.txt) :"
    echo "  data.py -> dncnn.py -> train_dncnn.py -> VERIF DU RESIDU -> loss.py"
    echo "  -> cov_model.py -> train_cov.py -> eval_cov.py -> denoise.py"
    exit 3
fi

# --resume : les scripts doivent repartir de checkpoints/<etape>_last.pt s'il
# existe, et écrire un fichier vide checkpoints/<etape>_FINI quand la dernière
# epoch est atteinte.
set +e
python "$SCRIPT_PY" --resume "${ARGS[@]}"
CODE=$?
set -e

echo "=============================================================="
echo " Fin      : $(date)   code de sortie : $CODE"
echo "=============================================================="

# --- Relance ---------------------------------------------------------------
if [ "$CODE" -ne 0 ]; then
    echo "Le script a échoué : pas de relance."
    exit "$CODE"
fi
if [ -f "checkpoints/${ETAPE}_FINI" ]; then
    echo "Entraînement terminé."
    exit 0
fi
if [ "$AUTO_RELANCE" = "1" ]; then
    if [ "$RELANCE" -ge "$MAX_RELANCES" ]; then
        echo "Plafond de $MAX_RELANCES relances atteint : on s'arrête."
        exit 0
    fi
    echo "Entraînement non terminé : resoumission (relance $((RELANCE + 1)))."
    sbatch --export=ALL,RELANCE=$((RELANCE + 1)),AUTO_RELANCE=1 \
           "$PROJET/train.bash" "$ETAPE"
else
    echo "Entraînement non terminé. Relance à la main :"
    echo "  sbatch train.bash $ETAPE"
fi
