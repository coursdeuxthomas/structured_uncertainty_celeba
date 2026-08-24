#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Prépare l'environnement Python du projet sur le cluster.
#
#   bash setup_cluster.sh
#
# Idempotent : relançable sans risque.
#
# Variables d'environnement reconnues :
#   CELEBA_VENV        emplacement du venv          (défaut : ./venv)
#   CELEBA_TORCH_INDEX index des wheels torch       (défaut : cu121)
#   CELEBA_USE_CONDA=1 utilise conda au lieu du venv
# ---------------------------------------------------------------------------
set -euo pipefail

PROJET="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${CELEBA_VENV:-$PROJET/venv}"
TORCH_INDEX="${CELEBA_TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
PAQUETS=(numpy matplotlib pillow kaggle)

echo "=============================================================="
echo " Projet   : $PROJET"
echo " Machine  : $(hostname)"
echo "=============================================================="

# --- 1. Modules ------------------------------------------------------------
# Sur la plupart des clusters SLURM, python et CUDA viennent de 'module'.
# Adapte les noms ci-dessous à ceux de TON cluster : 'module avail python'.
if command -v module >/dev/null 2>&1; then
    echo
    echo "-- 'module' détecté. Modules python/cuda disponibles :"
    (module avail python cuda 2>&1 || true) | head -25
    echo
    echo "   Si le python trouvé plus bas n'est pas le bon, lance"
    echo "   'module load <ton_module_python>' puis relance ce script."
    # Décommente et adapte une fois que tu connais les noms exacts :
    # module load python/3.11
    # module load cuda/12.1
fi

echo
echo "-- Interpréteur : $(command -v python3)  ($(python3 -V 2>&1))"

# --- 2. Environnement ------------------------------------------------------
if [ "${CELEBA_USE_CONDA:-0}" = "1" ]; then
    # ---- variante conda ----
    ENVNAME="${CELEBA_CONDA_ENV:-celeba}"
    echo
    echo "-- conda : environnement '$ENVNAME'"
    if ! conda env list | grep -qE "^$ENVNAME[[:space:]]"; then
        conda create -y -n "$ENVNAME" python=3.11
    fi
    # 'conda activate' n'est pas disponible dans un script non interactif
    # sans cette ligne :
    eval "$(conda shell.bash hook)"
    conda activate "$ENVNAME"
    ACTIVATION="conda activate $ENVNAME"
else
    # ---- variante venv (par défaut) ----
    echo
    if [ ! -d "$VENV" ]; then
        echo "-- Création du venv : $VENV"
        python3 -m venv "$VENV"
    else
        echo "-- venv déjà présent : $VENV"
    fi
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    ACTIVATION="source $VENV/bin/activate"
fi

python -m pip install --upgrade pip -q

# --- 3. PyTorch ------------------------------------------------------------
echo
if python -c "import torch" 2>/dev/null; then
    echo "-- torch déjà installé : $(python -c 'import torch; print(torch.__version__)')"
else
    echo "-- Installation de torch depuis $TORCH_INDEX"
    echo "   (si la version de CUDA ne correspond pas à celle du cluster,"
    echo "    relance avec CELEBA_TORCH_INDEX=.../whl/cu118 par exemple)"
    python -m pip install torch --index-url "$TORCH_INDEX"
fi

echo "-- Installation des autres dépendances"
python -m pip install -q "${PAQUETS[@]}"

# --- 4. Vérification ------------------------------------------------------
echo
echo "-- Vérification :"
python - <<'PY'
import torch, numpy, PIL, matplotlib
print("   numpy      %s" % numpy.__version__)
print("   torch      %s" % torch.__version__)
print("   CUDA compilée dans torch : %s" % torch.version.cuda)
dispo = torch.cuda.is_available()
print("   GPU visible : %s" % dispo)
if dispo:
    print("   -> %s" % torch.cuda.get_device_name(0))
else:
    print("   (normal sur un nœud de connexion : les GPU ne sont visibles")
    print("    que dans un job. Vérifie avec : srun --gres=gpu:1 --pty bash)")
PY

echo
echo "=============================================================="
echo " Terminé. Pour activer l'environnement plus tard :"
echo "   $ACTIVATION"
echo "=============================================================="
