#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Récupère CelebA sur le cluster et construit le cache 64x64 gris.
#
#   bash get_data.sh
#
# Idempotent : si les images ou le cache sont déjà là, les étapes sont sautées.
#
# Variables d'environnement reconnues :
#   CELEBA_DIR    dossier des JPEG        (défaut : ./img_align_celeba)
#   CELEBA_CACHE  fichier .npy du cache   (défaut : ./celeba_64_gray.npy)
#   CELEBA_RAW    dossier de décompression Kaggle (défaut : ./celeba_raw)
#
# Sur un cluster, mets les données sur le SCRATCH et non dans le home
# (830 Mo de cache + 1,4 Go de JPEG dépassent souvent le quota) :
#   export CELEBA_DIR=$SCRATCH/celeba/img_align_celeba
#   export CELEBA_CACHE=$SCRATCH/celeba/celeba_64_gray.npy
# ---------------------------------------------------------------------------
set -euo pipefail

PROJET="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${CELEBA_DIR:-$PROJET/img_align_celeba}"
CACHE="${CELEBA_CACHE:-$PROJET/celeba_64_gray.npy}"
RAW="${CELEBA_RAW:-$PROJET/celeba_raw}"
N_ATTENDU=202599

compte_jpg() { find "$1" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l; }

echo "=============================================================="
echo " JPEG  : $DEST"
echo " Cache : $CACHE"
echo "=============================================================="

# --- 1. Les images sont-elles déjà là et complètes ? -----------------------
N=$(compte_jpg "$DEST")
if [ "$N" -eq "$N_ATTENDU" ]; then
    echo "-- $N JPEG déjà présents : téléchargement inutile."
else
    if [ "$N" -gt 0 ]; then
        echo "-- $N JPEG présents sur $N_ATTENDU : téléchargement INCOMPLET,"
        echo "   on relance le téléchargement complet."
    fi

    # --- 2. Identifiants Kaggle -------------------------------------------
    if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
        echo
        echo "ERREUR : $HOME/.kaggle/kaggle.json est absent."
        echo
        echo "  Sur kaggle.com : Settings > API > Create New Token."
        echo "  Envoie le fichier obtenu sur le cluster, puis :"
        echo "      mkdir -p ~/.kaggle"
        echo "      mv kaggle.json ~/.kaggle/"
        echo "      chmod 600 ~/.kaggle/kaggle.json"
        exit 1
    fi
    chmod 600 "$HOME/.kaggle/kaggle.json"

    if ! command -v kaggle >/dev/null 2>&1; then
        echo "ERREUR : la commande 'kaggle' est introuvable."
        echo "  Active d'abord l'environnement : source venv/bin/activate"
        exit 1
    fi

    # --- 3. Téléchargement -------------------------------------------------
    echo "-- Téléchargement du dataset (~1,4 Go) vers $RAW"
    mkdir -p "$RAW"
    kaggle datasets download -d jessicali9530/celeba-dataset -p "$RAW" --unzip

    # --- 4. Aplatissement --------------------------------------------------
    # L'archive Kaggle contient img_align_celeba/img_align_celeba/*.jpg
    SRC="$RAW/img_align_celeba/img_align_celeba"
    [ -d "$SRC" ] || SRC="$RAW/img_align_celeba"
    if [ ! -d "$SRC" ]; then
        echo "ERREUR : impossible de trouver les JPEG dans $RAW"
        find "$RAW" -maxdepth 2 -type d
        exit 1
    fi
    mkdir -p "$DEST"
    if [ "$(cd "$SRC" && pwd)" != "$(cd "$DEST" && pwd)" ]; then
        echo "-- Déplacement des JPEG vers $DEST"
        find "$SRC" -maxdepth 1 -name '*.jpg' -exec mv -t "$DEST" {} +
    fi

    # Le fichier de partition officiel, utile pour vérifier le split.
    for f in list_eval_partition.csv list_attr_celeba.csv; do
        [ -f "$RAW/$f" ] && cp -n "$RAW/$f" "$PROJET/" 2>/dev/null || true
    done
fi

# --- 5. Vérification du compte --------------------------------------------
N=$(compte_jpg "$DEST")
echo
echo "-- JPEG dans $DEST : $N (attendu $N_ATTENDU)"
if [ "$N" -ne "$N_ATTENDU" ]; then
    echo
    echo "  /!\\ Le compte ne tombe pas juste."
    echo "      data.py découpe train/test PAR INDICE en supposant les"
    echo "      202 599 images présentes et triées : le split serait FAUX."
    echo "      Corrige avant d'entraîner quoi que ce soit."
    exit 1
fi

# Contrôle des bornes : les noms doivent aller de 000001.jpg à 202599.jpg.
PREMIER=$(find "$DEST" -maxdepth 1 -name '*.jpg' -printf '%f\n' | sort | head -1)
DERNIER=$(find "$DEST" -maxdepth 1 -name '*.jpg' -printf '%f\n' | sort | tail -1)
echo "-- Bornes : $PREMIER .. $DERNIER  (attendu 000001.jpg .. 202599.jpg)"

# --- 6. Construction du cache ---------------------------------------------
echo
if [ -f "$CACHE" ]; then
    echo "-- Cache déjà présent : $CACHE ($(du -h "$CACHE" | cut -f1))"
else
    echo "-- Construction du cache (10 à 20 min, une seule fois)"
    CELEBA_DIR="$DEST" CELEBA_CACHE="$CACHE" python "$PROJET/data.py" --build
fi

echo
echo "=============================================================="
echo " Données prêtes."
echo " Pense à exporter ces variables avant d'entraîner :"
echo "   export CELEBA_DIR=$DEST"
echo "   export CELEBA_CACHE=$CACHE"
echo "=============================================================="
