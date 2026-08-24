"""
Prétraitement de CelebA et Dataset PyTorch pour le débruitage.

Chaîne : JPEG couleur 178x218  ->  recadrage centré  ->  64x64  ->  gris  ->
cache uint8 unique.

Le cache est construit UNE SEULE FOIS (~10 à 20 minutes), puis chargé en
mémoire à chaque run (~830 Mo en uint8). Relire 200 000 JPEG à chaque epoch
serait le goulot d'étranglement de tout le projet.

Usage :
    python data.py --build        # construit le cache (à faire une fois)
    python data.py                # vérifie le cache et sauvegarde un aperçu
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import Dataset

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
RAW_DIR = "img_align_celeba"          # dossier des JPEG bruts
CACHE_PATH = "celeba_64_gray.npy"     # cache uint8 [N, 64, 64]

IMAGE_SIZE = 64      # résolution finale (article)
CROP = 148           # recadrage centré avant redimensionnement

# Split officiel CelebA. Les fichiers sont numérotés séquentiellement et la
# partition officielle est contiguë :
#   000001 .. 162770  -> train  (partition 0)
#   162771 .. 182637  -> valid  (partition 1)
#   182638 .. 202599  -> test   (partition 2)
# L'article utilise train + valid pour l'entraînement, et test pour le test.
N_TRAIN = 182637     # = 162 770 + 19 867
N_TOTAL = 202599

# Bruit : sigma est exprimé en unités [0, 1] (convention DnCNN standard).
# ATTENTION : les images sont normalisées dans [-1, 1], donc l'échelle y est
# DEUX FOIS plus grande. La conversion est faite dans CelebADataset.
SIGMA = 25.0 / 255.0


# --------------------------------------------------------------------------
# Construction du cache
# --------------------------------------------------------------------------
def build_cache(raw_dir=RAW_DIR, cache_path=CACHE_PATH,
                image_size=IMAGE_SIZE, crop=CROP):
    """
    Parcourt tous les JPEG, les prétraite, et écrit un unique tableau uint8.

    Recadrage : les images alignées font 178x218. On prend un carré centré de
    `crop` x `crop` (148 par défaut, convention usuelle pour CelebA 64x64 :
    cela recentre sur le visage et supprime le fond et le haut du crâne), puis
    on redimensionne en `image_size`.
    """
    from PIL import Image

    fichiers = sorted(f for f in os.listdir(raw_dir) if f.lower().endswith(".jpg"))
    n = len(fichiers)
    print("JPEG trouvés : %d (attendu %d)" % (n, N_TOTAL))
    if n != N_TOTAL:
        print("  /!\\ nombre inattendu : le split par indice sera faux.")

    out = np.zeros((n, image_size, image_size), dtype=np.uint8)

    for k, nom in enumerate(fichiers):
        img = Image.open(os.path.join(raw_dir, nom))

        # Recadrage centré carré.
        largeur, hauteur = img.size
        gauche = (largeur - crop) // 2
        haut = (hauteur - crop) // 2
        img = img.crop((gauche, haut, gauche + crop, haut + crop))

        # Redimensionnement puis niveaux de gris.
        img = img.resize((image_size, image_size), Image.BILINEAR).convert("L")

        out[k] = np.asarray(img, dtype=np.uint8)

        if (k + 1) % 10000 == 0:
            print("  %6d / %d" % (k + 1, n), flush=True)

    np.save(cache_path, out)
    print("Cache écrit : %s  (%.0f Mo)" % (cache_path, out.nbytes / 1e6))
    return out


def load_cache(cache_path=CACHE_PATH):
    """Charge le cache uint8 [N, 64, 64]. Lève une erreur s'il n'existe pas."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            "Cache introuvable (%s). Lancez d'abord : python data.py --build"
            % cache_path
        )
    return np.load(cache_path)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class CelebADataset(Dataset):
    """
    Renvoie, pour chaque exemple :
        x : image propre    [1, 64, 64], float32 dans [-1, 1]
        y : image bruitée   [1, 64, 64], y = x + sigma * N(0, I)

    Le bruit est tiré A LA VOLEE à chaque accès. C'est volontaire : le réseau
    voit une infinité de réalisations du bruit pour une même image, ce qui
    empêche toute mémorisation du résidu et joue le rôle d'augmentation de
    données. Le coût est nul (un simple randn).

    split : "train" (182 637 premières images) ou "test" (19 962 dernières).
    """

    def __init__(self, split="train", cache_path=CACHE_PATH, sigma=SIGMA,
                 n_train=N_TRAIN):
        images = load_cache(cache_path)

        if split == "train":
            self.images = images[:n_train]
        elif split == "test":
            self.images = images[n_train:]
        else:
            raise ValueError("split doit valoir 'train' ou 'test'")

        # Les images sont dans [-1, 1] : l'écart-type du bruit y est donc le
        # double de sa valeur exprimée en unités [0, 1].
        self.sigma = 2.0 * sigma
        self.split = split

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # uint8 [0, 255]  ->  float32 [-1, 1]
        x = self.images[idx].astype(np.float32) / 127.5 - 1.0
        x = torch.from_numpy(x).unsqueeze(0)          # [1, 64, 64]

        y = x + self.sigma * torch.randn_like(x)
        return {"x": x, "y": y}


# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prétraitement CelebA.")
    parser.add_argument("--build", action="store_true",
                        help="construit le cache depuis les JPEG (une seule fois).")
    args = parser.parse_args()

    if args.build:
        build_cache()

    # Vérifications.
    train_ds = CelebADataset("train")
    test_ds = CelebADataset("test")
    print("train : %d exemples | test : %d exemples" % (len(train_ds), len(test_ds)))
    print("attendu article : 182637 / 19962")

    s = train_ds[0]
    print("x : %s  dans [%.2f, %.2f]" % (tuple(s["x"].shape), s["x"].min(), s["x"].max()))
    print("y : %s  dans [%.2f, %.2f]" % (tuple(s["y"].shape), s["y"].min(), s["y"].max()))
    print("sigma effectif (unités [-1,1]) : %.4f" % train_ds.sigma)
    print("bruit mesuré (y - x) std       : %.4f" % (s["y"] - s["x"]).std().item())

    # Le bruit doit être différent à chaque accès.
    a, b = train_ds[0]["y"], train_ds[0]["y"]
    print("bruit retiré à chaque accès    :", not torch.equal(a, b))

    # Aperçu visuel.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs("results", exist_ok=True)
        fig, axes = plt.subplots(2, 6, figsize=(14, 5))
        for k in range(6):
            s = train_ds[k * 1000]
            for r, (img, titre) in enumerate([(s["x"], "x propre"), (s["y"], "y bruitée")]):
                axes[r, k].imshow(img[0].numpy(), cmap="gray", vmin=-1, vmax=1)
                axes[r, k].set_title(titre, fontsize=9)
                axes[r, k].axis("off")
        plt.tight_layout()
        plt.savefig("results/data_preview.png", dpi=140)
        plt.close()
        print("aperçu : results/data_preview.png")
    except Exception as e:
        print("aperçu non généré :", e)
