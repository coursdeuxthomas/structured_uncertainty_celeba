"""
CelebA preprocessing and PyTorch Dataset for denoising.

Pipeline: colour JPEG 178x218  ->  centre crop  ->  64x64  ->  grayscale  ->
single uint8 cache.

The cache is built ONLY ONCE (~10 to 20 minutes), then loaded into memory on
every run (~830 MB as uint8). Re-reading 200,000 JPEGs at every epoch would
be the bottleneck of the whole project.

Usage:
    python data.py --build        # builds the cache (to be done once)
    python data.py                # checks the cache and saves a preview
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import Dataset

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
# Paths overridable through environment variables. On a cluster, the JPEGs
# (1.4 GB) and the cache (830 MB) must go to the scratch space and not to the
# home directory, whose quota is small:
#     export CELEBA_DIR=$SCRATCH/celeba/img_align_celeba
#     export CELEBA_CACHE=$SCRATCH/celeba/celeba_64_gray.npy
RAW_DIR = os.environ.get("CELEBA_DIR", "img_align_celeba")      # raw JPEGs
CACHE_PATH = os.environ.get("CELEBA_CACHE", "celeba_64_gray.npy")  # uint8 [N,64,64]

IMAGE_SIZE = 64      # final resolution (paper)
CROP = 148           # centre crop before resizing

# Official CelebA split. The files are numbered sequentially and the official
# partition is contiguous:
#   000001 .. 162770  -> train  (partition 0)
#   162771 .. 182637  -> valid  (partition 1)
#   182638 .. 202599  -> test   (partition 2)
# The paper uses train + valid for training, and test for testing.
N_TRAIN = 182637     # = 162,770 + 19,867
N_TOTAL = 202599

# Noise: sigma is expressed in [0, 1] units (standard DnCNN convention).
# CAUTION: the images are normalised to [-1, 1], so the scale there is TWICE
# as large. The conversion is done in CelebADataset.
SIGMA = 25.0 / 255.0


# --------------------------------------------------------------------------
# Cache construction
# --------------------------------------------------------------------------
def build_cache(raw_dir=RAW_DIR, cache_path=CACHE_PATH,
                image_size=IMAGE_SIZE, crop=CROP):
    """
    Walks through every JPEG, preprocesses it, and writes a single uint8 array.

    Cropping: the aligned images are 178x218. We take a centred square of
    `crop` x `crop` (148 by default, the usual convention for CelebA 64x64:
    it recentres on the face and removes the background and the top of the
    skull), then we resize to `image_size`.
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

        # Square centre crop.
        largeur, hauteur = img.size
        gauche = (largeur - crop) // 2
        haut = (hauteur - crop) // 2
        img = img.crop((gauche, haut, gauche + crop, haut + crop))

        # Resizing, then grayscale.
        img = img.resize((image_size, image_size), Image.BILINEAR).convert("L")

        out[k] = np.asarray(img, dtype=np.uint8)

        if (k + 1) % 10000 == 0:
            print("  %6d / %d" % (k + 1, n), flush=True)

    np.save(cache_path, out)
    print("Cache écrit : %s  (%.0f Mo)" % (cache_path, out.nbytes / 1e6))
    return out


def load_cache(cache_path=CACHE_PATH):
    """Load the uint8 cache [N, 64, 64]. Raises an error if it is missing."""
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
    Returns, for each example:
        x : clean image     [1, 64, 64], float32 in [-1, 1]
        y : noisy image     [1, 64, 64], y = x + sigma * N(0, I)

    The noise is drawn ON THE FLY at every access. This is deliberate: the
    network sees an infinity of noise realisations for one and the same
    image, which prevents any memorisation of the residual and plays the role
    of data augmentation. The cost is nil (a single randn).

    split : "train" (first 182,637 images) or "test" (last 19,962).
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

        # The images live in [-1, 1]: the standard deviation of the noise is
        # therefore twice its value expressed in [0, 1] units.
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

    # Checks.
    train_ds = CelebADataset("train")
    test_ds = CelebADataset("test")
    print("train : %d exemples | test : %d exemples" % (len(train_ds), len(test_ds)))
    print("attendu article : 182637 / 19962")

    s = train_ds[0]
    print("x : %s  dans [%.2f, %.2f]" % (tuple(s["x"].shape), s["x"].min(), s["x"].max()))
    print("y : %s  dans [%.2f, %.2f]" % (tuple(s["y"].shape), s["y"].min(), s["y"].max()))
    print("sigma effectif (unités [-1,1]) : %.4f" % train_ds.sigma)
    print("bruit mesuré (y - x) std       : %.4f" % (s["y"] - s["x"]).std().item())

    # The noise must be different at every access.
    a, b = train_ds[0]["y"], train_ds[0]["y"]
    print("bruit retiré à chaque accès    :", not torch.equal(a, b))

    # Visual preview.
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
