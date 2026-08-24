"""
Preuve empirique que les visages de CelebA sont alignés.

    export CELEBA_CACHE=$HOME/data/celeba/celeba_64_gray.npy
    python verifier_alignement.py

Principe : si les visages sont alignés, l'image MOYENNE du dataset est un
visage net — yeux, nez et bouche restent visibles parce qu'ils tombent au même
endroit dans toutes les images. S'ils ne l'étaient pas, la moyenne serait une
bouillie grise.

Le script affiche côte à côte la vraie moyenne et la moyenne obtenue après
décalage aléatoire des mêmes images : c'est à quoi ressemblerait le dataset
sans l'alignement fait en amont par les auteurs de CelebA.

Écrit results/alignement.png
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE = os.environ.get("CELEBA_CACHE", "celeba_64_gray.npy")
N = 20000          # échantillon : la moyenne est déjà stable bien avant
DECALAGE = 6       # amplitude du désalignement simulé, en pixels


def moyenne_desalignee(images, amplitude, graine=0):
    """Décale chaque image d'un vecteur aléatoire, puis moyenne."""
    rng = np.random.default_rng(graine)
    dy = rng.integers(-amplitude, amplitude + 1, size=len(images))
    dx = rng.integers(-amplitude, amplitude + 1, size=len(images))
    acc = np.zeros(images.shape[1:], dtype=np.float64)
    for k in range(len(images)):
        acc += np.roll(images[k], (dy[k], dx[k]), axis=(0, 1))
    return acc / len(images)


# mmap_mode : on ne charge pas les 830 Mo pour lire les 20 000 premières.
images = np.load(CACHE, mmap_mode="r")
print("cache  : %s" % CACHE)
print("forme  : %s  dtype %s" % (images.shape, images.dtype))

sub = np.asarray(images[:N], dtype=np.float32)
n_reel = len(sub)
print("échantillon : %d images" % n_reel)

moyenne = sub.mean(axis=0)
ecart = sub.std(axis=0)
moyenne_floue = moyenne_desalignee(sub, DECALAGE)

# Le contraste de l'image moyenne chiffre l'alignement : plus il est élevé,
# plus les visages se superposent bien.
c_align = moyenne.std()
c_desalign = moyenne_floue.std()
print("contraste de la moyenne alignée    : %.2f" % c_align)
print("contraste après décalage +/-%d px   : %.2f" % (DECALAGE, c_desalign))
print("rapport                            : %.2f" % (c_align / c_desalign))

os.makedirs("results", exist_ok=True)
fig, axes = plt.subplots(1, 5, figsize=(16, 4.0))

panneaux = [
    (sub[0], "une image", dict(cmap="gray", vmin=0, vmax=255)),
    (sub[1], "une autre", dict(cmap="gray", vmin=0, vmax=255)),
    (moyenne, "MOYENNE des %d\n(contraste %.1f)" % (n_reel, c_align),
     dict(cmap="gray", vmin=0, vmax=255)),
    (moyenne_floue, "moyenne si décalées\nde +/-%d px (%.1f)" % (DECALAGE, c_desalign),
     dict(cmap="gray", vmin=0, vmax=255)),
    (ecart, "écart-type par pixel", dict(cmap="magma")),
]
for ax, (img, titre, kw) in zip(axes, panneaux):
    ax.imshow(img, **kw)
    ax.set_title(titre, fontsize=10)
    ax.axis("off")

plt.tight_layout(pad=1.4)
plt.savefig("results/alignement.png", dpi=140)
print("figure : results/alignement.png")
