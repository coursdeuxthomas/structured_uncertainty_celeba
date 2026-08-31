"""
Empirical proof that the CelebA faces are aligned.

    export CELEBA_CACHE=$HOME/data/celeba/celeba_64_gray.npy
    python verifier_alignement.py

Principle: if the faces are aligned, the MEAN image of the dataset is a sharp
face — eyes, nose and mouth stay visible because they fall at the same place
in every image. If they were not, the mean would be a grey mush.

The script displays side by side the true mean and the mean obtained after
randomly shifting the same images: this is what the dataset would look like
without the alignment done upstream by the authors of CelebA.

Writes results/alignement.png
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE = os.environ.get("CELEBA_CACHE", "celeba_64_gray.npy")
N = 20000          # sample: the mean is already stable well before that
DECALAGE = 6       # amplitude of the simulated misalignment, in pixels


def moyenne_desalignee(images, amplitude, graine=0):
    """Shift each image by a random vector, then average."""
    rng = np.random.default_rng(graine)
    dy = rng.integers(-amplitude, amplitude + 1, size=len(images))
    dx = rng.integers(-amplitude, amplitude + 1, size=len(images))
    acc = np.zeros(images.shape[1:], dtype=np.float64)
    for k in range(len(images)):
        acc += np.roll(images[k], (dy[k], dx[k]), axis=(0, 1))
    return acc / len(images)


# mmap_mode: we do not load the 830 MB just to read the first 20 000.
images = np.load(CACHE, mmap_mode="r")
print("cache  : %s" % CACHE)
print("forme  : %s  dtype %s" % (images.shape, images.dtype))

sub = np.asarray(images[:N], dtype=np.float32)
n_reel = len(sub)
print("échantillon : %d images" % n_reel)

moyenne = sub.mean(axis=0)
ecart = sub.std(axis=0)
moyenne_floue = moyenne_desalignee(sub, DECALAGE)

# The contrast of the mean image quantifies the alignment: the higher it is,
# the better the faces overlap.
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
