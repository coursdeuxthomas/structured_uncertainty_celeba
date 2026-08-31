"""
THE CRITICAL CHECK.

    python verifier_residu.py --checkpoint checkpoints/dncnn_best.pt

The whole project rests on one hypothesis: the residual `r = x - DnCNN(y)`,
what the denoiser removed in excess, contains SPATIAL STRUCTURE. If `r` looks
like white noise, a diagonal covariance would be enough to describe it and
there is nothing to learn: the project has no purpose.

This script settles the question, visually and numerically.

What has to be understood before reading the result: `r` is a MIXTURE of two
things.

    r = (residual noise not removed)  +  (image detail destroyed on the way)
         white component                  structured component

Its autocorrelation therefore shows a narrow peak at the centre — the white
part — and, if the hypothesis holds, a BROADENING around that peak. It is
that broadening which justifies the project, not the height of the peak.

Outputs:
    results/residu.png
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import CelebADataset
from dncnn import DnCNN


def autocorrelation_2d(champs):
    """
    Mean spatial autocorrelation of a batch of images [N, H, W].

    Going through the FFT: the autocorrelation is the inverse transform of
    the power spectrum. Each image is centred first, otherwise the mean
    dominates everything. Normalisation by the value at zero shift, so that
    the centre equals 1 and the map reads as correlations.

    Note: the FFT assumes periodic borders. At 64x64 the effect is marginal
    for short shifts, which are the only ones we are interested in.
    """
    c = champs - champs.mean(axis=(-2, -1), keepdims=True)
    spectre = np.abs(np.fft.fft2(c)) ** 2
    a = np.fft.ifft2(spectre).real
    a = np.fft.fftshift(a, axes=(-2, -1))
    centre = a.shape[-1] // 2
    a = a / a[:, centre, centre][:, None, None]
    return a.mean(axis=0)


def profil_radial(carte):
    """Mean of the autocorrelation map over rings of integer radius."""
    h, w = carte.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    d_ent = d.round().astype(int)
    rayons = np.arange(0, 16)
    return rayons, np.array([carte[d_ent == r].mean() for r in rayons])


def main():
    p = argparse.ArgumentParser(description="Vérification critique du résidu.")
    p.add_argument("--checkpoint", default="checkpoints/dncnn_best.pt")
    p.add_argument("--n", type=int, default=512, help="images de test utilisées.")
    p.add_argument("--graine", type=int, default=0)
    args = p.parse_args()

    appareil = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jeu = CelebADataset("test")
    sigma = jeu.sigma

    modele = DnCNN().to(appareil)
    try:
        etat = torch.load(args.checkpoint, map_location=appareil,
                          weights_only=False)
    except TypeError:                                     # very old torch
        etat = torch.load(args.checkpoint, map_location=appareil)
    modele.load_state_dict(etat["modele"])
    modele.eval()
    print("checkpoint : %s  (epoch %d)" % (args.checkpoint, etat["epoch"]))

    # Frozen noise: the diagnostic must be reproducible.
    generateur = torch.Generator().manual_seed(args.graine)
    brut = np.stack([jeu.images[i] for i in range(args.n)])
    x = torch.from_numpy(brut.astype(np.float32) / 127.5 - 1.0).unsqueeze(1)
    y = x + sigma * torch.randn(x.shape, generator=generateur)

    with torch.no_grad():
        mu = modele(y.to(appareil)).cpu()

    r = (x - mu).squeeze(1).numpy()          # [N, 64, 64]
    bruit_ajoute = (y - x).squeeze(1).numpy()

    # Control: white noise with the same standard deviation as r. It is the
    # reference against which "structured" means something.
    temoin = np.random.default_rng(args.graine).normal(0.0, r.std(), r.shape)

    a_r = autocorrelation_2d(r)
    a_t = autocorrelation_2d(temoin)
    rayons, prof_r = profil_radial(a_r)
    _, prof_t = profil_radial(a_t)

    c = a_r.shape[0] // 2
    a1_h, a1_v = a_r[c, c + 1], a_r[c + 1, c]
    sous_seuil = np.where(prof_r < 0.1)[0]
    longueur = rayons[sous_seuil[0]] if len(sous_seuil) else rayons[-1]

    print()
    print("écart-type du bruit ajouté      : %.4f" % bruit_ajoute.std())
    print("écart-type du résidu r          : %.4f" % r.std())
    if r.std() < bruit_ajoute.std():
        print("  r est plus petit que le bruit ajouté : le débruiteur")
        print("  fait son travail.")
    else:
        print("  /!\\ r est PLUS GRAND que le bruit ajouté : le débruiteur")
        print("  DÉGRADE l'image au lieu de l'améliorer. Modèle sous-entraîné")
        print("  ou bug. Le diagnostic ci-dessous n'a aucun sens dans cet état.")
    print()
    print("autocorrélation à 1 pixel  horizontal %.3f   vertical %.3f"
          % (a1_h, a1_v))
    print("  témoin bruit blanc                  %.3f" % a_t[c, c + 1])
    print("portée : rayon où l'autocorrélation passe sous 0,1 : %d pixels"
          % longueur)
    print()
    if max(a1_h, a1_v) > 0.15 and longueur >= 2:
        print("VERDICT : le résidu est STRUCTURÉ. Le projet a un objet.")
    else:
        print("VERDICT : le résidu est quasi blanc. ALERTE — relire le")
        print("          protocole avant d'aller plus loin.")

    # ---- figure ----------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    gris = dict(cmap="gray", vmin=-1, vmax=1)
    k = 0

    axes[0, 0].imshow(x[k, 0], **gris)
    axes[0, 0].set_title("x  (propre)")
    axes[0, 1].imshow(y[k, 0], **gris)
    axes[0, 1].set_title("y  (bruitée)")
    axes[0, 2].imshow(mu[k, 0], **gris)
    axes[0, 2].set_title("mu = DnCNN(y)")

    ampl = 3.0 * r.std()
    axes[1, 0].imshow(r[k], cmap="RdBu_r", vmin=-ampl, vmax=ampl)
    axes[1, 0].set_title("r = x - mu  (contraste amplifié)")

    demi = 7
    vue = a_r[c - demi:c + demi + 1, c - demi:c + demi + 1]
    im = axes[1, 1].imshow(vue, cmap="magma", vmin=0, vmax=1,
                           extent=[-demi, demi, demi, -demi])
    axes[1, 1].set_title("autocorrélation de r  (15 x 15)")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046)

    axes[1, 2].plot(rayons, prof_r, "o-", label="résidu r")
    axes[1, 2].plot(rayons, prof_t, "s--", label="témoin bruit blanc")
    axes[1, 2].axhline(0.1, color="grey", lw=0.8, ls=":")
    axes[1, 2].set_xlabel("décalage (pixels)")
    axes[1, 2].set_ylabel("autocorrélation")
    axes[1, 2].set_title("profil radial")
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)

    for ax in [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("results/residu.png", dpi=140)
    print("figure : results/residu.png")


if __name__ == "__main__":
    main()
