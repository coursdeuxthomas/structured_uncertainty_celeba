"""
Evaluation of the covariance network — phase 6 of the project.

    python eval_cov.py                  # 2000 test images, both models
    python eval_cov.py --n 0            # the whole test set (19,962 images)
    python eval_cov.py --verif          # numerical self-tests, no data

The difficulty of this phase is conceptual before it is technical: on real
data `Sigma_true` DOES NOT EXIST. Neither the Frobenius distance nor the KL
divergence — the two metrics of the ellipses project — survives without it.
Four things replace them, and they are enough (§7 of docs/tuteur.txt):

    (1) the NLL on the TEST set;
    (2) the DIAGONAL BASELINE, the only yardstick that gives (1) a meaning;
    (3) the CALIBRATION of the whitened residual w = L^T r, which must
        follow N(0, I);
    (4) the figures: mu + eps diagonal against mu + eps structured.

Point (3) is the most informative of the four. If the covariance were right,
`w` would be white noise of variance 1: its variance measures the model's
over- or under-confidence, and its AUTOCORRELATION tells whether the spatial
structure of the residual has been captured. The diagonal model can, by
construction, only recalibrate the amplitude pixel by pixel: its `w` keeps the
autocorrelation of the residual. This is the most direct measure of what the
24 off-diagonal channels bring.

NO n x n MATRIX IS EVER FORMED. At n = 4096 a single one weighs 67 MB. The NLL
and the calibration go through `apply_LT` (cost O(n*m)), the sampling through
a sparse back substitution.

Outputs:
    results/eval_cov.json        every number
    results/echantillons.png     the 5 columns (Fig. 7 and 19 of the article)
    results/calibration.png      variance maps and autocorrelations
"""

import argparse
import json
import math
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cov_model import SparseCholeskyNet
from data import CelebADataset
from loss import (IMAGE_SIZE, VOISINAGE, apply_LT, build_neighbor_indices,
                  causal_offsets, clamp_log_diag, structured_gaussian_nll)
# Same reasons as in train_cov.py: here these functions face exactly the same
# problem as at training time (tolerant checkpoint loading, frozen validation
# noise, frozen denoiser), so importing them beats copying them out.
from train_cov import charger_dncnn
from train_dncnn import charger_checkpoint, construire_validation
from verifier_residu import autocorrelation_2d, profil_radial

DOSSIER_RES = "results"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def charger_cov(chemin, appareil):
    """
    Loads a covariance network and restores the MODE it was trained with
    (structured, or diagonal only).

    The mode is read from the checkpoint, never guessed: instantiating a
    structured network from diagonal weights would not crash, it would simply
    produce wrong results, silently.
    """
    etat = charger_checkpoint(chemin, appareil)
    if "diagonale" not in etat:
        raise SystemExit(
            "ERREUR : %s ne contient pas la clé 'diagonale'. Ce checkpoint "
            "n'a pas été écrit par train_cov.py." % chemin)
    reseau = SparseCholeskyNet(diagonale_seule=etat["diagonale"]).to(appareil)
    reseau.load_state_dict(etat["modele"])
    reseau.eval()
    for p in reseau.parameters():
        p.requires_grad_(False)
    return reseau, etat


# --------------------------------------------------------------------------
# Product L v  (the transpose of apply_LT, which lives in loss.py)
# --------------------------------------------------------------------------
def apply_L(log_diag, offdiag, vecteur, neighbor_idx, mask):
    """
    Computes `z = L v` with no dense matrix, in O(n*m).

    `apply_LT` from loss.py scatters; this one gathers:

        z_i = l_ii * v_i + sum_k offdiag[i, k] * v[neighbor_idx[i, k]]

    The two together give the product by the precision, `Lambda v = L L^T v`,
    which denoise.py needs for its Wiener filter. This is the one building
    block loss.py was missing, since it only ever needed L^T for the NLL.

    Args:
        log_diag : [B, n]      offdiag : [B, n, m]      vecteur : [B, n]
    Returns:
        z : [B, n]
    """
    diag = torch.exp(clamp_log_diag(log_diag))
    voisins = vecteur[:, neighbor_idx]                       # [B, n, m]
    return diag * vecteur + (offdiag * mask.unsqueeze(0) * voisins).sum(dim=2)


def appliquer_lambda(log_diag, offdiag, vecteur, neighbor_idx, mask):
    """`Lambda v = L (L^T v)`. Symmetric positive definite, never formed."""
    return apply_L(log_diag, offdiag,
                   apply_LT(log_diag, offdiag, vecteur, neighbor_idx, mask),
                   neighbor_idx, mask)


# --------------------------------------------------------------------------
# Sampling: solve L^T eps = u
# --------------------------------------------------------------------------
def build_anticausal_indices(image_size=IMAGE_SIZE, f=VOISINAGE, device=None):
    """
    The sparsity pattern read the other way round.

    `build_neighbor_indices` answers "which pixels j (before i) influence row i
    of L". For the back substitution the reverse question is needed: "which
    pixels j (AFTER i) have i as a causal neighbour", that is, which non-zero
    values sit in COLUMN i of L.

        neighbor_idx[j, k] = i   <=>   j = i - offset_k

    Returns:
        anti_idx  : [n, m], anti_idx[i, k] = j such that
                    neighbor_idx[j, k] = i (or i itself if that j falls
                    outside the image),
        anti_mask : [n, m], 1.0 if that j exists.

    The coefficient to use is then `offdiag[anti_idx[i, k], k]`: the same index
    k on both sides, since it is the same offset.
    """
    S = image_size
    n = S * S
    offsets = causal_offsets(f)
    m = len(offsets)

    anti_idx = np.zeros((n, m), dtype=np.int64)
    anti_mask = np.zeros((n, m), dtype=np.float32)

    for i in range(n):
        r, c = divmod(i, S)
        for k, (dr, dc) in enumerate(offsets):
            rr, cc = r - dr, c - dc          # opposite sign: that is all
            if 0 <= rr < S and 0 <= cc < S:
                anti_idx[i, k] = rr * S + cc
                anti_mask[i, k] = 1.0
            else:
                anti_idx[i, k] = i
                anti_mask[i, k] = 0.0

    anti_idx = torch.from_numpy(anti_idx)
    anti_mask = torch.from_numpy(anti_mask)
    if device is not None:
        anti_idx = anti_idx.to(device)
        anti_mask = anti_mask.to(device)
    return anti_idx, anti_mask


@torch.no_grad()
def echantillonner_residu(log_diag, offdiag, anti_idx, anti_mask, u):
    """
    Draws a residual `eps ~ N(0, Sigma)` by solving `L^T eps = u`,
    `u ~ N(0, I)`.

    Why this works: if `L^T eps = u` then
    `cov(eps) = L^-T I L^-1 = (L L^T)^-1 = Lambda^-1 = Sigma`. So nothing is
    ever inverted, a TRIANGULAR system is solved instead — and a sparse one,
    with 24 coefficients per row.

    `L^T` is UPPER triangular: the substitution starts at the last pixel and
    walks back up the raster order.

        eps_i = ( u_i - sum_{j > i} L[j, i] eps_j ) / l_ii

    COST: the loop is inherently sequential, n = 4096 Python iterations no
    matter how many images there are (the batch is vectorised). Expect a few
    seconds. It does not matter: sampling only ever happens for the figures,
    never inside a training loop.

    `offdiag = None` short-circuits the loop: for the diagonal model L is
    diagonal and eps can be read off directly.
    """
    inv_diag = torch.exp(-clamp_log_diag(log_diag))          # 1 / l_ii
    if offdiag is None:
        return u * inv_diag

    B, n = u.shape
    m = offdiag.shape[2]
    ks = torch.arange(m, device=u.device)
    eps = torch.zeros_like(u)

    for i in range(n - 1, -1, -1):
        j = anti_idx[i]                                      # [m]
        coef = offdiag[:, j, ks] * anti_mask[i]              # [B, m]
        somme = (coef * eps[:, j]).sum(dim=1)                # [B]
        eps[:, i] = (u[:, i] - somme) * inv_diag[:, i]
    return eps


# --------------------------------------------------------------------------
# Walking through the test set
# --------------------------------------------------------------------------
@torch.no_grad()
def calculer_mu_et_residu(dncnn, x, y, batch):
    """mu = DnCNN(y) and r = x - mu, in batches so as not to fill memory."""
    mus = []
    for d in range(0, len(x), batch):
        mus.append(dncnn(y[d:d + batch]))
    mu = torch.cat(mus)
    return mu, (x - mu).flatten(1)


@torch.no_grad()
def nll_et_blanchi(cov_net, mu, r, batch, neighbor_idx, mask):
    """
    Per-image NLL and whitened residual `w = L^T r`, over the whole set.

    Returns:
        nll : [N] (nats per IMAGE, the n*log(2*pi) constant included)
        w   : [N, n] on the CPU (33 MB for 2000 images)
    """
    nlls, ws = [], []
    for d in range(0, len(mu), batch):
        log_diag, offdiag = cov_net(mu[d:d + batch])
        rr = r[d:d + batch]
        nlls.append(structured_gaussian_nll(log_diag, offdiag, rr,
                                            neighbor_idx, mask,
                                            mean_batch=False).cpu())
        ws.append(apply_LT(log_diag, offdiag, rr, neighbor_idx, mask).cpu())
    return torch.cat(nlls), torch.cat(ws)


def statistiques_calibration(w, n_cote=IMAGE_SIZE):
    """
    What `w = L^T r` must satisfy if the predicted covariance is right:
    mean 0, variance 1, no spatial correlation whatsoever.

    The per-pixel variance is the CONFIDENCE MAP: above 1 the model was
    over-confident (it predicted too small a covariance), below 1 it was not
    confident enough.
    """
    champs = w.reshape(-1, n_cote, n_cote).numpy()
    carte_var = champs.var(axis=0)
    carte_moy = champs.mean(axis=0)
    auto = autocorrelation_2d(champs)
    c = auto.shape[0] // 2
    rayons, profil = profil_radial(auto)
    return {
        "moyenne": float(champs.mean()),
        "variance": float(champs.var()),
        "carte_var_min": float(carte_var.min()),
        "carte_var_max": float(carte_var.max()),
        "carte_var_mediane": float(np.median(carte_var)),
        "carte_moy_absmax": float(np.abs(carte_moy).max()),
        "autocorr_1px_h": float(auto[c, c + 1]),
        "autocorr_1px_v": float(auto[c + 1, c]),
        "_carte_var": carte_var,
        "_profil": (rayons, profil),
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def figure_echantillons(x, y, mu, echantillons, chemin):
    """
    The figure that sums up the project (Fig. 7 and 19 of the article).

        x | y | mu | mu + eps diagonal | mu + eps structured

    The last two columns carry the same starting noise `u`: what separates
    them is ONLY the covariance used to transform it. The diagonal column
    should show snow (each pixel drawn independently), the structured column
    coherent detail — hair, edges, skin grain.
    """
    colonnes = [("x  (propre)", x), ("y  (bruitée)", y), ("mu = DnCNN(y)", mu)]
    colonnes += echantillons
    n_lignes = x.shape[0]
    n_col = len(colonnes)

    fig, axes = plt.subplots(n_lignes, n_col,
                             figsize=(2.3 * n_col, 2.4 * n_lignes))
    axes = np.atleast_2d(axes)
    for j, (titre, images) in enumerate(colonnes):
        for i in range(n_lignes):
            axes[i, j].imshow(images[i], cmap="gray", vmin=-1, vmax=1)
            axes[i, j].axis("off")
        axes[0, j].set_title(titre, fontsize=10)
    plt.tight_layout()
    plt.savefig(chemin, dpi=140)
    plt.close()


def figure_calibration(stats, profil_residu, chemin):
    """
    Two variance maps and two autocorrelation profiles.

    The panel that matters is the one with the profiles: the diagonal model
    recalibrates the amplitude pixel by pixel, so its `w` keeps the spatial
    correlation of the residual (~0.55 at 1 pixel). The structured model has
    to crush it down to 0. That is the proof, in a single curve, that the 24
    off-diagonal channels earn their keep.
    """
    noms = list(stats.keys())
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # log2 scale: the variance of w is a RATIO to 1, so "twice too much" and
    # "twice too little" must read symmetrically. The bounds adapt to the
    # data, otherwise a badly calibrated model gives a uniformly saturated
    # image that teaches nothing.
    cartes = [np.log2(np.maximum(stats[nom]["_carte_var"], 1e-6))
              for nom in noms[:2]]
    limite = max(1.0, max(float(np.percentile(np.abs(c), 99)) for c in cartes))
    for k, nom in enumerate(noms[:2]):
        im = axes[0, k].imshow(cartes[k], cmap="RdBu_r",
                               vmin=-limite, vmax=limite)
        axes[0, k].set_title("log2 var(w) par pixel — %s\n"
                             "(0 = calibré ; médiane de var(w) = %.2f)"
                             % (nom, np.median(stats[nom]["_carte_var"])),
                             fontsize=10)
        axes[0, k].axis("off")
        fig.colorbar(im, ax=axes[0, k], fraction=0.046)
    if len(noms) < 2:
        axes[0, 1].axis("off")

    # Bound at least equal to 4: that is the scale of N(0, 1), the one we want
    # to read when the model is calibrated. It widens if w overflows, rather
    # than squashing all the disagreement onto the two edge bars.
    borne = max(4.0, max(float(np.percentile(np.abs(stats[nom]["_echantillon_w"]),
                                             99.5)) for nom in noms))
    for nom in noms:
        axes[1, 0].hist(stats[nom]["_echantillon_w"], bins=120,
                        range=(-borne, borne), density=True,
                        histtype="step", label="w  %s" % nom)
    grille = np.linspace(-borne, borne, 400)
    axes[1, 0].plot(grille, np.exp(-grille ** 2 / 2) / math.sqrt(2 * math.pi),
                    "k--", lw=1, label="N(0, 1) visé")
    axes[1, 0].set_title("distribution de w = L^T r", fontsize=10)
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    rayons, profil = profil_residu
    axes[1, 1].plot(rayons, profil, "o-", color="0.4", label="résidu r")
    for nom in noms:
        r_w, p_w = stats[nom]["_profil"]
        axes[1, 1].plot(r_w, p_w, "s-", label="w  %s" % nom)
    axes[1, 1].axhline(0.0, color="grey", lw=0.8, ls=":")
    axes[1, 1].set_xlabel("décalage (pixels)")
    axes[1, 1].set_ylabel("autocorrélation")
    axes[1, 1].set_title("autocorrélation spatiale : le résidu contre son\n"
                         "blanchi (0 partout = structure capturée)", fontsize=10)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(chemin, dpi=140)
    plt.close()


# --------------------------------------------------------------------------
# Numerical self-tests (--verif)
# --------------------------------------------------------------------------
def verifications():
    """
    Checks the two building blocks added here, in 16x16 where the dense matrix
    fits in 262 kB and serves as the reference truth.

    None of this depends on the size of the image: validating it at 256 pixels
    validates it at 4096, exactly as for the tests in loss.py.
    """
    from loss import build_L_dense, predicted_precision_and_covariance

    torch.manual_seed(0)
    S, B = 16, 3
    n = S * S
    neighbor_idx, mask = build_neighbor_indices(S, VOISINAGE)
    anti_idx, anti_mask = build_anticausal_indices(S, VOISINAGE)
    m = neighbor_idx.shape[1]

    log_diag = 0.3 * torch.randn(B, n)
    offdiag = 0.1 * torch.randn(B, n, m)
    v = torch.randn(B, n)

    print("=== A) apply_L contre la matrice dense ===")
    L = build_L_dense(log_diag, offdiag, neighbor_idx, mask)
    z_creux = apply_L(log_diag, offdiag, v, neighbor_idx, mask)
    z_dense = torch.bmm(L, v.unsqueeze(2)).squeeze(2)
    err = (z_creux - z_dense).abs().max().item()
    print("erreur max : %.2e" % err)
    assert err < 1e-4

    print()
    print("=== B) apply_L et apply_LT sont bien transposés l'un de l'autre ===")
    # <L a, b> = <a, L^T b> for all a, b: an identity that cannot hold by
    # accident, and that tests both functions in one go.
    a, b = torch.randn(B, n), torch.randn(B, n)
    g = (apply_L(log_diag, offdiag, a, neighbor_idx, mask) * b).sum(1)
    d = (a * apply_LT(log_diag, offdiag, b, neighbor_idx, mask)).sum(1)
    err = (g - d).abs().max().item()
    print("écart max : %.2e" % err)
    assert err < 1e-3

    print()
    print("=== C) l'échantillonnage résout bien L^T eps = u ===")
    u = torch.randn(B, n)
    eps = echantillonner_residu(log_diag, offdiag, anti_idx, anti_mask, u)
    verif = apply_LT(log_diag, offdiag, eps, neighbor_idx, mask)
    err = (verif - u).abs().max().item()
    print("erreur max sur L^T eps - u : %.2e" % err)
    assert err < 1e-3

    print()
    print("=== D) la covariance empirique des tirages est bien Sigma ===")
    # 4,000 draws for ONE image: the empirical covariance must converge to the
    # predicted Sigma. Wide tolerance, this is Monte-Carlo.
    K = 4000
    ld = log_diag[:1].expand(K, n).contiguous()
    od = offdiag[:1].expand(K, n, m).contiguous()
    u = torch.randn(K, n)
    tirages = echantillonner_residu(ld, od, anti_idx, anti_mask, u)
    _, Sigma = predicted_precision_and_covariance(log_diag[:1], offdiag[:1],
                                                  neighbor_idx, mask)
    Sigma = Sigma[0]
    emp = torch.from_numpy(np.cov(tirages.numpy().T)).float()
    err_diag = ((emp.diagonal() - Sigma.diagonal()).abs()
                / Sigma.diagonal()).mean().item()
    err_rel = (emp - Sigma).norm() / Sigma.norm()
    print("erreur relative sur la diagonale : %.3f  (attendu ~%.3f à K=%d)"
          % (err_diag, math.sqrt(2.0 / K), K))
    print("erreur relative de Frobenius     : %.3f" % err_rel.item())
    assert err_diag < 0.05

    print()
    print("OK — apply_L, la transposition et l'échantillonnage sont justes.")


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Évaluation du réseau de covariance.")
    p.add_argument("--cov", default="checkpoints/cov_best.pt",
                   help="modèle structuré.")
    p.add_argument("--covdiag", default="checkpoints/covdiag_best.pt",
                   help="référence diagonale (second entraînement complet).")
    p.add_argument("--dncnn", default="checkpoints/dncnn_best.pt")
    p.add_argument("--n", type=int, default=2000,
                   help="images de test évaluées ; 0 = toutes (19 962).")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--figures", type=int, default=5,
                   help="lignes de la figure d'échantillons.")
    p.add_argument("--graine", type=int, default=0)
    p.add_argument("--verif", action="store_true",
                   help="auto-tests numériques en 16x16, puis sortie.")
    args = p.parse_args()

    if args.verif:
        verifications()
        return

    appareil = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.graine)
    os.makedirs(DOSSIER_RES, exist_ok=True)
    if appareil.type == "cpu":
        print("/!\\ aucun GPU visible : l'évaluation tournera sur CPU et sera")
        print("    lente. Sur le cluster, passez par srun ou sbatch.")
    else:
        print("GPU : %s" % torch.cuda.get_device_name(0))

    # ---- models ----------------------------------------------------------
    dncnn, epoch_dncnn = charger_dncnn(args.dncnn, appareil)
    print("DnCNN : %s (epoch %d), gelé" % (args.dncnn, epoch_dncnn))

    modeles = {}
    cov, etat = charger_cov(args.cov, appareil)
    if etat["diagonale"]:
        raise SystemExit("ERREUR : %s est un modèle DIAGONAL, pas structuré."
                         % args.cov)
    modeles["structuré"] = (cov, etat)
    print("covariance structurée : %s (epoch %d, meilleure NLL val %+.4f "
          "nat/pixel)" % (args.cov, etat["epoch"],
                          etat["meilleur"] / (IMAGE_SIZE ** 2)))

    if os.path.exists(args.covdiag):
        covd, etatd = charger_cov(args.covdiag, appareil)
        if not etatd["diagonale"]:
            raise SystemExit("ERREUR : %s n'est PAS un modèle diagonal."
                             % args.covdiag)
        modeles["diagonal"] = (covd, etatd)
        print("référence diagonale   : %s (epoch %d, meilleure NLL val %+.4f "
              "nat/pixel)" % (args.covdiag, etatd["epoch"],
                              etatd["meilleur"] / (IMAGE_SIZE ** 2)))
    else:
        print()
        print("/!\\ RÉFÉRENCE DIAGONALE ABSENTE (%s introuvable)." % args.covdiag)
        print("    La NLL du modèle structuré seule NE DÉMONTRE RIEN : on ne")
        print("    peut pas savoir si elle est basse grâce à la structure ou")
        print("    simplement grâce à la capacité du réseau. Lancez le second")
        print("    entraînement :  sbatch train_cov.bash --diagonale")

    # A covariance network is only valid for THE denoiser it was trained on:
    # mu changes, so r changes, so the learned law no longer matches anything.
    # The mistake would produce no message at all.
    for nom, (_, e) in modeles.items():
        if e.get("dncnn") and e["dncnn"] != args.dncnn:
            print("/!\\ %s a été entraîné avec %s, on évalue avec %s."
                  % (nom, e["dncnn"], args.dncnn))

    # ---- test data, frozen noise -----------------------------------------
    jeu = CelebADataset("test")
    n_test = len(jeu) if args.n == 0 else min(args.n, len(jeu))
    indices = list(range(n_test))
    x, y = construire_validation(jeu, indices, jeu.sigma, args.graine, appareil)
    mu, r = calculer_mu_et_residu(dncnn, x, y, args.batch)
    n = r.shape[1]

    std_r = r.std().item()
    nll_isotrope = 0.5 * (1.0 + math.log(2.0 * math.pi * std_r ** 2))
    print()
    print("jeu de test : %d images (sur %d)" % (n_test, len(jeu)))
    print("résidu : écart-type %.4f" % std_r)
    print("plancher d'une gaussienne isotrope ajustée : %+.4f nat/pixel"
          % nll_isotrope)

    neighbor_idx, mask = build_neighbor_indices(device=appareil)

    # ---- NLL and calibration ---------------------------------------------
    resultats, stats = {}, {}
    for nom, (reseau, _) in modeles.items():
        nll, w = nll_et_blanchi(reseau, mu, r, args.batch, neighbor_idx, mask)
        nll_pixel = nll / n
        resultats[nom] = {
            "nll_image_moyenne": float(nll.mean()),
            "nll_pixel_moyenne": float(nll_pixel.mean()),
            "nll_pixel_ecart_type": float(nll_pixel.std()),
            "sigma_equivalent": float(math.exp(nll_pixel.mean() - 1.4189385)),
        }
        stats[nom] = statistiques_calibration(w)
        # Sub-sample for the histogram: 200,000 values are plenty to draw a
        # density, and 8 million would make the figure needlessly heavy.
        plat = w.flatten()
        pas = max(1, plat.numel() // 200000)
        stats[nom]["_echantillon_w"] = plat[::pas].numpy()

    # The residual must be measured EXACTLY like w, otherwise the rows of the
    # table are not comparable. Pitfall: the radial profile at radius 1
    # averages the 4 orthogonal neighbours AND the 4 diagonal ones (distance
    # 1.41, rounded to 1), which gives a markedly lower figure than the
    # horizontal/vertical autocorrelation of verifier_residu.py — 0.46 instead
    # of 0.55 on the DnCNN residual. We keep both: the h/v average for the
    # table, the radial profile for the figure.
    auto_residu = autocorrelation_2d(
        r.reshape(-1, IMAGE_SIZE, IMAGE_SIZE).cpu().numpy())
    centre = auto_residu.shape[0] // 2
    autocorr_residu_hv = 0.5 * float(auto_residu[centre, centre + 1]
                                     + auto_residu[centre + 1, centre])
    profil_residu = profil_radial(auto_residu)

    # ---- figures ---------------------------------------------------------
    # Images spread over the whole test set rather than the first ones: five
    # consecutive CelebA faces look far too much alike to be convincing.
    idx_fig = [int(v) for v in np.linspace(0, len(jeu) - 1, args.figures)]
    xf, yf = construire_validation(jeu, idx_fig, jeu.sigma,
                                   args.graine + 7, appareil)
    with torch.no_grad():
        muf = dncnn(yf)

    # The SAME u for both models: only the covariance changes from one column
    # to the other, which makes the comparison readable.
    u = torch.randn(len(idx_fig), n,
                    generator=torch.Generator().manual_seed(args.graine + 11))
    u = u.to(appareil)
    anti_idx, anti_mask = build_anticausal_indices(device=appareil)

    colonnes_ech = []
    for nom in ("diagonal", "structuré"):
        if nom not in modeles:
            continue
        with torch.no_grad():
            log_diag, offdiag = modeles[nom][0](muf)
        eps = echantillonner_residu(
            log_diag, None if nom == "diagonal" else offdiag,
            anti_idx, anti_mask, u)
        image = (muf.flatten(1) + eps).reshape(-1, IMAGE_SIZE, IMAGE_SIZE)
        colonnes_ech.append(("mu + eps  %s" % nom, image.cpu().numpy()))

    figure_echantillons(xf[:, 0].cpu().numpy(), yf[:, 0].cpu().numpy(),
                        muf[:, 0].cpu().numpy(), colonnes_ech,
                        os.path.join(DOSSIER_RES, "echantillons.png"))
    figure_calibration(stats, profil_residu,
                       os.path.join(DOSSIER_RES, "calibration.png"))

    # ---- summary ---------------------------------------------------------
    print()
    print("=" * 78)
    print("%-12s %14s %10s %10s %12s" % ("modèle", "NLL nat/pixel", "sigma eq.",
                                         "var(w)", "autocorr 1px"))
    print("-" * 78)
    for nom in ("structuré", "diagonal"):
        if nom not in resultats:
            continue
        R, S = resultats[nom], stats[nom]
        print("%-12s %+9.4f ±%.3f %10.4f %10.3f %12.3f"
              % (nom, R["nll_pixel_moyenne"], R["nll_pixel_ecart_type"],
                 R["sigma_equivalent"], S["variance"],
                 0.5 * (S["autocorr_1px_h"] + S["autocorr_1px_v"])))
    # Last row: the starting point. NLL of an isotropic Gaussian, and the
    # autocorrelation of the RAW residual, the one whitening must wipe out.
    print("%-12s %+9.4f %6s %10.4f %10s %12.3f"
          % ("résidu brut", nll_isotrope, "", std_r, "-", autocorr_residu_hv))
    print("=" * 78)

    sortie = {
        "n_images": n_test,
        "sigma": jeu.sigma,
        "std_residu": std_r,
        "nll_isotrope_pixel": nll_isotrope,
        "dncnn": {"chemin": args.dncnn, "epoch": epoch_dncnn},
        "modeles": resultats,
        "calibration": {nom: {k: v for k, v in S.items()
                              if not k.startswith("_")}
                        for nom, S in stats.items()},
        "autocorr_residu_1px": autocorr_residu_hv,
        "autocorr_residu_1px_radial": float(profil_residu[1][1]),
    }

    if "diagonal" in resultats:
        gain = (resultats["diagonal"]["nll_pixel_moyenne"]
                - resultats["structuré"]["nll_pixel_moyenne"])
        sortie["gain_structure_nat_par_pixel"] = gain
        print()
        print("APPORT DE LA STRUCTURE : %+.4f nat/pixel" % gain)
        print("  soit un facteur %.2f sur l'écart-type équivalent, et %.0f nat"
              " par image." % (math.exp(gain), gain * n))
        if gain <= 0:
            print("  /!\\ NÉGATIF : le modèle structuré ne bat pas le modèle")
            print("      diagonal. Vérifier l'ordre des 24 canaux, le masque,")
            print("      et que les deux runs ont bien le même nombre d'epochs.")
        ah = stats["structuré"]["autocorr_1px_h"]
        if abs(ah) > 0.15:
            print("  /!\\ le blanchi garde %.2f d'autocorrélation à 1 pixel :"
                  % ah)
            print("      la structure spatiale n'est pas entièrement capturée.")

    with open(os.path.join(DOSSIER_RES, "eval_cov.json"), "w") as f:
        json.dump(sortie, f, indent=2)
    print()
    print("écrit : results/eval_cov.json, results/echantillons.png,"
          " results/calibration.png")


if __name__ == "__main__":
    main()
