"""
Évaluation du réseau de covariance — phase 6 du projet.

    python eval_cov.py                  # 2000 images de test, les deux modèles
    python eval_cov.py --n 0            # tout le jeu de test (19 962 images)
    python eval_cov.py --verif          # auto-tests numériques, sans données

La difficulté de cette phase est conceptuelle avant d'être technique : sur
données réelles `Sigma_true` N'EXISTE PAS. Ni distance de Frobenius, ni
divergence de KL — les deux métriques du projet ellipses disparaissent avec
elle. Quatre choses les remplacent, et elles suffisent (§7 de tuteur.txt) :

    (1) la NLL sur le jeu de TEST ;
    (2) la RÉFÉRENCE DIAGONALE, seul étalon qui donne un sens à (1) ;
    (3) la CALIBRATION du résidu blanchi w = L^T r, qui doit suivre N(0, I) ;
    (4) les figures : mu + eps diagonal contre mu + eps structuré.

Le point (3) est le plus informatif des quatre. Si la covariance était juste,
`w` serait un bruit blanc de variance 1 : sa variance mesure la sur- ou
sous-confiance du modèle, et son AUTOCORRÉLATION dit si la structure spatiale
du résidu a été capturée. Le modèle diagonal ne peut, par construction, que
recalibrer l'amplitude pixel par pixel : son `w` garde l'autocorrélation du
résidu. C'est la mesure la plus directe de ce que les 24 canaux hors-diagonale
apportent.

AUCUNE MATRICE n x n N'EST FORMÉE. À n = 4096 une seule pèse 67 Mo. La NLL et
la calibration passent par `apply_LT` (coût O(n*m)), l'échantillonnage par une
substitution arrière creuse.

Sorties :
    results/eval_cov.json        tous les chiffres
    results/echantillons.png     les 5 colonnes (Fig. 7 et 19 de l'article)
    results/calibration.png      cartes de variance et autocorrélations
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
# Mêmes raisons que dans train_cov.py : ces fonctions posent ici exactement le
# même problème qu'à l'entraînement (chargement tolérant des checkpoints, bruit
# de validation figé, débruiteur gelé), autant les importer que les recopier.
from train_cov import charger_dncnn
from train_dncnn import charger_checkpoint, construire_validation
from verifier_residu import autocorrelation_2d, profil_radial

DOSSIER_RES = "results"


# --------------------------------------------------------------------------
# Chargement
# --------------------------------------------------------------------------
def charger_cov(chemin, appareil):
    """
    Charge un réseau de covariance et rétablit le MODE avec lequel il a été
    entraîné (structuré ou diagonale seule).

    Le mode est lu dans le checkpoint, jamais deviné : instancier un réseau
    structuré à partir de poids diagonaux ne planterait pas, cela donnerait
    simplement des résultats faux et silencieux.
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
# Produit L v  (le transposé de apply_LT, qui vit dans loss.py)
# --------------------------------------------------------------------------
def apply_L(log_diag, offdiag, vecteur, neighbor_idx, mask):
    """
    Calcule `z = L v` sans matrice dense, en O(n*m).

    `apply_LT` de loss.py disperse (scatter) ; celui-ci rassemble (gather) :

        z_i = l_ii * v_i + sum_k offdiag[i, k] * v[neighbor_idx[i, k]]

    Les deux ensemble donnent le produit par la précision, `Lambda v = L L^T v`,
    dont denoise.py a besoin pour son filtre de Wiener. C'est la seule brique
    qui manquait à loss.py, qui n'avait besoin que de L^T pour la NLL.

    Args :
        log_diag : [B, n]      offdiag : [B, n, m]      vecteur : [B, n]
    Retour :
        z : [B, n]
    """
    diag = torch.exp(clamp_log_diag(log_diag))
    voisins = vecteur[:, neighbor_idx]                       # [B, n, m]
    return diag * vecteur + (offdiag * mask.unsqueeze(0) * voisins).sum(dim=2)


def appliquer_lambda(log_diag, offdiag, vecteur, neighbor_idx, mask):
    """`Lambda v = L (L^T v)`. Symétrique définie positive, jamais formée."""
    return apply_L(log_diag, offdiag,
                   apply_LT(log_diag, offdiag, vecteur, neighbor_idx, mask),
                   neighbor_idx, mask)


# --------------------------------------------------------------------------
# Échantillonnage : résoudre L^T eps = u
# --------------------------------------------------------------------------
def build_anticausal_indices(image_size=IMAGE_SIZE, f=VOISINAGE, device=None):
    """
    Le motif de parcimonie lu dans l'autre sens.

    `build_neighbor_indices` répond à « quels pixels j (avant i) influencent la
    ligne i de L ». Pour la substitution arrière il faut la question inverse :
    « quels pixels j (APRÈS i) ont i pour voisin causal », c'est-à-dire quelles
    valeurs non nulles se trouvent dans la COLONNE i de L.

        neighbor_idx[j, k] = i   <=>   j = i - offset_k

    Retour :
        anti_idx  : [n, m], anti_idx[i, k] = j tel que neighbor_idx[j, k] = i
                    (ou i lui-même si ce j sort de l'image),
        anti_mask : [n, m], 1.0 si ce j existe.

    Le coefficient à utiliser est alors `offdiag[anti_idx[i, k], k]` : le même
    indice k des deux côtés, puisque c'est le même décalage.
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
            rr, cc = r - dr, c - dc          # signe opposé : c'est tout
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
    Tire un résidu `eps ~ N(0, Sigma)` en résolvant `L^T eps = u`, `u ~ N(0, I)`.

    Pourquoi cela marche : si `L^T eps = u` alors
    `cov(eps) = L^-T I L^-1 = (L L^T)^-1 = Lambda^-1 = Sigma`. On n'inverse
    donc jamais rien, on résout un système TRIANGULAIRE — et creux, avec 24
    coefficients par ligne.

    `L^T` est triangulaire SUPÉRIEURE : la substitution part du dernier pixel
    et remonte l'ordre raster.

        eps_i = ( u_i - sum_{j > i} L[j, i] eps_j ) / l_ii

    COÛT : la boucle est intrinsèquement séquentielle, n = 4096 itérations
    Python quel que soit le nombre d'images (le batch est vectorisé). Comptez
    quelques secondes. C'est sans importance : on n'échantillonne que pour les
    figures, jamais dans une boucle d'entraînement.

    `offdiag = None` court-circuite la boucle : pour le modèle diagonal, L est
    diagonale et eps se lit directement.
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
# Parcours du jeu de test
# --------------------------------------------------------------------------
@torch.no_grad()
def calculer_mu_et_residu(dncnn, x, y, batch):
    """mu = DnCNN(y) et r = x - mu, par lots pour ne pas saturer la mémoire."""
    mus = []
    for d in range(0, len(x), batch):
        mus.append(dncnn(y[d:d + batch]))
    mu = torch.cat(mus)
    return mu, (x - mu).flatten(1)


@torch.no_grad()
def nll_et_blanchi(cov_net, mu, r, batch, neighbor_idx, mask):
    """
    NLL par image et résidu blanchi `w = L^T r`, sur tout le jeu.

    Retour :
        nll : [N] (nat par IMAGE, constante n*log(2*pi) comprise)
        w   : [N, n] sur le CPU (33 Mo pour 2000 images)
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
    Ce que doit vérifier `w = L^T r` si la covariance prédite est juste :
    moyenne 0, variance 1, aucune corrélation spatiale.

    La variance par pixel est la CARTE DE CONFIANCE : au-dessus de 1 le modèle
    était trop confiant (il a prédit une covariance trop petite), en dessous il
    ne l'était pas assez.
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
    La figure qui résume le projet (Fig. 7 et 19 de l'article).

        x | y | mu | mu + eps diagonal | mu + eps structuré

    Les deux dernières colonnes portent le même bruit de départ `u` : ce qui
    les sépare est UNIQUEMENT la covariance utilisée pour le transformer. La
    colonne diagonale doit montrer de la neige (chaque pixel tiré
    indépendamment), la colonne structurée du détail cohérent — des cheveux,
    des contours, du grain de peau.
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
    Deux cartes de variance et deux profils d'autocorrélation.

    Le panneau qui compte est celui des profils : le modèle diagonal recalibre
    l'amplitude pixel par pixel, donc son `w` garde la corrélation spatiale du
    résidu (~0,55 à 1 pixel). Le modèle structuré doit l'écraser vers 0. C'est
    la preuve, en une courbe, que les 24 canaux hors-diagonale servent.
    """
    noms = list(stats.keys())
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # Échelle en log2 : la variance de w est un RAPPORT à 1, donc « deux fois
    # trop » et « deux fois trop peu » doivent se lire symétriquement. Les
    # bornes s'adaptent aux données, sinon un modèle mal calibré donne une
    # image uniformément saturée qui n'apprend rien.
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

    # Borne au moins egale a 4 : c'est l'echelle de N(0, 1), celle qu'on veut
    # lire quand le modele est calibre. Elle s'elargit si w deborde, plutot que
    # d'ecraser tout le desaccord sur les deux barres du bord.
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
# Auto-tests numériques (--verif)
# --------------------------------------------------------------------------
def verifications():
    """
    Contrôle des deux briques ajoutées ici, en 16x16 où la matrice dense tient
    dans 262 ko et sert de vérité de référence.

    Rien de tout cela ne dépend de la taille de l'image : le valider en 256
    pixels le valide en 4096, exactement comme pour les tests de loss.py.
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
    # <L a, b> = <a, L^T b> pour tout a, b : identité qui ne peut être vraie
    # par hasard, et qui teste les deux fonctions d'un coup.
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
    # 4 000 tirages pour UNE image : la covariance empirique doit converger
    # vers la Sigma prédite. Tolérance large, c'est du Monte-Carlo.
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

    # ---- modèles ---------------------------------------------------------
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

    # Un réseau de covariance n'est valable que pour LE débruiteur sur lequel
    # il a été entraîné : mu change, donc r change, donc la loi apprise ne
    # correspond plus à rien. L'erreur ne produirait aucun message.
    for nom, (_, e) in modeles.items():
        if e.get("dncnn") and e["dncnn"] != args.dncnn:
            print("/!\\ %s a été entraîné avec %s, on évalue avec %s."
                  % (nom, e["dncnn"], args.dncnn))

    # ---- données de test, bruit figé -------------------------------------
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

    # ---- NLL et calibration ---------------------------------------------
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
        # Sous-échantillon pour l'histogramme : 200 000 valeurs suffisent
        # largement à dessiner une densité, et 8 millions rendraient la figure
        # inutilement lourde.
        plat = w.flatten()
        pas = max(1, plat.numel() // 200000)
        stats[nom]["_echantillon_w"] = plat[::pas].numpy()

    # Le résidu doit être mesuré EXACTEMENT comme w, sinon les lignes du
    # tableau ne sont pas comparables. Piège : le profil radial au rayon 1
    # moyenne les 4 voisins orthogonaux ET les 4 diagonaux (distance 1,41,
    # arrondie à 1), ce qui donne un chiffre nettement plus bas que
    # l'autocorrélation horizontale/verticale de verifier_residu.py — 0,46 au
    # lieu de 0,55 sur le résidu du DnCNN. On garde les deux : la moyenne h/v
    # pour le tableau, le profil radial pour la figure.
    auto_residu = autocorrelation_2d(
        r.reshape(-1, IMAGE_SIZE, IMAGE_SIZE).cpu().numpy())
    centre = auto_residu.shape[0] // 2
    autocorr_residu_hv = 0.5 * float(auto_residu[centre, centre + 1]
                                     + auto_residu[centre + 1, centre])
    profil_residu = profil_radial(auto_residu)

    # ---- figures ---------------------------------------------------------
    # Images réparties sur tout le jeu de test plutôt que les premières :
    # cinq visages consécutifs de CelebA se ressemblent trop pour être
    # convaincants.
    idx_fig = [int(v) for v in np.linspace(0, len(jeu) - 1, args.figures)]
    xf, yf = construire_validation(jeu, idx_fig, jeu.sigma,
                                   args.graine + 7, appareil)
    with torch.no_grad():
        muf = dncnn(yf)

    # Le MÊME u pour les deux modèles : seule la covariance change d'une
    # colonne à l'autre, ce qui rend la comparaison lisible.
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

    # ---- résumé ----------------------------------------------------------
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
    # Dernière ligne : le point de départ. NLL d'une gaussienne isotrope, et
    # l'autocorrélation du résidu BRUT, celle que le blanchi doit effacer.
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
