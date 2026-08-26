"""
Test de surapprentissage volontaire — critère d'arrêt de la phase 4.

On prend 8 images, on fige le bruit, on calcule le résidu `r = x - DnCNN(y)`
UNE FOIS, et on entraîne le réseau de covariance sur ces 8 résidus et rien
d'autre. Avec 518 681 paramètres pour 8 exemples, la NLL doit s'effondrer.

    Si elle ne descend pas sur 8 images, le problème est dans le code, pas dans
    les données.

C'est le seul test qui vérifie loss.py et cov_model.py ENSEMBLE, et il coûte
deux minutes contre cinq heures pour un vrai entraînement.

Le même test tourne pour le modèle structuré et pour la référence diagonale.
L'écart entre les deux est une première mesure, sur 8 images, de ce que la
structure apporte — la vraie mesure viendra de eval_cov.py sur le jeu de test.

Usage :
    python surapprentissage.py --checkpoint checkpoints/dncnn_best.pt
"""

import argparse
import math

import torch

from cov_model import SparseCholeskyNet
from data import CelebADataset
from dncnn import DnCNN
from loss import build_neighbor_indices, apply_LT, structured_gaussian_nll


def charger_dncnn(chemin, appareil):
    """Charge le débruiteur entraîné, le passe en eval et GÈLE ses poids."""
    etat = torch.load(chemin, map_location=appareil, weights_only=False)
    modele = DnCNN().to(appareil)
    modele.load_state_dict(etat["modele"])
    modele.eval()
    for p in modele.parameters():
        p.requires_grad_(False)
    return modele, etat.get("epoch", -1)


def surapprendre(residu, mu, neighbor_idx, mask, diagonale_seule,
                 iterations, lr, appareil, etiquette):
    """
    Entraîne un réseau de covariance neuf sur un lot FIXE de résidus.

    Renvoie (nll_initiale, nll_finale, variance de w = L^T r).
    """
    reseau = SparseCholeskyNet(diagonale_seule=diagonale_seule).to(appareil)
    opt = torch.optim.Adam(reseau.parameters(), lr=lr)
    n = residu.shape[1]

    nll_init = None
    for it in range(1, iterations + 1):
        opt.zero_grad()
        log_diag, offdiag = reseau(mu)
        nll = structured_gaussian_nll(log_diag, offdiag, residu,
                                      neighbor_idx, mask)
        nll.backward()
        opt.step()

        if nll_init is None:
            nll_init = nll.item()
        if it % max(1, iterations // 10) == 0 or it == 1:
            print("    %-12s iter %5d   NLL/pixel %+8.4f"
                  % (etiquette, it, nll.item() / n))

    # Variance de w = L^T r, à titre indicatif SEULEMENT.
    #
    # Sur données tenues à l'écart, w doit suivre N(0, I) et sa variance vaut
    # 1 : c'est le test de calibration de eval_cov.py. ICI, PAS DU TOUT. Sur
    # 8 résidus figés, 518 681 paramètres peuvent faire tendre la précision
    # vers l'infini et la NLL vers moins l'infini : la variance de w s'effondre
    # bien en dessous de 1, et c'est la PREUVE que le surapprentissage marche.
    # Ne pas lire ce chiffre comme un défaut de calibration.
    with torch.no_grad():
        log_diag, offdiag = reseau(mu)
        w = apply_LT(log_diag, offdiag, residu, neighbor_idx, mask)
        nll_fin = structured_gaussian_nll(log_diag, offdiag, residu,
                                          neighbor_idx, mask).item()
    return nll_init / n, nll_fin / n, w.var().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/dncnn_best.pt")
    p.add_argument("--n", type=int, default=8, help="images surapprises.")
    p.add_argument("--iterations", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--graine", type=int, default=0)
    p.add_argument("--split", default="test", choices=["train", "test"])
    args = p.parse_args()

    appareil = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.graine)

    # Sur le nœud de connexion du cluster il n'y a pas de GPU : torch bascule
    # silencieusement sur le CPU et le test met un quart d'heure au lieu de
    # deux minutes. Autant le dire tout de suite plutôt que de laisser
    # attendre devant un écran muet.
    if appareil.type == "cpu":
        print("/!\\ aucun GPU visible : exécution sur CPU, comptez ~15 min.")
        print("    Sur le cluster, cela signifie que vous êtes resté sur le")
        print("    nœud de connexion. Passez par srun ou sbatch.")
    else:
        print("GPU : %s" % torch.cuda.get_device_name(0))

    dncnn, epoch = charger_dncnn(args.checkpoint, appareil)
    print("DnCNN chargé : %s (epoch %d), gelé." % (args.checkpoint, epoch))

    # --- le lot, avec un bruit tiré UNE SEULE FOIS ------------------------
    # Point crucial. CelebADataset retire le bruit à chaque accès, ce qui est
    # exactement ce qu'on veut à l'entraînement et exactement ce qu'on ne veut
    # PAS ici : si r changeait à chaque itération, il n'y aurait rien à
    # mémoriser et le test ne prouverait rien.
    jeu = CelebADataset(split=args.split)
    lot = [jeu[i] for i in range(args.n)]
    x = torch.stack([e["x"] for e in lot]).to(appareil)
    y = torch.stack([e["y"] for e in lot]).to(appareil)

    with torch.no_grad():
        mu = dncnn(y)
    r = (x - mu).flatten(1)                    # [B, 4096]
    mu = mu.detach()

    n = r.shape[1]
    std_r = r.std().item()
    print("résidu sur ces %d images : écart-type %.4f" % (args.n, std_r))
    print("  -> init_log_diag idéal pour ce lot : %.4f" % (-math.log(std_r)))

    # Plancher de référence : la meilleure gaussienne ISOTROPE possible sur ce
    # résidu. Tout modèle qui ne fait pas mieux que ça n'a rien appris.
    nll_isotrope = 0.5 * (1.0 + math.log(2.0 * math.pi * std_r ** 2))
    print("  -> NLL/pixel d'une gaussienne isotrope ajustée : %+.4f"
          % nll_isotrope)
    print()

    neighbor_idx, mask = build_neighbor_indices(device=appareil)

    resultats = {}
    for etiquette, diag_seule in (("structuré", False), ("diagonal", True)):
        print("  %s :" % etiquette)
        resultats[etiquette] = surapprendre(
            r, mu, neighbor_idx, mask, diag_seule,
            args.iterations, args.lr, appareil, etiquette)
        print()

    # --- verdict ----------------------------------------------------------
    print("=" * 72)
    print("%-12s %12s %12s %12s" % ("modèle", "NLL init", "NLL finale",
                                    "var(L^T r)"))
    print("(var(L^T r) << 1 est ATTENDU ici : voir le commentaire du code.)")
    for etiquette in ("structuré", "diagonal"):
        init, fin, var = resultats[etiquette]
        print("%-12s %+12.4f %+12.4f %12.4f" % (etiquette, init, fin, var))
    print("%-12s %+12s %+12.4f" % ("isotrope", "-", nll_isotrope))
    print("=" * 72)

    fin_struct = resultats["structuré"][1]
    fin_diag = resultats["diagonal"][1]

    if fin_struct > nll_isotrope - 0.05:
        print("ÉCHEC : le modèle structuré ne bat même pas une gaussienne")
        print("        isotrope sur 8 images. Le bug est dans le code.")
    elif fin_struct >= fin_diag:
        print("ALERTE : le modèle structuré ne fait pas mieux que le")
        print("         diagonal. Les 24 canaux hors-diagonale ne servent à")
        print("         rien — vérifier l'ordre des canaux et le masque.")
    else:
        print("SUCCÈS : NLL/pixel %+.4f structuré contre %+.4f diagonal,"
              % (fin_struct, fin_diag))
        print("         soit %.4f nat par pixel gagné par la structure."
              % (fin_diag - fin_struct))
        print("         loss.py et cov_model.py fonctionnent ensemble.")


if __name__ == "__main__":
    main()
