"""
Training of the covariance network (step 2 of the project).

    python train_cov.py --epochs 50 --resume                # structured model
    python train_cov.py --epochs 50 --resume --diagonale    # reference

TWO-STAGE training, as in the paper (Eq. 4, "keeping the generative model
parameters theta fixed"): the DnCNN is loaded, switched to eval() and FROZEN.
Only the covariance network learns.

    y  = x + sigma * randn
    mu = dncnn(y)                     <- under torch.no_grad()
    r  = x - mu
    log_diag, offdiag = cov_net(mu)
    loss = structured_gaussian_nll(log_diag, offdiag, r, neighbor_idx, mask)

TWO RUNS ARE NEEDED, not one.  `--diagonale` retrains the same network with
offdiag forced to zero: this is the classical heteroscedastic reference, and
the NLL gap between the two runs is what quantifies what the structure adds.
Without it the results demonstrate nothing (§6.2 of docs/tuteur.txt). The two
runs write under distinct names, they do not overwrite each other.

Outputs (prefixe = cov, or covdiag with --diagonale):
    checkpoints/<prefixe>_last.pt     full state, rewritten at each epoch
    checkpoints/<prefixe>_best.pt     best validation NLL
    results/history_<prefixe>.json    curves, rewritten at each epoch
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from cov_model import SparseCholeskyNet
from data import CelebADataset, SIGMA
from dncnn import DnCNN
from loss import build_neighbor_indices, structured_gaussian_nll

# Reused from train_dncnn.py rather than copied over: atomic saving, the
# SLURM guard, tolerant checkpoint loading and the frozen-noise validation
# set raise exactly the same problems here. A fix applied over there
# automatically benefits this script.
from train_dncnn import (charger_checkpoint, construire_validation,
                         sauvegarde_atomique, verifier_noeud_de_calcul)

DOSSIER_CKPT = "checkpoints"
DOSSIER_RES = "results"


def charger_dncnn(chemin, appareil):
    """Load the denoiser, switch it to eval() and FREEZE its weights."""
    etat = charger_checkpoint(chemin, appareil)
    modele = DnCNN().to(appareil)
    modele.load_state_dict(etat["modele"])
    modele.eval()
    # requires_grad_(False) on top of the loop's no_grad(): no_grad prevents
    # the graph from being built, this prevents the optimiser from touching
    # the weights even if one day we made a mistake while building it.
    for p in modele.parameters():
        p.requires_grad_(False)
    return modele, etat.get("epoch", -1)


@torch.no_grad()
def preparer_validation(dncnn, x_val, y_val, batch):
    """
    Compute mu and r on the validation set ONLY ONCE.

    The DnCNN is frozen and the validation noise is fixed: mu will never
    change. Recomputing it at every epoch would be GPU time thrown away, and
    the validation curve must in any case be exactly reproducible.
    """
    mus = []
    for debut in range(0, len(x_val), batch):
        mus.append(dncnn(y_val[debut:debut + batch]))
    mu = torch.cat(mus)
    r = (x_val - mu).flatten(1)
    return mu, r


@torch.no_grad()
def evaluer(cov_net, mu_val, r_val, batch, neighbor_idx, mask):
    """Mean NLL per image on the validation set."""
    cov_net.eval()
    total, n = 0.0, 0
    for debut in range(0, len(mu_val), batch):
        mu = mu_val[debut:debut + batch]
        r = r_val[debut:debut + batch]
        log_diag, offdiag = cov_net(mu)
        nll = structured_gaussian_nll(log_diag, offdiag, r,
                                      neighbor_idx, mask)
        total += nll.item() * len(mu)
        n += len(mu)
    cov_net.train()
    return total / n


def main():
    p = argparse.ArgumentParser(description="Entraînement du réseau de covariance.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dncnn", default="checkpoints/dncnn_best.pt",
                   help="le débruiteur gelé.")
    p.add_argument("--diagonale", action="store_true",
                   help="référence diagonale : offdiag forcé à zéro.")
    p.add_argument("--val", type=int, default=2000)
    p.add_argument("--limite", type=int, default=0,
                   help="n'utiliser que les N premières images (test rapide).")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--graine", type=int, default=0)
    p.add_argument("--local", action="store_true",
                   help="autorise l'execution hors d'un job SLURM.")
    args = p.parse_args()

    verifier_noeud_de_calcul(args.local)

    torch.manual_seed(args.graine)
    np.random.seed(args.graine)
    os.makedirs(DOSSIER_CKPT, exist_ok=True)
    os.makedirs(DOSSIER_RES, exist_ok=True)

    # Distinct names for the two runs: otherwise the second would overwrite
    # the first and we would lose the comparison the project rests on.
    prefixe = "covdiag" if args.diagonale else "cov"
    chemin_last = os.path.join(DOSSIER_CKPT, "%s_last.pt" % prefixe)
    chemin_best = os.path.join(DOSSIER_CKPT, "%s_best.pt" % prefixe)
    chemin_hist = os.path.join(DOSSIER_RES, "history_%s.json" % prefixe)

    appareil = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("appareil : %s" % appareil)
    if appareil.type == "cuda":
        print("GPU      : %s" % torch.cuda.get_device_name(0))
    print("modèle   : %s" % ("référence DIAGONALE" if args.diagonale
                             else "covariance STRUCTURÉE"))

    # ---- the frozen denoiser --------------------------------------------
    dncnn, epoch_dncnn = charger_dncnn(args.dncnn, appareil)
    print("DnCNN    : %s (epoch %d), gelé" % (args.dncnn, epoch_dncnn))

    # ---- data ------------------------------------------------------------
    complet = CelebADataset("train")
    n_total = len(complet) if args.limite == 0 else min(args.limite, len(complet))
    n_val = min(args.val, n_total // 10)
    indices_val = list(range(n_total - n_val, n_total))
    indices_train = list(range(0, n_total - n_val))

    sigma = complet.sigma
    x_val, y_val = construire_validation(complet, indices_val, sigma,
                                         args.graine + 1, appareil)
    mu_val, r_val = preparer_validation(dncnn, x_val, y_val, args.batch)
    del x_val, y_val

    chargeur = DataLoader(
        Subset(complet, indices_train), batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=(appareil.type == "cuda"),
        drop_last=False, persistent_workers=(args.workers > 0),
    )

    n = r_val.shape[1]
    print("train    : %d images   validation : %d images" %
          (len(indices_train), n_val))
    print("sigma    : %.4f (unités [-1, 1]), soit %.1f/255" % (sigma, SIGMA * 255))
    print("itérations par epoch : %d" % len(chargeur))

    # Reference floor: the best ISOTROPIC Gaussian on this residual.
    # Any NLL above it means the model has learnt nothing at all.
    std_val = r_val.std().item()
    nll_isotrope = 0.5 * (1.0 + math.log(2.0 * math.pi * std_val ** 2))
    print("résidu de validation : écart-type %.4f" % std_val)
    print("NLL/pixel d'une gaussienne isotrope (plancher) : %+.4f"
          % nll_isotrope)

    # ---- sparsity pattern, computed once --------------------------------
    neighbor_idx, mask = build_neighbor_indices(device=appareil)

    # ---- model -----------------------------------------------------------
    cov_net = SparseCholeskyNet(diagonale_seule=args.diagonale).to(appareil)
    optimiseur = torch.optim.Adam(cov_net.parameters(), lr=args.lr)
    print("paramètres : %d" % sum(p.numel() for p in cov_net.parameters()))

    # No mixed precision here, unlike train_dncnn.py. The NLL sums 4096
    # terms then takes an exponential: in float16 the accumulation loses too
    # many digits and exp(log_diag) comes close to the format's limit.
    # The speed gain is not worth the risk of a NaN in the third hour.

    epoch_depart, meilleur, historique = 0, float("inf"), []
    if args.resume and os.path.exists(chemin_last):
        etat = charger_checkpoint(chemin_last, appareil)
        # Guard: resuming a structured run with --diagonale (or the other
        # way round) would give an inconsistent model with no message at all.
        if etat.get("diagonale") != args.diagonale:
            raise SystemExit(
                "ERREUR : le checkpoint %s a diagonale=%s, mais le run demande "
                "diagonale=%s." % (chemin_last, etat.get("diagonale"),
                                   args.diagonale))
        cov_net.load_state_dict(etat["modele"])
        optimiseur.load_state_dict(etat["optimiseur"])
        epoch_depart = etat["epoch"]
        meilleur = etat["meilleur"]
        historique = etat["historique"]
        print("reprise depuis l'epoch %d (meilleure NLL %.2f)" %
              (epoch_depart, meilleur))
    elif args.resume:
        print("--resume demandé mais aucun checkpoint : départ de zéro.")

    # ---- loop ------------------------------------------------------------
    for epoch in range(epoch_depart, args.epochs):
        t0 = time.time()
        cumul, vus = 0.0, 0

        for k, lot in enumerate(chargeur):
            x = lot["x"].to(appareil, non_blocking=True)
            y = lot["y"].to(appareil, non_blocking=True)

            # The noise is redrawn at every access by the Dataset: the network
            # never sees the same residual twice for a given image. That is
            # what stops it memorising r instead of learning its distribution.
            with torch.no_grad():
                mu = dncnn(y)
            r = (x - mu).flatten(1)

            optimiseur.zero_grad(set_to_none=True)
            log_diag, offdiag = cov_net(mu)
            cout = structured_gaussian_nll(log_diag, offdiag, r,
                                           neighbor_idx, mask)
            cout.backward()
            optimiseur.step()

            valeur = cout.item()
            if not math.isfinite(valeur):
                raise SystemExit(
                    "ERREUR : NLL non finie a l'epoch %d, iteration %d. Le "
                    "checkpoint n'est pas ecrit : %s reste utilisable."
                    % (epoch + 1, k + 1, chemin_last))

            cumul += valeur * len(x)
            vus += len(x)
            if (k + 1) % 200 == 0:
                print("  epoch %d  %5d/%d  NLL/pixel %+.4f" %
                      (epoch + 1, k + 1, len(chargeur), cumul / vus / n),
                      flush=True)

        nll_train = cumul / vus
        nll_val = evaluer(cov_net, mu_val, r_val, args.batch,
                          neighbor_idx, mask)
        duree = time.time() - t0

        historique.append({
            "epoch": epoch + 1,
            "nll_train": nll_train,
            "nll_val": nll_val,
            "nll_train_pixel": nll_train / n,
            "nll_val_pixel": nll_val / n,
            "secondes": duree,
        })
        print("epoch %2d/%d | train %+.4f | val %+.4f (nat/pixel) | %.0f s"
              % (epoch + 1, args.epochs, nll_train / n, nll_val / n, duree),
              flush=True)

        etat = {
            "epoch": epoch + 1,
            "modele": cov_net.state_dict(),
            "optimiseur": optimiseur.state_dict(),
            "meilleur": min(meilleur, nll_val),
            "historique": historique,
            "diagonale": args.diagonale,
            "dncnn": args.dncnn,
            "args": vars(args),
        }
        sauvegarde_atomique(etat, chemin_last)
        # Smaller is better: this is an NLL, not a PSNR.
        if nll_val < meilleur:
            meilleur = nll_val
            sauvegarde_atomique(etat, chemin_best)
            print("  meilleur modèle sauvegardé (%+.4f nat/pixel)"
                  % (meilleur / n))

        with open(chemin_hist, "w") as f:
            json.dump(historique, f, indent=2)

    print("terminé. meilleure NLL de validation : %+.4f nat/pixel (plancher "
          "isotrope %+.4f)" % (meilleur / n, nll_isotrope))


if __name__ == "__main__":
    main()
