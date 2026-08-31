"""
Training of the DnCNN (step 1 of the project).

    python train_dncnn.py --epochs 3 --limite 20000     # pipeline test
    python train_dncnn.py --epochs 50 --resume          # the real run

The cluster's `short` partition is capped at 1 h 55 and a complete training
run takes several hours: CHECKPOINT RESUMING is written from this very first
version onwards. `--resume` restarts from checkpoints/dncnn_last.pt, optimiser
and epoch number included.

Outputs:
    checkpoints/dncnn_last.pt     full state, rewritten at each epoch
    checkpoints/dncnn_best.pt     best validation PSNR
    results/history_dncnn.json    curves, rewritten at each epoch
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data import CelebADataset, SIGMA
from dncnn import DnCNN


# torch.amp replaced torch.cuda.amp in recent versions. We take the new API
# if it exists, the old one otherwise: the code runs on the cluster whatever
# version happens to be installed in the conda environment.
try:
    from torch.amp import GradScaler as _GradScaler, autocast as _autocast

    def creer_echelle(actif):
        return _GradScaler("cuda", enabled=actif)

    def contexte_amp(actif):
        return _autocast("cuda", enabled=actif)
except ImportError:                                       # torch < 2.3
    def creer_echelle(actif):
        return torch.cuda.amp.GradScaler(enabled=actif)

    def contexte_amp(actif):
        return torch.cuda.amp.autocast(enabled=actif)

DOSSIER_CKPT = "checkpoints"
DOSSIER_RES = "results"


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------
def psnr(mse_pm1):
    """
    PSNR in dB, expressed in the [0, 1] scale, which is the convention of the
    denoising literature.

    Our images live in [-1, 1]. Moving to [0, 1] divides the amplitude by 2,
    hence the squared error by 4. Without this correction we would report a
    PSNR that is wrong by 6 dB.
    """
    mse_01 = mse_pm1 / 4.0
    # float() and not np.float64: since PyTorch 2.6, torch.load refuses by
    # default to deserialise numpy scalars, and checkpoint resuming would
    # fail at the very moment we need it most.
    return float(10.0 * np.log10(1.0 / max(mse_01, 1e-12)))


def sauvegarde_atomique(objet, chemin):
    """
    Write a temporary file first, then rename it.

    A SLURM job can be killed at any moment. Without this precaution, an
    interruption during torch.save would leave a truncated checkpoint —
    that is, a lost run, discovered only when resuming.
    """
    tmp = chemin + ".tmp"
    torch.save(objet, tmp)
    os.replace(tmp, chemin)


def verifier_noeud_de_calcul(forcer):
    """
    Refuse to start outside a SLURM job.

    The GPUs are only visible from inside a job. Launching a training run
    on the login node saturates it for every user of the cluster, and it
    runs on the CPU anyway, hence tens of times more slowly. The guard
    costs two lines and avoids a mistake that one does not notice straight
    away.
    """
    if forcer or os.environ.get("SLURM_JOB_ID"):
        return
    print("ERREUR : aucun job SLURM detecte (SLURM_JOB_ID est vide).")
    print("  Tu es sur le noeud de connexion, ou il n'y a pas de GPU.")
    print()
    print("  Session interactive, pour un run court :")
    print("    srun --partition=short --gres=gpu:1 --cpus-per-task=4 \\")
    print("         --mem=16G --time=01:55:00 --pty bash")
    print()
    print("  Ou en tache de fond, pour un run long :")
    print("    sbatch train_dncnn.bash")
    print()
    print("  Pour passer outre volontairement (petit test sur CPU) : --local")
    raise SystemExit(1)


def charger_checkpoint(chemin, appareil):
    """
    Load a checkpoint written by this script.

    PyTorch 2.6 switched torch.load to weights_only=True by default, which
    refuses anything that is not a tensor or a basic type. Our checkpoints
    contain the history and the arguments: we explicitly disable the guard,
    which carries no risk since the file comes from us.
    """
    try:
        return torch.load(chemin, map_location=appareil, weights_only=False)
    except TypeError:                                     # very old torch
        return torch.load(chemin, map_location=appareil)


def construire_validation(dataset, indices, sigma, graine, appareil):
    """
    Prepare the validation set once and for all, with FROZEN noise.

    The Dataset draws its noise on the fly: two successive evaluations on the
    same images would not give the same PSNR, and the validation curve would
    be noisy for a reason that has nothing to do with learning.
    So we freeze one realisation, once and for all.
    """
    generateur = torch.Generator().manual_seed(graine)
    xs = [dataset.images[i] for i in indices]
    x = torch.from_numpy(np.stack(xs).astype(np.float32) / 127.5 - 1.0)
    x = x.unsqueeze(1)                                   # [N, 1, 64, 64]
    bruit = torch.randn(x.shape, generator=generateur)
    y = x + sigma * bruit
    return x.to(appareil), y.to(appareil)


@torch.no_grad()
def evaluer(modele, x_val, y_val, batch):
    """Mean MSE on the validation set."""
    modele.eval()
    total, n = 0.0, 0
    for debut in range(0, len(x_val), batch):
        x = x_val[debut:debut + batch]
        y = y_val[debut:debut + batch]
        mu = modele(y)
        total += torch.mean((mu - x) ** 2).item() * len(x)
        n += len(x)
    modele.train()
    return total / n


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Entraînement du DnCNN.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val", type=int, default=2000,
                   help="images retirées de la fin du train pour le suivi.")
    p.add_argument("--limite", type=int, default=0,
                   help="n'utiliser que les N premières images (test rapide).")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", action="store_true",
                   help="précision mixte : environ deux fois plus rapide.")
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

    appareil = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("appareil : %s" % appareil)
    if appareil.type == "cuda":
        print("GPU      : %s" % torch.cuda.get_device_name(0))

    # ---- data ------------------------------------------------------------
    complet = CelebADataset("train")
    n_total = len(complet) if args.limite == 0 else min(args.limite, len(complet))

    # Monitoring is done on images REMOVED from the train set, never seen by
    # gradient descent. We would rather lose 2,000 training images than watch
    # the model on the test set: the final NLL must remain an honest
    # measurement.
    n_val = min(args.val, n_total // 10)
    indices_val = list(range(n_total - n_val, n_total))
    indices_train = list(range(0, n_total - n_val))

    sigma = complet.sigma            # already converted to [-1, 1] units
    x_val, y_val = construire_validation(complet, indices_val, sigma,
                                         args.graine + 1, appareil)

    chargeur = DataLoader(
        Subset(complet, indices_train), batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=(appareil.type == "cuda"),
        drop_last=False, persistent_workers=(args.workers > 0),
    )

    print("train    : %d images   validation : %d images" %
          (len(indices_train), n_val))
    print("sigma    : %.4f (unités [-1, 1]),  soit %.1f/255" %
          (sigma, SIGMA * 255))
    print("itérations par epoch : %d" % len(chargeur))

    # Reference point: the PSNR of the noisy image itself. The model must do
    # markedly better, otherwise it is learning nothing.
    mse_bruit = torch.mean((y_val - x_val) ** 2).item()
    print("PSNR de y (référence à battre) : %.2f dB" % psnr(mse_bruit))

    # ---- model -----------------------------------------------------------
    modele = DnCNN().to(appareil)
    optimiseur = torch.optim.Adam(modele.parameters(), lr=args.lr)
    echelle = creer_echelle(args.amp and appareil.type == "cuda")

    epoch_depart, meilleur, historique = 0, -1.0, []
    chemin_last = os.path.join(DOSSIER_CKPT, "dncnn_last.pt")
    if args.resume and os.path.exists(chemin_last):
        etat = charger_checkpoint(chemin_last, appareil)
        modele.load_state_dict(etat["modele"])
        optimiseur.load_state_dict(etat["optimiseur"])
        epoch_depart = etat["epoch"]
        meilleur = etat["meilleur"]
        historique = etat["historique"]
        print("reprise depuis l'epoch %d (meilleur PSNR %.2f dB)" %
              (epoch_depart, meilleur))
    elif args.resume:
        print("--resume demandé mais aucun checkpoint : départ de zéro.")

    # ---- loop ------------------------------------------------------------
    perte = nn.MSELoss()
    for epoch in range(epoch_depart, args.epochs):
        t0 = time.time()
        cumul, vus = 0.0, 0

        for k, lot in enumerate(chargeur):
            x = lot["x"].to(appareil, non_blocking=True)
            y = lot["y"].to(appareil, non_blocking=True)

            optimiseur.zero_grad(set_to_none=True)
            with contexte_amp(echelle.is_enabled()):
                mu = modele(y)
                # MSE(mu, x) is identical to MSE(predicted noise, true
                # noise): mu - x = (y - n_pred) - x = n_true - n_pred.
                cout = perte(mu, x)
            echelle.scale(cout).backward()
            echelle.step(optimiseur)
            echelle.update()

            cumul += cout.item() * len(x)
            vus += len(x)
            if (k + 1) % 200 == 0:
                print("  epoch %d  %5d/%d  perte %.5f" %
                      (epoch + 1, k + 1, len(chargeur), cumul / vus), flush=True)

        mse_train = cumul / vus
        mse_val = evaluer(modele, x_val, y_val, args.batch)
        p_val = psnr(mse_val)
        duree = time.time() - t0

        historique.append({
            "epoch": epoch + 1,
            "mse_train": mse_train,
            "mse_val": mse_val,
            "psnr_val": p_val,
            "secondes": duree,
        })
        print("epoch %2d/%d | train %.5f | val %.5f | PSNR %.2f dB | %.0f s"
              % (epoch + 1, args.epochs, mse_train, mse_val, p_val, duree),
              flush=True)

        etat = {
            "epoch": epoch + 1,
            "modele": modele.state_dict(),
            "optimiseur": optimiseur.state_dict(),
            "meilleur": max(meilleur, p_val),
            "historique": historique,
            "args": vars(args),
        }
        sauvegarde_atomique(etat, chemin_last)
        if p_val > meilleur:
            meilleur = p_val
            sauvegarde_atomique(etat, os.path.join(DOSSIER_CKPT, "dncnn_best.pt"))
            print("  meilleur modèle sauvegardé (%.2f dB)" % meilleur)

        # Rewritten at each epoch, not only at the end: a killed job must not
        # take the curves down with it.
        with open(os.path.join(DOSSIER_RES, "history_dncnn.json"), "w") as f:
            json.dump(historique, f, indent=2)

    print("terminé. meilleur PSNR de validation : %.2f dB" % meilleur)


if __name__ == "__main__":
    main()
