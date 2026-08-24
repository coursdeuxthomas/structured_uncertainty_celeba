"""
Entraînement du DnCNN (étape 1 du projet).

    python train_dncnn.py --epochs 3 --limite 20000     # test de la chaîne
    python train_dncnn.py --epochs 50 --resume          # le vrai run

La partition `short` du cluster est limitée à 1 h 55 et un entraînement complet
demande plusieurs heures : la REPRISE SUR CHECKPOINT est écrite dès cette
première version. `--resume` repart de checkpoints/dncnn_last.pt, optimiseur et
numéro d'epoch compris.

Sorties :
    checkpoints/dncnn_last.pt     état complet, réécrit à chaque epoch
    checkpoints/dncnn_best.pt     meilleur PSNR de validation
    results/history_dncnn.json    courbes, réécrites à chaque epoch
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


# torch.amp a remplace torch.cuda.amp dans les versions recentes. On prend la
# nouvelle API si elle existe, l'ancienne sinon : le code tourne sur le cluster
# quelle que soit la version installee dans l'environnement conda.
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
# Utilitaires
# --------------------------------------------------------------------------
def psnr(mse_pm1):
    """
    PSNR en dB, exprimé dans l'échelle [0, 1] qui est la convention de la
    littérature sur le débruitage.

    Nos images vivent dans [-1, 1]. Le passage à [0, 1] divise l'amplitude par
    2, donc l'erreur quadratique par 4. Sans cette correction on annoncerait un
    PSNR faux de 6 dB.
    """
    mse_01 = mse_pm1 / 4.0
    # float() et pas np.float64 : depuis PyTorch 2.6, torch.load refuse par
    # defaut de deserialiser des scalaires numpy, et la reprise sur checkpoint
    # echouerait au moment ou on en a le plus besoin.
    return float(10.0 * np.log10(1.0 / max(mse_01, 1e-12)))


def sauvegarde_atomique(objet, chemin):
    """
    Écrit d'abord un fichier temporaire, puis le renomme.

    Un job SLURM peut être tué à n'importe quel instant. Sans cette précaution,
    une interruption pendant torch.save laisserait un checkpoint tronqué —
    c'est-à-dire un run perdu, découvert seulement à la reprise.
    """
    tmp = chemin + ".tmp"
    torch.save(objet, tmp)
    os.replace(tmp, chemin)


def verifier_noeud_de_calcul(forcer):
    """
    Refuse de demarrer hors d'un job SLURM.

    Les GPU ne sont visibles que dans un job. Lancer un entrainement sur le
    noeud de connexion le sature pour tous les utilisateurs du cluster, et
    tourne de toute facon sur CPU, donc des dizaines de fois plus lentement.
    Le garde-fou coute deux lignes et evite une erreur qu'on ne remarque pas
    tout de suite.
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
    Charge un checkpoint ecrit par ce script.

    PyTorch 2.6 a bascule torch.load sur weights_only=True par defaut, ce qui
    refuse tout ce qui n'est pas un tenseur ou un type de base. Nos checkpoints
    contiennent l'historique et les arguments : on desactive explicitement le
    garde-fou, ce qui est sans risque puisque le fichier vient de nous.
    """
    try:
        return torch.load(chemin, map_location=appareil, weights_only=False)
    except TypeError:                                     # torch tres ancien
        return torch.load(chemin, map_location=appareil)


def construire_validation(dataset, indices, sigma, graine, appareil):
    """
    Prépare une fois pour toutes le jeu de validation, avec un bruit FIGÉ.

    Le Dataset tire son bruit à la volée : deux évaluations successives sur les
    mêmes images ne donneraient pas le même PSNR, et la courbe de validation
    serait bruitée pour une raison qui n'a rien à voir avec l'apprentissage.
    On fige donc une réalisation, une bonne fois.
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
    """MSE moyenne sur le jeu de validation."""
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

    # ---- données ---------------------------------------------------------
    complet = CelebADataset("train")
    n_total = len(complet) if args.limite == 0 else min(args.limite, len(complet))

    # Le suivi se fait sur des images RETIRÉES du train, jamais vues par la
    # descente de gradient. On préfère perdre 2 000 images d'entraînement
    # plutôt que de surveiller le modèle sur le jeu de test : la NLL finale
    # doit rester une mesure honnête.
    n_val = min(args.val, n_total // 10)
    indices_val = list(range(n_total - n_val, n_total))
    indices_train = list(range(0, n_total - n_val))

    sigma = complet.sigma            # déjà converti en unités [-1, 1]
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

    # Repère : le PSNR de l'image bruitée elle-même. Le modèle doit faire
    # nettement mieux, sinon il n'apprend rien.
    mse_bruit = torch.mean((y_val - x_val) ** 2).item()
    print("PSNR de y (référence à battre) : %.2f dB" % psnr(mse_bruit))

    # ---- modèle ----------------------------------------------------------
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

    # ---- boucle ----------------------------------------------------------
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
                # MSE(mu, x) est identique à MSE(bruit prédit, bruit vrai) :
                # mu - x = (y - n_pred) - x = n_vrai - n_pred.
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

        # Réécrit à chaque epoch, pas seulement à la fin : un job tué ne doit
        # pas emporter les courbes avec lui.
        with open(os.path.join(DOSSIER_RES, "history_dncnn.json"), "w") as f:
            json.dump(historique, f, indent=2)

    print("terminé. meilleur PSNR de validation : %.2f dB" % meilleur)


if __name__ == "__main__":
    main()
