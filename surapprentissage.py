"""
Deliberate overfitting test — stopping criterion of phase 4.

We take 8 images, we freeze the noise, we compute the residual
`r = x - DnCNN(y)` ONCE, and we train the covariance network on those 8
residuals and nothing else. With 518 681 parameters for 8 examples, the NLL
must collapse.

    If it does not go down on 8 images, the problem is in the code, not in
    the data.

This is the only test that checks loss.py and cov_model.py TOGETHER, and it
costs two minutes against five hours for a real training run.

The same test runs for the structured model and for the diagonal baseline.
The gap between the two is a first measure, on 8 images, of what the
structure brings — the real measure will come from eval_cov.py on the test
set.

Usage:
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
    """Load the trained denoiser, switch it to eval and FREEZE its weights."""
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
    Train a brand-new covariance network on a FIXED batch of residuals.

    Returns (initial NLL, final NLL, variance of w = L^T r).
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

    # Variance of w = L^T r, for information ONLY.
    #
    # On held-out data, w must follow N(0, I) and its variance is 1: that is
    # the calibration test of eval_cov.py. HERE, NOT AT ALL. On 8 frozen
    # residuals, 518 681 parameters can drive the precision towards infinity
    # and the NLL towards minus infinity: the variance of w collapses well
    # below 1, and that is the PROOF that the overfitting works. Do not read
    # this figure as a calibration flaw.
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

    # On the login node of the cluster there is no GPU: torch silently falls
    # back to the CPU and the test takes a quarter of an hour instead of two
    # minutes. Better to say so straight away than to leave someone waiting
    # in front of a silent screen.
    if appareil.type == "cpu":
        print("/!\\ aucun GPU visible : exécution sur CPU, comptez ~15 min.")
        print("    Sur le cluster, cela signifie que vous êtes resté sur le")
        print("    nœud de connexion. Passez par srun ou sbatch.")
    else:
        print("GPU : %s" % torch.cuda.get_device_name(0))

    dncnn, epoch = charger_dncnn(args.checkpoint, appareil)
    print("DnCNN chargé : %s (epoch %d), gelé." % (args.checkpoint, epoch))

    # --- the batch, with the noise drawn ONLY ONCE ------------------------
    # Crucial point. CelebADataset redraws the noise on every access, which is
    # exactly what we want during training and exactly what we do NOT want
    # here: if r changed at every iteration, there would be nothing to
    # memorise and the test would prove nothing.
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

    # Reference floor: the best possible ISOTROPIC Gaussian on this residual.
    # Any model that does not do better than that has learned nothing.
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
