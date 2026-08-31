"""
The §5.3 application: recovering the detail the DnCNN erased.

    python denoise.py                   # sweep tau, then evaluate on 2000 images
    python denoise.py --tau 0.038       # tau imposed
    python denoise.py --verif           # solver self-test, no data needed

THE IDEA
    mu = DnCNN(y) is smooth: the denoiser removed the noise AND the
    high-frequency detail. What it took away is entirely contained in

        s = y - mu = (y - x) + (x - mu) = noise + residual

    Two things mixed together, only one of which is to be recovered. Sigma
    describes the shape of the residuals that are PLAUSIBLE for a face; white
    noise, for its part, resembles no plausible residual at all. Filtering s
    through Sigma therefore keeps the detail and throws the noise away. Final
    image: mu + f(s).

THE FILTER, AND WHY IT IS NOT THE ONE FROM THE PAPER
    The paper projects s onto the first 1000 eigenvectors of Sigma. A spectral
    decomposition of a 4096 x 4096 matrix per image costs O(n^3); on 2000
    images that is out of the question (and Sigma would have to be formed,
    which the whole project forbids itself).

    We take the Wiener filter, which has the same intent — attenuate whatever
    does not lie in the dominant directions of Sigma — and which simplifies
    remarkably:

        f(s) = Sigma (Sigma + tau I)^-1 s = (I + tau Lambda)^-1 s

    There is no Sigma left at all, only the PRECISION, which the network
    predicts directly. The system is symmetric positive definite: conjugate
    gradient, each product `Lambda v = L (L^T v)` costing O(n*m). No n x n
    matrix is ever formed, no eigenvalue is ever computed.

    Better still: tau is not an arbitrary knob. If one models s = r + noise
    with r ~ N(0, Sigma) and independent noise of variance sigma^2, then

        E[r | s] = Sigma (Sigma + sigma^2 I)^-1 s

    that is to say EXACTLY this filter with tau = sigma^2. The best tau found
    empirically therefore reads as a diagnostic: close to sigma^2, the
    predicted covariance is at the right scale; much larger, the model is
    overconfident and has to be reined in. (The independence between r and the
    noise is strictly speaking false — r depends on y, hence on the noise —
    but it is a reasonable approximation, and the tau sweep corrects for it.)

WHAT THE COMPARISON IS WORTH
    When tau tends to infinity, f(s) tends to 0 and we fall back exactly on
    the DnCNN alone. The baseline is therefore a LIMITING CASE of the method:
    a tau chosen on a validation set cannot do worse, and any measured gain is
    real.

AND WHY THAT GAIN WILL BE ALL BUT ZERO — read this before being surprised
    The DnCNN is trained in MSE, and the minimiser of the MSE is the
    CONDITIONAL MEAN: at the optimum, mu(y) = E[x | y], hence E[r | y] = 0.
    But s = y - mu(y) is a deterministic function of y. For any additive
    correction g(y):

        E||x - mu - g||^2 = E||r||^2 - 2 E[r . g(y)] + E||g||^2
                          = E||r||^2 + E||g||^2   >=  E||r||^2

    since E[r . g(y)] = E[ E[r|y] . g(y) ] = 0. NO correction computed from y
    can reduce the MSE of an MSE-optimal denoiser, and the best one is g = 0.
    The gain measured here therefore says nothing about the quality of the
    covariance: it measures THE DnCNN'S DISTANCE FROM THE OPTIMUM, which is
    still a result in itself.

    The paper does not have this problem: its mu comes from a VAE that has
    NEVER seen noise, hence one very far from E[x|y], which is where the 40 %
    of Table 4 come from. Our variant replaces that VAE with a genuine
    denoiser, and that is exactly what makes §5.3 structurally thankless here.

    What the covariance brings lies elsewhere, and is not measured in MSE: mu
    is blurry because a conditional mean is blurry, and mu + sampled eps is
    sharp. That is the perception / distortion trade-off, and it is the
    subject of the eval_cov.py figure, not of this table.

Outputs:
    results/denoise.json
    results/denoise.png
"""

import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import CelebADataset
from loss import IMAGE_SIZE, VOISINAGE, build_neighbor_indices
# Same reason as in train_cov.py and eval_cov.py: these building blocks pose
# the same problem here as elsewhere, so importing them beats copying them.
from eval_cov import appliquer_lambda, charger_cov
from train_cov import charger_dncnn
from train_dncnn import construire_validation, psnr

DOSSIER_RES = "results"


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------
@torch.no_grad()
def gradient_conjugue(matvec, b, iterations=60, tol=1e-6):
    """
    Solves `A f = b` for A symmetric positive definite, without forming A.

    A is known only through its action `matvec`. That is the whole reason
    conjugate gradient is used here: our A is `I + tau * L L^T`, whose product
    with a vector costs O(n*m) whereas the matrix itself would weigh 67 MB.

    Everything is vectorised over the batch: each image has its own system,
    its own alpha and beta coefficients, and the stopping criterion bears on
    the WORST image of the batch.

    Returns: (f, maximum relative residual, number of iterations performed).
    """
    f = torch.zeros_like(b)
    reste = b.clone()
    p = reste.clone()
    rs = (reste * reste).sum(dim=1)
    norme_b = rs.sqrt().clamp_min(1e-30)

    it = 0
    for it in range(1, iterations + 1):
        Ap = matvec(p)
        alpha = rs / (p * Ap).sum(dim=1).clamp_min(1e-30)
        f = f + alpha.unsqueeze(1) * p
        reste = reste - alpha.unsqueeze(1) * Ap
        rs_neuf = (reste * reste).sum(dim=1)
        if float((rs_neuf.sqrt() / norme_b).max()) < tol:
            rs = rs_neuf
            break
        p = reste + (rs_neuf / rs.clamp_min(1e-30)).unsqueeze(1) * p
        rs = rs_neuf

    return f, float((rs.sqrt() / norme_b).max()), it


@torch.no_grad()
def projeter(cov_net, mu, s, tau, neighbor_idx, mask, iterations, tol):
    """
    f(s) = (I + tau * Lambda)^-1 s, for a batch of images.

    `cov_net` is called ONCE per batch: the 25 x 4096 predicted coefficients
    serve the 60 iterations of the conjugate gradient.
    """
    log_diag, offdiag = cov_net(mu)

    def matvec(v):
        return v + tau * appliquer_lambda(log_diag, offdiag, v,
                                          neighbor_idx, mask)

    return gradient_conjugue(matvec, s, iterations, tol)


@torch.no_grad()
def mse_par_image(cov_net, dncnn, x, y, tau, batch, neighbor_idx, mask,
                  iterations, tol, garder=0):
    """
    MSE of `mu + f(s)` against `x`, image by image, in [-1, 1] units.

    Also returns the MSE of the DnCNN alone — measured on the SAME images and
    the SAME noise, without which the comparison would mean nothing — and, if
    `garder > 0`, the first `garder` reconstructed images for the figure.
    """
    mses, mses_dncnn, images = [], [], []
    gardees, reste_relatif, iterations_faites = 0, 0.0, 0

    for d in range(0, len(x), batch):
        xb, yb = x[d:d + batch], y[d:d + batch]
        mu = dncnn(yb)
        s = (yb - mu).flatten(1)

        f, res, nb = projeter(cov_net, mu, s, tau, neighbor_idx, mask,
                              iterations, tol)
        reste_relatif = max(reste_relatif, res)
        iterations_faites = max(iterations_faites, nb)

        xhat = mu.flatten(1) + f
        cible = xb.flatten(1)
        mses.append(((xhat - cible) ** 2).mean(dim=1).cpu())
        mses_dncnn.append(((mu.flatten(1) - cible) ** 2).mean(dim=1).cpu())

        if gardees < garder:
            manque = garder - gardees
            images.append(xhat[:manque].reshape(-1, IMAGE_SIZE,
                                                IMAGE_SIZE).cpu())
            gardees += min(manque, len(xb))

    sortie = {
        "mse": torch.cat(mses),
        "mse_dncnn": torch.cat(mses_dncnn),
        "residu_cg": reste_relatif,
        "iterations_cg": iterations_faites,
    }
    if garder:
        sortie["images"] = torch.cat(images).numpy()
    return sortie


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def figure_denoise(x, y, mu, colonnes, chemin):
    """x | y | mu | mu + f(s) for each available model."""
    toutes = [("x  (propre)", x), ("y  (bruitée)", y),
              ("mu = DnCNN(y)", mu)] + colonnes
    n_lignes, n_col = x.shape[0], len(toutes)

    fig, axes = plt.subplots(n_lignes, n_col,
                             figsize=(2.3 * n_col, 2.4 * n_lignes))
    axes = np.atleast_2d(axes)
    for j, (titre, images) in enumerate(toutes):
        for i in range(n_lignes):
            axes[i, j].imshow(images[i], cmap="gray", vmin=-1, vmax=1)
            axes[i, j].axis("off")
        axes[0, j].set_title(titre, fontsize=10)
    plt.tight_layout()
    plt.savefig(chemin, dpi=140)
    plt.close()


# --------------------------------------------------------------------------
# Self-test (--verif)
# --------------------------------------------------------------------------
def verifications():
    """
    The conjugate gradient against a dense solve, in 16x16.

    This is the only place in the project where the solver can be confronted
    with the truth: at n = 256 the matrix `I + tau L L^T` fits in 262 kB and
    `torch.linalg.solve` gives the exact answer.
    """
    from loss import build_L_dense

    torch.manual_seed(0)
    S, B, tau = 16, 3, 0.0385
    n = S * S
    neighbor_idx, mask = build_neighbor_indices(S, VOISINAGE)
    m = neighbor_idx.shape[1]

    log_diag = 2.0 + 0.3 * torch.randn(B, n)      # diagonal ~ exp(2.66)
    offdiag = 0.5 * torch.randn(B, n, m)
    s = torch.randn(B, n)

    L = build_L_dense(log_diag, offdiag, neighbor_idx, mask)
    A = torch.eye(n).expand(B, n, n) + tau * (L @ L.transpose(1, 2))
    exact = torch.linalg.solve(A, s.unsqueeze(2)).squeeze(2)

    def matvec(v):
        return v + tau * appliquer_lambda(log_diag, offdiag, v,
                                          neighbor_idx, mask)

    print("=== gradient conjugué contre résolution dense (16x16) ===")
    conditionnement = torch.linalg.eigvalsh(A)
    print("conditionnement de A : %.1f"
          % (conditionnement[..., -1].max() / conditionnement[..., 0].min()))
    for iterations in (10, 30, 100):
        f, res, nb = gradient_conjugue(matvec, s, iterations, 1e-10)
        err = (f - exact).norm() / exact.norm()
        print("  %3d itérations -> erreur relative %.2e (résidu CG %.1e)"
              % (nb, err.item(), res))
    assert err < 1e-4, "le gradient conjugué ne converge pas vers la solution"

    # The filter's two limiting cases, which must be exact with no computation.
    f0, _, _ = gradient_conjugue(lambda v: v + 0.0 * v, s, 5, 1e-12)
    assert (f0 - s).abs().max() < 1e-5, "tau = 0 doit rendre s inchangé"
    print("tau = 0 : f(s) = s               OK")
    print()
    print("OK — le solveur est juste.")


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Débruitage par projection (§5.3).")
    p.add_argument("--cov", default="checkpoints/cov_best.pt")
    p.add_argument("--covdiag", default="checkpoints/covdiag_best.pt")
    p.add_argument("--dncnn", default="checkpoints/dncnn_best.pt")
    p.add_argument("--tau", type=float, default=0.0,
                   help="0 = balayage sur le jeu de validation.")
    p.add_argument("--n", type=int, default=2000,
                   help="images de test évaluées (2000 = protocole Table 4).")
    p.add_argument("--n_val", type=int, default=256,
                   help="images de validation pour le choix de tau.")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--cg", type=int, default=60, help="itérations max du CG.")
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--figures", type=int, default=5)
    p.add_argument("--graine", type=int, default=0)
    p.add_argument("--verif", action="store_true")
    args = p.parse_args()

    if args.verif:
        verifications()
        return

    appareil = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.graine)
    os.makedirs(DOSSIER_RES, exist_ok=True)
    if appareil.type == "cpu":
        print("/!\\ aucun GPU visible : ce sera lent. Passez par srun/sbatch.")
    else:
        print("GPU : %s" % torch.cuda.get_device_name(0))

    dncnn, epoch_dncnn = charger_dncnn(args.dncnn, appareil)
    print("DnCNN : %s (epoch %d), gelé" % (args.dncnn, epoch_dncnn))

    modeles = {}
    cov, etat = charger_cov(args.cov, appareil)
    if etat["diagonale"]:
        raise SystemExit("ERREUR : %s est un modèle diagonal." % args.cov)
    modeles["structuré"] = cov
    if os.path.exists(args.covdiag):
        covd, etatd = charger_cov(args.covdiag, appareil)
        if not etatd["diagonale"]:
            raise SystemExit("ERREUR : %s n'est pas diagonal." % args.covdiag)
        modeles["diagonal"] = covd
        print("référence diagonale chargée : %s" % args.covdiag)
    else:
        print("/!\\ pas de référence diagonale (%s) : on ne saura pas si le"
              % args.covdiag)
        print("    gain vient de la STRUCTURE ou de la simple hétéroscédasticité.")

    neighbor_idx, mask = build_neighbor_indices(device=appareil)

    jeu_test = CelebADataset("test")
    sigma = jeu_test.sigma
    tau_theorique = sigma ** 2
    print("sigma = %.4f  ->  tau théorique (= sigma^2) = %.5f"
          % (sigma, tau_theorique))

    # ---- choosing tau, on images that are NOT the test set ---------------
    # We take the tail of the training split: these are the images that
    # train_cov.py removed from the gradient for its validation curve. Tuning
    # tau on the test set would make the final figure a rigged measurement.
    taus = {}
    if args.tau > 0:
        for nom in modeles:
            taus[nom] = args.tau
        print("tau imposé : %.5f (pas de balayage)" % args.tau)
    else:
        jeu_val = CelebADataset("train")
        idx_val = list(range(len(jeu_val) - args.n_val, len(jeu_val)))
        xv, yv = construire_validation(jeu_val, idx_val, sigma,
                                       args.graine + 3, appareil)
        # Logarithmic grid centred on the theoretical value: 13 symmetric
        # points, so sigma^2 is exactly one of them.
        grille = tau_theorique * np.logspace(-1.5, 1.5, 13)
        print()
        print("balayage de tau sur %d images de validation" % args.n_val)
        balayage = {}
        for nom, reseau in modeles.items():
            scores = []
            for tau in grille:
                res = mse_par_image(reseau, dncnn, xv, yv, float(tau),
                                    args.batch, neighbor_idx, mask,
                                    args.cg, args.tol)
                scores.append(float(res["mse"].mean()))
            k = int(np.argmin(scores))
            taus[nom] = float(grille[k])
            balayage[nom] = {"grille": [float(t) for t in grille],
                             "mse": scores, "tau": float(grille[k])}
            print("  %-10s tau* = %.5f  (%.2f x sigma^2)  MSE %.3e"
                  % (nom, grille[k], grille[k] / tau_theorique, scores[k]))
            # The two edges do not say the same thing, and conflating them
            # would pass an admission of failure off as a knob to fine-tune.
            if k == 0:
                print("    /!\\ optimum au bord BAS : f(s) tend vers s, donc")
                print("        mu + f(s) tend vers y. Le filtre préfère "
                      "l'image bruitée")
                print("        au débruitage — le DnCNN est mauvais, ou la "
                      "covariance est fausse.")
            elif k == len(grille) - 1:
                print("    /!\\ optimum au bord HAUT : f(s) tend vers 0, donc")
                print("        mu + f(s) tend vers mu. Le filtre refuse de "
                      "remettre quoi que ce soit :")
                print("        à ce niveau de bruit il n'y a rien à récupérer. "
                      "Résultat négatif, à dire tel quel.")
        del xv, yv

    # ---- evaluation on the test set --------------------------------------
    n_test = len(jeu_test) if args.n == 0 else min(args.n, len(jeu_test))
    x, y = construire_validation(jeu_test, list(range(n_test)), sigma,
                                 args.graine, appareil)
    print()
    print("évaluation sur %d images de test" % n_test)

    resultats, colonnes = {}, []
    for nom in ("diagonal", "structuré"):
        if nom not in modeles:
            continue
        res = mse_par_image(modeles[nom], dncnn, x, y, taus[nom], args.batch,
                            neighbor_idx, mask, args.cg, args.tol,
                            garder=args.figures)
        mse = res["mse"]
        resultats[nom] = {
            "tau": taus[nom],
            "tau_sur_sigma2": taus[nom] / tau_theorique,
            # Images live in [-1, 1]; the literature (and Table 4 of the
            # paper) counts in [0, 1], i.e. an MSE four times smaller. Both
            # are given to rule out any misunderstanding.
            "mse_pm1": float(mse.mean()),
            "mse_01": float(mse.mean()) / 4.0,
            "mse_01_ecart_type": float(mse.std()) / 4.0,
            "psnr": psnr(float(mse.mean())),
            "images_ameliorees": float((mse < res["mse_dncnn"]).float().mean()),
            "residu_cg": res["residu_cg"],
            "iterations_cg": res["iterations_cg"],
        }
        colonnes.append(("mu + f(s)  %s" % nom, res["images"]))
        if nom == "structuré":
            mse_dncnn = res["mse_dncnn"]

    reference = {
        "mse_pm1": float(mse_dncnn.mean()),
        "mse_01": float(mse_dncnn.mean()) / 4.0,
        "mse_01_ecart_type": float(mse_dncnn.std()) / 4.0,
        "psnr": psnr(float(mse_dncnn.mean())),
    }

    # ---- figure ----------------------------------------------------------
    figure_denoise(x[:args.figures, 0].cpu().numpy(),
                   y[:args.figures, 0].cpu().numpy(),
                   dncnn(y[:args.figures]).detach()[:, 0].cpu().numpy(),
                   colonnes, os.path.join(DOSSIER_RES, "denoise.png"))

    # ---- summary ---------------------------------------------------------
    print()
    print("=" * 76)
    print("%-22s %14s %10s %12s" % ("méthode", "MSE [0,1]", "PSNR dB",
                                    "images +"))
    print("-" * 76)
    print("%-22s %8.3e         %7.2f %12s"
          % ("DnCNN seul", reference["mse_01"], reference["psnr"], "-"))
    for nom in ("diagonal", "structuré"):
        if nom not in resultats:
            continue
        R = resultats[nom]
        print("%-22s %8.3e ±%.0e %7.2f %11.0f %%"
              % ("mu + f(s)  %s" % nom, R["mse_01"], R["mse_01_ecart_type"],
                 R["psnr"], 100 * R["images_ameliorees"]))
    print("=" * 76)
    print("rappel article (Table 4, protocole différent) : DAE 5.13e-3,")
    print("  covariance structurée 2.99e-3. Notre référence est un vrai")
    print("  débruiteur, pas un autoencodeur : l'adversaire est plus coriace.")

    if "structuré" in resultats:
        gain = 100.0 * (1.0 - resultats["structuré"]["mse_01"]
                        / reference["mse_01"])
        print()
        print("GAIN DU MODÈLE STRUCTURÉ SUR LE DnCNN SEUL : %+.2f %% de MSE,"
              " %+.3f dB" % (gain, resultats["structuré"]["psnr"]
                             - reference["psnr"]))
        if gain <= 1.0:
            print("  Gain quasi nul, et c'est ATTENDU : le DnCNN est entraîné")
            print("  en MSE, donc mu approche E[x|y] et E[r|y] = 0. Comme")
            print("  s = y - mu est une fonction de y, aucune correction")
            print("  additive ne peut réduire la MSE. Ce %+.2f %% mesure"
                  % gain)
            print("  l'écart du DnCNN à l'optimum, pas la qualité de Sigma.")
            print("  Voir le docstring de ce fichier et la section 7 de")
            print("  docs/tuteur.txt avant d'y voir un échec.")

    sortie = {
        "n_images": n_test,
        "sigma": sigma,
        "tau_theorique": tau_theorique,
        "dncnn_seul": reference,
        "modeles": resultats,
    }
    if args.tau <= 0:
        sortie["balayage"] = balayage
    with open(os.path.join(DOSSIER_RES, "denoise.json"), "w") as f:
        json.dump(sortie, f, indent=2)
    print()
    print("écrit : results/denoise.json, results/denoise.png")


if __name__ == "__main__":
    main()
