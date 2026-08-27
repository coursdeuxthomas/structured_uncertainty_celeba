"""
L'application du §5.3 : récupérer le détail que le DnCNN a effacé.

    python denoise.py                   # balaye tau, puis évalue sur 2000 images
    python denoise.py --tau 0.038       # tau imposé
    python denoise.py --verif           # auto-test du solveur, sans données

L'IDÉE
    mu = DnCNN(y) est lisse : le débruiteur a enlevé le bruit ET le détail
    haute-fréquence. Ce qu'il a retiré est entièrement contenu dans

        s = y - mu = (y - x) + (x - mu) = bruit + résidu

    Deux choses mélangées, dont une seule est à récupérer. Sigma décrit la
    forme des résidus PLAUSIBLES pour un visage ; le bruit blanc, lui, ne
    ressemble à aucun résidu plausible. Filtrer s à travers Sigma garde donc le
    détail et jette le bruit. Image finale : mu + f(s).

LE FILTRE, ET POURQUOI CE N'EST PAS CELUI DE L'ARTICLE
    L'article projette s sur les 1000 premiers vecteurs propres de Sigma. Une
    décomposition spectrale d'une matrice 4096 x 4096 par image coûte O(n^3) ;
    sur 2000 images c'est hors de question (et il faudrait former Sigma, ce que
    tout le projet s'interdit).

    On prend le filtre de Wiener, qui a la même intention — atténuer ce qui
    n'est pas dans les directions dominantes de Sigma — et qui se simplifie
    remarquablement :

        f(s) = Sigma (Sigma + tau I)^-1 s = (I + tau Lambda)^-1 s

    Il n'y a plus de Sigma du tout, seulement la PRÉCISION, que le réseau
    prédit directement. Le système est symétrique défini positif : gradient
    conjugué, chaque produit `Lambda v = L (L^T v)` coûtant O(n*m). Aucune
    matrice n x n n'est formée, aucune valeur propre n'est calculée.

    Mieux : tau n'est pas un réglage arbitraire. Si l'on modélise s = r + bruit
    avec r ~ N(0, Sigma) et un bruit de variance sigma^2 indépendant, alors

        E[r | s] = Sigma (Sigma + sigma^2 I)^-1 s

    c'est-à-dire EXACTEMENT ce filtre avec tau = sigma^2. Le meilleur tau
    trouvé empiriquement se lit donc comme un diagnostic : proche de sigma^2,
    la covariance prédite est à la bonne échelle ; beaucoup plus grand, le
    modèle est trop confiant et il faut le brider. (L'indépendance entre r et
    le bruit est fausse en toute rigueur — r dépend de y, donc du bruit — mais
    c'est une approximation raisonnable, et le balayage de tau la corrige.)

CE QUE VAUT LA COMPARAISON
    Quand tau tend vers l'infini, f(s) tend vers 0 et l'on retombe exactement
    sur le DnCNN seul. La référence est donc un CAS LIMITE de la méthode : un
    tau choisi sur un jeu de validation ne peut pas faire pire, et tout gain
    mesuré est réel.

ET POURQUOI CE GAIN SERA QUASI NUL — à lire avant de s'en étonner
    Le DnCNN est entraîné en MSE, et le minimiseur de la MSE est la MOYENNE
    CONDITIONNELLE : à l'optimum, mu(y) = E[x | y], donc E[r | y] = 0. Or
    s = y - mu(y) est une fonction déterministe de y. Pour toute correction
    additive g(y) :

        E||x - mu - g||^2 = E||r||^2 - 2 E[r . g(y)] + E||g||^2
                          = E||r||^2 + E||g||^2   >=  E||r||^2

    puisque E[r . g(y)] = E[ E[r|y] . g(y) ] = 0. AUCUNE correction calculée à
    partir de y ne peut réduire la MSE d'un débruiteur MSE-optimal, et la
    meilleure est g = 0. Le gain mesuré ici ne dit donc rien de la qualité de
    la covariance : il mesure L'ÉCART DU DnCNN À L'OPTIMUM, ce qui reste un
    résultat en soi.

    L'article n'a pas ce problème : son mu vient d'un VAE qui n'a JAMAIS vu de
    bruit, donc très loin de E[x|y], d'où les 40 % de la Table 4. Notre
    variante remplace ce VAE par un vrai débruiteur, et c'est exactement ce qui
    rend le §5.3 structurellement ingrat ici.

    Ce que la covariance apporte est ailleurs, et ne se mesure pas en MSE : mu
    est flou parce qu'une moyenne conditionnelle est floue, et mu + eps
    échantillonné est net. C'est le compromis perception / distorsion, et c'est
    l'objet de la figure de eval_cov.py, pas de ce tableau.

Sorties :
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
# Même raison que dans train_cov.py et eval_cov.py : ces briques posent ici le
# même problème qu'ailleurs, autant les importer que les recopier.
from eval_cov import appliquer_lambda, charger_cov
from train_cov import charger_dncnn
from train_dncnn import construire_validation, psnr

DOSSIER_RES = "results"


# --------------------------------------------------------------------------
# Solveur
# --------------------------------------------------------------------------
@torch.no_grad()
def gradient_conjugue(matvec, b, iterations=60, tol=1e-6):
    """
    Résout `A f = b` pour A symétrique définie positive, sans former A.

    A n'est connue que par son action `matvec`. C'est toute la raison d'être du
    gradient conjugué ici : notre A vaut `I + tau * L L^T`, dont le produit
    par un vecteur coûte O(n*m) alors que la matrice pèserait 67 Mo.

    Tout est vectorisé sur le batch : chaque image a son propre système, ses
    propres coefficients alpha et beta, et le critère d'arrêt porte sur la PIRE
    image du lot.

    Retour : (f, résidu relatif maximal, nombre d'itérations effectuées).
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
    f(s) = (I + tau * Lambda)^-1 s, pour un lot d'images.

    `cov_net` est appelé UNE fois par lot : les 25 x 4096 coefficients prédits
    servent aux 60 itérations du gradient conjugué.
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
    MSE de `mu + f(s)` contre `x`, image par image, en unités [-1, 1].

    Renvoie aussi la MSE du DnCNN seul — mesurée sur les MÊMES images et le
    MÊME bruit, sans quoi la comparaison ne voudrait rien dire — et, si
    `garder > 0`, les `garder` premières images reconstruites pour la figure.
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
    """x | y | mu | mu + f(s) pour chaque modèle disponible."""
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
# Auto-test (--verif)
# --------------------------------------------------------------------------
def verifications():
    """
    Le gradient conjugué contre une résolution dense, en 16x16.

    C'est le seul endroit du projet où l'on peut confronter le solveur à la
    vérité : à n = 256 la matrice `I + tau L L^T` tient dans 262 ko et
    `torch.linalg.solve` donne la réponse exacte.
    """
    from loss import build_L_dense

    torch.manual_seed(0)
    S, B, tau = 16, 3, 0.0385
    n = S * S
    neighbor_idx, mask = build_neighbor_indices(S, VOISINAGE)
    m = neighbor_idx.shape[1]

    log_diag = 2.0 + 0.3 * torch.randn(B, n)      # diagonale ~ exp(2.66)
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

    # Les deux cas limites du filtre, qui doivent tomber juste sans calcul.
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

    # ---- choix de tau, sur des images qui ne sont PAS le jeu de test ------
    # On prend la fin du split d'entraînement : ce sont les images que
    # train_cov.py a retirées du gradient pour sa courbe de validation. Régler
    # tau sur le jeu de test ferait du chiffre final une mesure truquée.
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
        # Grille logarithmique centrée sur la valeur théorique : 13 points
        # symétriques, donc sigma^2 en fait exactement partie.
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
            # Les deux bords ne disent pas la même chose, et confondre les
            # deux ferait passer un aveu d'échec pour un réglage à affiner.
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

    # ---- évaluation sur le jeu de test -----------------------------------
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
            # Les images vivent dans [-1, 1] ; la littérature (et la Table 4 de
            # l'article) compte en [0, 1], soit une MSE quatre fois plus
            # petite. On donne les deux pour éviter tout malentendu.
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

    # ---- résumé ----------------------------------------------------------
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
            print("  tuteur.txt avant d'y voir un échec.")

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
