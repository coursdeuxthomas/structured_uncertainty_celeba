"""
Orchestration : où en est le projet, et que lancer ensuite.

    python main.py                # tableau de bord
    python main.py --evaluation   # lance eval_cov.py puis denoise.py

Le projet se déroule en huit étapes, dont trois sont de longs entraînements sur
GPU. Ce script ne les lance PAS : un entraînement de cinq heures passe par
sbatch, pas par un sous-processus qui mourrait avec la session SSH. Il lit ce
qui a déjà été produit, affiche les chiffres obtenus, et donne la prochaine
commande à taper.

Il n'importe volontairement ni torch ni numpy : tout se lit dans les fichiers
JSON écrits à chaque epoch. Le tableau de bord fonctionne donc partout, y
compris sur le nœud de connexion sans environnement conda activé.
"""

import argparse
import json
import os
import subprocess
import sys

DOSSIER_CKPT = "checkpoints"
DOSSIER_RES = "results"
N_PIXELS = 64 * 64


def lire_json(chemin):
    """Renvoie le contenu du JSON, ou None s'il n'existe pas ou est illisible."""
    try:
        with open(chemin) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def taille(chemin):
    try:
        return "%.0f Mo" % (os.path.getsize(chemin) / 1e6)
    except OSError:
        return "?"


# --------------------------------------------------------------------------
# Les huit étapes, dans l'ordre où elles doivent être franchies
# --------------------------------------------------------------------------
def etapes():
    """
    Chaque étape : (nom, fichier témoin, commande, fonction de résumé).

    Le « fichier témoin » est ce qui prouve que l'étape a été franchie. Pour
    les entraînements c'est le checkpoint du meilleur modèle, pas le dernier :
    un run interrompu à l'epoch 3 a bien un dncnn_last.pt, mais l'étape n'est
    pas terminée pour autant.
    """
    cache = os.environ.get("CELEBA_CACHE", "celeba_64_gray.npy")

    def resume_dncnn():
        h = lire_json(os.path.join(DOSSIER_RES, "history_dncnn.json"))
        if not h:
            return ""
        d = h[-1]
        return "%d epochs, PSNR val %.2f dB (meilleur %.2f)" % (
            d["epoch"], d["psnr_val"], max(e["psnr_val"] for e in h))

    def resume_cov(prefixe):
        def f():
            h = lire_json(os.path.join(DOSSIER_RES,
                                       "history_%s.json" % prefixe))
            if not h:
                return ""
            d = h[-1]
            meilleur = min(e["nll_val_pixel"] for e in h)
            return "%d epochs, NLL val %+.4f nat/pixel (meilleure %+.4f)" % (
                d["epoch"], d["nll_val_pixel"], meilleur)
        return f

    def resume_eval():
        d = lire_json(os.path.join(DOSSIER_RES, "eval_cov.json"))
        if not d:
            return ""
        bout = "%d images" % d["n_images"]
        for nom, r in d["modeles"].items():
            bout += ", %s %+.4f" % (nom, r["nll_pixel_moyenne"])
        if "gain_structure_nat_par_pixel" in d:
            bout += "  ->  structure %+.4f nat/pixel" % d[
                "gain_structure_nat_par_pixel"]
        return bout

    def resume_denoise():
        d = lire_json(os.path.join(DOSSIER_RES, "denoise.json"))
        if not d:
            return ""
        bout = "DnCNN seul %.3e" % d["dncnn_seul"]["mse_01"]
        for nom, r in d["modeles"].items():
            bout += ", %s %.3e" % (nom, r["mse_01"])
        return bout + "  (MSE en unités [0, 1])"

    return [
        ("données : cache 64x64 gris", cache,
         "python data.py --build", lambda: taille(cache)),
        ("DnCNN entraîné", os.path.join(DOSSIER_CKPT, "dncnn_best.pt"),
         "sbatch train_dncnn.bash", resume_dncnn),
        ("vérification critique du résidu",
         os.path.join(DOSSIER_RES, "residu.png"),
         "python verifier_residu.py", lambda: "figure présente"),
        ("test de surapprentissage", None,
         "sbatch surapprentissage.bash", None),
        ("covariance structurée entraînée",
         os.path.join(DOSSIER_CKPT, "cov_best.pt"),
         "sbatch train_cov.bash", resume_cov("cov")),
        ("référence diagonale entraînée",
         os.path.join(DOSSIER_CKPT, "covdiag_best.pt"),
         "sbatch train_cov.bash --diagonale", resume_cov("covdiag")),
        ("évaluation : NLL, calibration, figures",
         os.path.join(DOSSIER_RES, "eval_cov.json"),
         "python eval_cov.py", resume_eval),
        ("débruitage : projection du résidu",
         os.path.join(DOSSIER_RES, "denoise.json"),
         "python denoise.py", resume_denoise),
    ]


def tableau_de_bord():
    """Affiche l'état de chaque étape et la prochaine commande à lancer."""
    print("=" * 78)
    print("CELEBA — STRUCTURED UNCERTAINTY : ÉTAT DU PROJET")
    print("=" * 78)

    suivante = None
    for nom, temoin, commande, resume in etapes():
        if temoin is None:
            # Étape qui ne produit aucun fichier : on ne peut rien affirmer.
            print("[?] %-38s %s" % (nom, "(ne laisse pas de trace)"))
            continue
        fait = os.path.exists(temoin)
        detail = resume() if (fait and resume) else ""
        print("[%s] %-38s %s" % ("x" if fait else " ", nom, detail))
        if not fait and suivante is None:
            suivante = (nom, commande)

    print("=" * 78)
    if suivante is None:
        print("Toutes les étapes sont franchies.")
        print("  Les chiffres à reprendre dans tuteur.txt sont dans")
        print("  results/eval_cov.json et results/denoise.json.")
    else:
        print("PROCHAINE ACTION : %s" % suivante[0])
        print("    %s" % suivante[1])
    print()
    print("Rappel : les entraînements passent par sbatch (partition short,")
    print("coupée à 1 h 55, --resume est déjà dans les scripts). Ce fichier ne")
    print("les lance pas.")


def evaluation():
    """
    Enchaîne les deux étapes courtes : eval_cov.py puis denoise.py.

    Elles tiennent en quelques minutes sur GPU et n'ont pas besoin de sbatch,
    mais elles vont ensemble : la seconde n'a de sens que si la première a
    confirmé que le modèle vaut quelque chose. On s'arrête au premier échec.
    """
    for script in ("eval_cov.py", "denoise.py"):
        print()
        print(">>> %s" % script)
        code = subprocess.call([sys.executable, "-u", script])
        if code != 0:
            print("%s a échoué (code %d). On s'arrête là." % (script, code))
            return code
    return 0


def main():
    p = argparse.ArgumentParser(description="Orchestration du projet CelebA.")
    p.add_argument("--evaluation", action="store_true",
                   help="lance eval_cov.py puis denoise.py.")
    args = p.parse_args()

    if args.evaluation:
        raise SystemExit(evaluation())
    tableau_de_bord()


if __name__ == "__main__":
    main()
