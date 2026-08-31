"""
Orchestration: where the project stands, and what to run next.

    python main.py                # dashboard
    python main.py --evaluation   # runs eval_cov.py then denoise.py

The project unfolds in eight steps, three of which are long trainings on GPU.
This script does NOT launch them: a five-hour training goes through sbatch,
not through a subprocess that would die with the SSH session. It reads what
has already been produced, displays the numbers obtained, and gives the next
command to type.

It deliberately imports neither torch nor numpy: everything is read from the
JSON files written at each epoch. The dashboard therefore works everywhere,
including on the login node with no conda environment activated.
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
    """Returns the contents of the JSON, or None if missing or unreadable."""
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
# The eight steps, in the order in which they must be cleared
# --------------------------------------------------------------------------
def etapes():
    """
    Each step: (name, witness file, command, summary function).

    The "witness file" is what proves the step has been cleared. For the
    trainings it is the checkpoint of the best model, not the last one: a run
    interrupted at epoch 3 does have a dncnn_last.pt, but the step is not
    finished for all that.
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
    """Displays the state of each step and the next command to launch."""
    print("=" * 78)
    print("CELEBA — STRUCTURED UNCERTAINTY : ÉTAT DU PROJET")
    print("=" * 78)

    suivante = None
    for nom, temoin, commande, resume in etapes():
        if temoin is None:
            # Step that produces no file: nothing can be asserted about it.
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
        print("  Les chiffres à reprendre dans docs/tuteur.txt sont dans")
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
    Chains the two short steps: eval_cov.py then denoise.py.

    They take a few minutes on GPU and do not need sbatch, but they belong
    together: the second only makes sense if the first has confirmed that the
    model is worth something. We stop at the first failure.
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
