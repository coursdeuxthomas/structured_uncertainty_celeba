# Structured Uncertainty — Débruitage CelebA

Application de **Structured Uncertainty Prediction Networks** (Dorta et al.,
CVPR 2018, [arXiv:1802.07079](https://arxiv.org/abs/1802.07079)) au
**débruitage d'images**, avec un DnCNN comme modèle de base.

Troisième volet d'une série de trois projets partageant la même architecture
et les mêmes conventions :

| Projet | Données | Dépôt |
|---|---|---|
| splines | 1D | [structured_uncertainty_spline](https://github.com/coursdeuxthomas/structured_uncertainty_spline) |
| ellipses | 2D synthétique | [structured_uncertainty_ellispses](https://github.com/coursdeuxthomas/structured_uncertainty_ellipse) |
| **celebA** | **2D réel, débruitage** | **ce dépôt** |

## Objectif

Un DnCNN débruite une image : il enlève le bruit, mais aussi le **détail
haute-fréquence** (cheveux, texture, contours fins). On entraîne un second
réseau qui apprend la **covariance structurée du résidu** `r = x - mu`, où
`mu = DnCNN(y)` est l'image débruitée et `x` l'image propre.

Cette covariance décrit à quoi ressemble le détail perdu. On peut alors :

1. **échantillonner** un résidu plausible et le rajouter → image nette au lieu
   de lisse ;
2. **projeter** ce que le DnCNN a retiré sur l'espace des résidus plausibles →
   récupérer le détail sans récupérer le bruit (§5.3 de l'article).

La covariance vraie n'existe pas sur données réelles : l'entraînement se fait
uniquement par maximum de vraisemblance sur les résidus observés.

## Le modèle

- Résidu modélisé par une gaussienne pleine : `r ~ N(0, Sigma)`.
- Le réseau prédit la **précision** `Lambda = Sigma^-1` via sa Cholesky
  `Lambda = L Lᵀ`, `L` triangulaire inférieure.
- Positivité : le réseau sort `log(l_ii)`, puis `exp`.
- NLL minimisée sans jamais inverser `Sigma` :

```
loss = -2 * sum_i log(l_ii) + || Lᵀ r ||²   (+ n·log(2π))
```

**Parcimonie.** Avec `n = 64×64 = 4096` pixels, une Cholesky dense demanderait
8 390 656 valeurs par image. On impose le motif de l'article : `l_ij` non nul
seulement si `i >= j` et si les pixels `i, j` sont voisins dans un patch `f×f`
avec `f = 7`. Il reste 24 valeurs hors-diagonale plus la diagonale, soit
**25 valeurs par pixel = 102 400 par image** (facteur 82).

> À `n = 4096`, une matrice `n×n` en float32 pèse 67 Mo : **aucune matrice
> dense n'est jamais formée**, ni à l'entraînement ni à l'évaluation.

## Données

- CelebA « aligned & cropped », recadrage centré, redimensionnement en
  **64×64**, niveaux de gris, normalisé dans `[-1, 1]`.
- Split de l'article : **182 637 train / 19 962 test** (`train + valid`
  officiels pour l'entraînement, `test` officiel pour le test).
- Bruit : `y = x + sigma · N(0, I)` avec `sigma = 25/255`, tiré **à la volée**
  à chaque accès.
- Images prétraitées mises en cache en `uint8` (≈ 750 Mo, tient en RAM).

Le dataset n'est pas versionné. Pour le récupérer :

```bash
python download.py   # renseigner ses identifiants Kaggle au préalable
```

## Structure

| Fichier | Rôle |
|---|---|
| `data.py` | prétraitement, cache, `CelebADataset` (renvoie `x` et `y`) |
| `dncnn.py` | le débruiteur (17 couches Conv-BN-ReLU, apprentissage résiduel) |
| `train_dncnn.py` | entraînement du débruiteur (étape 1) |
| `loss.py` | NLL gaussienne structurée, `f = 7` |
| `cov_model.py` | réseau de covariance : `mu` → `log_diag` et `offdiag` |
| `train_cov.py` | entraînement du réseau de covariance, DnCNN gelé |
| `eval_cov.py` | NLL, référence diagonale, calibration, figures |
| `denoise.py` | l'application §5.3 (projection du résidu) |
| `main.py` | orchestration |

Diagnostics et outillage :

| Fichier | Rôle |
|---|---|
| `verifier_alignement.py` | preuve empirique de l'alignement des visages |
| `verifier_residu.py` | **la vérification critique** : structure du résidu |
| `build_data.bash` | job SLURM de construction du cache |
| `train_dncnn.bash` | job SLURM d'entraînement du débruiteur |
| `PROCEDURE.md` | récupérer les images et construire le cache |
| `explication_data.txt` | commentaire détaillé de `data.py` |
| `explication_dncnn.txt` | commentaire détaillé de l'étape 1 |
| `tuteur.txt` | notes de présentation : modèle, chiffres, décisions |

## Entraînement en deux temps

Le DnCNN est entraîné **puis gelé**. Le réseau de covariance est entraîné
séparément, comme dans l'article (Eq. 4) :

```python
y = x + sigma * torch.randn_like(x)
with torch.no_grad():
    mu = dncnn(y)
r = x - mu
log_diag, offdiag = cov_net(mu)
loss = structured_gaussian_nll(log_diag, offdiag, r, neighbor_idx, mask)
```

Hyperparamètres de l'article : lr `1e-3`, 50 epochs, batch 64, Adam.

## Évaluation

`Sigma_true` n'existe pas : ni Frobenius ni KL. À la place :

1. **NLL** sur le jeu de test ;
2. **référence diagonale** — même réseau avec `offdiag = 0`, réentraîné :
   l'écart de NLL chiffre ce que la structure apporte ;
3. **calibration** — `w = Lᵀ r` doit suivre `N(0, I)` ;
4. **figures** — `x` / `y` / `mu` / `mu + eps` diagonal / `mu + eps` structuré.

Pour le débruitage : MSE contre `x` propre, comparée à DnCNN seul.

## Installation

```bash
python -m venv venv
source venv/bin/activate      # Windows : venv\Scripts\activate
pip install numpy torch matplotlib pillow kaggle
```

## Sur le cluster

```bash
git clone https://github.com/coursdeuxthomas/structured_uncertainty_celeba.git
cd structured_uncertainty_celeba
```

Le dépôt est public : pas d'authentification pour cloner. Il ne contient que le
code — les images, le cache `.npy`, les checkpoints et les logs restent sur le
cluster et sont ignorés par git.

**Les données.** `data.py` lit `img_align_celeba/` et écrit
`celeba_64_gray.npy` dans le dossier courant. Les deux chemins se surchargent
par variable d'environnement, ce qui évite de remplir le home :

```bash
export CELEBA_DIR=$SCRATCH/celeba/img_align_celeba
export CELEBA_CACHE=$SCRATCH/celeba/celeba_64_gray.npy
python data.py --build         # une seule fois, 10 à 20 min
```

Le dataset doit compter **202 599 images**. `data.py` découpe train/test par
indice : avec un téléchargement incomplet, le split serait faux sans aucun
message d'erreur.

**Pousser depuis le cluster.** Une fois par machine, créer une clé SSH et la
déclarer sur GitHub (Settings > SSH and GPG keys) :

```bash
git config --global user.name  "Thomas"
git config --global user.email "ts.bouru@gmail.com"

ssh-keygen -t ed25519 -C "ts.bouru@gmail.com"   # Entrée trois fois
cat ~/.ssh/id_ed25519.pub                       # à coller sur GitHub

ssh -T git@github.com                           # doit répondre "Hi coursdeuxthomas!"
git remote set-url origin git@github.com:coursdeuxthomas/structured_uncertainty_celeba.git
```

Ensuite, dans les deux sens :

```bash
git add -A && git commit -m "..." && git push    # cluster  -> GitHub
git pull                                         # GitHub   -> poste local
```

Toujours faire `git pull` avant de reprendre le travail sur l'autre machine :
c'est le mélange de deux copies modifiées sans `pull` qui produit les conflits.

## État d'avancement

- [ ] `data.py` — téléchargement, prétraitement 64×64 gris, cache
- [ ] `dncnn.py` + `train_dncnn.py` — le débruiteur
- [ ] **vérification critique** — le résidu `r = x - DnCNN(y)` doit contenir des
      structures visibles, pas du bruit blanc
- [ ] `loss.py` — `f = 7`, clamp `log_diag.clamp(-10, 10)`
- [ ] `cov_model.py` — U-Net 3 niveaux, tête 25 canaux
- [ ] `train_cov.py` — entraînement en deux temps
- [ ] `eval_cov.py` — NLL, référence diagonale, calibration, figures
- [ ] `denoise.py` — projection et MSE finale

## Référence

Dorta, C., Vicente, S., Agapito, L., Campbell, N. D. F., Simpson, I.
*Structured Uncertainty Prediction Networks*. CVPR 2018.
