# Structured Uncertainty — Débruitage CelebA

Application de **Structured Uncertainty Prediction Networks** (Dorta et al.,
CVPR 2018, [arXiv:1802.07079](https://arxiv.org/abs/1802.07079)) au
**débruitage d'images**, avec un DnCNN comme modèle de base.

Troisième volet d'une série de trois projets partageant la même architecture
et les mêmes conventions :

| Projet | Données | Dépôt |
|---|---|---|
| splines | 1D | [structured_uncertainty](https://github.com/coursdeuxthomas/structured_uncertainty) |
| ellipses | 2D synthétique | *(local)* |
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
| `setup_cluster.sh` | installation de l'environnement sur le cluster |
| `get_data.sh` | téléchargement Kaggle, vérification, construction du cache |
| `train.bash` | job SLURM (partition `short`, 1 GPU, 1h55, reprise) |

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
# 1. Cloner (dépôt public : aucune authentification nécessaire)
git clone https://github.com/coursdeuxthomas/structured_uncertainty_celeba.git
cd structured_uncertainty_celeba

# 2. Environnement Python
bash setup_cluster.sh          # crée ./venv et installe torch + dépendances
source venv/bin/activate

# 3. Données : sur le scratch, JAMAIS dans le home
export CELEBA_DIR=$SCRATCH/celeba/img_align_celeba
export CELEBA_CACHE=$SCRATCH/celeba/celeba_64_gray.npy
bash get_data.sh               # Kaggle + vérification + cache 64x64 gris

# 4. Lancer un entraînement
sbatch train.bash dncnn
squeue -u $USER
tail -f logs/celeba-<jobid>.out
```

**Trois points d'attention.**

*Les modules.* `setup_cluster.sh` et `train.bash` contiennent des lignes
`module load` commentées : remplace-les par les noms réels de ton cluster
(`module avail python cuda`) avant de lancer quoi que ce soit.

*Le scratch.* Les JPEG pèsent 1,4 Go et le cache 830 Mo. Le quota du home est
en général trop petit — d'où `CELEBA_DIR` et `CELEBA_CACHE`. La variable
`$SCRATCH` n'existe pas partout : elle peut s'appeler `$WORK`, `$SCRATCHDIR`,
ou correspondre à un chemin en dur.

*La limite de 1h55.* Un entraînement de 50 epochs ne tient pas dans un job de
la partition `short`. Les scripts d'entraînement **doivent** sauvegarder
l'optimiseur et le numéro d'epoch dès la première version, et accepter
`--resume`. `train.bash` sait alors se resoumettre tout seul :

```bash
sbatch --export=ALL,AUTO_RELANCE=1 train.bash dncnn
```

Il s'arrête quand le script python crée `checkpoints/dncnn_FINI`, ou au bout de
20 relances.

**Aller-retour code.** Développe en local, pousse, récupère sur le cluster :

```bash
# poste local
git add -A && git commit -m "..." && git push

# cluster
git pull
```

Ne modifie pas le code directement sur le cluster : tu perdrais l'historique et
les deux copies divergeraient. Ce qui reste côté cluster, ce sont les données,
les checkpoints et les logs — tous ignorés par git.

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
