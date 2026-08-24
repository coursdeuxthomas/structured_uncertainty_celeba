# Structured Uncertainty — Débruitage CelebA

Application de **Structured Uncertainty Prediction Networks** (Dorta et al.,
CVPR 2018, arXiv:1802.07079) au **débruitage d'images**, avec un DnCNN comme
modèle de base.

Troisième volet d'une série de trois projets qui partagent la même architecture
de fichiers et les mêmes conventions :

- splines (1D) : https://github.com/coursdeuxthomas/structured_uncertainty
- ellipses (2D synthétique) : `../structured_uncertainty_ellipse_local`
- **celebA (2D réel, débruitage) : ce dépôt**

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

La covariance vraie n'existe pas sur données réelles : on entraîne uniquement
par maximum de vraisemblance sur les résidus observés.

## Le modèle mathématique

Identique aux deux projets frères.

- Résidu modélisé par une gaussienne pleine : `r ~ N(0, Sigma)`.
- Le réseau prédit la **précision** `Lambda = Sigma^{-1}` via sa Cholesky
  `Lambda = L L^T`, `L` triangulaire inférieure.
- Positivité : le réseau sort `log(l_ii)`, puis `exp`.
- NLL à minimiser, sans jamais inverser `Sigma` :

```
loss = -2 * sum_i log(l_ii) + || L^T r ||^2   (+ n*log(2*pi))
```

**Parcimonie.** `n = 64*64 = 4096` pixels. Une Cholesky dense demanderait
`n(n+1)/2 = 8 390 656` valeurs par image. On impose le motif de l'article :
`l_ij` non nul seulement si `i >= j` ET les pixels `i, j` voisins dans un patch
`f x f` avec **`f = 7`**. Il reste `(f²-1)/2 = 24` valeurs hors-diagonale plus
la diagonale, soit **25 valeurs par pixel = 102 400 par image** (facteur 82).

## Les données

- CelebA « aligned & cropped », recadrage centré puis redimensionnement en
  **64×64**, converti en **niveaux de gris**, normalisé dans `[-1, 1]`.
- Split de l'article : **182 637 train / 19 962 test**, c'est-à-dire
  `train + valid` officiels pour l'entraînement, `test` officiel pour le test.
- Bruit : `y = x + sigma * N(0, I)` avec **`sigma = 25/255`** (standard DnCNN),
  tiré **à la volée** à chaque accès (augmentation gratuite, évite que le réseau
  mémorise le résidu d'une image).
- Les images prétraitées sont mises en cache en `uint8` (≈ 750 Mo, tient en RAM).
  Ne jamais relire les JPEG à chaque epoch.

## Architecture des fichiers

- `data.py` — prétraitement, cache, `CelebADataset` (renvoie `x` et `y`).
- `dncnn.py` — le débruiteur (17 couches Conv-BN-ReLU, apprentissage résiduel).
- `train_dncnn.py` — entraînement du débruiteur (étape 1).
- `loss.py` — **repris tel quel du projet ellipses**, avec `f = 7`.
- `cov_model.py` — réseau de covariance : `mu [B,1,64,64]` → `log_diag [B,4096]`
  et `offdiag [B,4096,24]`.
- `train_cov.py` — entraînement du réseau de covariance, DnCNN **gelé**.
- `eval_cov.py` — NLL, référence diagonale, calibration, figures.
- `denoise.py` — l'application §5.3 (projection du résidu).
- `main.py` — orchestration.
- `results/` — sorties horodatées (ignoré par git).

## Entraînement en deux temps

Le DnCNN est entraîné **puis gelé**. Le réseau de covariance est entraîné
séparément, comme dans l'article (Eq. 4 : « keeping the generative model
parameters theta fixed ») :

```python
y  = x + sigma * torch.randn_like(x)
with torch.no_grad():
    mu = dncnn(y)
r = x - mu
log_diag, offdiag = cov_net(mu)
loss = structured_gaussian_nll(log_diag, offdiag, r, neighbor_idx, mask)
```

Hyperparamètres article : lr `1e-3`, 50 epochs, batch 64, Adam.

## Conventions

- Commentaires et docstrings en **français**, comme les deux projets frères.
- `L` encode la **précision**, PAS la covariance. Ne pas confondre avec la
  Cholesky de la covariance utilisée pour échantillonner du bruit.
- Le réseau de covariance prend `mu` **aplati** `[B, n]` ou en image
  `[B, 1, 64, 64]` ; `r` reste un vecteur `[B, n]` pour la loss.
- **Ne jamais donner `x` (image propre) au réseau de covariance.** Il pourrait
  reconstruire `r` directement et la NLL s'effondrerait sans rien apprendre.
- Ordre raster partout : indice `i = ligne * 64 + colonne`.

## Contrainte technique majeure : aucune matrice dense

À `n = 4096`, une seule matrice `n×n` en float32 pèse **67 Mo**. Un batch de 64
en pèserait 4.3 Go.

**L'évaluation doit donc être écrite sans jamais former `Sigma`.**
`build_L_dense` et `predicted_precision_and_covariance` (hérités du projet
ellipses) ne servent qu'aux tests unitaires en 16×16.

Ce qui reste possible et suffit :

- NLL via `apply_LT` — `O(n·m)` ;
- calibration via `w = L^T r` — `O(n·m)` ;
- échantillonnage en résolvant `L^T eps = u` (substitution arrière creuse).

## Évaluation (sans vérité terrain)

`Sigma_true` n'existe pas : ni Frobenius ni KL. Quatre choses à la place :

1. **NLL** sur le jeu de test.
2. **Référence diagonale** — même réseau avec `offdiag = 0`, réentraîné.
   L'écart de NLL chiffre ce que la structure apporte. *Sans cette comparaison,
   les résultats ne démontrent rien.*
3. **Calibration** — `w = L^T r` doit suivre `N(0, I)` : moyenne ~0, variance ~1
   par pixel, autocorrélation spatiale nulle. Seul diagnostic quantitatif
   valable sur données réelles.
4. **Figures** — `x` / `y` / `mu` / `mu + eps` diagonal / `mu + eps` structuré.

Pour le débruitage : MSE contre `x` propre, comparée à DnCNN seul.

## Environnement / commandes

- Python via le venv local ; dépendances : `numpy`, `torch`, `matplotlib`,
  `pillow`.
- Cluster SLURM (partition `short`, limite 1h55, 1 GPU) : voir `train.bash`.
  **Prévoir la reprise sur checkpoint** (sauvegarde de l'optimiseur et de
  l'epoch) dès le début, sinon des runs seront perdus.
- Tester un module isolément : `python loss.py`, `python cov_model.py`.

## État d'avancement

- [ ] `data.py` — téléchargement, prétraitement 64×64 gris, cache.
- [ ] `dncnn.py` + `train_dncnn.py` — le débruiteur.
- [ ] **VÉRIFICATION CRITIQUE** — afficher `r = x - DnCNN(y)`.
      Si le résidu ressemble à du bruit blanc, **le projet n'a pas d'objet**.
      Il doit contenir des structures visibles (contours, cheveux, texture).
      Vérifier aussi que son autocorrélation spatiale s'étale sur plusieurs
      pixels. **Ne pas coder la suite avant d'avoir passé ce test.**
- [ ] `loss.py` — reprise depuis le projet ellipses, `f = 7`, tests relancés.
      Ajouter le clamp manquant : `log_diag.clamp(-10, 10)`.
- [ ] `cov_model.py` — U-Net 3 niveaux, tête 25 canaux.
- [ ] `train_cov.py` — entraînement 2 temps.
- [ ] `eval_cov.py` — NLL, référence diagonale, calibration, figures.
- [ ] `denoise.py` — projection et MSE finale.

## Références

- `article.pdf` — Dorta et al., 2018. §5.2 (CelebA), §5.3 (débruitage),
  annexe D (figures).
- `roadmap_celeba_dncnn.txt` — plan détaillé phase par phase.
- `../structured_uncertainty_ellipse_local/` — le projet ellipses, dont
  `loss.py` est réutilisé tel quel et dont les fichiers `explication_*.txt`
  documentent la partie mathématique.
