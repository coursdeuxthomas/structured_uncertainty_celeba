# Récupérer CelebA sur le cluster et construire les données

Tout se fait sur le cluster, dans l'environnement conda `dncnn`. Compter
environ 30 minutes en tout, dont 20 d'attente.

Ce qu'on veut à la fin : `202 599` JPEG sur le disque, et un fichier
`celeba_64_gray.npy` de 830 Mo que `data.py` sait charger en RAM.

---

## 0. Vérifier la place disponible

Il faut ~2,3 Go : 1,4 Go de JPEG plus 830 Mo de cache.

```bash
quota -s          # si la commande existe
df -h $HOME
du -sh $HOME
```

Si le home est trop juste, remplace `$HOME/data` par un chemin de scratch dans
tout ce qui suit — rien d'autre ne change, les chemins sont des variables.

---

## 1. Le token Kaggle

CelebA n'a pas d'URL de téléchargement direct fiable : le lien Google Drive
officiel est régulièrement bloqué pour cause de trafic. Kaggle est le moyen
propre, mais il demande un compte et un token.

**Attention, Kaggle a changé de système.** Il n'y a plus de `kaggle.json` qui
se télécharge tout seul : on obtient maintenant un jeton texte de la forme
`KGAT_...`. L'ancien fichier existe encore derrière un bouton
« Create Legacy API Key », mais autant utiliser le nouveau.

**Sur kaggle.com** : avatar en haut à droite → **Settings** → section **API**
→ créer un token. La chaîne `KGAT_...` s'affiche **une seule fois** : copie-la
immédiatement.

**Sur le cluster**, écris-la dans un fichier :

```bash
mkdir -p ~/.kaggle
nano ~/.kaggle/access_token      # colle la chaîne KGAT_..., rien d'autre
chmod 600 ~/.kaggle/access_token
```

Passe par `nano` plutôt que par `echo KGAT_... > fichier` : tout ce que tu
tapes dans le terminal atterrit dans `~/.bash_history`, lisible par
l'administrateur du cluster et conservé indéfiniment. Même remarque pour
`export KAGGLE_API_TOKEN=KGAT_...` — ça marche, mais ça laisse le jeton en
clair dans l'historique.

**Le CLI doit être récent.** Le format `KGAT_` n'est compris qu'à partir de
la version 2.x du paquet `kaggle` (2.2.4 en juillet 2026). Les versions 1.7 et
antérieures ne connaissent que `kaggle.json` et échouent avec
`Could not find kaggle.json`.

```bash
pip install --upgrade kaggle
pip show kaggle | head -2         # doit afficher 2.x
```

Si pip reste bloqué en 1.7.x, c'est que la version de Python de l'env est trop
ancienne pour le paquet 2.x. Deux issues : créer un env conda avec un Python
plus récent, ou — plus simple — utiliser la clé héritée. Sur kaggle.com,
Settings → API → **Create Legacy API Key** : un `kaggle.json` se télécharge,
avec `{"username":"...","key":"..."}`. Colle-le dans `~/.kaggle/kaggle.json`,
`chmod 600`, et le CLI ancien s'authentifie sans rien changer d'autre.

Vérifie que l'authentification passe :

```bash
kaggle competitions list
```

Une liste de compétitions s'affiche : c'est bon. Une erreur 401 : le fichier
contient autre chose que la chaîne, ou un espace de trop.

**Si un jeton a fuité** (collé dans un chat, poussé sur GitHub, tapé dans une
commande partagée) : Settings → API → révoque-le et crée-en un autre. Un jeton
Kaggle donne un accès complet au compte, pas seulement en lecture.

---

## 2. Télécharger

```bash
conda activate dncnn
pip install kaggle

mkdir -p ~/data/celeba
kaggle datasets download -d jessicali9530/celeba-dataset -p ~/data/celeba --unzip
```

1,4 Go, quelques minutes. C'est le seul moment où l'on travaille sur le nœud
de connexion : c'est du réseau, pas du calcul, personne ne t'en voudra.

---

## 3. Aplatir l'arborescence

L'archive Kaggle emboîte le dossier deux fois : `img_align_celeba/img_align_celeba/*.jpg`.

```bash
cd ~/data/celeba
ls                                    # pour voir ce que tu as vraiment

mv img_align_celeba img_tmp
mv img_tmp/img_align_celeba .
rm -rf img_tmp
```

---

## 4. Vérifier — l'étape à ne pas sauter

```bash
find ~/data/celeba/img_align_celeba -name '*.jpg' | wc -l
```

**Doit afficher exactement `202599`.** Et les bornes :

```bash
ls ~/data/celeba/img_align_celeba | head -1     # 000001.jpg
ls ~/data/celeba/img_align_celeba | tail -1     # 202599.jpg
```

Pourquoi c'est critique : `data.py` ne lit aucun fichier de partition, il
découpe par indice.

```python
self.images = images[:182637]     # train
self.images = images[182637:]     # test
```

Avec 150 000 images au lieu de 202 599, le code tourne, n'affiche aucune
erreur, et entraîne sur un découpage qui n'a rien à voir avec le split de
l'article. C'est exactement ce qui est arrivé à la copie locale, coupée à
83 470 images.

---

## 5. Construire le cache

Le décodage de 202 599 JPEG prend 10 à 20 minutes de CPU. Ça ne se fait pas
sur le nœud de connexion : on passe par un job.

Copie `build_data.bash` dans ton dépôt cloné (le script fait un
`cd ~/structured_uncertainty_celeba` : adapte la ligne si tu as cloné
ailleurs), puis :

```bash
cd ~/structured_uncertainty_celeba
mkdir -p logs
sbatch build_data.bash

squeue -u $USER
tail -f logs/celeba_data_<jobid>.out
```

Le job ne demande **pas** de GPU : c'est du décodage d'images, et un job sans
GPU est servi bien plus vite par la file.

---

## 6. Lire le résultat

À la fin du log tu dois voir :

```
JPEG trouvés : 202599 (attendu 202599)
Cache écrit : /home/tbouru/data/celeba/celeba_64_gray.npy  (830 Mo)
train : 182637 exemples | test : 19962 exemples
attendu article : 182637 / 19962
x : (1, 64, 64)  dans [-1.00, 1.00]
sigma effectif (unités [-1,1]) : 0.1961
bruit mesuré (y - x) std       : 0.196x
bruit retiré à chaque accès    : True
```

Trois lignes à lire vraiment :

- **`182637 / 19962`** — le split de l'article est respecté.
- **`sigma effectif 0.1961`** — c'est `2 x 25/255`. Le facteur 2 est normal :
  les images sont dans `[-1, 1]`, pas dans `[0, 1]`, donc l'écart-type du
  bruit y est deux fois plus grand.
- **`bruit retiré à chaque accès : True`** — le bruit est bien retiré à
  chaque `__getitem__`. Si c'était `False`, le réseau de covariance pourrait
  mémoriser le résidu au lieu d'en apprendre la structure.

Et une figure d'aperçu dans `results/data_preview.png` : six visages propres
en haut, les mêmes bruités en dessous. Regarde-la — c'est le seul contrôle
qui attrape un recadrage raté ou une inversion de niveaux de gris.

Pour la rapatrier sur Windows :

```bash
scp tbouru@<cluster>:~/structured_uncertainty_celeba/results/data_preview.png .
```

---

## 7. Ensuite

Le cache est construit une fois pour toutes. Dans tous les scripts suivants,
il suffit d'exporter les deux variables avant de lancer :

```bash
export CELEBA_DIR=$HOME/data/celeba/img_align_celeba
export CELEBA_CACHE=$HOME/data/celeba/celeba_64_gray.npy
```

Prochaine étape du projet : `dncnn.py` puis `train_dncnn.py`, et surtout la
vérification critique de la roadmap — afficher `r = x - DnCNN(y)` et
s'assurer qu'il contient des structures visibles et pas du bruit blanc. Si le
résidu est blanc, le projet n'a pas d'objet.

## Git : aller et venir entre le poste et le cluster

Ce bloc vivait dans le README ; il en a ete retire quand celui-ci est passe
en anglais et s'est recentre sur le projet lui-meme.

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
c'est le melange de deux copies modifiees sans `pull` qui produit les conflits.
