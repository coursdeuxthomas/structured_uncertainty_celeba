"""
Loss NLL gaussienne structurée pour CelebA (Cholesky CREUSE de la précision).

Reprise du projet ellipses, avec deux changements de constantes et un garde-fou
numérique. Les deux fonctions centrales (`apply_LT` et `structured_gaussian_nll`)
sont indépendantes de la taille d'image et du voisinage : elles passent de
16x16 à 64x64 sans être touchées.

    ellipses : S = 16, n =  256, f = 5, m = 12  ->  13 valeurs par pixel
    celebA   : S = 64, n = 4096, f = 7, m = 24  ->  25 valeurs par pixel

Motif de parcimonie de l'article (Dorta et al., 2018, §5.1) :

    L[i, j] != 0  seulement si  i >= j  ET  les pixels i, j sont voisins
    dans un patch f x f (ici f = 7).

Pour chaque pixel `i`, `L` n'a donc de valeurs non nulles que sur :
- la diagonale `L[i, i]`, paramétrée par `log(l_ii)` (l'`exp` garantit > 0) ;
- les `L[i, j]` pour `j` voisin CAUSAL de `i` (j vient avant i en ordre raster
  ET dans le patch f x f). Pour f = 7 il y a 24 voisins causaux.

La précision est `Lambda = L L^T` (symétrique définie positive par
construction). La NLL gaussienne (Eq. 4 de l'article) se calcule SANS inverser
Sigma :

    nll = 0.5 * [ log|Sigma| + r^T Lambda r + n*log(2*pi) ]
        = 0.5 * [ -2*sum_i log(l_ii) + || L^T r ||^2 + n*log(2*pi) ]

avec ici `r = x - mu`, le résidu du DnCNN (et non `x - mu` d'une spline).

`L^T r` se calcule directement à partir des valeurs non nulles, via un
`scatter_add` : aucune matrice n x n n'est jamais formée. C'est la condition de
survie du projet — à n = 4096, une seule matrice dense float32 pèse 67 Mo, et
un batch de 64 en pèserait 4.3 Go.

CHANGEMENT PAR RAPPORT AUX ELLIPSES : `log_diag` est désormais borné à
[-10, +10] avant l'`exp`. À 4096 pixels et sur données réelles, une valeur de
diagonale qui part à +inf fait exploser la NLL en une itération ; en 16x16 sur
données synthétiques on pouvait s'en passer, plus ici.

Usage :
    python loss.py          # tests unitaires 16x16 + tests de forme 64x64
"""

import math

import numpy as np
import torch


# Constantes du projet (cf. CLAUDE.md, phase 0 de la roadmap).
IMAGE_SIZE = 64      # images 64x64 -> n = 4096 pixels
VOISINAGE = 7        # patch f x f de l'article -> m = 24 voisins causaux
CLAMP_LOG_DIAG = 10.0

# Au-delà de cette taille, on refuse de construire une matrice dense : à
# n = 4096 elle pèse 67 Mo par exemple. Les fonctions denses ne servent qu'aux
# tests unitaires sur images jouets.
N_MAX_DENSE = 1024


def clamp_log_diag(log_diag, limite=CLAMP_LOG_DIAG):
    """
    Borne `log(l_ii)` dans [-limite, +limite] avant tout `exp`.

    En dehors de l'intervalle le gradient est nul : c'est exactement l'effet
    voulu, une diagonale partie trop loin cesse d'être poussée plus loin. Les
    bornes correspondent à `l_ii` entre `exp(-10) = 4.5e-5` et
    `exp(10) = 2.2e4`, ce qui couvre très largement les échelles utiles pour un
    résidu dans [-2, 2].
    """
    return torch.clamp(log_diag, min=-limite, max=limite)


# ---------------------------------------------------------------------------
# Motif de parcimonie : voisins causaux dans un patch f x f
# ---------------------------------------------------------------------------
def causal_offsets(f=VOISINAGE):
    """
    Décalages (dr, dc) des voisins CAUSAUX (hors pixel courant) d'un patch f x f.

    Un voisin `j = i + (dr, dc)` est causal si son indice raster est < celui de
    `i`, c.-à-d. `dr < 0`, ou (`dr == 0` et `dc < 0`). En ordre raster
    (index = row * S + col), cela garantit que `L` est bien triangulaire
    inférieure.

    Pour f = 7 (h = 3) il y a 24 décalages :
    - dr in {-3, -2, -1}, dc in {-3, ..., 3}  -> 21
    - dr == 0,            dc in {-3, -2, -1}  ->  3

    Retour :
        offsets : list[(dr, dc)] de longueur (f*f - 1) // 2
    """
    h = f // 2
    offsets = []
    for dr in range(-h, 1):
        for dc in range(-h, h + 1):
            if dr < 0 or (dr == 0 and dc < 0):
                offsets.append((dr, dc))
    return offsets


def build_neighbor_indices(image_size=IMAGE_SIZE, f=VOISINAGE, device=None):
    """
    Précalcule, pour chaque pixel `i`, l'indice raster de ses voisins causaux et
    un masque de validité (pixels hors image = invalides).

    Retour :
        neighbor_idx : LongTensor [n, m]  (m = 24 pour f = 7)
                       neighbor_idx[i, k] = indice raster du k-ième voisin causal
                       de i, ou `i` lui-même si le voisin sort de l'image (valeur
                       neutralisée par le masque, la diagonale étant écrasée
                       séparément).
        mask         : FloatTensor [n, m], 1.0 si le voisin est dans l'image.

    Ces tenseurs ne dépendent pas de l'exemple : on les calcule UNE FOIS au début
    de l'entraînement et on les réutilise pour tout le batch et toute
    l'expérience. En 64x64 ils font [4096, 24], soit moins de 500 ko.
    """
    S = image_size
    n = S * S
    offsets = causal_offsets(f)
    m = len(offsets)

    neighbor_idx = np.zeros((n, m), dtype=np.int64)
    mask = np.zeros((n, m), dtype=np.float32)

    for i in range(n):
        r, c = divmod(i, S)
        for k, (dr, dc) in enumerate(offsets):
            rr, cc = r + dr, c + dc
            if 0 <= rr < S and 0 <= cc < S:
                neighbor_idx[i, k] = rr * S + cc
                mask[i, k] = 1.0
            else:
                # Voisin hors image : on le renvoie sur `i` (colonne quelconque)
                # et on le neutralise via le masque (0 -> contribution nulle).
                neighbor_idx[i, k] = i
                mask[i, k] = 0.0

    neighbor_idx = torch.from_numpy(neighbor_idx)
    mask = torch.from_numpy(mask)
    if device is not None:
        neighbor_idx = neighbor_idx.to(device)
        mask = mask.to(device)
    return neighbor_idx, mask


# ---------------------------------------------------------------------------
# Produit L^T r sans matrice dense
# ---------------------------------------------------------------------------
def apply_LT(log_diag, offdiag, residual, neighbor_idx, mask):
    """
    Calcule `w = L^T r` à partir des valeurs non nulles de `L`, sans jamais
    construire la matrice n x n. Coût O(n*m) au lieu de O(n^2).

    Rappel : `L[i, i] = exp(log_diag[i])` et `L[i, neighbor_idx[i, k]] = offdiag[i, k]`.
    On a `w_c = (L^T r)_c = sum_i L[i, c] r_i`, d'où :
    - diagonale : `w_i += exp(log_diag_i) * r_i` ;
    - hors-diag : chaque `L[i, j] * r_i` s'ajoute à `w_j` (scatter_add sur j).

    Cette fonction sert aussi à l'ÉVALUATION : `w = L^T r` doit suivre `N(0, I)`
    si la covariance est bien calibrée (cf. eval_cov.py).

    Args :
        log_diag     : [B, n]      valeurs log de la diagonale de L (bornées ici).
        offdiag      : [B, n, m]   valeurs hors-diagonale (une par voisin causal).
        residual     : [B, n]      r = x - mu.
        neighbor_idx : [n, m]      indices des voisins causaux.
        mask         : [n, m]      masque de validité des voisins.

    Retour :
        w : [B, n],  w = L^T r.
    """
    B, n = residual.shape
    m = offdiag.shape[2]

    # Terme diagonal. Le clamp est appliqué ici aussi pour que la fonction soit
    # sûre quand on l'appelle directement (calibration), pas seulement via la NLL.
    # Il est idempotent : le ré-appliquer ne change rien.
    w = torch.exp(clamp_log_diag(log_diag)) * residual  # [B, n]

    # Termes hors-diagonale : contribution de chaque pixel i à ses voisins j.
    contrib = offdiag * mask.unsqueeze(0) * residual.unsqueeze(2)  # [B, n, m]

    # scatter_add sur la dimension des pixels :
    # w[b, neighbor_idx[i, k]] += contrib[b, i, k].
    idx = neighbor_idx.reshape(1, n * m).expand(B, n * m)
    w = w.scatter_add(1, idx, contrib.reshape(B, n * m))
    return w


# ---------------------------------------------------------------------------
# NLL gaussienne structurée
# ---------------------------------------------------------------------------
def structured_gaussian_nll(
    log_diag,
    offdiag,
    residual,
    neighbor_idx,
    mask,
    include_const=True,
    mean_batch=True,
):
    """
    NLL gaussienne (Eq. 4 de l'article) pour la précision structurée creuse.

        nll = 0.5 * [ -2*sum_i log(l_ii) + ||L^T r||^2 + n*log(2*pi) ]

    avec `log(l_ii) = log_diag_i` (donc `log|Sigma| = -2 sum_i log_diag_i`) et
    `||L^T r||^2 = r^T Lambda r`. Aucune inversion, aucune matrice n x n.

    Les deux termes se lisent l'un contre l'autre : le terme quadratique pousse
    la précision vers 0 (une covariance énorme rend tout résidu probable), le
    log-déterminant l'en empêche (il récompense les grandes diagonales). Leur
    équilibre est ce que le réseau apprend.

    IMPORTANT : `log_diag` est borné AVANT d'être utilisé, et la MÊME valeur
    bornée sert au log-déterminant et à `apply_LT`. Si les deux termes ne
    voyaient pas la même diagonale, la loss ne serait plus une NLL cohérente.

    Args :
        log_diag, offdiag : sorties du réseau de covariance ([B, n] et [B, n, m]).
        residual          : r = x - mu, [B, n]  (mu = DnCNN(y), gelé).
        neighbor_idx, mask: motif de parcimonie (cf. build_neighbor_indices).
        include_const     : ajoute le terme constant n*log(2*pi) (vraie NLL).
                            À laisser True pour comparer des NLL entre modèles.
        mean_batch        : moyenne sur le batch (sinon retour par exemple [B]).

    Retour :
        nll : scalaire (mean_batch=True) ou [B].
    """
    n = residual.shape[1]

    log_diag = clamp_log_diag(log_diag)

    w = apply_LT(log_diag, offdiag, residual, neighbor_idx, mask)  # [B, n]

    quad = (w ** 2).sum(dim=1)                   # ||L^T r||^2, [B]
    log_det_sigma = -2.0 * log_diag.sum(dim=1)   # log|Sigma|, [B]

    nll = log_det_sigma + quad
    if include_const:
        nll = nll + n * math.log(2.0 * math.pi)
    nll = 0.5 * nll

    if mean_batch:
        return nll.mean()
    return nll


# ---------------------------------------------------------------------------
# Reconstruction dense — TESTS UNITAIRES UNIQUEMENT, jamais en 64x64
# ---------------------------------------------------------------------------
def build_L_dense(log_diag, offdiag, neighbor_idx, mask, force=False):
    """
    Reconstruit la matrice de Cholesky creuse `L` sous forme DENSE [B, n, n].

    RÉSERVÉ AUX TESTS sur images jouets (16x16, n = 256, 262 ko par exemple).
    En 64x64 la matrice pèse 67 Mo par exemple : la fonction refuse de la
    construire, sauf `force=True` assumé.

    Retour :
        L : [B, n, n], triangulaire inférieure, diagonale > 0.
    """
    B, n = log_diag.shape
    if n > N_MAX_DENSE and not force:
        raise ValueError(
            "build_L_dense refuse n = %d : la matrice dense ferait %.1f Mo par "
            "exemple. Cette fonction ne sert qu'aux tests unitaires en 16x16 ; "
            "en 64x64, tout passe par apply_LT. (force=True pour outrepasser.)"
            % (n, n * n * 4 / 1e6)
        )

    m = offdiag.shape[2]
    device = log_diag.device

    L = torch.zeros(B, n, n, dtype=log_diag.dtype, device=device)

    # Hors-diagonale : L[b, i, neighbor_idx[i, k]] += offdiag[b, i, k] (masqué).
    rows = torch.arange(n, device=device).view(n, 1).expand(n, m).reshape(-1)  # [n*m]
    cols = neighbor_idx.reshape(-1)                                            # [n*m]
    vals = (offdiag * mask.unsqueeze(0)).reshape(B, n * m)                     # [B, n*m]

    bidx = torch.arange(B, device=device).view(B, 1).expand(B, n * m).reshape(-1)
    ridx = rows.view(1, -1).expand(B, -1).reshape(-1)
    cidx = cols.view(1, -1).expand(B, -1).reshape(-1)
    L.index_put_((bidx, ridx, cidx), vals.reshape(-1), accumulate=True)

    # Diagonale : écrase toute valeur parasite (les voisins hors-image pointent
    # sur i, mais leur valeur masquée est nulle -> pas d'effet ici).
    diag = torch.exp(clamp_log_diag(log_diag))  # [B, n]
    idx = torch.arange(n, device=device)
    L[:, idx, idx] = diag
    return L


def predicted_precision_and_covariance(log_diag, offdiag, neighbor_idx, mask,
                                       force=False):
    """
    Renvoie (Lambda, Sigma) prédites à partir des sorties du réseau.

        Lambda = L L^T           (précision)
        Sigma  = Lambda^{-1}     (covariance)

    RÉSERVÉ AUX TESTS, comme `build_L_dense` : l'inversion est en O(n^3), soit
    6.9e10 flops par image en 64x64. L'évaluation réelle n'inverse jamais.

    Retour :
        Lambda : [B, n, n]
        Sigma  : [B, n, n]
    """
    L = build_L_dense(log_diag, offdiag, neighbor_idx, mask, force=force)
    Lambda = L @ L.transpose(1, 2)
    Sigma = torch.linalg.inv(Lambda)
    return Lambda, Sigma


if __name__ == "__main__":
    torch.manual_seed(0)

    # =======================================================================
    # A) Tests unitaires de l'ALGORITHME, en 16x16 avec le voisinage f = 7 du
    #    projet. L'algorithme ne dépend pas de la taille d'image : le valider
    #    en 256 pixels le valide en 4096, et la matrice dense reste à 262 ko.
    # =======================================================================
    S, f, B = 16, VOISINAGE, 4
    n = S * S
    neighbor_idx, mask = build_neighbor_indices(S, f)
    m = neighbor_idx.shape[1]

    print("=== A) tests algorithme, S=%d f=%d ===" % (S, f))
    print("n = %d | voisins causaux m = %d (attendu 24 pour f=7)" % (n, m))
    assert m == (f * f - 1) // 2, "nombre de voisins causaux incorrect"

    log_diag = 0.1 * torch.randn(B, n)
    offdiag = 0.1 * torch.randn(B, n, m)
    r = torch.randn(B, n)

    # 1) L^T r creux == L^T r dense.
    w_sparse = apply_LT(log_diag, offdiag, r, neighbor_idx, mask)
    L = build_L_dense(log_diag, offdiag, neighbor_idx, mask)
    w_dense = torch.bmm(L.transpose(1, 2), r.unsqueeze(2)).squeeze(2)
    err = (w_sparse - w_dense).abs().max().item()
    print("erreur max apply_LT vs dense : %.2e (doit etre ~0)" % err)
    assert err < 1e-4

    # 2) L est bien triangulaire inférieure avec diagonale exp(log_diag).
    upper = torch.triu(L, diagonal=1).abs().max().item()
    diag_err = (torch.diagonal(L, dim1=1, dim2=2) - torch.exp(log_diag)).abs().max().item()
    print("masse triangle superieur : %.2e (doit etre 0)" % upper)
    print("erreur diagonale         : %.2e (doit etre 0)" % diag_err)
    assert upper == 0.0 and diag_err < 1e-6

    # 3) Lambda = L L^T définie positive.
    Lambda, Sigma = predicted_precision_and_covariance(log_diag, offdiag,
                                                       neighbor_idx, mask)
    eigmin = torch.linalg.eigvalsh(Lambda)[..., 0].min().item()
    print("valeur propre min de Lambda : %.3e (doit etre > 0)" % eigmin)
    assert eigmin > 0

    # 4) NLL dense de référence vs structurée.
    nll_struct = structured_gaussian_nll(log_diag, offdiag, r, neighbor_idx, mask)
    log_det_sigma = torch.logdet(Sigma)
    quad = torch.einsum("bi,bij,bj->b", r, Lambda, r)
    nll_ref = 0.5 * (log_det_sigma + quad + n * math.log(2 * math.pi))
    print("NLL structuree : %.4f | NLL dense ref : %.4f"
          % (nll_struct.item(), nll_ref.mean().item()))
    assert abs(nll_struct.item() - nll_ref.mean().item()) < 1e-1

    # =======================================================================
    # B) Tests de FORME en 64x64, la taille réelle. Aucune matrice dense ici :
    #    on vérifie les dimensions, la finitude et le passage du gradient.
    # =======================================================================
    S, B = IMAGE_SIZE, 8
    n = S * S
    neighbor_idx, mask = build_neighbor_indices(S, VOISINAGE)
    m = neighbor_idx.shape[1]

    print()
    print("=== B) tests de forme, S=%d f=%d ===" % (S, VOISINAGE))
    print("n = %d | m = %d | valeurs predites par image : %d"
          % (n, m, n * (m + 1)))
    assert (n, m) == (4096, 24)
    assert neighbor_idx.shape == (n, m) and mask.shape == (n, m)

    log_diag = (0.1 * torch.randn(B, n)).requires_grad_(True)
    offdiag = (0.1 * torch.randn(B, n, m)).requires_grad_(True)
    r = torch.randn(B, n)

    w = apply_LT(log_diag, offdiag, r, neighbor_idx, mask)
    print("apply_LT : %s -> %s" % (tuple(r.shape), tuple(w.shape)))
    assert w.shape == (B, n)

    nll = structured_gaussian_nll(log_diag, offdiag, r, neighbor_idx, mask)
    print("NLL (init aleatoire) : %.1f" % nll.item())
    assert torch.isfinite(nll), "NLL non finie"

    nll.backward()
    assert log_diag.grad is not None and torch.isfinite(log_diag.grad).all()
    assert offdiag.grad is not None and torch.isfinite(offdiag.grad).all()
    print("gradients : log_diag %s, offdiag %s (finis)"
          % (tuple(log_diag.grad.shape), tuple(offdiag.grad.shape)))

    # NLL par exemple, utile à eval_cov.py.
    nll_par_ex = structured_gaussian_nll(log_diag, offdiag, r, neighbor_idx, mask,
                                         mean_batch=False)
    assert nll_par_ex.shape == (B,)

    # =======================================================================
    # C) Le clamp : une diagonale absurde ne doit pas produire d'inf/NaN.
    #    C'est l'ajout par rapport au projet ellipses.
    # =======================================================================
    print()
    print("=== C) clamp de log_diag ===")
    log_diag_fou = torch.full((2, n), 500.0, requires_grad=True)
    offdiag_zero = torch.zeros(2, n, m)
    r2 = torch.randn(2, n)
    nll_fou = structured_gaussian_nll(log_diag_fou, offdiag_zero, r2,
                                      neighbor_idx, mask)
    print("log_diag = 500 -> NLL = %.3e (doit etre finie)" % nll_fou.item())
    assert torch.isfinite(nll_fou), "le clamp n'a pas protege la NLL"

    nll_fou.backward()
    grad_max = log_diag_fou.grad.abs().max().item()
    print("gradient sur log_diag sature : %.2e (doit etre 0)" % grad_max)
    assert grad_max == 0.0, "le clamp doit couper le gradient hors bornes"

    # Le garde-fou dense doit refuser n = 4096.
    try:
        build_L_dense(log_diag[:1].detach(), offdiag[:1].detach(),
                      neighbor_idx, mask)
        raise AssertionError("build_L_dense aurait du refuser n = 4096")
    except ValueError:
        print("build_L_dense refuse bien n = 4096")

    print()
    print("OK")