"""
Structured Gaussian NLL loss for CelebA (SPARSE Cholesky of the precision).

Carried over from the ellipses project, with two constant changes and one
numerical safeguard. The two central functions (`apply_LT` and
`structured_gaussian_nll`) do not depend on the image size nor on the
neighborhood: they go from 16x16 to 64x64 untouched.

    ellipses : S = 16, n =  256, f = 5, m = 12  ->  13 values per pixel
    celebA   : S = 64, n = 4096, f = 7, m = 24  ->  25 values per pixel

Sparsity pattern of the article (Dorta et al., 2018, §5.1):

    L[i, j] != 0  only if  i >= j  AND the pixels i, j are neighbors
    inside an f x f patch (here f = 7).

For each pixel `i`, `L` therefore has non-zero values only on:
- the diagonal `L[i, i]`, parameterized by `log(l_ii)` (the `exp` guarantees
  > 0);
- the `L[i, j]` for `j` a CAUSAL neighbor of `i` (j comes before i in raster
  order AND lies inside the f x f patch). For f = 7 there are 24 causal
  neighbors.

The precision is `Lambda = L L^T` (symmetric positive definite by
construction). The Gaussian NLL (Eq. 4 of the article) is computed WITHOUT
inverting Sigma:

    nll = 0.5 * [ log|Sigma| + r^T Lambda r + n*log(2*pi) ]
        = 0.5 * [ -2*sum_i log(l_ii) + || L^T r ||^2 + n*log(2*pi) ]

with here `r = x - mu`, the residual of the DnCNN (and not the `x - mu` of a
spline).

`L^T r` is computed directly from the non-zero values, through a
`scatter_add`: no n x n matrix is ever formed. This is the survival condition
of the project — at n = 4096, a single dense float32 matrix weighs 67 MB, and
a batch of 64 would weigh 4.3 GB.

CHANGE WITH RESPECT TO THE ELLIPSES: `log_diag` is now clamped to
[-10, +10] before the `exp`. At 4096 pixels and on real data, a diagonal
value that runs off to +inf makes the NLL explode in a single iteration; in
16x16 on synthetic data one could do without it, no longer here.

Usage:
    python loss.py          # 16x16 unit tests + 64x64 shape tests
"""

import math

import numpy as np
import torch


# Constants of the project (cf. CLAUDE.md, phase 0 of the roadmap).
IMAGE_SIZE = 64      # 64x64 images -> n = 4096 pixels
VOISINAGE = 7        # f x f patch of the article -> m = 24 causal neighbors
CLAMP_LOG_DIAG = 10.0

# Beyond this size we refuse to build a dense matrix: at n = 4096 it weighs
# 67 MB per example. The dense functions are only useful for the unit tests on
# toy images.
N_MAX_DENSE = 1024


def clamp_log_diag(log_diag, limite=CLAMP_LOG_DIAG):
    """
    Clamps `log(l_ii)` into [-limite, +limite] before any `exp`.

    Outside the interval the gradient is zero: this is exactly the intended
    effect, a diagonal that has gone too far stops being pushed further. The
    bounds correspond to `l_ii` between `exp(-10) = 4.5e-5` and
    `exp(10) = 2.2e4`, which covers by far the useful scales for a residual
    in [-2, 2].
    """
    return torch.clamp(log_diag, min=-limite, max=limite)


# ---------------------------------------------------------------------------
# Sparsity pattern: causal neighbors inside an f x f patch
# ---------------------------------------------------------------------------
def causal_offsets(f=VOISINAGE):
    """
    Offsets (dr, dc) of the CAUSAL neighbors (current pixel excluded) of an
    f x f patch.

    A neighbor `j = i + (dr, dc)` is causal if its raster index is < that of
    `i`, i.e. `dr < 0`, or (`dr == 0` and `dc < 0`). In raster order
    (index = row * S + col), this guarantees that `L` is indeed lower
    triangular.

    For f = 7 (h = 3) there are 24 offsets:
    - dr in {-3, -2, -1}, dc in {-3, ..., 3}  -> 21
    - dr == 0,            dc in {-3, -2, -1}  ->  3

    Return:
        offsets : list[(dr, dc)] of length (f*f - 1) // 2
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
    Precomputes, for each pixel `i`, the raster index of its causal neighbors
    and a validity mask (pixels outside the image = invalid).

    Return:
        neighbor_idx : LongTensor [n, m]  (m = 24 for f = 7)
                       neighbor_idx[i, k] = raster index of the k-th causal
                       neighbor of i, or `i` itself if the neighbor falls out
                       of the image (a value neutralized by the mask, the
                       diagonal being overwritten separately).
        mask         : FloatTensor [n, m], 1.0 if the neighbor is in the image.

    These tensors do not depend on the example: they are computed ONCE at the
    beginning of training and reused for the whole batch and the whole
    experiment. In 64x64 they are [4096, 24], that is less than 500 kB.
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
                # Neighbor out of the image: we send it back onto `i` (any
                # column would do) and neutralize it with the mask (0 -> null
                # contribution).
                neighbor_idx[i, k] = i
                mask[i, k] = 0.0

    neighbor_idx = torch.from_numpy(neighbor_idx)
    mask = torch.from_numpy(mask)
    if device is not None:
        neighbor_idx = neighbor_idx.to(device)
        mask = mask.to(device)
    return neighbor_idx, mask


# ---------------------------------------------------------------------------
# Product L^T r without a dense matrix
# ---------------------------------------------------------------------------
def apply_LT(log_diag, offdiag, residual, neighbor_idx, mask):
    """
    Computes `w = L^T r` from the non-zero values of `L`, without ever
    building the n x n matrix. Cost O(n*m) instead of O(n^2).

    Reminder: `L[i, i] = exp(log_diag[i])` and
    `L[i, neighbor_idx[i, k]] = offdiag[i, k]`.
    We have `w_c = (L^T r)_c = sum_i L[i, c] r_i`, hence:
    - diagonal : `w_i += exp(log_diag_i) * r_i`;
    - off-diag : each `L[i, j] * r_i` is added to `w_j` (scatter_add on j).

    This function also serves for EVALUATION: `w = L^T r` must follow
    `N(0, I)` if the covariance is well calibrated (cf. eval_cov.py).

    Args:
        log_diag     : [B, n]      log values of the diagonal of L (clamped
                                   here).
        offdiag      : [B, n, m]   off-diagonal values (one per causal
                                   neighbor).
        residual     : [B, n]      r = x - mu.
        neighbor_idx : [n, m]      indices of the causal neighbors.
        mask         : [n, m]      validity mask of the neighbors.

    Return:
        w : [B, n],  w = L^T r.
    """
    B, n = residual.shape
    m = offdiag.shape[2]

    # Diagonal term. The clamp is applied here as well so that the function is
    # safe when called directly (calibration), not only through the NLL.
    # It is idempotent: re-applying it changes nothing.
    w = torch.exp(clamp_log_diag(log_diag)) * residual  # [B, n]

    # Off-diagonal terms: contribution of each pixel i to its neighbors j.
    contrib = offdiag * mask.unsqueeze(0) * residual.unsqueeze(2)  # [B, n, m]

    # scatter_add over the pixel dimension:
    # w[b, neighbor_idx[i, k]] += contrib[b, i, k].
    idx = neighbor_idx.reshape(1, n * m).expand(B, n * m)
    w = w.scatter_add(1, idx, contrib.reshape(B, n * m))
    return w


# ---------------------------------------------------------------------------
# Structured Gaussian NLL
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
    Gaussian NLL (Eq. 4 of the article) for the sparse structured precision.

        nll = 0.5 * [ -2*sum_i log(l_ii) + ||L^T r||^2 + n*log(2*pi) ]

    with `log(l_ii) = log_diag_i` (hence `log|Sigma| = -2 sum_i log_diag_i`)
    and `||L^T r||^2 = r^T Lambda r`. No inversion, no n x n matrix.

    The two terms are read against each other: the quadratic term pushes the
    precision towards 0 (a huge covariance makes any residual probable), the
    log-determinant prevents it (it rewards large diagonals). Their balance is
    what the network learns.

    IMPORTANT: `log_diag` is clamped BEFORE being used, and the SAME clamped
    value serves the log-determinant and `apply_LT`. If the two terms did not
    see the same diagonal, the loss would no longer be a coherent NLL.

    Args:
        log_diag, offdiag : outputs of the covariance network ([B, n] and
                            [B, n, m]).
        residual          : r = x - mu, [B, n]  (mu = DnCNN(y), frozen).
        neighbor_idx, mask: sparsity pattern (cf. build_neighbor_indices).
        include_const     : adds the constant term n*log(2*pi) (true NLL).
                            To be left True to compare NLLs across models.
        mean_batch        : average over the batch (otherwise per-example
                            return [B]).

    Return:
        nll : scalar (mean_batch=True) or [B].
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
# Dense reconstruction — UNIT TESTS ONLY, never in 64x64
# ---------------------------------------------------------------------------
def build_L_dense(log_diag, offdiag, neighbor_idx, mask, force=False):
    """
    Rebuilds the sparse Cholesky matrix `L` in DENSE form [B, n, n].

    RESERVED FOR THE TESTS on toy images (16x16, n = 256, 262 kB per example).
    In 64x64 the matrix weighs 67 MB per example: the function refuses to
    build it, unless a deliberately assumed `force=True`.

    Return:
        L : [B, n, n], lower triangular, diagonal > 0.
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

    # Off-diagonal: L[b, i, neighbor_idx[i, k]] += offdiag[b, i, k] (masked).
    rows = torch.arange(n, device=device).view(n, 1).expand(n, m).reshape(-1)  # [n*m]
    cols = neighbor_idx.reshape(-1)                                            # [n*m]
    vals = (offdiag * mask.unsqueeze(0)).reshape(B, n * m)                     # [B, n*m]

    bidx = torch.arange(B, device=device).view(B, 1).expand(B, n * m).reshape(-1)
    ridx = rows.view(1, -1).expand(B, -1).reshape(-1)
    cidx = cols.view(1, -1).expand(B, -1).reshape(-1)
    L.index_put_((bidx, ridx, cidx), vals.reshape(-1), accumulate=True)

    # Diagonal: overwrites any spurious value (the out-of-image neighbors point
    # onto i, but their masked value is null -> no effect here).
    diag = torch.exp(clamp_log_diag(log_diag))  # [B, n]
    idx = torch.arange(n, device=device)
    L[:, idx, idx] = diag
    return L


def predicted_precision_and_covariance(log_diag, offdiag, neighbor_idx, mask,
                                       force=False):
    """
    Returns the (Lambda, Sigma) predicted from the outputs of the network.

        Lambda = L L^T           (precision)
        Sigma  = Lambda^{-1}     (covariance)

    RESERVED FOR THE TESTS, like `build_L_dense`: the inversion is O(n^3),
    that is 6.9e10 flops per image in 64x64. The real evaluation never
    inverts.

    Return:
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
    # A) Unit tests of the ALGORITHM, in 16x16 with the f = 7 neighborhood of
    #    the project. The algorithm does not depend on the image size:
    #    validating it on 256 pixels validates it on 4096, and the dense
    #    matrix stays at 262 kB.
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

    # 1) sparse L^T r == dense L^T r.
    w_sparse = apply_LT(log_diag, offdiag, r, neighbor_idx, mask)
    L = build_L_dense(log_diag, offdiag, neighbor_idx, mask)
    w_dense = torch.bmm(L.transpose(1, 2), r.unsqueeze(2)).squeeze(2)
    err = (w_sparse - w_dense).abs().max().item()
    print("erreur max apply_LT vs dense : %.2e (doit etre ~0)" % err)
    assert err < 1e-4

    # 2) L is indeed lower triangular with diagonal exp(log_diag).
    upper = torch.triu(L, diagonal=1).abs().max().item()
    diag_err = (torch.diagonal(L, dim1=1, dim2=2) - torch.exp(log_diag)).abs().max().item()
    print("masse triangle superieur : %.2e (doit etre 0)" % upper)
    print("erreur diagonale         : %.2e (doit etre 0)" % diag_err)
    assert upper == 0.0 and diag_err < 1e-6

    # 3) Lambda = L L^T positive definite.
    Lambda, Sigma = predicted_precision_and_covariance(log_diag, offdiag,
                                                       neighbor_idx, mask)
    eigmin = torch.linalg.eigvalsh(Lambda)[..., 0].min().item()
    print("valeur propre min de Lambda : %.3e (doit etre > 0)" % eigmin)
    assert eigmin > 0

    # 4) Reference dense NLL vs structured one.
    nll_struct = structured_gaussian_nll(log_diag, offdiag, r, neighbor_idx, mask)
    log_det_sigma = torch.logdet(Sigma)
    quad = torch.einsum("bi,bij,bj->b", r, Lambda, r)
    nll_ref = 0.5 * (log_det_sigma + quad + n * math.log(2 * math.pi))
    print("NLL structuree : %.4f | NLL dense ref : %.4f"
          % (nll_struct.item(), nll_ref.mean().item()))
    assert abs(nll_struct.item() - nll_ref.mean().item()) < 1e-1

    # =======================================================================
    # B) SHAPE tests in 64x64, the real size. No dense matrix here: we check
    #    the dimensions, the finiteness and that the gradient flows through.
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

    # Per-example NLL, useful to eval_cov.py.
    nll_par_ex = structured_gaussian_nll(log_diag, offdiag, r, neighbor_idx, mask,
                                         mean_batch=False)
    assert nll_par_ex.shape == (B,)

    # =======================================================================
    # C) The clamp: an absurd diagonal must not produce inf/NaN.
    #    This is the addition with respect to the ellipses project.
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

    # The dense safeguard must refuse n = 4096.
    try:
        build_L_dense(log_diag[:1].detach(), offdiag[:1].detach(),
                      neighbor_idx, mask)
        raise AssertionError("build_L_dense aurait du refuser n = 4096")
    except ValueError:
        print("build_L_dense refuse bien n = 4096")

    print()
    print("OK")