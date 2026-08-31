"""
Covariance network — step 2 of the project.

The DnCNN is trained and then FROZEN. This network learns the covariance of
the residual `r = x - mu` that it leaves behind, that is to say the
high-frequency detail the denoising has erased.

Input / output:

    mu        [B, 1, 64, 64]   (or [B, 4096] flattened)
    ->
    log_diag  [B, 4096]        log of the diagonal terms of L
    offdiag   [B, 4096, 24]    the 24 off-diagonal terms per pixel

`L` is the Cholesky factor of the PRECISION `Lambda = Sigma^{-1} = L L^T`,
never that of the covariance. The network NEVER sees the clean image `x`:
with `x` it could reconstruct `r` directly and the NLL would collapse without
anything being learned about the structure.

Usage:
    python cov_model.py      # shapes, parameter count, gradient test
"""

import math

import torch
import torch.nn as nn

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
TAILLE_IMAGE = 64
N_PIXELS = TAILLE_IMAGE * TAILLE_IMAGE          # 4096

# Sparsity pattern from the paper: l_ij is non-zero only if i >= j and if the
# pixels i, j are neighbours inside an f x f patch. With f = 7 that leaves
# (49 - 1) / 2 = 24 off-diagonal terms per pixel, plus the diagonal.
TAILLE_PATCH = 7
N_VOISINS = (TAILLE_PATCH ** 2 - 1) // 2        # 24
N_SORTIES = 1 + N_VOISINS                       # 25 channels at the head

# Width of the three U-Net levels.
CANAUX = (32, 64, 128)

# Standard deviation of the DnCNN residual, measured on 26 August on the FINAL
# MODEL (dncnn_best.pt, best validation epoch: 45 out of 50).
# Used for initialisation only. See docs/tuteur.txt §4.
#
# Previous value: 0.0698, measured on a short 3-epoch run. The 47 epochs that
# follow only gain 8 % on the residual — the DnCNN plateaus early, which is
# its known behaviour. To be recalibrated only if the denoiser changes.
STD_RESIDU = 0.0640
INIT_LOG_DIAG = -math.log(STD_RESIDU)           # ~ 2.66


# --------------------------------------------------------------------------
# Basic building block
# --------------------------------------------------------------------------
class Bloc(nn.Module):
    """
    Two 3x3 convolutions, each followed by BatchNorm and ReLU.

    `bias=False` on the convolutions: the BatchNorm that follows already has
    its own shift (beta), so a convolution bias would be redundant. Same
    reasoning as in dncnn.py.
    """

    def __init__(self, entree, sortie):
        super().__init__()
        self.bloc = nn.Sequential(
            nn.Conv2d(entree, sortie, 3, padding=1, bias=False),
            nn.BatchNorm2d(sortie),
            nn.ReLU(inplace=True),
            nn.Conv2d(sortie, sortie, 3, padding=1, bias=False),
            nn.BatchNorm2d(sortie),
            nn.ReLU(inplace=True),
        )

    def forward(self, v):
        return self.bloc(v)


def _remontee(entree, sortie):
    """
    Going up one level: bilinear interpolation x2 then a 3x3 convolution.

    NO ConvTranspose2d. A deconvolution with kernel 2 and stride 2 produces
    checkerboard artefacts; on a precision map that would give a periodic
    pattern in the predicted covariance, which one would mistake for learned
    structure. Interpolation followed by convolution avoids the problem for
    the same cost.
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(entree, sortie, 3, padding=1, bias=False),
        nn.BatchNorm2d(sortie),
        nn.ReLU(inplace=True),
    )


# --------------------------------------------------------------------------
# The network
# --------------------------------------------------------------------------
class SparseCholeskyNet(nn.Module):
    """
    3-level U-Net, 1x1 convolutional head with 25 channels.

        64x64  enc1   32 channels  ------------------------------+
          |    pool                                              |
        32x32  enc2   64 channels  --------------+               |
          |    pool                              |               |
        16x16  fond  128 channels                |               |
          |    upsample                          |               |
        32x32  dec2   64 channels  <-- concat ---+               |
          |    upsample                                          |
        64x64  dec1   32 channels  <-- concat -------------------+
          |
        1x1 conv head -> 25 channels

    WHY A U-NET AND NOT A FLAT STACK LIKE THE DnCNN.
    The covariance of the residual depends on the context: hair texture, the
    contour of an eyelid and a smooth cheek do not have the same statistics.
    One therefore has to see wide. A flat stack of 3x3 convolutions gains
    2 pixels of receptive field per layer; the downsamplings gain twice as
    much at every level, for far less computation.

    RECEPTIVE FIELD, along the deepest path:
        enc1  ->  5      pool  ->  6      enc2  -> 14      pool -> 16
        fond  -> 32      dec2  -> 44      dec1  -> 50
    That is about 50 x 50 pixels on a 64-pixel image, and 32 x 32 already at
    the bottom of the U. Comparable to the DnCNN's 35 x 35, which is
    consistent: both networks have to recognise the same structures.

    ORDER OF THE 25 CHANNELS — this is the contract with loss.py, do not
    break it:
        channel 0        -> log_diag, the log of the diagonal term l_ii
        channels 1 to 24 -> offdiag, in the EXACT ORDER of
                            causal_offsets(f=7), hence
                            offdiag[:, i, k] = l_ij with
                            j = neighbor_idx[i, k].
    Permuting these 24 channels does not crash the code: it merely yields a
    model that learns a wrong matrix, silently.

    No clamp here. `log_diag.clamp(-10, 10)` belongs to loss.py, in one single
    place, otherwise one no longer knows which of the two protects what.
    """

    def __init__(self, canaux=CANAUX, init_log_diag=INIT_LOG_DIAG,
                 diagonale_seule=False):
        super().__init__()
        c1, c2, c3 = canaux
        self.diagonale_seule = diagonale_seule
        self.init_log_diag = init_log_diag

        # Going down
        self.enc1 = Bloc(1, c1)
        self.enc2 = Bloc(c1, c2)
        self.fond = Bloc(c2, c3)
        self.pool = nn.MaxPool2d(2)

        # Going up. After each concatenation the number of channels doubles,
        # hence the `2 * c` at the input of the decoder blocks.
        self.up2 = _remontee(c3, c2)
        self.dec2 = Bloc(2 * c2, c2)
        self.up1 = _remontee(c2, c1)
        self.dec1 = Bloc(2 * c1, c1)

        # Head: a single 1x1 convolution. The 25 values of a pixel are read
        # from the channels at that location, with no further spatial mixing —
        # all the context has already been aggregated by the U.
        self.tete = nn.Conv2d(c1, N_SORTIES, 1, bias=True)
        self._initialiser()

    def _initialiser(self):
        """
        Kaiming everywhere, EXCEPT the head, initialised to zero with a bias
        on channel 0.

        Consequence: on the very first pass the network predicts
        log_diag = init_log_diag everywhere and offdiag = 0, that is to say an
        isotropic Gaussian of standard deviation std(r). The model therefore
        starts from the dumbest possible answer but already at the right
        scale, and only has to move away from it. Without this the first
        epochs start from an arbitrary precision and the NLL does anything.

        Side effect worth knowing: a head with zero weights backpropagates NO
        gradient at all to the trunk on the first iteration (the gradient at
        the input of the head is W^T g = 0). Only the head's weights move. As
        soon as they are non-zero, the trunk learns normally. This is
        expected, not a bug — the __main__ test checks exactly that.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in",
                                        nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.zeros_(self.tete.weight)
        nn.init.zeros_(self.tete.bias)
        with torch.no_grad():
            self.tete.bias[0] = self.init_log_diag

    def forward(self, mu):
        """
        mu : [B, 1, 64, 64] or [B, 4096]
        returns (log_diag [B, 4096], offdiag [B, 4096, 24])
        """
        if mu.dim() == 2:
            mu = mu.view(-1, 1, TAILLE_IMAGE, TAILLE_IMAGE)
        if mu.dim() != 4 or mu.shape[1] != 1:
            raise ValueError("mu attendu en [B, 1, 64, 64] ou [B, 4096], "
                             "reçu %s" % (tuple(mu.shape),))

        e1 = self.enc1(mu)                                   # [B, 32, 64, 64]
        e2 = self.enc2(self.pool(e1))                        # [B, 64, 32, 32]
        f = self.fond(self.pool(e2))                         # [B, 128, 16, 16]

        d2 = self.dec2(torch.cat([self.up2(f), e2], dim=1))   # [B, 64, 32, 32]
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # [B, 32, 64, 64]

        sortie = self.tete(d1)                               # [B, 25, 64, 64]
        batch = sortie.shape[0]

        # Raster order: i = row * 64 + column. This is exactly the order of
        # `view` on the last two dimensions, no permutation needed.
        log_diag = sortie[:, 0].reshape(batch, N_PIXELS)

        if self.diagonale_seule:
            # Diagonal baseline (§6.2): the same network, but with offdiag
            # forced to zero. The NLL gap with the full model quantifies what
            # the structure brings. With no gradient to send back, the 24
            # corresponding channels simply stay inert.
            offdiag = torch.zeros(batch, N_PIXELS, N_VOISINS,
                                  device=sortie.device, dtype=sortie.dtype)
        else:
            # [B, 24, 64, 64] -> [B, 24, 4096] -> [B, 4096, 24]
            # `.contiguous()` because the loss indexes this tensor neighbour
            # by neighbour; a transposed view would cost more there than the
            # copy.
            offdiag = (sortie[:, 1:]
                       .reshape(batch, N_VOISINS, N_PIXELS)
                       .permute(0, 2, 1)
                       .contiguous())

        return log_diag, offdiag

    def champ_receptif(self):
        """Approximate receptive field, in pixels. See the docstring table."""
        return 50


# --------------------------------------------------------------------------
# Module test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    modele = SparseCholeskyNet()
    n_param = sum(p.numel() for p in modele.parameters())

    print("SparseCholeskyNet")
    print("  paramètres      : %d  (%.2f M)" % (n_param, n_param / 1e6))
    print("  champ réceptif  : ~%d x %d pixels" % (modele.champ_receptif(),
                                                   modele.champ_receptif()))
    print("  sorties         : %d canaux = 1 + %d" % (N_SORTIES, N_VOISINS))
    print("  init_log_diag   : %.4f  (= -log %.4f)" % (INIT_LOG_DIAG,
                                                       STD_RESIDU))

    # --- shapes ----------------------------------------------------------
    mu = torch.randn(4, 1, TAILLE_IMAGE, TAILLE_IMAGE)
    log_diag, offdiag = modele(mu)
    print("  entrée %s -> log_diag %s, offdiag %s"
          % (tuple(mu.shape), tuple(log_diag.shape), tuple(offdiag.shape)))
    assert log_diag.shape == (4, N_PIXELS)
    assert offdiag.shape == (4, N_PIXELS, N_VOISINS)

    # The flattened input must give exactly the same result.
    log_diag_plat, _ = modele(mu.view(4, N_PIXELS))
    assert torch.allclose(log_diag, log_diag_plat), "entrée aplatie divergente"

    # --- initialisation --------------------------------------------------
    # Constant output and at the right scale: isotropic Gaussian of standard
    # deviation std(r). This is what the zeroed head must guarantee.
    assert torch.allclose(log_diag, torch.full_like(log_diag, INIT_LOG_DIAG)), \
        "log_diag non constant à l'initialisation"
    assert offdiag.abs().max() == 0, "offdiag non nul à l'initialisation"
    sigma_init = math.exp(-INIT_LOG_DIAG)
    print("  à l'init : sigma prédit = %.4f, résidu mesuré = %.4f"
          % (sigma_init, STD_RESIDU))

    # --- gradients -------------------------------------------------------
    # Fake criterion, only there to check that gradients flow. The real NLL
    # is in loss.py.
    cible_d = torch.randn(4, N_PIXELS)
    cible_o = torch.randn(4, N_PIXELS, N_VOISINS)

    def faux_critere(modele):
        ld, od = modele(mu)
        return ((ld - cible_d) ** 2).mean() + ((od - cible_o) ** 2).mean()

    perte = faux_critere(modele)
    perte.backward()

    # Expected: the head receives gradient, the trunk does not get any yet.
    g_tete = modele.tete.weight.grad.abs().max().item()
    g_tronc = modele.enc1.bloc[0].weight.grad.abs().max().item()
    print("  passe 1 : |grad| tête = %.3e, tronc = %.3e" % (g_tete, g_tronc))
    assert g_tete > 0, "la tête ne reçoit pas de gradient"
    assert g_tronc == 0, "le tronc ne devrait pas encore bouger (tête à zéro)"

    # After one step the head is no longer zero: the trunk must learn.
    opt = torch.optim.Adam(modele.parameters(), lr=1e-3)
    opt.step()
    opt.zero_grad()
    faux_critere(modele).backward()
    g_tronc = modele.enc1.bloc[0].weight.grad.abs().max().item()
    print("  passe 2 : |grad| tronc = %.3e" % g_tronc)
    assert g_tronc > 0, "le tronc n'apprend pas après le premier pas"

    # --- diagonal baseline -----------------------------------------------
    diag_seul = SparseCholeskyNet(diagonale_seule=True)
    ld, od = diag_seul(mu)
    assert od.shape == (4, N_PIXELS, N_VOISINS) and od.abs().max() == 0
    print("  référence diagonale : offdiag identiquement nul   OK")

    # --- memory cost of a real batch -------------------------------------
    octets = 64 * N_PIXELS * N_SORTIES * 4
    print("  sortie pour un batch de 64 : %.0f Mo" % (octets / 1e6))
    print("  OK")