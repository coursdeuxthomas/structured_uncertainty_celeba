"""
DnCNN — the denoiser (step 1 of the project).

Architecture of Zhang et al., "Beyond a Gaussian Denoiser" (TIP 2017),
adapted to 64x64 grayscale images.

The central point is RESIDUAL LEARNING: the network does not predict the
clean image, it predicts the NOISE. The denoised image is then obtained by
subtraction:

    bruit_predit = reseau(y)
    mu           = y - bruit_predit

This is easier to learn. The target `n = y - x` is a noise image,
statistically simple and without structure; the target `x` would be a face,
with all its complexity. The content of the image, which is already present
in `y`, does not have to be reconstructed: it is enough to let it through.

Usage:
    python dncnn.py          # shape test and parameter count
"""

import torch
import torch.nn as nn

PROFONDEUR = 17      # number of convolutional layers (article: 17 for a
                     # known noise level, 20 for blind noise)
CANAUX = 64          # width of the internal layers
NOYAU = 3            # size of the filters


class DnCNN(nn.Module):
    """
    Stack of `profondeur` 3x3 convolutions without any downsampling.

    Structure:
        layer 1               Conv + ReLU
        layers 2 .. D-1       Conv + BatchNorm + ReLU
        layer D               Conv                       -> the predicted noise

    The resolution never changes: `padding = 1` with a 3x3 kernel preserves
    64x64 from one end to the other. No pooling, no internal skip connections
    — the only residual connection is the final subtraction.
    """

    def __init__(self, profondeur=PROFONDEUR, canaux=CANAUX,
                 canaux_image=1, noyau=NOYAU):
        super().__init__()
        rembourrage = noyau // 2
        couches = []

        # First layer: no BatchNorm. It sees the raw image, whose statistics
        # we do not want to normalize — it is precisely the noise level that
        # carries the information.
        couches.append(nn.Conv2d(canaux_image, canaux, noyau,
                                 padding=rembourrage, bias=True))
        couches.append(nn.ReLU(inplace=True))

        # Intermediate layers: Conv + BN + ReLU.
        # bias=False because the BatchNorm that follows already has its own
        # shift (beta): a convolution bias would be redundant and without
        # effect.
        for _ in range(profondeur - 2):
            couches.append(nn.Conv2d(canaux, canaux, noyau,
                                     padding=rembourrage, bias=False))
            couches.append(nn.BatchNorm2d(canaux))
            couches.append(nn.ReLU(inplace=True))

        # Last layer: a convolution alone, without activation. The output is
        # the predicted noise, which must be able to be negative — a final
        # ReLU would truncate it to zero.
        couches.append(nn.Conv2d(canaux, canaux_image, noyau,
                                 padding=rembourrage, bias=False))

        self.reseau = nn.Sequential(*couches)
        self.profondeur = profondeur
        self._initialiser()

    def _initialiser(self):
        """Kaiming initialization, suited to ReLU."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in",
                                        nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, y):
        """
        y : noisy image [B, 1, 64, 64]
        returns mu : denoised image [B, 1, 64, 64]
        """
        bruit_predit = self.reseau(y)
        return y - bruit_predit

    def bruit(self, y):
        """The network's raw output, before subtraction. Useful for diagnosis."""
        return self.reseau(y)

    def champ_receptif(self):
        """
        Size of the receptive field, in pixels.

        Each 3x3 convolution adds 1 pixel on each side, hence 2 per layer.
        With 17 layers: 1 + 2*17 = 35 pixels. On a 64-pixel image, each output
        pixel therefore sees a little more than half of the image — largely
        enough to capture a local texture.
        """
        return 1 + 2 * self.profondeur


if __name__ == "__main__":
    modele = DnCNN()
    n_param = sum(p.numel() for p in modele.parameters())

    print("DnCNN")
    print("  profondeur      : %d couches" % modele.profondeur)
    print("  paramètres      : %d  (%.2f M)" % (n_param, n_param / 1e6))
    print("  champ réceptif  : %d x %d pixels" % (modele.champ_receptif(),
                                                  modele.champ_receptif()))

    # Shape test: the input and the output must have exactly the same size.
    # If the padding were wrong, the error would show up here.
    y = torch.randn(4, 1, 64, 64)
    with torch.no_grad():
        mu = modele(y)
    print("  entrée %s -> sortie %s" % (tuple(y.shape), tuple(mu.shape)))
    assert y.shape == mu.shape, "la forme doit être conservée"

    # At initialization, the network predicts an arbitrary noise: mu has no
    # reason to resemble y. We only check that nothing blows up.
    print("  mu : moyenne %.3f, écart-type %.3f" % (mu.mean(), mu.std()))
    assert torch.isfinite(mu).all(), "sortie non finie à l'initialisation"
    print("  OK")
