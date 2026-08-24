"""
DnCNN — le débruiteur (étape 1 du projet).

Architecture de Zhang et al., « Beyond a Gaussian Denoiser » (TIP 2017),
adaptée aux images 64x64 en niveaux de gris.

Le point central est l'APPRENTISSAGE RÉSIDUEL : le réseau ne prédit pas
l'image propre, il prédit le BRUIT. L'image débruitée s'obtient ensuite par
soustraction :

    bruit_predit = reseau(y)
    mu           = y - bruit_predit

C'est plus facile à apprendre. La cible `n = y - x` est une image de bruit,
statistiquement simple et sans structure ; la cible `x` serait un visage, avec
toute sa complexité. Le contenu de l'image, qui est déjà présent dans `y`,
n'a pas à être reconstruit : il suffit de le laisser passer.

Usage :
    python dncnn.py          # test de forme et compte des paramètres
"""

import torch
import torch.nn as nn

PROFONDEUR = 17      # nombre de couches convolutives (article : 17 pour un
                     # niveau de bruit connu, 20 pour du bruit aveugle)
CANAUX = 64          # largeur des couches internes
NOYAU = 3            # taille des filtres


class DnCNN(nn.Module):
    """
    Empilement de `profondeur` convolutions 3x3 sans aucun sous-échantillonnage.

    Structure :
        couche 1              Conv + ReLU
        couches 2 .. D-1      Conv + BatchNorm + ReLU
        couche D              Conv                       -> le bruit prédit

    La résolution ne change jamais : `padding = 1` avec un noyau 3x3 conserve
    64x64 d'un bout à l'autre. Pas de pooling, pas de skip connections internes
    — la seule connexion résiduelle est la soustraction finale.
    """

    def __init__(self, profondeur=PROFONDEUR, canaux=CANAUX,
                 canaux_image=1, noyau=NOYAU):
        super().__init__()
        rembourrage = noyau // 2
        couches = []

        # Première couche : pas de BatchNorm. Elle voit l'image brute, dont on
        # ne veut pas normaliser la statistique — c'est justement le niveau de
        # bruit qui porte l'information.
        couches.append(nn.Conv2d(canaux_image, canaux, noyau,
                                 padding=rembourrage, bias=True))
        couches.append(nn.ReLU(inplace=True))

        # Couches intermédiaires : Conv + BN + ReLU.
        # bias=False parce que le BatchNorm qui suit a déjà son propre décalage
        # (beta) : un biais de convolution serait redondant et sans effet.
        for _ in range(profondeur - 2):
            couches.append(nn.Conv2d(canaux, canaux, noyau,
                                     padding=rembourrage, bias=False))
            couches.append(nn.BatchNorm2d(canaux))
            couches.append(nn.ReLU(inplace=True))

        # Dernière couche : une convolution seule, sans activation. La sortie
        # est le bruit prédit, qui doit pouvoir être négatif — un ReLU final
        # le tronquerait à zéro.
        couches.append(nn.Conv2d(canaux, canaux_image, noyau,
                                 padding=rembourrage, bias=False))

        self.reseau = nn.Sequential(*couches)
        self.profondeur = profondeur
        self._initialiser()

    def _initialiser(self):
        """Initialisation de Kaiming, adaptée aux ReLU."""
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
        y : image bruitée [B, 1, 64, 64]
        renvoie mu : image débruitée [B, 1, 64, 64]
        """
        bruit_predit = self.reseau(y)
        return y - bruit_predit

    def bruit(self, y):
        """La sortie brute du réseau, avant soustraction. Utile au diagnostic."""
        return self.reseau(y)

    def champ_receptif(self):
        """
        Taille du champ réceptif, en pixels.

        Chaque convolution 3x3 ajoute 1 pixel de chaque côté, donc 2 par
        couche. Avec 17 couches : 1 + 2*17 = 35 pixels. Sur une image de 64,
        chaque pixel de sortie voit donc un peu plus de la moitié de l'image —
        largement de quoi capturer une texture locale.
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

    # Test de forme : l'entrée et la sortie doivent avoir exactement la même
    # taille. Si le padding était faux, l'erreur apparaîtrait ici.
    y = torch.randn(4, 1, 64, 64)
    with torch.no_grad():
        mu = modele(y)
    print("  entrée %s -> sortie %s" % (tuple(y.shape), tuple(mu.shape)))
    assert y.shape == mu.shape, "la forme doit être conservée"

    # À l'initialisation, le réseau prédit un bruit quelconque : mu n'a aucune
    # raison de ressembler à y. On vérifie seulement que rien n'explose.
    print("  mu : moyenne %.3f, écart-type %.3f" % (mu.mean(), mu.std()))
    assert torch.isfinite(mu).all(), "sortie non finie à l'initialisation"
    print("  OK")
