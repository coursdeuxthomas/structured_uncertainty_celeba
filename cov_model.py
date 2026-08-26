"""
Réseau de covariance — étape 2 du projet.

Le DnCNN est entraîné puis GELÉ. Ce réseau-ci apprend la covariance du résidu
`r = x - mu` qu'il laisse derrière lui, c'est-à-dire le détail haute-fréquence
que le débruitage a effacé.

Entrée / sortie :

    mu        [B, 1, 64, 64]   (ou [B, 4096] aplati)
    ->
    log_diag  [B, 4096]        log des termes diagonaux de L
    offdiag   [B, 4096, 24]    les 24 termes hors-diagonale par pixel

`L` est la Cholesky de la PRÉCISION `Lambda = Sigma^{-1} = L L^T`, jamais celle
de la covariance. Le réseau ne voit JAMAIS l'image propre `x` : avec `x` il
pourrait reconstruire `r` directement et la NLL s'effondrerait sans que rien
soit appris sur la structure.

Usage :
    python cov_model.py      # formes, compte des paramètres, test de gradient
"""

import math

import torch
import torch.nn as nn

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
TAILLE_IMAGE = 64
N_PIXELS = TAILLE_IMAGE * TAILLE_IMAGE          # 4096

# Motif de parcimonie de l'article : l_ij non nul seulement si i >= j et si les
# pixels i, j sont voisins dans un patch f x f. Avec f = 7 il reste
# (49 - 1) / 2 = 24 termes hors-diagonale par pixel, plus la diagonale.
TAILLE_PATCH = 7
N_VOISINS = (TAILLE_PATCH ** 2 - 1) // 2        # 24
N_SORTIES = 1 + N_VOISINS                       # 25 canaux en tête de réseau

# Largeur des trois niveaux du U-Net.
CANAUX = (32, 64, 128)

# Écart-type du résidu du DnCNN, mesuré le 26 août sur le MODÈLE FINAL
# (dncnn_best.pt, meilleure epoch de validation : 45 sur 50).
# Sert uniquement à l'initialisation. Voir tuteur.txt §4.
#
# Valeur précédente : 0,0698, mesurée sur un run court de 3 epochs. Les 47
# epochs suivantes ne gagnent que 8 % sur le résidu — le DnCNN plafonne tôt,
# c'est son comportement connu. À ne recalibrer que si le débruiteur change.
STD_RESIDU = 0.0640
INIT_LOG_DIAG = -math.log(STD_RESIDU)           # ~ 2,66


# --------------------------------------------------------------------------
# Brique de base
# --------------------------------------------------------------------------
class Bloc(nn.Module):
    """
    Deux convolutions 3x3, chacune suivie de BatchNorm et ReLU.

    `bias=False` sur les convolutions : le BatchNorm qui suit a déjà son propre
    décalage (beta), un biais de convolution serait redondant. Même
    raisonnement que dans dncnn.py.
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
    Remontée d'un niveau : interpolation bilinéaire x2 puis convolution 3x3.

    PAS de ConvTranspose2d. Une déconvolution à noyau 2 et pas 2 produit des
    artefacts en damier ; sur une carte de précision cela donnerait un motif
    périodique dans la covariance prédite, qu'on prendrait pour de la structure
    apprise. L'interpolation puis convolution évite le problème pour le même
    coût.
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(entree, sortie, 3, padding=1, bias=False),
        nn.BatchNorm2d(sortie),
        nn.ReLU(inplace=True),
    )


# --------------------------------------------------------------------------
# Le réseau
# --------------------------------------------------------------------------
class SparseCholeskyNet(nn.Module):
    """
    U-Net à 3 niveaux, tête convolutive 1x1 à 25 canaux.

        64x64  enc1   32 canaux  ------------------------------+
          |    pool                                            |
        32x32  enc2   64 canaux  --------------+               |
          |    pool                            |               |
        16x16  fond  128 canaux                |               |
          |    remontée                        |               |
        32x32  dec2   64 canaux  <-- concat ---+               |
          |    remontée                                        |
        64x64  dec1   32 canaux  <-- concat --------------------+
          |
        tête conv 1x1 -> 25 canaux

    POURQUOI UN U-NET ET PAS UN EMPILEMENT PLAT COMME LE DnCNN.
    La covariance du résidu dépend du contexte : la texture des cheveux, un
    contour de paupière et une joue lisse n'ont pas la même statistique. Il
    faut donc voir large. Un empilement plat de convolutions 3x3 gagne 2 pixels
    de champ réceptif par couche ; les sous-échantillonnages en gagnent le
    double à chaque niveau, pour bien moins de calcul.

    CHAMP RÉCEPTIF, le long du chemin le plus profond :
        enc1  ->  5      pool  ->  6      enc2  -> 14      pool -> 16
        fond  -> 32      dec2  -> 44      dec1  -> 50
    Soit environ 50 x 50 pixels sur une image de 64, et 32 x 32 dès le fond du
    U. Comparable au 35 x 35 du DnCNN, ce qui est cohérent : les deux réseaux
    doivent reconnaître les mêmes structures.

    ORDRE DES 25 CANAUX — c'est le contrat avec loss.py, à ne pas casser :
        canal 0        -> log_diag, le log du terme diagonal l_ii
        canaux 1 à 24  -> offdiag, dans l'ORDRE EXACT de causal_offsets(f=7),
                          donc offdiag[:, i, k] = l_ij avec
                          j = neighbor_idx[i, k].
    Une permutation de ces 24 canaux ne fait pas planter le code : elle donne
    juste un modèle qui apprend une matrice fausse, silencieusement.

    Aucun clamp ici. `log_diag.clamp(-10, 10)` appartient à loss.py, un seul
    endroit, sinon on ne sait plus lequel des deux protège quoi.
    """

    def __init__(self, canaux=CANAUX, init_log_diag=INIT_LOG_DIAG,
                 diagonale_seule=False):
        super().__init__()
        c1, c2, c3 = canaux
        self.diagonale_seule = diagonale_seule
        self.init_log_diag = init_log_diag

        # Descente
        self.enc1 = Bloc(1, c1)
        self.enc2 = Bloc(c1, c2)
        self.fond = Bloc(c2, c3)
        self.pool = nn.MaxPool2d(2)

        # Remontée. Après chaque concaténation le nombre de canaux double,
        # d'où les `2 * c` en entrée des blocs du décodeur.
        self.up2 = _remontee(c3, c2)
        self.dec2 = Bloc(2 * c2, c2)
        self.up1 = _remontee(c2, c1)
        self.dec1 = Bloc(2 * c1, c1)

        # Tête : une convolution 1x1. Les 25 valeurs d'un pixel sont lues dans
        # les canaux à cet emplacement, sans mélange spatial supplémentaire —
        # tout le contexte a déjà été agrégé par le U.
        self.tete = nn.Conv2d(c1, N_SORTIES, 1, bias=True)
        self._initialiser()

    def _initialiser(self):
        """
        Kaiming partout, SAUF la tête, initialisée à zéro avec un biais sur le
        canal 0.

        Conséquence : au tout premier passage le réseau prédit
        log_diag = init_log_diag partout et offdiag = 0, c'est-à-dire une
        gaussienne isotrope d'écart-type std(r). Le modèle démarre donc sur la
        réponse la plus bête possible mais déjà à la bonne échelle, et n'a plus
        qu'à s'en écarter. Sans cela les premières epochs partent d'une
        précision arbitraire et la NLL fait n'importe quoi.

        Effet de bord à connaître : une tête à poids nuls ne rétropropage
        AUCUN gradient vers le tronc à la première itération (le gradient
        d'entrée de la tête vaut W^T g = 0). Seuls les poids de la tête
        bougent. Dès qu'ils sont non nuls, le tronc apprend normalement. C'est
        attendu, pas un bug — le test du __main__ vérifie exactement ça.
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
        mu : [B, 1, 64, 64] ou [B, 4096]
        renvoie (log_diag [B, 4096], offdiag [B, 4096, 24])
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

        # Ordre raster : i = ligne * 64 + colonne. C'est exactement l'ordre de
        # `view` sur les deux dernières dimensions, aucune permutation à faire.
        log_diag = sortie[:, 0].reshape(batch, N_PIXELS)

        if self.diagonale_seule:
            # Référence diagonale (§6.2) : le même réseau, mais offdiag forcé à
            # zéro. L'écart de NLL avec le modèle complet chiffre ce que la
            # structure apporte. Sans zéro gradient à remonter, les 24 canaux
            # correspondants restent simplement inertes.
            offdiag = torch.zeros(batch, N_PIXELS, N_VOISINS,
                                  device=sortie.device, dtype=sortie.dtype)
        else:
            # [B, 24, 64, 64] -> [B, 24, 4096] -> [B, 4096, 24]
            # `.contiguous()` parce que la loss indexe ce tenseur voisin par
            # voisin ; une vue transposée y coûterait plus cher que la copie.
            offdiag = (sortie[:, 1:]
                       .reshape(batch, N_VOISINS, N_PIXELS)
                       .permute(0, 2, 1)
                       .contiguous())

        return log_diag, offdiag

    def champ_receptif(self):
        """Champ réceptif approché, en pixels. Voir le tableau du docstring."""
        return 50


# --------------------------------------------------------------------------
# Test du module
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

    # --- formes ----------------------------------------------------------
    mu = torch.randn(4, 1, TAILLE_IMAGE, TAILLE_IMAGE)
    log_diag, offdiag = modele(mu)
    print("  entrée %s -> log_diag %s, offdiag %s"
          % (tuple(mu.shape), tuple(log_diag.shape), tuple(offdiag.shape)))
    assert log_diag.shape == (4, N_PIXELS)
    assert offdiag.shape == (4, N_PIXELS, N_VOISINS)

    # L'entrée aplatie doit donner exactement le même résultat.
    log_diag_plat, _ = modele(mu.view(4, N_PIXELS))
    assert torch.allclose(log_diag, log_diag_plat), "entrée aplatie divergente"

    # --- initialisation --------------------------------------------------
    # Sortie constante et à la bonne échelle : gaussienne isotrope d'écart-type
    # std(r). C'est ce que la tête à zéro doit garantir.
    assert torch.allclose(log_diag, torch.full_like(log_diag, INIT_LOG_DIAG)), \
        "log_diag non constant à l'initialisation"
    assert offdiag.abs().max() == 0, "offdiag non nul à l'initialisation"
    sigma_init = math.exp(-INIT_LOG_DIAG)
    print("  à l'init : sigma prédit = %.4f, résidu mesuré = %.4f"
          % (sigma_init, STD_RESIDU))

    # --- gradients -------------------------------------------------------
    # Faux critère, uniquement pour vérifier que les gradients circulent. La
    # vraie NLL est dans loss.py.
    cible_d = torch.randn(4, N_PIXELS)
    cible_o = torch.randn(4, N_PIXELS, N_VOISINS)

    def faux_critere(modele):
        ld, od = modele(mu)
        return ((ld - cible_d) ** 2).mean() + ((od - cible_o) ** 2).mean()

    perte = faux_critere(modele)
    perte.backward()

    # Attendu : la tête reçoit du gradient, le tronc n'en reçoit pas encore.
    g_tete = modele.tete.weight.grad.abs().max().item()
    g_tronc = modele.enc1.bloc[0].weight.grad.abs().max().item()
    print("  passe 1 : |grad| tête = %.3e, tronc = %.3e" % (g_tete, g_tronc))
    assert g_tete > 0, "la tête ne reçoit pas de gradient"
    assert g_tronc == 0, "le tronc ne devrait pas encore bouger (tête à zéro)"

    # Après un pas, la tête n'est plus nulle : le tronc doit apprendre.
    opt = torch.optim.Adam(modele.parameters(), lr=1e-3)
    opt.step()
    opt.zero_grad()
    faux_critere(modele).backward()
    g_tronc = modele.enc1.bloc[0].weight.grad.abs().max().item()
    print("  passe 2 : |grad| tronc = %.3e" % g_tronc)
    assert g_tronc > 0, "le tronc n'apprend pas après le premier pas"

    # --- référence diagonale ---------------------------------------------
    diag_seul = SparseCholeskyNet(diagonale_seule=True)
    ld, od = diag_seul(mu)
    assert od.shape == (4, N_PIXELS, N_VOISINS) and od.abs().max() == 0
    print("  référence diagonale : offdiag identiquement nul   OK")

    # --- coût mémoire d'un batch réel ------------------------------------
    octets = 64 * N_PIXELS * N_SORTIES * 4
    print("  sortie pour un batch de 64 : %.0f Mo" % (octets / 1e6))
    print("  OK")