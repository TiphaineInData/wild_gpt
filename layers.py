"""
RMSNorm : Root Mean Square Layer Normalization

Pourquoi normaliser entre chaque couche :
------------------------------------------
Lors de l'entraînement, les activations des neurones peuvent devenir très grandes ou très petites.
Cela provoque :
- soit des gradients explosifs (instabilité),
- soit des gradients nuls (blocage de l'apprentissage).

La normalisation régule l’échelle des activations pour :
- stabiliser la descente de gradient,
- faciliter la convergence,
- rendre le modèle plus robuste et plus facile à entraîner.

Pourquoi RMSNorm plutôt que LayerNorm :
----------------------------------------
LayerNorm centre les activations (soustraction de la moyenne) **et** ajuste leur échelle.

Avec MoE (Mixture of Experts), chaque token passe par un sous-ensemble d'experts différents.
Soustraire la moyenne (comme dans LayerNorm) peut :
- casser la diversité entre ces chemins dynamiques,
- rendre les activations trop homogènes.

RMSNorm ne centre pas les vecteurs :
- il garde la direction des activations,
- il ajuste seulement leur taille globale (norme),
- il est plus rapide (pas de calcul de moyenne),
- et il est plus stable dans des architectures sparsifiées comme MoE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, ndim, eps=1e-6):
        """
        RMSNorm = Root Mean Square Layer Normalization.

        Ne centre pas les activations (pas de soustraction de moyenne),
        mais les remet à l’échelle en divisant par la racine de la moyenne des carrés.
        Moins coûteux que LayerNorm et plus stable avec des architectures sparsifiées comme MoE.

        Arguments :
        - ndim : taille de la dernière dimension (embedding dim)
        - eps : petite constante pour éviter la division par zéro
        """
        super().__init__() # hérite de nn.Module
        self.eps = eps  # évite une division par 0
        self.weight = nn.Parameter(torch.ones(ndim))  # vecteur entraînable pour chaque dimension, initialisé à 1

    def forward(self, x):
        # x : (batch, seq_len, embedding_dim)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()  # racine de la moyenne des carrés
        x_normed = x / rms  # division élément par élément
        return self.weight * x_normed  # scaling appris

class RotaryEmbedding(nn.Module):
    """
    RotaryEmbedding : génère les valeurs cosinus et sinus utilisées pour appliquer un encodage positionnel rotatif (RoPE).

    RoPE est une méthode d'encodage positionnel pour les modèles Transformer.
    Contrairement aux méthodes classiques (où on ajoute un vecteur de position), RoPE modifie directement les vecteurs
    Q et K de l'attention en les faisant tourner (par paire de dimensions) selon leur position dans la séquence.

    Ce module ne modifie pas les vecteurs lui-même. Il produit simplement deux matrices (`cos`, `sin`) qui sont utilisées
    plus tard dans l’attention pour injecter la position dans les vecteurs.

    Pourquoi utiliser RoPE :
    - Meilleure gestion des longues séquences,
    - Plus stable que les encodages additifs,
    - Compatible avec des architectures comme Mixture of Experts (MoE),
    - Préserve la structure directionnelle des vecteurs, utile pour l’attention.

    Arguments :
    - dim (int) : taille du vecteur à encoder (souvent la dimension d’une tête d’attention)
    - max_seq_len (int) : longueur maximale de séquence prévue (par défaut : 2048)

    Méthode forward(x, seq_len) :
    - x : n’importe quel tenseur (sert à récupérer le device)
    - seq_len : longueur réelle de la séquence (optionnel, sinon déduite de x)
    - Retourne : deux matrices (cos, sin) de forme (seq_len, dim // 2)
    """
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]  # récupère la longueur de séquence à partir de x
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)  # produit la matrice des fréquences positionnelles
        cos, sin = freqs.cos(), freqs.sin()
        return cos, sin
    
def apply_rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class SwiGLU(nn.Module):
    """
    Implémentation de SwiGLU (Swish-Gated Linear Unit), une variante moderne du MLP
    utilisée dans les architectures Transformer (PaLM, LLaMA, DeepSeek...).

    Fonctionnement :
    - L'entrée x est projetée deux fois :
        • une première projection crée un vecteur appelé "porte" (gate)
        • une deuxième projection produit le "contenu à transmettre" (up)
    - La porte passe ensuite dans la fonction d’activation SiLU (Swish), appliquée à **chaque élément**
      → Cela donne un vecteur de coefficients souples, qui agissent comme
        des **amplificateurs ou des filtres doux** sur le contenu.
    - Ce vecteur activé est multiplié **élément par élément** avec le contenu `up`
      → Chaque valeur du contenu est alors **boostée, atténuée, annulée ou inversée**, 
        selon ce que décide la porte.
    - Le résultat est enfin projeté vers la taille de sortie finale.

    Résultat : une transformation non-linéaire dynamique, plus expressive qu’un MLP classique.

    Args:
        in_features (int): Taille du vecteur d'entrée
        hidden_features (int): Taille intermédiaire pour les projections gate et up
        out_features (int): Taille de sortie finale
        bias (bool): Utilise ou non un biais dans les couches linéaires
    """

    def __init__(self, in_features, hidden_features, out_features, bias=True):
        super().__init__()

        # Projette l'entrée pour créer un vecteur "porte" qui sera activé (SiLU)
        self.gate_proj = nn.Linear(in_features, hidden_features, bias=bias)

        # Projette l'entrée pour créer un vecteur "contenu principal"
        self.up_proj = nn.Linear(in_features, hidden_features, bias=bias)

        # Projette le signal filtré vers la dimension de sortie attendue
        self.down_proj = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        """
    Implémentation de SwiGLU (Swish-Gated Linear Unit), utilisée dans les Transformers modernes 
    comme alternative plus expressive au MLP classique.

    Fonctionnement (sur un seul vecteur x) :

    - On part d’un vecteur `x` de taille `in_features` (par ex. 768)
    - On le projette deux fois :
        • `gate_proj(x)` produit un **vecteur porte** (gate), de taille `hidden_features` (par ex. 2048)
        • `up_proj(x)` produit un **vecteur de contenu** (up), aussi de taille `hidden_features`
    - La porte est activée avec la fonction `SiLU` (Swish), appliquée **élément par élément**
      → Cela donne un vecteur de coefficients souples qui vont **amplifier, réduire ou annuler** le contenu
    - Le contenu est ensuite modulé dynamiquement : `filtered = SiLU(gate) * up`
    - Enfin, on passe ce vecteur modulé dans `down_proj`, qui ramène la taille de `hidden_features` à `out_features` 
      (souvent égal à `in_features` pour rester compatible avec le reste du modèle)

    Résultat : une transformation contrôlée, plus fine et plus expressive qu'un MLP classique.

    Args:
        in_features (int): Taille du vecteur d'entrée
        hidden_features (int): Taille intermédiaire (souvent plus grande, ex: 4×)
        out_features (int): Taille de sortie finale (souvent égale à in_features)
        bias (bool): Si True, ajoute un biais dans les couches linéaires
        """

        # Construction de la porte et du contenu
        gate = self.gate_proj(x)  # vecteur de filtrage (peut avoir des valeurs < 0 ou > 1)
        up = self.up_proj(x)      # vecteur de signal principal

        # Activation douce de la porte avec SiLU, puis modulation du signal
        # Chaque élément de `up` est multiplié par un coefficient donné par la porte activée
        filtered = F.silu(gate) * up

        # Projection finale du contenu modulé vers la sortie
        return self.down_proj(filtered)
