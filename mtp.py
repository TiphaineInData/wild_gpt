import torch.nn as nn
from .layers import RMSNorm
from .attention import MultiHeadLatentAttention
from .moe import MOELayer


"""
    Module de prédiction enrichie utilisé dans le cadre du Multi-Token Prediction (MTP).

    Cette tête de prédiction combine deux sources d'information :
    - `prev_hidden` : l'état caché produit par le modèle à une position donnée,
    - `future_token_embed` : l'embedding d'un ou plusieurs tokens cibles (tokens futurs connus pendant l'entraînement).

    L'objectif est de produire une représentation plus riche qui tienne compte de la supervision explicite 
    via les tokens cibles, en amont de la prédiction.

    Architecture :
    1. Normalisation RMS des deux entrées.
    2. Concatenation des deux vecteurs (prev_hidden + future_token_embed), suivie d'une projection linéaire.
    3. Passage dans une couche d'attention latente (MultiHeadLatentAttention) avec résidu.
    4. Passage dans une couche MLP de type Mixture-of-Experts (MoELayer), également avec résidu.

    Ce bloc peut être utilisé comme une tête intermédiaire avant la tête de langage finale (lm_head),
    dans le cadre d'un apprentissage renforcé par la cible (e.g., multi-token loss ou alignement renforcé).

    Paramètres
    ----------
    config : object
        Configuration du modèle contenant les dimensions d'embedding et les paramètres d'architecture.
    depth : int
        Indique combien de positions futures cette tête est censée utiliser pour apprendre à prédire.
        Il est utilisé par le modèle principal pour sélectionner les `future_token_embeds` décalés de `depth`.
        Dans cette classe, il est stocké mais n'influence pas le calcul (peut être utilisé à des fins de log ou d'adaptation future).

    Entrées
    -------
    prev_hidden : torch.Tensor de forme (batch_size, seq_len, n_embd)
        États cachés produits par les blocs transformer précédents.
    future_token_embed : torch.Tensor de forme (batch_size, seq_len, n_embd)
        Embeddings des tokens cibles (labels), extraits via la table d'embedding du modèle.

    Sortie
    ------
    hidden : torch.Tensor de forme (batch_size, seq_len, n_embd)
        Représentation enrichie combinant contexte passé et cible supervisée, prête pour la prédiction finale.
"""
import torch
import torch.nn as nn
from .layers import RMSNorm
from .attention import MultiHeadLatentAttention
from .moe import MOELayer

class MultiTokenPredictionHead(nn.Module):
    def __init__(self, config, depth):
        super().__init__()
        self.depth = depth  # Stocké à titre indicatif, mais pas utilisé dans ce bloc
        self.n_embd = config.n_embd

        # Projette la concaténation [prev_hidden ; future_token_embed] vers un vecteur de taille n_embd
        # => permet au modèle d’apprendre une combinaison non triviale entre contexte et supervision
        self.combine_proj = nn.Linear(2 * config.n_embd, config.n_embd, bias=config.bias)

        # Normalisation RMS des deux entrées avant concaténation
        self.norm1 = RMSNorm(config.n_embd)  # pour prev_hidden
        self.norm2 = RMSNorm(config.n_embd)  # pour future_token_embed

        # Composants transformeur appliqués après combinaison
        self.attn = MultiHeadLatentAttention(config)  # attention spéciale (souvent avec LoRA, RoPE...)
        self.attn_norm = RMSNorm(config.n_embd)       # normalisation avant attention

        self.mlp = MOELayer(config)                   # MLP de type Mixture-of-Experts
        self.mlp_norm = RMSNorm(config.n_embd)        # normalisation avant MLP

    def forward(self, prev_hidden, future_token_embed):
        """
        Combine les deux sources d'information (contexte + cible) pour produire
        une représentation enrichie.

        Args:
            prev_hidden: (batch, seq_len, n_embd) — état produit par les blocs transformer
            future_token_embed: (batch, seq_len, n_embd) — embedding du token cible à horizon t + depth

        Returns:
            hidden: (batch, seq_len, n_embd) — vecteur fusionné, enrichi, prêt pour la prédiction finale
        """

        # Étape 1 — Normalise séparément les deux entrées
        prev_norm = self.norm1(prev_hidden)
        future_norm = self.norm2(future_token_embed)

        # Étape 2 — Concatène les vecteurs le long de la dimension d’embedding : [prev ; future]
        combined = torch.cat([prev_norm, future_norm], dim=-1)  # → shape (batch, seq_len, 2 * n_embd)

        # Étape 3 — Applique une projection linéaire pour ramener à n_embd
        hidden = self.combine_proj(combined)

        # Étape 4 — Attention avec résidu : hidden ← hidden + attn(norm(hidden))
        hidden = hidden + self.attn(self.attn_norm(hidden))

        # Étape 5 — MLP MoE avec résidu : hidden ← hidden + mlp(norm(hidden))
        hidden = hidden + self.mlp(self.mlp_norm(hidden))

        return hidden
