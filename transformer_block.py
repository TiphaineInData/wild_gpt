import torch.nn as nn
from .layers import RMSNorm
from .attention import MultiHeadLatentAttention
from .moe import MOELayer

class WildBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Première normalisation (avant attention)
        # RMSNorm est une alternative à LayerNorm, sans soustraction de la moyenne
        self.ln_1 = RMSNorm(config.n_embd)

        # Mécanisme d'attention maison (multi-head latent attention avec RoPE par ex)
        self.attn = MultiHeadLatentAttention(config)

        # Deuxième normalisation (avant MLP)
        self.ln_2 = RMSNorm(config.n_embd)

        # MLP remplacé ici par une couche Mixture of Experts (MoE)
        # Elle contient plusieurs MLP appelés "experts", avec un router
        self.mlp = MOELayer(config)

    def forward(self, x):
        # Skip connection + attention
        # Normalisation avant attention (PreNorm), puis résidu
        x = x + self.attn(self.ln_1(x))

        # Skip connection + MoE
        # Normalisation avant MoE (PreNorm), puis résidu
        x = x + self.mlp(self.ln_2(x))

        return x
