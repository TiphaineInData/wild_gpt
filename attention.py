import torch.nn as nn
import torch
import math
from .layers import RMSNorm, RotaryEmbedding, apply_rope
import torch.nn.functional as F

class MultiHeadLatentAttention(nn.Module):
    """
    Implémente une attention multi-têtes avec compression/décompression low-rank (type LoRA), 
    RMSNorm, et positionnement rotary (RoPE).

    Chaque token est représenté par un vecteur de taille n_embd.
    Ce vecteur est découpé en n_head sous-vecteurs (de taille head_dim), chacun traité 
    par une tête d’attention dans un sous-espace différent.

    RoPE (Rotary Positional Embedding) encode la position en appliquant des rotations 
    (sinus/cosinus) à une sous-partie du vecteur Q et K (rope_dim), au lieu d’ajouter 
    directement un vecteur de position.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # Dimensions de compression (low-rank)
        self.kv_lora_rank = config.kv_lora_rank  # dim réduite pour K/V
        self.q_lora_rank = config.q_lora_rank    # dim réduite pour Q
        self.rope_dim = config.rope_dim          # nombre de dimensions sur lesquelles on applique RoPE

        # Compression K/V : on réduit n_embd → kv_lora_rank pour alléger le calcul attention
        self.kv_proj = nn.Linear(self.n_embd, self.kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank)

        # Décompression K/V : on reconstruit des vecteurs K et V utilisables par les têtes
        self.k_decompress = nn.Linear(self.kv_lora_rank, self.n_head * self.head_dim, bias=False)
        self.v_decompress = nn.Linear(self.kv_lora_rank, self.n_head * self.head_dim, bias=False)

        # Compression Q : même idée, mais avec sa propre dimension q_lora_rank
        self.q_proj = nn.Linear(self.n_embd, self.q_lora_rank, bias=False)

        # Décompression Q : retourne à la forme multi-têtes
        self.q_decompress = nn.Linear(self.q_lora_rank, self.n_head * self.head_dim, bias=False)

        # RoPE projections
        self.k_rope_proj = nn.Linear(self.n_embd, self.n_head * self.rope_dim, bias=False)
        self.q_rope_proj = nn.Linear(self.q_lora_rank, self.n_head * self.rope_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(self.n_head * self.head_dim, self.n_embd, bias=config.bias)

        # Dropout sur les poids d’attention (masque des liens entre tokens)
        self.attn_dropout = nn.Dropout(config.dropout)

        # Dropout classique sur la sortie finale du bloc (régularisation globale)
        self.resid_dropout = nn.Dropout(config.dropout)

        # RoPE
        self.rope = RotaryEmbedding(self.rope_dim, config.block_size)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            )
        )

    def forward(self, x):
        # x : (B, T, C)
        # B = batch size (nb de séquences dans le batch)
        # T = longueur de la séquence (nb de tokens)
        # C = dimension d'embedding (n_embd)
        B, T, C = x.size()

        # Phase de compression : on réduit la taille des vecteurs K/V et Q
        kv_compressed = self.kv_norm(self.kv_proj(x))  # (B, T, kv_lora_rank)
        q_compressed = self.q_proj(x)                  # (B, T, q_lora_rank)

        # Décompression : on recrée des vecteurs K, V, Q exploitables pour l’attention
        k_content = self.k_decompress(kv_compressed)   # (B, T, n_head * head_dim)
        v = self.v_decompress(kv_compressed)           # (B, T, n_head * head_dim)
        q_content = self.q_decompress(q_compressed)    # (B, T, n_head * head_dim)

        # Composantes pour le positionnement rotatif (RoPE)
        k_rope = self.k_rope_proj(x)                   # (B, T, n_head * rope_dim)
        q_rope = self.q_rope_proj(q_compressed)        # (B, T, n_head * rope_dim)

        # Reshape pour obtenir la forme multi-head : (B, n_head, T, head_dim)
        k_content = k_content.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q_content = q_content.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k_rope = k_rope.view(B, T, self.n_head, self.rope_dim).transpose(1, 2)
        q_rope = q_rope.view(B, T, self.n_head, self.rope_dim).transpose(1, 2)

        # Application du positionnement RoPE sur une sous-partie des vecteurs Q et K
        cos, sin = self.rope(x, T)  # sinus/cosinus pré-calculés selon la position
        q_rope = apply_rope(q_rope, cos, sin)
        k_rope = apply_rope(k_rope, cos, sin)

        # Concaténation de la partie "contenu" et "positionnelle"
        q = torch.cat([q_content, q_rope], dim=-1)     # (B, n_head, T, head_dim + rope_dim)
        k = torch.cat([k_content, k_rope], dim=-1)

        # Calcul des scores d’attention (produit scalaire entre Q et K)
        scale = 1.0 / math.sqrt(q.size(-1))            # normalisation
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, n_head, T, T)

        # Application du masque causal : un token ne peut voir que ses positions passées
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))

        # Poids d’attention : softmax + dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Application de l’attention : combinaison pondérée des vecteurs V
        out = torch.matmul(attn_weights, v)            # (B, n_head, T, head_dim)

        # Remise en forme + projection finale + dropout résiduel
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        out = self.resid_dropout(self.o_proj(out))     # (B, T, n_embd)

        return out

