import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformer_block import WildBlock
from layers import RMSNorm
from mtp import MultiTokenPredictionHead

class Wild_GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # word token Embedding : matrice d'embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # word position embedding : matrice de position
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # liste de blocs de transformeurs (nbe : n_layer)
        self.h = nn.ModuleList([WildBlock(config) for _ in range(config.n_layer)])

        # normalisation finale appliquée à la sortie du dernier block avant la prédiction
        self.ln_f = RMSNorm(config.n_embd)

        # Tête de langage : projette la sortie du modèle vers un vecteur de taille vocab_size (logits)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # poids partagés : on impose que les poids d'entrée wte soient similaires aux poids de sortie lm_head
        # → réduit le nbe de params, améliore généralisation, modèle plus cohérent
        self.wte.weight = self.lm_head.weight

        # Multi-Token Prediction heads
        if config.mtp_num_heads > 0:
            self.mtp_heads = nn.ModuleList([
                MultiTokenPredictionHead(config, depth)
                for depth in range(1, config.mtp_num_heads + 1)
            ])
        else:
            self.mtp_heads = None

        # Initialise les poids sur tous les sous-modules
        self.apply(self._init_weights)

        # Parcourt tous les params du modèle et pr o_proj et down_proj car grosses dimensions (attention/MLP) : évite gradient exploding
        for pn, p in self.named_parameters():
            if pn.endswith('o_proj.weight') or pn.endswith('down_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """encode les index, prédit les logits, calcule les pertes et gère le MTP si il est activé"""
        device = idx.device  # CPU ou cuda
        b, t = idx.size()  # b = batch, t = longueur de la séquence (nbe tokens par ligne)

        if t == 0:
            # Séquence vide → on retourne des tenseurs vides compatibles
            empty_logits = torch.empty(b, 0, self.config.vocab_size, device=device)
            return empty_logits, None, None, None

        assert t <= self.config.block_size  # sécurité : on ne dépasse pas la longueur max

        # Token et Position Embedding
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # position [0, 1, ..., t-1]
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)  # sum token + position + dropout

        # Transformer blocks
        for block in self.h:
            x = block(x)

        # Norme finale
        x = self.ln_f(x)

        # Prédiction principale
        main_logits = self.lm_head(x)
        main_loss = None

        # si on a des cibles (donc en entraînement, pas en génération)
        if targets is not None:
            # on aplatit pour comparer chaque token
            main_loss = F.cross_entropy(
                main_logits.view(-1, main_logits.size(-1)),
                targets.view(-1),
                ignore_index=4
            )

        # Multi Token Prediction
        mtp_loss = None
        if self.mtp_heads is not None and targets is not None:
            mtp_losses = []
            current_hidden = x

            # Pour chaque profondeur (1, 2, 3, ...) et chaque tête MTP
            for depth, mtp_head in enumerate(self.mtp_heads, 1):
                if t > depth:
                    future_indices = idx[:, depth:]
                    future_embeds = self.wte(future_indices)

                    # padding avec le token de fin si trop court
                    if future_embeds.size(1) < current_hidden.size(1):
                        pad_size = current_hidden.size(1) - future_embeds.size(1)
                        eof_token_id = 4
                        eof_embed = self.wte(torch.full((1,), eof_token_id, device=device)).squeeze(0)
                        padding = eof_embed.expand(b, pad_size, self.config.n_embd)
                        future_embeds = torch.cat([future_embeds, padding], dim=1)

                    elif future_embeds.size(1) > current_hidden.size(1):
                        future_embeds = future_embeds[:, :current_hidden.size(1)]

                    current_hidden = mtp_head(current_hidden, future_embeds)
                    mtp_logits = self.lm_head(current_hidden)

                    # décalage pour la perte
                    if t > depth + 1:
                        shift_logits = mtp_logits[..., :-(depth + 1), :].contiguous()
                        shift_labels = targets[..., depth + 1:].contiguous()

                        if shift_labels.numel() > 0:
                            mtp_loss_single = F.cross_entropy(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1),
                                ignore_index=4
                            )
                            mtp_losses.append(mtp_loss_single)

            if mtp_losses:
                mtp_loss = torch.stack(mtp_losses).mean()

        # si on a des cibles (durant l'entraînement et pas la génération)
        if targets is not None:
            # Et si on a aussi calculé les pertes MTP
            if mtp_loss is not None:
                # On combine les pertes principales + MTP pondérée
                total_loss = main_loss + self.config.mtp_loss_weight * mtp_loss
                return main_logits, total_loss, main_loss, mtp_loss
            else:
                return main_logits, main_loss, main_loss, None
        else:
            # En génération : on ne garde que les logits du dernier token
            return main_logits[:, [-1], :], None, None, None

    @torch.no_grad()  # on désactive le calcul des gradients, car on ne fait pas d'entraînement ici
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # on refuse les entrées vides
        if idx.size(1) == 0:
            raise ValueError("La séquence d'entrée est vide. Impossible de générer.")

        # on va générer un token à la fois, jusqu'à en avoir ajouté max_new_tokens
        for _ in range(max_new_tokens):
            # si la séquence est trop longue, on tronque pour respecter block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # passage dans le modèle (on ne garde que les logits)
            logits, _, _, _ = self(idx_cond)

            # on récupère les logits du dernier token de la séquence
            logits = logits[:, -1, :] / temperature  # plus la température est haute, plus c'est aléatoire

            # si top_k est activé, on garde seulement les k tokens avec les meilleurs scores
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')  # les autres deviennent très improbables

            # on transforme les scores en probabilités
            probs = F.softmax(logits, dim=-1)

            # on tire au hasard un token en fonction des probabilités
            idx_next = torch.multinomial(probs, num_samples=1)

            # on ajoute ce nouveau token à la séquence d'entrée
            idx = torch.cat((idx, idx_next), dim=1)

        # on retourne la séquence complète (départ + nouveaux tokens générés)
        return idx
