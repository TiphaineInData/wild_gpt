import torch.nn as nn
from layers import SwiGLU
import torch
import torch.nn.functional as F
import torch.nn as nn

class MOELayer(nn.Module):
    def __init__(self, config):
        """
        Couche Mixture of Experts (MoE) avec top-k routing.

        Cette couche contient :
        - Un routeur linéaire (nn.Linear) qui attribue à chaque token un score pour chaque expert. 
          Il agit comme un classifieur, en sortie il donne les log-probabilités d'activation des experts.
        - Une liste d'experts, chacun étant un petit MLP (Multi-Layer Perceptron) de type SwiGLU.
          Ces MLP utilisent aussi des nn.Linear, mais ici pour transformer les données.
        - Un expert partagé optionnel, commun à tous les tokens, activé pour chaque passage.
        - Un mécanisme de balance de charge sans perte auxiliaire, qui adapte dynamiquement les biais du routeur.

        Pour chaque token, seuls 'top_k' experts sont activés parmi 'n_experts'. Cela permet
        d'augmenter la capacité du modèle tout en gardant un coût de calcul constant.
        """
        super().__init__()
        self.config = config
        self.n_experts = config.n_experts # Nombre total d'experts disponibles dans la couche
        self.top_k = config.n_experts_per_token # Nombre d'experts sélectionnés pour traiter chaque token (top-k routing) : haque token est routé vers ses top_k experts les plus "pertinents".
        self.n_embd = config.n_embd # Taille de l'embedding (dimension des vecteurs d'entrée/sortie)

        # Routeur
        self.router = nn.Linear(config.n_embd, config.n_experts, bias=False)

        # Expert MLPs

        self.experts = nn.ModuleList([
            SwiGLU(
                config.n_embd,
                config.expert_intermediate_size,
                config.n_embd,
                config.bias
            ) for _ in range(config.n_experts)
        ])

        # Shared expert
        if config.use_shared_expert:
            self.shared_expert = SwiGLU(
                config.n_embd,
                config.shared_expert_intermediate_size,
                config.n_embd,
                config.bias
            )
        else:
            self.shared_expert = None

        # Mécanisme de balance de charge sans perte auxiliaire :
        # un biais par expert est appris passivement pour équilibrer leur activation.
        # Ce biais est mis à jour manuellement durant l'entraînement (pas via backpropagation).
        self.register_buffer('expert_bias', torch.zeros(config.n_experts))  # Biais par expert
        self.bias_update_rate = 0.001  # Taux de mise à jour des biais (valeur fixe, petit déplacement à chaque pas)

    def forward(self, x):
        # x : [batch_size, seq_len, hidden_size]
        batch_size, seq_len, hidden_size = x.shape

        # Aplatit le batch et la séquence pour traiter tous les tokens à la fois
        x_flat = x.view(-1, hidden_size)  # [batch_size * seq_len, hidden_size]

        # === Phase de routage ===
        # Le routeur donne un score pour chaque expert, par token
        router_logits = self.router(x_flat) + self.expert_bias  # [batch_size * seq_len, n_experts]

        # Sélection des top-k experts avec les plus hauts scores pour chaque token
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)  # [n_tokens, top_k]

        # On initialise une matrice de poids à 0, puis on remplit uniquement les top-k avec la softmax
        routing_weights = torch.zeros_like(router_logits)  # [n_tokens, n_experts]
        routing_weights.scatter_(
            -1,
            top_k_indices,
            F.softmax(top_k_logits, dim=-1)  # softmax sur les top-k uniquement
        )

        # === Phase de passage dans les experts ===
        output = torch.zeros_like(x_flat)  # initialisation de la sortie [n_tokens, hidden_size]
        expert_usage = torch.zeros(self.n_experts, device=x.device)  # compteur d'utilisation des experts

        # Pour chaque expert : on applique l'expert aux tokens qui l'ont sélectionné
        for expert_idx in range(self.n_experts):
            # Mask des tokens qui ont sélectionné cet expert (au moins une fois dans leur top-k)
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)  # [n_tokens]
            expert_usage[expert_idx] = expert_mask.sum().float()  # nombre de tokens traités par cet expert

            if expert_mask.any():
                # On extrait les entrées concernées
                expert_input = x_flat[expert_mask]  # [n_selected_tokens, hidden_size]

                # On passe ces entrées dans l'expert
                expert_output = self.experts[expert_idx](expert_input)  # [n_selected_tokens, hidden_size]

                # On applique les poids de routage correspondants
                weights = routing_weights[expert_mask, expert_idx].unsqueeze(-1)  # [n_selected_tokens, 1]
                output[expert_mask] += expert_output * weights  # on ajoute à la sortie pondérée

        # === Expert partagé optionnel ===
        if self.shared_expert is not None:
            shared_output = self.shared_expert(x_flat)  # [n_tokens, hidden_size]
            output += shared_output  # ajout de la sortie de l'expert partagé

        # === Équilibrage des experts (sans perte auxiliaire) ===
        if self.training:
            with torch.no_grad():  # pas de backprop ici
                avg_usage = expert_usage.mean()
                for i in range(self.n_experts):
                    if expert_usage[i] > avg_usage:
                        self.expert_bias[i] -= self.bias_update_rate
                    else:
                        self.expert_bias[i] += self.bias_update_rate

        # On remet la forme d'origine [batch_size, seq_len, hidden_size]
        return output.view(batch_size, seq_len, hidden_size)
