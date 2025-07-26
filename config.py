from dataclasses import dataclass

@dataclass
class Wild_GPT_config:
    # Model architecture
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512

    # MLA configuration
    kv_lora_rank: int = 128
    q_lora_rank: int = 192
    rope_dim: int = 32

    # MoE configuration
    n_experts: int = 8
    n_experts_per_token: int = 2
    expert_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 768
    use_shared_expert: bool = True

    # MTP configuration
    mtp_num_heads: int = 1

    # Training parameters
    dropout: float = 0.15
    bias: bool = True
    aux_loss_weight: float = 0.0
    mtp_loss_weight: float = 0.3
