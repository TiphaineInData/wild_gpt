import torch
import math
import os
import wandb
import time
from tqdm.auto import tqdm
from contextlib import nullcontext
from .config import Wild_GPT_config
from .model import Wild_GPT
from .data_loader import get_batch, estimate_loss
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def compute_perplexity(loss: float) -> float:
    return math.exp(loss) if loss is not None and loss < 100 else float("inf")

def train_model():
    config = Wild_GPT_config(
        vocab_size=50257,
        block_size=1024,
        n_layer=8,
        n_head=8,
        n_embd=512,
        kv_lora_rank=128,
        q_lora_rank=192,
        n_experts=8,
        n_experts_per_token=2,
        mtp_num_heads=1,
        dropout=0.15
    )

    # Entraînement
    learning_rate = 3e-4
    max_iters = 20000
    warmup_steps = 3000
    min_lr = 1e-5
    eval_interval = 1800
    eval_iters = 250
    batch_size = 32
    gradient_accumulation_steps = 8
    patience = 12

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.cuda.amp.autocast(dtype=ptdtype)

    wandb.init(project="wild_gpt_full_train", config=config.__dict__)
    torch.manual_seed(42)

    model = Wild_GPT(config).to(device)

    # Charger les poids depuis DeepSeek sans embeddings si tokenizer ≠
    pretrained_path = "best_deepseek_v3.pt"
    state_dict = torch.load(pretrained_path, map_location=device)
    model_dict = model.state_dict()
    filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and "wte" not in k and "wpe" not in k}
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict, strict=False)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Wild_GPT model with {total_params:,} parameters")
    wandb.log({"total_parameters": total_params})

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-9)
    scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))

    model.train()
    best_val_loss = float('inf')
    best_model_path = "models/Wild_GPT.pt"
    best_optimizer_path = "models/Wild_GPT_optimizer.pt"
    no_improvement_steps = 0

    last_backup_time = time.time()
    backup_interval = 7200  # 2h

    for step in tqdm(range(max_iters)):
        X, y = get_batch("train", config, batch_size, device_type, device)
        with ctx:
            _, total_loss, main_loss, mtp_loss = model(X, y)
            loss = total_loss / gradient_accumulation_steps
            scaler.scale(loss).backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step < warmup_steps:
            lr = learning_rate * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / (max_iters - warmup_steps)
            lr = min_lr + (learning_rate - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        wandb.log({
            "step": step,
            "total_loss": total_loss.item(),
            "main_loss": main_loss.item(),
            "mtp_loss": mtp_loss.item() if mtp_loss else 0.0,
            "learning_rate": lr
        })

        # 💾 Sauvegarde automatique toutes les 2h même sans amélioration
        if time.time() - last_backup_time > backup_interval:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            torch.save(model.state_dict(), f"models/Wild_GPT_backup_{timestamp}.pt")
            with open("models/training_log.txt", "a", encoding="utf-8") as f:
                f.write(f"📦 Backup automatique à {timestamp} (step {step})\n")
            last_backup_time = time.time()

        if step % eval_interval == 0 and step > 0:
            losses = estimate_loss(model, config, eval_iters, batch_size, device_type, device, ctx)
            train_perplexity = compute_perplexity(losses['train'])
            val_perplexity = compute_perplexity(losses['val'])

            print(f"Step {step}: train {losses['train']:.4f} (PPL {train_perplexity:.2f}), val {losses['val']:.4f} (PPL {val_perplexity:.2f})")

            wandb.log({
                "val_loss": losses['val'],
                "train_loss_eval": losses['train'],
                "val_perplexity": val_perplexity,
                "train_perplexity": train_perplexity
            })

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                no_improvement_steps = 0
                torch.save(model.state_dict(), best_model_path)
                torch.save(optimizer.state_dict(), best_optimizer_path)
                torch.save(model, "models/Wild_GPT_full.pt")
                wandb.log({"best_val_loss": best_val_loss})
            else:
                no_improvement_steps += 1
                if no_improvement_steps >= patience:
                    print("Early stopping triggered.")
                    wandb.log({"early_stopping_step": step})
                    with open("models/training_log.txt", "a", encoding="utf-8") as f:
                        f.write(f"⛔ Early stopping triggered at step {step}\n")
                    break

    # Sauvegarde finale
    torch.save(model, "models/Wild_GPT_final.pt")
    wandb.finish()
    print("Training complete.")
    return model, config

if __name__ == "__main__":
    train_model()
