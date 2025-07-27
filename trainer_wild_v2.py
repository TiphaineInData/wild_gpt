import torch
import math
import os
import wandb
import time
from tqdm.auto import tqdm
from contextlib import nullcontext
from config import Wild_GPT_config
from model import Wild_GPT
from data_loader import get_batch, estimate_loss
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def compute_perplexity(loss: float) -> float:
    return math.exp(loss) if loss is not None and loss < 100 else float("inf")

def train_model_v2():
    config = Wild_GPT_config(
        vocab_size=50257,
        block_size=1024,
        n_layer=8,
        n_head=8,
        n_embd=512,
        kv_lora_rank=128,
        q_lora_rank=192,
        n_experts=8,  # ✅ MÊME que v1 pour compatibilité
        n_experts_per_token=2,  # ✅ MÊME que v1 pour compatibilité
        mtp_num_heads=1,
        dropout=0.1  # ⬇️ Moins de dropout : 0.15 → 0.1
    )

    # Hyperparamètres optimisés pour continuation
    learning_rate = 1e-4  # ⬇️ LR plus petit pour fine-tuning : 3e-4 → 1e-4
    max_iters = 15000  # ⬇️ Moins d'itérations : 20k → 15k
    warmup_steps = 1000  # ⬇️ Warmup plus court : 3k → 1k
    min_lr = 5e-6  # ⬇️ LR min plus petit : 1e-5 → 5e-6
    eval_interval = 1200  # ⬇️ Eval plus fréquent : 1800 → 1200
    eval_iters = 250
    batch_size = 16  # ⬇️ Batch plus petit : 32 → 16
    gradient_accumulation_steps = 16  # ⬆️ Plus d'accumulation : 8 → 16
    patience = 15  # ⬆️ Plus de patience : 12 → 15

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.cuda.amp.autocast(dtype=ptdtype)

    wandb.init(project="wild_gpt_v2_train", config=config.__dict__)
    torch.manual_seed(42)

    model = Wild_GPT(config).to(device)

    # 🔥 CHARGER LES POIDS DE WILD_GPT V1 (votre modèle entraîné)
    checkpoint_path = "models/Wild_GPT.pt"  # Votre meilleur modèle v1
    
    # Créer le dossier models_v2 pour sauvegardes v2
    os.makedirs("models_v2", exist_ok=True)
    
    if os.path.exists(checkpoint_path):
        print(f"🚀 Loading Wild_GPT v1 checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Chargement direct - architectures identiques
        model.load_state_dict(checkpoint, strict=False)
        print(f"✅ Loaded Wild_GPT v1 weights successfully!")
        
        initial_val_loss = 6.70  # Votre meilleur résultat v1
        print(f"🎯 Starting from v1 val_loss: {initial_val_loss}")
        
    else:
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        print("🔄 Loading DeepSeek v3 base weights instead")
        
        # Fallback : charger DeepSeek si pas de checkpoint v1
        pretrained_path = "best_deepseek_v3.pt"
        state_dict = torch.load(pretrained_path, map_location=device)
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and "wte" not in k and "wpe" not in k}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict, strict=False)
        initial_val_loss = 11.0

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Wild_GPT v2 model with {total_params:,} parameters")
    wandb.log({"total_parameters": total_params, "initial_val_loss": initial_val_loss})

    # Optimiseur avec LR scheduling plus doux
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        betas=(0.9, 0.98),  # Beta2 plus élevé pour plus de stabilité
        weight_decay=0.05,  # Weight decay plus faible
        eps=1e-9
    )
    scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))

    model.train()
    best_val_loss = initial_val_loss  # Commencer depuis la perf v1
    best_model_path = "models_v2/Wild_GPT_v2.pt"
    best_optimizer_path = "models_v2/Wild_GPT_v2_optimizer.pt"
    no_improvement_steps = 0

    last_backup_time = time.time()
    backup_interval = 7200  # 2h

    print(f"🚀 Starting Wild_GPT v2 training...")
    print(f"🎯 Target: < 5.5 val_loss (current best: {best_val_loss})")

    for step in tqdm(range(max_iters)):
        X, y = get_batch("train", config, batch_size, device_type, device)
        with ctx:
            _, total_loss, main_loss, mtp_loss = model(X, y)
            loss = total_loss / gradient_accumulation_steps
            scaler.scale(loss).backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Gradient clipping plus strict
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # LR scheduling plus doux (cosine avec warmup)
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
            "learning_rate": lr,
            "version": "v2"
        })

        # 💾 Sauvegarde automatique toutes les 2h
        if time.time() - last_backup_time > backup_interval:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            torch.save(model.state_dict(), f"models_v2/Wild_GPT_v2_backup_{timestamp}.pt")
            with open("models_v2/log_v2.txt", "a", encoding="utf-8") as f:
                f.write(f"📦 Backup v2 automatique à {timestamp} (step {step})\n")
            last_backup_time = time.time()

        # Évaluation plus fréquente
        if step % eval_interval == 0 and step > 0:
            losses = estimate_loss(model, config, eval_iters, batch_size, device_type, device, ctx)
            train_perplexity = compute_perplexity(losses['train'])
            val_perplexity = compute_perplexity(losses['val'])

            improvement = best_val_loss - losses['val']
            print(f"Step {step}: train {losses['train']:.4f} (PPL {train_perplexity:.2f}), val {losses['val']:.4f} (PPL {val_perplexity:.2f})")
            if improvement > 0:
                print(f"🎉 Improvement: -{improvement:.4f} (best: {best_val_loss:.4f} → {losses['val']:.4f})")
            
            wandb.log({
                "val_loss": losses['val'],
                "train_loss_eval": losses['train'],
                "val_perplexity": val_perplexity,
                "train_perplexity": train_perplexity,
                "improvement": improvement
            })

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                no_improvement_steps = 0
                torch.save(model.state_dict(), best_model_path)
                torch.save(optimizer.state_dict(), best_optimizer_path)
                torch.save(model, "models_v2/Wild_GPT_v2_full.pt")
                wandb.log({"best_val_loss": best_val_loss})
                
                # Log des milestones
                if best_val_loss < 6.0:
                    print("🔥 MILESTONE: Val_loss < 6.0 !")
                if best_val_loss < 5.5:
                    print("🚀 TARGET REACHED: Val_loss < 5.5 !")
                    
            else:
                no_improvement_steps += 1
                if no_improvement_steps >= patience:
                    print("Early stopping triggered.")
                    wandb.log({"early_stopping_step": step})
                    with open("models_v2/log_v2.txt", "a", encoding="utf-8") as f:
                        f.write(f"⛔ Early stopping v2 triggered at step {step}\n")
                        f.write(f"🏆 Final best val_loss: {best_val_loss:.4f}\n")
                    break

    # Sauvegarde finale
    torch.save(model, "models_v2/Wild_GPT_v2_final.pt")
    wandb.finish()
    
    print("="*50)
    print("🎉 Wild_GPT v2 Training Complete!")
    print(f"🏆 Best val_loss: {best_val_loss:.4f}")
    print(f"📈 Improvement from v1: {6.70 - best_val_loss:.4f}")
    if best_val_loss < 5.5:
        print("🚀 TARGET ACHIEVED: < 5.5 val_loss!")
    else:
        print(f"🎯 Target progress: {((6.70 - best_val_loss) / (6.70 - 5.5)) * 100:.1f}% towards 5.5")
    print("="*50)
    
    return model, config

if __name__ == "__main__":
    train_model_v2()
