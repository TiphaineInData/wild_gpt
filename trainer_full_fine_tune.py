import torch
import math
import os
import wandb
import time
from tqdm.auto import tqdm
from contextlib import nullcontext
from config import Wild_GPT_config
from model import Wild_GPT   # ✅ Classe définie dans model.py
from data_loader_fine_tune import get_batch, estimate_loss
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def compute_perplexity(loss: float) -> float:
    return math.exp(loss) if loss is not None and loss < 100 else float("inf")

def run_evaluation(model, config, step, eval_iters, batch_size, device_type, device, ctx, epoch=None):
    """Effectue une évaluation et logge sur wandb."""
    losses = estimate_loss(model, config, eval_iters, batch_size, device_type, device, ctx)
    train_loss = losses['train']
    val_loss = losses['val']
    train_ppl = compute_perplexity(train_loss)
    val_ppl = compute_perplexity(val_loss)

    if epoch is not None:
        print(f"\n📊 EVAL Epoch {epoch}: train {train_loss:.4f} (PPL {train_ppl:.2f}), "
              f"val {val_loss:.4f} (PPL {val_ppl:.2f})")
        wandb.log({
            "epoch": epoch,
            "val_loss": val_loss,
            "train_loss_eval": train_loss,
            "val_perplexity": val_ppl,
            "train_perplexity": train_ppl
        })
    else:
        print(f"📊 EVAL Step {step}: train {train_loss:.4f} (PPL {train_ppl:.2f}), "
              f"val {val_loss:.4f} (PPL {val_ppl:.2f})")
        wandb.log({
            "step": step,
            "val_loss": val_loss,
            "train_loss_eval": train_loss,
            "val_perplexity": val_ppl,
            "train_perplexity": train_ppl
        })

    return val_loss, val_ppl

def train_full_finetune():
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
        dropout=0.10
    )

    # Hyperparamètres full fine-tuning
    learning_rate = 5e-5
    max_iters = 10000
    warmup_steps = 1000
    min_lr = 1e-6
    eval_interval = 1800
    eval_iters = 250
    batch_size = 32
    gradient_accumulation_steps = 8
    weight_decay = 0.15
    max_epochs = 5  # ⚠️ Nouveau : découpage en epochs

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.cuda.amp.autocast(dtype=ptdtype)

    wandb.init(project="wild_gpt_full_finetune", config=config.__dict__)
    torch.manual_seed(42)

    # 🔄 Charger le modèle
    print("🔄 Initialisation du modèle...")
    model = Wild_GPT(config).to(device)

    pretrained_path = "Wild_GPT_v2.pt"
    if os.path.exists(pretrained_path):
        print(f"🚀 Chargement des poids depuis {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print("✅ Poids V2 chargés avec succès !")
    else:
        print(f"❌ ERREUR: {pretrained_path} introuvable !")
        exit(1)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔥 FULL FINE-TUNING activé !")
    print(f"📊 Paramètres total: {total_params:,}")
    print(f"📊 Paramètres entraînables: {trainable_params:,} (100%)")

    wandb.log({
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": 100.0
    })

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
        eps=1e-9
    )
    scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))

    model.train()
    best_val_loss = float('inf')

    os.makedirs("finetune_models", exist_ok=True)
    best_model_path = "finetune_models/Wild_GPT_finetune_best.pt"
    log_file = "finetune_models/training_log.txt"

    print("🚀 Début du FULL fine-tuning...")

    steps_per_epoch = max_iters // max_epochs
    global_step = 0

    for epoch in range(1, max_epochs + 1):
        for step in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}/{max_epochs}"):
            X, y = get_batch("train", config, batch_size, device_type, device)

            with ctx:
                logits, total_loss, main_loss, mtp_loss = model(X, y)
                loss = total_loss / gradient_accumulation_steps
                scaler.scale(loss).backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Scheduler cosine
            if global_step < warmup_steps:
                lr = learning_rate * (global_step + 1) / warmup_steps
            else:
                progress = (global_step - warmup_steps) / (max_iters - warmup_steps)
                lr = min_lr + (learning_rate - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            current_loss = total_loss.item()
            wandb.log({
                "step": global_step,
                "epoch": epoch,
                "total_loss": current_loss,
                "main_loss": main_loss.item(),
                "mtp_loss": mtp_loss.item() if mtp_loss else 0.0,
                "learning_rate": lr,
                "perplexity": compute_perplexity(current_loss)
            })

            if global_step % 50 == 0:
                ppl = compute_perplexity(current_loss)
                print(f"Step {global_step}: loss={current_loss:.4f}, lr={lr:.2e}, ppl={ppl:.1f}")

            # 🔍 Évaluation régulière
            if global_step % eval_interval == 0 and global_step > 0:
                val_loss, val_ppl = run_evaluation(model, config, global_step,
                                                   eval_iters, batch_size, device_type, device, ctx)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), best_model_path)
                    print(f"🏆 NOUVEAU RECORD ! Val loss: {val_loss:.4f} (PPL: {val_ppl:.1f})")
                    wandb.log({"best_val_loss": best_val_loss})

            global_step += 1

        # 📊 Évaluation fin d’epoch
        val_loss, val_ppl = run_evaluation(model, config, global_step,
                                           eval_iters, batch_size, device_type, device, ctx, epoch=epoch)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"🏆 [Epoch {epoch}] NOUVEAU RECORD ! Val loss: {val_loss:.4f} (PPL: {val_ppl:.1f})")
            wandb.log({"best_val_loss": best_val_loss})

    # Sauvegarde finale
    torch.save(model.state_dict(), "finetune_models/Wild_GPT_finetune_final.pt")
    torch.save(model, "finetune_models/Wild_GPT_finetune_complete_final.pt")
    wandb.finish()
    print("🎉 Full fine-tuning terminé !")
    print(f"🏆 Meilleure val loss: {best_val_loss:.4f}")

    return model, config

if __name__ == "__main__":
    train_full_finetune()
