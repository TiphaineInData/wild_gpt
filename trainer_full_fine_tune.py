import torch
import math
import os
import wandb
import time
from tqdm.auto import tqdm
from contextlib import nullcontext
from config import Wild_GPT_config
from model import Wild_GPT   # ✅ importer la classe du modèle
from data_loader_fine_tune import get_batch, estimate_loss
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def compute_perplexity(loss: float) -> float:
    return math.exp(loss) if loss is not None and loss < 100 else float("inf")

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
        dropout=0.10   # FT → un peu moins de régularisation
    )

    # Hyperparamètres full fine-tuning
    learning_rate = 5e-5
    max_iters = 10000
    warmup_steps = 1000
    min_lr = 1e-6
    eval_interval = 1800
    eval_iters = 250
    batch_size = 32
    gradient_accumulation_steps = 8  # batch effectif = 256
    weight_decay = 0.15

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.cuda.amp.autocast(dtype=ptdtype)

    wandb.init(project="wild_gpt_full_finetune", config=config.__dict__)
    torch.manual_seed(42)

    # Charger le modèle
    print("🔄 Initialisation du modèle...")
    model = Wild_GPT(config).to(device)

    # Charger les poids du modèle pré-entraîné (Wild_GPT_v2.pt)
    pretrained_path = "./Wild_GPT_v2.pt"
    if os.path.exists(pretrained_path):
        state_dict = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ Poids Wild_GPT_v2 chargés depuis {pretrained_path}")
    else:
        print(f"🚨 ERREUR: Fichier {pretrained_path} non trouvé !")
        exit(1)

    # Tous les paramètres sont entraînables
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

    # Optimiseur
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

    # Chemins de sauvegarde
    os.makedirs("finetune_models", exist_ok=True)
    best_model_path = "./finetune_models/Wild_GPT_finetune_best.pt"
    best_optimizer_path = "./finetune_models/Wild_GPT_finetune_optimizer_best.pt"
    log_file = "./finetune_models/training_log.txt"

    last_backup_time = time.time()
    backup_interval = 7200  # 2h

    print("🚀 Début du FULL fine-tuning...")

    for step in tqdm(range(max_iters), desc="Full Fine-tuning",
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):

        # Récupérer un batch
        X, y = get_batch("train", config, batch_size, device_type, device)

        with ctx:
            logits, total_loss, main_loss, mtp_loss = model(X, y)

            if total_loss is None:
                print("🚨 ERREUR: total_loss est None !")
                exit(1)

            loss = total_loss / gradient_accumulation_steps
            scaler.scale(loss).backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Scheduler cosine
        if step < warmup_steps:
            lr = learning_rate * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / (max_iters - warmup_steps)
            lr = min_lr + (learning_rate - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        current_loss = total_loss.item()
        wandb.log({
            "step": step,
            "total_loss": current_loss,
            "main_loss": main_loss.item(),
            "mtp_loss": mtp_loss.item() if mtp_loss else 0.0,
            "learning_rate": lr,
            "perplexity": compute_perplexity(current_loss)
        })

        if step % 50 == 0:
            ppl = compute_perplexity(current_loss)
            print(f"Step {step}: loss={current_loss:.4f}, lr={lr:.2e}, ppl={ppl:.1f}")

        # 💾 Backup auto toutes les 2h
        if time.time() - last_backup_time > backup_interval:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            torch.save(model.state_dict(), f"./finetune_models/Wild_GPT_finetune_backup_{timestamp}.pt")
            torch.save(optimizer.state_dict(), f"./finetune_models/Wild_GPT_finetune_optimizer_backup_{timestamp}.pt")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"📦 Backup auto à {timestamp} (step {step})\n")
            print(f"💾 Backup automatique sauvé: {timestamp}")
            last_backup_time = time.time()

        # Évaluation
        if step % eval_interval == 0 and step > 0:
            losses = estimate_loss(model, config, eval_iters, batch_size, device_type, device, ctx)
            train_loss = losses['train']
            val_loss = losses['val']

            train_perplexity = compute_perplexity(train_loss)
            val_perplexity = compute_perplexity(val_loss)

            print(f"📊 EVAL Step {step}: train {train_loss:.4f} (PPL {train_perplexity:.2f}), "
                  f"val {val_loss:.4f} (PPL {val_perplexity:.2f})")

            wandb.log({
                "val_loss": val_loss,
                "train_loss_eval": train_loss,
                "val_perplexity": val_perplexity,
                "train_perplexity": train_perplexity
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                torch.save(optimizer.state_dict(), best_optimizer_path)

                # sauvegarde supplémentaire instruct
                instruct_best = "./finetune_models/best_wild_gpt_v_instruct.pt"
                torch.save(model.state_dict(), instruct_best)

                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"🏆 Nouveau record à step {step}: val_loss {val_loss:.4f}, "
                            f"PPL {val_perplexity:.2f}\n")
                    f.write(f"➡️ Best checkpoint: {instruct_best}\n")

                print(f"🏆 NOUVEAU RECORD ! Val loss: {val_loss:.4f} (PPL: {val_perplexity:.1f})")
                wandb.log({"best_val_loss": best_val_loss})

    # Sauvegarde finale
    final_model = "./finetune_models/Wild_GPT_finetune_final.pt"
    final_opt = "./finetune_models/Wild_GPT_finetune_optimizer_final.pt"
    final_complete = "./finetune_models/Wild_GPT_finetune_complete_final.pt"
    final_instruct = "./finetune_models/wild_gpt_v_instruct_final.pt"

    torch.save(model.state_dict(), final_model)
    torch.save(optimizer.state_dict(), final_opt)
    torch.save(model, final_complete)
    torch.save(model.state_dict(), final_instruct)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("🎉 Fine-tuning terminé !\n")
        f.write(f"📦 Final checkpoints:\n")
        f.write(f"  - {final_model}\n")
        f.write(f"  - {final_opt}\n")
        f.write(f"  - {final_complete}\n")
        f.write(f"  - {final_instruct}\n")

    wandb.finish()
    print("🎉 Full fine-tuning terminé !")
    print(f"🏆 Meilleure val loss: {best_val_loss:.4f}")

    return model, config

if __name__ == "__main__":
    train_full_finetune()
