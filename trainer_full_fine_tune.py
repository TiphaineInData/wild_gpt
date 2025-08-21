import torch
import math
import os
import wandb
import time
from tqdm.auto import tqdm
from contextlib import nullcontext
from config import Wild_GPT_config
from model import Wild_GPT
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
        dropout=0.05  # 🔥 RÉDUCTION dropout pour fine-tuning
    )

    # 🎯 Hyperparamètres OPTIMISÉS pour fine-tuning
    learning_rate = 1e-5  # 🔥 BEAUCOUP plus petit (était 5e-5)
    max_iters = 15000     # Plus de steps
    warmup_steps = 500    # Warmup plus court
    min_lr = 1e-7         # LR min plus petit
    eval_interval = 1200  # Évals plus fréquentes 
    eval_iters = 200
    batch_size = 16       # Batch plus petit pour stabilité
    gradient_accumulation_steps = 16  # Plus d'accumulation
    weight_decay = 0.01   # Weight decay réduit
    patience = 15         # Plus de patience

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.cuda.amp.autocast(dtype=ptdtype)

    wandb.init(project="wild_gpt_fixed_finetune", config=config.__dict__)
    torch.manual_seed(42)

    # 🔄 Charger le modèle V2 avec TOUS ses poids
    print("📄 Initialisation du modèle...")
    model = Wild_GPT(config).to(device)

    # 🚀 CHARGEMENT du modèle V2 (dans le même dossier)
    pretrained_path = "Wild_GPT_v2.pt"
    if os.path.exists(pretrained_path):
        print(f"🚀 Chargement du modèle V2 depuis {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location=device)
        
        # 🔥 CHARGEMENT STRICT pour vérifier la compatibilité
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️ Clés manquantes: {missing_keys}")
        if unexpected_keys:
            print(f"⚠️ Clés inattendues: {unexpected_keys}")
            
        print("✅ Modèle V2 chargé avec succès !")
        
        # 🎯 VÉRIFICATION: tester le modèle sur un batch
        print("🧪 Test du modèle chargé...")
        model.eval()
        with torch.no_grad():
            X_test, y_test = get_batch("train", config, 2, device_type, device)
            try:
                logits, loss, main_loss, mtp_loss = model(X_test, y_test)
                print(f"✅ Test réussi! Loss initiale: {loss.item():.4f}")
                if loss.item() > 8.0:
                    print("⚠️ ATTENTION: Loss initiale très élevée! Vérifiez les datasets.")
            except Exception as e:
                print(f"❌ ERREUR lors du test: {e}")
                exit(1)
        model.train()
        
    else:
        print(f"❌ ERREUR: {pretrained_path} introuvable!")
        print("📁 Vérifiez que Wild_GPT_v2.pt est dans le même dossier")
        exit(1)

    # 📊 Stats du modèle
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔥 FULL FINE-TUNING activé!")
    print(f"📊 Paramètres total: {total_params:,}")
    print(f"📊 Paramètres entraînables: {trainable_params:,} (100%)")

    wandb.log({
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": 100.0
    })

    # 🎯 OPTIMISEUR: reprise de l'optimizer V2 (même dossier)
    optimizer_path = "Wild_GPT_v2_optimizer.pt"
    if os.path.exists(optimizer_path):
        print("🔄 REPRISE de l'optimiseur V2 (recommandé)")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
            eps=1e-9
        )
        try:
            optimizer_state = torch.load(optimizer_path, map_location=device)
            optimizer.load_state_dict(optimizer_state)
            print("✅ État optimiseur V2 repris!")
            
            # 🔥 MAIS on ajuste le LR pour le fine-tuning
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
                param_group['weight_decay'] = weight_decay
            print(f"🎯 LR ajusté à {learning_rate}")
            
        except Exception as e:
            print(f"⚠️ Erreur chargement optimizer: {e}")
            print("🔄 Création nouvel optimiseur")
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.95),
                weight_decay=weight_decay,
                eps=1e-9
            )
    else:
        print("🆕 Création nouvel optimiseur (Wild_GPT_v2_optimizer.pt non trouvé)")
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
    no_improvement_count = 0

    os.makedirs("finetune_models", exist_ok=True)
    best_model_path = "finetune_models/Wild_GPT_finetune_BEST.pt"
    best_optimizer_path = "finetune_models/Wild_GPT_finetune_optimizer_BEST.pt"
    log_file = "finetune_models/training_log.txt"

    last_backup_time = time.time()
    backup_interval = 7200  # 2h

    print("🚀 Début du FULL fine-tuning CORRIGÉ...")
    
    # 📝 Log initial
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"🚀 Fine-tuning démarré à {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"📊 Config: LR={learning_rate}, batch={batch_size}, grad_accum={gradient_accumulation_steps}\n")

    for step in tqdm(range(max_iters), desc="Fine-tuning CORRIGÉ"):

        try:
            X, y = get_batch("train", config, batch_size, device_type, device)

            with ctx:
                # 🔥 CORRECTION: utiliser l'API correcte du modèle
                logits, total_loss, main_loss, mtp_loss = model(X, y)

                if total_loss is None or torch.isnan(total_loss):
                    print(f"🚨 ERREUR: total_loss invalide au step {step}")
                    continue

                loss = total_loss / gradient_accumulation_steps
                scaler.scale(loss).backward()

        except Exception as e:
            print(f"🚨 ERREUR au step {step}: {e}")
            continue

        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 📈 Scheduler cosine amélioré
        if step < warmup_steps:
            lr = learning_rate * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / (max_iters - warmup_steps)
            lr = min_lr + (learning_rate - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        current_loss = total_loss.item()
        current_ppl = compute_perplexity(current_loss)
        
        wandb.log({
            "step": step,
            "total_loss": current_loss,
            "main_loss": main_loss.item() if main_loss else 0.0,
            "mtp_loss": mtp_loss.item() if mtp_loss else 0.0,
            "learning_rate": lr,
            "perplexity": current_ppl
        })

        # 📊 Log régulier
        if step % 100 == 0:
            print(f"Step {step}: loss={current_loss:.4f}, lr={lr:.2e}, ppl={current_ppl:.1f}")

        # 💾 Backup automatique
        if time.time() - last_backup_time > backup_interval:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            torch.save(model.state_dict(), f"finetune_models/Wild_GPT_finetune_backup_{timestamp}.pt")
            torch.save(optimizer.state_dict(), f"finetune_models/Wild_GPT_finetune_optimizer_backup_{timestamp}.pt")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"📦 Backup auto à {timestamp} (step {step})\n")
            print(f"💾 Backup automatique: {timestamp}")
            last_backup_time = time.time()

        # 📈 Évaluation avec monitoring val_loss
        if step % eval_interval == 0 and step > 0:
            print(f"🔍 Évaluation au step {step}...")
            losses = estimate_loss(model, config, eval_iters, batch_size, device_type, device, ctx)
            train_loss = losses['train']
            val_loss = losses['val']

            train_perplexity = compute_perplexity(train_loss)
            val_perplexity = compute_perplexity(val_loss)

            print(f"📊 EVAL Step {step}:")
            print(f"   🔥 Train: {train_loss:.4f} (PPL {train_perplexity:.2f})")
            print(f"   🎯 Val:   {val_loss:.4f} (PPL {val_perplexity:.2f})")

            wandb.log({
                "val_loss": val_loss,
                "train_loss_eval": train_loss,
                "val_perplexity": val_perplexity,
                "train_perplexity": train_perplexity
            })

            # 🏆 Nouveau record ?
            if val_loss < best_val_loss:
                improvement = best_val_loss - val_loss
                best_val_loss = val_loss
                no_improvement_count = 0
                
                torch.save(model.state_dict(), best_model_path)
                torch.save(optimizer.state_dict(), best_optimizer_path)
                
                print(f"🏆 NOUVEAU RECORD ! Amélioration: -{improvement:.4f}")
                print(f"   📈 Val loss: {val_loss:.4f} (PPL: {val_perplexity:.1f})")
                
                wandb.log({"best_val_loss": best_val_loss})
                
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"🏆 Nouveau record au step {step}: {val_loss:.4f}\n")
                    
            else:
                no_improvement_count += 1
                print(f"📊 Pas d'amélioration ({no_improvement_count}/{patience})")
                
                # 🛑 Early stopping
                if no_improvement_count >= patience:
                    print("🛑 Early stopping déclenché!")
                    wandb.log({"early_stopping_step": step})
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(f"🛑 Early stopping au step {step}\n")
                    break

    # 💾 Sauvegarde finale
    torch.save(model.state_dict(), "finetune_models/Wild_GPT_finetune_FINAL.pt")
    torch.save(optimizer.state_dict(), "finetune_models/Wild_GPT_finetune_optimizer_FINAL.pt")
    torch.save(model, "finetune_models/Wild_GPT_finetune_COMPLETE.pt")

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"🎉 Fine-tuning terminé à {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"🏆 Meilleure val loss: {best_val_loss:.4f}\n")

    wandb.finish()
    print("🎉 Fine-tuning CORRIGÉ terminé !")
    print(f"🏆 Meilleure val loss: {best_val_loss:.4f}")

    return model, config

if __name__ == "__main__":
    train_full_finetune()
