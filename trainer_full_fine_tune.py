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

def train_intensif_12h():
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
        dropout=0.02  # 🔥 Dropout bas pour maximum apprentissage
    )

    # 🎯 HYPERPARAMÈTRES INTENSIFS 12h (Budget $25)
    learning_rate = 5e-5      # 🔥 AGRESSIF pour aller vite
    max_iters = 30000         # ~12h d'entraînement
    warmup_steps = 1500       # Warmup rapide
    min_lr = 5e-7             # LR minimal
    eval_interval = 1500      # Évals moins fréquentes = plus de training
    eval_iters = 100          # Évals rapides
    batch_size = 32           # 🚀 MAXIMUM GPU usage
    gradient_accumulation_steps = 8   # Effective batch = 256
    weight_decay = 0.08       # Un peu plus de régularisation
    patience = 8              # 🔥 Early stopping AGRESSIF si plateau

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.cuda.amp.autocast(dtype=ptdtype)

    wandb.init(project="wild_gpt_INTENSIF_12H", config=config.__dict__)
    torch.manual_seed(42)

    # 🔄 Charger le modèle V2
    print("📄 Initialisation du modèle...")
    model = Wild_GPT(config).to(device)

    # 🚀 Chargement du modèle V2 (6.21 val_loss)
    pretrained_path = "Wild_GPT_v2.pt"
    if os.path.exists(pretrained_path):
        print(f"🚀 Chargement du modèle V2 depuis {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print("✅ Modèle V2 chargé avec succès !")
        
        # Test rapide
        model.eval()
        with torch.no_grad():
            X_test, y_test = get_batch("train", config, 2, device_type, device)
            logits, loss, main_loss, mtp_loss = model(X_test, y_test)
            print(f"✅ Test réussi! Loss initiale: {loss.item():.4f}")
        model.train()
        
    else:
        print(f"❌ ERREUR: {pretrained_path} introuvable!")
        exit(1)

    # 📊 Stats du modèle
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🔥 TRAINING INTENSIF 12h activé!")
    print(f"📊 Paramètres: {total_params:,}")
    print(f"💰 Budget: $25 (~13h max à $1.80/h)")
    print(f"🎯 Objectif AMBITIEUX: Val_loss < 4.5, Perplexité < 150")

    wandb.log({
        "total_parameters": total_params,
        "budget_dollars": 25,
        "hourly_cost": 1.80,
        "max_hours": 13.8,
        "target_val_loss": 4.5,
        "target_perplexity": 150
    })

    # 🎯 Optimiseur AGRESSIF
    optimizer_path = "Wild_GPT_v2_optimizer.pt"
    if os.path.exists(optimizer_path):
        print("🔄 REPRISE de l'optimiseur V2")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),  # Standard pour vitesse
            weight_decay=weight_decay,
            eps=1e-8
        )
        try:
            optimizer_state = torch.load(optimizer_path, map_location=device)
            optimizer.load_state_dict(optimizer_state)
            
            # Ajuster LR pour training intensif
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
                param_group['weight_decay'] = weight_decay
            print(f"🎯 LR AGRESSIF ajusté à {learning_rate}")
            
        except Exception as e:
            print(f"⚠️ Erreur chargement optimizer: {e}")
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999),
                weight_decay=weight_decay,
                eps=1e-8
            )
    else:
        print("🆕 Création nouvel optimiseur AGRESSIF")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
            eps=1e-8
        )

    scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))

    model.train()
    best_val_loss = float('inf')
    no_improvement_count = 0
    start_time = time.time()

    os.makedirs("intensif_models", exist_ok=True)
    best_model_path = "intensif_models/Wild_GPT_INTENSIF_BEST.pt"
    best_optimizer_path = "intensif_models/Wild_GPT_INTENSIF_optimizer_BEST.pt"
    log_file = "intensif_models/intensif_log.txt"

    # 🎯 Checkpoints toutes les 2h pour économiser
    last_backup_time = time.time()
    backup_interval = 7200  # 2h

    print("🚀 TRAINING INTENSIF 12h - MAXIMUM PERFORMANCE!")
    print(f"⚡ LR agressif: {learning_rate}, Batch: {batch_size}")
    
    # Log initial
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"🚀 TRAINING INTENSIF démarré à {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"💰 Budget: $25 pour ~12h à $1.80/h\n")
        f.write(f"🎯 OBJECTIF: val_loss < 4.5, perplexité < 150\n")
        f.write(f"⚡ Config AGRESSIF: LR={learning_rate}, batch={batch_size}x{gradient_accumulation_steps}\n")

    best_milestone_loss = 8.0  # Pour tracker les milestones

    for step in tqdm(range(max_iters), desc="Training INTENSIF", 
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {rate_fmt}"):

        try:
            X, y = get_batch("train", config, batch_size, device_type, device)

            with ctx:
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

        # 📈 Scheduler cosine AGRESSIF
        if step < warmup_steps:
            lr = learning_rate * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / (max_iters - warmup_steps)
            lr = min_lr + (learning_rate - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        current_loss = total_loss.item()
        current_ppl = compute_perplexity(current_loss)
        
        # 📊 Monitoring INTENSIF
        wandb.log({
            "step": step,
            "total_loss": current_loss,
            "main_loss": main_loss.item() if main_loss else 0.0,
            "mtp_loss": mtp_loss.item() if mtp_loss else 0.0,
            "learning_rate": lr,
            "perplexity": current_ppl,
            "hours_elapsed": (time.time() - start_time) / 3600,
            "cost_spent": ((time.time() - start_time) / 3600) * 1.80,
            "progress_to_target": max(0, (6.5 - current_loss) / (6.5 - 4.5))
        })

        # 🎯 Milestones tracker
        if current_loss < best_milestone_loss:
            print(f"🎯 MILESTONE! Loss dropped to {current_loss:.4f} (PPL: {current_ppl:.1f})")
            best_milestone_loss = current_loss
            
        # Log détaillé moins fréquent pour économiser temps
        if step % 300 == 0:
            hours_elapsed = (time.time() - start_time) / 3600
            cost_spent = hours_elapsed * 1.80
            remaining_budget = 25 - cost_spent
            
            print(f"Step {step}: loss={current_loss:.4f}, ppl={current_ppl:.1f}, "
                  f"h={hours_elapsed:.1f}, cost=${cost_spent:.1f}, left=${remaining_budget:.1f}")
            
            # 🎯 Alertes objectif
            if current_ppl < 200:
                print(f"🔥 EXCELLENT! Perplexité sous 200: {current_ppl:.1f}")
            if current_loss < 5.0:
                print(f"🎯 GÉNIAL! Loss sous 5.0: {current_loss:.4f}")
            if current_loss < 4.5:
                print(f"🏆 OBJECTIF ATTEINT! Loss sous 4.5: {current_loss:.4f}")

        # 💾 Backup périodique
        if time.time() - last_backup_time > backup_interval:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            torch.save(model.state_dict(), f"intensif_models/Wild_GPT_backup_{timestamp}.pt")
            hours_elapsed = (time.time() - start_time) / 3600
            cost_spent = hours_elapsed * 1.80
            
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"💾 Backup à {timestamp} - Step {step}, Cost: ${cost_spent:.1f}\n")
            print(f"💾 Backup automatique: {timestamp} (${cost_spent:.1f} dépensés)")
            last_backup_time = time.time()

        # 📈 Évaluation INTENSIF
        if step % eval_interval == 0 and step > 0:
            hours_elapsed = (time.time() - start_time) / 3600
            cost_spent = hours_elapsed * 1.80
            
            print(f"🔍 ÉVAL Step {step} (${cost_spent:.1f} dépensés)...")
            losses = estimate_loss(model, config, eval_iters, batch_size, device_type, device, ctx)
            train_loss = losses['train']
            val_loss = losses['val']

            train_perplexity = compute_perplexity(train_loss)
            val_perplexity = compute_perplexity(val_loss)

            print(f"📊 EVAL Step {step}:")
            print(f"   🔥 Train: {train_loss:.4f} (PPL {train_perplexity:.1f})")
            print(f"   🎯 Val:   {val_loss:.4f} (PPL {val_perplexity:.1f})")
            print(f"   💰 Cost:  ${cost_spent:.1f} / $25")

            wandb.log({
                "val_loss": val_loss,
                "train_loss_eval": train_loss,
                "val_perplexity": val_perplexity,
                "train_perplexity": train_perplexity,
                "cost_at_eval": cost_spent
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
                    f.write(f"🏆 Record au step {step}: {val_loss:.4f} (${cost_spent:.1f})\n")
                
                # 🎯 Check si objectif atteint
                if val_loss < 4.5:
                    print("🎉 OBJECTIF VAL_LOSS < 4.5 ATTEINT !")
                    wandb.log({"objective_reached": True})
                    
            else:
                no_improvement_count += 1
                print(f"📊 Pas d'amélioration ({no_improvement_count}/{patience})")
                
                # 🛑 Early stopping AGRESSIF
                if no_improvement_count >= patience:
                    print("🛑 Early stopping - Plateau détecté!")
                    print(f"💰 Coût final: ${cost_spent:.1f}")
                    wandb.log({"early_stopping_step": step, "final_cost": cost_spent})
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(f"🛑 Early stopping au step {step}, coût: ${cost_spent:.1f}\n")
                    break
            
            # 💰 Check budget limite
            if cost_spent >= 24.0:  # Garde $1 de marge
                print("💰 BUDGET LIMITE ATTEINT! Sauvegarde finale...")
                torch.save(model.state_dict(), "intensif_models/Wild_GPT_BUDGET_FINAL.pt")
                wandb.log({"budget_exhausted": True, "final_cost": cost_spent})
                break

    # 💾 Sauvegarde finale
    final_hours = (time.time() - start_time) / 3600
    final_cost = final_hours * 1.80
    
    torch.save(model.state_dict(), "intensif_models/Wild_GPT_INTENSIF_FINAL.pt")
    torch.save(optimizer.state_dict(), "intensif_models/Wild_GPT_INTENSIF_optimizer_FINAL.pt")
    torch.save(model, "intensif_models/Wild_GPT_INTENSIF_COMPLETE.pt")

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"🎉 Training terminé à {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"🏆 Meilleure val loss: {best_val_loss:.4f}\n")
        f.write(f"💰 Coût total: ${final_cost:.2f} / $25\n")
        f.write(f"⏱️ Durée: {final_hours:.1f}h\n")

    wandb.finish()
    
    print("🎉 TRAINING INTENSIF terminé !")
    print(f"🏆 Meilleure val loss: {best_val_loss:.4f}")
    print(f"💰 Coût total: ${final_cost:.2f}")
    print(f"⏱️ Durée: {final_hours:.1f}h")
    
    # 🎯 Évaluation finale
    if best_val_loss < 4.5:
        print("🎊 OBJECTIF ATTEINT! Val_loss < 4.5 !")
    elif best_val_loss < 5.0:
        print("🔥 EXCELLENT! Très proche de l'objectif!")
    else:
        print("💪 Bon progrès! Prêt pour session 2?")

    return model, config

if __name__ == "__main__":
    train_intensif_12h()
