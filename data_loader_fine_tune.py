import torch
import numpy as np
import os

def get_batch(split, config, batch_size, device_type, device):
    """
    Data loader pour les fichiers de fine-tuning
    Utilise train_finetune.bin et val_finetune.bin
    """
    base_path = os.path.dirname(__file__)
    
    # Chemins vers les fichiers de fine-tuning
    if split == 'train':
        path = os.path.join(base_path, 'train_finetune.bin')
    elif split == 'val' or split == 'validation':
        path = os.path.join(base_path, 'val_finetune.bin')
    else:
        raise ValueError(f"Split '{split}' non reconnu. Utilisez 'train' ou 'val'.")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier {path} non trouvé. Avez-vous lancé data_prep_finetune.py ?")

    data = np.memmap(path, dtype=np.uint16, mode='r')
    block_size = config.block_size
    eof_token = 4

    x_batch = []
    y_batch = []

    while len(x_batch) < batch_size:
        i = torch.randint(len(data) - 1, (1,)).item()

        # Séquence d'entrée (x) et cible (y)
        seq = data[i : i + block_size + 1]

        # Si on est en fin de fichier ou que la séquence est trop courte :
        if len(seq) < block_size + 1:
            continue

        x_seq = seq[:-1]
        y_seq = seq[1:]

        # Vérifie qu'on a au moins un token utile dans y
        if (y_seq != eof_token).any():
            x_tensor = torch.from_numpy(x_seq.astype(np.int64))
            y_tensor = torch.from_numpy(y_seq.astype(np.int64))
            x_batch.append(x_tensor)
            y_batch.append(y_tensor)

    x = torch.stack(x_batch)
    y = torch.stack(y_batch)

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y

@torch.no_grad()
def estimate_loss(model, config, eval_iters, batch_size, device_type, device, ctx):
    """
    Estimation de la loss pour PEFT/LoRA models
    """
    model.eval()
    out = {}

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            X, Y = get_batch(split, config, batch_size, device_type, device)

            with ctx:
                # Pour PEFT, on utilise l'API standard
                outputs = model(input_ids=X, labels=Y)
                losses[k] = outputs.loss.item()

        out[split] = losses.mean()

    model.train()
    return out