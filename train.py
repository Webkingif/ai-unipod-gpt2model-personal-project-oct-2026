import torch
import os
from tokenizers import Tokenizer
from dataset import get_dataloader
from model import GPT

def train():
    # --- 1. Hyperparameters & Config ---
    BATCH_SIZE = 16        # Balanced for your 6GB Arc A380 VRAM
    MAX_SEQ_LEN = 256
    D_MODEL = 384          # Enhanced capacity for the 7.1MB specialized domain corpus
    N_LAYER = 6            # Expanded depth from 4 to 6 blocks
    N_HEAD = 6             # Clean split: 384 / 6 = 64 head size
    DROPOUT = 0.1
    LEARNING_RATE = 3e-4
    MAX_ITERS = 40000      # Scaled up to thoroughly process your ~1.8M token dataset
    
    # --- 2. Hardware Device Selection (Targeting Intel Arc A380) ---
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        # gpu0 is UHD 770 (integrated), gpu1 is Intel Arc A380 (discrete)
        device = torch.device('xpu:1' if torch.xpu.device_count() > 1 else 'xpu:0')
        print(f"🚀 Using Intel GPU: {torch.xpu.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("⚠️ Intel XPU backend not found, falling back to CPU.")

    # --- 3. Load Data and Tokenizer ---
    dataloader = get_dataloader("corpus.txt", "custom_gpt2_tokenizer.json", MAX_SEQ_LEN, BATCH_SIZE)
    
    tokenizer = Tokenizer.from_file("custom_gpt2_tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary Size: {vocab_size}")

    # --- 4. Initialize Model & Optimizer ---
    model = GPT(
        vocab_size=vocab_size,
        max_seq_len=MAX_SEQ_LEN,
        d_model=D_MODEL,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        dropout=DROPOUT
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Ensure checkpoint directory exists
    os.makedirs("checkpoints", exist_ok=True)

    # --- 5. Training Loop ---
    model.train()
    print("🏁 Starting training loop on your Intel Arc setup...")
    
    iter_count = 0
    while iter_count < MAX_ITERS:
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            # Forward pass: compute logits and loss
            logits, loss = model(x, y)

            # Backward pass & weight update
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Print progress periodically
            if iter_count % 100 == 0:
                print(f"Iteration {iter_count}/{MAX_ITERS} | Loss: {loss.item():.4f}")

            # Save periodic checkpoints every 2,000 steps
            if iter_count > 0 and iter_count % 2000 == 0:
                ckpt_path = f"checkpoints/gpt2_model_iter_{iter_count}.pt"
                torch.save(model.state_dict(), ckpt_path)
                print(f"💾 Checkpoint saved: {ckpt_path}")

            iter_count += 1
            if iter_count >= MAX_ITERS:
                break

    # --- 6. Save Final Model Checkpoint ---
    final_path = "checkpoints/gpt2_model.pt"
    torch.save(model.state_dict(), final_path)
    print(f"✅ Training complete! Final model saved to '{final_path}'.")

if __name__ == "__main__":
    train()
