import torch
import os
from tokenizers import Tokenizer
from dataset import get_dataloader
from model import GPT

def train():
    # --- 1. Hyperparameters & Config ---
    BATCH_SIZE = 8
    MAX_SEQ_LEN = 128
    D_MODEL = 256
    N_LAYER = 4
    N_HEAD = 4
    DROPOUT = 0.1
    LEARNING_RATE = 3e-4
    MAX_ITERS = 11000  # Increase this if you have a larger dataset!
    
    # Select hardware device (CUDA for Nvidia, MPS for Apple Silicon, else CPU)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"🚀 Using device: {device}")

    # --- 2. Load Data and Tokenizer ---
    dataloader = get_dataloader("corpus.txt", "custom_gpt2_tokenizer.json", MAX_SEQ_LEN, BATCH_SIZE)
    
    tokenizer = Tokenizer.from_file("custom_gpt2_tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary Size: {vocab_size}")

    # --- 3. Initialize Model & Optimizer ---
    model = GPT(
        vocab_size=vocab_size,
        max_seq_len=MAX_SEQ_LEN,
        d_model=D_MODEL,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        dropout=DROPOUT
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # --- 4. Training Loop ---
    model.train()
    print("🏁 Starting training loop...")
    
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
            if iter_count % 50 == 0:
                print(f"Iteration {iter_count}/{MAX_ITERS} | Loss: {loss.item():.4f}")

            iter_count += 1
            if iter_count >= MAX_ITERS:
                break

    # --- 5. Save Model Checkpoint ---
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/gpt2_model.pt")
    print("✅ Training complete! Model saved to 'checkpoints/gpt2_model.pt'.")

if __name__ == "__main__":
    train()
