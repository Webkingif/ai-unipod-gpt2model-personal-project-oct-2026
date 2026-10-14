import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from model import GPT

def generate_text(prompt, max_new_tokens=50, temperature=0.6, top_k=50):
    # --- 1. Config & Device Setup ---
    MAX_SEQ_LEN = 128
    D_MODEL = 256
    N_LAYER = 4
    N_HEAD = 4
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"🚀 Using device: {device}")

    # --- 2. Load Tokenizer & Model Checkpoint ---
    tokenizer_path = "custom_gpt2_tokenizer.json"
    model_path = "checkpoints/gpt2_model.pt"
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    model = GPT(
        vocab_size=vocab_size,
        max_seq_len=MAX_SEQ_LEN,
        d_model=D_MODEL,
        n_layer=N_LAYER,
        n_head=N_HEAD
    ).to(device)

    # Load trained weights safely
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Model weights loaded successfully!")

    # --- 3. Encode Prompt ---
    encoded_prompt = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded_prompt.ids], dtype=torch.long, device=device)

    # --- 4. Generation Loop ---
    print(f"\n📝 Prompt: \"{prompt}\"")
    print("🤖 Generating response...\n")

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop context if it exceeds max sequence length
            cond_idx = input_ids[:, -MAX_SEQ_LEN:]
            
            # Forward pass to get logits
            logits, _ = model(cond_idx)
            
            # Focus on the last token's predictions and scale by temperature
            logits = logits[:, -1, :] / temperature
            
            # Optional: Top-K filtering to prevent weird random tokens
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample the next token from the probability distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to running sequence
            input_ids = torch.cat((input_ids, next_token), dim=1)

            # Stop generating early if model outputs the <|endoftext|> token
            # (Assuming <|endoftext|> was added during tokenizer training)
            decoded_token = tokenizer.decode(next_token[0].tolist())
            if "<|endoftext|>" in decoded_token:
                break

    # --- 5. Decode Final Token Sequence ---
    full_output_ids = input_ids[0].tolist()
    raw_text = tokenizer.decode(full_output_ids)
    generated_text = raw_text.replace("Ġ", " ").strip()
    
    print(f"✨ Output:\n{generated_text}")
    return generated_text

if __name__ == "__main__":
    # Test generation with a sample prompt from your domain
    sample_prompt = "AI UniPod UNILAG is an artificial intelligence"
    generate_text(sample_prompt, max_new_tokens=60, temperature=0.7)
