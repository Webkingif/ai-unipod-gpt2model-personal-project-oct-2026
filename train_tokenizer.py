from tokenizers import Tokenizer, models, trainers, pre_tokenizers

def train_custom_tokenizer():
    # 1. Initialize an empty BPE model
    tokenizer = Tokenizer(models.BPE())

    # 2. Use the ByteLevel pre-tokenizer (standard for GPT-2 to handle spaces and raw bytes cleanly)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 3. Configure the BPE Trainer
    # Note: If your dataset is small (e.g., Tiny Shakespeare), keep vocab_size around 5000.
    # If your dataset is huge, you can scale this up to 16000, 32000, or 50257.
    trainer = trainers.BpeTrainer(
        vocab_size=16384,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # 4. Point to your local dataset file(s)
    files = ["corpus.txt"]

    print("🚀 Training custom BPE tokenizer... (This should only take a moment)")
    tokenizer.train(files, trainer)

    # 5. Save the trained tokenizer configuration to disk
    tokenizer.save("custom_gpt2_tokenizer.json")
    print("✅ Tokenizer successfully trained and saved as 'custom_gpt2_tokenizer.json'!")

    # 6. Quick test to verify it works
    output = tokenizer.encode("Hello, world! <|endoftext|>")
    print("\n--- Test Encoding ---")
    print("Tokens:", output.tokens)
    print("IDs:", output.ids)
    
    # Decode back to text
    decoded = tokenizer.decode(output.ids)
    print("Decoded text:", decoded)

if __name__ == "__main__":
    train_custom_tokenizer()
