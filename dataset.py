import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

class GPTDataset(Dataset):
    def __init__(self, file_path, tokenizer_path, max_seq_len):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_seq_len = max_seq_len

        print(f"Loading text from {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        print("tokenizing corpus ...")
        encoded = self.tokenizer.encode(text)
        self.tokens = torch.tensor(encoded.ids, dtype=torch.long)
        print(f"Total tokens in dataset:{len(self.tokens)}:")

    def __len__(self):
        return max(0, len(self.tokens)-self.max_seq_len)

    def __getitem__(self, idx):
        chunk = self.tokens[idx: idx+self.max_seq_len+1]
        x = chunk[:-1]
        y = chunk[1:]

        return x, y




def get_dataloader(file_path, tokenizer_path, max_seq_len, batch_size, shuffle=True):
    dataset = GPTDataset(file_path, tokenizer_path, max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

if __name__ == "__main__":
    # Quick test of the dataset pipeline
    BATCH_SIZE = 4
    MAX_SEQ_LEN = 64
    
    loader = get_dataloader("corpus.txt", "custom_gpt2_tokenizer.json", MAX_SEQ_LEN, BATCH_SIZE)
    
    for x_batch, y_batch in loader:
        print("\n--- Batch Sample Test ---")
        print("Input shape (x):", x_batch.shape)
        print("Target shape (y):", y_batch.shape)
        break


    
        
