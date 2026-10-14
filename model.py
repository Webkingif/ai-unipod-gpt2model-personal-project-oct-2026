import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, max_seq_len, dropout):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_model = d_model
        
        # Key, Query, Value projections all in one matrix for efficiency
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        # Output projection
        self.c_proj = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Causal mask to ensure attention only looks at past tokens
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )

    def forward(self, x):
        B, T, C = x.shape  # Batch size, Sequence length, Embedding dimension (d_model)
        
        # Calculate query, key, values
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        # Reshape for multi-head attention: (B, nh, T, hs)
        hs = C // self.n_head
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        
        # Causal self-attention: (Q @ K^T) / sqrt(hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Re-assemble all head outputs
        
        return self.resid_dropout(self.c_proj(y))

class MLP(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.c_fc = nn.Linear(d_model, 4 * d_model)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)

class Block(nn.Module):
    def __init__(self, d_model, n_head, max_seq_len, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, max_seq_len, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)

    def forward(self, x):
        # Pre-LN architecture (LayerNorm applied before sub-layers)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, n_layer, n_head, dropout=0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, d_model),
            wpe = nn.Embedding(max_seq_len, d_model),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([Block(d_model, n_head, max_seq_len, dropout) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(d_model),
        ))
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying (share weights between token embedding and final output projection)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Cannot forward sequence of length {T}, max is {self.max_seq_len}"
        
        # Forward pass through token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Pass through transformer blocks
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss
