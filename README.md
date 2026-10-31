# Custom Domain-Specific GPT-2 Model (Nigerian Higher Education, AI & Robotics)

A custom-built, lightweight GPT-2 transformer model trained from scratch on local hardware to specialize in generating coherent technical and academic text centered around Nigerian higher education, artificial intelligence, and robotics.

---

## 🚀 Project Overview

This project implements a complete local Large Language Model (LLM) pipeline—from raw corpus curation and custom tokenizer training to model architecture design, PyTorch training/resumption loops, and inference generation. 

### Key Specifications:
* **Architecture:** Custom Pre-LN GPT-2 (`d_model=384`, `n_layer=6`, `n_head=6`, ~15M parameters).
* **Context Window:** 256 tokens.
* **Tokenizer:** Byte-Pair Encoding (BPE) with an optimized vocabulary size of 16,384 to eliminate sub-word fragmentation.
* **Hardware Acceleration:** Designed for local accelerator setups, specifically optimized for **Intel Arc A380 GPUs** via the PyTorch `xpu` backend.

---

## 📁 Repository Structure

```text
├── corpus.txt                 # 7.1 MB domain-specific corpus
├── train_tokenizer.py         # Script to train the custom BPE tokenizer
├── tokenizer.json             # Trained tokenizer configuration
├── train.py                   # Main model training script with XPU & checkpoint resume support
├── generate.py                # Inference script with temperature scaling & sentence trimming
├── dataset.py                 # PyTorch DataLoader and batching logic
├── model.py                   # Custom GPT-2 transformer architecture definition
└── checkpoints/               # Directory for periodic model checkpoints (.pt)
```
⚙️ Prerequisites & Environment Setup
1. Python Environment
Ensure you have Python 3.10+ installed. Install the required dependencies:
pip install torch torchvision torchaudio tokenizers regex

2. Intel XPU Support (For Intel Arc A380 GPUs)
If you are running on dedicated Intel Arc graphics hardware, ensure your PyTorch environment is configured with the Intel XPU backend:
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/xpu](https://download.pytorch.org/whl/xpu)

🛠️ Step-by-Step Guide to Recreate the Project
Step 1: Prepare the Domain Corpus
Place your raw text data into a single file named corpus.txt in the root directory. Ensure the text focuses on your target domain (e.g., Nigerian higher education policies, robotics documentation, AI curricula).
Step 2: Train the Custom Tokenizer
To prevent word fragmentation (such as splitting words into awkward sub-words), train a custom BPE tokenizer with a vocabulary size of 16,384:
# train_tokenizer.py example snippet
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="<|endoftext|>"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=16384,
    special_tokens=["<|endoftext|>"]
)

tokenizer.train(["corpus.txt"], trainer)
tokenizer.save("custom_gpt2_tokenizer.json")

Run the script:
python train_tokenizer.py

Step 3: Run Model Training
Configure your hyperparameters in train.py (BATCH_SIZE = 16, MAX_SEQ_LEN = 256, D_MODEL = 384, N_LAYER = 6, N_HEAD = 6) and start training:
python train.py

 * Resuming from Checkpoints: If your training session is interrupted (e.g., power outage), the training script automatically supports loading from the latest saved checkpoint (e.g., checkpoints/gpt2_model_iter_6000.pt) and resumes counting towards your target iterations.
Step 4: Generate Text
Once training is complete, use the inference script to generate domain-specific text. Use low temperature settings (0.3 to 0.5) and regex-based sentence trimming to ensure clean, professional output:
python generate.py

💡 Key Implementation Highlights
 * Hardware-Aware Routing: Automatically detects available Intel XPU hardware devices (xpu:0 / xpu:1) with automatic fallback to CPU if necessary.
 * Fault Tolerant Training: Periodic checkpointing (checkpoints/gpt2_model_iter_X.pt) ensures zero loss of progress during long training runs.
 * Clean Inference Post-Processing: Automatically strips incomplete trailing sentence fragments to guarantee grammatically complete outputs.
📝 Development Context
Personal project developed during a research internship at AI Unipod Unilag.

