#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys

# ----------------------------------------------------------------------
# 1. AUTO-INSTALL MISSING PACKAGES (before any torch import)
# ----------------------------------------------------------------------
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

required_packages = [
    "torch", "torchvision", "torchaudio", "einops", "tqdm",
    "datasets", "transformers", "tokenizers", "sentencepiece",
    "accelerate", "huggingface_hub", "requests"
]

print("Checking / installing required packages...")
for pkg in required_packages:
    try:
        __import__(pkg.replace("-", "_"))
        print(f"✓ {pkg} already installed")
    except ImportError:
        print(f"✗ Installing {pkg} ...")
        install(pkg)
        print(f"✓ {pkg} installed")

# ----------------------------------------------------------------------
# 2. IMPORTS
# ----------------------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import warnings
import os
import requests
import zipfile
import io

warnings.filterwarnings("ignore")

# Device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
print(f"Using device: {device}")

# ----------------------------------------------------------------------
# 3. MODEL DEFINITION (Sparse Griffin-Hybrid) – unchanged from before
# ----------------------------------------------------------------------
class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features, sparsity=0.85):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.sparsity = sparsity
    def forward(self, x):
        mask = torch.rand_like(x) > self.sparsity
        x = x * mask
        return self.linear(x)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

class RG_LRU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.i_proj = nn.Linear(dim, 3 * hidden_dim, bias=True)
        self.recurrent_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.log_alpha = nn.Parameter(torch.zeros(hidden_dim))
        self.log_gate = nn.Parameter(torch.zeros(hidden_dim))
        self.o_proj = nn.Linear(hidden_dim, dim, bias=True)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.i_proj.weight)
        nn.init.xavier_uniform_(self.recurrent_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        nn.init.zeros_(self.i_proj.bias)
        nn.init.zeros_(self.o_proj.bias)
        nn.init.zeros_(self.log_alpha)
        nn.init.zeros_(self.log_gate)
    def forward(self, x, state=None):
        batch, seq_len, _ = x.shape
        i_gate, i_rec, i_out = torch.split(self.i_proj(x), self.hidden_dim, dim=-1)
        i_gate = torch.sigmoid(i_gate)
        alpha = torch.exp(-torch.exp(self.log_alpha))
        gate = torch.sigmoid(self.log_gate)
        if state is None:
            state = torch.zeros(batch, self.hidden_dim, device=x.device)
        outputs = []
        for t in range(seq_len):
            u_t = i_rec[:, t, :] + self.recurrent_proj(state)
            state = gate * state + (1 - gate) * alpha * u_t
            o_t = i_out[:, t, :] * i_gate[:, t, :] + state
            outputs.append(o_t)
        output = torch.stack(outputs, dim=1)
        output = self.o_proj(output)
        return output, state

class HawkBlock(nn.Module):
    def __init__(self, dim: int, mlp_hidden: int = None, sparse_mlp: bool = True, sparsity: float = 0.85):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = 4 * dim
        self.norm1 = RMSNorm(dim)
        self.rglru = RG_LRU(dim, dim)
        self.norm2 = RMSNorm(dim)
        if sparse_mlp:
            self.mlp = nn.Sequential(
                SparseLinear(dim, mlp_hidden, sparsity=sparsity),
                nn.GELU(),
                SparseLinear(mlp_hidden, dim, sparsity=sparsity)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_hidden),
                nn.GELU(),
                nn.Linear(mlp_hidden, dim)
            )
    def forward(self, x, state=None):
        norm_x = self.norm1(x)
        rglru_out, new_state = self.rglru(norm_x, state)
        x = x + rglru_out
        norm_x2 = self.norm2(x)
        mlp_out = self.mlp(norm_x2)
        x = x + mlp_out
        return x, new_state

class SparseGriffinHybrid(nn.Module):
    def __init__(self, vocab_size=50257, dim=768, depth=12, mlp_hidden=3072, sparsity=0.85, max_seq_len=512):
        super().__init__()
        self.dim, self.depth, self.max_seq_len = dim, depth, max_seq_len
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, dim))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.blocks = nn.ModuleList([HawkBlock(dim, mlp_hidden, True, sparsity) for _ in range(depth)])
        self.final_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.reset_states()
    def reset_states(self):
        self.states = [None] * self.depth
    def forward(self, x, states=None, return_states=False):
        batch, seq_len = x.shape
        h = self.token_emb(x) + self.pos_emb[:, :seq_len, :]
        new_states = []
        for i, block in enumerate(self.blocks):
            state = states[i] if states is not None else None
            h, new_state = block(h, state)
            new_states.append(new_state)
        h = self.final_norm(h)
        logits = self.lm_head(h)
        if return_states:
            return logits, new_states
        return logits
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=50):
        self.reset_states()
        generated = input_ids.tolist() if input_ids.dim() == 1 else input_ids[0].tolist()
        for _ in range(max_new_tokens):
            if len(generated) > 1:
                x = torch.tensor([[generated[-1]]], dtype=torch.long, device=next(self.parameters()).device)
            else:
                x = torch.tensor([generated], dtype=torch.long, device=next(self.parameters()).device)
            with torch.no_grad():
                logits, new_states = self.forward(x, states=self.states, return_states=True)
                self.states = new_states
            logits = logits[0, -1, :] / temperature
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
        return generated
    def clone_for_adapt(self):
        clone = SparseGriffinHybrid(self.token_emb.num_embeddings, self.dim, self.depth, 3072, 0.85, self.max_seq_len)
        clone.load_state_dict(self.state_dict())
        return clone
    def adapt(self, support_inputs, support_labels, steps=3, lr=0.001):
        model_copy = self.clone_for_adapt()
        optimizer = torch.optim.SGD(model_copy.parameters(), lr=lr)
        for _ in range(steps):
            logits = model_copy(support_inputs)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), support_labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.load_state_dict(model_copy.state_dict())
        return self

# ----------------------------------------------------------------------
# 4. ROBUST DATASET (works even if wikitext download fails)
# ----------------------------------------------------------------------
def download_wikitext2_raw():
    """Download wikitext-2-raw from Hugging Face using the correct URL."""
    # Correct URL as of 2025 (wikitext dataset on HF)
    url = "https://huggingface.co/datasets/wikitext/resolve/main/wikitext-2-raw-v1.zip?download=1"
    print(f"Downloading wikitext-2-raw from {url} ...")
    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        # The archive contains 'wikitext-2-raw/wiki.raw'
        text = z.read("wikitext-2-raw/wiki.raw").decode("utf-8")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        print(f"Loaded {len(lines)} lines from wikitext-2-raw")
        return lines
    except Exception as e:
        print(f"Failed to download wikitext: {e}")
        return None

def get_fallback_texts():
    """A tiny built-in corpus so training never fails."""
    print("Using fallback built-in text corpus (short Wikipedia excerpt).")
    return [
        "Machine learning is a field of artificial intelligence. It enables computers to learn from data.",
        "Natural language processing helps computers understand human language. Applications include translation and chatbots.",
        "Recurrent neural networks are good at processing sequences. They have internal memory.",
        "Transformers use attention mechanisms. They are powerful for language tasks.",
        "The Sparse Griffin Hybrid model combines recurrent and sparse components for efficiency.",
        "Meta-learning allows models to adapt quickly to new tasks with few examples.",
        "Infinite context is achieved by carrying a recurrent state across arbitrarily long sequences.",
    ]

class MixedDataset(Dataset):
    def __init__(self, tokenizer, max_length=512, num_samples=10000):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Try to get wikitext, otherwise use fallback
        wiki_lines = download_wikitext2_raw()
        if wiki_lines is None:
            wiki_lines = get_fallback_texts()
        # Take a subset (70% of requested samples)
        target_wiki = int(num_samples * 0.7)
        if len(wiki_lines) > target_wiki:
            wiki_samples = wiki_lines[:target_wiki]
        else:
            wiki_samples = wiki_lines * (target_wiki // len(wiki_lines) + 1)
            wiki_samples = wiki_samples[:target_wiki]

        # Dialog: try UltraChat, fallback to synthetic
        try:
            from datasets import load_dataset
            print("Loading UltraChat (first 5000 samples)...")
            dialog = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:5000]")
            dialog_texts = []
            for sample in dialog:
                conv = [f"{msg['role'].capitalize()}: {msg['content']}" for msg in sample["messages"]]
                dialog_texts.append("\n".join(conv))
            dialog_samples = dialog_texts[:int(num_samples * 0.3)]
            print(f"Loaded {len(dialog_samples)} real dialog samples.")
        except Exception as e:
            print(f"UltraChat loading failed ({e}), using synthetic dialog fallback.")
            # Synthetic conversations – repeat to reach needed count
            base_dialog = [
                "User: Hello\nAssistant: Hi! How can I help?",
                "User: What is your name?\nAssistant: I am a helpful AI assistant.",
                "User: Tell me a joke.\nAssistant: Why don't scientists trust atoms? Because they make up everything.",
            ]
            target_dialog = int(num_samples * 0.3)
            dialog_samples = (base_dialog * (target_dialog // len(base_dialog) + 1))[:target_dialog]

        self.texts = wiki_samples + dialog_samples
        print(f"Total combined samples: {len(self.texts)}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return enc.input_ids.squeeze(0), enc.attention_mask.squeeze(0)

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# ----------------------------------------------------------------------
# 5. TRAINING LOOP
# ----------------------------------------------------------------------
def train(model, train_loader, epochs=3, lr=1e-4, grad_acc_steps=8, device="cpu"):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
    model.train()
    global_step = 0
    for epoch in range(epochs):
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad()
        for step, (input_ids, _) in enumerate(progress):
            input_ids = input_ids.to(device)
            labels = input_ids[:, 1:].contiguous()
            inputs = input_ids[:, :-1].contiguous()
            logits = model(inputs)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
            loss = loss / grad_acc_steps
            loss.backward()
            total_loss += loss.item() * grad_acc_steps
            if (step + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            progress.set_postfix(loss=total_loss / (step + 1))
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Avg loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), f"sgh_epoch_{epoch+1}.pt")

# ----------------------------------------------------------------------
# 6. MAIN
# ----------------------------------------------------------------------
def main():
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size

    model = SparseGriffinHybrid(
        vocab_size=vocab_size,
        dim=768,
        depth=12,
        mlp_hidden=3072,
        sparsity=0.85,
        max_seq_len=512
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,} (~{total_params/1e6:.1f}M)")

    print("Building dataset (with fallbacks for robustness)...")
    dataset = MixedDataset(tokenizer, max_length=512, num_samples=15000)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    print("\nStarting training...")
    train(model, dataloader, epochs=3, lr=1e-4, grad_acc_steps=8, device=device)

    torch.save(model.state_dict(), "sgh_final.pt")
    print("Training finished. Model saved as sgh_final.pt")

    # Demo generation
    print("\n--- Generation demo ---")
    model.eval()
    model.reset_states()
    prompt = "The future of artificial intelligence is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_ids = model.generate(input_ids, max_new_tokens=50, temperature=0.8, top_k=50)
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Prompt: {prompt}\nGenerated: {generated_text}")

    # Meta-learning demo
    print("\n--- Meta-learning demo (few-shot adaptation) ---")
    support_text = "User: What is the capital of France?\nAssistant: Paris is the capital of France."
    support_ids = tokenizer.encode(support_text, return_tensors="pt").to(device)
    support_labels = support_ids[:, 1:].contiguous()
    support_inputs = support_ids[:, :-1].contiguous()
    print("Adapting model (3 inner steps)...")
    model.adapt(support_inputs, support_labels, steps=3, lr=0.001)
    print("Adaptation complete.")

if __name__ == "__main__":
    main()