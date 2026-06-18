#!/usr/bin/env python3
"""
UCLT‑MAML – proof‑of‑concept that learns a specific paragraph and answers cleanly.
Runs on CPU. Resume‑friendly. No repetition in output.
"""

import os, sys, math, time, random, subprocess
from typing import Optional, Dict

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg],
                          stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
for p in ["torch", "numpy", "tokenizers", "tqdm", "datasets"]:
    install(p)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset, WeightedRandomSampler, ConcatDataset
from torch.amp import GradScaler, autocast
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tqdm import tqdm
from datasets import load_dataset

# ===================== TUNED FOR CLEAN OUTPUT =====================
DEVICE = torch.device("cpu")
GRAD_ACCUM_STEPS = 1
MICRO_BATCH = 2
SEQ_LEN = 64
VOCAB_SIZE = 1000
D_MODEL = 64
NHEAD = 2
MAX_ACT_STEPS = 3
MEM_SLOTS = 16
PONDER_LAMBDA = 0.01
LR = 5e-4                         # slightly lower
MAML_INNER_LR = 0.01
MAML_INNER_STEPS = 2
WARMUP_STEPS = 100
PHASE1_STEPS = 400                # more warm‑up
PHASE2_STEPS = 1000               # deep overfitting
CHECKPOINT_INTERVAL = 100
NUM_WORKERS = 0

# ===================== 1. Tokenizer =====================
def train_tokenizer(save_path=None, vocab_size=VOCAB_SIZE):
    if save_path is None:
        save_path = f"tokenizer_vocab{vocab_size}.json"
    if os.path.exists(save_path):
        return Tokenizer.from_file(save_path)

    texts = []
    try:
        c4 = load_dataset("allenai/c4", "en", split="train", streaming=True)
        for i, sample in enumerate(c4):
            if i >= 1000: break
            t = sample["text"]
            if t and len(t.strip()) > 50:
                texts.append(t.strip())
    except:
        nouns = ["dog", "cat", "house", "car", "tree", "computer", "book", "city", "ocean", "mountain"]
        verbs = ["runs", "jumps", "reads", "writes", "builds", "destroys", "loves", "hates", "eats", "drinks"]
        for _ in range(1000):
            texts.append(f"The {random.choice(nouns)} {random.choice(verbs)} the {random.choice(nouns)}. " * 2)
    if len(texts) < 100:
        texts = ["The quick brown fox jumps over the lazy dog. " * 10] * 500

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<reason>", "<fast>"],
    )
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.decoder = decoders.BPEDecoder()
    tokenizer.save(save_path)
    return tokenizer

# ===================== 2. Datasets =====================
class C4StreamingDataset(IterableDataset):
    def __init__(self, tokenizer, max_len=SEQ_LEN):
        self.tokenizer = tokenizer
        self.max_len = max_len
        try:
            self.dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
        except:
            self.dataset = None

    def __iter__(self):
        if self.dataset is not None:
            for sample in self.dataset:
                text = sample["text"]
                if not text or len(text) < 50:
                    continue
                ids = self.tokenizer.encode(text).ids
                for i in range(0, len(ids), self.max_len):
                    chunk = ids[i:i+self.max_len]
                    if len(chunk) > 1:
                        if len(chunk) < self.max_len:
                            chunk += [self.tokenizer.token_to_id("<pad>")] * (self.max_len - len(chunk))
                        yield torch.tensor(chunk, dtype=torch.long)
        else:
            while True:
                text = " ".join(["the quick brown fox jumps over the lazy dog."] * 4)
                ids = self.tokenizer.encode(text).ids[:self.max_len]
                if len(ids) > 1:
                    if len(ids) < self.max_len:
                        ids += [self.tokenizer.token_to_id("<pad>")] * (self.max_len - len(ids))
                    yield torch.tensor(ids, dtype=torch.long)

class ChunkedDataset(Dataset):
    def __init__(self, tokenizer, texts, max_len=SEQ_LEN):
        self.input_ids = []
        for text in tqdm(texts, desc="Chunking"):
            try:
                ids = tokenizer.encode(text).ids
            except:
                continue
            for i in range(0, len(ids), max_len):
                chunk = ids[i:i+max_len]
                if len(chunk) > 1:
                    if len(chunk) < max_len:
                        chunk += [tokenizer.token_to_id("<pad>")] * (max_len - len(chunk))
                    self.input_ids.append(torch.tensor(chunk, dtype=torch.long))
        if not self.input_ids:
            self.input_ids = [torch.zeros(max_len, dtype=torch.long)]

    def __len__(self): return len(self.input_ids)
    def __getitem__(self, idx): return self.input_ids[idx]

# ===================== 3. Model (unchanged) =====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        res = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, x, x)
        x = res + self.dropout(attn_out)
        res = x
        x = self.norm2(x)
        ff = self.linear2(F.gelu(self.linear1(x)))
        x = res + self.dropout(ff)
        return x

class UniversalTransformerACT(nn.Module):
    def __init__(self, d_model, nhead, max_steps=MAX_ACT_STEPS):
        super().__init__()
        self.block = TransformerBlock(d_model, nhead)
        self.halt_linear = nn.Linear(d_model, 1)
        self.max_steps = max_steps
    def forward(self, x):
        batch, seq_len, d_model = x.shape
        halting_prob = torch.zeros(batch, seq_len, 1, device=x.device)
        n_up = torch.zeros_like(halting_prob)
        output = torch.zeros_like(x)
        state = x
        ponder_cost = 0.0
        for t in range(self.max_steps):
            state = self.block(state)
            p = torch.sigmoid(self.halt_linear(state))
            running = (halting_prob < 1.0).float()
            new_halted = running * p * ((halting_prob + p) <= 1.0).float()
            halting_prob += new_halted
            n_up += running
            output += new_halted * state
            ponder_cost += running.sum()
            if halting_prob.ge(1.0).all():
                break
        remainder = 1.0 - halting_prob
        output += remainder * state
        n_up += remainder
        output = output / n_up.clamp(min=1e-8)
        ponder_cost = ponder_cost / (batch * seq_len)
        return output, ponder_cost

class FastWeightMemory(nn.Module):
    def __init__(self, base_model, max_skills=5):
        super().__init__()
        self.base_model = base_model
        self.fast_memory: Dict[str, Dict[str, torch.Tensor]] = {}
        self.max_skills = max_skills
    def get_adapted_weights(self, skill_key):
        if skill_key not in self.fast_memory:
            delta = {n: torch.zeros_like(p) for n, p in self.base_model.named_parameters()}
            self.fast_memory[skill_key] = delta
        return self.fast_memory[skill_key]
    def update_delta(self, skill_key, delta):
        self.fast_memory[skill_key] = delta
        if len(self.fast_memory) > self.max_skills:
            oldest = next(iter(self.fast_memory))
            del self.fast_memory[oldest]

class ExternalMemory(nn.Module):
    def __init__(self, num_slots, slot_dim):
        super().__init__()
        self.register_buffer("slots", torch.randn(num_slots, slot_dim) * 0.02)
        self.register_buffer("usage", torch.zeros(num_slots))
    def read(self, query):
        query = F.normalize(query, dim=-1)
        slots_norm = F.normalize(self.slots, dim=-1)
        attn = torch.matmul(query, slots_norm.T)
        attn = F.softmax(attn / 0.1, dim=-1)
        return torch.matmul(attn, self.slots)
    def write(self, value):
        with torch.no_grad():
            idx = torch.argmin(self.usage)
            self.slots[idx] = value.mean(dim=0)
            self.usage[idx] += 1.0
            self.usage *= 0.99

class UCLT_MAML(nn.Module):
    def __init__(self, vocab_size, d_model=D_MODEL, nhead=NHEAD,
                 pad_token_id=0, reason_token_id=None, fast_token_id=None):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.reason_token_id = reason_token_id
        self.fast_token_id = fast_token_id

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.ut = UniversalTransformerACT(d_model, nhead)
        self.memory = ExternalMemory(MEM_SLOTS, d_model)
        self.mem_proj = nn.Linear(d_model, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.reason_head = nn.Linear(d_model, vocab_size)
        self.logic_head = nn.Linear(d_model, 1)
        self.curiosity_head = nn.Linear(d_model, d_model)
        self.lm_head.weight = self.embedding.weight
        self.dropout = nn.Dropout(0.1)
        self.fast_memory = FastWeightMemory(self.ut)

    def forward(self, input_ids, use_memory=True, use_fast_weights=True, skill_key=None):
        mask = (input_ids != self.pad_token_id).float()
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)
        x = self.dropout(x)

        if use_fast_weights and skill_key is not None:
            delta = self.fast_memory.get_adapted_weights(skill_key)
            with torch.no_grad():
                for n, p in self.ut.named_parameters():
                    p.data = p.data + delta[n].to(p.device)

        out, ponder = self.ut(x)

        if use_fast_weights and skill_key is not None:
            with torch.no_grad():
                for n, p in self.ut.named_parameters():
                    p.data = p.data - delta[n].to(p.device)

        seq_repr = (out * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        if use_memory:
            mem = self.memory.read(seq_repr)
            mem = self.mem_proj(mem)
            out = out + mem.unsqueeze(1)

        return {
            "logits": self.lm_head(out),
            "reason_logits": self.reason_head(out),
            "logic_score": torch.sigmoid(self.logic_head(out)),
            "hidden": out,
            "seq_repr": seq_repr,
            "ponder": ponder,
        }

    def write_memory(self, embedding):
        self.memory.write(embedding)

# ===================== 4. MAML helpers =====================
from torch.func import functional_call

def maml_inner_loop(model, input_ids, loss_fn, inner_lr=MAML_INNER_LR, steps=MAML_INNER_STEPS):
    adapted = {n: p.clone().detach().requires_grad_(True)
               for n, p in model.ut.named_parameters()}

    for _ in range(steps):
        def forward_with_adapted(adapted_weights):
            x = model.embedding(input_ids)
            x = model.pos_encoder(x)
            x = model.dropout(x)
            out, ponder = functional_call(model.ut, adapted_weights, (x,))
            mask = (input_ids != model.pad_token_id).float()
            seq_repr = (out * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            mem = model.mem_proj(model.memory.read(seq_repr))
            out = out + mem.unsqueeze(1)
            return {
                "logits": model.lm_head(out),
                "reason_logits": model.reason_head(out),
                "ponder": ponder,
            }

        out = forward_with_adapted(adapted)
        loss = loss_fn(out, input_ids)
        grads = torch.autograd.grad(loss, list(adapted.values()), create_graph=False, allow_unused=False)
        adapted = {n: adapted[n] - inner_lr * g
                   for (n, _), g in zip(adapted.items(), grads)}

    original = {n: p.data for n, p in model.ut.named_parameters()}
    delta = {n: (adapted[n].detach() - original[n]) for n in original}
    return delta

def get_cosine_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def save_checkpoint(obj, path):
    torch.save(obj, path)

def load_checkpoint(path, device):
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device)
        print(f"Loaded checkpoint from {path} (step {ckpt.get('step',0)})")
        return ckpt
    return None

def generate_cot_batch(tokenizer, batch_size=MICRO_BATCH):
    examples = []
    templates = [
        ("What is {a} + {b}?", "Let's add {a} and {b}. {a} + {b} = {c}. The answer is {c}."),
        ("If I have {a} apples and give away {b}, how many left?",
         "Start with {a} apples, give away {b}, so {a} - {b} = {c}. The answer is {c}."),
    ]
    for _ in range(batch_size):
        a = np.random.randint(1, 10)
        b = np.random.randint(1, 10)
        op = "+" if np.random.rand() < 0.5 else "-"
        c = a + b if op == "+" else a - b
        t = templates[np.random.randint(len(templates))]
        examples.append(t[0].format(a=a, b=b) + " " + t[1].format(a=a, b=b, c=c))
    enc = tokenizer.encode_batch(examples)
    max_len = max(len(e.ids) for e in enc)
    input_ids = []
    for e in enc:
        ids = e.ids + [tokenizer.token_to_id("<pad>")] * (max_len - len(e.ids))
        input_ids.append(ids)
    return torch.tensor(input_ids, dtype=torch.long)

# ===================== 5. Main =====================
def main():
    device = DEVICE
    scaler = GradScaler('cpu')
    print(f"Device: {device}")

    # Tokenizer
    tokenizer = train_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("<pad>")
    reason_id = tokenizer.token_to_id("<reason>")
    fast_id = tokenizer.token_to_id("<fast>")
    print(f"Vocabulary size: {vocab_size}")

    ckpt_path = f"uclt_maml_vocab{vocab_size}_latest.pt"

    # Model
    model = UCLT_MAML(vocab_size, pad_token_id=pad_id, reason_token_id=reason_id,
                     fast_token_id=fast_id).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = PHASE1_STEPS + PHASE2_STEPS
    scheduler = get_cosine_schedule(optimizer, WARMUP_STEPS, total_steps)

    # Resume checkpoint
    ckpt = load_checkpoint(ckpt_path, device)
    start_step = 0
    if ckpt is not None:
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_step = ckpt['step']
        print(f"Resumed from step {start_step}")

    # Phase 1: C4 pre‑training
    if start_step < PHASE1_STEPS:
        print("\n=== Phase 1: C4 pre‑training ===")
        c4_dataset = C4StreamingDataset(tokenizer)
        c4_loader = DataLoader(c4_dataset, batch_size=MICRO_BATCH, num_workers=0)
        model.train()
        optimizer.zero_grad()
        pbar = tqdm(c4_loader, initial=start_step, total=PHASE1_STEPS, desc="C4 training")
        for step, input_ids in enumerate(pbar, start=start_step):
            if step >= PHASE1_STEPS: break
            input_ids = input_ids.to(device)
            with autocast('cpu'):
                out = model(input_ids, use_fast_weights=False)
                shift_logits = out["logits"][:, :-1, :]
                shift_labels = input_ids[:, 1:]
                loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)),
                                       shift_labels.reshape(-1), ignore_index=pad_id)
                ponder_loss = PONDER_LAMBDA * out["ponder"]
                loss = loss + ponder_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix(loss=f"{loss.item():.3f}", lr=optimizer.param_groups[0]['lr'])
            if step % CHECKPOINT_INTERVAL == 0 and step > start_step:
                save_checkpoint({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss.item()
                }, ckpt_path)
        save_checkpoint({
            'step': PHASE1_STEPS,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': 0.0
        }, ckpt_path)
        start_step = PHASE1_STEPS

    # Phase 2: Learn the custom AI paragraph
    print("\n=== Phase 2: Learning the custom AI paragraph ===")
    custom_paragraph = (
        "Artificial intelligence (AI) is the simulation of human intelligence in machines. "
        "AI systems can learn from data, recognize patterns, and make decisions. "
        "Modern AI uses deep learning and neural networks. "
        "Applications include natural language processing, computer vision, and robotics. "
        "AI is transforming industries like healthcare, finance, and transportation."
    )
    # Repeat many times – the model needs to see it frequently
    texts = [custom_paragraph] * 500
    custom_dataset = ChunkedDataset(tokenizer, texts, max_len=SEQ_LEN)
    print(f"Custom dataset size: {len(custom_dataset)} chunks")
    custom_loader = DataLoader(custom_dataset, batch_size=MICRO_BATCH, shuffle=True, num_workers=0)

    def cot_generator():
        while True:
            yield generate_cot_batch(tokenizer, MICRO_BATCH)
    cot_iter = iter(cot_generator())

    model.train()
    optimizer.zero_grad()
    pbar = tqdm(range(PHASE2_STEPS), desc="Custom paragraph + MAML")
    for step in pbar:
        # 80% custom paragraph, 20% CoT (keeps reasoning heads active)
        if np.random.rand() < 0.8:
            try:
                batch = next(iter(custom_loader))
            except StopIteration:
                continue
            input_ids = batch.to(device)
            use_reason = False
        else:
            input_ids = next(cot_iter).to(device)
            use_reason = True

        def loss_fn(out, input_ids):
            shift_logits = out["reason_logits"][:, :-1, :] if use_reason else out["logits"][:, :-1, :]
            shift_labels = input_ids[:, 1:]
            return F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)),
                                   shift_labels.reshape(-1), ignore_index=pad_id)

        # MAML inner loop
        delta = maml_inner_loop(model, input_ids, loss_fn)
        model.fast_memory.update_delta("task_default", delta)

        # Outer step
        with autocast('cpu'):
            out = model(input_ids, use_fast_weights=True, skill_key="task_default")
            loss = loss_fn(out, input_ids)
            ponder_loss = PONDER_LAMBDA * out["ponder"]
            loss = loss + ponder_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        pbar.set_postfix(loss=f"{loss.item():.3f}")

        if step % CHECKPOINT_INTERVAL == 0:
            save_checkpoint({
                'step': start_step + step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss.item()
            }, ckpt_path)

    save_checkpoint({
        'step': total_steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': 0.0
    }, ckpt_path)

    # ===================== GENERATION with n‑gram blocking =====================
    print("\nGenerating final answer (no repetition)...")
    model.eval()
    prompt = "What is artificial intelligence?"
    enc = tokenizer.encode(prompt)
    input_ids = torch.tensor([enc.ids], device=device)
    eos_id = tokenizer.token_to_id("</s>")
    pad_id = tokenizer.token_to_id("<pad>")

    generated = enc.ids.copy()   # list of token IDs for n‑gram checking

    with torch.no_grad():
        for _ in range(40):
            out = model(input_ids, use_fast_weights=True, skill_key="task_default")
            logits = out["logits"][0, -1, :] / 0.8

            # Block pad & EOS
            logits[pad_id] = -float('Inf')
            # Block 2‑gram repetitions (prevents loops)
            if len(generated) >= 2:
                last_bigram = tuple(generated[-2:])
                # Find all occurrences of this bigram earlier in the sequence
                for i in range(len(generated)-2):
                    if tuple(generated[i:i+2]) == last_bigram:
                        next_tok = generated[i+2] if i+2 < len(generated) else None
                        if next_tok is not None and next_tok != pad_id:
                            logits[next_tok] -= 1e5

            # Pick the highest probability token
            next_token = torch.argmax(logits).item()
            generated.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
            if next_token == eos_id:
                break

    answer = tokenizer.decode(generated, skip_special_tokens=True)
    # Clean up prompt from answer
    if answer.startswith(prompt):
        answer = answer[len(prompt):].strip()
    print(f"Prompt: {prompt}")
    print(f"Response: {answer}")
    print("(The model should now output a clean, coherent definition.)")

if __name__ == "__main__":
    main()