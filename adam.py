import numpy as np
from collections import defaultdict, Counter
import re, random, sys, threading, os, torch, torch.nn as nn, torch.optim as optim
from torch.nn import functional as F
from concurrent.futures import ThreadPoolExecutor
from math import sqrt

# ============================================================
# 1. UNIVERSAL TRANSFORMER (LARGER CAPACITY)
# ============================================================
class UniversalTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.2):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class UniversalTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, d_ff=512, num_steps=4, max_len=256, dropout=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_steps = num_steps
        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.block = UniversalTransformerBlock(d_model, n_heads, d_ff, dropout)
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        b, t = input_ids.size()
        positions = torch.arange(0, t, device=input_ids.device).unsqueeze(0).expand(b, -1)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        causal_mask = torch.triu(torch.ones(t, t, device=input_ids.device) * float('-inf'), diagonal=1)
        for _ in range(self.num_steps):
            x = self.block(x, mask=causal_mask)
        x = self.ln_final(x)
        return self.head(x)

    def probability(self, context_ids, token_id):
        self.eval()
        with torch.no_grad():
            input_tensor = torch.tensor([context_ids], dtype=torch.long)
            logits = self.forward(input_tensor)
            last_logits = logits[0, -1, :] / 0.8          # temperature 0.8
            probs = F.softmax(last_logits, dim=-1)
            return probs[token_id].item()

    def generate_token(self, context_ids, top_p=0.9):
        """Nucleus sampling: returns (token_id, token_str)."""
        self.eval()
        with torch.no_grad():
            input_tensor = torch.tensor([context_ids], dtype=torch.long)
            logits = self.forward(input_tensor)
            last_logits = logits[0, -1, :] / 0.8
            probs = F.softmax(last_logits, dim=-1)

            # Top‑p (nucleus) filtering
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumsum > top_p
            cutoff[0] = False
            filtered_probs = sorted_probs.clone()
            filtered_probs[cutoff] = 0.0
            filtered_probs = filtered_probs / filtered_probs.sum()
            idx = torch.multinomial(filtered_probs, 1).item()
            token_id = sorted_indices[idx].item()
            return token_id


# ============================================================
# 2. TRAINING UTILITIES
# ============================================================
def tokenize_sentence(text, word2idx):
    tokens = re.findall(r"\b[\w']+\b|[.?!]", text.lower())
    return [word2idx.get(t, word2idx['<unk>']) for t in tokens]

def build_vocab(texts):
    word2idx = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
    idx2word = ['<pad>', '<unk>', '<s>', '</s>']
    for text in texts:
        for token in re.findall(r"\b[\w']+\b|[.?!]", text.lower()):
            if token not in word2idx:
                word2idx[token] = len(word2idx)
                idx2word.append(token)
    return word2idx, idx2word

def train_universal_transformer(model, texts, word2idx, epochs=100, lr=2e-3, device='cpu'):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])

    sequences = []
    for text in texts:
        ids = tokenize_sentence(text, word2idx)
        if len(ids) > 0:
            sequences.append(ids)

    for epoch in range(epochs):
        total_loss = 0.0
        random.shuffle(sequences)
        for seq in sequences:
            if len(seq) < 2:
                continue
            input_ids = torch.tensor([seq[:-1]], dtype=torch.long).to(device)
            target_ids = torch.tensor([seq[1:]], dtype=torch.long).to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, model.vocab_size), target_ids.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch+1) % 10 == 0:
            print(f"  UT epoch {epoch+1}/{epochs}, loss: {total_loss/len(sequences):.4f}")
    print("Universal Transformer trained.")


# ============================================================
# 3. THE N‑GRAM + UT HYBRID (only UT used for scoring)
# ============================================================
class AdamV36Consistency:
    def __init__(self, orders: list[int] = [2, 3, 4, 5, 6, 7, 8]):
        self.orders = orders
        self.models = {n: defaultdict(Counter) for n in orders}
        self.context_totals = {n: defaultdict(int) for n in orders}
        self.vocabulary = set()
        self.domain_map = defaultdict(set)
        self.global_counts = Counter()
        self.total_tokens = 0
        self.doc_counts = []
        self.doc_norms = None
        self.idf = {}

        self.stop_tokens = {".", "!", "?"}
        self.global_sequence_blacklist = set()
        self.run_history = Counter()
        self.lock = threading.Lock()

        self.ut_model = None
        self.word2idx = None
        self.idx2word = None

    def tokenize(self, text: str) -> list[str]:
        text = text.lower()
        return re.findall(r"\b[\w']+\b|[.?!]", text)

    def _train_single_doc(self, idx: int, text: str):
        tokens = self.tokenize(text)
        local_vocab = set(tokens)
        local_counts = Counter(tokens)
        local_domain_map = defaultdict(set)
        local_models = {n: defaultdict(Counter) for n in self.orders}
        local_context_totals = {n: defaultdict(int) for n in self.orders}

        for t in tokens:
            local_domain_map[t].add(idx)

        for n in self.orders:
            for i in range(len(tokens) - n + 1):
                ctx = tuple(tokens[i:i+n-1])
                nxt = tokens[i+n-1]
                local_models[n][ctx][nxt] += 1
                local_context_totals[n][ctx] += 1

        return {
            'vocab': local_vocab,
            'counts': local_counts,
            'domain_map': local_domain_map,
            'models': local_models,
            'context_totals': local_context_totals
        }

    def train_adam(self, text_list: list[str]):
        self.num_docs = len(text_list)
        print(f"Training on {self.num_docs} documents with threading...")
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(self._train_single_doc, idx, text)
                       for idx, text in enumerate(text_list)]
            results = [future.result() for future in futures]

        print("Merging training results...")
        self.doc_counts = [None] * self.num_docs
        for idx, result in enumerate(results):
            self.vocabulary.update(result['vocab'])
            self.global_counts.update(result['counts'])
            self.total_tokens += sum(result['counts'].values())
            self.doc_counts[idx] = result['counts']

            for token, docs in result['domain_map'].items():
                self.domain_map[token].update(docs)

            for n in self.orders:
                for ctx, counter in result['models'][n].items():
                    self.models[n][ctx].update(counter)
                for ctx, total in result['context_totals'][n].items():
                    self.context_totals[n][ctx] += total

        # IDF and norms (kept for potential use)
        doc_freq = Counter()
        for token in self.domain_map:
            doc_freq[token] = len(self.domain_map[token])
        for token, df in doc_freq.items():
            self.idf[token] = np.log((self.num_docs + 1) / (df + 1)) + 1.0
        self.doc_norms = np.zeros(self.num_docs)
        for i, cnt in enumerate(self.doc_counts):
            norm_sq = sum((tf * self.idf.get(t, 1.0))**2 for t, tf in cnt.items())
            self.doc_norms[i] = sqrt(norm_sq)

        print("N‑gram tables built.")

        # ---- Build vocab and train/save UT ----
        self.word2idx, self.idx2word = build_vocab(text_list)
        vocab_size = len(self.word2idx)
        self.ut_model = UniversalTransformerLM(vocab_size, d_model=256, n_heads=8, d_ff=512,
                                               num_steps=4, max_len=256, dropout=0.2)

        model_path = "ut_adam.pt"
        if os.path.exists(model_path):
            print(f"Loading pre‑trained UT from {model_path}...")
            self.ut_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.ut_model.eval()
        else:
            print("Training Universal Transformer on", len(text_list), "utterances...")
            train_universal_transformer(self.ut_model, text_list, self.word2idx,
                                        epochs=100, lr=2e-3, device='cpu')
            torch.save(self.ut_model.state_dict(), model_path)
            print(f"UT saved to {model_path}")

    # ------------------------------------------------------------
    # Cosine TF‑IDF (still used for topic locking)
    # ------------------------------------------------------------
    def _cosine_doc_scores(self, tokens: list[str]) -> np.ndarray:
        prompt_tf = Counter(tokens)
        prompt_vec = {}
        for t, tf in prompt_tf.items():
            if t in self.idf:
                prompt_vec[t] = tf * self.idf[t]
        norm = sqrt(sum(v*v for v in prompt_vec.values()))
        if norm == 0: return np.zeros(self.num_docs)
        scores = np.zeros(self.num_docs)
        for i in range(self.num_docs):
            cnt = self.doc_counts[i]
            dot = 0.0
            for t, w in prompt_vec.items():
                if t in cnt:
                    dot += w * (cnt[t] * self.idf[t])
            doc_norm = self.doc_norms[i] if self.doc_norms[i] > 0 else 1e-9
            scores[i] = dot / (norm * doc_norm)
        return scores

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def detect_repetition_loop(self, response, window=10):
        if len(response) < window: return False
        recent = response[-window:]
        return len(set(recent)) / len(recent) < 0.4

    def get_token_rarity_weight(self, token):
        occ = self.global_counts.get(token, 1)
        return 1.0 / (1.0 + np.log(occ + 1))

    def calculate_diversity_score(self, response, window=15):
        if len(response) < 5: return 1.0
        recent = response[-window:]
        return len(set(recent)) / len(recent) if recent else 1.0

    def is_token_recently_used(self, token, response, lookback=5):
        if len(response) < lookback: return token in response
        return token in response[-lookback:]

    # ------------------------------------------------------------
    # Generation loop (UT with nucleus sampling)
    # ------------------------------------------------------------
    def chat(self, user_input: str, max_tokens: int = 70, min_tokens: int = 20):
        input_tokens = self.tokenize(user_input)

        # Topic locking (still done via TF‑IDF for vocabulary filter, optional)
        doc_scores = self._cosine_doc_scores(input_tokens)
        if doc_scores.max() > 0:
            threshold = doc_scores.max() * 0.2
            relevant = set(np.where(doc_scores >= threshold)[0])
            if len(relevant) > 12:
                idx_top = np.argsort(doc_scores)[-12:]
                relevant = set(idx_top)
        else:
            relevant = set()
        if not relevant:
            document_momentum = Counter()
            for t in input_tokens:
                if t in self.domain_map:
                    rarity_weight = self.get_token_rarity_weight(t)
                    base_weight = 20.0 if rarity_weight > 0.5 else 15.0
                    for doc_id in self.domain_map[t]:
                        document_momentum[doc_id] += rarity_weight * base_weight
            relevant = {doc for doc, _ in document_momentum.most_common(10)}

        # We'll use a mild vocabulary filter based on locked docs (still statistical)
        allowed_vocab = set()
        for d in relevant:
            allowed_vocab.update(self.doc_counts[d].keys())
        allowed_vocab.update(self.stop_tokens)
        # Add any token that appears in at least 30% of all docs (common words)
        for token, df in self.idf.items():
            if df >= 0.3 * self.num_docs:
                allowed_vocab.add(token)

        context = input_tokens.copy()
        response = []
        loop_detected = 0
        ended_properly = False

        for i in range(max_tokens):
            if i > 0 and response[-1] in self.stop_tokens:
                if i >= min_tokens:
                    ended_properly = True
                    break

            if self.detect_repetition_loop(response):
                loop_detected += 1
                if loop_detected >= 4:
                    if response and response[-1] not in self.stop_tokens:
                        response.append(".")
                        ended_properly = True
                    break
            else:
                loop_detected = max(0, loop_detected - 1)

            context_ids = [self.word2idx.get(t, self.word2idx['<unk>']) for t in context]
            # Get next token from UT using nucleus sampling
            token_id = self.ut_model.generate_token(context_ids, top_p=0.9)
            selected = self.idx2word[token_id]

            # If token is not in allowed_vocab, re‑sample (softly)
            if selected not in allowed_vocab and selected not in self.stop_tokens:
                # Re‑sample a few times to avoid a hard block
                for _ in range(5):
                    token_id = self.ut_model.generate_token(context_ids, top_p=0.9)
                    selected = self.idx2word[token_id]
                    if selected in allowed_vocab or selected in self.stop_tokens:
                        break

            if selected in self.stop_tokens and len(response) < min_tokens:
                continue   # don't stop too early

            sys.stdout.write(f"\r[T{i:03d}] {selected}...")
            sys.stdout.flush()

            response.append(selected)
            context.append(selected)

            if selected in self.stop_tokens:
                ended_properly = True
                break

        if not ended_properly and response:
            if response[-1] not in self.stop_tokens:
                response.append(".")

        sys.stdout.write("\n")
        result = self.cleanup_response(" ".join(response))
        return result

    def cleanup_response(self, response):
        words = response.split()
        cleaned = []
        seen_bigrams = set()
        for w in words:
            if len(cleaned) >= 1 and cleaned[-1] == w: continue
            if len(cleaned) >= 1:
                bigram = (cleaned[-1], w)
                if bigram in seen_bigrams and len(cleaned) > 8: continue
                seen_bigrams.add(bigram)
            if len(cleaned) >= 6:
                if tuple(cleaned[-3:]) == tuple(cleaned[-6:-3]): continue
            cleaned.append(w)
        return re.sub(r"^[.?!, ]+", "", " ".join(cleaned)).strip().capitalize()


if __name__ == "__main__":
    from data import dialog as data

    adam = AdamV36Consistency()
    adam.train_adam(data)

    test_prompts = [
        "who am I, where am I??",
        "what do I see?",
        "where am I?",
        "who am I?",
        "what do I see?",
        "where am I?",
        "Am I a Human?",
        "Whats AI",
        "Am I and artificial",
        "Do I have a name?",
        "What's my name",
        "I like to go out with my friends",
        "My favourite food is",
        "Can you tell me what did you eat?",
        "Whats your favourite color?",
    ]

    for run in range(1, 6):
        prompt = random.choice(test_prompts)
        print(f"\nRUN {run:02d} | Input: {prompt}")
        result = adam.chat(prompt, max_tokens=70, min_tokens=20)
        for w in adam.tokenize(result):
            adam.run_history[w] += 0.5
        print(f"FINAL OUTPUT: {result}")
        print("-" * 85)