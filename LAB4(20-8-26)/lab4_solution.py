import os
import math
import random
from collections import defaultdict, Counter
import numpy as np

# Use exact same splits and seed as Lab 3
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "LAB3(13-8-26)", "gujarati_sentences.txt")

def length_stratified_split(sentences, dev_size=1000, test_size=1000, seed=RANDOM_SEED):
    rng = random.Random(seed)
    indexed = [(i, len(s.split())) for i, s in enumerate(sentences)]
    lengths = [wc for _, wc in indexed]
    percentiles = np.percentile(lengths, np.arange(0, 101, 10))
    bin_edges = sorted(set(percentiles.astype(int)))
    bins = defaultdict(list)
    for idx, wc in indexed:
        b = 0
        for j in range(len(bin_edges) - 1):
            if wc >= bin_edges[j]:
                b = j
        bins[b].append(idx)
    for b in bins:
        rng.shuffle(bins[b])
    total = sum(len(v) for v in bins.values())
    needed = dev_size + test_size
    dev_indices, test_indices = [], []
    for b in sorted(bins.keys()):
        proportion = len(bins[b]) / total
        n_from_bin = max(1, int(proportion * (needed // 2)))
        taken_dev = bins[b][:n_from_bin]
        taken_test = bins[b][n_from_bin:n_from_bin * 2]
        dev_indices.extend(taken_dev)
        test_indices.extend(taken_test)
    rng.shuffle(dev_indices)
    rng.shuffle(test_indices)
    dev_indices = set(dev_indices[:dev_size])
    test_indices = set(test_indices[:test_size])
    test_indices -= dev_indices
    used = dev_indices | test_indices
    remaining = [i for i in range(len(sentences)) if i not in used]
    rng.shuffle(remaining)
    while len(dev_indices) < dev_size and remaining:
        dev_indices.add(remaining.pop())
    while len(test_indices) < test_size and remaining:
        test_indices.add(remaining.pop())
    train_indices = [i for i in range(len(sentences)) if i not in dev_indices and i not in test_indices]
    dev = [sentences[i] for i in sorted(dev_indices)]
    test = [sentences[i] for i in sorted(test_indices)]
    train = [sentences[i] for i in train_indices]
    return train, dev, test

def tokenize(text):
    return text.strip().split()

class NGramModel:
    def __init__(self, n, smoothing="none", k=0.1, lam=0.5, d=0.75):
        self.n = n
        self.smoothing = smoothing
        self.k = k
        self.lam = lam
        self.d = d
        
        self.ngrams = Counter()
        self.context_counts = Counter()
        self.vocab = set()
        self.vocab_size = 0
        
        # For Kneser-Ney
        self.continuation_counts = Counter() # N( • w )
        self.context_types = Counter()       # N( ctx • )
        
        self.lower_model = None

    def get_ngrams(self, tokens):
        padded = ['<s>'] * (self.n - 1) + tokens + ['</s>']
        return [tuple(padded[i:i+self.n]) for i in range(len(padded) - self.n + 1)]
    
    def train(self, sentences):
        print(f"Training {self.n}-gram model with {self.smoothing} smoothing...")
        
        # First pass: build vocab
        for s in sentences:
            tokens = tokenize(s)
            self.vocab.update(tokens)
        self.vocab.add('<s>')
        self.vocab.add('</s>')
        self.vocab.add('<UNK>')
        self.vocab_size = len(self.vocab)
        
        # Build lower order models for interpolation and KN
        if self.smoothing in ["interpolated", "kneser-ney"] and self.n > 1:
            self.lower_model = NGramModel(self.n - 1, smoothing=self.smoothing, k=self.k, lam=self.lam, d=self.d)
            self.lower_model.vocab = self.vocab
            self.lower_model.vocab_size = self.vocab_size
            curr = self.lower_model
            while curr.n > 1:
                curr.lower_model = NGramModel(curr.n - 1, smoothing=self.smoothing, k=self.k, lam=self.lam, d=self.d)
                curr.lower_model.vocab = self.vocab
                curr.lower_model.vocab_size = self.vocab_size
                curr = curr.lower_model
                
        # Second pass: count ngrams
        for s in sentences:
            tokens = tokenize(s)
            
            for ngram in self.get_ngrams(tokens):
                self.ngrams[ngram] += 1
                context = ngram[:-1]
                self.context_counts[context] += 1
                
                # For KN:
                if self.smoothing == "kneser-ney":
                    word = ngram[-1]
                    # N( • w ) : number of different contexts w has appeared in
                    if self.ngrams[ngram] == 1: 
                        self.continuation_counts[word] += 1
                        self.context_types[context] += 1
                        
            # Train lower models inline
            if self.smoothing in ["interpolated", "kneser-ney"] and self.n > 1:
                curr = self.lower_model
                while curr is not None:
                    for ngram in curr.get_ngrams(tokens):
                        curr.ngrams[ngram] += 1
                        context = ngram[:-1]
                        curr.context_counts[context] += 1
                        if curr.smoothing == "kneser-ney":
                            word = ngram[-1]
                            if curr.ngrams[ngram] == 1:
                                curr.continuation_counts[word] += 1
                                curr.context_types[context] += 1
                    curr = curr.lower_model

    def _prob_unsmoothed(self, ngram):
        context = ngram[:-1]
        count = self.ngrams[ngram]
        ctx_count = self.context_counts[context] if self.n > 1 else sum(self.ngrams.values())
        return count / ctx_count if ctx_count > 0 else 0.0

    def _prob_add_k(self, ngram, k):
        context = ngram[:-1]
        count = self.ngrams[ngram]
        ctx_count = self.context_counts[context] if self.n > 1 else sum(self.ngrams.values())
        return (count + k) / (ctx_count + k * self.vocab_size)

    def _prob_interpolated(self, ngram):
        p_ml = self._prob_unsmoothed(ngram)
        if self.n == 1:
            return self.lam * p_ml + (1 - self.lam) / self.vocab_size
        return self.lam * p_ml + (1 - self.lam) * self.lower_model._prob_interpolated(ngram[1:])

    def _prob_kneser_ney(self, ngram, is_highest_order=True):
        """
        Chen & Goodman (1998) Kneser-Ney.
        - Highest-order: discounted raw count / context count.
        - Lower-order backoff: continuation probability P_KN(w).
        - Unigram base: add-1 MLE (highest-order) | continuation (backoff).
        """
        if self.n == 1:
            if is_highest_order:
                return self._prob_add_k(ngram, 1.0)   # FIX: proper add-1, not tiny-k hack
            # Continuation probability
            word = ngram[-1]
            total_types = sum(self.continuation_counts.values())
            if total_types == 0:
                return 1.0 / self.vocab_size
            c = self.continuation_counts.get(word, 0)
            if c == 0:
                return 1.0 / self.vocab_size
            return c / total_types

        context = ngram[:-1]
        count = self.ngrams[ngram]
        ctx_count = self.context_counts[context]
        num_types = self.context_types[context]

        if ctx_count == 0 or num_types == 0:
            discounted_prob, alpha = 0.0, 1.0
        else:
            discounted_prob = max(count - self.d, 0.0) / ctx_count
            alpha = (self.d / ctx_count) * num_types

        p_lower = self.lower_model._prob_kneser_ney(ngram[1:], is_highest_order=False)
        return discounted_prob + alpha * p_lower

    def prob(self, ngram):
        if self.smoothing == "none":
            return self._prob_unsmoothed(ngram)
        elif self.smoothing == "add-1":
            return self._prob_add_k(ngram, 1.0)
        elif self.smoothing == "add-k":
            return self._prob_add_k(ngram, self.k)
        elif self.smoothing == "interpolated":
            return self._prob_interpolated(ngram)
        elif self.smoothing == "kneser-ney":
            return self._prob_kneser_ney(ngram)
        return 0.0

    def perplexity(self, sentences):
        log_prob_sum = 0.0
        N = 0
        for s in sentences:
            tokens = tokenize(s)
            ngrams = self.get_ngrams(tokens)
            for ngram in ngrams:
                p = self.prob(ngram)
                if p == 0:
                    return float('inf')
                log_prob_sum += math.log(p)
                N += 1
        if N == 0: return 0.0
        return math.exp(-log_prob_sum / N)

def run_lab4():
    print("Loading data...")
    if not os.path.exists(CACHE_FILE):
        print(f"Error: {CACHE_FILE} not found. Please run Lab 3 first.")
        return
        
    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
        
    print(f"Total sentences: {len(sentences)}")
    train, dev, test = length_stratified_split(sentences)
    
    # Fast test
    train = train[:10000] 
    test = test[:100]
    
    print(f"Train: {len(train)}, Test: {len(test)}")
    
    smoothings = ["none", "add-1", "add-k", "interpolated", "kneser-ney"]
    ns = [1, 2, 3, 4]
    
    results = []
    
    for n in ns:
        print(f"\\n--- {n}-Gram Models ---")
        for s in smoothings:
            model = NGramModel(n, smoothing=s, k=0.1, lam=0.5, d=0.75)
            model.train(train)
            ppl = model.perplexity(test)
            results.append((n, s, ppl))
            print(f"Smoothing={s:15s} | Perplexity={ppl:.2f}")

if __name__ == "__main__":
    run_lab4()
