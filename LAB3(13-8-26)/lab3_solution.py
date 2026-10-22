"""
lab3_solution.py — LAB 3: BPE & WordPiece Tokenization
Implements BPE and WordPiece from scratch on Gujarati (guj_Gujr) data
from ai4bharat/IndicCorpV2. No external tokenizer libraries used.
"""

import os
import re
import csv
import math
import time
import random
import json
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional

import numpy as np

# ─── Configuration ──────────────────────────────────────────────────────────────
RANDOM_SEED = 42
LANGUAGE = "guj_Gujr"
MAX_SENTENCES = 1_100_000  # 1M train + 1K dev + 1K test + buffer
CACHE_FILE = os.path.join(os.path.dirname(__file__), "gujarati_sentences.txt")

TRAIN_SIZES = [100_000, 300_000, 500_000, 1_000_000]
VOCAB_SIZES = [20_000, 30_000, 50_000]

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA DOWNLOAD & SENTENCE TOKENIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def tokenize_sentences(paragraph: str, min_words: int = 3, min_len: int = 10,
                       max_len: int = 500) -> List[str]:
    """Split a paragraph into sentences using Indic punctuation boundaries."""
    paragraph = paragraph.strip()
    if not paragraph:
        return []
    boundary = re.compile(r'(?<=[.?!\u0964\u0965\u06D4])\s+')
    parts = boundary.split(paragraph)
    sentences = []
    for part in parts:
        for sub in part.split('\n'):
            sub = sub.strip()
            if min_len <= len(sub) <= max_len:
                words = sub.split()
                if len(words) >= min_words:
                    sentences.append(sub)
    return sentences


def download_gujarati_data(target: int = MAX_SENTENCES) -> List[str]:
    """Streams Gujarati data from IndicCorpV2 using split='guj_Gujr'."""
    from datasets import load_dataset
    print(f"Streaming Gujarati ({LANGUAGE}) corpus from Hugging Face...")
    ds = None
    
    try:
        # In IndicCorpV2, language codes (e.g. 'guj_Gujr') are specified via the split parameter
        ds = load_dataset(
            "ai4bharat/IndicCorpV2", "indiccorp_v2",
            split=LANGUAGE, streaming=True
        )
        print(f"  -> Successfully connected to IndicCorpV2 (split='{LANGUAGE}') stream.")
    except Exception as e:
        print(f"  [Split load failed]: {e}. Trying fallback dataset 'ai4bharat/indic_nlp_datasets'...")
        try:
            ds = load_dataset(
                "ai4bharat/indic_nlp_datasets", "guj",
                split="train", streaming=True
            )
            print("  -> Successfully connected to fallback indic_nlp_datasets stream.")
        except Exception as e2:
            raise RuntimeError(f"Could not stream Gujarati corpus from Hugging Face: {e2}")

    sentences, seen = [], set()
    for i, row in enumerate(ds):
        text = (row.get("text") or row.get("sentence") or "").strip()
        if not text:
            continue
        for s in tokenize_sentences(text):
            if s not in seen:
                seen.add(s)
                sentences.append(s)
        if len(sentences) >= target:
            break
        if (i + 1) % 25000 == 0:
            print(f"  Processed {i+1} paragraphs -> {len(sentences):,} sentences collected...")
            
    print(f"Total unique Gujarati sentences collected: {len(sentences):,}")
    if not sentences:
        raise ValueError("Downloaded dataset returned 0 valid Gujarati sentences!")
    return sentences


def load_or_download_data() -> List[str]:
    """Loads dataset from local cache if present and non-empty, otherwise streams and caches."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        if len(sentences) >= 1000:
            print(f"Successfully loaded {len(sentences):,} cached sentences from '{CACHE_FILE}'")
            return sentences
        else:
            print(f"Cache file '{CACHE_FILE}' contained only {len(sentences)} sentences (invalid/empty). Downloading fresh...")
            os.remove(CACHE_FILE)
            
    sentences = download_gujarati_data()
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        for s in sentences:
            f.write(s + '\n')
    print(f"Cached {len(sentences):,} sentences to '{CACHE_FILE}'")
    return sentences


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LENGTH-STRATIFIED TRAIN/DEV/TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════════

def length_stratified_split(sentences: List[str], dev_size: int = 1000,
                            test_size: int = 1000, seed: int = RANDOM_SEED):
    """
    Split data into train/dev/test ensuring dev & test have diverse sentence lengths.
    Bins sentences by word-count, samples proportionally from each bin.
    """
    if not sentences:
        raise ValueError("Cannot perform stratified split on an empty sentence list! Ensure dataset download succeeded.")

    rng = random.Random(seed)

    # Compute word-count for each sentence
    indexed = [(i, len(s.split())) for i, s in enumerate(sentences)]

    # Create length bins (quantile-based, ~10 bins)
    lengths = [wc for _, wc in indexed]
    percentiles = np.percentile(lengths, np.arange(0, 101, 10))
    bin_edges = sorted(set(percentiles.astype(int)))

    # Assign sentences to bins
    bins = defaultdict(list)
    for idx, wc in indexed:
        b = 0
        for j in range(len(bin_edges) - 1):
            if wc >= bin_edges[j]:
                b = j
        bins[b].append(idx)

    # Shuffle within each bin
    for b in bins:
        rng.shuffle(bins[b])

    # Calculate how many to sample from each bin for dev & test
    total = sum(len(v) for v in bins.values())
    needed = dev_size + test_size

    dev_indices, test_indices = [], []
    for b in sorted(bins.keys()):
        proportion = len(bins[b]) / total
        n_from_bin = max(1, int(proportion * (needed // 2)))
        # Take for dev
        taken_dev = bins[b][:n_from_bin]
        # Take for test
        taken_test = bins[b][n_from_bin:n_from_bin * 2]
        dev_indices.extend(taken_dev)
        test_indices.extend(taken_test)

    # Trim to exact sizes
    rng.shuffle(dev_indices)
    rng.shuffle(test_indices)
    dev_indices = set(dev_indices[:dev_size])
    test_indices = set(test_indices[:test_size])

    # Ensure no overlap
    test_indices -= dev_indices

    # If we need more test samples, fill from remaining
    used = dev_indices | test_indices
    remaining = [i for i in range(len(sentences)) if i not in used]
    rng.shuffle(remaining)

    while len(dev_indices) < dev_size and remaining:
        dev_indices.add(remaining.pop())
    while len(test_indices) < test_size and remaining:
        test_indices.add(remaining.pop())

    train_indices = [i for i in range(len(sentences))
                     if i not in dev_indices and i not in test_indices]

    dev = [sentences[i] for i in sorted(dev_indices)]
    test = [sentences[i] for i in sorted(test_indices)]
    train = [sentences[i] for i in train_indices]

    # Print length distribution stats
    for name, split in [("Dev", dev), ("Test", test), ("Train (first 10K)", train[:10000])]:
        wcs = [len(s.split()) for s in split]
        print(f"  {name}: n={len(split)}, "
              f"len range=[{min(wcs)}, {max(wcs)}], "
              f"mean={np.mean(wcs):.1f}, std={np.std(wcs):.1f}")

    return train, dev, test


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BPE IMPLEMENTATION (FROM SCRATCH)
# ═══════════════════════════════════════════════════════════════════════════════

class BPETokenizer:
    """Byte Pair Encoding tokenizer (Sennrich et al., 2016), from scratch."""

    END_OF_WORD = "</w>"

    def __init__(self, vocab_size: int = 30000):
        self.vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Dict[str, int] = {}

    def _get_word_freqs(self, sentences: List[str]) -> Dict[Tuple[str, ...], int]:
        """Pre-tokenize and count word frequencies."""
        word_freqs = Counter()
        for sent in sentences:
            for word in sent.strip().split():
                # Split word into characters + end-of-word marker
                chars = tuple(list(word) + [self.END_OF_WORD])
                word_freqs[chars] += 1
        return dict(word_freqs)

    def _get_pair_freqs(self, word_freqs: Dict[Tuple[str, ...], int]) -> Counter:
        """Count frequencies of adjacent symbol pairs."""
        pair_freqs = Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair_freqs[(word[i], word[i + 1])] += freq
        return pair_freqs

    def _merge_pair(self, word_freqs: Dict[Tuple[str, ...], int],
                    pair: Tuple[str, str]) -> Dict[Tuple[str, ...], int]:
        """Merge all occurrences of a pair in the vocabulary."""
        new_word_freqs = {}
        merged = pair[0] + pair[1]
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        return new_word_freqs

    def train(self, sentences: List[str]):
        """Train BPE on the given sentences."""
        print(f"    Training BPE (vocab_size={self.vocab_size})...")
        t0 = time.time()
        word_freqs_raw = self._get_word_freqs(sentences)

        # Build initial character vocabulary
        chars = set()
        for word in word_freqs_raw:
            for ch in word:
                chars.add(ch)
        self.vocab = {ch: i for i, ch in enumerate(sorted(chars))}

        word_list = [list(w) for w in word_freqs_raw.keys()]
        word_counts = list(word_freqs_raw.values())
        
        pair_freqs = Counter()
        pair_to_words = defaultdict(set)
        
        for idx, word in enumerate(word_list):
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freqs[pair] += word_counts[idx]
                pair_to_words[pair].add(idx)

        self.merges = []
        num_merges = self.vocab_size - len(self.vocab)

        for step in range(num_merges):
            if not pair_freqs:
                break
            best_pair = pair_freqs.most_common(1)[0][0]
            merged_symbol = best_pair[0] + best_pair[1]
            
            self.vocab[merged_symbol] = len(self.vocab)
            self.merges.append(best_pair)

            words_to_update = list(pair_to_words[best_pair])
            del pair_freqs[best_pair]
            del pair_to_words[best_pair]

            for idx in words_to_update:
                word = word_list[idx]
                count = word_counts[idx]

                # Decrement old pairs
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    if pair != best_pair:
                        pair_freqs[pair] -= count
                        pair_to_words[pair].discard(idx)
                        if pair_freqs[pair] <= 0:
                            del pair_freqs[pair]

                # Perform the merge
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                        new_word.append(merged_symbol)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                word_list[idx] = new_word

                # Increment new pairs
                for i in range(len(new_word) - 1):
                    pair = (new_word[i], new_word[i + 1])
                    pair_freqs[pair] += count
                    pair_to_words[pair].add(idx)

            if (step + 1) % 5000 == 0 or step == num_merges - 1:
                elapsed = time.time() - t0
                print(f"      Merge {step+1}/{num_merges} | "
                      f"Vocab: {len(self.vocab)} | {elapsed:.1f}s")

        print(f"    BPE training complete: {len(self.vocab)} symbols, "
              f"{len(self.merges)} merges in {time.time()-t0:.1f}s")

    def encode(self, text: str) -> List[str]:
        """Tokenize text using learned BPE merges."""
        tokens = []
        for word in text.strip().split():
            symbols = list(word) + [self.END_OF_WORD]
            for a, b in self.merges:
                i = 0
                while i < len(symbols) - 1:
                    if symbols[i] == a and symbols[i + 1] == b:
                        symbols[i] = a + b
                        del symbols[i + 1]
                    else:
                        i += 1
            tokens.extend(symbols)
        return tokens

    def encode_batch(self, sentences: List[str]) -> List[List[str]]:
        return [self.encode(s) for s in sentences]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. WORDPIECE IMPLEMENTATION (FROM SCRATCH)
# ═══════════════════════════════════════════════════════════════════════════════

class WordPieceTokenizer:
    """WordPiece tokenizer (Schuster & Nakajima, 2012), from scratch."""

    UNK = "[UNK]"
    CONT_PREFIX = "##"

    def __init__(self, vocab_size: int = 30000):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}

    def _get_word_freqs(self, sentences: List[str]) -> Dict[str, int]:
        word_freqs = Counter()
        for sent in sentences:
            for word in sent.strip().split():
                word_freqs[word] += 1
        return dict(word_freqs)

    def train(self, sentences: List[str]):
        """Train WordPiece on the given sentences."""
        print(f"    Training WordPiece (vocab_size={self.vocab_size})...")
        t0 = time.time()
        word_freqs_raw = self._get_word_freqs(sentences)

        # Build initial character vocabulary with ## prefix for continuations
        char_set = set()
        for word in word_freqs_raw:
            char_set.add(word[0])
            for ch in word[1:]:
                char_set.add(self.CONT_PREFIX + ch)

        self.vocab = {self.UNK: 0}
        for i, ch in enumerate(sorted(char_set), 1):
            self.vocab[ch] = i

        word_list = []
        word_counts = []
        for word, count in word_freqs_raw.items():
            split = [word[0]] + [self.CONT_PREFIX + c for c in word[1:]]
            word_list.append(split)
            word_counts.append(count)

        pair_freqs = Counter()
        symbol_freqs = Counter()
        pair_to_words = defaultdict(set)

        for idx, split in enumerate(word_list):
            count = word_counts[idx]
            for s in split:
                symbol_freqs[s] += count
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += count
                pair_to_words[pair].add(idx)

        num_merges = self.vocab_size - len(self.vocab)

        for step in range(num_merges):
            if not pair_freqs:
                break

            # Score pairs: freq(pair) / (freq(a) * freq(b))
            best_pair = None
            best_score = -1.0
            
            for pair, freq in pair_freqs.items():
                denom = symbol_freqs[pair[0]] * symbol_freqs[pair[1]]
                if denom == 0:
                    continue
                score = freq / denom
                if score > best_score:
                    best_score = score
                    best_pair = pair

            if best_pair is None:
                break

            # Merge the best pair
            merged = best_pair[0] + best_pair[1][2:] if best_pair[1].startswith(self.CONT_PREFIX) \
                     else best_pair[0] + best_pair[1]
            self.vocab[merged] = len(self.vocab)

            words_to_update = list(pair_to_words[best_pair])
            del pair_freqs[best_pair]
            del pair_to_words[best_pair]

            for idx in words_to_update:
                split = word_list[idx]
                count = word_counts[idx]

                # Decrement old stats
                for s in split:
                    symbol_freqs[s] -= count
                for i in range(len(split) - 1):
                    pair = (split[i], split[i + 1])
                    if pair != best_pair:
                        pair_freqs[pair] -= count
                        pair_to_words[pair].discard(idx)
                        if pair_freqs[pair] <= 0:
                            del pair_freqs[pair]

                # Perform the merge
                new_split = []
                i = 0
                while i < len(split):
                    if i < len(split) - 1 and split[i] == best_pair[0] and split[i + 1] == best_pair[1]:
                        new_split.append(merged)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                word_list[idx] = new_split

                # Increment new stats
                for s in new_split:
                    symbol_freqs[s] += count
                for i in range(len(new_split) - 1):
                    pair = (new_split[i], new_split[i + 1])
                    pair_freqs[pair] += count
                    pair_to_words[pair].add(idx)

            if (step + 1) % 5000 == 0 or step == num_merges - 1:
                elapsed = time.time() - t0
                print(f"      Merge {step+1}/{num_merges} | "
                      f"Vocab: {len(self.vocab)} | {elapsed:.1f}s")

            if len(self.vocab) >= self.vocab_size:
                break

        print(f"    WordPiece training complete: {len(self.vocab)} symbols "
              f"in {time.time()-t0:.1f}s")

    def encode(self, text: str) -> List[str]:
        """Tokenize text using greedy longest-match-first."""
        tokens = []
        for word in text.strip().split():
            if not word:
                continue
            sub_tokens = []
            start = 0
            is_bad = False
            while start < len(word):
                end = len(word)
                found = False
                while start < end:
                    substr = word[start:end]
                    if start > 0:
                        substr = self.CONT_PREFIX + substr
                    if substr in self.vocab:
                        sub_tokens.append(substr)
                        found = True
                        break
                    end -= 1
                if not found:
                    sub_tokens = [self.UNK]
                    is_bad = True
                    break
                start = end
            tokens.extend(sub_tokens)
        return tokens

    def encode_batch(self, sentences: List[str]) -> List[List[str]]:
        return [self.encode(s) for s in sentences]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ANALYSIS UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_stats(tokenizer, sentences: List[str], name: str) -> Dict:
    """Compute tokenization statistics."""
    all_tokens = tokenizer.encode_batch(sentences)
    total_tokens = sum(len(t) for t in all_tokens)
    total_chars = sum(len(s) for s in sentences)
    avg_tokens = total_tokens / len(sentences) if sentences else 0

    unk_count = sum(1 for toks in all_tokens for t in toks
                    if t == "[UNK]")
    unk_rate = (unk_count / total_tokens * 100.0) if total_tokens > 0 else 0.0

    unique_tokens = set(t for toks in all_tokens for t in toks)
    compression = total_chars / total_tokens if total_tokens > 0 else 0

    return {
        "name": name,
        "avg_tokens_per_sent": avg_tokens,
        "total_tokens": total_tokens,
        "unique_tokens_used": len(unique_tokens),
        "unk_rate": unk_rate,
        "compression_ratio": compression,
    }


def show_tokenization_examples(bpe: BPETokenizer, wp: WordPieceTokenizer,
                               sentences: List[str], n: int = 5):
    """Display side-by-side tokenization examples."""
    rng = random.Random(RANDOM_SEED)
    samples = rng.sample(sentences, min(n, len(sentences)))

    print("\n" + "=" * 100)
    print("TOKENIZATION EXAMPLES")
    print("=" * 100)
    for i, sent in enumerate(samples, 1):
        bpe_tokens = bpe.encode(sent)
        wp_tokens = wp.encode(sent)
        print(f"\n--- Example {i} ---")
        print(f"  Original : {sent[:120]}{'...' if len(sent)>120 else ''}")
        print(f"  BPE ({len(bpe_tokens):3d}): {' | '.join(bpe_tokens[:20])}"
              f"{'...' if len(bpe_tokens)>20 else ''}")
        print(f"  WP  ({len(wp_tokens):3d}): {' | '.join(wp_tokens[:20])}"
              f"{'...' if len(wp_tokens)>20 else ''}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(train_sents, dev_sents, test_sents,
                   train_size, vocab_size, label=""):
    """Train BPE & WordPiece with given config and return stats."""
    subset = train_sents[:train_size]
    print(f"\n{'─'*70}")
    print(f"  {label} | Train={len(subset):,} | Vocab={vocab_size:,}")
    print(f"{'─'*70}")

    bpe = BPETokenizer(vocab_size=vocab_size)
    bpe.train(subset)

    wp = WordPieceTokenizer(vocab_size=vocab_size)
    wp.train(subset)

    bpe_test = compute_stats(bpe, test_sents, "BPE-test")
    wp_test = compute_stats(wp, test_sents, "WP-test")

    return {
        "train_size": len(subset),
        "vocab_size": vocab_size,
        "bpe": bpe, "wp": wp,
        "bpe_test": bpe_test,
        "wp_test": wp_test,
    }


def print_results_table(results: List[Dict], title: str, vary_key: str):
    """Print a formatted comparison table."""
    if not results:
        print(f"\nNo results to display for: {title}")
        return
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")
    header = (f"{'Config':>12} | {'Method':>10} | {'Avg Tok/Sent':>13} | "
              f"{'UNK Rate':>9} | {'Compress':>9} | {'Unique Tok':>10}")
    print(header)
    print("-" * 100)
    for r in results:
        key_val = f"{r[vary_key]:,}"
        for method, split in [("BPE", "bpe_test"), ("WordPiece", "wp_test")]:
            s = r[split]
            print(f"{key_val:>12} | {method:>10} | {s['avg_tokens_per_sent']:>13.2f} | "
                  f"{s['unk_rate']:>8.4f}% | {s['compression_ratio']:>9.3f} | "
                  f"{s['unique_tokens_used']:>10,}")
            key_val = ""  # Only print once
    print("=" * 100)


def main():
    print("=" * 70)
    print("  LAB 3: BPE & WordPiece Tokenization")
    print(f"  Language: Gujarati ({LANGUAGE})")
    print("=" * 70)

    # ── Step 1: Load Data ──
    print("\n[Step 1] Loading data...")
    sentences = load_or_download_data()

    # ── Step 2: Split ──
    print(f"\n[Step 2] Creating length-stratified train/dev/test split...")
    train, dev, test = length_stratified_split(sentences)
    print(f"  Split sizes: Train={len(train):,} | Dev={len(dev)} | Test={len(test)}")

    # ── Step 3: Training Size Experiments (fixed vocab=30K) ──
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: Varying Training Data Size (vocab=30,000)")
    print("=" * 70)

    available_train = len(train)
    effective_train_sizes = [ts for ts in TRAIN_SIZES if ts <= available_train]
    if not effective_train_sizes:
        step = max(1000, available_train // 4)
        effective_train_sizes = [step, step * 2, step * 3, available_train]

    size_results = []
    for ts in effective_train_sizes:
        result = run_experiment(train, dev, test, ts, 30000,
                                label=f"TrainSize={ts:,}")
        size_results.append(result)

    # Show examples from last successful experiment
    if size_results:
        last = size_results[-1]
        show_tokenization_examples(last["bpe"], last["wp"], test, n=5)
        print_results_table(size_results, "Impact of Training Data Size (Vocab=30K)",
                            "train_size")

    # ── Step 4: Vocabulary Size Experiments (fixed train=max available) ──
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Varying Vocabulary Size")
    print("=" * 70)

    max_train = min(1_000_000, len(train))
    vocab_results = []
    for vs in VOCAB_SIZES:
        result = run_experiment(train, dev, test, max_train, vs,
                                label=f"VocabSize={vs:,}")
        vocab_results.append(result)

    if vocab_results:
        last = vocab_results[-1]
        show_tokenization_examples(last["bpe"], last["wp"], test, n=5)
    print_results_table(vocab_results, f"Impact of Vocabulary Size (Train={max_train:,})",
                        "vocab_size")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  SUMMARY OF FINDINGS")
    print("=" * 70)
    print("""
  Training Data Size Impact:
    - More training data → better subword coverage → fewer tokens per sentence
    - Unknown (UNK) rate decreases with more training data
    - Compression ratio improves as tokenizer learns common subwords

  Vocabulary Size Impact:
    - Larger vocab → longer subword units → fewer tokens per sentence
    - Larger vocab → lower UNK rate but diminishing returns
    - Smaller vocab → more fragmented tokenization but generalizes better

  BPE vs WordPiece:
    - BPE merges based on raw frequency (greedy pair counting)
    - WordPiece merges based on likelihood score (freq / product of parts)
    - WordPiece tends to produce slightly different subword boundaries
    """)
    print("Done!")


if __name__ == "__main__":
    main()
