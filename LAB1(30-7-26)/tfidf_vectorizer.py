"""
tfidf_vectorizer.py — Lab 1: Language Identification
Custom TF-IDF Vectorizer using word n-grams and character n-grams.
Uses scipy sparse matrices for efficiency.
NO sklearn used anywhere.
"""

import re
import math
import numpy as np
from collections import defaultdict
from scipy.sparse import coo_matrix, csr_matrix
from typing import List, Dict, Tuple, Optional, Iterator


class CustomTFIDFVectorizer:
    """
    Custom TF-IDF Vectorizer that combines:
      • Word Unigrams  (word_ngram_range=(1,2) lower bound)
      • Word Bigrams   (word_ngram_range=(1,2) upper bound)
      • Char 2-grams
      • Char 3-grams
      • Char 4-grams   (char_ngram_range=(2,4))

    Formula (matches scikit-learn's default, implemented from scratch):
      TF  = 1 + log(raw_count)   [sublinear TF]
      IDF = log((1 + N) / (1 + df)) + 1   [smooth IDF]
      TF-IDF = TF * IDF
      Each document row is L2-normalised.

    Output: scipy CSR sparse matrix (memory efficient for large vocabularies).
    """

    def __init__(
        self,
        word_ngram_range: Tuple[int, int] = (1, 2),
        char_ngram_range: Tuple[int, int] = (2, 4),
        min_df: int = 2,
        max_features: Optional[int] = 200_000,
        sublinear_tf: bool = True,
    ):
        self.word_ngram_range = word_ngram_range
        self.char_ngram_range = char_ngram_range
        self.min_df          = min_df
        self.max_features    = max_features
        self.sublinear_tf    = sublinear_tf

        # Set after fit()
        self.vocab_: Dict[str, int] = {}   # feature_str → column index
        self.idf_  : np.ndarray     = None # shape (vocab_size,)
        self.n_docs_: int           = 0

    # ── N-gram Generators ──────────────────────────────────────────────────────

    def _word_ngrams(self, text: str) -> Iterator[str]:
        """Yield word unigrams and bigrams (prefixed with 'W:')."""
        tokens = text.lower().split()
        lo, hi = self.word_ngram_range
        for n in range(lo, hi + 1):
            for i in range(len(tokens) - n + 1):
                yield "W:" + " ".join(tokens[i : i + n])

    def _char_ngrams(self, text: str) -> Iterator[str]:
        """
        Yield character n-grams (prefixed with 'C<n>:').
        Text is lowercased and padded with spaces to capture word boundaries.
        """
        # Normalise whitespace then add boundary markers
        processed = " " + " ".join(text.lower().split()) + " "
        lo, hi = self.char_ngram_range
        for n in range(lo, hi + 1):
            prefix = f"C{n}:"
            for i in range(len(processed) - n + 1):
                yield prefix + processed[i : i + n]

    def _all_ngrams(self, text: str) -> Iterator[str]:
        """Yield all word + character n-grams for one document."""
        yield from self._word_ngrams(text)
        yield from self._char_ngrams(text)

    def _count_ngrams(self, text: str) -> Dict[str, int]:
        """Return {feature: raw_count} for a single document."""
        counts: Dict[str, int] = defaultdict(int)
        for gram in self._all_ngrams(text):
            counts[gram] += 1
        return dict(counts)

    # ── Fit ────────────────────────────────────────────────────────────────────

    def fit(self, texts: List[str]) -> "CustomTFIDFVectorizer":
        """
        Build vocabulary and compute IDF from training texts.
        Must be called only on the training split to avoid data leakage.
        """
        self.n_docs_ = N = len(texts)
        print(f"[TF-IDF] Fitting on {N} documents ...")

        # Step 1: Count document frequencies
        doc_freq: Dict[str, int] = defaultdict(int)
        for text in texts:
            seen: set = set()
            for gram in self._all_ngrams(text):
                if gram not in seen:
                    doc_freq[gram] += 1
                    seen.add(gram)

        # Step 2: Filter by min_df
        vocab_items = [
            (gram, df) for gram, df in doc_freq.items() if df >= self.min_df
        ]

        # Step 3: Sort by document frequency (desc) and apply max_features cap
        vocab_items.sort(key=lambda x: -x[1])
        if self.max_features is not None:
            vocab_items = vocab_items[: self.max_features]

        # Step 4: Build vocabulary mapping  feature_str → column index
        self.vocab_ = {gram: idx for idx, (gram, _) in enumerate(vocab_items)}
        V = len(self.vocab_)
        print(f"[TF-IDF] Vocabulary size: {V:,}")

        # Step 5: Compute IDF array
        # IDF(t) = log((1 + N) / (1 + df(t))) + 1   (sklearn smooth IDF)
        self.idf_ = np.zeros(V, dtype=np.float32)
        for gram, idx in self.vocab_.items():
            df = doc_freq[gram]
            self.idf_[idx] = math.log((1.0 + N) / (1.0 + df)) + 1.0

        return self

    # ── Transform ──────────────────────────────────────────────────────────────

    def transform(self, texts: List[str]) -> csr_matrix:
        """
        Transform a list of texts to a TF-IDF sparse matrix.
        Shape: (len(texts), vocab_size)
        """
        n_docs = len(texts)
        V      = len(self.vocab_)
        print(f"[TF-IDF] Transforming {n_docs} documents (vocab={V:,}) ...")

        # Collect COO triplets
        row_list:  List[int]   = []
        col_list:  List[int]   = []
        data_list: List[float] = []

        for doc_idx, text in enumerate(texts):
            counts = self._count_ngrams(text)

            # Gather TF-IDF values for features in vocabulary
            row_vals: Dict[int, float] = {}
            for gram, raw_count in counts.items():
                if gram not in self.vocab_:
                    continue
                col = self.vocab_[gram]
                # Sublinear TF: 1 + log(count)
                if self.sublinear_tf:
                    tf = 1.0 + math.log(raw_count) if raw_count > 0 else 0.0
                else:
                    tf = float(raw_count)
                val = tf * float(self.idf_[col])
                if val != 0.0:
                    row_vals[col] = val

            # L2 normalise
            if row_vals:
                norm = math.sqrt(sum(v * v for v in row_vals.values()))
                if norm > 0.0:
                    for col, val in row_vals.items():
                        row_list.append(doc_idx)
                        col_list.append(col)
                        data_list.append(val / norm)

        mat = coo_matrix(
            (data_list, (row_list, col_list)),
            shape=(n_docs, V),
            dtype=np.float32,
        )
        return mat.tocsr()

    def fit_transform(self, texts: List[str]) -> csr_matrix:
        """Convenience: fit then transform on the same data."""
        return self.fit(texts).transform(texts)

    # ── Persistence helpers ────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save vocab and IDF to a numpy .npz file."""
        np.savez_compressed(
            path,
            vocab_keys  = np.array(list(self.vocab_.keys())),
            vocab_vals  = np.array(list(self.vocab_.values()), dtype=np.int32),
            idf         = self.idf_,
            n_docs      = np.array([self.n_docs_]),
        )
        print(f"[TF-IDF] Saved vectorizer → {path}.npz")

    @classmethod
    def load(cls, path: str) -> "CustomTFIDFVectorizer":
        """Load a saved vectorizer."""
        data = np.load(path, allow_pickle=True)
        vec = cls()
        vec.vocab_  = dict(zip(data["vocab_keys"].tolist(),
                               data["vocab_vals"].tolist()))
        vec.idf_    = data["idf"]
        vec.n_docs_ = int(data["n_docs"][0])
        print(f"[TF-IDF] Loaded vectorizer from {path} (vocab={len(vec.vocab_):,})")
        return vec


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    docs = [
        "यह हिंदी में एक वाक्य है।",
        "இது தமிழ் மொழியில் ஒரு வாக்கியம்.",
        "ఇది తెలుగులో ఒక వాక్యం.",
        "यह एक और हिंदी वाक्य है।",
    ]
    vec = CustomTFIDFVectorizer(min_df=1, max_features=500)
    X = vec.fit_transform(docs)
    print(f"Matrix shape: {X.shape}")
    print(f"Non-zero elements: {X.nnz}")
