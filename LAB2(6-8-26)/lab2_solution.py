"""
lab2_solution.py — LAB 2: TF-IDF Normalization Comparisons
This script implements a self-contained pipeline to evaluate the 6 combinations of
Term Frequency (TF) and Inverse Document Frequency (IDF) normalizations.
No scikit-learn is used.
"""

import os
import csv
import math
import time
import random
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional

import numpy as np
import scipy.sparse as sp

# Set fixed seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DATA_PATH = "../LAB1(30-7-26)/data/sentences.csv"

LANGUAGE_CONFIGS = [
    "asm_Beng", "ben_Beng", "brx_Deva", "doi_Deva", "gom_Deva", "guj_Gujr",
    "hin_Deva", "kan_Knda", "kas_Arab", "kas_Deva", "mai_Deva", "mal_Mlym",
    "mar_Deva", "mni_Beng", "mni_Mtei", "npi_Deva", "ory_Orya", "pan_Guru",
    "san_Deva", "sat_Olck", "snd_Arab", "tam_Taml", "tel_Telu", "urd_Arab"
]

SENT_BOUNDARY = re_boundary = None


# ── 1. Data Loader & Stratified Split ──────────────────────────────────────────

def tokenize_sentences(paragraph: str, min_words: int = 3, min_len: int = 10, max_len: int = 500) -> List[str]:
    import re
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


def collect_and_build_dataset(target_per_lang: int = 1000, seed: int = 42) -> List[Tuple[str, str]]:
    from datasets import load_dataset
    all_data = []
    for lang in LANGUAGE_CONFIGS:
        print(f"Streaming configurations for: {lang}...", end="", flush=True)
        try:
            ds = load_dataset(
                "ai4bharat/IndicCorpV2",
                "indiccorp_v2",
                data_dir=f"data/{lang}",
                split="train",
                streaming=True,
                trust_remote_code=True
            )
        except Exception as e:
            print(f" Failed: {e}")
            continue
        
        sentences = []
        for row in ds:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            sents = tokenize_sentences(text)
            sentences.extend(sents)
            if len(sentences) >= target_per_lang * 3:
                break
        
        seen = set()
        unique = []
        for s in sentences:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        
        rng = random.Random(seed)
        rng.shuffle(unique)
        selected = unique[:target_per_lang]
        print(f" Collected {len(selected)} sentences.")
        for s in selected:
            all_data.append((s, lang))
    return all_data


def load_dataset(path: str) -> List[Tuple[str, str]]:
    """Loads the dataset. Auto-streams from Hugging Face if cached CSV is missing or empty."""
    paths_to_try = [
        path,
        "../LAB1(30-7-26)/data/sentences.csv",
        "LAB1(30-7-26)/data/sentences.csv",
        "data/sentences.csv"
    ]
    
    valid_path = None
    for p in paths_to_try:
        if os.path.exists(p) and os.path.getsize(p) > 100:
            valid_path = p
            break
            
    if valid_path is not None:
        dataset = []
        with open(valid_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if row:
                    dataset.append((row[0], row[1]))
        return dataset

    # Fallback to downloading and caching
    print("Cached sentences file not found, empty, or corrupted. Downloading dataset...")
    os.makedirs("../LAB1(30-7-26)/data", exist_ok=True)
    dataset = collect_and_build_dataset(target_per_lang=1000, seed=RANDOM_SEED)
    
    # Save cache to make it available for future runs
    save_path = "../LAB1(30-7-26)/data/sentences.csv"
    with open(save_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(dataset)
    print(f"Saved dataset to {save_path}")
    
    return dataset


def perform_stratified_split(data: List[Tuple[str, str]], train_ratio=0.8, val_ratio=0.1, seed=42):
    """Splits dataset into stratified partitions: train/val/test."""
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for text, label in data:
        by_label[label].append(text)
        
    train_data, val_data, test_data = [], [], []
    
    for label, texts in sorted(by_label.items()):
        rng.shuffle(texts)
        n = len(texts)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_data.extend([(t, label) for t in texts[:n_train]])
        val_data.extend([(t, label) for t in texts[n_train:n_train + n_val]])
        test_data.extend([(t, label) for t in texts[n_train + n_val:]])
        
    rng.shuffle(train_data)
    rng.shuffle(val_data)
    rng.shuffle(test_data)
    
    return train_data, val_data, test_data


# ── 2. N-gram Feature Extraction Helpers ────────────────────────────────────────

def get_word_ngrams(text: str) -> List[str]:
    """Extracts word unigrams and bigrams from lowercase text."""
    words = text.lower().split()
    grams = []
    for w in words:
        grams.append(f"W1::{w}")
    for i in range(len(words) - 1):
        grams.append(f"W2::{words[i]} {words[i+1]}")
    return grams


def get_char_ngrams(text: str) -> List[str]:
    """Extracts character 2, 3, and 4-grams with space padding."""
    processed = " " + " ".join(text.lower().split()) + " "
    grams = []
    for n in [2, 3, 4]:
        prefix = f"C{n}::"
        for i in range(len(processed) - n + 1):
            grams.append(prefix + processed[i:i+n])
    return grams


def get_all_ngrams(text: str) -> List[str]:
    """Combines word and character n-grams."""
    return get_word_ngrams(text) + get_char_ngrams(text)


# ── 3. Custom TF-IDF Vectorizer with Normalization Options ──────────────────────

class CustomTfidfVectorizer:
    def __init__(
        self, 
        min_df: int = 2, 
        max_features: Optional[int] = 100000,
        tf_mode: str = "raw",
        idf_mode: str = "unnorm"
    ):
        self.min_df = min_df
        self.max_features = max_features
        self.tf_mode = tf_mode
        self.idf_mode = idf_mode
        
        self.vocab = {}
        self.idf = None
        self.n_docs = 0
        self.max_df_word = 0

    def fit(self, texts: List[str]) -> 'CustomTfidfVectorizer':
        self.n_docs = len(texts)
        doc_freqs = defaultdict(int)
        
        for text in texts:
            unique_ngrams = set(get_all_ngrams(text))
            for gram in unique_ngrams:
                doc_freqs[gram] += 1

        filtered_vocab = [gram for gram, df in doc_freqs.items() if df >= self.min_df]
        filtered_vocab.sort(key=lambda x: (-doc_freqs[x], x))
        
        if self.max_features is not None:
            filtered_vocab = filtered_vocab[:self.max_features]
            
        self.vocab = {gram: i for i, gram in enumerate(filtered_vocab)}
        
        word_dfs = [doc_freqs[gram] for gram in self.vocab if gram.startswith("W1::")]
        self.max_df_word = max(word_dfs) if word_dfs else self.n_docs
        
        self.idf = np.zeros(len(self.vocab), dtype=np.float32)
        for gram, idx in self.vocab.items():
            df = doc_freqs[gram]
            if self.idf_mode == "unnorm":
                self.idf[idx] = math.log((1.0 + self.n_docs) / (1.0 + df)) + 1.0
            elif self.idf_mode == "norm_max_df":
                self.idf[idx] = math.log((1.0 + self.max_df_word) / (1.0 + df)) + 1.0
                
        return self

    def transform(self, texts: List[str]) -> sp.csr_matrix:
        rows, cols, vals = [], [], []
        for row_idx, text in enumerate(texts):
            words = text.lower().split()
            total_words = len(words)
            word_counts = Counter(words)
            max_word_freq = max(word_counts.values()) if word_counts else 1
            
            counts = Counter(get_all_ngrams(text))
            row_entries = {}
            
            for gram, count in counts.items():
                if gram in self.vocab:
                    col_idx = self.vocab[gram]
                    
                    if self.tf_mode == "raw":
                        tf = float(count)
                    elif self.tf_mode == "total_words":
                        tf = float(count) / max(total_words, 1)
                    elif self.tf_mode == "max_word_freq":
                        tf = float(count) / max(max_word_freq, 1)
                    
                    row_entries[col_idx] = tf * self.idf[col_idx]
            
            if row_entries:
                norm = math.sqrt(sum(v*v for v in row_entries.values()))
                if norm > 0.0:
                    for col_idx, tfidf in row_entries.items():
                        rows.append(row_idx)
                        cols.append(col_idx)
                        vals.append(tfidf / norm)
                        
        shape = (len(texts), len(self.vocab))
        coo = sp.coo_matrix((vals, (rows, cols)), shape=shape, dtype=np.float32)
        return coo.tocsr()


# ── 4. Custom Softmax Logistic Regression ───────────────────────────────────────

class SoftmaxLogisticRegression:
    def __init__(
        self,
        lr: float = 0.5,
        epochs: int = 40,
        batch_size: int = 128,
        lambda_reg: float = 1e-4,
        lr_decay_step: int = 10,
        lr_decay_rate: float = 0.5,
        early_stopping_patience: int = 5,
        random_state: int = 42
    ):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.lr_decay_step = lr_decay_step
        self.lr_decay_rate = lr_decay_rate
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        
        self.W = None
        self.b = None

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exps = np.exp(shifted)
        return exps / exps.sum(axis=1, keepdims=True)

    def fit(self, X_train: sp.csr_matrix, y_train: np.ndarray, X_val: sp.csr_matrix, y_val: np.ndarray) -> 'SoftmaxLogisticRegression':
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X_train.shape
        n_classes = len(np.unique(y_train))
        
        scale = 1.0 / np.sqrt(n_features)
        self.W = rng.randn(n_classes, n_features).astype(np.float32) * scale
        self.b = np.zeros(n_classes, dtype=np.float32)
        
        Y_oh = np.zeros((n_samples, n_classes), dtype=np.float32)
        Y_oh[np.arange(n_samples), y_train] = 1.0
        
        best_f1 = -1.0
        best_W, best_b = None, None
        patience_counter = 0
        current_lr = self.lr
        
        for epoch in range(1, self.epochs + 1):
            if epoch > 1 and epoch % self.lr_decay_step == 0:
                current_lr *= self.lr_decay_rate
                
            indices = rng.permutation(n_samples)
            
            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                X_batch = X_train[batch_idx]
                Y_batch = Y_oh[batch_idx]
                
                logits = X_batch.dot(self.W.T) + self.b
                probs = self._softmax(logits)
                
                delta = (probs - Y_batch) / len(batch_idx)
                dW = delta.T.dot(X_batch).toarray().astype(np.float32)
                db = delta.sum(axis=0).astype(np.float32)
                
                dW += self.lambda_reg * self.W
                
                self.W -= current_lr * dW
                self.b -= current_lr * db
                
            val_preds = self.predict(X_val)
            val_f1 = calculate_macro_f1(y_val, val_preds)
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_W = self.W.copy()
                best_b = self.b.copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    break
                    
        if best_W is not None:
            self.W = best_W
            self.b = best_b
            
        return self

    def predict(self, X: sp.csr_matrix) -> np.ndarray:
        logits = X.dot(self.W.T) + self.b
        probs = self._softmax(logits)
        return probs.argmax(axis=1)


# ── 5. Evaluation Metrics ───────────────────────────────────────────────────────

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def calculate_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes_in_data = np.unique(np.concatenate([y_true, y_pred]))
    if len(classes_in_data) == 0:
        return 0.0
    
    max_lbl = int(np.max(np.concatenate([y_true, y_pred])))
    cm = compute_confusion_matrix(y_true, y_pred, max_lbl + 1)
    
    f1_scores = []
    for c in classes_in_data:
        TP = cm[c, c]
        FP = cm[:, c].sum() - TP
        FN = cm[c, :].sum() - TP
        
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0.0 else 0.0
        f1_scores.append(f1)
        
    return float(np.mean(f1_scores))


# ── 6. Main Runner ─────────────────────────────────────────────────────────────

def main():
    print("==================================================")
    print("  LAB 2: TF-IDF Normalization Combinations")
    print("==================================================")

    print("Loading dataset...")
    raw_dataset = load_dataset(DATA_PATH)
    print(f"Loaded {len(raw_dataset)} sentences.")

    train_set, val_set, test_set = perform_stratified_split(raw_dataset, seed=RANDOM_SEED)
    train_texts, train_labels = zip(*train_set)
    val_texts, val_labels = zip(*val_set)
    test_texts, test_labels = zip(*test_set)

    classes = sorted(list(set(train_labels)))
    label_to_idx = {c: i for i, c in enumerate(classes)}
    
    y_train = np.array([label_to_idx[l] for l in train_labels], dtype=np.int32)
    y_val = np.array([label_to_idx[l] for l in val_labels], dtype=np.int32)
    y_test = np.array([label_to_idx[l] for l in test_labels], dtype=np.int32)

    combinations = [
        {"name": "Combo 1: TF Raw & IDF Unnormalized", "tf": "raw", "idf": "unnorm"},
        {"name": "Combo 2: TF Raw & IDF Normalized", "tf": "raw", "idf": "norm_max_df"},
        {"name": "Combo 3: TF Total Words & IDF Unnormalized", "tf": "total_words", "idf": "unnorm"},
        {"name": "Combo 4: TF Total Words & IDF Normalized", "tf": "total_words", "idf": "norm_max_df"},
        {"name": "Combo 5: TF Max Word Freq & IDF Unnormalized", "tf": "max_word_freq", "idf": "unnorm"},
        {"name": "Combo 6: TF Max Word Freq & IDF Normalized", "tf": "max_word_freq", "idf": "norm_max_df"},
    ]

    results = []

    print("\nStarting evaluation of combinations...")
    for idx, combo in enumerate(combinations, 1):
        print(f"\n[{idx}/6] Evaluating {combo['name']}...")
        
        t0 = time.time()
        vectorizer = CustomTfidfVectorizer(
            min_df=2, 
            max_features=80000, 
            tf_mode=combo["tf"], 
            idf_mode=combo["idf"]
        )
        vectorizer.fit(train_texts)
        
        X_train = vectorizer.transform(train_texts)
        X_val = vectorizer.transform(val_texts)
        X_test = vectorizer.transform(test_texts)
        
        clf = SoftmaxLogisticRegression(
            lr=0.5, 
            epochs=25, 
            batch_size=256, 
            lambda_reg=1e-4, 
            random_state=RANDOM_SEED
        )
        clf.fit(X_train, y_train, X_val, y_val)
        
        val_preds = clf.predict(X_val)
        val_f1 = calculate_macro_f1(y_val, val_preds)
        val_acc = np.mean(val_preds == y_val)
        
        test_preds = clf.predict(X_test)
        test_f1 = calculate_macro_f1(y_test, test_preds)
        test_acc = np.mean(test_preds == y_test)
        
        elapsed = time.time() - t0
        print(f"Done in {elapsed:.1f}s | Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f}")
        
        results.append({
            "Combo": idx,
            "Name": combo["name"],
            "TF Normalization": combo["tf"],
            "IDF Normalization": combo["idf"],
            "Val Macro-F1": val_f1,
            "Val Accuracy": val_acc,
            "Test Macro-F1": test_f1,
            "Test Accuracy": test_acc
        })

    print("\n" + "="*80)
    print("                           COMPARATIVE RESULTS TABLE")
    print("="*80)
    print(f"{'Combo':<5} | {'TF Mode':<15} | {'IDF Mode':<12} | {'Val F1':<8} | {'Test F1':<8} | {'Val Acc':<8} | {'Test Acc':<8}")
    print("-" * 80)
    for r in results:
        print(f"{r['Combo']:<5} | {r['TF Normalization']:<15} | {r['IDF Normalization']:<12} | {r['Val Macro-F1']:<8.4f} | {r['Test Macro-F1']:<8.4f} | {r['Val Accuracy']:<8.4f} | {r['Test Accuracy']:<8.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
