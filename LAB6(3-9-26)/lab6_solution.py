"""
Lab 6: Language Identification with FastText Word Vectors
=========================================================
Classifiers:
  1. Logistic Regression  (averaged word vectors, pure numpy)
  2. MLP  n=0  (linear — no hidden layers)
  3. MLP  n=1  (1 hidden layer)
  4. MLP  n=2  (2 hidden layers)
  5. LSTM      (PyTorch)
  6. GRU       (PyTorch)
  7. FastText Classification  (fasttext library)

Metric: Custom Macro-F1  (no sklearn)
"""

import os, sys, csv, gzip, math, time, random, urllib.request, struct
import numpy as np
from collections import defaultdict, Counter

# ── Paths ─────────────────────────────────────────────────────────────────────
LAB1_DIR     = os.path.join(os.path.dirname(__file__), '..', 'LAB1(30-7-26)')
DATA_CSV     = os.path.join(os.path.dirname(__file__), '..', 'data', 'sentences.csv')
VEC_CACHE    = os.path.join(os.path.dirname(__file__), 'fasttext_vecs.npz')
FT_TRAIN     = os.path.join(os.path.dirname(__file__), 'ft_train.txt')
FT_MODEL_OUT = os.path.join(os.path.dirname(__file__), 'ft_model')
VEC_DIR      = os.path.join(os.path.dirname(__file__), 'vectors')

RANDOM_SEED = 42
DIM = 300

# ── Language → FastText wiki language code ────────────────────────────────────
LANG_TO_FT = {
    "asm_Beng": "as", "ben_Beng": "bn", "brx_Deva": "hi", "doi_Deva": "hi",
    "gom_Deva": "hi", "guj_Gujr": "gu", "hin_Deva": "hi", "kan_Knda": "kn",
    "kas_Arab": "ur", "kas_Deva": "hi", "mai_Deva": "hi", "mal_Mlym": "ml",
    "mar_Deva": "mr", "mni_Beng": "bn", "mni_Mtei": "hi", "npi_Deva": "ne",
    "ory_Orya": "or", "pan_Guru": "pa", "san_Deva": "sa", "sat_Olck": "hi",
    "snd_Arab": "sd", "tam_Taml": "ta", "tel_Telu": "te", "urd_Arab": "ur",
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path=DATA_CSV):
    data = []
    with open(path, encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            t, l = row['text'].strip(), row['label'].strip()
            if t and l:
                data.append((t, l))
    return data


def stratified_split(data, train_r=0.8, val_r=0.1, seed=RANDOM_SEED):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for text, lbl in data:
        by_label[lbl].append(text)
    train, val, test = [], [], []
    for lbl, texts in sorted(by_label.items()):
        rng.shuffle(texts)
        n = len(texts); nt = int(n*train_r); nv = int(n*val_r)
        train += [(t, lbl) for t in texts[:nt]]
        val   += [(t, lbl) for t in texts[nt:nt+nv]]
        test  += [(t, lbl) for t in texts[nt+nv:]]
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# 2. FASTTEXT WORD VECTORS  (download .vec.gz → extract vocab subset → cache)
# ─────────────────────────────────────────────────────────────────────────────

def build_vocab(sentences):
    vocab = set()
    for text, _ in sentences:
        vocab.update(text.strip().split())
    return vocab


def _vec_url(lang_code):
    return f"https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.{lang_code}.vec"


def _download_partial_vec(lang_code, vocab, vec_dir=VEC_DIR):
    """Download wiki.{lang}.vec and keep only words in vocab. Returns dict word→np.array."""
    os.makedirs(vec_dir, exist_ok=True)
    cache_file = os.path.join(vec_dir, f"{lang_code}.npz")
    if os.path.exists(cache_file):
        d = np.load(cache_file, allow_pickle=True)
        return {w: d[w] for w in d}

    url = _vec_url(lang_code)
    print(f"  Downloading vectors for '{lang_code}' from fasttext CDN ...", flush=True)
    vecs = {}
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            first = True
            for raw_line in resp:
                line = raw_line.decode('utf-8', errors='ignore').rstrip('\n')
                if first:
                    first = False
                    parts = line.split()
                    if len(parts) == 2:   # header line
                        continue
                parts = line.split(' ')
                if len(parts) < DIM + 1:
                    continue
                word = parts[0]
                if word in vocab:
                    try:
                        vecs[word] = np.array(parts[1:DIM+1], dtype=np.float32)
                    except ValueError:
                        pass
        if vecs:
            np.savez_compressed(cache_file, **vecs)
            print(f"    Saved {len(vecs)} vectors → {cache_file}")
    except Exception as e:
        print(f"    WARNING: could not download '{lang_code}': {e}")
    return vecs


def load_fasttext_vecs(all_data):
    """Return dict: word → 300-dim vector (merged across all needed languages)."""
    if os.path.exists(VEC_CACHE):
        print(f"[Vecs] Loading cached word vectors from {VEC_CACHE}")
        d = np.load(VEC_CACHE, allow_pickle=True)
        return {w: d[w] for w in d}

    vocab = build_vocab(all_data)
    print(f"[Vecs] Vocabulary size: {len(vocab):,}")

    needed_codes = set(LANG_TO_FT.values())
    merged = {}
    for code in sorted(needed_codes):
        vecs = _download_partial_vec(code, vocab)
        for w, v in vecs.items():
            if w not in merged:
                merged[w] = v
        print(f"    [{code}] {len(vecs)} vectors found  (total: {len(merged):,})")

    if merged:
        np.savez_compressed(VEC_CACHE, **merged)
        print(f"[Vecs] Cached {len(merged):,} vectors → {VEC_CACHE}")
    return merged


def avg_vector(text, vecs, dim=DIM):
    """Average word vectors for a sentence. Unknown words → zero."""
    tokens = text.strip().split()
    mat = [vecs[t] for t in tokens if t in vecs]
    if not mat:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(mat, axis=0).astype(np.float32)


def make_avg_features(dataset, vecs):
    X = np.stack([avg_vector(t, vecs) for t, _ in dataset])
    labels = [lbl for _, lbl in dataset]
    return X, labels


def make_seq_features(dataset, vecs, max_len=50, dim=DIM):
    """Return (N, max_len, dim) array + labels + lengths."""
    seqs, labels, lengths = [], [], []
    for text, lbl in dataset:
        tokens = text.strip().split()[:max_len]
        mat = np.zeros((max_len, dim), dtype=np.float32)
        for i, t in enumerate(tokens):
            if t in vecs:
                mat[i] = vecs[t]
        seqs.append(mat)
        labels.append(lbl)
        lengths.append(max(1, min(len(tokens), max_len)))
    return np.stack(seqs), labels, np.array(lengths, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# 3. CUSTOM MACRO-F1  (no sklearn)
# ─────────────────────────────────────────────────────────────────────────────

def macro_f1(y_true, y_pred):
    classes = sorted(set(y_true) | set(y_pred))
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    cm = np.zeros((n, n), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    f1s = []
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2*pr*rc/(pr+rc) if (pr+rc) > 0 else 0.0)
    return float(np.mean(f1s))


# ─────────────────────────────────────────────────────────────────────────────
# 4. LABEL ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class LabelEncoder:
    def fit(self, labels):
        self.classes_ = sorted(set(labels))
        self._l2i = {c: i for i, c in enumerate(self.classes_)}
        return self
    def transform(self, labels):
        return np.array([self._l2i[l] for l in labels], dtype=np.int64)
    def inverse_transform(self, ids):
        return [self.classes_[i] for i in ids]


# ─────────────────────────────────────────────────────────────────────────────
# 5. LOGISTIC REGRESSION  (pure numpy, softmax + SGD)
# ─────────────────────────────────────────────────────────────────────────────

class LogisticRegression:
    def __init__(self, n_classes, dim, lr=0.1, epochs=30, batch=256, seed=RANDOM_SEED):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.01, (n_classes, dim)).astype(np.float32)
        self.b = np.zeros(n_classes, dtype=np.float32)
        self.lr = lr; self.epochs = epochs; self.batch = batch

    def _softmax(self, z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        n = len(X)
        for ep in range(self.epochs):
            idx = np.random.permutation(n)
            X, y = X[idx], y[idx]
            for i in range(0, n, self.batch):
                Xb, yb = X[i:i+self.batch], y[i:i+self.batch]
                logits = Xb @ self.W.T + self.b          # (B, C)
                probs  = self._softmax(logits)
                probs[np.arange(len(yb)), yb] -= 1.0    # gradient
                probs /= len(yb)
                self.W -= self.lr * (probs.T @ Xb)
                self.b -= self.lr * probs.sum(0)
            if (ep+1) % 10 == 0:
                preds = self.predict(X)
                acc = (preds == y).mean()
                print(f"    Epoch {ep+1:3d}  train-acc={acc:.4f}")

    def predict(self, X):
        return np.argmax(X @ self.W.T + self.b, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# 6. MLP / LSTM / GRU  (PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

def _try_import_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError:
        print("PyTorch not installed. Run: pip install torch")
        return None, None


def _train_torch_model(model, X_train, y_train, X_val, y_val,
                        le, epochs=20, batch=64, lr=1e-3, seq=False):
    torch, nn = _try_import_torch()
    if torch is None:
        return None
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if seq:
        X_tr, lens_tr = X_train
        X_va, lens_va = X_val
    else:
        X_tr, lens_tr = X_train, None
        X_va, lens_va = X_val, None

    n = len(X_tr)
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(n)
        total_loss = 0
        for i in range(0, n, batch):
            bi = idx[i:i+batch]
            xb = torch.tensor(X_tr[bi], dtype=torch.float32, device=device)
            yb = torch.tensor(y_train[bi], dtype=torch.long,  device=device)
            lb = torch.tensor(lens_tr[bi], dtype=torch.long,  device=device) if lens_tr is not None else None
            opt.zero_grad()
            out = model(xb, lb) if lb is not None else model(xb)
            loss = loss_fn(out, yb)
            loss.backward(); opt.step()
            total_loss += loss.item()
        if (ep+1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                xv = torch.tensor(X_va, dtype=torch.float32, device=device)
                lv = torch.tensor(lens_va, dtype=torch.long, device=device) if lens_va is not None else None
                ov = model(xv, lv) if lv is not None else model(xv)
                preds = ov.argmax(1).cpu().numpy()
            pred_lbl = le.inverse_transform(preds)
            true_lbl = le.inverse_transform(y_val)
            f1 = macro_f1(true_lbl, pred_lbl)
            print(f"    Epoch {ep+1:3d}  loss={total_loss/n:.4f}  val-F1={f1:.4f}")
    return model


def _predict_torch(model, X, lens=None):
    torch, nn = _try_import_torch()
    if torch is None:
        return np.zeros(len(X), dtype=np.int64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        xv = torch.tensor(X, dtype=torch.float32, device=device)
        lv = torch.tensor(lens, dtype=torch.long, device=device) if lens is not None else None
        out = model(xv, lv) if lv is not None else model(xv)
    return out.argmax(1).cpu().numpy()


def build_mlp(n_hidden, dim, n_classes):
    torch, nn = _try_import_torch()
    if torch is None:
        return None
    layers = []
    in_dim = dim
    hidden_sizes = [256, 128]
    for i in range(n_hidden):
        layers += [nn.Linear(in_dim, hidden_sizes[i]), nn.ReLU(), nn.Dropout(0.3)]
        in_dim = hidden_sizes[i]
    layers.append(nn.Linear(in_dim, n_classes))
    return nn.Sequential(*layers)


class LSTMClassifier:
    class _Net:
        pass

    @staticmethod
    def build(dim, n_classes, hidden=128, layers=1):
        torch, nn = _try_import_torch()
        if torch is None:
            return None
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(dim, hidden, layers, batch_first=True,
                                    dropout=0.3 if layers > 1 else 0.0)
                self.fc = nn.Linear(hidden, n_classes)
            def forward(self, x, lengths=None):
                if lengths is not None:
                    from torch.nn.utils.rnn import pack_padded_sequence
                    lengths_cpu = lengths.cpu()
                    packed = pack_padded_sequence(x, lengths_cpu, batch_first=True,
                                                  enforce_sorted=False)
                    _, (h, _) = self.lstm(packed)
                else:
                    _, (h, _) = self.lstm(x)
                return self.fc(h[-1])
        return Net()


class GRUClassifier:
    @staticmethod
    def build(dim, n_classes, hidden=128, layers=1):
        torch, nn = _try_import_torch()
        if torch is None:
            return None
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(dim, hidden, layers, batch_first=True,
                                   dropout=0.3 if layers > 1 else 0.0)
                self.fc = nn.Linear(hidden, n_classes)
            def forward(self, x, lengths=None):
                if lengths is not None:
                    from torch.nn.utils.rnn import pack_padded_sequence
                    lengths_cpu = lengths.cpu()
                    packed = pack_padded_sequence(x, lengths_cpu, batch_first=True,
                                                  enforce_sorted=False)
                    _, h = self.gru(packed)
                else:
                    _, h = self.gru(x)
                return self.fc(h[-1])
        return Net()


# ─────────────────────────────────────────────────────────────────────────────
# 7. FASTTEXT CLASSIFICATION  (fasttext library)
# ─────────────────────────────────────────────────────────────────────────────

def run_fasttext_classifier(train_data, test_data):
    try:
        import fasttext
    except ImportError:
        print("fasttext not installed. Run: pip install fasttext")
        return None

    # Write fasttext-format training file
    with open(FT_TRAIN, 'w', encoding='utf-8') as f:
        for text, lbl in train_data:
            f.write(f"__label__{lbl} {text}\n")

    model = fasttext.train_supervised(
        input=FT_TRAIN,
        epoch=25,
        lr=0.5,
        wordNgrams=2,
        dim=100,
        loss='softmax',
        verbose=0,
    )
    preds = [model.predict(t)[0][0].replace('__label__', '') for t, _ in test_data]
    return preds


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_lab6():
    rng_state = random.getstate()
    np_state  = np.random.get_state()
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 65)
    print("  Lab 6: Language Identification — FastText Word Vectors")
    print("=" * 65)

    # ── Load data ──────────────────────────────────────────────────────
    print("\n[1] Loading dataset from Lab 1 ...")
    data = load_data()
    print(f"    Total: {len(data)}  Languages: {len(set(l for _,l in data))}")

    train_data, val_data, test_data = stratified_split(data)
    print(f"    Train={len(train_data)}  Val={len(val_data)}  Test={len(test_data)}")

    # ── Load FastText vectors ──────────────────────────────────────────
    print("\n[2] Loading FastText wiki word vectors ...")
    vecs = load_fasttext_vecs(train_data + val_data + test_data)
    cov = sum(1 for t, _ in train_data for w in t.split() if w in vecs)
    tot = sum(len(t.split()) for t, _ in train_data)
    print(f"    Vector coverage (train): {cov/tot*100:.1f}%")

    # ── Label encoding ─────────────────────────────────────────────────
    le = LabelEncoder().fit([l for _, l in data])
    n_classes = len(le.classes_)
    print(f"    Classes: {n_classes}")

    def encode(dataset):
        return le.transform([l for _, l in dataset])

    y_train = encode(train_data)
    y_val   = encode(val_data)
    y_test  = encode(test_data)

    # ── Averaged features (for LR + MLP) ──────────────────────────────
    print("\n[3] Computing averaged word-vector features ...")
    X_tr_avg, _ = make_avg_features(train_data, vecs)
    X_va_avg, _ = make_avg_features(val_data,   vecs)
    X_te_avg, _ = make_avg_features(test_data,  vecs)

    # ── Sequence features (for LSTM + GRU) ────────────────────────────
    print("[4] Computing sequence features ...")
    X_tr_seq, _, L_tr = make_seq_features(train_data, vecs)
    X_va_seq, _, L_va = make_seq_features(val_data,   vecs)
    X_te_seq, _, L_te = make_seq_features(test_data,  vecs)

    results = {}

    # ─────────────────────────────────────────────────────────────────
    # Classifier 1: Logistic Regression
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "─"*65)
    print("  Classifier 1: Logistic Regression (averaged vectors)")
    print("─"*65)
    lr_clf = LogisticRegression(n_classes=n_classes, dim=DIM, lr=0.2, epochs=30)
    lr_clf.fit(X_tr_avg, y_train)
    preds = le.inverse_transform(lr_clf.predict(X_te_avg))
    true  = [l for _, l in test_data]
    f1 = macro_f1(true, preds)
    results['Logistic Regression'] = f1
    print(f"  ► Test Macro-F1: {f1:.4f}")

    # ─────────────────────────────────────────────────────────────────
    # Classifiers 2-4: MLP n=0,1,2
    # ─────────────────────────────────────────────────────────────────
    torch, nn = _try_import_torch()
    if torch is not None:
        for n_hidden in [0, 1, 2]:
            name = f"MLP n={n_hidden}"
            print(f"\n{'─'*65}")
            print(f"  Classifier: {name}")
            print("─"*65)
            model = build_mlp(n_hidden, DIM, n_classes)
            _train_torch_model(model, X_tr_avg, y_train, X_va_avg, y_val, le,
                               epochs=20, batch=128, lr=1e-3, seq=False)
            raw = _predict_torch(model, X_te_avg)
            preds = le.inverse_transform(raw)
            f1 = macro_f1(true, preds)
            results[name] = f1
            print(f"  ► Test Macro-F1: {f1:.4f}")

        # ─────────────────────────────────────────────────────────────
        # Classifier 5: LSTM
        # ─────────────────────────────────────────────────────────────
        print(f"\n{'─'*65}")
        print("  Classifier: LSTM")
        print("─"*65)
        lstm_model = LSTMClassifier.build(DIM, n_classes, hidden=128)
        _train_torch_model(lstm_model,
                           (X_tr_seq, L_tr), y_train,
                           (X_va_seq, L_va), y_val,
                           le, epochs=20, batch=64, lr=1e-3, seq=True)
        raw   = _predict_torch(lstm_model, X_te_seq, L_te)
        preds = le.inverse_transform(raw)
        f1 = macro_f1(true, preds)
        results['LSTM'] = f1
        print(f"  ► Test Macro-F1: {f1:.4f}")

        # ─────────────────────────────────────────────────────────────
        # Classifier 6: GRU
        # ─────────────────────────────────────────────────────────────
        print(f"\n{'─'*65}")
        print("  Classifier: GRU")
        print("─"*65)
        gru_model = GRUClassifier.build(DIM, n_classes, hidden=128)
        _train_torch_model(gru_model,
                           (X_tr_seq, L_tr), y_train,
                           (X_va_seq, L_va), y_val,
                           le, epochs=20, batch=64, lr=1e-3, seq=True)
        raw   = _predict_torch(gru_model, X_te_seq, L_te)
        preds = le.inverse_transform(raw)
        f1 = macro_f1(true, preds)
        results['GRU'] = f1
        print(f"  ► Test Macro-F1: {f1:.4f}")
    else:
        print("\nSkipping MLP/LSTM/GRU — PyTorch not installed.")

    # ─────────────────────────────────────────────────────────────────
    # Classifier 7: FastText Classification
    # ─────────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  Classifier: FastText Classification")
    print("─"*65)
    ft_preds = run_fasttext_classifier(train_data + val_data, test_data)
    if ft_preds is not None:
        f1 = macro_f1(true, ft_preds)
        results['FastText Classification'] = f1
        print(f"  ► Test Macro-F1: {f1:.4f}")

    # ─────────────────────────────────────────────────────────────────
    # Summary Table
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FINAL RESULTS — Macro-F1 on Test Set")
    print("=" * 65)
    print(f"  {'Classifier':<30} {'Macro-F1':>10}")
    print(f"  {'-'*42}")
    for name, f1 in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:<30} {f1:>10.4f}")
    print("=" * 65)

    random.setstate(rng_state)
    np.random.set_state(np_state)
    return results


if __name__ == "__main__":
    run_lab6()
