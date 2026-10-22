"""
data_utils.py — Lab 1: Indic Language Identification
Dataset   : ai4bharat/IndicCorpV2  (config = "indiccorp_v2")
Sampling  : Reservoir sampling per language split (exact assignment spec)
Split     : Stratified 80 / 10 / 10
NO sklearn used anywhere.
"""

import re
import hashlib
import random
import csv
import os
from collections import defaultdict
from typing import List, Tuple, Dict

# ── Force UTF-8 on Windows ────────────────────────────────────────────────────
import sys
if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")

# ── Random seed (set by caller, default here for standalone use) ──────────────
RANDOM_SEED = 42

# ── 24 Indic language-script splits in IndicCorpV2 ───────────────────────────
LANGUAGE_CONFIGS = [
    "asm_Beng",  # Assamese  – Bengali script
    "ben_Beng",  # Bengali   – Bengali script
    "brx_Deva",  # Bodo      – Devanagari
    "doi_Deva",  # Dogri     – Devanagari
    "gom_Deva",  # Konkani   – Devanagari
    "guj_Gujr",  # Gujarati  – Gujarati script
    "hin_Deva",  # Hindi     – Devanagari
    "kan_Knda",  # Kannada   – Kannada script
    "kas_Arab",  # Kashmiri  – Arabic script
    "kas_Deva",  # Kashmiri  – Devanagari
    "mai_Deva",  # Maithili  – Devanagari
    "mal_Mlym",  # Malayalam – Malayalam script
    "mar_Deva",  # Marathi   – Devanagari
    "mni_Beng",  # Manipuri  – Bengali script
    "mni_Mtei",  # Manipuri  – Meitei script
    "npi_Deva",  # Nepali    – Devanagari
    "ory_Orya",  # Odia      – Odia script
    "pan_Guru",  # Punjabi   – Gurmukhi script
    "san_Deva",  # Sanskrit  – Devanagari
    "sat_Olck",  # Santali   – Ol Chiki script
    "snd_Arab",  # Sindhi    – Arabic script
    "tam_Taml",  # Tamil     – Tamil script
    "tel_Telu",  # Telugu    – Telugu script
    "urd_Arab",  # Urdu      – Arabic script
]

# ── Script-aware sentence boundary regex ─────────────────────────────────────
# Covers: Devanagari danda (।), double danda (॥), Arabic full stop (۔), .?!
_SENT_BOUNDARY = re.compile(r'(?<=[.?!\u0964\u0965\u06D4])\s+')


def sentence_split(text: str) -> List[str]:
    """
    Split a paragraph into sentences using a script-aware regex.
    Secondary split on newlines for paragraph-formatted corpora.
    """
    text = text.strip()
    if not text:
        return []
    parts = _SENT_BOUNDARY.split(text)
    out = []
    for part in parts:
        for sub in part.split('\n'):
            sub = sub.strip()
            if sub:
                out.append(sub)
    return out


# ── Assignment-specified collection function ──────────────────────────────────

def collect_sentences(split_name: str,
                      target: int = 1000,
                      min_len: int = 20,
                      max_len: int = 300,
                      max_paragraphs: int = 15000,
                      seed: int = RANDOM_SEED) -> Tuple[List[str], dict]:
    """
    Reservoir-sample `target` sentences from a streamed prefix of a language split.

    Uses the exact approach from the Lab 1 assignment:
      - Dataset  : ai4bharat/IndicCorpV2  (config = indiccorp_v2)
      - Split    : split_name  (e.g. "hin_Deva", "tam_Taml", …)
      - Sampling : Vitter's reservoir sampling for uniform random selection
      - Dedup    : MD5 hash-based exact deduplication

    Returns (reservoir, stats_dict).
    """
    from datasets import load_dataset

    rng = random.Random(seed)

    # Load the language-specific split from IndicCorpV2 (no trust_remote_code)
    ds = load_dataset(
        "ai4bharat/IndicCorpV2",
        "indiccorp_v2",
        split=split_name,
        streaming=True,
    )

    reservoir       = []
    seen_hashes     = set()
    n_paragraphs    = 0
    n_valid_seen    = 0   # drives reservoir replacement index

    for row in ds:
        n_paragraphs += 1
        text = row["text"].strip()

        if text:
            for sent in sentence_split(text):
                sent = sent.strip()
                if not (min_len <= len(sent) <= max_len):
                    continue

                h = hashlib.md5(sent.encode("utf-8")).hexdigest()
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                n_valid_seen += 1
                if len(reservoir) < target:
                    reservoir.append(sent)
                else:
                    # Vitter reservoir replacement
                    j = rng.randint(0, n_valid_seen - 1)
                    if j < target:
                        reservoir[j] = sent

        if n_paragraphs >= max_paragraphs:
            break

    stats = {
        "n_paragraphs_scanned":     n_paragraphs,
        "n_valid_unique_sentences": n_valid_seen,
    }
    return reservoir, stats


# ── Full dataset builder ──────────────────────────────────────────────────────

def build_dataset(target_per_lang: int = 1000,
                   output_path: str = "data/sentences.csv",
                   seed: int = RANDOM_SEED) -> List[Tuple[str, str]]:
    """
    Call collect_sentences() for every language in LANGUAGE_CONFIGS,
    combine into (text, label) pairs, save to CSV, and return the list.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print("=" * 60)
    print("  Collecting sentences from ai4bharat/IndicCorpV2")
    print(f"  Target: {target_per_lang} sentences per language")
    print("=" * 60)

    all_data: List[Tuple[str, str]] = []

    for lang_code in LANGUAGE_CONFIGS:
        print(f"  [{lang_code}] streaming ...", end=" ", flush=True)
        try:
            sentences, stats = collect_sentences(
                split_name    = lang_code,
                target        = target_per_lang,
                seed          = seed,
            )
            n = len(sentences)
            print(f"collected {n} sentences "
                  f"(scanned {stats['n_paragraphs_scanned']:,} paragraphs, "
                  f"{stats['n_valid_unique_sentences']:,} valid unique)")
            if n < target_per_lang:
                print(f"    WARNING: only {n}/{target_per_lang} collected for {lang_code}")
            all_data.extend((s, lang_code) for s in sentences)

        except Exception as exc:
            print(f"FAILED — {exc}")
            print(f"    Skipping {lang_code}.")

    # Save CSV
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(['text', 'label'])
        writer.writerows(all_data)

    print(f"\nSaved {len(all_data)} sentences → {output_path}")
    return all_data


# ── CSV loader (for cached runs) ──────────────────────────────────────────────

def load_csv(path: str) -> List[Tuple[str, str]]:
    """Load (text, label) pairs from a cached CSV file."""
    data: List[Tuple[str, str]] = []
    with open(path, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text  = row.get('text',  '').strip()
            label = row.get('label', '').strip()
            if text and label:
                data.append((text, label))
    return data


# ── Stratified Split ──────────────────────────────────────────────────────────

def stratified_split(
    data: List[Tuple[str, str]],
    train_ratio: float = 0.8,
    val_ratio:   float = 0.1,
    seed:        int   = RANDOM_SEED,
) -> Tuple[List, List, List]:
    """
    Stratified split into train / val / test partitions.
    With 1000 sentences/language: 800 train, 100 val, 100 test per language.
    Returns: (train_data, val_data, test_data) each as List[(text, label)].
    """
    rng = random.Random(seed)

    by_label: Dict[str, List[str]] = defaultdict(list)
    for text, label in data:
        by_label[label].append(text)

    train_data, val_data, test_data = [], [], []

    for label, texts in sorted(by_label.items()):
        rng.shuffle(texts)
        n       = len(texts)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        train_data.extend((t, label) for t in texts[:n_train])
        val_data.extend(  (t, label) for t in texts[n_train:n_train + n_val])
        test_data.extend( (t, label) for t in texts[n_train + n_val:])

    rng.shuffle(train_data)
    rng.shuffle(val_data)
    rng.shuffle(test_data)

    return train_data, val_data, test_data


# ── Label Encoder ─────────────────────────────────────────────────────────────

class LabelEncoder:
    """Maps string labels ↔ integer indices. No sklearn."""

    def __init__(self):
        self.classes_: List[str] = []
        self._label_to_idx: Dict[str, int] = {}

    def fit(self, labels: List[str]) -> 'LabelEncoder':
        self.classes_ = sorted(set(labels))
        self._label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        return self

    def transform(self, labels: List[str]) -> List[int]:
        return [self._label_to_idx[l] for l in labels]

    def inverse_transform(self, indices: List[int]) -> List[str]:
        return [self.classes_[i] for i in indices]

    def fit_transform(self, labels: List[str]) -> List[int]:
        return self.fit(labels).transform(labels)


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "यह एक परीक्षण वाक्य है। यहाँ दूसरा वाक्य है! क्या यह काम करता है?"
    sents  = sentence_split(sample)
    print("Tokenized sentences:")
    for s in sents:
        print(f"  {s!r}")
