"""
lab1_solution.py — Lab 1: Indic Language Identification
Main pipeline: data collection → TF-IDF → Logistic Regression → Evaluation.

Usage:
    # First run (downloads data):
    python lab1_solution.py

    # Subsequent runs (uses cached data):
    python lab1_solution.py --use-cache

    # Skip data download, use existing data/sentences.csv:
    python lab1_solution.py --use-cache --data-path data/sentences.csv
"""

import os
import sys

# ── Force UTF-8 on Windows ────────────────────────────────────────────────────
# Windows defaults to cp1252 which can't decode Indic Unicode.
# Setting PYTHONUTF8 propagates to all subprocesses; reconfiguring stdout/stderr
# fixes the interactive terminal display as well.
if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import time
import argparse
import numpy as np
from scipy.sparse import save_npz, load_npz

# ── Local modules (no sklearn) ────────────────────────────────────────────────
from data_utils          import build_dataset, load_csv, stratified_split, LabelEncoder
from tfidf_vectorizer    import CustomTFIDFVectorizer
from logistic_regression import CustomLogisticRegression
from evaluation          import (
    print_evaluation_report,
    print_confusion_matrix,
    macro_f1_score,
    accuracy_score,
)

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_PATH        = "data/sentences.csv"
TFIDF_SAVE_PATH  = "data/tfidf_vectorizer"
MODEL_SAVE_PATH  = "data/lr_model"
TRAIN_X_PATH     = "data/X_train.npz"
VAL_X_PATH       = "data/X_val.npz"
TEST_X_PATH      = "data/X_test.npz"

RANDOM_SEED      = 42
SENTENCES_PER_LANG = 1000

# TF-IDF hyper-parameters
TFIDF_CONFIG = dict(
    word_ngram_range = (1, 2),    # unigrams + bigrams
    char_ngram_range = (2, 4),    # 2-grams, 3-grams, 4-grams
    min_df           = 2,
    max_features     = 200_000,
    sublinear_tf     = True,
)

# Logistic Regression hyper-parameters
LR_CONFIG = dict(
    learning_rate = 0.5,
    epochs        = 100,
    batch_size    = 256,
    lambda_reg    = 1e-3,
    lr_decay      = 0.95,
    verbose       = True,
    print_every   = 10,
    random_state  = RANDOM_SEED,
)


# ── Step 1: Data Collection ───────────────────────────────────────────────────

def step1_collect_data(use_cache: bool, data_path: str):
    """Download / load the sentence dataset."""
    print("\n" + "=" * 60)
    print("  STEP 1 — Data Collection & Sentence Tokenisation")
    print("=" * 60)

    if use_cache and os.path.exists(data_path) and os.path.getsize(data_path) > 100:
        print(f"[Data] Loading cached dataset from {data_path} ...")
        data = load_csv(data_path)
    else:
        print("[Data] Streaming IndicCorpV2 from HuggingFace ...")
        data = build_dataset(
            target_per_lang = SENTENCES_PER_LANG,
            output_path     = data_path,
            seed            = RANDOM_SEED,
        )

    # Summary
    from collections import Counter
    label_counts = Counter(lbl for _, lbl in data)
    print(f"\n[Data] Total sentences: {len(data)}")
    print(f"[Data] Languages found: {len(label_counts)}")
    print(f"[Data] Sentences per language (min/max): "
          f"{min(label_counts.values())} / {max(label_counts.values())}")
    return data


# ── Step 2: Stratified Split ──────────────────────────────────────────────────

def step2_split(data):
    """Stratified 80:10:10 split."""
    print("\n" + "=" * 60)
    print("  STEP 2 — Stratified Train / Val / Test Split (80:10:10)")
    print("=" * 60)

    train_data, val_data, test_data = stratified_split(
        data,
        train_ratio = 0.8,
        val_ratio   = 0.1,
        seed        = RANDOM_SEED,
    )

    print(f"[Split] Train : {len(train_data)} samples")
    print(f"[Split] Val   : {len(val_data)} samples")
    print(f"[Split] Test  : {len(test_data)} samples")

    # Unpack
    X_train_raw, y_train = zip(*train_data)
    X_val_raw,   y_val   = zip(*val_data)
    X_test_raw,  y_test  = zip(*test_data)

    return (list(X_train_raw), list(y_train),
            list(X_val_raw),   list(y_val),
            list(X_test_raw),  list(y_test))


# ── Step 3: TF-IDF Vectorisation ─────────────────────────────────────────────

def step3_tfidf(X_train_raw, X_val_raw, X_test_raw, use_cache: bool):
    """Fit custom TF-IDF on train and transform all splits."""
    print("\n" + "=" * 60)
    print("  STEP 3 — Custom TF-IDF Vectorisation")
    print("=" * 60)
    print(f"  Features: Word 1-grams + 2-grams | Char 2/3/4-grams")
    print(f"  max_features={TFIDF_CONFIG['max_features']:,} | "
          f"min_df={TFIDF_CONFIG['min_df']} | "
          f"sublinear_tf={TFIDF_CONFIG['sublinear_tf']}")

    cache_exists = (
        use_cache
        and os.path.exists(TRAIN_X_PATH)
        and os.path.exists(VAL_X_PATH)
        and os.path.exists(TEST_X_PATH)
    )

    if cache_exists:
        print("[TF-IDF] Loading cached sparse matrices ...")
        X_train = load_npz(TRAIN_X_PATH)
        X_val   = load_npz(VAL_X_PATH)
        X_test  = load_npz(TEST_X_PATH)
        vec     = CustomTFIDFVectorizer.load(TFIDF_SAVE_PATH + ".npz")
    else:
        t0  = time.time()
        vec = CustomTFIDFVectorizer(**TFIDF_CONFIG)

        # Fit ONLY on training data (no data leakage)
        X_train = vec.fit_transform(X_train_raw)
        print(f"[TF-IDF] Transforming val  set ...")
        X_val   = vec.transform(X_val_raw)
        print(f"[TF-IDF] Transforming test set ...")
        X_test  = vec.transform(X_test_raw)

        elapsed = time.time() - t0
        print(f"[TF-IDF] Done in {elapsed:.1f}s")

        # Cache results
        os.makedirs("data", exist_ok=True)
        save_npz(TRAIN_X_PATH, X_train)
        save_npz(VAL_X_PATH,   X_val)
        save_npz(TEST_X_PATH,  X_test)
        vec.save(TFIDF_SAVE_PATH)

    print(f"\n[TF-IDF] Matrix shapes:")
    print(f"  X_train : {X_train.shape}  nnz={X_train.nnz:,}")
    print(f"  X_val   : {X_val.shape}   nnz={X_val.nnz:,}")
    print(f"  X_test  : {X_test.shape}  nnz={X_test.nnz:,}")

    return vec, X_train, X_val, X_test


# ── Step 4: Train Logistic Regression ────────────────────────────────────────

def step4_train(X_train, y_train, use_cache: bool):
    """Train custom multinomial logistic regression."""
    print("\n" + "=" * 60)
    print("  STEP 4 — Custom Logistic Regression Training")
    print("=" * 60)

    model_file = MODEL_SAVE_PATH + ".npz"

    if use_cache and os.path.exists(model_file):
        print("[LR] Loading cached model ...")
        clf = CustomLogisticRegression.load(model_file)
    else:
        clf = CustomLogisticRegression(**LR_CONFIG)
        t0  = time.time()
        clf.fit(X_train, list(y_train))
        elapsed = time.time() - t0
        print(f"[LR] Training finished in {elapsed:.1f}s")
        clf.save(MODEL_SAVE_PATH)

    return clf


# ── Step 5: Evaluation ────────────────────────────────────────────────────────

def step5_evaluate(clf, X_val, y_val, X_test, y_test):
    """Evaluate model on validation and test sets."""
    print("\n" + "=" * 60)
    print("  STEP 5 — Evaluation (Custom Macro-F1)")
    print("=" * 60)

    # Validation
    val_preds = clf.predict(X_val).tolist()
    mac_f1_val, acc_val = print_evaluation_report(
        list(y_val), val_preds, split_name="Validation"
    )

    # Test
    test_preds = clf.predict(X_test).tolist()
    mac_f1_test, acc_test = print_evaluation_report(
        list(y_test), test_preds, split_name="Test"
    )

    # Confusion matrix on test set
    print_confusion_matrix(list(y_test), test_preds)

    # Final summary
    print("\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Validation Macro-F1 : {mac_f1_val:.4f}")
    print(f"  Test       Macro-F1 : {mac_f1_test:.4f}")
    print(f"  Validation Accuracy : {acc_val:.4f}")
    print(f"  Test       Accuracy : {acc_test:.4f}")
    print("=" * 60)

    return mac_f1_test


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Lab 1 — Indic Language Identification Pipeline"
    )
    parser.add_argument(
        "--use-cache", action="store_true",
        help="Load cached data/TF-IDF matrices/model instead of recomputing."
    )
    parser.add_argument(
        "--data-path", type=str, default=DATA_PATH,
        help=f"Path to sentences CSV (default: {DATA_PATH})"
    )
    args = parser.parse_args()

    total_start = time.time()

    print("\n" + "=" * 60)
    print("  Lab 1: Indic Language Identification")
    print("  Dataset : IndicCorpV2 (ai4bharat/IndicCorpV2)")
    print("  Model   : Custom TF-IDF + Custom Logistic Regression")
    print("  Metric  : Custom Macro-F1 (no sklearn)")
    print("=" * 60)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    data = step1_collect_data(args.use_cache, args.data_path)

    (X_train_raw, y_train,
     X_val_raw,   y_val,
     X_test_raw,  y_test) = step2_split(data)

    vec, X_train, X_val, X_test = step3_tfidf(
        X_train_raw, X_val_raw, X_test_raw, args.use_cache
    )

    clf = step4_train(X_train, y_train, args.use_cache)

    macro_f1 = step5_evaluate(clf, X_val, y_val, X_test, y_test)

    total_time = time.time() - total_start
    print(f"\n  Total wall-clock time : {total_time / 60:.1f} min")
    print(f"  Final Test Macro-F1  : {macro_f1:.4f}\n")


if __name__ == "__main__":
    main()
