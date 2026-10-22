"""
logistic_regression.py — Lab 1: Language Identification
Custom Multinomial Logistic Regression (Softmax Regression).
Trained with mini-batch SGD + L2 regularisation.
NO sklearn used anywhere.
"""

import numpy as np
from scipy.sparse import issparse, csr_matrix
from typing import List, Optional


class CustomLogisticRegression:
    """
    Multinomial (Softmax) Logistic Regression classifier.

    Model:
        logits = X @ W.T + b          shape: (batch, n_classes)
        probs  = softmax(logits)

    Loss:
        L = CrossEntropy(probs, y_onehot) + (lambda/2) * ||W||_F^2

    Optimiser:
        Mini-batch SGD with step-decay learning rate schedule.

    Parameters
    ----------
    learning_rate : float
        Initial learning rate.
    epochs : int
        Number of full passes over training data.
    batch_size : int
        Mini-batch size for SGD.
    lambda_reg : float
        L2 regularisation strength.
    lr_decay : float
        Multiplicative LR decay applied every epoch.
    verbose : bool
        Print loss + accuracy every `print_every` epochs.
    print_every : int
        Frequency of logging.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        learning_rate: float = 0.5,
        epochs: int          = 100,
        batch_size: int      = 256,
        lambda_reg: float    = 1e-3,
        lr_decay: float      = 0.95,
        verbose: bool        = True,
        print_every: int     = 10,
        random_state: int    = 42,
    ):
        self.learning_rate = learning_rate
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.lambda_reg    = lambda_reg
        self.lr_decay      = lr_decay
        self.verbose       = verbose
        self.print_every   = print_every
        self.random_state  = random_state

        # Learned parameters (set after fit)
        self.W: Optional[np.ndarray] = None  # (n_classes, n_features)  float32
        self.b: Optional[np.ndarray] = None  # (n_classes,)             float32
        self.classes_: Optional[np.ndarray] = None
        self._label_to_idx: dict = {}
        self.train_loss_history: List[float] = []

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Row-wise numerically stable softmax."""
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_x   = np.exp(shifted)
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def _encode_labels(self, y: List[str]) -> np.ndarray:
        """String labels → integer indices (int32)."""
        return np.array([self._label_to_idx[lbl] for lbl in y], dtype=np.int32)

    @staticmethod
    def _sparse_matmul(X, W: np.ndarray) -> np.ndarray:
        """
        Compute X @ W efficiently, supporting both dense and CSR sparse X.
        Returns dense ndarray.
        """
        if issparse(X):
            return X.dot(W)          # scipy handles sparse × dense
        return X @ W

    # ── Fit ────────────────────────────────────────────────────────────────────

    def fit(self, X, y: List[str]) -> "CustomLogisticRegression":
        """
        Train the classifier.

        Parameters
        ----------
        X : array-like or scipy CSR matrix, shape (n_samples, n_features)
        y : list of string class labels, length n_samples
        """
        rng = np.random.RandomState(self.random_state)

        # ── Setup ──────────────────────────────────────────────────────────────
        self.classes_      = np.array(sorted(set(y)))
        self._label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        n_classes  = len(self.classes_)
        n_features = X.shape[1]
        n_samples  = X.shape[0]

        # Weight initialisation: small random values (Xavier-style)
        scale   = 1.0 / np.sqrt(n_features)
        self.W  = rng.randn(n_classes, n_features).astype(np.float32) * scale
        self.b  = np.zeros(n_classes, dtype=np.float32)

        # Encode labels & one-hot matrix
        y_enc   = self._encode_labels(y)
        Y_oh    = np.zeros((n_samples, n_classes), dtype=np.float32)
        Y_oh[np.arange(n_samples), y_enc] = 1.0

        self.train_loss_history = []
        lr = float(self.learning_rate)

        print(f"\n[LR] Training: {n_samples} samples | "
              f"{n_classes} classes | {n_features:,} features")
        print(f"[LR] epochs={self.epochs}, batch={self.batch_size}, "
              f"lr={lr}, λ={self.lambda_reg}\n")

        # ── Training loop ──────────────────────────────────────────────────────
        for epoch in range(1, self.epochs + 1):
            perm       = rng.permutation(n_samples)
            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, n_samples, self.batch_size):
                batch_idx = perm[start : start + self.batch_size]
                batch_n   = len(batch_idx)

                X_b = X[batch_idx]          # sparse slice still CSR
                Y_b = Y_oh[batch_idx]       # (batch, n_classes)

                # ── Forward pass ───────────────────────────────────────────────
                # logits: (batch, n_classes)
                logits = self._sparse_matmul(X_b, self.W.T) + self.b
                probs  = self._softmax(logits)               # (batch, n_classes)

                # ── Loss ───────────────────────────────────────────────────────
                log_p    = np.log(probs + 1e-10)
                ce_loss  = -np.sum(Y_b * log_p) / batch_n
                reg_loss = 0.5 * self.lambda_reg * float(np.sum(self.W ** 2))
                loss     = ce_loss + reg_loss

                epoch_loss += loss
                n_batches  += 1

                # ── Gradients ──────────────────────────────────────────────────
                delta = (probs - Y_b) / batch_n              # (batch, n_classes)

                # dW: (n_classes, n_features)
                if issparse(X_b):
                    # Sparse.dot(Dense) is safer than Dense.dot(Sparse) in scipy
                    dW = (X_b.T.dot(delta)).T.astype(np.float32)
                else:
                    dW = (delta.T @ X_b).astype(np.float32)

                dW += self.lambda_reg * self.W               # L2 gradient term
                db  = delta.sum(axis=0).astype(np.float32)  # (n_classes,)

                # ── Parameter update ───────────────────────────────────────────
                self.W -= lr * dW
                self.b -= lr * db

            # Step-decay learning rate
            lr *= self.lr_decay

            avg_loss = epoch_loss / max(n_batches, 1)
            self.train_loss_history.append(avg_loss)

            if self.verbose and epoch % self.print_every == 0:
                train_acc = self._batch_accuracy(X, y, rng, max_samples=2000)
                print(f"  Epoch {epoch:4d}/{self.epochs} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Train Acc (sample): {train_acc:.4f} | "
                      f"lr: {lr:.5f}")

        print("\n[LR] Training complete.")
        return self

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict_proba(self, X) -> np.ndarray:
        """
        Return class probability matrix.
        Shape: (n_samples, n_classes)
        """
        logits = self._sparse_matmul(X, self.W.T) + self.b
        return self._softmax(logits)

    def predict(self, X) -> np.ndarray:
        """
        Return predicted string labels for each sample.
        Shape: (n_samples,)
        """
        probs   = self.predict_proba(X)
        indices = probs.argmax(axis=1)
        return self.classes_[indices]

    # ── Utility ────────────────────────────────────────────────────────────────

    def _batch_accuracy(self, X, y: List[str],
                         rng: np.random.RandomState,
                         max_samples: int = 2000) -> float:
        """Fast accuracy estimate on a random subset (avoids slow full pass)."""
        n = X.shape[0]
        if n <= max_samples:
            idx = np.arange(n)
        else:
            idx = rng.choice(n, max_samples, replace=False)
        preds = self.predict(X[idx])
        y_sub = np.array(y)[idx]
        return float(np.mean(preds == y_sub))

    def score(self, X, y: List[str]) -> float:
        """Return exact accuracy over full dataset."""
        preds = self.predict(X)
        return float(np.mean(preds == np.array(y)))

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save model weights and metadata."""
        np.savez_compressed(
            path,
            W       = self.W,
            b       = self.b,
            classes = self.classes_,
        )
        print(f"[LR] Model saved → {path}.npz")

    @classmethod
    def load(cls, path: str) -> "CustomLogisticRegression":
        """Load a saved model."""
        data = np.load(path, allow_pickle=True)
        model = cls()
        model.W        = data["W"]
        model.b        = data["b"]
        model.classes_ = data["classes"]
        model._label_to_idx = {c: i for i, c in enumerate(model.classes_)}
        print(f"[LR] Model loaded from {path}")
        return model


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from scipy.sparse import random as sp_random
    rng  = np.random.RandomState(0)
    X    = sp_random(200, 500, density=0.1, format="csr", random_state=0)
    y    = [f"lang_{i % 5}" for i in range(200)]
    clf  = CustomLogisticRegression(epochs=20, verbose=True, print_every=5)
    clf.fit(X, y)
    preds = clf.predict(X[:5])
    print("Sample predictions:", preds)
