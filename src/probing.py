from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def pool_layer(hidden_states: np.ndarray, strategy: str) -> np.ndarray:
    """Pool (sequence, hidden) states while retaining a named experiment choice."""
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape (seq_len, hidden_dim)")
    if strategy == "mean":
        return hidden_states.mean(axis=0)
    if strategy == "last":
        return hidden_states[-1]
    if strategy == "mean_max":
        return np.concatenate((hidden_states.mean(axis=0), hidden_states.max(axis=0)))
    raise ValueError("strategy must be 'mean', 'last', or 'mean_max'")


def build_feature_matrix(results: list[Any], layer: int, strategy: str) -> tuple[np.ndarray, np.ndarray]:
    usable = [result for result in results if result.parsed_answer is not None]
    if not usable:
        return np.empty((0, 0)), np.empty(0, dtype=int)
    X = np.vstack([pool_layer(result.hidden_states[layer], strategy) for result in usable])
    y = np.asarray([result.parsed_answer == result.ground_truth for result in usable], dtype=int)
    return X, y


def train_val_split(results: list[Any], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    by_type: dict[str, list[int]] = {}
    for index, result in enumerate(results):
        if result.parsed_answer is not None:
            by_type.setdefault(result.question_type, []).append(index)
    train, validation = [], []
    for indices in by_type.values():
        shuffled = np.asarray(indices)[rng.permutation(len(indices))]
        n_val = max(1, int(round(len(indices) * val_fraction)))
        validation.extend(int(value) for value in shuffled[:n_val])
        train.extend(int(value) for value in shuffled[n_val:])
    return sorted(train), sorted(validation)


def train_probe(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    from sklearn.linear_model import LogisticRegression
    probe = LogisticRegression(max_iter=2000, random_state=0)
    return probe.fit(X_train, y_train)


def evaluate_probe(probe: Any, X_val: np.ndarray, y_val: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, roc_auc_score
    scores = probe.predict_proba(X_val)[:, 1]
    try:
        auroc = float(roc_auc_score(y_val, scores))
    except ValueError:
        auroc = float("nan")
    return {"accuracy": float(accuracy_score(y_val, scores >= 0.5)), "auroc": auroc}


def save_split(train_indices: list[int], val_indices: list[int], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"train_indices": train_indices, "val_indices": val_indices}, indent=2) + "\n", encoding="utf-8")


def load_split(path: str) -> tuple[list[int], list[int]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["train_indices"], data["val_indices"]