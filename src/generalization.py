from __future__ import annotations

from typing import Any

import numpy as np


def cross_category_split(results: list[Any], train_types: list[str], test_types: list[str], val_indices: list[int]) -> tuple[list[int], list[int]]:
    held_out = set(val_indices)
    train = [index for index in val_indices if results[index].question_type in train_types]
    test = [index for index in val_indices if results[index].question_type in test_types]
    if set(train) & set(test) or not held_out.issuperset(train + test):
        raise AssertionError("cross-category split violates held-out-set isolation")
    return train, test


def compare_probe_to_baseline(probe_predictions: np.ndarray, confidence_predictions: np.ndarray,
                              y_true: np.ndarray, metadata: list[dict]) -> Any:
    import pandas as pd
    if not (len(probe_predictions) == len(confidence_predictions) == len(y_true) == len(metadata)):
        raise ValueError("prediction arrays and metadata must have equal length")
    rows = []
    for probe, baseline, truth, item in zip(probe_predictions, confidence_predictions, y_true, metadata):
        if bool(probe) != bool(baseline):
            rows.append({**item, "ground_truth": bool(truth), "probe_prediction": bool(probe),
                         "confidence_prediction": bool(baseline)})
    return pd.DataFrame(rows)