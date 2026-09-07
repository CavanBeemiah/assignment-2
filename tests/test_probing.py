import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from probing import build_feature_matrix, train_val_split


def test_split_is_stratified_and_feature_labels_are_correct():
    results = [SimpleNamespace(question_type=kind, parsed_answer=True, ground_truth=True,
                               hidden_states={0: np.ones((2, 3))})
               for kind in ("present", "absent_random", "absent_adversarial") for _ in range(4)]
    train, validation = train_val_split(results, 0.25, 9)
    assert set(train).isdisjoint(validation)
    assert sorted(train + validation) == list(range(12))
    X, y = build_feature_matrix(results, 0, "mean")
    assert X.shape == (12, 3)
    assert np.all(y == 1)
    print("Tests passed")

if __name__ == "__main__":
    test_split_is_stratified_and_feature_labels_are_correct()