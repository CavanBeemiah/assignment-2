import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from generalization import compare_probe_to_baseline, cross_category_split


def test_generalization_uses_only_held_out_indices():
    results = [SimpleNamespace(question_type=kind) for kind in ("present", "absent_random", "absent_adversarial", "present")]
    train, test = cross_category_split(results, ["present", "absent_random"], ["absent_adversarial"], [0, 1, 2])
    assert train == [0, 1] and test == [2]


def test_disagreement_table_is_generated():
    pytest.importorskip("pandas")
    table = compare_probe_to_baseline(np.array([1, 0]), np.array([0, 0]), np.array([1, 1]), [{"image_id": 1}, {"image_id": 2}])
    assert len(table) == 1 and table.iloc[0]["image_id"] == 1