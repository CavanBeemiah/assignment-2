import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from probing import pool_layer


def test_pooling_shapes_and_values():
    states = np.arange(12, dtype=float).reshape(3, 4)
    assert np.array_equal(pool_layer(states, "mean"), [4, 5, 6, 7])
    assert pool_layer(states, "last").shape == (4,)
    assert pool_layer(states, "mean_max").shape == (8,)
    print("Tests passed")

if __name__ == "__main__":
    test_pooling_shapes_and_values()