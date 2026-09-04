from pathlib import Path
import sys

import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from inference import load_results
from probing import build_feature_matrix, load_split, train_probe
from generalization import compare_probe_to_baseline, cross_category_split


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="data/inference_results.pkl")
    parser.add_argument("--split", default="data/train_val_split.json")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", default="data/disagreements.csv")
    args = parser.parse_args()
    results = load_results(args.results)
    _, val_indices = load_split(args.split)
    train_indices, test_indices = cross_category_split(
        results, ["present", "absent_random"], ["absent_adversarial"], val_indices)
    usable = [index for index, result in enumerate(results) if result.parsed_answer is not None]
    positions = {index: position for position, index in enumerate(usable)}
    X, y = build_feature_matrix(results, args.layer, "mean")
    train_positions = [positions[index] for index in train_indices if index in positions]
    test_positions = [positions[index] for index in test_indices if index in positions]
    probe = train_probe(X[train_positions], y[train_positions])
    probe_predictions = probe.predict(X[test_positions])
    confidence_predictions = np.asarray([results[index].confidence >= 0.5 for index in test_indices if index in positions])
    metadata = [{"image_id": results[index].image_id, "category": results[index].category,
                 "question_type": results[index].question_type, "confidence": results[index].confidence}
                for index in test_indices if index in positions]
    table = compare_probe_to_baseline(probe_predictions, confidence_predictions, y[test_positions], metadata)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"wrote {len(table)} disagreements to {args.output}")