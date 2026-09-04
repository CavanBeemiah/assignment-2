from pathlib import Path
import csv
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from inference import load_results
from probing import build_feature_matrix, evaluate_probe, load_split, save_split, train_probe, train_val_split


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="data/inference_results.pkl")
    parser.add_argument("--split", default="data/train_val_split.json")
    parser.add_argument("--output", default="data/layer_auroc.csv")
    parser.add_argument("--strategy", default="mean")
    parser.add_argument("--seed", type=int, default=3043)
    args = parser.parse_args()
    results = load_results(args.results)
    split_path = Path(args.split)
    if split_path.exists():
        train_indices, val_indices = load_split(args.split)
    else:
        train_indices, val_indices = train_val_split(results, 0.3, args.seed)
        save_split(train_indices, val_indices, args.split)
    layers = sorted(set(layer for result in results for layer in result.hidden_states))
    rows = []
    for layer in layers:
        X, y = build_feature_matrix(results, layer, args.strategy)
        train = np.asarray([index for index in train_indices if results[index].parsed_answer is not None])
        validation = np.asarray([index for index in val_indices if results[index].parsed_answer is not None])
        usable = [index for index, result in enumerate(results) if result.parsed_answer is not None]
        positions = {index: position for position, index in enumerate(usable)}
        probe = train_probe(X[[positions[index] for index in train]], y[[positions[index] for index in train]])
        metrics = evaluate_probe(probe, X[[positions[index] for index in validation]], y[[positions[index] for index in validation]])
        rows.append({"layer": layer, **metrics})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["layer", "accuracy", "auroc"])
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} layer results to {args.output}")