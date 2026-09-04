from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/layer_auroc.csv")
    parser.add_argument("--output", default="data/layer_auroc.png")
    args = parser.parse_args()
    table = pd.read_csv(args.input).sort_values("layer")
    plt.figure(figsize=(7, 4))
    plt.plot(table["layer"], table["auroc"], marker="o", label="AUROC")
    plt.plot(table["layer"], table["accuracy"], marker="s", label="Accuracy")
    plt.xlabel("Layer"); plt.ylabel("Score"); plt.ylim(0, 1); plt.legend(); plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=180)