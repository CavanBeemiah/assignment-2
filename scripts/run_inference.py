from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from dataset_construction import load_manifest
from inference import load_model, run_inference_on_manifest, save_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--images", required=True)
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="data/inference_results.pkl")
    args = parser.parse_args()
    model, processor = load_model(args.model, args.device)
    results = run_inference_on_manifest(model, processor, load_manifest(args.manifest), args.images)
    save_results(results, args.output)
    print(f"wrote {len(results)} results to {args.output}")