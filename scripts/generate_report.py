"""Generate an answer-ready Q1-Q10 draft from saved checkpoint artifacts."""

from __future__ import annotations

import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from dataset_construction import seed, student_number

STUDENT_NUMBER = student_number
SEED = seed


ROOT = Path(__file__).parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pct(value: float | None) -> str:
    return "not available" if value is None else f"{value:.3f}"


def result_dicts(path: Path) -> list[dict]:
    with path.open("rb") as handle:
        results = pickle.load(handle)
    return [{"image_id": r.image_id, "category": r.category, "question_type": r.question_type,
             "ground_truth": r.ground_truth, "generated_text": r.generated_text,
             "parsed_answer": r.parsed_answer, "confidence": r.confidence}
            for r in results]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--results", default="data/inference_results.pkl")
    parser.add_argument("--layers", default="data/layer_auroc.csv")
    parser.add_argument("--disagreements", default="data/disagreements.csv")
    parser.add_argument("--output", default="report_draft.md")
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    questions = read_csv(manifest_path)
    counts = Counter(row["question_type"] for row in questions)
    images = sorted({row["image_id"] for row in questions})
    report = [
        "# CSC3043S Assignment 2 Draft Answers",
        "",
        "Replace bracketed interpretation prompts with your own discussion after inspecting the generated examples.",
        "",
        "## Q1",
        f"Student number: `{STUDENT_NUMBER}`. Seed: `{SEED}`.",
        "",
        "| Question type | Count |",
        "|---|---:|",
    ]
    report.extend(f"| {kind} | {counts.get(kind, 0)} |" for kind in ("present", "absent_random", "absent_adversarial"))
    report.extend(["", f"The manifest contains {len(images)} images and {len(questions)} QA pairs. The three question types are balanced: {len(set(counts.values())) == 1}.", "",
                   "## Q2", "For each image, I computed category co-occurrence by counting each unordered pair once per image across the complete COCO annotation subset. The adversarial negative is the absent category with the largest co-occurrence count with any category present in that image; ties are resolved deterministically.", ""])
    example = next((row for row in questions if row["question_type"] == "absent_adversarial"), None)
    if example:
        report.append(f"Example to verify against the annotation statistics: image `{example['image_id']}`, adversarial category `{example['category']}`. Add the selected present category and exact frequency from your co-occurrence table here.")

    results_path = ROOT / args.results
    if not results_path.exists():
        report.extend(["", "## Q3-Q10", "Inference and probing artifacts are not present yet. Run the commands below, then rerun this script:", "", "```powershell", "python scripts/run_inference.py --images data/val2017", "python scripts/summarize_results.py", "python scripts/run_probes.py", "python scripts/run_generalization.py --layer <best-layer>", "python scripts/generate_report.py", "```"])
    else:
        results = result_dicts(results_path)
        answered = [row for row in results if row["parsed_answer"] is not None]
        unclear = len(results) - len(answered)
        report.extend(["", "## Q3", f"The unclear-answer rate was {unclear / len(results):.3f} ({unclear}/{len(results)}). Unclear generations were excluded from probe training and accuracy calculations because they do not provide a binary answer.", "", "## Q4", "| Question type | Accuracy |", "|---|---:|"])
        for kind in ("present", "absent_random", "absent_adversarial"):
            typed = [row for row in answered if row["question_type"] == kind]
            accuracy = sum((row["parsed_answer"] is True) == (str(row["ground_truth"]).lower() == "true")
                           for row in typed) / len(typed) if typed else None
            report.append(f"| {kind} | {pct(accuracy)} |")
        report.extend(["", "The hardest question type is [fill in from the table]. This is [consistent/inconsistent] with the expectation that adversarial negatives exploit category co-occurrence.", "", "## Q5", "I used mean pooling over the raw token hidden states to obtain one fixed-length vector per example. Mean pooling uses information across the sequence and is less sensitive to the final-token position than last-token pooling; the named strategy remains configurable in the code.", "", "## Q6"])
        layers_path = ROOT / args.layers
        if layers_path.exists():
            layer_rows = read_csv(layers_path)
            report.extend(["| Layer | Accuracy | AUROC |", "|---:|---:|---:|"])
            report.extend(f"| {row['layer']} | {row['accuracy']} | {row['auroc']} |" for row in layer_rows)
            best = max(layer_rows, key=lambda row: float(row["auroc"]))
            report.append(f"\nThe best tested layer by AUROC was layer {best['layer']} ({best['auroc']}). Interpret its depth as [early/middle/late] after checking the model's layer count.")
        else:
            report.append("Run `scripts/run_probes.py` to create the layer-versus-AUROC table.")
        report.extend(["", "## Q7", "Compare the best-layer AUROC with the confidence-baseline AUROC from the saved predictions. The difference is [probe AUROC - baseline AUROC].", "", "## Q8", "Compare the within-category validation result with the cross-category result in the saved generalisation output. The gap is [value]. A large gap would suggest category-construction-specific information rather than a fully general correctness signal.", "", "## Q9"])
        disagreements_path = ROOT / args.disagreements
        if disagreements_path.exists():
            disagreements = read_csv(disagreements_path)
            report.extend(["| Image | Category | Question type | Confidence | Ground truth | Probe prediction |", "|---:|---|---|---:|---|---|"])
            report.extend(f"| {row.get('image_id', '')} | {row.get('category', '')} | {row.get('question_type', '')} | {row.get('confidence', '')} | {row.get('ground_truth', '')} | {row.get('probe_prediction', '')} |" for row in disagreements[:3])
        else:
            report.append("Run `scripts/run_generalization.py` to create the disagreement table.")
        report.extend(["", "## Q10", "The evidence [does/does not] support the claim that hidden states contain information about hallucination correctness beyond stated confidence. This conclusion is limited by the 600-example sample, one frozen VLM, the single COCO subset, and the specific random/adversarial negative construction. It should therefore be treated as evidence about this controlled experiment, not as a general claim about all vision-language models."])

    output = ROOT / args.output
    output.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()