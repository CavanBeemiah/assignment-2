# CSC3043S Assignment 2 Draft Answers

Replace bracketed interpretation prompts with your own discussion after inspecting the generated examples.

## Q1
Student number: `BMHCAV001`. Seed: `1223053883751971958833`.

| Question type | Count |
|---|---:|
| present | 200 |
| absent_random | 200 |
| absent_adversarial | 200 |

The manifest contains 200 images and 600 QA pairs. The three question types are balanced: True.

## Q2
For each image, I computed category co-occurrence by counting each unordered pair once per image across the complete COCO annotation subset. The adversarial negative is the absent category with the largest co-occurrence count with any category present in that image; ties are resolved deterministically.

Example to verify against the annotation statistics: image `548267`, adversarial category `person`. Add the selected present category and exact frequency from your co-occurrence table here.

## Q3-Q10
Inference and probing artifacts are not present yet. Run the commands below, then rerun this script:

```powershell
python scripts/run_inference.py --images data/val2017
python scripts/summarize_results.py
python scripts/run_probes.py
python scripts/run_generalization.py --layer <best-layer>
python scripts/generate_report.py
```
