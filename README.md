# CSC3043S Assignment 2

This repository implements the seeded COCO question set, frozen SmolVLM inference, hidden-state probing, and cross-category analysis required by the assignment.

## Run

Install dependencies with `python -m pip install -r requirements.txt`. Put the COCO annotations and `val2017` images outside git, then run:

```text
python scripts/build_dataset.py --annotations path/to/instances_val2017.json --images path/to/val2017
python scripts/run_inference.py --images path/to/val2017
python scripts/run_probes.py
```

The student seed is configured in `src/dataset_construction.py`. The raw hidden-state result cache is deliberately serialized as a pickle rather than JSON because it contains large NumPy arrays.