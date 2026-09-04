from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from dataset_construction import (SEED, build_question_set, compute_cooccurrence,
                                  load_coco_subset, sample_image_ids, save_manifest)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", default="data/manifest.csv")
    parser.add_argument("--n-images", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    coco = load_coco_subset(args.annotations, args.images)
    image_ids = sample_image_ids(coco, args.n_images, args.seed)
    questions = build_question_set(coco, image_ids, compute_cooccurrence(coco), args.seed)
    save_manifest(questions, args.output)
    print(f"wrote {len(questions)} questions to {args.output}")