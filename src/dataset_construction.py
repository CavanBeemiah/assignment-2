from __future__ import annotations

import csv
import importlib
from pycocotools.coco import COCO
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any
import numpy as np


student_number = "BMHCAV001"
seed = int.from_bytes(
student_number.encode("utf-8"),
byteorder="big"
)
rng = np.random.default_rng(seed=seed)
QUESTION_TEMPLATE = "Is there a {object} in this image? Answer yes or no."


@dataclass
class COCOSubset:
    
    def __init__(self, coco: Any, image_dir: Path, category_names: dict[int, str]) -> None:
        self.coco = coco
        self.image_dir = image_dir
        self.category_names = category_names
        
    def categories_for_image(self, image_id: int) -> set[str]:
        annotation_ids = self.coco.getAnnIds(imgIds=[image_id], iscrowd=None)
        annotations = self.coco.loadAnns(annotation_ids)
        return set(self.category_names[a["category_id"]] for a in annotations)

    def image_path(self, image_id: int) -> Path:
        info = self.coco.loadImgs([image_id])[0]
        return self.image_dir / info["file_name"]


def load_coco_subset(annotation_path: str, image_dir: str) -> COCOSubset:
    """Wrap pycocotools access to the provided COCO subset."""
    coco = COCO(annotation_path)
    categories = coco.loadCats(coco.getCatIds())
    category_names = {category["id"]: category["name"] for category in categories}
    return COCOSubset(coco, Path(image_dir), category_names)


def compute_cooccurrence(coco: COCOSubset) -> dict[tuple[str, str], int]:
    """Return symmetric pairwise category co-occurrence counts per image."""
    counts: dict[tuple[str, str], int] = {}
    for image_id in sorted(coco.coco.getImgIds()):
        names = sorted(coco.categories_for_image(image_id))
        for first, second in combinations(names, 2):
            counts[(first, second)] = counts.get((first, second), 0) + 1
            counts[(second, first)] = counts[(first, second)]
    return counts


def sample_image_ids(coco: COCOSubset, n_images: int, seed: int) -> list[int]:
    """Sample image IDs deterministically without replacement."""
    image_ids = np.asarray(sorted(coco.coco.getImgIds()), dtype=np.int64)
    if n_images < 0 or n_images > len(image_ids):
        raise ValueError(f"n_images must be between 0 and {len(image_ids)}")
    return [int(value) for value in np.random.default_rng(seed).choice(image_ids, n_images, replace=False)]


def build_question_set(coco: COCOSubset, image_ids: list[int],
                       cooccurrence: dict[tuple[str, str], int], seed: int) -> list[dict[str, Any]]:
    """Build three fixed-template, labelled existence questions per image."""
    rng = np.random.default_rng(seed)
    all_categories = sorted(coco.category_names.values())
    questions: list[dict[str, Any]] = []
    for image_id in image_ids:
        present = sorted(coco.categories_for_image(image_id))
        if not present:
            raise ValueError(f"Image {image_id} has no annotated categories")
        absent = sorted(set(all_categories) - set(present))
        if len(absent) < 2:
            raise ValueError(f"Image {image_id} needs at least two absent categories")
        present_category = str(rng.choice(present))
        random_category = str(rng.choice(absent))
        scores = {candidate: max((cooccurrence.get((candidate, name), 0) for name in present), default=0)
              for candidate in absent if candidate != random_category}
        best_score = max(scores.values())
        best_candidates = sorted(candidate for candidate, score in scores.items() if score == best_score)
        adversarial_category = str(rng.choice(best_candidates))
        selected = ((present_category, "present", True),
                    (random_category, "absent_random", False),
                    (adversarial_category, "absent_adversarial", False))
        for category, question_type, ground_truth in selected:
            questions.append({"image_id": int(image_id), "category": category,
                              "question": QUESTION_TEMPLATE.format(object=category),
                              "question_type": question_type, "ground_truth": ground_truth})
    return questions


MANIFEST_FIELDS = ["image_id", "category", "question", "question_type", "ground_truth"]


def save_manifest(questions: list[dict], path: str) -> None:
    """Write a stable CSV manifest with a fixed column order."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: question[field] for field in MANIFEST_FIELDS} for question in questions)


def load_manifest(path: str) -> list[dict]:
    """Load a manifest and restore numeric and boolean fields."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [{**row, "image_id": int(row["image_id"]), "ground_truth": row["ground_truth"].lower() == "true"}
            for row in rows]
    