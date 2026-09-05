import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from dataset_construction import COCOSubset, build_question_set, compute_cooccurrence, sample_image_ids


class FakeCOCO:
    def __init__(self):
        self.images = {1: {"file_name": "000000000001.jpg"}, 2: {"file_name": "000000000002.jpg"}}
        self.anns = {1: [{"category_id": 1}, {"category_id": 2}], 2: [{"category_id": 1}, {"category_id": 3}]}
    def getImgIds(self): return list(self.images)
    def getAnnIds(self, imgIds, iscrowd=None): return [id(self.anns[imgIds[0]])]
    def loadAnns(self, ids): return next(value for value in self.anns.values() if id(value) in ids)
    def loadImgs(self, ids): return [self.images[ids[0]]]


def test_dataset_is_balanced_and_deterministic():
    coco = COCOSubset(FakeCOCO(), Path("."), {1: "cat", 2: "dog", 3: "car", 4: "tree"})
    assert sample_image_ids(coco, 2, 4) == sample_image_ids(coco, 2, 4)
    questions = build_question_set(coco, [1, 2], compute_cooccurrence(coco), 4)
    assert [row["question_type"] for row in questions].count("present") == 2
    assert len({(row["image_id"], row["category"]) for row in questions}) == 6
    assert all(not row["ground_truth"] for row in questions if row["question_type"] != "present")
    print("All checks passed")

if __name__ == "__main__":
    test_dataset_is_balanced_and_deterministic()