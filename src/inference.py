from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from transformers import AutoModelForImageTextToText, AutoProcessor
import numpy as np


@dataclass
class InferenceResult:
    image_id: int
    category: str
    question_type: str
    ground_truth: bool
    generated_text: str
    parsed_answer: bool | None
    confidence: float
    hidden_states: dict[int, np.ndarray]


def load_model(model_name: str, device: str = "cpu") -> tuple[Any, Any]:
    """Load a frozen Transformers model and processor once."""        
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(model_name)
    model.to(device).eval()
    return model, processor


def _parse_answer(text: str) -> bool | None:
    match = re.search(r"\b(yes|no)\b", text.lower())
    return None if match is None else match.group(1) == "yes"


def _extract_hidden_states(output: Any) -> dict[int, np.ndarray]:
    states = getattr(output, "hidden_states", None)
    if states is None:
        raise RuntimeError("The model did not return hidden states")
    if states and isinstance(states[0], (tuple, list)):
        states = states[-1]
    return {layer: tensor.detach().float().cpu().numpy()[0] for layer, tensor in enumerate(states)}


def run_single_example(model: Any, processor: Any, image: Any, question: str) -> InferenceResult:
    """Run one image question; callers attach manifest metadata to the result."""
    inputs = processor(text=question, images=image, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with __import__("torch").inference_mode():
        generated = model.generate(**inputs, max_new_tokens=8, return_dict_in_generate=True, output_scores=True)
        token_ids = generated.sequences[:, inputs["input_ids"].shape[1]:]
        text = processor.batch_decode(token_ids, skip_special_tokens=True)[0].strip()
        scores = generated.scores[0] if generated.scores else None
        confidence = 0.0 if scores is None else float(scores.softmax(dim=-1).max().item())
        full_ids = generated.sequences
        forward_inputs = dict(inputs)
        forward_inputs["input_ids"] = full_ids
        output = model(**forward_inputs, output_hidden_states=True, return_dict=True)
    return InferenceResult(0, "", "", False, text, _parse_answer(text), confidence, _extract_hidden_states(output))


def run_inference_on_manifest(model: Any, processor: Any, manifest: list[dict], image_dir: str) -> list[InferenceResult]:
    from PIL import Image

    results: list[InferenceResult] = []
    image_cache: dict[int, Any] = {}
    for row in manifest:
        image_id = int(row["image_id"])
        if image_id not in image_cache:
            matches = list(Path(image_dir).glob(f"{image_id:012d}.*"))
            if not matches:
                raise FileNotFoundError(f"No COCO image found for image_id={image_id}")
            image_cache[image_id] = Image.open(matches[0]).convert("RGB")
        result = run_single_example(model, processor, image_cache[image_id], row["question"])
        result.image_id = image_id
        result.category = row["category"]
        result.question_type = row["question_type"]
        result.ground_truth = bool(row["ground_truth"])
        results.append(result)
    return results


def save_results(results: list[InferenceResult], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_results(path: str) -> list[InferenceResult]:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)