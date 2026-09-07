from __future__ import annotations

import pickle
import re
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image


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
    transformers = importlib.import_module("transformers")
    AutoModelForImageTextToText = transformers.AutoModelForImageTextToText
    AutoProcessor = transformers.AutoProcessor
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
    return {layer: tensor.detach().float().cpu().numpy()[0] for layer, tensor in enumerate(states)}


def run_image_questions(model: Any, processor: Any, image: Any, questions: list[str],
                        pooling: str = "mean") -> list[InferenceResult]:
    """Run all questions for one image in a single batch."""
    prompts = []
    for question in questions:
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ]}]
        prompts.append(processor.apply_chat_template(messages, add_generation_prompt=True)
                       if hasattr(processor, "apply_chat_template") else f"<image>\n{question}")
    images = [image] * len(questions)
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    import torch

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
        )
        token_ids = generated.sequences[:, inputs["input_ids"].shape[1]:]
        texts = processor.batch_decode(token_ids, skip_special_tokens=True)
        batch_states = getattr(generated, "hidden_states", None)
        if not batch_states or not isinstance(batch_states[0], (tuple, list)):
            raise RuntimeError("The model did not return batched prompt hidden states")
        prompt_states = batch_states[0]
        results = []
        for row_index, (text, token_row) in enumerate(zip(texts, token_ids)):
            confidence = 0.0
            if generated.scores:
                token_scores = [scores.log_softmax(dim=-1)[row_index, token_id]
                                for scores, token_id in zip(generated.scores, token_row)]
                confidence = float(torch.stack(token_scores).mean().exp().item())
            states: dict[int, np.ndarray] = {}
            for layer, tensor in enumerate(prompt_states):
                layer_states = tensor[row_index].detach().float().cpu().numpy()
                states[layer] = _pool_hidden_state(layer_states, pooling)
            text = text.strip()
            results.append(InferenceResult(0, "", "", False, text, _parse_answer(text), confidence, states))
    return results


def _pool_hidden_state(hidden_states: np.ndarray, strategy: str) -> np.ndarray:
    if strategy == "mean":
        return hidden_states.mean(axis=0)
    if strategy == "last":
        return hidden_states[-1]
    if strategy == "mean_max":
        return np.concatenate((hidden_states.mean(axis=0), hidden_states.max(axis=0)))
    if strategy == "raw":
        return hidden_states
    raise ValueError("pooling must be 'mean', 'last', 'mean_max', or 'raw'")


def run_single_example(model: Any, processor: Any, image: Any, question: str) -> InferenceResult:
    """Run one image question; callers attach manifest metadata to the result."""
    return run_image_questions(model, processor, image, [question])[0]


def _extract_hidden_states_from_generation(generated: Any) -> dict[int, np.ndarray]:
    """Return hidden states for the prompt tokens recorded during generation."""
    states = getattr(generated, "hidden_states", None)
    if states is None or not states:
        raise RuntimeError("The model did not return hidden states during generation")
    if not isinstance(states[0], (tuple, list)):
        return _extract_hidden_states(generated)
    prompt_states = states[0]
    return {layer: tensor.detach().float().cpu().numpy()[0] for layer, tensor in enumerate(prompt_states)}


def run_inference_on_manifest(model: Any, processor: Any, manifest: list[dict], image_dir: str,
                              checkpoint_path: str | None = None, pooling: str = "mean") -> list[InferenceResult]:

    image_rows: dict[int, list[dict]] = {}
    for row in manifest:
        image_rows.setdefault(int(row["image_id"]), []).append(row)
    completed: set[int] = set()
    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            with Path(checkpoint_path).open("rb") as handle:
                while True:
                    payload = pickle.load(handle)
                    completed.add(int(payload["image_id"]))
        except EOFError:
            pass
    checkpoint_handle = Path(checkpoint_path).open("ab") if checkpoint_path else None
    try:
        for image_position, (image_id, rows) in enumerate(image_rows.items(), start=1):
            if image_id in completed:
                continue
            matches = list(Path(image_dir).glob(f"{image_id:012d}.*"))
            if not matches:
                raise FileNotFoundError(f"No COCO image found for image_id={image_id}")
            image = Image.open(matches[0]).convert("RGB")
            image_results = run_image_questions(model, processor, image, [row["question"] for row in rows], pooling)
            for row, result in zip(rows, image_results):
                result.image_id = image_id
                result.category = row["category"]
                result.question_type = row["question_type"]
                result.ground_truth = bool(row["ground_truth"])
            if checkpoint_handle:
                pickle.dump({"image_id": image_id, "results": image_results}, checkpoint_handle, protocol=pickle.HIGHEST_PROTOCOL)
                checkpoint_handle.flush()
                print(f"checkpoint: {image_position}/{len(image_rows)} images", flush=True)
    finally:
        if checkpoint_handle:
            checkpoint_handle.close()
    if not checkpoint_path:
        return [result for payload in [] for result in payload]
    return load_results(checkpoint_path)


def save_results(results: list[InferenceResult], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_results(path: str) -> list[InferenceResult]:
    with Path(path).open("rb") as handle:
        try:
            loaded = pickle.load(handle)
            if isinstance(loaded, list):
                return loaded
        except (EOFError, pickle.UnpicklingError):
            pass
    results: list[InferenceResult] = []
    with Path(path).open("rb") as handle:
        while True:
            try:
                payload = pickle.load(handle)
            except EOFError:
                break
            if isinstance(payload, dict) and "results" in payload:
                results.extend(payload["results"])
    return results


def summarize_results(results: list[InferenceResult]) -> dict[str, Any]:
    """Compute Q3/Q4 metrics directly from saved inference results."""
    unclear = [result for result in results if result.parsed_answer is None]
    accuracy: dict[str, float] = {}
    for question_type in sorted({result.question_type for result in results}):
        typed = [result for result in results if result.question_type == question_type]
        answered = [result for result in typed if result.parsed_answer is not None]
        accuracy[question_type] = (sum(result.parsed_answer == result.ground_truth for result in answered) / len(answered)
                       if answered else None)
    answered = [result for result in results if result.parsed_answer is not None]
    return {
        "total": len(results),
        "unclear_count": len(unclear),
        "unclear_rate": len(unclear) / len(results) if results else 0.0,
        "accuracy": sum(result.parsed_answer == result.ground_truth for result in answered) / len(answered)
        if answered else None,
        "accuracy_by_question_type": accuracy,
    }