"""Weight-space soups: average fine-tuned Ultralytics checkpoints.

All members are fine-tuned from the same baseline, so their weights can be
averaged directly. Averaged BatchNorm running statistics do not exactly match
the averaged weights, so they can optionally be re-estimated on training
images.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load an Ultralytics checkpoint dictionary (model stored under `model`)."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("model") is None:
        raise ValueError(f"not an Ultralytics checkpoint with a model: {path}")
    return checkpoint


def average_checkpoints(paths: Sequence[Path]) -> dict[str, Any]:
    """Return a checkpoint whose model is the uniform average of the given models.

    Floating-point parameters and buffers (including BatchNorm statistics) are
    averaged in fp32; integer buffers such as `num_batches_tracked` are copied
    from the first model. The other checkpoint fields come from the first.
    """

    if not paths:
        raise ValueError("cannot average zero checkpoints")
    checkpoints = [load_checkpoint(path) for path in paths]
    states = [checkpoint["model"].float().state_dict() for checkpoint in checkpoints]
    if any(state.keys() != states[0].keys() for state in states):
        raise ValueError("checkpoints have different architectures and cannot be averaged")
    averaged = {
        key: (sum(state[key] for state in states) / len(states)) if value.is_floating_point() else value.clone()
        for key, value in states[0].items()
    }
    model = checkpoints[0]["model"]
    model.load_state_dict(averaged)
    return {**checkpoints[0], "model": model}


def _letterboxed_batch(images: Sequence[Path], imgsz: int) -> torch.Tensor:
    """Load images as an RGB float batch, letterboxed to a square `imgsz` like Ultralytics."""

    import cv2
    from ultralytics.data.augment import LetterBox

    letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False)
    arrays = []
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"could not read image: {path}")
        arrays.append(letterbox(image=image)[..., ::-1].transpose(2, 0, 1))  # BGR HWC -> RGB CHW
    return torch.from_numpy(np.ascontiguousarray(np.stack(arrays))).float() / 255.0


def recalibrate_batchnorm(
    model: torch.nn.Module, images: Sequence[Path], *, imgsz: int, batch: int, device: Any,
) -> None:
    """Re-estimate BatchNorm running statistics from un-augmented `images`.

    Statistics are reset and accumulated as an exact average over all
    batches (momentum None), with gradients off. Weights are not changed.
    """

    norms = [module for module in model.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    if not norms or not images:
        return
    momenta = [norm.momentum for norm in norms]
    for norm in norms:
        norm.reset_running_stats()
        norm.momentum = None
    model.to(device).train()
    with torch.no_grad():
        for start in range(0, len(images), batch):
            model(_letterboxed_batch(images[start:start + batch], imgsz).to(device))
    for norm, momentum in zip(norms, momenta):
        norm.momentum = momentum
    model.eval().cpu()


def build_soup(
    paths: Sequence[Path], target: Path, *, calibration_images: Sequence[Path], imgsz: int, batch: int,
    device: Any,
) -> Path:
    """Average `paths` into a checkpoint at `target`, recalibrating BatchNorm if images are given."""

    from ultralytics.utils.torch_utils import select_device

    checkpoint = average_checkpoints(paths)
    if calibration_images:
        recalibrate_batchnorm(
            checkpoint["model"], calibration_images, imgsz=imgsz, batch=batch,
            device=select_device(device, verbose=False),
        )
    checkpoint["model"].half()  # Ultralytics stores inference checkpoints in fp16
    partial = target.with_name(target.name + ".part")
    torch.save(checkpoint, partial)
    return partial.replace(target)


def sample_images(images: Sequence[Path], count: int, seed: int) -> list[Path]:
    """Deterministically sample up to `count` images."""

    return random.Random(seed).sample(list(images), min(count, len(images)))


def greedy_selection(
    ranked: Sequence[Path], start_score: float, score_subset: Callable[[list[Path]], float],
    report: Callable[[Path, float, bool], None] = lambda path, score, kept: None,
) -> list[Path]:
    """Greedy soup selection (Wortsman et al., 2022).

    `ranked` lists members best first and `start_score` is the best member's
    own score. Each further member is kept only if the soup of the members
    kept so far plus that member scores strictly higher. `report` is called
    after each trial.
    """

    selected, best = [ranked[0]], start_score
    for candidate in ranked[1:]:
        score = score_subset([*selected, candidate])
        kept = score > best
        if kept:
            selected, best = [*selected, candidate], score
        report(candidate, score, kept)
    return selected
