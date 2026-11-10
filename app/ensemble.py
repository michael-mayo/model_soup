"""Output-space ensembles: fuse several models' detections image by image.

Ensembles are the reference that weight-space soups are compared against:
they usually score well but cost one forward pass per member.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np
import torch

from app.evaluation import MAX_DETECTIONS, Detections, Predictions


def _nms_image(parts: Sequence[Detections], iou_threshold: float) -> Detections:
    """Pool all members' detections and keep the class-wise NMS survivors."""

    import torchvision

    boxes = np.concatenate([part.boxes for part in parts])
    if not len(boxes):
        return Detections.empty()
    scores = np.concatenate([part.scores for part in parts])
    classes = np.concatenate([part.classes for part in parts])
    keep = torchvision.ops.batched_nms(
        torch.from_numpy(boxes), torch.from_numpy(scores), torch.from_numpy(classes), iou_threshold,
    ).numpy()
    return Detections(boxes[keep], scores[keep], classes[keep]).top(MAX_DETECTIONS)


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU between one xyxy box and an (N, 4) array of xyxy boxes."""

    width = np.clip(np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]), 0, None)
    height = np.clip(np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]), 0, None)
    intersection = width * height
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(area + areas - intersection, 1e-9)


def _wbf_image(parts: Sequence[Detections], iou_threshold: float) -> Detections:
    """Weighted boxes fusion (Solovyev et al., 2021) for one image, all members weighted equally.

    Per class, boxes are visited in descending confidence. Each joins the
    first cluster whose current fused box overlaps it by more than
    `iou_threshold`, otherwise it starts a new cluster. A fused box is the
    confidence-weighted mean of its cluster; its score is the mean cluster
    confidence scaled by min(cluster size, members) / members, so boxes
    found by few members are down-weighted.
    """

    members = len(parts)
    boxes = np.concatenate([part.boxes for part in parts])
    if not len(boxes):
        return Detections.empty()
    scores = np.concatenate([part.scores for part in parts])
    classes = np.concatenate([part.classes for part in parts])

    fused_boxes, fused_scores, fused_classes = [], [], []
    for class_id in np.unique(classes):
        index = np.flatnonzero(classes == class_id)
        index = index[np.argsort(-scores[index], kind="stable")]
        weighted_sum = np.zeros((len(index), 4))  # sum of score * box per cluster
        score_sum = np.zeros(len(index))
        count = np.zeros(len(index), dtype=np.int64)
        fused = np.zeros((len(index), 4))  # current fused box per cluster
        clusters = 0
        for i in index:
            box, score = boxes[i].astype(np.float64), float(scores[i])
            target = -1
            if clusters:
                overlaps = _iou_one_to_many(box, fused[:clusters])
                best = int(overlaps.argmax())
                if overlaps[best] > iou_threshold:
                    target = best
            if target < 0:
                target, clusters = clusters, clusters + 1
            weighted_sum[target] += score * box
            score_sum[target] += score
            count[target] += 1
            fused[target] = weighted_sum[target] / score_sum[target]
        mean_score = score_sum[:clusters] / count[:clusters]
        fused_boxes.append(fused[:clusters])
        fused_scores.append(mean_score * np.minimum(count[:clusters], members) / members)
        fused_classes.append(np.full(clusters, class_id, dtype=np.int64))
    return Detections(
        np.concatenate(fused_boxes).astype(np.float32), np.concatenate(fused_scores).astype(np.float32),
        np.concatenate(fused_classes),
    ).top(MAX_DETECTIONS)


def _fuse(
    members: Sequence[Predictions], fuse_image: Callable[[Sequence[Detections]], Detections],
) -> Predictions:
    """Fuse members image by image; time is every member's prediction time plus fusion."""

    started = time.perf_counter()
    detections = [fuse_image(parts) for parts in zip(*(member.detections for member in members))]
    seconds = sum(member.seconds for member in members) + time.perf_counter() - started
    return Predictions(detections, seconds)


def ensemble_nms(members: Sequence[Predictions], iou_threshold: float) -> Predictions:
    """Ensemble by pooling every member's detections and applying class-wise NMS."""

    return _fuse(members, lambda parts: _nms_image(parts, iou_threshold))


def ensemble_wbf(members: Sequence[Predictions], iou_threshold: float) -> Predictions:
    """Ensemble by weighted boxes fusion of every member's detections."""

    return _fuse(members, lambda parts: _wbf_image(parts, iou_threshold))
