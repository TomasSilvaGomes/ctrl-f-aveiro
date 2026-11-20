"""Fotografia completa e máscaras das deteções na mesma imagem."""

from __future__ import annotations

import io
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image

from photo_preview import aligned_box, full_catalog_image, jpeg_bytes, normalised_box, prepare_masks


@dataclass(frozen=True)
class PhotoScene:
    image: bytes
    size: tuple[int, int]
    overlay: bytes | None
    objects: tuple


@lru_cache(maxsize=2)
def _classified_index(path, modified_ns):
    index = defaultdict(list)
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("image_id") and item.get("subtype") and item.get("bbox"):
                index[str(item["image_id"])].append(item)
    return index


def classified_objects(catalog_path, image_id):
    root = Path(os.environ.get("DATA_DIR") or catalog_path.parent.parent)
    path = root / "preds" / "owlv2_classified.jsonl"
    if not path.is_file():
        return ()
    return _classified_index(path, path.stat().st_mtime_ns).get(str(image_id), ())


def _compose(image, size, detections):
    combined = Image.new("RGBA", size)
    objects = []
    for label, box, mask in detections:
        if box:
            objects.append((label, box))
        if mask:
            with Image.open(io.BytesIO(mask)) as layer:
                if layer.size == size:
                    combined.alpha_composite(layer.convert("RGBA"))
    output = io.BytesIO()
    overlay = None
    if combined.getchannel("A").getbbox():
        combined.save(output, format="PNG", optimize=True)
        overlay = output.getvalue()
    return PhotoScene(image, size, overlay, tuple(objects))


def build_catalog_scene(catalog_path, image_id, detections, token):
    photo, remote = full_catalog_image(catalog_path, {"imagem_id": image_id}, token)
    original = photo.size
    photo.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    boxes = []
    for label, box in detections:
        aligned = aligned_box(catalog_path, {"imagem_id": image_id, "imagem_bbox": box}, original, remote=remote)
        if aligned:
            relative = normalised_box(aligned, original)
            if relative[0] < relative[2] and relative[1] < relative[3]:
                boxes.append((label, relative))
    return _segment_scene(photo, jpeg_bytes(photo), boxes)


def _segment_objects(photo, boxes):
    for start in range(0, len(boxes), 6):
        batch = boxes[start:start + 6]
        pixels = [tuple(value * size for value, size in zip(box, photo.size * 2)) for _, box in batch]
        yield from (
            (label, box, mask) for (label, box), mask in zip(batch, prepare_masks(photo, pixels))
        )


def _segment_scene(photo, encoded, boxes):
    return _compose(encoded, photo.size, _segment_objects(photo, boxes))


def build_contribution_scene(previews):
    if not previews:
        raise ValueError("Só existe o recorte deste elemento.")
    encoded = previews[0][1].image
    with Image.open(io.BytesIO(encoded)) as source:
        photo = source.convert("RGB")
    masks = [(label, preview.box, preview.mask) for label, preview in previews if preview.mask]
    missing = [(label, preview.box) for label, preview in previews if preview.box and not preview.mask]
    masks.extend(_segment_objects(photo, missing))
    return _compose(encoded, photo.size, masks)
