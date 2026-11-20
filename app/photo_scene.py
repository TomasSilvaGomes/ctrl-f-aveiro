"""Fotografia completa e máscaras das deteções na mesma imagem."""

from __future__ import annotations

import io
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from photo_preview import (
    SegmentationError, _overlay, aligned_box, full_catalog_image, jpeg_bytes, normalised_box, prepare_masks,
)


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


@lru_cache(maxsize=2)
def _contour_index(root, stamps):
    """'imagem|x0,y0,x1,y1' -> polígonos na foto original, dos contornos já calculados pela pipeline
    (preds/contorno_cache*.jsonl; primeiro o SAM 2.1 small, como o imagens.py)."""
    order = {"contorno_cache_small.jsonl": 0, "contorno_cache.jsonl": 1}
    index = {}
    for name, _ in sorted(stamps, key=lambda item: (order.get(item[0], 2), item[0])):
        with (Path(root) / "preds" / name).open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    item = json.loads(line)
                    if item.get("poly"):
                        index.setdefault(item["k"], item["poly"])
    return index


def precomputed_contours(catalog_path):
    root = Path(os.environ.get("DATA_DIR") or catalog_path.parent.parent)
    files = sorted((root / "preds").glob("contorno_cache*.jsonl"))
    return _contour_index(str(root), tuple((path.name, path.stat().st_mtime_ns) for path in files))


def _contour_mask(polygons, original, size):
    """Polígonos na foto original -> camada da app no tamanho mostrado. Nas 360° desenha também
    deslocado de uma volta (contornos que passam a costura)."""
    sx, sy = size[0] / original[0], size[1] / original[1]
    silhouette = Image.new("L", size, 0)
    draw = ImageDraw.Draw(silhouette)
    for polygon in polygons:
        for shift in (-original[0], 0, original[0]):
            draw.polygon([((u + shift) * sx, v * sy) for u, v in polygon], fill=1)
    return _overlay(np.asarray(silhouette, dtype=bool))


def _compose(image, size, detections):
    combined = Image.new("RGBA", size)
    objects = []
    for label, box, mask in detections:
        if box:
            objects.append((label, box, bool(mask)))   # sem máscara: a app desenha a caixa
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


def build_catalog_scene(catalog_path, image_id, detections, token, anonymize=None):
    """Máscara de cada objeto: o contorno já calculado pela pipeline (igual em qualquer PC, sem modelos);
    os que não o têm vão ao SAM 2 se houver; senão fica a caixa.
    anonymize: PIL -> PIL com caras e matrículas desfocadas (photo_analysis.anonymize)."""
    photo, remote = full_catalog_image(catalog_path, {"imagem_id": image_id}, token)
    original = photo.size
    photo.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    if anonymize is not None:
        photo = anonymize(photo)
    contours = {} if remote else precomputed_contours(catalog_path)   # contornos na resolução da foto local
    ready, missing = [], []
    for label, box in detections:
        aligned = aligned_box(catalog_path, {"imagem_id": image_id, "imagem_bbox": box}, original, remote=remote)
        if aligned:
            relative = normalised_box(aligned, original)
            if relative[0] < relative[2] and relative[1] < relative[3]:
                polygons = contours.get(f"{image_id}|" + ",".join(str(int(round(v))) for v in box))
                if polygons:
                    ready.append((label, relative, _contour_mask(polygons, original, photo.size)))
                else:
                    missing.append((label, relative))
    return _compose(jpeg_bytes(photo), photo.size, [*ready, *_segment_objects(photo, missing)])


def _segment_objects(photo, boxes):
    """Máscara SAM 2 de cada caixa; sem SAM 2 (ou se falhar) ficam só as caixas com o nome."""
    for start in range(0, len(boxes), 6):
        batch = boxes[start:start + 6]
        pixels = [tuple(value * size for value, size in zip(box, photo.size * 2)) for _, box in batch]
        try:
            masks = prepare_masks(photo, pixels)
        except SegmentationError:
            masks = [None] * len(batch)
        yield from ((label, box, mask) for (label, box), mask in zip(batch, masks))


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
