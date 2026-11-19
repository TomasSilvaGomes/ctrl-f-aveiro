"""Deteção e anonimização das fotografias enviadas através da aplicação."""

from __future__ import annotations

import io
import math
import os
from functools import partial
from pathlib import Path
from threading import Lock

import numpy as np
import torch
import torchvision  # regista as operações usadas pelos modelos TorchScript
from PIL import Image, ImageDraw

from contributions import PrivacyModelsMissing
from photo_preview import SAM2_MODEL, PhotoPreview, jpeg_bytes, normalised_box, prepare_masks
from photo_privacy import LocalPrivacyDetector
from scripts.classify_subtype import Classifier, load_taxonomy, square_crop, suppress_contained
from scripts.evaluate import nms_by_category
from scripts.ler_limites import SpeedReader
from scripts.run_detector import PROMPTS, Owlv2, prepare_views, to_original

ROOT = Path(__file__).resolve().parent
PRIVACY_WEIGHTS = ("ego_blur_face", "ego_blur_lp")


def egoblur_regions(models, device, rgb):
    bgr = np.ascontiguousarray(rgb[:, :, ::-1].transpose(2, 0, 1))
    tensor = torch.from_numpy(bgr).to(device)
    regions = []
    for model in models:
        boxes, _, scores, _ = model(tensor)
        regions.extend(boxes[scores >= 0.30].tolist())
    return regions


class PhotoAnalyzer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.taxonomy = load_taxonomy()
        self.classifications = {
            (category, item["id"]): item
            for category, spec in self.taxonomy["categorias"].items()
            for item in spec["subtipos"]
        }
        self.detector = Owlv2(self.device)
        self.classifier = None
        self.speed_reader = None
        self.privacy_regions = None
        self.lock = Lock()

    def _load_privacy_models(self):
        directory = Path(os.environ.get("CTRL_F_EGOBLUR_DIR") or ROOT / "weights" / "egoblur")
        paths = []
        for name in PRIVACY_WEIGHTS:
            path = directory / f"{name}_gen1.jit"
            paths.append(path if path.is_file() else directory / f"{name}.jit")
        if all(path.is_file() for path in paths):
            models = [torch.jit.load(str(path), map_location=self.device).eval() for path in paths]
            return partial(egoblur_regions, models, self.device)
        try:
            return LocalPrivacyDetector().regions
        except Exception as error:
            raise PrivacyModelsMissing(
                "Não foi possível preparar a anonimização. Confirma o runtime ONNX e os pesos em weights/privacy."
            ) from error

    @torch.inference_mode()
    def _safe_jpeg(self, crop, max_size=640):
        if self.privacy_regions is None:
            self.privacy_regions = self._load_privacy_models()
        crop.thumbnail((960, 960), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(crop)
        for left, top, right, bottom in self.privacy_regions(np.asarray(crop)):
            pad_x = (right - left) * 0.20
            pad_y = (bottom - top) * 0.20
            bounds = (
                math.floor(left - pad_x), math.floor(top - pad_y),
                math.ceil(right + pad_x), math.ceil(bottom + pad_y),
            )
            draw.rectangle(bounds, fill="#202020")
        crop.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return jpeg_bytes(crop)

    @staticmethod
    def _published_jpeg(image, max_size=640):
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return jpeg_bytes(image)

    def analyze(self, image, *, already_anonymized=False, panoramic=False):
        with self.lock:
            save_photo = self._published_jpeg if already_anonymized else self._safe_jpeg
            return self._analyze(image, save_photo, panoramic)

    def _analyze(self, image, save_photo, panoramic):
        detections = []
        for pixels, view in prepare_views(image, panoramic=panoramic):
            (raw_detections,) = self.detector([pixels], threshold=0.20)
            for box, score, prompt in raw_detections:
                bounds = to_original(box, view)
                if bounds is None or max(bounds[2] - bounds[0], bounds[3] - bounds[1]) < 16:
                    continue
                category, _ = PROMPTS[prompt]
                detections.append({"bbox": bounds, "score": score, "category": category})
        detections = suppress_contained(nms_by_category(detections))[:64]
        if not detections:
            return []
        if self.classifier is None:
            self.classifier = Classifier(self.taxonomy, self.device)

        elements = []
        for start in range(0, len(detections), 8):
            batch = detections[start:start + 8]
            crops = [square_crop(image, detection["bbox"]) for detection in batch]
            categories = [detection["category"] for detection in batch]
            results = self.classifier([square for square, _ in crops], categories)
            for detection, (square, crop), (subtype, confidence, _) in zip(batch, crops, results):
                if subtype is None:
                    continue
                category = detection["category"]
                spec = self.classifications[category, subtype]
                label, value = spec["label"], None
                if subtype == "limite_velocidade":
                    if self.speed_reader is None:
                        self.speed_reader = SpeedReader(self.device, self.classifier)
                    speed, speed_confidence = self.speed_reader(np.asarray(square))
                    if speed_confidence >= 0.60:
                        value = speed
                        label = label.replace("{n}", str(speed))
                    else:
                        label = label.replace(" {n} km/h", "")
                properties = {
                    "categoria": category,
                    "categoria_label": self.taxonomy["categorias"][category]["label"],
                    "subtipo": subtype,
                    "label": label,
                    "codigo": spec.get("codigo"),
                    "valor": value,
                    "confianca": round(min(1.0, detection["score"] / 0.5) * confidence, 4),
                }
                elements.append((properties, save_photo(crop), detection["bbox"]))
        if not elements:
            return []
        photo = save_photo(image.copy(), max_size=960)
        boxes = [normalised_box(box, image.size) for _, _, box in elements]
        with Image.open(io.BytesIO(photo)) as preview:
            masks = prepare_masks(preview, [
                tuple(value * scale for value, scale in zip(box, preview.size * 2))
                for box in boxes
            ])
        return [
            ({**properties, "segmentacao_modelo": SAM2_MODEL}, crop, PhotoPreview(photo, mask, box))
            for (properties, crop, _), mask, box in zip(elements, masks, boxes)
        ]
