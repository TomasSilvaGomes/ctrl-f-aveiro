"""Deteção e anonimização das fotografias enviadas através da aplicação."""

from __future__ import annotations

import io
import math
import sys
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from PIL import Image, ImageDraw

from contributions import PrivacyModelsMissing
from photo_preview import SAM2_MODEL, PhotoPreview, jpeg_bytes, normalised_box, prepare_masks

ROOT = Path(__file__).resolve().parents[1]   # raiz do projeto
# A análise das fotos enviadas usa os MESMOS módulos da pipeline (scripts/), para dar o mesmo
# resultado que o catálogo: OWLv2 (detetar.py) -> SigLIP (classificar.py) -> limites (sinais.py).
sys.path.insert(0, str(ROOT / "scripts"))
from classificar import Classifier, square_crop, suppress_contained  # noqa: E402
from comum import Desfocar, load_taxonomy, nms_by_category  # noqa: E402
from detetar import PROMPTS, Owlv2, prepare_views, to_original  # noqa: E402
from sinais import SpeedReader  # noqa: E402


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
        """Caras e matrículas: EgoBlur Gen2 (comum.Desfocar, o mesmo da pipeline, Apache 2.0).
        Sem os pesos a análise pára: nunca se guarda uma foto sem anonimizar."""
        try:
            desfocar = Desfocar(self.device)
        except FileNotFoundError as error:
            raise PrivacyModelsMissing(
                "A anonimização está indisponível: faltam os pesos EgoBlur Gen2 em weights/ "
                "(ego_blur_face_gen2.jit e ego_blur_lp_gen2.jit)."
            ) from error
        return lambda rgb: desfocar.caixas(np.ascontiguousarray(rgb[:, :, ::-1]))

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
