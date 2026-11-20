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
from photo_preview import SAM2_MODEL, PhotoPreview, SegmentationError, jpeg_bytes, normalised_box, prepare_masks

ROOT = Path(__file__).resolve().parents[1]   # raiz do projeto
# A análise das fotos enviadas usa os MESMOS módulos da pipeline (scripts/), para dar o mesmo
# resultado que o catálogo: OWLv2 (detetar.py) -> SigLIP (classificar.py) -> limites (sinais.py).
sys.path.insert(0, str(ROOT / "scripts"))
from classificar import Classifier, square_crop, suppress_contained  # noqa: E402
from comum import Desfocar, load_taxonomy, nms_by_category  # noqa: E402
from detetar import PROMPTS, Owlv2, prepare_views, to_original  # noqa: E402
from sinais import SpeedReader  # noqa: E402

_PRIVACY = None
_PRIVACY_LOCK = Lock()


def privacy_model():
    """Caras e matrículas: EgoBlur Gen2 (comum.Desfocar, o mesmo da pipeline, Apache 2.0), carregado
    uma vez por processo. Sem os pesos pára tudo: nunca se guarda nem mostra uma foto sem anonimizar."""
    global _PRIVACY
    if _PRIVACY is None:
        try:
            _PRIVACY = Desfocar("cuda" if torch.cuda.is_available() else "cpu")
        except FileNotFoundError as error:
            raise PrivacyModelsMissing(
                "A anonimização está indisponível: faltam os pesos EgoBlur Gen2 em weights/ "
                "(ego_blur_face_gen2.jit e ego_blur_lp_gen2.jit)."
            ) from error
    return _PRIVACY


def privacy_boxes(rgb):
    """Imagem RGB (numpy) -> caixas [x0, y0, x1, y1] de caras e matrículas."""
    with _PRIVACY_LOCK:
        return privacy_model().caixas(np.ascontiguousarray(rgb[:, :, ::-1]))


def anonymize(image):
    """PIL -> cópia RGB com caras e matrículas desfocadas na foto inteira (elipse desfocada, como a
    pipeline faz às fotos próprias). Também nas do Mapillary: o desfoque deles às vezes falha."""
    bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
    with _PRIVACY_LOCK:
        blurred, _ = privacy_model()(bgr)
    return Image.fromarray(np.ascontiguousarray(blurred[:, :, ::-1]))


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
        self.lock = Lock()

    @torch.inference_mode()
    def _safe_jpeg(self, crop, max_size=640):
        """2.ª passagem, em cada recorte e na foto guardada: o recorte vê as caras maiores do que a foto
        inteira reduzida a 1200 px, por isso apanha as que a 1.ª passagem (anonymize) deixou."""
        crop.thumbnail((960, 960), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(crop)
        for left, top, right, bottom in privacy_boxes(np.asarray(crop)):
            pad_x = (right - left) * 0.20
            pad_y = (bottom - top) * 0.20
            bounds = (
                math.floor(left - pad_x), math.floor(top - pad_y),
                math.ceil(right + pad_x), math.ceil(bottom + pad_y),
            )
            draw.rectangle(bounds, fill="#202020")
        crop.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return jpeg_bytes(crop)

    def warm_up(self):
        """Carrega já o que a 1.ª análise carregaria a meio (EgoBlur, SigLIP, SAM 2), para a app poder
        mostrar "a carregar os modelos" em vez de uma barra parada. Sem SAM 2 continua (só caixas)."""
        privacy_model()
        with self.lock:
            if self.classifier is None:
                self.classifier = Classifier(self.taxonomy, self.device)
        try:
            prepare_masks(Image.new("RGB", (64, 64)), [(8, 8, 56, 56)])
        except SegmentationError:
            pass

    def analyze(self, image, *, panoramic=False, on_step=None):
        """Desfoca a foto inteira ANTES da deteção (como a pipeline) e cada imagem guardada outra vez.
        on_step(fração 0-1, texto): progresso para a app."""
        step = on_step or (lambda fraction, text: None)
        with self.lock:
            step(0.05, "a desfocar caras e matrículas")
            return self._analyze(anonymize(image), self._safe_jpeg, panoramic, step)

    def preview_without_detections(self, image):
        with self.lock:
            return PhotoPreview(self._safe_jpeg(anonymize(image), max_size=960))

    def _analyze(self, image, save_photo, panoramic, step):
        step(0.25, "a procurar mobiliário urbano")
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

        step(0.5, f"a classificar {len(detections)} deteções")
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
        step(0.8, "a contornar os objetos (SAM 2)")
        model = SAM2_MODEL
        with Image.open(io.BytesIO(photo)) as preview:
            try:
                masks = prepare_masks(preview, [
                    tuple(value * scale for value, scale in zip(box, preview.size * 2))
                    for box in boxes
                ])
            except SegmentationError:
                # sem SAM 2 a foto segue na mesma, só com a caixa de cada deteção (a causa fica no registo)
                masks, model = [None] * len(boxes), None
        return [
            ({**properties, "segmentacao_modelo": model}, crop, PhotoPreview(photo, mask, box))
            for (properties, crop, _), mask, box in zip(elements, masks, boxes)
        ]
