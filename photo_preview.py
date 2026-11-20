"""Prepara e guarda as fotografias e máscaras da ficha do mapa."""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import tempfile
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from threading import Lock

from PIL import Image, ImageChops, ImageFilter

from mapillary_photos import MapillaryPhotoError, fetch_photo, image_key

ROOT = Path(__file__).resolve().parent
SAM2_MODEL = "facebook/sam2.1-hiera-tiny"
LOGGER = logging.getLogger(__name__)
MODEL_LOAD_LOCK = Lock()


@dataclass(frozen=True)
class PhotoPreview:
    image: bytes
    mask: bytes | None = None
    box: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class PreviewResult:
    preview: PhotoPreview | None
    notice: str | None = None


class SegmentationError(RuntimeError):
    pass


def jpeg_bytes(image):
    output = io.BytesIO()
    image.convert("RGB").save(output, format="JPEG", quality=87, optimize=True)
    return output.getvalue()


def _overlay(mask):
    silhouette = Image.fromarray(mask.astype("uint8") * 255)
    outline = ImageChops.subtract(silhouette.filter(ImageFilter.MaxFilter(3)), silhouette.filter(ImageFilter.MinFilter(3)))
    fill = silhouette.point(lambda value: value * 0.27)
    layer = Image.new("RGBA", silhouette.size, (17, 119, 255))
    layer.putalpha(fill)
    border = Image.new("RGBA", silhouette.size, (92, 218, 255))
    border.putalpha(outline)
    output = io.BytesIO()
    Image.alpha_composite(layer, border).save(output, format="PNG", optimize=True)
    return output.getvalue()


class Segmenter:
    def __init__(self, directory):
        import torch
        from transformers import Sam2Model, Sam2Processor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Sam2Processor.from_pretrained(directory, local_files_only=True)
        self.model = Sam2Model.from_pretrained(directory, local_files_only=True).to(self.device).eval()
        self.lock = Lock()

    def masks(self, image, boxes):
        with self.lock, self.torch.inference_mode():
            inputs = self.processor(images=image, input_boxes=[[list(box) for box in boxes]], return_tensors="pt")
            sizes = inputs.pop("original_sizes")
            result = self.model(**inputs.to(self.device), multimask_output=False)
            masks = self.processor.post_process_masks(result.pred_masks.cpu(), sizes)[0]
            return [_overlay(mask[0].numpy()) if mask.any() else None for mask in masks]


@lru_cache(maxsize=1)
def segmenter(directory):
    return Segmenter(directory)


def segmentation_directory():
    return Path(os.environ.get("CTRL_F_SAM2_DIR") or ROOT / "weights" / "sam2")


def prepare_masks(image, boxes):
    directory = segmentation_directory()
    if not all((directory / name).is_file() for name in ("config.json", "preprocessor_config.json", "model.safetensors")):
        raise SegmentationError("A segmentação está indisponível. Faltam os pesos do SAM 2 no servidor.")
    try:
        with MODEL_LOAD_LOCK:
            model = segmenter(directory)
        return model.masks(image, boxes)
    except (OSError, RuntimeError, ImportError, ValueError) as error:
        raise SegmentationError("O SAM 2 não conseguiu gerar a máscara nesta tentativa.") from error


def normalised_box(box, size):
    width, height = size
    return tuple(max(0.0, min(1.0, float(value) / scale)) for value, scale in zip(box, (width, height, width, height)))


def preview_path(catalog_path, props, *, legacy=False):
    identity = [props.get(key) for key in ("id", "imagem_id", "imagem_bbox")]
    key = sha256(json.dumps(identity).encode("utf-8")).hexdigest()[:24]
    directory = catalog_path.parent / "previews"
    if not legacy:
        directory /= "sam2"
    return directory / f"{key}.jpg"


def catalog_photo_views(props):
    yield props
    for view in props.get("vistas_foto") or ():
        yield {
            "id": props.get("id"), "imagem_id": view["imagem_id"],
            "imagem_bbox": view["imagem_bbox"], "imagem_url": view.get("imagem_url"),
            "autor": view.get("autor"), "licenca": view.get("licenca"),
        }


@lru_cache(maxsize=4)
def _reference_index(directory):
    metadata = directory / "fontes.csv"
    if not metadata.is_file():
        return {}
    with metadata.open(encoding="utf-8-sig") as file:
        index = {}
        for row in csv.DictReader(file):
            path = (directory / row["ficheiro"]).resolve()
            if path.is_relative_to(directory):
                index[row["id_catalogo"], row["imagem_id"]] = path
        return index


def _data_directory(catalog_path):
    return Path(os.environ.get("DATA_DIR") or catalog_path.parent.parent).resolve()


@lru_cache(maxsize=4)
def _sample_sizes(directory):
    path = directory / "meta" / "sample_images.csv"
    if not path.is_file():
        return {}
    sizes = {}
    with path.open(encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            try:
                width, height = int(row["thumb_w"]), int(row["thumb_h"])
            except (KeyError, TypeError, ValueError):
                continue
            if width > 0 and height > 0:
                sizes[row["id"]] = (width, height)
    return sizes


def aligned_box(catalog_path, props, size, *, remote=False):
    box = props.get("imagem_bbox")
    if not box or len(box) != 4:
        return None
    original = _sample_sizes(_data_directory(catalog_path)).get(str(props.get("imagem_id")))
    if original is None:
        return None if remote else box
    width, height = size
    if not math.isclose(original[0] / original[1], width / height, rel_tol=0.025):
        return None
    return tuple(value * dimension for value, dimension in zip(normalised_box(box, original), size * 2))


def _local_image(catalog_path, props):
    image_id = str(props.get("imagem_id") or "")
    if not image_id.isdecimal():
        return None, None, False
    root = _data_directory(catalog_path)
    image_path = root / "images" / f"{image_id}.jpg"
    box = props.get("imagem_bbox")
    if image_path.is_file():
        return image_path, box, False
    candidates = (root / "referencias" / "aveiro", ROOT / "data" / "referencias" / "aveiro")
    for directory in dict.fromkeys(candidates):
        reference = _reference_index(directory.resolve()).get((props.get("id"), image_id))
        if reference and reference.is_file():
            return reference, box, True
    return None, None, False


def local_preview_image(catalog_path, props):
    path, box, reference = _local_image(catalog_path, props)
    if path is None:
        return None, None
    with Image.open(path) as file:
        image = file.convert("RGB")
    if not reference:
        box = aligned_box(catalog_path, props, image.size)
    return _frame_image(image, box, reference)


def full_catalog_image(catalog_path, props, token=""):
    path, _, reference = _local_image(catalog_path, props)
    if path and not reference:
        with Image.open(path) as source:
            return source.convert("RGB"), False
    key = image_key(props.get("imagem_id"))
    if not key or not token:
        raise ValueError("A fotografia completa ainda não está disponível.")
    photo = fetch_photo(key, token)
    with Image.open(io.BytesIO(photo.data), formats=("JPEG", "PNG", "WEBP")) as source:
        if source.width * source.height > 16_000_000:
            raise ValueError("A fotografia completa é demasiado grande.")
        return source.convert("RGB"), True


def _frame_image(image, box, reference=False):
    if box and len(box) == 4:
        x0, y0, x1, y1 = box
        extent = max(x1 - x0, y1 - y0)
        if reference:
            margin = 0.15 * extent
            left, top = int(x0 - margin), int(y0 - margin)
            expected = (int(x1 + margin) - left, int(y1 + margin) - top)
            if min(expected) > 0:
                relative = normalised_box((x0 - left, y0 - top, x1 - left, y1 - top), expected)
            else:
                relative = None
        else:
            margin = max(65, 1.55 * extent)
            left, top = max(0, math.floor(x0 - margin)), max(0, math.floor(y0 - margin))
            right, bottom = min(image.width, math.ceil(x1 + margin)), min(image.height, math.ceil(y1 + margin))
            if right > left and bottom > top:
                image = image.crop((left, top, right, bottom))
                relative = normalised_box((x0 - left, y0 - top, x1 - left, y1 - top), image.size)
            else:
                relative = None
    else:
        relative = None
    image.thumbnail((960, 960), Image.Resampling.LANCZOS)
    if relative and relative[0] < relative[2] and relative[1] < relative[3]:
        return image, tuple(value * size for value, size in zip(relative, image.size * 2))
    return image, None


def load_catalog_preview(catalog_path, props, *, prepared_only=False):
    prepared = _read_prepared(preview_path(catalog_path, props))
    if prepared or prepared_only:
        return prepared
    image, box = local_preview_image(catalog_path, props)
    if image is not None:
        return PhotoPreview(jpeg_bytes(image), box=normalised_box(box, image.size) if box else None)
    return _read_prepared(preview_path(catalog_path, props, legacy=True), load_mask=False)


def has_catalog_preview(catalog_path, props):
    path = preview_path(catalog_path, props)
    return path.is_file() or _local_image(catalog_path, props)[0] is not None or (
        preview_path(catalog_path, props, legacy=True).is_file()
    )


def _read_prepared(path, *, load_mask=True):
    mask = path.with_suffix(".png")
    metadata = path.with_suffix(".json")
    if not path.is_file():
        return None
    box = json.loads(metadata.read_text(encoding="utf-8")).get("box") if metadata.is_file() else None
    return PhotoPreview(path.read_bytes(), mask.read_bytes() if load_mask and mask.is_file() else None, tuple(box) if box else None)


def write_atomic(path, data):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(data)
            file.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def store_preview(catalog_path, props, preview):
    path = preview_path(catalog_path, props)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path.with_suffix(".json"), json.dumps({"box": preview.box}).encode("utf-8"))
    if preview.mask:
        write_atomic(path.with_suffix(".png"), preview.mask)
    write_atomic(path, preview.image)


def _prepare_selected_preview(catalog_path, props, preview, token):
    try:
        if preview is None:
            photo = fetch_photo(image_key(props.get("imagem_id")), token)
            with Image.open(io.BytesIO(photo.data), formats=("JPEG", "PNG", "WEBP")) as source:
                if source.width * source.height > 16_000_000:
                    raise MapillaryPhotoError("A fotografia devolvida pela API é demasiado grande.")
                box = aligned_box(catalog_path, props, source.size, remote=True)
                image, box = _frame_image(source.convert("RGB"), box)
            relative = normalised_box(box, image.size) if box else None
            preview = PhotoPreview(jpeg_bytes(image), box=relative)
            if relative is None:
                store_preview(catalog_path, props, preview)
                return PreviewResult(preview)

        with Image.open(io.BytesIO(preview.image)) as image:
            box = [value * size for value, size in zip(preview.box, image.size * 2)]
            (mask,) = prepare_masks(image.convert("RGB"), [box])
        if mask is None:
            return PreviewResult(preview, "O SAM 2 não encontrou uma máscara para esta caixa de deteção.")
        preview = PhotoPreview(preview.image, mask, preview.box)
        store_preview(catalog_path, props, preview)
    except MapillaryPhotoError as error:
        return PreviewResult(preview, str(error))
    except SegmentationError as error:
        LOGGER.exception("Não foi possível calcular a máscara")
        return PreviewResult(preview, str(error))
    except (OSError, ValueError, Image.DecompressionBombError):
        LOGGER.exception("Não foi possível preparar ou guardar a fotografia")
        return PreviewResult(preview, "Não foi possível preparar ou guardar a máscara.")
    return PreviewResult(preview)


class PreviewQueue:
    """Partilha a preparação de uma fotografia entre cliques e sessões."""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="fotografias")
        self.jobs = OrderedDict()
        self.lock = Lock()

    def submit(self, catalog_path, props, preview, *, token="", retry=False):
        key = preview_path(catalog_path, props)
        with self.lock:
            job = self.jobs.get(key)
            if job:
                if not job.done() or (not retry and (preview is None or job.result().preview is not None)):
                    self.jobs.move_to_end(key)
                    return job
            if len(self.jobs) >= 32 and key not in self.jobs:
                finished = next((candidate for candidate, pending in self.jobs.items() if pending.done()), None)
                if finished is None:
                    return None
                self.jobs.pop(finished)
            job = self.executor.submit(_prepare_selected_preview, catalog_path, props, preview, token)
            self.jobs[key] = job
            return job
