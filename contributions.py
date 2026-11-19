"""Entrada de fotografias e armazenamento dos recortes publicados."""

from __future__ import annotations

import io
import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from PIL import ExifTags, Image, ImageOps, UnidentifiedImageError

from photo_preview import PhotoPreview

SOURCE = "Contribuição"
PHOTO_CATEGORY = "fotografias"
UPLOAD_TYPES = ["jpg", "jpeg", "png", "webp", "heic", "heif"]
MAX_UPLOAD_MB = 25
MAX_PIXELS = 50_000_000
FORMATS = ("JPEG", "PNG", "WEBP", "HEIF")


class PrivacyModelsMissing(RuntimeError):
    pass


def open_photo(data: bytes) -> Image.Image:
    if not data or len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"Escolha uma fotografia com até {MAX_UPLOAD_MB} MB.")
    if data[4:8] == b"ftyp":
        try:
            from pillow_heif import register_heif_opener
        except ImportError as error:
            raise ValueError("O suporte HEIC está indisponível. Envie a foto em JPEG.") from error
        register_heif_opener()
    try:
        image = Image.open(io.BytesIO(data), formats=FORMATS)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError, SyntaxError) as error:
        raise ValueError("O ficheiro não é uma fotografia válida nos formatos aceites.") from error
    if image.width * image.height > MAX_PIXELS or min(image.size) < 16:
        image.close()
        raise ValueError("A imagem deve ter pelo menos 16 píxeis por lado e até 50 megapíxeis.")
    return image


def _decimal_degrees(values, ref, positive, negative, limit):
    degrees, minutes, seconds = map(float, values)
    ref = ref.decode("ascii") if isinstance(ref, bytes) else ref
    ref = ref.strip("\x00 ").upper()
    if ref not in (positive, negative) or not (
        0 <= degrees <= limit and 0 <= minutes < 60 and 0 <= seconds < 60
    ):
        raise ValueError
    value = degrees + minutes / 60 + seconds / 3600
    if value > limit:
        raise ValueError
    return -value if ref == negative else value


def photo_coordinates(data: bytes) -> tuple[float, float] | None:
    """Devolve (latitude, longitude) a partir do GPS EXIF, quando válido."""
    with open_photo(data) as image:
        try:
            gps = image.getexif().get_ifd(ExifTags.IFD.GPSInfo)
            lat = _decimal_degrees(gps[2], gps[1], "N", "S", 90)
            lon = _decimal_degrees(gps[4], gps[3], "E", "W", 180)
        except (ArithmeticError, AttributeError, KeyError, OSError, TypeError, ValueError, SyntaxError):
            return None
    return lat, lon


def decode_photo(data: bytes) -> Image.Image:
    with open_photo(data) as image:
        try:
            image = ImageOps.exif_transpose(image).convert("RGB")
        except (OSError, ValueError, SyntaxError) as error:
            raise ValueError("Não foi possível ler os píxeis desta fotografia.") from error
        image.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
        return image


class ContributionStore:
    def __init__(self, path: Path):
        self.path = path

    def _read(self, sql, parameters=(), table="elements"):
        if not self.path.exists():
            return []
        with closing(sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True)) as db:
            if not db.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (table,)).fetchone():
                return []
            return db.execute(sql, parameters).fetchall()

    def features(self):
        return [json.loads(row[0]) for row in self._read("SELECT feature FROM elements ORDER BY rowid")]

    def crop(self, element_id):
        rows = self._read("SELECT crop FROM elements WHERE id = ?", (element_id,))
        return rows[0][0] if rows else None

    def preview(self, element_id, box=None):
        rows = self._read(
            "SELECT image, mask FROM preview_masks JOIN preview_photos ON preview_photos.id = preview_masks.photo_id "
            "WHERE preview_masks.id = ?",
            (element_id,), table="preview_masks",
        )
        return PhotoPreview(*rows[0], box=box) if rows else None

    def has_photo(self, digest):
        return bool(self._read("SELECT 1 FROM photos WHERE digest = ?", (digest,)))

    def publish(self, digest, latitude, longitude, location_source, detections):
        if not (math.isfinite(latitude) and -90 <= latitude <= 90
                and math.isfinite(longitude) and -180 <= longitude <= 180):
            raise ValueError("Indique coordenadas válidas para a fotografia.")
        upload_id = f"FOTO-{uuid4().hex[:12]}"
        if not detections:
            detections = [({
                "categoria": PHOTO_CATEGORY,
                "categoria_label": "Fotografias",
                "subtipo": "sem_deteccoes",
                "label": "Fotografia sem elementos detetados",
                "confianca": None,
            }, None)]
        rows, masks = [], []
        preview_image = None
        for index, detection in enumerate(detections, start=1):
            properties, crop = detection[:2]
            preview = detection[2] if len(detection) > 2 else None
            element_id = f"{upload_id}-{index}"
            feature = {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
                "properties": {
                    **properties,
                    "id": element_id,
                    "contribuicao_id": upload_id,
                    "fonte": SOURCE,
                    "origem_coordenadas": location_source,
                    "estado": "automatico",
                    "n_imagens": 1,
                    "n_vistas": 1,
                },
            }
            if preview:
                preview_image = preview.image
                masks.append((element_id, upload_id, preview.mask))
                feature["properties"]["imagem_bbox_relativa"] = preview.box
            rows.append((element_id, json.dumps(feature, ensure_ascii=False), crop))

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("CREATE TABLE IF NOT EXISTS photos (digest TEXT PRIMARY KEY)")
            db.execute(
                "CREATE TABLE IF NOT EXISTS elements (id TEXT PRIMARY KEY, feature TEXT NOT NULL, crop BLOB)"
            )
            result = db.execute("INSERT OR IGNORE INTO photos VALUES (?)", (digest,))
            if not result.rowcount:
                return None
            db.executemany("INSERT INTO elements VALUES (?, ?, ?)", rows)
            if masks:
                db.execute("CREATE TABLE IF NOT EXISTS preview_photos (id TEXT PRIMARY KEY, image BLOB NOT NULL)")
                db.execute("CREATE TABLE IF NOT EXISTS preview_masks (id TEXT PRIMARY KEY, photo_id TEXT NOT NULL, mask BLOB)")
                db.execute("INSERT INTO preview_photos VALUES (?, ?)", (upload_id, preview_image))
                db.executemany("INSERT INTO preview_masks VALUES (?, ?, ?)", masks)
        return upload_id

    def update(self, feature):
        with closing(sqlite3.connect(self.path)) as db, db:
            result = db.execute(
                "UPDATE elements SET feature = ? WHERE id = ?",
                (json.dumps(feature, ensure_ascii=False), feature["properties"]["id"]),
            )
            return bool(result.rowcount)
