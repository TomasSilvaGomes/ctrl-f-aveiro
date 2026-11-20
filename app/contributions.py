"""Entrada, revisão e publicação das contribuições fotográficas."""

from __future__ import annotations

import io
import json
import math
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
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


def photo_camera(data: bytes) -> dict:
    """Para onde a câmara apontava (GPSImgDirection, graus) e focal equivalente a 35 mm, se o EXIF os tiver.
    Sem o rumo não dá para pôr os objetos à distância certa: ficam no local da foto."""
    camera = {}
    with open_photo(data) as image:
        try:
            exif = image.getexif()
            heading = exif.get_ifd(ExifTags.IFD.GPSInfo).get(17)
            if heading is not None and math.isfinite(float(heading)):
                camera["heading"] = float(heading) % 360
            focal = exif.get_ifd(ExifTags.IFD.Exif).get(0xA405)
            if focal and 8 <= float(focal) <= 300:
                camera["focal35"] = float(focal)
        except (ArithmeticError, AttributeError, OSError, TypeError, ValueError, SyntaxError):
            pass
    return camera


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
            db.row_factory = sqlite3.Row
            if not db.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (table,)).fetchone():
                return []
            return db.execute(sql, parameters).fetchall()

    def features(self):
        return [json.loads(row[0]) for row in self._read("SELECT feature FROM elements ORDER BY rowid")]

    def crop(self, element_id):
        rows = self._read("SELECT crop FROM elements WHERE id = ?", (element_id,))
        if not rows:
            rows = self._read(
                "SELECT crop FROM request_elements WHERE id = ?", (element_id,), table="request_elements",
            )
        return rows[0][0] if rows else None

    def preview(self, element_id, box=None):
        rows = self._read(
            "SELECT image, mask FROM preview_masks JOIN preview_photos ON preview_photos.id = preview_masks.photo_id "
            "WHERE preview_masks.id = ?",
            (element_id,), table="preview_masks",
        )
        return PhotoPreview(*rows[0], box=box) if rows else None

    def has_photo(self, digest):
        if not self.path.exists():
            return False
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            self._prepare_photo_links(db)
            return db.execute("SELECT 1 FROM photos WHERE digest = ?", (digest,)).fetchone() is not None

    def _prepare_photo_links(self, db):
        columns = db.execute("PRAGMA table_info(photos)").fetchall()
        if not columns:
            db.execute("CREATE TABLE photos (digest TEXT PRIMARY KEY, request_id TEXT)")
            return
        if any(column[1] == "request_id" for column in columns):
            return
        db.execute("ALTER TABLE photos ADD COLUMN request_id TEXT")
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name = 'photo_requests'").fetchone():
            return
        # A versão anterior inseria o hash e o pedido juntos, pela mesma ordem.
        unpaired = db.execute("""
            SELECT EXISTS(SELECT rowid FROM photos EXCEPT SELECT rowid FROM photo_requests)
                OR EXISTS(SELECT rowid FROM photo_requests EXCEPT SELECT rowid FROM photos)
        """).fetchone()[0]
        if unpaired:
            return
        db.execute("""
            UPDATE photos SET request_id = (SELECT id FROM photo_requests WHERE rowid = photos.rowid)
        """)
        preview_table = db.execute("SELECT 1 FROM sqlite_master WHERE name = 'preview_photos'").fetchone()
        keep_preview = (
            "AND NOT EXISTS (SELECT 1 FROM preview_photos WHERE id = photos.request_id)"
            if preview_table else ""
        )
        db.execute("""
            DELETE FROM photos
            WHERE request_id IN (SELECT id FROM photo_requests WHERE status = 'aprovado')
            AND NOT EXISTS (
                SELECT 1 FROM elements
                WHERE json_extract(feature, '$.properties.contribuicao_id') = photos.request_id
                AND COALESCE(json_extract(feature, '$.properties.sem_fotografia'), 0) = 0
            )
        """ + keep_preview)

    def submit(self, digest, latitude, longitude, location_source, detections, *, name, source, preview=None, credit=None):
        if not (math.isfinite(latitude) and -90 <= latitude <= 90
                and math.isfinite(longitude) and -180 <= longitude <= 180):
            raise ValueError("Indique coordenadas válidas para a fotografia.")
        upload_id = f"FOTO-{uuid4().hex[:12]}"
        submitted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        detection_count = len(detections)
        if not detections:
            detections = [({
                "categoria": PHOTO_CATEGORY,
                "categoria_label": "Fotografias",
                "subtipo": "sem_deteccoes",
                "label": "Fotografia sem elementos detetados",
                "confianca": None,
            }, None, preview)]
        rows, masks = [], []
        preview_image = None
        for index, detection in enumerate(detections, start=1):
            properties, crop = detection[:2]
            preview = detection[2] if len(detection) > 2 else None
            element_id = f"{upload_id}-{index}"
            point = properties.get("_lonlat") or [longitude, latitude]   # posição calculada, ou o local da foto
            feature = {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": point},
                "properties": {
                    **{key: value for key, value in properties.items() if key != "_lonlat"},
                    **(credit or {}),
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
            rows.append((element_id, upload_id, json.dumps(feature, ensure_ascii=False), crop))

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "CREATE TABLE IF NOT EXISTS elements (id TEXT PRIMARY KEY, feature TEXT NOT NULL, crop BLOB)"
            )
            db.execute("""
                CREATE TABLE IF NOT EXISTS photo_requests (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, source TEXT NOT NULL,
                    latitude REAL NOT NULL, longitude REAL NOT NULL, detection_count INTEGER NOT NULL,
                    status TEXT NOT NULL, submitted_at TEXT NOT NULL
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS request_elements (
                    id TEXT PRIMARY KEY, request_id TEXT NOT NULL, feature TEXT NOT NULL, crop BLOB
                )
            """)
            db.execute("CREATE INDEX IF NOT EXISTS request_elements_by_request ON request_elements (request_id)")
            db.execute("""
                CREATE TABLE IF NOT EXISTS photo_review_log (
                    id INTEGER PRIMARY KEY, request_id TEXT NOT NULL, action TEXT NOT NULL,
                    at TEXT NOT NULL, note TEXT NOT NULL
                )
            """)
            self._prepare_photo_links(db)
            result = db.execute(
                "INSERT OR IGNORE INTO photos (digest, request_id) VALUES (?, ?)", (digest, upload_id),
            )
            if not result.rowcount:
                return None
            db.execute(
                "INSERT INTO photo_requests VALUES (?, ?, ?, ?, ?, ?, 'pendente', ?)",
                (upload_id, name, source, latitude, longitude, detection_count, submitted_at),
            )
            db.executemany("INSERT INTO request_elements VALUES (?, ?, ?, ?)", rows)
            if masks:
                db.execute("CREATE TABLE IF NOT EXISTS preview_photos (id TEXT PRIMARY KEY, image BLOB NOT NULL)")
                db.execute("CREATE TABLE IF NOT EXISTS preview_masks (id TEXT PRIMARY KEY, photo_id TEXT NOT NULL, mask BLOB)")
                db.execute("INSERT INTO preview_photos VALUES (?, ?)", (upload_id, preview_image))
                db.executemany("INSERT INTO preview_masks VALUES (?, ?, ?)", masks)
            db.execute(
                "INSERT INTO photo_review_log (request_id, action, at, note) VALUES (?, 'submetido', ?, '')",
                (upload_id, submitted_at),
            )
        return upload_id

    def requests(self, status=None):
        where = " WHERE status = ?" if status is not None else ""
        rows = self._read(
            "SELECT * FROM photo_requests" + where + " ORDER BY submitted_at DESC, rowid DESC",
            (status,) if status is not None else (), table="photo_requests",
        )
        return [dict(row) for row in rows]

    def pending_count(self):
        rows = self._read("SELECT COUNT(*) FROM photo_requests WHERE status = 'pendente'", table="photo_requests")
        return rows[0][0] if rows else 0

    def request_features(self, request_id):
        rows = self._read("""
            SELECT feature FROM request_elements WHERE request_id = ?
            UNION ALL
            SELECT feature FROM elements WHERE json_extract(feature, '$.properties.contribuicao_id') = ?
        """, (request_id, request_id), table="request_elements")
        return [json.loads(row[0]) for row in rows]

    def review_log(self, request_id=None):
        sql = """
            SELECT l.request_id, r.name, r.source, l.action, l.at, l.note
            FROM photo_review_log AS l JOIN photo_requests AS r ON r.id = l.request_id
        """
        where = " WHERE l.request_id = ?" if request_id else ""
        rows = self._read(
            sql + where + " ORDER BY l.id DESC LIMIT 200", (request_id,) if request_id else (),
            table="photo_review_log",
        )
        return [dict(row) for row in rows]

    def review(self, request_id, decision, note):
        if decision not in ("aprovado", "rejeitado"):
            raise ValueError("Decisão inválida.")
        reviewed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            result = db.execute(
                "UPDATE photo_requests SET status = ? WHERE id = ? AND status = 'pendente'",
                (decision, request_id),
            )
            if not result.rowcount:
                return False
            if decision == "aprovado":
                db.execute("""
                    INSERT INTO elements (id, feature, crop)
                    SELECT id, json_set(feature, '$.properties.estado', 'confirmado'), crop
                    FROM request_elements WHERE request_id = ? ORDER BY rowid
                """, (request_id,))
                db.execute("DELETE FROM request_elements WHERE request_id = ?", (request_id,))
            db.execute(
                "INSERT INTO photo_review_log (request_id, action, at, note) VALUES (?, ?, ?, ?)",
                (request_id, decision, reviewed_at, note.strip()),
            )
        return True

    def update(self, element_id, edit, *, remove_photo=False, remove_object=False):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT feature FROM elements WHERE id = ?", (element_id,)).fetchone()
            if row is None:
                return False
            if remove_object or remove_photo:
                item_ids = [element_id]
                if remove_photo and (request_id := json.loads(row[0])["properties"].get("contribuicao_id")):
                    item_ids = [
                        item_id for (item_id,) in db.execute(
                            "SELECT id FROM elements WHERE id = ? "
                            "OR json_extract(feature, '$.properties.contribuicao_id') = ?",
                            (element_id, request_id),
                        )
                    ]
                db.executemany("DELETE FROM elements WHERE id = ?", ((item_id,) for item_id in item_ids))
                for item_id in item_ids:
                    self._remove_preview(db, item_id, keep_photo=remove_object)
                if remove_photo and request_id:
                    self._prepare_photo_links(db)
                    db.execute("DELETE FROM photos WHERE request_id = ?", (request_id,))
            else:
                db.execute(
                    "UPDATE elements SET feature = ? WHERE id = ?",
                    (json.dumps(edit(json.loads(row[0])), ensure_ascii=False), element_id),
                )
        return True

    def _remove_preview(self, db, element_id, *, keep_photo=False):
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name = 'preview_masks'").fetchone():
            return
        preview = db.execute("SELECT photo_id FROM preview_masks WHERE id = ?", (element_id,)).fetchone()
        if preview is None:
            return
        db.execute("DELETE FROM preview_masks WHERE id = ?", (element_id,))
        if not keep_photo:
            db.execute(
                "DELETE FROM preview_photos WHERE id = ? "
                "AND NOT EXISTS (SELECT 1 FROM preview_masks WHERE photo_id = preview_photos.id)",
                (preview[0],),
            )
