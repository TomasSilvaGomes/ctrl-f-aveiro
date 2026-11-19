"""Fotografias da API oficial usadas como entrada para os modelos."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import requests
from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[1]   # raiz do projeto (.env)
MAX_IMAGE_BYTES = 12 * 1024 * 1024
PHOTO_LICENSE = "CC BY-SA 4.0"


class MapillaryPhotoError(ValueError):
    pass


@dataclass(frozen=True)
class MapillaryPhoto:
    data: bytes
    coordinates: tuple[float, float] | None
    author: str | None
    panoramic: bool = False


def mapillary_token():
    token = os.environ.get("MAPILLARY_TOKEN", "").strip()
    if not token and (ROOT / ".env").is_file():
        token = (dotenv_values(ROOT / ".env").get("MAPILLARY_TOKEN") or "").strip()
    return token if token != "MLY|COLA_AQUI" else ""


def image_key(value):
    value = str(value or "").strip()
    if re.fullmatch(r"[0-9]{1,24}", value):
        return value
    try:
        url = urlsplit(value)
    except ValueError:
        return None
    if url.scheme == "https" and url.hostname in ("mapillary.com", "www.mapillary.com") and url.path.rstrip("/") == "/app":
        keys = parse_qs(url.query).get("pKey", ())
        if len(keys) == 1 and re.fullmatch(r"[0-9]{1,24}", keys[0]):
            return keys[0]
    return None


def photo_link(key):
    return f"https://www.mapillary.com/app/?pKey={key}"


def _coordinates(metadata):
    for key in ("computed_geometry", "geometry"):
        geometry = metadata.get(key)
        if not isinstance(geometry, dict) or geometry.get("type") != "Point":
            continue
        coords = geometry.get("coordinates")
        if not isinstance(coords, list) or len(coords) < 2:
            continue
        try:
            lon, lat = map(float, coords[:2])
        except (TypeError, ValueError):
            continue
        if math.isfinite(lat) and -90 <= lat <= 90 and math.isfinite(lon) and -180 <= lon <= 180:
            return lat, lon
    return None


def _thumbnail(url):
    if not isinstance(url, str):
        raise MapillaryPhotoError("A API não devolveu uma fotografia para este ID.")
    try:
        address = urlsplit(url)
    except ValueError:
        raise MapillaryPhotoError("A API devolveu um endereço de fotografia inválido.") from None
    host = address.hostname or ""
    domains = ("fbcdn.net", "mapillary.com")
    if address.scheme != "https" or not any(host == domain or host.endswith(f".{domain}") for domain in domains):
        raise MapillaryPhotoError("A API devolveu um endereço de fotografia inválido.")
    data = bytearray()
    with requests.get(url, timeout=(4, 12), stream=True, allow_redirects=False) as reply:
        if reply.status_code != 200:
            raise MapillaryPhotoError("Não foi possível descarregar esta fotografia.")
        for block in reply.iter_content(chunk_size=64 * 1024):
            if len(data) + len(block) > MAX_IMAGE_BYTES:
                raise MapillaryPhotoError("A fotografia devolvida pela API é demasiado grande.")
            data.extend(block)
    return bytes(data)


def fetch_photo(key, token):
    if not isinstance(key, str) or not re.fullmatch(r"[0-9]{1,24}", key):
        raise MapillaryPhotoError("Indica um ID ou link de uma fotografia do Mapillary.")
    if not token:
        raise MapillaryPhotoError("Falta configurar MAPILLARY_TOKEN no servidor.")
    try:
        with requests.get(
            f"https://graph.mapillary.com/{key}",
            params={
                "access_token": token,
                "fields": "thumb_2048_url,geometry,computed_geometry,creator,is_pano,camera_type",
            },
            timeout=(4, 10), allow_redirects=False,
        ) as reply:
            if reply.status_code in (401, 403):
                raise MapillaryPhotoError("O Mapillary recusou o acesso à fotografia. Verifica o token do servidor.")
            if reply.status_code == 429:
                raise MapillaryPhotoError("O Mapillary atingiu o limite de pedidos. Tenta novamente mais tarde.")
            if reply.status_code != 200:
                raise MapillaryPhotoError("O Mapillary não disponibilizou esta fotografia. Verifica o ID e o token.")
            metadata = reply.json()
        if not isinstance(metadata, dict):
            raise MapillaryPhotoError("A API devolveu dados de fotografia inválidos.")
        image = _thumbnail(metadata.get("thumb_2048_url"))
    except requests.RequestException:
        raise MapillaryPhotoError("Não foi possível ligar à API do Mapillary. Tenta novamente.") from None
    creator = metadata.get("creator")
    author = creator.get("username") if isinstance(creator, dict) else None
    panoramic = metadata.get("is_pano") is True or metadata.get("camera_type") in ("spherical", "equirectangular")
    return MapillaryPhoto(image, _coordinates(metadata), author if isinstance(author, str) else None, panoramic)
