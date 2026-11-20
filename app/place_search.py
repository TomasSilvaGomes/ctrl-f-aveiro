"""Pesquisa local de localidades portuguesas."""

from __future__ import annotations

import csv
import unicodedata
from functools import lru_cache
from pathlib import Path


# Localidades do GeoNames (PT.zip, CC BY 4.0): ver docs/LICENCAS.md
INDEX = Path(__file__).resolve().parents[1] / "data" / "meta" / "localidades_pt.csv"


def normalized_search(value):
    text = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in text if not unicodedata.combining(char))


@lru_cache(maxsize=1)
def _places():
    with INDEX.open(encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    options, lookup = [], {}
    for row in rows:
        name = row["name"]
        label = name
        if normalized_search(label) in lookup:
            council, parish = row["council"], row["parish"]
            if council and normalized_search(council) != normalized_search(name):
                label += f" · {council}"
            if normalized_search(label) in lookup and parish and parish not in (name, council):
                label += f" · {parish}"
        point = (float(row["lat"]), float(row["lon"]))
        if normalized_search(label) in lookup:
            label += f" ({point[0]:.4f}, {point[1]:.4f})"
        if normalized_search(label) not in lookup:
            options.append(label)
            lookup[normalized_search(label)] = (point, label)
    return tuple(options), lookup


def place_suggestions():
    return _places()[0]


def find_place(query):
    return _places()[1].get(normalized_search(query.strip()))
