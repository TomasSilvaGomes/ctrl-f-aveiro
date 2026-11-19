"""Prepara fotos e máscaras do catálogo a partir das imagens locais."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from photo_preview import (
    SAM2_MODEL, PhotoPreview, SegmentationError, catalog_photo_views, jpeg_bytes, local_preview_image,
    normalised_box, prepare_masks, preview_path, segmentation_directory, store_preview,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogo", type=Path, default=ROOT / "data" / "catalogo" / "catalogo.geojson")
    parser.add_argument("--descarregar-modelo", action="store_true", help="Descarrega os pesos do SAM 2.")
    args = parser.parse_args()
    if args.descarregar_modelo:
        from huggingface_hub import snapshot_download

        snapshot_download(
            SAM2_MODEL, local_dir=segmentation_directory(),
            allow_patterns=["config.json", "preprocessor_config.json", "processor_config.json", "model.safetensors"],
        )
    with args.catalogo.open(encoding="utf-8") as file:
        features = json.load(file)["features"]
    generated = missing = skipped = 0
    views = (view for feature in features for view in catalog_photo_views(feature["properties"]))
    for props in views:
        target = preview_path(args.catalogo, props)
        mask_path = target.with_suffix(".png")
        if target.is_file() and mask_path.is_file():
            skipped += 1
            continue
        image, box = local_preview_image(args.catalogo, props)
        if image is None or box is None:
            missing += 1
            continue
        try:
            (mask,) = prepare_masks(image, [box])
        except SegmentationError as error:
            parser.exit(1, f"{error}\n")
        if mask is None:
            missing += 1
            continue
        store_preview(args.catalogo, props, PhotoPreview(jpeg_bytes(image), mask, normalised_box(box, image.size)))
        generated += 1
    print(f"{generated} máscaras geradas, {skipped} já preparadas, {missing} sem máscara local.")


if __name__ == "__main__":
    main()
