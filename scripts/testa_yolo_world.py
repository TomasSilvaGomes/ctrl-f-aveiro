"""
Teste numa imagem: YOLO-World deteta categorias e SigLIP classifica subtipos.

Esta variante é experimental: não altera o OWLv2 nem o catálogo principal.
As categorias e os subtipos são lidos da taxonomia do projeto.

Instalação opcional:
    python -m pip install "ultralytics>=8.2.0"

Classes padrão: prompts oficiais de deteção por categoria na taxonomia.

Uso:
    python scripts/testa_yolo_world.py --image caminho\foto.jpg
    python scripts/testa_yolo_world.py --image caminho\foto.jpg --classes "stop sign,street light pole,waste container"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = Path(r"C:\Users\Utilizador\Desktop\2.png")
TAXONOMIA_PATH = ROOT / "config" / "taxonomia.yaml"


def load_taxonomy():
    with TAXONOMIA_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_detector_classes(tax):
    prompts = []
    lookup = {}
    for category, category_prompts in tax["detetor"].items():
        for prompt in category_prompts:
            prompts.append(prompt)
            lookup[prompt] = {
                "category": category,
                "category_label": tax["categorias"][category]["label"],
            }
    return prompts, lookup


def iou(a, b):
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def box_to_dict(box, prompts, lookup):
    x0, y0, x1, y1 = [round(float(v), 1) for v in box.xyxy[0].tolist()]
    score = round(float(box.conf[0]), 4)
    prompt = prompts[int(box.cls[0])]
    info = lookup[prompt]
    return {
        "bbox": [x0, y0, x1, y1],
        "score": score,
        "prompt": prompt,
        "category": info["category"],
        "category_label": info["category_label"],
        "subtype": None,
    }


def nms(detections, threshold=0.5):
    keep = []
    for det in sorted(detections, key=lambda d: -d["score"]):
        if all(
            det["category"] != k["category"] or iou(det["bbox"], k["bbox"]) < threshold
            for k in keep
        ):
            keep.append(det)
    return keep


def overlap_ratio(a, b):
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if not inter:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / min(area_a, area_b)


def deduplicate(detections):
    """Remove caixas repetidas da mesma categoria quando representam o mesmo objeto."""
    keep = []
    for det in sorted(detections, key=lambda d: -d["score"]):
        duplicate = next((
            existing for existing in keep
            if (
                existing["category"] == det["category"]
                or (existing["subtype"] in ("poste_iluminacao", "candeeiro")
                    and det["subtype"] in ("poste_iluminacao", "candeeiro"))
            )
            and (
                iou(existing["bbox"], det["bbox"]) >= 0.25
                or overlap_ratio(existing["bbox"], det["bbox"]) >= 0.6
            )
            and (
                det["category"] != "iluminacao"
                or abs(
                    (existing["bbox"][0] + existing["bbox"][2]) / 2
                    - (det["bbox"][0] + det["bbox"][2]) / 2
                ) <= max(
                    existing["bbox"][2] - existing["bbox"][0],
                    det["bbox"][2] - det["bbox"][0],
                )
            )
        ), None)
        if duplicate is None:
            keep.append(det)
            continue
        if det["category"] == "iluminacao":
            duplicate["bbox"] = [
                min(duplicate["bbox"][0], det["bbox"][0]),
                min(duplicate["bbox"][1], det["bbox"][1]),
                max(duplicate["bbox"][2], det["bbox"][2]),
                max(duplicate["bbox"][3], det["bbox"][3]),
            ]
        if det["score"] > duplicate["score"]:
            duplicate.update({k: v for k, v in det.items() if k != "bbox"})
    return keep


def shape_filter(detection):
    x0, y0, x1, y1 = detection["bbox"]
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    ratio = h / w
    subtype = detection["subtype"]
    if subtype == "stop":
        return 0.65 <= ratio <= 1.35
    if subtype == "poste_iluminacao":
        return ratio >= 1.5
    if subtype == "candeeiro":
        return ratio >= 0.35
    if subtype in (
        "ilha_reciclagem",
        "ecoponto_papel",
        "ecoponto_embalagens",
        "ecoponto_vidro",
        "indiferenciado",
        "subterraneo",
    ):
        return 0.7 <= ratio <= 3.0
    if detection["category"] == "sinalizacao":
        return 0.35 <= ratio <= 2.5
    if subtype == "semaforo":
        return 0.35 <= ratio <= 3.5
    return True


def confidence_filter(detection):
    threshold = {
        "stop": 0.10,
        "sentido_proibido": 0.10,
        "passagem_peoes": 0.10,
        "poste_iluminacao": 0.30,
        "candeeiro": 0.30,
        "semaforo": 0.30,
        "ilha_reciclagem": 0.25,
        "ecoponto_papel": 0.25,
        "ecoponto_embalagens": 0.25,
        "ecoponto_vidro": 0.25,
        "indiferenciado": 0.25,
        "subterraneo": 0.25,
    }.get(detection["subtype"], 0.15 if detection["category"] == "sinalizacao" else 0.35)
    return detection["score"] >= threshold


def tile_starts(length, tile_size):
    if length <= tile_size:
        return [0]
    stride = max(1, tile_size * 3 // 4)
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def detect_tiled(model, image, prompts, lookup, confidence, tile_size, imgsz):
    height, width = image.shape[:2]
    detections = []
    if tile_size is None:
        tiles = [(0, 0, image)]
    else:
        tiles = [
            (
                x0,
                y0,
                image[y0:min(y0 + tile_size, height), x0:min(x0 + tile_size, width)],
            )
            for y0 in tile_starts(height, tile_size)
            for x0 in tile_starts(width, tile_size)
        ]
    for x0, y0, tile in tiles:
        result = model.predict(
            source=tile,
            conf=confidence,
            iou=0.5,
            imgsz=imgsz,
            save=False,
            verbose=False,
        )[0]
        for box in result.boxes:
            det = box_to_dict(box, prompts, lookup)
            x1, y1, x2, y2 = det["bbox"]
            det["bbox"] = [
                round(x1 + x0, 1), round(y1 + y0, 1),
                round(x2 + x0, 1), round(y2 + y0, 1),
            ]
            detections.append(det)
    return detections


def classify_detections(image_path, detections, tax, device):
    import torch
    from PIL import Image

    from classify_subtype import Classifier, square_crop

    classifier = Classifier(tax, device)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        crops = [square_crop(image, det["bbox"])[0] for det in detections]
    predictions = classifier(crops, [det["category"] for det in detections])
    classified = []
    subtype_info = {
        (category, subtype["id"]): subtype
        for category, spec in tax["categorias"].items()
        for subtype in spec["subtipos"]
    }
    for det, (subtype, subtype_score, negative_score) in zip(detections, predictions):
        if subtype is None:
            continue
        info = subtype_info[(det["category"], subtype)]
        label = info["label"].replace(" {n} km/h", "")
        det.update({
            "subtype": subtype,
            "label": label,
            "sub_score": round(subtype_score, 4),
            "negative_score": round(negative_score, 4),
        })
        classified.append(det)
    return classified


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--model", default="yolov8m-world.pt")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Tamanho dos tiles; por omissão, processa a imagem inteira.",
    )
    parser.add_argument("--classes", default=None,
                        help="Lista separada por vírgulas de prompts oficiais da taxonomia.")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "relatorios" / "teste_yolo_world.jpg")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.image.exists():
        raise SystemExit(f"Imagem não encontrada: {args.image}")

    try:
        from ultralytics import YOLO
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "YOLO-World não está instalado. Instale com:\n"
            'python -m pip install "ultralytics>=8.2.0"'
        ) from exc
    except ImportError as exc:
        raise SystemExit(
            "Não foi possível carregar PyTorch/Ultralytics neste ambiente Python. "
            "Verifique a instalação do PyTorch."
        ) from exc

    tax = load_taxonomy()
    catalog_prompts, lookup = load_detector_classes(tax)
    if args.classes is None:
        prompts = catalog_prompts
    else:
        prompts = [v.strip() for v in args.classes.split(",") if v.strip()]
        lookup = {p: lookup[p] for p in prompts if p in lookup} if lookup else {}
    if not prompts:
        raise SystemExit("A lista --classes não pode ficar vazia.")
    missing = [p for p in prompts if p not in lookup]
    if missing:
        raise SystemExit(
            "Prompts não encontrados na taxonomia:\n  " + "\n  ".join(missing)
        )

    model = YOLO(args.model)
    model.set_classes(prompts)
    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Não foi possível ler a imagem: {args.image}")
    detections = detect_tiled(
        model, image, prompts, lookup, args.threshold, args.tile_size, args.imgsz
    )
    detections = deduplicate(detections)
    detections = nms(detections)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detections = classify_detections(args.image, detections, tax, device)
    detections = [det for det in detections if shape_filter(det)]
    detections = [det for det in detections if confidence_filter(det)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    annotated = image.copy()
    for det in detections:
        x0, y0, x1, y1 = [int(v) for v in det["bbox"]]
        text = f"{det['label']} {det['score']:.2f}"
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 200, 255), 2)
        font_scale, font_thickness = 1.6, 5
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        ty = max(0, y0 - th - 10)
        cv2.rectangle(
            annotated, (x0, ty), (x0 + tw + 16, ty + th + baseline + 10),
            (0, 200, 255), -1
        )
        cv2.putText(
            annotated, text, (x0 + 8, ty + th + 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20),
            font_thickness, cv2.LINE_AA
        )
    cv2.imwrite(str(args.output), annotated)

    json_path = args.output.with_suffix(".json")
    public_detections = [
        {
            "caixa": det["bbox"],
            "confianca": det["score"],
            "categoria": det["category_label"],
            "subtipo": det["subtype"],
            "rotulo": det["label"],
            "confianca_subtipo": det["sub_score"],
            "prompt_tecnico": det["prompt"],
        }
        for det in detections
    ]
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "imagem": str(args.image),
            "modelo": args.model,
            "classes": sorted({lookup[p]["category_label"] for p in prompts}),
            "prompts_tecnicos": prompts,
            "deteccoes": public_detections,
        }, f, ensure_ascii=False, indent=2)

    print(f"Imagem: {args.image}")
    print(f"Deteções: {len(detections)}")
    for det in detections:
        print(f"  {det['label']:<38} {det['score']:.0%}  {det['bbox']}")
    print(f"Saída: {args.output}")
    print(f"JSON:  {json_path}")


if __name__ == "__main__":
    main()
