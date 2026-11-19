"""Deteção local de rostos e matrículas quando o EgoBlur está indisponível."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np
import requests

from photo_preview import write_atomic


WEIGHTS = Path(__file__).resolve().parent / "weights" / "privacy"
FACE_MODEL = (
    "https://huggingface.co/opencv/opencv_zoo/resolve/39a1451/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
)
PLATE_MODEL = (
    "https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-s-608-license-plates-end2end.onnx",
    "2b878b38d9aa07b6ddc3ea75c4ffcb39869bc5c218e0a14002f60ab2f7b0be9a",
)
PLATE_SIZE = 608


def _weights_file(url, checksum):
    path = WEIGHTS / url.rsplit("/", 1)[-1]
    if path.is_file() and sha256(path.read_bytes()).hexdigest() == checksum:
        return path
    response = requests.get(url, timeout=(5, 25))
    response.raise_for_status()
    data = response.content
    if sha256(data).hexdigest() != checksum:
        raise ValueError(f"Os pesos recebidos não correspondem a {path.name}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, data)
    return path


class LocalPrivacyDetector:
    def __init__(self):
        import onnxruntime as ort

        face_path = _weights_file(*FACE_MODEL)
        plate_path = _weights_file(*PLATE_MODEL)
        self.faces = cv2.FaceDetectorYN.create(str(face_path), "", (320, 320), score_threshold=0.55)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.plates = ort.InferenceSession(str(plate_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.plate_input = self.plates.get_inputs()[0].name

    def regions(self, rgb):
        height, width = rgb.shape[:2]
        self.faces.setInputSize((width, height))
        _, faces = self.faces.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        regions = []
        if faces is not None:
            regions.extend((x, y, x + w, y + h) for x, y, w, h in faces[:, :4])
        regions.extend(self.plate_regions(rgb))
        return regions

    def plate_regions(self, rgb):
        height, width = rgb.shape[:2]
        ratio = PLATE_SIZE / max(width, height)
        scaled_width = max(1, round(width * ratio))
        scaled_height = max(1, round(height * ratio))
        left = (PLATE_SIZE - scaled_width) // 2
        top = (PLATE_SIZE - scaled_height) // 2
        canvas = np.full((PLATE_SIZE, PLATE_SIZE, 3), 114, dtype=np.uint8)
        canvas[top:top + scaled_height, left:left + scaled_width] = cv2.resize(rgb, (scaled_width, scaled_height))
        pixels = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        (predictions,) = self.plates.run(None, {self.plate_input: pixels})
        boxes = predictions[predictions[:, 6] >= 0.25, 1:5]
        scale_x, scale_y = scaled_width / width, scaled_height / height
        return (boxes - (left, top, left, top)) / (scale_x, scale_y, scale_x, scale_y)
