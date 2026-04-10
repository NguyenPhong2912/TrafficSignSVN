"""
Standalone inference backend for the GitHub desktop package.
"""

import base64
import json
import os
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory, make_response


BASE_DIR = Path(__file__).resolve().parent


def resolve_runtime_dir(raw_value: str | None, default_relative: str) -> Path:
    candidate = Path(raw_value) if raw_value else BASE_DIR / default_relative
    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate
    return candidate.resolve()


def resolve_runtime_file(raw_value: str | None, *relative_candidates: str) -> tuple[Path, list[str]]:
    candidates: list[Path] = []
    seen: set[str] = set()

    values = [raw_value] if raw_value else list(relative_candidates)
    for value in values:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (BASE_DIR / candidate).resolve()
        else:
            candidate = candidate.resolve()

        key = str(candidate).lower()
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)

    if not candidates:
        raise ValueError("At least one candidate path is required")

    for candidate in candidates:
        if candidate.exists():
            return candidate, [str(path) for path in candidates]

    return candidates[0], [str(path) for path in candidates]

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CAPTURES_DIR = resolve_runtime_dir(os.environ.get("CAPTURES_DIR"), "captures")
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Model paths (set via env vars on Render) ───────────────────────────────────
YOLO_SIGN_PATH, YOLO_SIGN_CANDIDATES = resolve_runtime_file(
    os.environ.get("YOLO_SIGN_PATH"),
    "weights/yolo_sign_best.pt",
)
YOLO_PLATE_PATH, YOLO_PLATE_CANDIDATES = resolve_runtime_file(
    os.environ.get("YOLO_PLATE_PATH"),
    "weights/yolo_plate_best.pt",
)
CLIP_CLF_PATH, CLIP_CLF_CANDIDATES = resolve_runtime_file(
    os.environ.get("CLIP_CLF_PATH"),
    "weights/clip_classifier_v7.pt",
)
CLIP_MODEL_NAME = os.environ.get("CLIP_MODEL",      "ViT-L-14")
CLIP_PRETRAIN   = os.environ.get("CLIP_PRETRAIN",   "laion2b_s32b_b82k")
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESH     = float(os.environ.get("CONF_THRESH", "0.25"))
PREVIEW_CONF_THRESH = float(os.environ.get("PREVIEW_CONF_THRESH", "0.12"))
MIN_BBOX        = int(os.environ.get("MIN_BBOX", "20"))
PREVIEW_MIN_BBOX = int(os.environ.get("PREVIEW_MIN_BBOX", "10"))
YOLO_SIGN_IMGSZ = int(os.environ.get("YOLO_SIGN_IMGSZ", "1280"))
YOLO_PLATE_IMGSZ = int(os.environ.get("YOLO_PLATE_IMGSZ", "1280"))

# ── Global model state ─────────────────────────────────────────────────────────
models = {
    "yolo_sign":  None,
    "yolo_plate": None,
    "clip_clf":   None,
    "clip_pre":   None,
    "class_names": [],
    "ocr_reader": None,
    "loaded": False,
    "error": None,
    "ready_full_flow": False,
    "component_errors": {},
    "paths": {
        "captures_dir": str(CAPTURES_DIR),
        "yolo_sign": str(YOLO_SIGN_PATH),
        "yolo_plate": str(YOLO_PLATE_PATH),
        "clip_clf": str(CLIP_CLF_PATH),
    },
    "path_candidates": {
        "yolo_sign": YOLO_SIGN_CANDIDATES,
        "yolo_plate": YOLO_PLATE_CANDIDATES,
        "clip_clf": CLIP_CLF_CANDIDATES,
    },
}


# ── CLIP Classifier head (must match notebook Cell 7) ─────────────────────────
class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, clip_dim, num_classes, freeze_backbone=True):
        super().__init__()
        self.clip = clip_model
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.clip.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(clip_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        feats = self.clip.encode_image(x).float()
        return self.head(feats)


def set_component_error(name: str, message: str) -> None:
    models["component_errors"][name] = message
    print(f"[WARN] {name}: {message}")


def _legacy_load_models_unused():
    """Load all models once at startup."""
    try:
        from ultralytics import YOLO
        import open_clip
        import easyocr

        print(f"[LOAD] Device: {DEVICE}")

        # YOLO sign
        if Path(YOLO_SIGN_PATH).exists():
            models["yolo_sign"] = YOLO(YOLO_SIGN_PATH)
            print(f"[LOAD] YOLO sign: {YOLO_SIGN_PATH}")
        else:
            print(f"[WARN] YOLO sign not found: {YOLO_SIGN_PATH}")

        # YOLO plate
        if Path(YOLO_PLATE_PATH).exists():
            models["yolo_plate"] = YOLO(YOLO_PLATE_PATH)
            print(f"[LOAD] YOLO plate: {YOLO_PLATE_PATH}")
        else:
            print(f"[WARN] YOLO plate not found: {YOLO_PLATE_PATH}")

        # CLIP classifier
        if Path(CLIP_CLF_PATH).exists():
            ckpt = torch.load(CLIP_CLF_PATH, map_location=DEVICE)
            clip_model, _, preprocess = open_clip.create_model_and_transforms(
                ckpt.get("clip_model", CLIP_MODEL_NAME),
                pretrained=CLIP_PRETRAIN,
            )
            clip_model = clip_model.to(DEVICE).eval()

            clf = CLIPClassifier(
                clip_model,
                ckpt.get("clip_dim", 768),
                ckpt["num_classes"],
            ).to(DEVICE)
            clf.load_state_dict(ckpt["state_dict"])
            clf.eval()

            models["clip_clf"]    = clf
            models["clip_pre"]    = preprocess
            models["class_names"] = ckpt["class_names"]
            print(f"[LOAD] CLIP classifier: {len(ckpt['class_names'])} classes")
        else:
            print(f"[WARN] CLIP classifier not found: {CLIP_CLF_PATH}")

        # EasyOCR
        models["ocr_reader"] = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
        print("[LOAD] EasyOCR ready")

        models["loaded"] = True
        print("[LOAD] All models ready ✓")

    except Exception as e:
        models["error"] = str(e)
        print(f"[ERROR] load_models: {e}")
        traceback.print_exc()


def clean_plate(text: str) -> str:
    return "".join(ch for ch in text.upper() if ch.isalnum() or ch in ".-")


def box_iou_xyxy(box_a: list[int], box_b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
    area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    denom = area_a + area_b - inter_area
    return inter_area / denom if denom > 0 else 0.0


def nms_xyxy(detections: list[dict], iou_thresh: float = 0.45) -> list[dict]:
    kept: list[dict] = []
    for det in sorted(detections, key=lambda item: item["conf"], reverse=True):
        if all(box_iou_xyxy(det["bbox"], prev["bbox"]) < iou_thresh for prev in kept):
            kept.append(det)
    return kept


def collect_yolo_detections(model, image_bgr: np.ndarray, *, conf_thresh: float, imgsz: int) -> list[dict]:
    result = model(image_bgr, conf=conf_thresh, imgsz=imgsz, verbose=False)[0]
    boxes = result.boxes
    detections: list[dict] = []
    if boxes is None:
        return detections
    for j in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[j].cpu().numpy().astype(int)
        detections.append(
            {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "conf": float(boxes.conf[j]),
                "cls": int(boxes.cls[j]),
            }
        )
    return detections


def iter_tile_starts(total: int, tile: int, stride: int) -> list[int]:
    if total <= tile:
        return [0]
    starts = list(range(0, max(total - tile, 0) + 1, max(1, stride)))
    last = total - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def collect_tiled_detections(
    model,
    image_bgr: np.ndarray,
    *,
    conf_thresh: float,
    imgsz: int,
    tile_size: int,
    stride: int,
) -> list[dict]:
    h, w = image_bgr.shape[:2]
    detections: list[dict] = []
    for y0 in iter_tile_starts(h, tile_size, stride):
        for x0 in iter_tile_starts(w, tile_size, stride):
            tile = image_bgr[y0:min(h, y0 + tile_size), x0:min(w, x0 + tile_size)]
            for det in collect_yolo_detections(model, tile, conf_thresh=conf_thresh, imgsz=imgsz):
                x1, y1, x2, y2 = det["bbox"]
                det["bbox"] = [x1 + x0, y1 + y0, x2 + x0, y2 + y0]
                detections.append(det)
    return detections


def collect_plate_detections(image_bgr: np.ndarray, *, conf_thresh: float, preview_mode: bool, dense: bool = False) -> list[dict]:
    if models["yolo_plate"] is None:
        return []

    detections = collect_yolo_detections(
        models["yolo_plate"],
        image_bgr,
        conf_thresh=conf_thresh,
        imgsz=YOLO_PLATE_IMGSZ,
    )

    if not preview_mode:
        scaled = cv2.resize(image_bgr, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        for det in collect_yolo_detections(
            models["yolo_plate"],
            scaled,
            conf_thresh=max(0.10, conf_thresh - 0.05),
            imgsz=max(YOLO_PLATE_IMGSZ, 1536),
        ):
            x1, y1, x2, y2 = det["bbox"]
            det["bbox"] = [
                int(round(x1 / 1.5)),
                int(round(y1 / 1.5)),
                int(round(x2 / 1.5)),
                int(round(y2 / 1.5)),
            ]
            detections.append(det)

    if dense:
        h, w = image_bgr.shape[:2]
        tile_size = 768 if preview_mode else 960
        stride = int(tile_size * (0.55 if preview_mode else 0.6))
        tile_size = min(tile_size, max(h, w))
        detections.extend(
            collect_tiled_detections(
                models["yolo_plate"],
                image_bgr,
                conf_thresh=max(0.08, conf_thresh - 0.06),
                imgsz=max(YOLO_PLATE_IMGSZ, 1408 if preview_mode else 1600),
                tile_size=tile_size,
                stride=max(256, stride),
            )
        )

    return nms_xyxy(detections, iou_thresh=0.42)


def build_plate_ocr_variants(crop: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 7, 60, 60)
    enlarged_color = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    enlarged_gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    _, otsu = cv2.threshold(enlarged_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(enlarged_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
    return [crop, gray, enlarged_color, enlarged_gray, otsu, adaptive]


def score_plate_candidate(text: str, conf: float) -> float:
    cleaned = clean_plate(text)
    compact = cleaned.replace(".", "").replace("-", "")
    if len(compact) < 2:
        return -1.0

    length = len(compact)
    has_digit = any(ch.isdigit() for ch in compact)
    has_alpha = any(ch.isalpha() for ch in compact)
    trailing_digits = len(compact) - len(compact.rstrip("0123456789"))

    score = float(conf)
    score += min(length, 10) * 0.035
    if has_digit and has_alpha:
        score += 0.22
    if compact[:2].isdigit():
        score += 0.18
    if any(ch.isalpha() for ch in compact[:5]):
        score += 0.12
    if trailing_digits >= 4:
        score += 0.20
    if trailing_digits == 0:
        score -= 0.18
    if length > 10:
        score -= (length - 10) * 0.18
    return score


def assemble_plate_text(ocr_results: list[tuple]) -> tuple[str, float]:
    deduped: dict[tuple[str, int, int], dict[str, float | str]] = {}
    for bbox, raw_text, conf in ocr_results:
        cleaned = clean_plate(raw_text)
        if len(cleaned) < 2:
            continue
        xs = [float(point[0]) for point in bbox]
        ys = [float(point[1]) for point in bbox]
        token_h = max(1.0, max(ys) - min(ys))
        token = {
            "text": cleaned,
            "conf": float(conf),
            "x": float(sum(xs) / len(xs)),
            "y": float(sum(ys) / len(ys)),
            "h": token_h,
        }
        dedupe_key = (
            cleaned,
            int(round(float(token["x"]) / 18.0)),
            int(round(float(token["y"]) / 18.0)),
        )
        previous = deduped.get(dedupe_key)
        if previous is None or float(token["conf"]) > float(previous["conf"]):
            deduped[dedupe_key] = token

    tokens = list(deduped.values())

    if not tokens:
        return "", 0.0

    candidates: list[tuple[str, float, str]] = []
    best_single = max(tokens, key=lambda item: float(item["conf"]))
    candidates.append((str(best_single["text"]), float(best_single["conf"]), "single"))

    by_y = sorted(tokens, key=lambda item: (float(item["y"]), float(item["x"])))
    lines: list[list[dict[str, float | str]]] = []
    for token in by_y:
        placed = False
        for line in lines:
            line_y = sum(float(item["y"]) for item in line) / len(line)
            line_h = max(float(item["h"]) for item in line)
            if abs(float(token["y"]) - line_y) <= max(18.0, line_h * 0.7):
                line.append(token)
                placed = True
                break
        if not placed:
            lines.append([token])

    ordered_lines = sorted(
        [sorted(line, key=lambda token: float(token["x"])) for line in lines],
        key=lambda line: sum(float(item["y"]) for item in line) / len(line),
    )
    merged_text = "".join("".join(str(item["text"]) for item in line) for line in ordered_lines)
    merged_conf = sum(float(item["conf"]) for line in ordered_lines for item in line) / sum(len(line) for line in ordered_lines)
    candidates.append((merged_text, merged_conf, "merged"))

    for line in ordered_lines:
        line_text = "".join(str(item["text"]) for item in line)
        line_conf = sum(float(item["conf"]) for item in line) / len(line)
        candidates.append((line_text, line_conf, "line"))

    best_text, best_score = "", -1.0
    best_conf = 0.0
    for candidate_text, candidate_conf, candidate_kind in candidates:
        candidate_text = clean_plate(candidate_text)
        if len(candidate_text) < 2:
            continue
        length = len(candidate_text)
        length_bonus = min(length, 10) * 0.04
        over_penalty = max(0, length - 12) * 0.25
        has_digit = any(ch.isdigit() for ch in candidate_text)
        has_alpha = any(ch.isalpha() for ch in candidate_text)
        mix_bonus = 0.22 if has_digit and has_alpha else 0.0
        merged_bonus = 0.28 if candidate_kind == "merged" and len(ordered_lines) > 1 else 0.0
        score = float(candidate_conf) + length_bonus + mix_bonus + merged_bonus - over_penalty
        if score > best_score:
            best_text = candidate_text
            best_conf = float(candidate_conf)
            best_score = score

    return best_text, round(best_conf, 3)


def b64_to_cv2(b64: str) -> np.ndarray:
    """Decode base64 image string to OpenCV BGR array."""
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    img_bytes = base64.b64decode(b64)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def cv2_to_b64(img: np.ndarray, quality: int = 75) -> str:
    """Encode OpenCV BGR image to base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def load_models():
    """Load all models once at startup."""
    from ultralytics import YOLO
    import easyocr
    import open_clip

    models["yolo_sign"] = None
    models["yolo_plate"] = None
    models["clip_clf"] = None
    models["clip_pre"] = None
    models["class_names"] = []
    models["ocr_reader"] = None
    models["loaded"] = False
    models["ready_full_flow"] = False
    models["error"] = None
    models["component_errors"] = {}

    print(f"[LOAD] Device: {DEVICE}")

    if YOLO_SIGN_PATH.exists():
        try:
            models["yolo_sign"] = YOLO(str(YOLO_SIGN_PATH))
            print(f"[LOAD] YOLO sign: {YOLO_SIGN_PATH}")
        except Exception as exc:
            set_component_error("yolo_sign", str(exc))
            traceback.print_exc()
    else:
        set_component_error("yolo_sign", f"Missing file: {YOLO_SIGN_PATH}")

    if YOLO_PLATE_PATH.exists():
        try:
            models["yolo_plate"] = YOLO(str(YOLO_PLATE_PATH))
            print(f"[LOAD] YOLO plate: {YOLO_PLATE_PATH}")
        except Exception as exc:
            set_component_error("yolo_plate", str(exc))
            traceback.print_exc()
    else:
        set_component_error("yolo_plate", f"Missing file: {YOLO_PLATE_PATH}")

    if CLIP_CLF_PATH.exists():
        try:
            # Torch 2.6 defaults to weights_only=True; this checkpoint stores metadata too.
            ckpt = torch.load(str(CLIP_CLF_PATH), map_location=DEVICE, weights_only=False)
            clip_model, _, preprocess = open_clip.create_model_and_transforms(
                ckpt.get("clip_model", CLIP_MODEL_NAME),
                pretrained=CLIP_PRETRAIN,
            )
            clip_model = clip_model.to(DEVICE).eval()

            clf = CLIPClassifier(
                clip_model,
                ckpt.get("clip_dim", 768),
                ckpt["num_classes"],
            ).to(DEVICE)
            clf.load_state_dict(ckpt["state_dict"])
            clf.eval()

            models["clip_clf"] = clf
            models["clip_pre"] = preprocess
            models["class_names"] = ckpt["class_names"]
            print(f"[LOAD] CLIP classifier: {len(ckpt['class_names'])} classes")
        except Exception as exc:
            set_component_error("clip_clf", str(exc))
            traceback.print_exc()
    else:
        set_component_error("clip_clf", f"Missing file: {CLIP_CLF_PATH}")

    try:
        models["ocr_reader"] = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
        print("[LOAD] EasyOCR ready")
    except Exception as exc:
        set_component_error("ocr_reader", str(exc))
        traceback.print_exc()

    models["loaded"] = any([
        models["yolo_sign"] is not None,
        models["yolo_plate"] is not None,
    ])
    models["ready_full_flow"] = all([
        models["yolo_sign"] is not None,
        models["yolo_plate"] is not None,
        models["clip_clf"] is not None,
        models["ocr_reader"] is not None,
    ])

    if models["component_errors"]:
        models["error"] = " | ".join(
            f"{name}: {message}" for name, message in models["component_errors"].items()
        )

    if models["ready_full_flow"]:
        print("[LOAD] Full pipeline ready")
    elif models["loaded"]:
        print("[LOAD] Partial pipeline ready")
    else:
        print("[LOAD] No detector model is available")


@torch.no_grad()
def run_pipeline(
    image_bgr: np.ndarray,
    *,
    enable_clip: bool = True,
    enable_ocr: bool = True,
    include_plate_crop: bool = True,
    preview_mode: bool = False,
    dense_plate: bool = False,
) -> dict:
    """Core detection pipeline with fast preview/full modes."""
    h, w = image_bgr.shape[:2]
    signs, plates = [], []
    detect_conf = PREVIEW_CONF_THRESH if preview_mode else CONF_THRESH
    min_sign_bbox = PREVIEW_MIN_BBOX if preview_mode else MIN_BBOX
    min_plate_w = 10 if preview_mode else 15
    min_plate_h = 8 if preview_mode else 10

    # ── A: Traffic signs ────────────────────────────────────────────────────
    if models["yolo_sign"] is not None:
        sr = models["yolo_sign"](image_bgr, conf=detect_conf, imgsz=YOLO_SIGN_IMGSZ, verbose=False)
        sboxes = sr[0].boxes
        yolo_classes = models["yolo_sign"].names  # dict {id: name}

        if sboxes is not None:
            for j in range(len(sboxes)):
                x1, y1, x2, y2 = sboxes.xyxy[j].cpu().numpy().astype(int)
                yconf  = float(sboxes.conf[j])
                ycls   = int(sboxes.cls[j])
                yname  = yolo_classes.get(ycls, f"cls_{ycls}")
                bw, bh = x2 - x1, y2 - y1
                if bw < min_sign_bbox or bh < min_sign_bbox:
                    continue

                # CLIP classify
                clip_label, clip_score, top3 = yname, yconf, []
                if enable_clip and models["clip_clf"] is not None:
                    pad = max(3, int(min(bw, bh) * 0.1))
                    roi = image_bgr[
                        max(0, y1 - pad):min(h, y2 + pad),
                        max(0, x1 - pad):min(w, x2 + pad),
                    ]
                    if roi.shape[0] >= 10 and roi.shape[1] >= 10:
                        rpil  = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                        tensor = models["clip_pre"](rpil).unsqueeze(0).to(DEVICE)
                        logits = models["clip_clf"](tensor)
                        probs  = torch.softmax(logits, dim=1)[0]
                        tv, ti = probs.topk(min(3, len(models["class_names"])))
                        clip_label = models["class_names"][ti[0]]
                        clip_score = float(tv[0])
                        top3 = [
                            {"label": models["class_names"][ti[k]], "score": round(float(tv[k]), 3)}
                            for k in range(len(tv))
                        ]

                final = clip_label if clip_score >= 0.4 else yname
                fscore = clip_score if clip_score >= 0.4 else yconf

                signs.append({
                    "bbox":       [int(x1), int(y1), int(x2), int(y2)],
                    "label":      final,
                    "score":      round(fscore, 3),
                    "yolo_class": yname,
                    "yolo_conf":  round(yconf, 3),
                    "clip_label": clip_label,
                    "clip_score": round(clip_score, 3),
                    "top3":       top3,
                })

    # ── B: License plates ───────────────────────────────────────────────────
    if models["yolo_plate"] is not None:
        for det in collect_plate_detections(
            image_bgr,
            conf_thresh=detect_conf,
            preview_mode=preview_mode,
            dense=dense_plate,
        ):
            x1, y1, x2, y2 = det["bbox"]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(x1 + 1, min(w, x2))
            y2 = max(y1 + 1, min(h, y2))
            pconf = float(det["conf"])
            bw, bh = x2 - x1, y2 - y1
            if bw < min_plate_w or bh < min_plate_h:
                continue

            pad_x = max(8, int(round(bw * 0.10)))
            pad_y = max(8, int(round(bh * 0.18)))
            crop = image_bgr[
                max(0, y1 - pad_y):min(h, y2 + pad_y),
                max(0, x1 - pad_x):min(w, x2 + pad_x),
            ]

            plate_text, plate_conf = "", 0.0
            if enable_ocr and models["ocr_reader"] is not None and crop.shape[0] >= 10:
                best_candidate_score = -1.0
                for variant in build_plate_ocr_variants(crop):
                    try:
                        variant_ocr = models["ocr_reader"].readtext(
                            variant,
                            detail=1,
                            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-",
                        )
                    except TypeError:
                        variant_ocr = models["ocr_reader"].readtext(variant, detail=1)
                    candidate_text, candidate_conf = assemble_plate_text(variant_ocr)
                    candidate_score = score_plate_candidate(candidate_text, candidate_conf)
                    if candidate_score > best_candidate_score:
                        best_candidate_score = candidate_score
                        plate_text = candidate_text
                        plate_conf = candidate_conf

            plates.append({
                "bbox":        [int(x1), int(y1), int(x2), int(y2)],
                "plate_text":  plate_text,
                "plate_conf":  plate_conf,
                "detect_conf": round(pconf, 3),
                "crop_b64":    cv2_to_b64(crop, quality=80) if include_plate_crop and crop.size > 0 else "",
            })

    return {"signs": signs, "plates": plates}


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Tránh Safari/iPhone giữ bản HTML cũ (lỗi getUserMedia trên http://IP)."""
    html = render_template("index.html")
    resp = make_response(html)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/status")
def status():
    return jsonify({
        "loaded":    models["loaded"],
        "ready_full_flow": models["ready_full_flow"],
        "error":     models["error"],
        "device":    DEVICE,
        "yolo_sign":  models["yolo_sign"]  is not None,
        "yolo_plate": models["yolo_plate"] is not None,
        "clip_clf":   models["clip_clf"]   is not None,
        "ocr":        models["ocr_reader"] is not None,
        "classes":    len(models["class_names"]),
        "missing_components": sorted(models["component_errors"].keys()),
        "paths": models["paths"],
        "path_candidates": models["path_candidates"],
    })


@app.route("/detect", methods=["POST"])
def detect():
    """
    POST JSON: { "image": "<base64 jpeg>" }
    Returns: { "signs": [...], "plates": [...] }
    """
    try:
        if models["yolo_sign"] is None and models["yolo_plate"] is None:
            return jsonify({
                "error": "No detection model is loaded",
                "details": models["component_errors"],
            }), 503

        data = request.get_json(force=True)
        if not data or "image" not in data:
            return jsonify({"error": "Missing 'image' field"}), 400

        image_bgr = b64_to_cv2(data["image"])
        if image_bgr is None:
            return jsonify({"error": "Cannot decode image"}), 400

        mode = str(data.get("mode", "full")).lower()
        preview_mode = mode == "preview"

        result = run_pipeline(
            image_bgr,
            enable_clip=not preview_mode,
            enable_ocr=not preview_mode,
            include_plate_crop=not preview_mode,
            preview_mode=preview_mode,
        )
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/capture", methods=["POST"])
def capture():
    """
    Save a capture record with image + metadata.
    POST JSON:
    {
      "image":     "<base64 jpeg>",
      "timestamp": "2024-01-01T12:00:00",
      "latitude":  10.8231,
      "longitude": 106.6297,
      "accuracy":  15.0,
      "signs":     [...],
      "plates":    [...]
    }
    """
    try:
        data = request.get_json(force=True)
        if not data or "image" not in data:
            return jsonify({"error": "Missing 'image' field"}), 400

        capture_id = str(uuid.uuid4())[:8]
        ts = data.get("timestamp", datetime.now(timezone.utc).isoformat())

        # Save image
        image_bgr = b64_to_cv2(data["image"])
        img_path  = CAPTURES_DIR / f"{capture_id}.jpg"
        if image_bgr is not None:
            cv2.imwrite(str(img_path), image_bgr)

        # Save metadata JSON
        meta = {
            "id":        capture_id,
            "timestamp": ts,
            "latitude":  data.get("latitude"),
            "longitude": data.get("longitude"),
            "accuracy":  data.get("accuracy"),
            "signs":     data.get("signs", []),
            "plates":    data.get("plates", []),
            "image_file": img_path.name,
        }
        meta_path = CAPTURES_DIR / f"{capture_id}.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        return jsonify({"ok": True, "id": capture_id, "saved_to": str(img_path)})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/captures")
def list_captures():
    """Return list of all saved captures (metadata only)."""
    records = []
    for f in sorted(CAPTURES_DIR.glob("*.json"), reverse=True):
        try:
            records.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return jsonify(records)


@app.route("/captures/<path:filename>")
def serve_capture(filename):
    return send_from_directory(CAPTURES_DIR, filename)


# ── Startup ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_models()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
