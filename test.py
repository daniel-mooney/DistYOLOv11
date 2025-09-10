# test_view.py
from ultralytics import YOLO
import cv2
import os
import sys
import yaml
from pathlib import Path
from typing import List, Optional


# ---------- Config / data helpers ----------

def load_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    for k in ("weights", "data", "split"):
        if not cfg.get(k):
            raise ValueError(f"Missing required key '{k}' in {cfg_path}")
    return cfg


def get_data_yaml(data_yaml_path: str) -> dict:
    with open(data_yaml_path, "r") as f:
        return yaml.safe_load(f)


def list_test_images_from_txt(data_yaml_path: str, split: str = "test") -> List[str]:
    data = get_data_yaml(data_yaml_path)
    split_file = data.get(split)
    if not split_file or not os.path.exists(split_file):
        raise FileNotFoundError(f"{split}.txt not found or missing in data.yaml: {split_file}")
    with open(split_file, "r") as f:
        imgs = [ln.strip() for ln in f if ln.strip()]
    if not imgs:
        raise ValueError(f"No images listed in {split_file}")
    return imgs


def get_class_names_from_data_yaml(data_yaml_path: str) -> Optional[List[str]]:
    try:
        data = get_data_yaml(data_yaml_path)
        names = data.get("names", None)
        if isinstance(names, dict):
            names = [name for _, name in sorted(names.items(), key=lambda kv: int(kv[0]))]
        return names if isinstance(names, list) else None
    except Exception:
        return None


# ---------- Labels / drawing ----------

def infer_label_path_from_image(image_path: str) -> str:
    """Heuristics to find YOLO txt label path from an image path."""
    p = Path(image_path)
    # Try .../images/... -> .../labels/...
    parts = list(p.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        lp = Path(*parts).with_suffix(".txt")
        if lp.exists():
            return str(lp)
    # Try sibling 'labels' directory next to image parent
    lp = p.with_suffix(".txt")
    labels_dir = p.parent.parent / "labels" / p.name
    labels_dir = labels_dir.with_suffix(".txt")
    if labels_dir.exists():
        return str(labels_dir)
    # Fallback: alongside image (ad-hoc sets)
    return str(lp)


def draw_gt_boxes(bgr, label_file: str, class_names: Optional[List[str]] = None):
    if not os.path.exists(label_file):
        return bgr
    h, w = bgr.shape[:2]
    try:
        with open(label_file, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    except Exception:
        return bgr

    for ln in lines:
        parts = ln.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        x, y, bw, bh = map(float, parts[1:5])  # normalized cx,cy,w,h
        cx, cy, pw, ph = x * w, y * h, bw * w, bh * h
        x1 = int(max(0, cx - pw / 2))
        y1 = int(max(0, cy - ph / 2))
        x2 = int(min(w - 1, cx + pw / 2))
        y2 = int(min(h - 1, cy + ph / 2))
        # GT = green
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = class_names[cls] if class_names and 0 <= cls < len(class_names) else str(cls)
        cv2.putText(bgr, f"GT:{label}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return bgr


# ---------- Metrics ----------

def compute_metrics(model: YOLO, cfg: dict) -> dict:
    res = model.val(
        data=cfg["data"],
        split=cfg.get("split", "test"),
        imgsz=cfg.get("imgsz", 640),
        conf=cfg.get("conf", 0.25),
        iou=cfg.get("iou", 0.7),
        device=cfg.get("device", 0),
        batch=cfg.get("batch", 16),
        verbose=False,
    )

    out = {}
    if hasattr(res, "results_dict") and isinstance(res.results_dict, dict):
        rd = res.results_dict
        out["metrics/precision(B)"]   = float(rd.get("metrics/precision(B)", rd.get("metrics/precision", 0.0)))
        out["metrics/recall(B)"]      = float(rd.get("metrics/recall(B)", rd.get("metrics/recall", 0.0)))
        out["metrics/mAP50(B)"]       = float(rd.get("metrics/mAP50(B)", rd.get("metrics/mAP50", 0.0)))
        out["metrics/mAP50-95(B)"]    = float(rd.get("metrics/mAP50-95(B)", rd.get("metrics/mAP50-95", 0.0)))
        out["fitness"]                = float(rd.get("fitness", 0.0))
    else:
        # Fallback (older/newer API variations)
        try:
            mAP50_95 = float(getattr(getattr(res, "box", None), "map", 0.0))
            maps = getattr(getattr(res, "box", None), "maps", None)
            mAP50 = float(maps[0]) if maps and len(maps) > 0 else 0.0
        except Exception:
            mAP50_95, mAP50 = 0.0, 0.0
        out["metrics/precision(B)"] = 0.0
        out["metrics/recall(B)"] = 0.0
        out["metrics/mAP50(B)"] = mAP50
        out["metrics/mAP50-95(B)"] = mAP50_95
        out["fitness"] = 0.0

    return out


# ---------- Viewer ----------

def viewer_loop(model: YOLO, cfg: dict):
    imgs = list_test_images_from_txt(cfg["data"], cfg.get("split", "test"))
    names = get_class_names_from_data_yaml(cfg["data"])

    conf = cfg.get("conf", 0.25)
    iou = cfg.get("iou", 0.7)
    imgsz = cfg.get("imgsz", 640)
    device = cfg.get("device", 0)

    idx = 0
    show_gt = True
    win = "YOLOv11 Test Viewer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    help_text = "Keys: n/→ next | p/← prev | g toggle GT | s save | q quit"

    print(type(model.model))

    while True:
        img_path = imgs[idx]

        preds = model.predict(
            source=img_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )

        disp = cv2.imread(img_path) if not preds else preds[0].plot()  # BGR with preds
        if disp is None:
            # Skip unreadable image
            idx = (idx + 1) % len(imgs)
            continue

        if show_gt:
            lab = infer_label_path_from_image(img_path)
            disp = draw_gt_boxes(disp, lab, names)

        overlay = disp.copy()
        text = f"[{idx+1}/{len(imgs)}] {Path(img_path).name} | {help_text}"
        cv2.putText(overlay, text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(win, overlay)
        k = cv2.waitKey(0) & 0xFF

        if k in (ord("q"), 27):  # q or ESC
            break
        elif k in (ord("n"), 83):  # n or Right arrow
            idx = (idx + 1) % len(imgs)
        elif k in (ord("p"), 81):  # p or Left arrow
            idx = (idx - 1) % len(imgs)
        elif k == ord("g"):
            show_gt = not show_gt
        elif k == ord("s"):
            out_path = str(Path(imgs[idx]).with_suffix("")) + "_pred.jpg"
            cv2.imwrite(out_path, overlay)
            print(f"Saved: {out_path}")

    cv2.destroyAllWindows()


# ---------- Main ----------

def main() -> None:
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "test.yaml"
    cfg = load_cfg(cfg_path)

    model = YOLO(cfg["weights"])

    # 1) Compute + print metrics in the requested format
    metrics = compute_metrics(model, cfg)
    print("results_dict:", metrics)

    # 2) Launch interactive viewer over images listed in test.txt
    viewer_loop(model, cfg)


if __name__ == "__main__":
    main()
