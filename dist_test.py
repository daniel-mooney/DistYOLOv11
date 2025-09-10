from distyolo import DistYOLO
from distyolo.results import BoxDistribution
from typing import List
import yaml
import sys
import torch
import cv2
import numpy as np


def main(cfg_yaml: str) -> None:
    with open(cfg_yaml, 'r') as f:
        cfg = yaml.safe_load(f)
    
    conf = cfg.get("conf", 0.25)
    iou = cfg.get("iou", 0.7)
    imgsz = cfg.get("imgsz", 640)
    device = cfg.get("device", 0)

    data_cfg_file = cfg["data"]

    with open(data_cfg_file, "r") as f:
        data_cfg = yaml.safe_load(f)
    
    with open(data_cfg['test'], "r") as f:
        test_images = [line.strip() for line in f if line.strip()]
    
    model = DistYOLO(cfg['weights'])
    idx = 0

    win = "YOLOv11 Test Viewer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        img_path = test_images[idx % len(test_images)]
        disp = cv2.imread(img_path)

        preds = model.predict(
            source=img_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device
        )

        for pred in preds:
            boxes: List[BoxDistribution] = pred.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf_score = box.conf[0].item()
                label = f"{data_cfg['names'][cls_id]} {conf_score:.2f}"
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(disp, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                dist = box.distribution[0].cpu().numpy()
                l, t,r, b = np.split(dist, 4)

                # Perform softmax over each side's distribution
                l = np.exp(l) / np.sum(np.exp(l))
                t = np.exp(t) / np.sum(np.exp(t))
                r = np.exp(r) / np.sum(np.exp(r))
                b = np.exp(b) / np.sum(np.exp(b))

                print(f"left: {l}")
                print(f"top: {t}")
                print(f"right: {r}")
                print(f"bottom: {b}")

        cv2.imshow(win, disp)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'):
            break
        idx += 1

    cv2.destroyAllWindows()
if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True, linewidth=200)

    cfg_yaml = sys.argv[1]
    main(cfg_yaml)