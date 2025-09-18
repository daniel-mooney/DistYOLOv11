from distyolo import DistYOLO
from distyolo.results import BoxDistribution
from typing import List
import yaml
import sys
import torch
import cv2
import numpy as np


def dist_mean(p: np.ndarray, support: np.ndarray) -> float:
    """Calculate the mean of a 1D discrete probability distribution

    Args:
        p (np.ndarray): An array of probabilities
        support (np.ndarray): The distribution support

    Returns:
        float: The distribution mean
    """
    return np.sum(p * support)

def dist_cov(p: np.ndarray, support: np.ndarray) -> float:
    """Calculate the covariance of a discrete probability distribution"""
    mean = dist_mean(p, support)
    return np.sum(p * support**2) - mean**2


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

                print(f"Box {cls_id}, confidence {conf_score:.2f}, stride {box.stride[0].item()}:")

                # Print side statistics
                dist = box.distribution[0].cpu().numpy()
                support = np.arange(box.max_reg) * box.stride[0].cpu().item()

                labels = ['left', 'top', 'right', 'bottom']

                for i, side in enumerate(np.split(dist, 4)):
                    side = np.exp(side) / np.sum(np.exp(side))      # softmax

                    mean = dist_mean(side, support)
                    cov = dist_cov(side, support)

                    print(f"{labels[i]:<8} {mean=:6.2f}\t{cov=:6.2f}")
                
                width = (x2 - x1)
                height = (y2 - y1)

                print(f"{width=}")
                print(f"{height=}", end="\n\n")

                l, t, r, b = [np.exp(d) / np.sum(np.exp(d)) for d in np.split(dist, 4)]
                print(f"{l=}")
                print(f"{t=}")
                print(f"{r=}")
                print(f"{b=}")

                print("----")
        
        # Write the index in the top left corner
        cv2.putText(disp, f"Image {idx+1}/{len(test_images)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(win, disp)
        k = cv2.waitKey(0) & 0xFF
        if k in (ord("q"), 27):  # q or ESC
            break
        elif k in (ord("n"), 83):  # n or Right arrow
            idx = (idx + 1) % len(test_images)
        elif k in (ord("p"), 81):  # p or Left arrow
            idx = (idx - 1) % len(test_images)
        

    cv2.destroyAllWindows()
if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True, linewidth=200)

    cfg_yaml = sys.argv[1]
    main(cfg_yaml)