from ultralytics import YOLO
import sys

def main() -> None:
    train_cfg = sys.argv[1]
    model = YOLO()
    results = model.train(cfg=train_cfg)
    print(results)

if __name__ == "__main__":
    main()