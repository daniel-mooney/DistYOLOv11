from ultralytics import YOLO
import sys
import yaml

def main() -> None:
    train_cfg = sys.argv[1]

    with open(train_cfg, 'r') as f:
        override = yaml.safe_load(f)

    model = YOLO(override['model'])
    results = model.train(**override)
    print(results)

if __name__ == "__main__":
    main()