from ultralytics import YOLO
import sys
import yaml

from distyolo import DistYOLO

def main() -> None:
    train_cfg = sys.argv[1]

    with open(train_cfg, 'r') as f:
        override = yaml.safe_load(f)

    model = DistYOLO(override['model'])
    results = model.train(**override)
    print(results)

if __name__ == "__main__":
    main()