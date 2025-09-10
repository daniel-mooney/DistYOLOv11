from distyolo import DistYOLO


def main() -> None:
    model = DistYOLO("yolo11s.pt")

    results = model.predict()
    # print(results)

if __name__ == "__main__":
    main()